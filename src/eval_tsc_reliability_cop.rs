use std::{
    hint,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};

use anyhow::Context;
use libc::cpu_set_t;

use crate::*;

/// A single TSC probe
#[derive(Clone, Copy, Default)]
struct TscProbe {
    /// TSC value
    tsc_val: u64,
    /// Position in a globally-ordered sequence on TSC probes
    seq_num: u64,
}

/// Shared state for TSC probe threads
struct TscProbeSharedState {
    /// Global TSC probe sequence counter
    seq_counter: AtomicU64,

    /// A reference to a variable shared by all TSC probe threads. The variable plays a
    /// role of semaphore. Each thread increments it atomically only once to signal that
    /// it's ready to collect probes. The threads don't start collecting probes until
    /// they notice that all the threads incremented the counter. Use of the counter
    /// ensures that threads start collecting TSC probes more or less simultaneously
    ready_counter: AtomicU32,

    /// The number of TSC probe threads. Serves as a target value for the "ready counter".
    /// The threads start collecting probes only when they notice that the counter is
    /// equal to the number of threads
    num_threads: u32,
}


///
/// Thread function that collects TSC probes
///
/// NOTE: TSC probe threads must allow asynchronous cancelability at any time.
///       Explicit memory allocation is not allowed inside these threads.
///       Synchronization methods should be thought through carefully.
///
fn tsc_probe_thread_fn(
    cpu_set: &cpu_set_t,
    shared_state: &TscProbeSharedState,
    probes: &mut [TscProbe],
) -> Result<(), anyhow::Error> {
    unsafe {
        let thread_self = libc::pthread_self();
        let cpu_set_size = std::mem::size_of::<libc::cpu_set_t>();

        // Switch to a CPU designated for this thread
        if libc::pthread_setaffinity_np(thread_self, cpu_set_size, cpu_set) != 0 {
            bail!("Couldn't bind itself to a designated CPU")
        }

        // At this point the thread is ready to collect TSC probes. But it doesn't start
        // doing that until all other threads are also ready. We use a shared counter to
        // ensure that all threads start collecting probes more or less simultaneously.
        // Each thread increments the counter when it is ready to collect probes. Then
        // the thread waits until the counter reaches its target value (which is equal to
        // the number of threads)
        shared_state.ready_counter.fetch_add(1, Ordering::AcqRel);

        // Just spin (and burn CPU cycles, but hopefully for not so long)
        while shared_state.ready_counter.load(Ordering::Acquire) < shared_state.num_threads {
            hint::spin_loop();
        }

        // Well, can collect TSC probes finally. This loop should be as tight as
        // possible. The less operations inside the better
        for probe in probes.iter_mut() {
            let mut seq_num;
            let mut tsc_val;

            loop {
                seq_num = shared_state.seq_counter.load(Ordering::Acquire);
                tsc_val = get_tsc();

                if shared_state
                    .seq_counter
                    .compare_exchange(seq_num, seq_num + 1, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    break;
                }
            }

            probe.seq_num = seq_num;
            probe.tsc_val = tsc_val.0;
        }

        Ok(())
    }
}


/// Print sequence of collected TSC probes
fn print_tsc_probe_sequence(tsc_probes: &[Vec<TscProbe>]) {
    println!("Sequence of TSC probes");
    println!("(CPU index may not be equal to CPU ID)");
    println!("(CPU index <-> CPU ID mapping should be printed above)");

    let num_threads = tsc_probes.len();
    let probes_count = tsc_probes[0].len();

    assert!(u64::MAX / probes_count as u64 > num_threads as u64);

    let mut indexes = vec![0usize; num_threads];
    let total_probes = num_threads * probes_count;

    for seq_num in 0..total_probes {
        // Find which CPU produced this probe
        for cpu_ind in 0..num_threads {
            if indexes[cpu_ind] < probes_count
                && tsc_probes[cpu_ind][indexes[cpu_ind]].seq_num == seq_num as u64
            {
                println!(
                    "Seq {}: CPU index {}, TSC value {}",
                    seq_num, cpu_ind, tsc_probes[cpu_ind][indexes[cpu_ind]].tsc_val
                );
                indexes[cpu_ind] += 1;
                break;
            }
        }
    }
}

/// Collect TSC probes
///
///   - probes_count probes is collected on each available CPU
///   - the probes are collected by concurrently running threads (1 thread per each
///     available CPU)
///   - the probes are sequentially ordered. The order is ensured by means of compare-and-
///     swap operation
fn collect_cas_ordered_tsc_probes(
    _num_cpus: i32,
    cpu_sets: &[libc::cpu_set_t],
    tsc_probes: &mut [Vec<TscProbe>],
    _probes_count: usize,
) -> Result<(), anyhow::Error> {
    let num_threads = cpu_sets.len();

    let shared_state = TscProbeSharedState {
        seq_counter: AtomicU64::new(0),
        ready_counter: AtomicU32::new(0),
        num_threads: num_threads as u32,
    };

    // Use scoped threads to allow borrowing without 'static requirement
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);

        // We need to split tsc_probes into mutable disjoint slices
        // Use indices to avoid double mutable borrow issues
        let tsc_probes_ptr = tsc_probes.as_mut_ptr();

        for i in 0..num_threads {
            let cpu_set = &cpu_sets[i];
            let shared = &shared_state;

            // SAFETY: Each thread accesses a different index of tsc_probes
            let probes = unsafe { &mut *tsc_probes_ptr.add(i) };

            let handle = s.spawn(move || tsc_probe_thread_fn(cpu_set, shared, probes));
            handles.push(handle);
        }

        // Wait for all threads to complete
        let mut result = Ok(());
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if result.is_ok() {
                        result = Err(e);
                    }
                }
                Err(_) => {
                    if result.is_ok() {
                        result = Err(anyhow::anyhow!("Thread panicked"));
                    }
                }
            }
        }
        result
    })
}


/// Generic evaluation of consistency of TSC probes
/// Returns true if TSC values DON'T vary (which is bad - inconsistency detected)
fn is_tsc_probes_consistency(tsc_probes: &[Vec<TscProbe>]) -> bool {
    // Make sure that collected TSC values do vary on each of the CPUs. That may not be
    // true, for example, in case when some CPUs consistently return "zero" for every TSC
    // test.
    // This check is really important. If all TSC values are equal, then both TSC "delta"
    // ranges and TSC monotonicity will be perfect, but at the same time TSC would be
    // absolutely inappropriate for measuring wall-clock time. (Global TSC monotonicity
    // evaluation and some other monotonicity checks existing in the library will give the
    // positive result because they don't require successively measured TSC values to
    // strictly grow. Overall, WTMLIB's requirements with respect to TSC monotonicity are
    // the following: TSC values must grow on a global scale and not decrease locally.
    // I.e. the library allows some successively measured TSC values to be equal to each
    // other)
    tsc_probes.iter().any(|probes| {
        if let (Some(first), Some(last)) = (probes.first(), probes.last()) {
            first.tsc_val == last.tsc_val
        } else {
            true // Empty is inconsistent
        }
    })
}

/// Calculate bounds of a shift between TSC on a given CPU and TSC on the base CPU
/// (assuming that TSC values were collected using the method of CAS-ordered probes)
///
/// We do that in the following way:
///   1) let's take from a globally ordered sequence of TSC probes two probes that were
///      collected successively on the base CPU. Let's denote these probes p1 and p2
///   2) in the globally ordered sequence of probes there may or may not be probes that
///      were collected between p1 and p2 (on the other CPU of course). Assume that a
///      sequence of such probes does exist. There may be just one probe in this sequence.
///      Let's denote the first probe in this sequence P1 and the last one P2 (P1 and P2
///      may refer to the same probe)
///   3) let's now denote TSC values corresponding to p1 and p2 t1 and t2. Also we denote
///      TSC values corresponding to P1 and P2 T1 and T2
///   4) let's choose some value T such that T1 <= T <= T2
///   5) when TSC on the given CPU was equal to T, TSC on the base CPU was somewhere
///      between t1 and t2. Let's denote that value of the base TSC "t"
///   6) we're interested in a difference "delta = T - t". Let's find bounds for this
///      difference
///   7) this difference is the smallest when t is the biggest. t is the biggest when it
///      is closest to t2. But t cannot be closer to t2 than "T2 - T". That's because t
///      cannot be closer to t2 than T to T2 (assuming that time runs at the same pace on
///      both CPUs). Thus, the maximum value for t is "t2 - (T2 - T)". And the minimum
///      value for "delta" is "T - (t2 - (T2 - T)) = T2 - t2"
///   8) similarly we find the upper bound for delta. delta is the biggest when t is the
///      smallest. t is the smallest when it is closest to t1. But t cannot be closer to t1
///      than T to T1. So, the minumim value for t is "t1 + (T - T1)". And the maximum
///      value for delta is "T - (t1 + (T - T1)) = T1 - t1"
///   9) that's how we find a range for "delta" based on a single sequence of TSC probes
///      collected on the given CPU and enclosed between a pair of TSC probes collected
///      successively on the base CPU
///  10) but in a globally ordered sequence of TSC probes there may exist multiple
///      sub-sequences with that property. Based on that, we can narrow the range of
///      possible "delta" values. To do that, we calculate an intersection of multiple
///      "delta" ranges calculated for different TSC probe sub-sequences of a globally
///      ordered TSC sequence
///
/// Input: tsc_probes[0] - array of TSC probes collected on the base CPU
///        tsc_probes[1] - array of TSC probes collected on some other CPU
fn calc_tsc_delta_range_cop(
    tsc_probes: &[Vec<TscProbe>],
) -> Result<(i64, i64), anyhow::Error> {
    let mut d_min = i64::MIN;
    let mut d_max = i64::MAX;
    // Indexes into arrays of TSC probes collected on the given and base CPUs
    let mut ig = 0usize;
    let mut seq_num: u64 = 0;
    // The number of produced independent "delta" range estimations
    let mut num_ranges = 0i64;
    let num_probes = tsc_probes[0].len();

    println!("Calculating shift between TSC counters of the two CPUs...");

    if is_tsc_probes_consistency(tsc_probes) {
        bail!("Data inconsistency: TSC values don't vary")
    }

    // Consistency check. Successive TSC values measured on the same CPU must
    // not decrease (unless TSC counter wraps)
    for i in 1..num_probes {
        if tsc_probes[0][i].tsc_val < tsc_probes[0][i - 1].tsc_val
            || tsc_probes[1][i].tsc_val < tsc_probes[1][i - 1].tsc_val
        {
            bail!("Detected decreasing successive TSC values (measured on the same CPU). Can be a result of TSC wrap")
        }
    }

    // Skip those TSC probes collected on the given CPU that are not enclosed between any probes collected on the base CPU
    if tsc_probes[1][0].seq_num == 0 {
        while ig < num_probes && tsc_probes[1][ig].seq_num == ig as u64 {
            ig += 1;
            seq_num += 1;
        }
    }

    // Sequence numbers must be sequential
    if tsc_probes[0][0].seq_num != seq_num {
        bail!("Sequence number mismatch");
    }

    // The loop counter starts from "one"
    let mut ib = 1usize;
    while ib < num_probes {
        // Check whether between the current and previous TSC probes collected on the base
        // CPU there are probes collected on the other CPU
        if tsc_probes[0][ib].seq_num == seq_num + 1 {
            seq_num = tsc_probes[0][ib].seq_num;
            ib += 1;
            continue;
        }

        if tsc_probes[0][ib].seq_num <= seq_num + 1 {
            ib += 1;
            continue;
        }

        num_ranges += 1;

        let tsc_base_prev = tsc_probes[0][ib - 1].tsc_val;
        let tsc_base_curr = tsc_probes[0][ib].tsc_val;

        // First and last indexes of TSC probes that were collected on the given CPU
        // between successive TSC probes collected on the base CPU
        let sub_seq_first = ig;

        while ig < num_probes && tsc_probes[1][ig].seq_num < tsc_probes[0][ib].seq_num {
            seq_num = tsc_probes[1][ig].seq_num;
            ig += 1;
        }

        let sub_seq_last = if ig > 0 { ig - 1 } else { 0 };

        if sub_seq_last < sub_seq_first {
            ib += 1;
            continue;
        }

        let tsc_given_min = tsc_probes[1][sub_seq_first].tsc_val;
        let tsc_given_max = tsc_probes[1][sub_seq_last].tsc_val;

        // Check that we will not get overflow while subtracting TSC values
        let diff1 = u64::abs_diff(tsc_given_min, tsc_base_prev);
        let diff2 = u64::abs_diff(tsc_given_max, tsc_base_curr);

        if diff1 > i64::MAX as u64 || diff2 > i64::MAX as u64 {
            bail!("TSC wrap detected. Cannot rely on TSC while measuring wall-clock time")
        }

        // Time interval between enclosing base probes must be bigger than time
        // interval between enclosed probes collected on the other CPU
        if tsc_base_curr - tsc_base_prev < tsc_given_max - tsc_given_min {
            bail!("Time runs at different pace on different CPUs or TSC wrap occurred");
        }

        let bound_min = tsc_given_max as i64 - tsc_base_curr as i64;
        let bound_max = tsc_given_min as i64 - tsc_base_prev as i64;

        // "delta" ranges calculated for different sub-sequences must intersect
        if bound_min > d_max || bound_max < d_min {
            bail!("TSC delta ranges don't intersect. Major TSC inconsistency detected");
        }

        println!(
            "The shift belongs to range: {} [{}, {}]",
            bound_max - bound_min,
            bound_min,
            bound_max,
        );

        d_min = d_min.max(bound_min);
        d_max = d_max.min(bound_max);

        seq_num = tsc_probes[0][ib].seq_num;
        ib += 1;
    }

    if num_ranges < WTMLIB_TSC_DELTA_RANGE_COUNT_THRESHOLD {
        bail!(
            "Couldn't observe the required amount of TSC probe sub-sequences with desired properties ({} required, {} found)",
            WTMLIB_TSC_DELTA_RANGE_COUNT_THRESHOLD,
            num_ranges,
        );
    }

    if d_min > d_max {
        bail!("Invalid delta range: min > max");
    }

    println!(
        "Combined range (intersection of all the above): {} [{}, {}]",
        d_max - d_min,
        d_min,
        d_max,
    );

    Ok((d_min, d_max))
}

/// Calculate size of enclosing TSC range (using a sequence of CAS-ordered probes)
///
/// "Size of enclosing TSC range" is a non-negative integer value such that if TSC
/// values are measured simultaneously on all the available CPUs, then difference
/// between the largest and the smallest will be not bigger than this value. In other
/// words, it's an estimated upper bound for shifts between TSC counters running on
/// different CPUs
///
/// To calculate "enclosing TSC range" we do the following:
///   1) for each CPU we calculate bounds that enclose a shift between this CPU's TSC
///      and TSC of some base CPU
///   2) we calculate the smallest range that encloses all ranges calculated during
///      the previous step
///
/// When "enclosing TSC range" is found, its size is calculated as a difference
/// between its upper and lower bounds
fn calc_tsc_enclosing_range_cop(
    num_cpus: i32,
    base_cpu: i32,
    cpu_constraint: &cpu_set_t,
    _cline_size: i32,
) -> Result<i64, anyhow::Error> {
    unsafe {
        let mut l_bound = i64::MAX;
        let mut u_bound = i64::MIN;
        let probes_count = WTMLIB_EVAL_TSC_MONOTCTY_PROBES_COUNT;

        println!("Calculating an upper bound for shifts between TSC counters running on different CPUs...");
        println!("Base CPU ID: {}", base_cpu);

        let mut cpu_sets: Vec<cpu_set_t> = vec![std::mem::zeroed(); 2];
        let mut tsc_probes: Vec<Vec<TscProbe>> =
            vec![vec![Default::default(); probes_count]; 2];

        libc::CPU_ZERO(&mut cpu_sets[0]);
        libc::CPU_SET(base_cpu as usize, &mut cpu_sets[0]);
        libc::CPU_ZERO(&mut cpu_sets[1]);

        for cpu_id in 0..num_cpus {
            if !libc::CPU_ISSET(cpu_id as usize, cpu_constraint) || cpu_id == base_cpu {
                continue;
            }

            libc::CPU_SET(cpu_id as usize, &mut cpu_sets[1]);
            println!(
                "Collecting TSC probes on CPUs {} and {}...",
                base_cpu, cpu_id
            );

            collect_cas_ordered_tsc_probes(num_cpus, &mut cpu_sets, &mut tsc_probes, probes_count)
                .context("Error while collecting CAS-ordered TSC probes")?;

            println!("CPU ID {} maps to CPU index {}", base_cpu, 0);
            println!("CPU ID {} maps to CPU index {}", cpu_id, 1);
            print_tsc_probe_sequence(&tsc_probes);

            let (delta_min, delta_max) = calc_tsc_delta_range_cop(&tsc_probes)
                .context("Calculation of TSC delta range failed")?;

            // Update bounds of the enclosing TSC range
            l_bound = l_bound.min(delta_min);
            u_bound = u_bound.max(delta_max);

            assert!(delta_max >= delta_min && u_bound >= l_bound);
            // Return CPU mask to the "clean" state
            libc::CPU_CLR(cpu_id as usize, &mut cpu_sets[1]);
        }

        println!(
            "Shift between TSC on any of the available CPUs and TSC on the base CPU belongs to range: [{}, {}]",
            l_bound, u_bound
        );
        println!(
            "Upper bound for shifts between TSCs is: {}",
            u_bound - l_bound
        );

        let range_size = u_bound - l_bound;
        Ok(range_size)
    }
}

/// Check whether TSC values monotonically increase along an ordered sequence of TSC
/// probes
///
/// Monotonicity is evaluated in a straightforward way:
///   - TSC probes have sequence numbers that reflect an order in which the probes were
///     collected
///   - the function traverses the probes in order of increasing sequence numbers and
///     examines whether TSC values also increase
///
/// Along with the examination described above the function also assess statistical
/// significance of the result. Let us use graph theory terms to explain how the
/// assessement is made (graph terminology is not important here; it just makes the
/// explanation simpler):
///   - assume a graph such that allowed CPUs are its vertices. The graph is fully
///     connected by undirected unweighted edges
///   - we treat the sequence of TSC probes as a path in this graph. (The sequence of
///     probes naturally transforms to a sequence of CPUs on which these probes were
///     collected. And the sequence of CPUs - which is a sequence of vertices of the
///     graph - represents some path in the graph)
///   - we introduce the notion of "full loop". "Full loop" is a sub-sequence consisting
///     of successive (!) points on the path such that:
///        1) first and last CPUs in the sub-sequence are the same
///        2) each of the available CPUs can be found at least once in the sub-sequence
///        3) there is no shorter sub-sequence of successive points that has the same
///           starting point and satisfies conditions 1) and 2)
///     In other words, "full loop" is a sub-path of the path such that it starts at
///     some CPU, passes through all other CPUs (all or some of them may be visited
///     several times), and returns to the starting CPU (which also might be visited
///     several times before the finish). And there should be no shorter sub-path with
///     the same properties which starts at the same starting point.
///   - positive result of monotonicity evaluation is considered more or less reliable
///     (or statistically significant) if a "full loop" exists on the path. (If all
///     available CPUs were visited on the path but no "full loop" exists, the result
///     cannot be considered reliable. It's easy to understand why).
///   - the more "full loops" exist the more reliable is the result
///   - so, to assess statistical significance of monotinicity evaluation we count the
///     number of "full loops" on the provided CPU path
///   - there can exist overlapping "full loops" on the path. We don't take them into
///     account. Taking them into account would complicate the algorithm significantly
///     while not improving reliability assessment too much. So, we count only "full
///     loops" that are located on the path strictly one after another
///   - and we introduce one more simplification. We require that all "full loops" start
///     with the same CPU (which is actually the first CPU in the TSC probe sequence).
///     This simplification allows us to build a very simple and fast algorithm. Its
///     complexity is O(num_probes) and additional memory is O(num_CPUs). But this
///     algorithm is less precise than algorithm that allows "full loops" to start with
///     arbitrary CPU. Consider an example. Assume that we have 4 CPUs and the following
///     path: 3 2 1 3 4 2. There is no "full loop" here that starts with CPU 3. So,
///     the algorithm implemented here will not find any "full loop" at all. But in the
///     sequence above there exists a "full loop" that starts with CPU 2.
///     An algorithm that doesn't impose constraint on the starting CPU can be easily
///     implemented in a way that is very similar to what is seen in the function below.
///     The complexity would be O(num_probes * num_CPUs) and additional memory would be
///     O(num_CPUs ^ 2). But currently the simplified version works well. There is no
///     need in higher precision. If higher precision will become a requirement, simple
///     modifications of the below code will allow to have it.
///
/// The algorithm we use to find "full loops" is the following:
///   - we iterate over TSC probes from the first to the last
///   - we have a counter of already found "full loops"
///   - for each available CPU we have a flag indicating whether this CPU was seen since
///     we have found the previous "full loop"
///   - also we a have a variable that tracks the number of different CPUs seen since
///     the last "full loop" was found
///   - if this variable becomes equal to the number of available CPUs, and if we
///     encounter the starting CPU after that, then we conclude that we have one more
///     "full loop"
///
fn is_probe_sequence_monotonic(tsc_probes: &[Vec<TscProbe>]) -> Result<bool, anyhow::Error> {
    let mut is_monotonic = true;

    print!("Testing monotonicity of the TSC probes sequence...");

    if !is_tsc_probes_consistency(tsc_probes) {
        bail!("TSC inconsistency")
    }

    // Number of "full loops" there were already found
    let mut num_loops = 0;
    // indexes[cpu_ind] stores current index to an array of TSC probes collected on a CPU that has index "cpu_ind"
    let mut indexes = vec![0; tsc_probes.len()];
    // Number of different CPUs seen while trying to find a new "full loop"
    let mut cpus_seen = 0;
    // Index of the first CPU in the TSC probes sequence
    let first_cpu_ind = tsc_probes
        .iter()
        .position(|i| i[0].seq_num == 0)
        .context("index of the first CPU not found")?;
    let mut prev_tsc_val = 0;
    let prob_num = tsc_probes[0].len();
    // If (cpu_seen_num[ind] == num_loops + 1) then it means that we've already seen CPU
    // with index "ind" while trying to find a new "full loop". CPU index may not be equal
    // to CPU ID. See the caller function to understand the difference
    let mut cpu_seen_num = vec![0; tsc_probes.len()];

    for i in 0..(prob_num * tsc_probes.len()) {
        let mut cpu_ind = 0;
        while cpu_ind < tsc_probes.len() {
            let tsc_probe = tsc_probes[cpu_ind][indexes[cpu_ind]];

            if tsc_probe.seq_num as usize != i {
                cpu_ind += 1;
                continue;
            }

            if tsc_probe.tsc_val < prev_tsc_val {
                is_monotonic = false;
                println!("TSC value growth breaks at sequence number {}", i);
                break;
            }

            indexes[cpu_ind] += 1;
            prev_tsc_val = tsc_probe.tsc_val;

            // Have we found the new "full loop"?
            if cpus_seen == tsc_probes.len() && cpu_ind == first_cpu_ind {
                num_loops += 1;
                cpus_seen = 0;
            }

            // Do we see the current CPU for the first time while trying to find a new "full loop"?
            if cpu_seen_num[cpu_ind] < num_loops + 1 {
                assert!(cpu_seen_num[cpu_ind] == num_loops);
                cpu_seen_num[cpu_ind] += 1;
                cpus_seen += 1;
                assert!(cpus_seen <= tsc_probes.len())
            } else {
                assert!(cpu_seen_num[cpu_ind] == num_loops + 1);
            }

            break;
        }

        if !is_monotonic {
            break;
        };

        if cpu_ind == tsc_probes.len() {
            bail!(
                "Internal inconsistency: couldn't find TSC probe with sequential number {}",
                i
            );
        }
    }

    if is_monotonic && num_loops < WTMLIB_FULL_LOOP_COUNT_THRESHOLD {
        bail!(
            "Couldn't observe the required amount of TSC probe sub-sequences with desired properties ({} required, {} found)",
            WTMLIB_FULL_LOOP_COUNT_THRESHOLD,
            num_loops,
        );
    }

    if is_monotonic {
        println!("The collected TSC values DO monotonically increase");
        return Ok(true);
    }

    return Ok(false);
}

/// Check whether TSC values measured on same/different CPUs one after another
/// monotonically increase
///
/// The algorithm is the following:
///   1) collect TSC probes using concurrently running threads (one thread per each
///      available CPU). All the measurements are sequentially ordered by means of
///      compare-and-swap operation
///   2) check whether TSC values monotonically increase along the sequence of collected
///      probes
///   3) at the same time evaluate statistical significance of the result
///
/// NOTE: if the function reports that collected TSC values do not monotonically increase,
///       that doesn't necessarily imply that TSCs are unreliable. In some cases the
///       observed decrease may be a result of TSC wrap
fn eval_tsc_monotonicity_cop(
    num_cpus: i32,
    cpu_constraint: &cpu_set_t,
    _cline_size: i32,
) -> Result<bool, anyhow::Error> {
    unsafe {
        println!("Evaluating TSC monotonicity...");

        // Calculate the number of CPUs available to the current thread
        let mut num_cpus_avail = 0;
        for cpu_id in 0..num_cpus as usize {
            if libc::CPU_ISSET(cpu_id, cpu_constraint) {
                num_cpus_avail += 1;
            }
        }

        let probes_count = WTMLIB_EVAL_TSC_MONOTCTY_PROBES_COUNT;
        let mut cpu_sets: Vec<cpu_set_t> = vec![std::mem::zeroed(); num_cpus_avail];
        let mut tsc_probes: Vec<Vec<TscProbe>> =
            vec![vec![Default::default(); probes_count]; num_cpus_avail];

        // Initialize CPU sets
        let mut set_inx = 0;
        for cpu_id in 0..num_cpus as usize {
            if !libc::CPU_ISSET(cpu_id, cpu_constraint) {
                continue;
            }

            assert!(set_inx < num_cpus_avail);
            println!("CPU ID {} maps to CPU index {}", cpu_id, set_inx);
            libc::CPU_ZERO(&mut cpu_sets[set_inx]);
            libc::CPU_SET(cpu_id, &mut cpu_sets[set_inx]);

            set_inx += 1;
        }

        collect_cas_ordered_tsc_probes(num_cpus, &mut cpu_sets, &mut tsc_probes, probes_count)
            .context("Error while collecting CAS-ordered TSC probes")?;

        print_tsc_probe_sequence(&tsc_probes);

        let is_monotonic = is_probe_sequence_monotonic(&tsc_probes)
            .context("Error while testing monotonicity of the TSC values sequence")?;

        Ok(is_monotonic)
    }
}

pub struct CpuSwitchingEstimate {
    /// Estimated maximum shift between TSC counters running on different CPUs
    pub tsc_range_length: i64,
    /// Whether TSC values measured successively on same or different CPUs
    /// monotonically increase. If the function sets (*is_monotonic) to
    /// "false", that doesn't necessarily imply that TSCs are unreliable.
    /// In rare cases the observed non-monotonicity may be a result of TSC
    /// wrap that occured on one/several CPUs right before or just in the
    /// middle of the computations
    pub is_monotonic: bool,
}

/// Evaluate reliability of TSC (the required data is collected using a method of
/// CAS-Ordered Probes - the measurements are done by concurrently running threads; one per
/// each available CPU. The measurements are sequentially ordered by means of compare-and-
/// swap operation)
pub fn eval_tsc_reliability_cop() -> Result<CpuSwitchingEstimate, anyhow::Error> {
    let ps_state = ProcAndSysState::new()
        .context("Couldn't obtain details of the system and process state")?;

    let tsc_range_length = calc_tsc_enclosing_range_cop(
        ps_state.num_cpus,
        ps_state.initial_cpu_id,
        &ps_state.initial_cpu_set,
        ps_state.cline_size,
    )
    .context("Error while calculating enclosing TSC range")?;

    let is_monotonic = eval_tsc_monotonicity_cop(
        ps_state.num_cpus,
        &ps_state.initial_cpu_set,
        ps_state.cline_size,
    )
    .context("Error while evaluating TSC monotonicity")?;

    Ok(CpuSwitchingEstimate {
        tsc_range_length,
        is_monotonic,
    })
}
