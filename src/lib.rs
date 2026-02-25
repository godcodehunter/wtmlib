use anyhow::{bail, Context, Result};
use crate::config::*;

mod eval_tsc_reliability_cop;
mod conversion_params;
mod config;

// Re-export public types
pub use eval_tsc_reliability_cop::{eval_tsc_reliability_cop, CpuSwitchingEstimate};
pub use conversion_params::ConversionParams;

#[derive(Debug, Clone, Copy)]
pub struct Timestamp(pub u64);

/// Get time-stamp counter
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub fn get_tsc() -> Timestamp {
    unsafe {
        use std::arch::x86_64::_rdtsc;

        let ret = _rdtsc();
        Timestamp(ret)
    }
}

// Helper to compute absolute value of difference between two integer values
fn abs_diff(a_: f64, b_: f64) -> f64 {
    if a_ > b_ {
        a_ - b_
    } else {
        b_ - a_
    }
}

/// Structure to keep values of selected parameters that describe
/// - hardware state
/// - operating system state
/// - process state
struct ProcAndSysState {
    /// Number of configured logical CPUs in the system
    num_cpus: i32,
    /// Initial CPU ID (a CPU that the current thread is executing 
    /// on when WTM library is called) 
    initial_cpu_id: i32,
    /// Initial CPU set (a CPU set that the current thread 
    /// is confined to when WTM library is called) 
    initial_cpu_set: libc::cpu_set_t,
    /// Cache line size 
    cline_size: i32,
}

impl ProcAndSysState {
    /// Get cache line size
    /// 
    /// The implementation of this function is platform-specific. But it's expected to
    /// work seamlessly on Linux
    ///
    /// It's expected that WTMLIB will be executed on a system with homogenous CPUs (or
    /// on a system with a single CPU). Currently there is no mechanism in WTMLIB to
    /// resolve false memory sharing issues on systems with heterogenous CPUs
    ///
    /// The function returns either the cache line size or some negative value (in case
    /// of error)
    fn get_cache_line_size() -> Result<i64, anyhow::Error> {
        unsafe {
            // Get cache line size using "sysconf"
            const _SC_LEVEL1_DCACHE_LINESIZE: libc::c_int= 190;
            let cline_size = libc::sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
            if cline_size == -1 {
                bail!("\"sysconf()\" returned an error");
            }

            Ok(cline_size)
        }
    }

    /// Get values of selected parameters that describe:
    ///   - hardware state
    ///   - OS state
    ///   - current process state
    fn new() -> Result<Self, anyhow::Error> {
        unsafe {
            // Get the number of configured logical CPUs in the system (not all of them
            // may be available at the moment; e.g. some of them may be offline)
            let num_cpus = libc::sysconf(libc::_SC_NPROCESSORS_CONF) as i32;
            if num_cpus <= 0 {
                bail!("Couldn't get the number of configured CPUs");
            }

            let initial_cpu_id = libc::sched_getcpu();
            if initial_cpu_id < 0 {
                bail!("Couldn't get ID of the current CPU");
            }

            let thread_self = libc::pthread_self();

            // Get thread's affinity mask. We're going to test TSC values only on the CPUs allowed
            // by the current thread's affinity mask
            let mut initial_cpu_set: libc::cpu_set_t = std::mem::zeroed();

            let cpu_set_size = std::mem::size_of::<libc::cpu_set_t>();
            if libc::pthread_getaffinity_np(thread_self, cpu_set_size, &mut initial_cpu_set) != 0 {
                bail!("Couldn't get CPU affinity of the current thread");
            }

            let cline_size = Self::get_cache_line_size()
                .context("Error while obtaining cache line size")?;

            Ok(Self {
                num_cpus,
                initial_cpu_id,
                initial_cpu_set,
                cline_size: cline_size as i32,
            })
        }
    }
}

impl Drop for ProcAndSysState {
    /// Restore initial state of the current process
    ///
    /// Some WTM library functions substantially affect the state of the current process. Such
    /// functions should save the process' state at the very beginning (using a call to
    /// wtmlib_GetProcAndSystemState() function) and recover the initial state at the very end
    /// (using this function)
    fn drop(&mut self) {
        unsafe {
            // Restore CPU affinity of the current thread. We do that in two steps:
            // 1) first we move the current thread to the initial CPU
            // 2) then we confine the thread to the specified set of CPUs
            // The second step alone is not enough to recover the initial thread state. While the
            // specified CPU set indeed must include the initial CPU, it may also include some
            // other CPUs. Hence, if we do the second step only and omit the first one, the thread
            // may end up on a CPU that is different from the initial one. But we do want to
            // return it to the initial CPU, because before calling WTMLIB the application could
            // store some data in the cache of that CPU. Other side effects may also exist.
            // Technically, it's not guaranteed that if we do the first step, then after the
            // second step the thread will remain on the initial CPU. But we believe the
            // probability of that is pretty high
            let thread_self = libc::pthread_self();
            let cpu_set_size = std::mem::size_of::<libc::cpu_set_t>();
            let mut cpu_set: libc::cpu_set_t = std::mem::zeroed();

            libc::CPU_SET(self.initial_cpu_id as usize, &mut cpu_set);

            if libc::pthread_setaffinity_np(thread_self, cpu_set_size, &cpu_set) != 0 {
                eprintln!("Warning: Couldn't return the current thread to the initial CPU");
            }

            if libc::pthread_setaffinity_np(thread_self, cpu_set_size, &self.initial_cpu_set) != 0 {
                eprintln!("Warning: Couldn't restore CPU affinity of the current thread");
            }
        }
    }
}


// /// Macro that evaluates to 1 if the library should use non-negative return values. In
// /// the opposite case the macro evaluates to -1.
// /// The library takes care of its return values' sign to avoid interference with special
// /// return values. Only one such 'special value' is relevant for now. Namely,
// /// PTHREAD_CANCELED
// const WTMLIB_RET_SIGN = ((long)PTHREAD_CANCELED < 0 ? 1 : -1);

// /// Return value that indicates generic error
// const WTMLIB_RET_GENERIC_ERR = (WTMLIB_RET_SIGN * 1);

// /// Return value indicating that major TSC inconsistency was detected
// ///
// /// Getting this return value doesn't necessarily imply that TSC is unreliable and cannot
// /// be used as a wall-clock time measure. This type of error in some cases may be caused
// /// by TSC wrap (which could has happened on some CPU in the middle of calculations; or it
// /// might has happened on one CPU before WTMLIB was called but hasn't yet happened on the
// /// other available CPUs; and so on)
// const WTMLIB_RET_TSC_INCONSISTENCY = (WTMLIB_RET_SIGN * 2);

// /// Return value indicating that configured statistical significance criteria were not met
// /// (data collected by the library doesn't contain enough specific patterns)
// const WTMLIB_RET_POOR_STAT = (WTMLIB_RET_SIGN * 3);



/// Collect TSC values on all available CPUs in a carousel manner
///
/// The algorithm is the following:
///   - the current thread is first moved to the first CPU, and TSC is measured on that CPU
///   - then the same thread is moved to the second CPU, and TSC is measured there
///   - so on, until all available CPUs are visited (and thus the first round of the
///     carousel is completed)
///   - further rounds (if num_rounds > 1) are then undertaken in exactly the same way
///   - after the last round completes, the thread is moved to the first CPU again, and
///     TSC is measured there again. Thus, first and last measurements taken by the
///     function are taken on the same CPU
///
/// CPU affinity of the current thread may change as a result of calling this function
fn collect_tsc_in_cpu_carousel(
    cpu_sets: &[libc::cpu_set_t],
    tsc_vals: &mut [Vec<u64>],
    _num_cpus: i32,
    num_rounds: i64,
) -> Result<(), anyhow::Error> {
    unsafe {
        let thread_self = libc::pthread_self();
        let cpu_set_size = std::mem::size_of::<libc::cpu_set_t>();

        // Move current thread across all CPUs described by CPU masks referenced from
        // cpu_sets array (each element of the array references a CPU mask that represents
        // a single CPU). The thread is moved from one CPU to another in a carousel fashion.
        // And the carousel is run the specified number of times
        for i in 0..num_rounds as usize {
            for (ind, cpu_set) in cpu_sets.iter().enumerate() {
                let ret = libc::pthread_setaffinity_np(thread_self, cpu_set_size, cpu_set);
                if ret != 0 {
                    bail!("Couldn't change CPU affinity of the current thread");
                }
                tsc_vals[ind][i] = get_tsc().0;
            }
        }

        // Move the thread to the first CPU in the sequence, so that the first and last
        // TSC values are measured on the same CPU
        if libc::pthread_setaffinity_np(thread_self, cpu_set_size, &cpu_sets[0]) != 0 {
            bail!("Couldn't change CPU affinity of the current thread");
        }

        tsc_vals[0][num_rounds as usize] = get_tsc().0;

        Ok(())
    }
}

/// Generic evaluation of consistency of TSC values collected in a CPU carousel
fn check_carousel_vals_consistency(
    tsc_vals: &[Vec<u64>],
    num_rounds: i64,
) -> Result<(), anyhow::Error> {
    // Make sure that collected TSC values do vary on each of the CPUs. That may not be
    // true, for example, in case when some CPUs consistently return "zero" for every TSC
    // test.
    // This check is really important. If all TSC values are equal, then both TSC "delta"
    // ranges and TSC monotonicity will be perfect, but at the same time TSC would be
    // absolutely inappropriate for measuring wall-clock time.

    // Separate check for a CPU with index "zero". Because there must be an extra value
    // measured on this CPU
    if tsc_vals[0][0] == tsc_vals[0][num_rounds as usize] {
        bail!("First and last TSC values collected on CPU with index 0 are equal");
    }

    // Check TSC values collected on all other CPUs
    for i in 1..tsc_vals.len() {
        if tsc_vals[i][0] == tsc_vals[i][(num_rounds - 1) as usize] {
            bail!("First and last TSC values collected on CPU with index {} are equal", i);
        }
    }

    Ok(())
}

/// Calculate bounds of a shift between TSC on the given CPU and TSC on the base CPU
/// (assuming that TSC values were collected using CPU carousel method)
///
/// It's done in the following way:
///   1) for each TSC value T measured on the given CPU there are TSC values t1 and t2
///      measured right before and right after T, but on the base CPU
///   2) so, we know that when T was measured, TSC on the base CPU was somewhere between
///      t1 and t2. Let's denote that value of base TSC "t"
///   3) we're interested in a difference "delta = T - t"
///   4) since "t belongs to [t1, t2]", "delta belongs to [T - t2; T - t1]"
///   5) that's how we find a range for "delta" based on a single round of CPU carousel
///   6) but we (possibly) had multiple rounds of CPU carousel. Based on that, we can
///      narrow the range of possible "delta" values. To do that, we calculate the
///      intersection of all "delta" ranges calculated in different carousel rounds
///
/// Input: tsc_vals[0] - array of base TSC values collected during CPU carousel
///        tsc_vals[1] - array of TSC values collected on some other CPU during
///                      the same carousel
fn calc_tsc_delta_range_cpu_switch(
    tsc_vals: &[Vec<u64>],
    num_rounds: i64,
) -> Result<(i64, i64), anyhow::Error> {
    let mut d_min = i64::MIN;
    let mut d_max = i64::MAX;

    println!("Calculating shift between TSC counters of the two given CPUs...");

    check_carousel_vals_consistency(tsc_vals, num_rounds)?;

    for i in 0..num_rounds as usize {
        // Consistency check. Successive TSC values measured on the same CPU must
        // not decrease (unless TSC counter wraps)
        if tsc_vals[0][i + 1] < tsc_vals[0][i]
            || (i > 0 && tsc_vals[1][i] < tsc_vals[1][i - 1])
        {
            bail!("Detected decreasing successive TSC values (measured on the same CPU). That may be a result of TSC wrap");
        }

        // Check that we will not get overflow while subtracting TSC values
        let diff1 = u64::abs_diff(tsc_vals[1][i], tsc_vals[0][i]);
        let diff2 = u64::abs_diff(tsc_vals[1][i], tsc_vals[0][i + 1]);

        if diff1 > i64::MAX as u64 || diff2 > i64::MAX as u64 {
            bail!("Difference between TSC values measured on different CPUs is too big. May be a result of TSC wrap");
        }

        let bound_min = tsc_vals[1][i] as i64 - tsc_vals[0][i + 1] as i64;
        let bound_max = tsc_vals[1][i] as i64 - tsc_vals[0][i] as i64;

        assert!(bound_min <= bound_max);

        // "delta" ranges calculated for different carousel rounds must overlap.
        // Otherwise, we have a major TSC inconsistency and cannot rely on TSC while
        // measuring wall-clock time
        if bound_min > d_max || bound_max < d_min {
            bail!("TSC delta ranges calculated for different carousel rounds don't overlap. May be a result of some major inconsistency");
        }

        println!("The shift belongs to range: {} [{}, {}]",
            bound_max - bound_min, bound_min, bound_max);

        d_min = if bound_min > d_min { bound_min } else { d_min };
        d_max = if bound_max < d_max { bound_max } else { d_max };
    }

    assert!(d_min <= d_max);
    println!("Combined range (intersection of all the above): {} [{}, {}]",
        d_max - d_min, d_min, d_max);

    Ok((d_min, d_max))
}

/// Calculate size of enclosing TSC range (using data collected by means of "CPU
/// Switching" method)
///
/// "Size of enclosing TSC range" is a non-negative integer value such that if TSC
/// values are measured simultaneously on all the available CPUs, then difference
/// between the largest and the smallest will be not bigger than this value. In other
/// words, it's an estimated upper bound for shifts between TSC counters running on
/// different CPUs
fn calc_tsc_enclosing_range_cpu_switch(
    num_cpus: i32,
    base_cpu: i32,
    cpu_constraint: &libc::cpu_set_t,
) -> Result<i64, anyhow::Error> {
    unsafe {
        let mut l_bound = i64::MAX;
        let mut u_bound = i64::MIN;

        println!("Calculating an upper bound for shifts between TSC counters running on different CPUs...");
        println!("Base CPU ID: {}", base_cpu);

        // We add 1 to WTMLIB_CALC_TSC_RANGE_ROUND_COUNT, because we produce "num_rounds + 1"
        // samples for the first CPU. The last sample is always taken on the first CPU in the carousel
        let num_samples = (WTMLIB_CALC_TSC_RANGE_ROUND_COUNT + 1) as usize;
        let mut cpu_sets: Vec<libc::cpu_set_t> = vec![std::mem::zeroed(); 2];
        let mut tsc_vals: Vec<Vec<u64>> = vec![vec![0u64; num_samples]; 2];

        libc::CPU_ZERO(&mut cpu_sets[0]);
        libc::CPU_SET(base_cpu as usize, &mut cpu_sets[0]);
        libc::CPU_ZERO(&mut cpu_sets[1]);

        for cpu_id in 0..num_cpus {
            if !libc::CPU_ISSET(cpu_id as usize, cpu_constraint) || cpu_id == base_cpu {
                continue;
            }

            libc::CPU_SET(cpu_id as usize, &mut cpu_sets[1]);
            println!("\nRunning carousel for CPUs {} and {}...", base_cpu, cpu_id);

            collect_tsc_in_cpu_carousel(
                &cpu_sets,
                &mut tsc_vals,
                num_cpus,
                WTMLIB_CALC_TSC_RANGE_ROUND_COUNT,
            )
            .context("CPU carousel failed")?;

            println!("CPU ID {} maps to CPU index {}", base_cpu, 0);
            println!("CPU ID {} maps to CPU index {}", cpu_id, 1);

            let (delta_min, delta_max) = calc_tsc_delta_range_cpu_switch(
                &tsc_vals,
                WTMLIB_CALC_TSC_RANGE_ROUND_COUNT,
            )
            .context("Calculation of TSC delta range failed")?;

            // Update bounds of the enclosing TSC range
            l_bound = if l_bound > delta_min { delta_min } else { l_bound };
            u_bound = if u_bound < delta_max { delta_max } else { u_bound };

            assert!(delta_max >= delta_min && u_bound >= l_bound);
            // Return CPU mask to the "clean" state
            libc::CPU_CLR(cpu_id as usize, &mut cpu_sets[1]);
        }

        println!("\nShift between TSC on any of the available CPUs and TSC on the base CPU belongs to range: [{}, {}]", l_bound, u_bound);
        println!("Upper bound for shifts between TSCs is: {}", u_bound - l_bound);

        Ok(u_bound - l_bound)
    }
}

/// Check whether TSC values measured on different CPUs one after another monotonically
/// increase
///
/// The algorithm is the following:
///   1) move the current thread across all available CPUs in a carousel manner and measure
///      TSC value after each migration
///   2) check whether collected TSC values monotonically increase
///
/// NOTE: if the function reports that collected TSC values do not monotonically increase,
///       that doesn't necessarily imply that TSCs are unreliable. In some cases the
///       observed decrease may be a result of TSC wrap
fn eval_tsc_monotonicity_cpu_switch(
    num_cpus: i32,
    cpu_constraint: &libc::cpu_set_t,
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

        // We add 1 to WTMLIB_EVAL_TSC_MONOTCTY_ROUND_COUNT, because we produce
        // "num_rounds + 1" samples for the first CPU.
        let num_samples = (WTMLIB_EVAL_TSC_MONOTCTY_ROUND_COUNT + 1) as usize;
        let mut cpu_sets: Vec<libc::cpu_set_t> = vec![std::mem::zeroed(); num_cpus_avail];
        let mut tsc_vals: Vec<Vec<u64>> = vec![vec![0u64; num_samples]; num_cpus_avail];

        // Initialize CPU sets
        let mut set_inx = 0;
        for cpu_id in 0..num_cpus as usize {
            if !libc::CPU_ISSET(cpu_id, cpu_constraint) {
                continue;
            }

            assert!(set_inx < num_cpus_avail);
            libc::CPU_ZERO(&mut cpu_sets[set_inx]);
            libc::CPU_SET(cpu_id, &mut cpu_sets[set_inx]);
            println!("CPU index {} maps to CPU ID {}", set_inx, cpu_id);
            set_inx += 1;
        }

        collect_tsc_in_cpu_carousel(
            &cpu_sets,
            &mut tsc_vals,
            0, // num_cpus not used
            WTMLIB_EVAL_TSC_MONOTCTY_ROUND_COUNT,
        )
        .context("CPU carousel failed")?;

        check_carousel_vals_consistency(&tsc_vals, WTMLIB_EVAL_TSC_MONOTCTY_ROUND_COUNT)?;

        // Check whether collected TSC values monotonically increase
        let mut is_monotonic = true;
        let mut prev_tsc_val = tsc_vals[0][0];

        'outer: for round in 0..WTMLIB_EVAL_TSC_MONOTCTY_ROUND_COUNT as usize {
            // tsc_series may not be equal to CPU IDs. Refer to the initialization of CPU
            // sets above to understand how tsc_series relate to CPU IDs
            for tsc_series in 0..num_cpus_avail {
                if tsc_vals[tsc_series][round] < prev_tsc_val {
                    // This condition doesn't necessarily imply that TSCs are unreliable.
                    // Non-monotonic TSC sequence may be a result of TSC wrap
                    is_monotonic = false;
                    println!("Monotonic increase broke at carousel round {}, CPU index {}",
                        round, tsc_series);
                    break 'outer;
                }
                prev_tsc_val = tsc_vals[tsc_series][round];
            }
        }

        // Check the last TSC value which is always measured on the same CPU as the first
        // value. This last check is insignificant if WTMLIB_EVAL_TSC_MONOTCTY_ROUND_COUNT
        // is large. But it's critical if WTMLIB_EVAL_TSC_MONOTCTY_ROUND_COUNT == 1
        if is_monotonic {
            if tsc_vals[0][WTMLIB_EVAL_TSC_MONOTCTY_ROUND_COUNT as usize] < prev_tsc_val {
                // This condition doesn't necessarily imply that TSCs are unreliable.
                // Non-monotonic TSC sequence may be a result of TSC wrap
                is_monotonic = false;
            }
        }

        if is_monotonic {
            println!("The collected TSC values DO monotonically increase");
        }

        Ok(is_monotonic)
    }
}

/// Evaluate reliability of TSC (the required data is collected using "CPU Switching"
/// method - a single thread jumps from one CPU to another and takes all needed
/// measurements)
///
/// Returns:
///   - tsc_range_length: estimated maximum shift between TSC counters running on
///                       different CPUs
///   - is_monotonic: whether TSC values measured successively on same or different CPUs
///                   monotonically increase. If "false", that doesn't necessarily imply
///                   that TSCs are unreliable. In rare cases the observed non-monotonicity
///                   may be a result of TSC wrap that occurred on one/several CPUs right
///                   before or just in the middle of the computations
pub fn eval_tsc_reliability_cpu_switch() -> Result<CpuSwitchingEstimate, anyhow::Error> {
    println!("Evaluating TSC reliability (the required data is collected using \"CPU Switching\" method)...");
    let ps_state = ProcAndSysState::new()
        .context("Couldn't obtain details of the system and process state")?;

    let tsc_range_length = calc_tsc_enclosing_range_cpu_switch(
        ps_state.num_cpus,
        ps_state.initial_cpu_id,
        &ps_state.initial_cpu_set,
    )
    .context("Error while calculating enclosing TSC range")?;

    let is_monotonic = eval_tsc_monotonicity_cpu_switch(
        ps_state.num_cpus,
        &ps_state.initial_cpu_set,
    )
    .context("Error while evaluating TSC monotonicity")?;

    Ok(CpuSwitchingEstimate {
        tsc_range_length,
        is_monotonic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tsc_returns_nonzero() {
        let tsc = get_tsc();
        assert!(tsc.0 > 0, "TSC should return a non-zero value");
    }

    #[test]
    fn test_get_tsc_is_monotonic() {
        let tsc1 = get_tsc();
        let tsc2 = get_tsc();
        let tsc3 = get_tsc();

        assert!(
            tsc2.0 >= tsc1.0,
            "TSC should be monotonically increasing: {} >= {}",
            tsc2.0,
            tsc1.0
        );
        assert!(
            tsc3.0 >= tsc2.0,
            "TSC should be monotonically increasing: {} >= {}",
            tsc3.0,
            tsc2.0
        );
    }

    #[test]
    fn test_abs_diff() {
        assert_eq!(abs_diff(5.0, 3.0), 2.0);
        assert_eq!(abs_diff(3.0, 5.0), 2.0);
        assert_eq!(abs_diff(5.0, 5.0), 0.0);
    }

    #[test]
    fn test_proc_and_sys_state_new() {
        let result = ProcAndSysState::new();
        assert!(result.is_ok(), "ProcAndSysState::new() should succeed");

        let state = result.unwrap();
        assert!(state.num_cpus > 0, "Should have at least 1 CPU");
        assert!(
            state.initial_cpu_id >= 0,
            "Initial CPU ID should be non-negative"
        );
        assert!(state.cline_size > 0, "Cache line size should be positive");
    }

    #[test]
    fn test_check_carousel_vals_consistency_valid() {
        // Create valid TSC values that vary
        let tsc_vals = vec![
            vec![100u64, 200, 300, 400, 500],
            vec![150u64, 250, 350, 450, 550],
        ];
        let result = check_carousel_vals_consistency(&tsc_vals, 4);
        assert!(result.is_ok(), "Should pass for varying TSC values");
    }

    #[test]
    fn test_check_carousel_vals_consistency_invalid_first_cpu() {
        // Create invalid TSC values where first CPU has equal first and last
        let tsc_vals = vec![
            vec![100u64, 200, 300, 400, 100], // First and last are equal
            vec![150u64, 250, 350, 450, 550],
        ];
        let result = check_carousel_vals_consistency(&tsc_vals, 4);
        assert!(result.is_err(), "Should fail when first CPU has equal first/last TSC");
    }

    #[test]
    fn test_timestamp_clone() {
        let ts1 = Timestamp(12345);
        let ts2 = ts1;
        assert_eq!(ts1.0, ts2.0);
    }
}












