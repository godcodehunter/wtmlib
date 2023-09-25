use crate::*;

/// A set of parameters used to convert TSC ticks into nanoseconds in a fast and
/// accurate way
pub struct ConversionParams {
    /// A multiplier: (tsc_remainder * mult)
    mult: u64,
    
    /// A shift: nsecs_per_tsc_remainder = (tsc_remainder * mult) >> shift
    shift: u64,
    
    /// Number of nanoseconds per TSC modulus: nsecs_per_tsc_moduli =
    /// (tsc_ticks >> tsc_remainder_length) * nsecs_per_tsc_modulus
    nsecs_per_tsc_modulus: u64,
    
    /// Length of TSC remainder in bits. Used to calculate the quotient of TSC ticks
    /// divided by TSC modulo: (tsc_ticks >> tsc_remainder_length)
    tsc_remainder_length: i64,

    /// A bitmask used to extract TSC remainder:
    /// tsc_remainder = tsc_ticks & tsc_remainder_bitmask
    tsc_remainder_bitmask: u64,

    /// Number of TSC ticks per second. The parameter is intended for clients willing to
    /// convert TSC ticks to (nano)seconds by means of simple division (either integer or
    /// floating-point
    tsc_ticks_per_sec: u64,
}

impl ConversionParams {
    /// Calculate parameters that aid in converting measured TSC ticks into nanoseconds. Given
    /// these parameters the TSC ticks can be converted into human-friendly time format using
    /// fast division-free integer arithmetic
    ///
    /// Let us use an example to explain how time measured in TSC ticks is converted into
    /// nanoseconds inside this library (the procedure described below is borrowed from FIO).
    ///
    /// Ideally, we'd like to do the following: ns_time = tsc_ticks / tsc_per_ns
    /// Also we'd like to use only integer arithmetic to keep conversion overhead low. The
    /// lower is the overhead the less it affects valuable computations.
    /// If tsc_per_ns = 3, then simple integer division works fine: ns_time = tsc_ticks / 3
    /// looks good. But what if tsc_per_ns = 3.333? The accuracy will be really poor if 3.333
    /// is rounded down to 3.
    /// We can mitigate this problem in the following way:
    ///      ns_time = (tsc_ticks * factor) / (3.333 * factor)
    /// If "factor" is "big enough" then accuracy must be good. What is not really good though,
    /// is conversion overhead. Integer division is a pretty expensive operation (it takes 10+
    /// clocks on x86 CPUs at the moment of writing; even worse, integer division operations
    /// may not be pipelined in some cases).
    /// Let's rewrite our expression in an equivalent form:
    ///      ns_time = (tsc_ticks * factor / 3.333) / factor
    /// First division is not a problem here. We can precompute (factor / 3.333) in advance.
    /// But the second division is still pain. To deal with it, let's choose "factor" to be
    /// equal to a power of 2. After that we can replace the last division by bitwise shift
    /// which is really fast. Must work well as long as "factor" is "big enough".
    /// But the problem is "factor" cannot be arbitrarily big. Namely, multiplication in the
    /// numerator must not overflow 64-bit integer type (we want to stay using built-in types
    /// because using wide arithmetic would negatively affect performance).
    /// Let's see how big "factor" can be in our example. Assume that we want our conversion
    /// to be valid for time periods up to 1 year. There is the following amount of TSC ticks
    /// in 1 year: 3.333 * 1000000000 * 60 * 60 * 24 * 365 = 105109488000000000. Dividing the
    /// maximum 64-bit value by this number, we get:
    /// 18446744073709551615 / 105109488000000000 ~ 175.5. Thus, (factor / 3.333) cannot be
    /// bigger than this value: factor <= 175.5 * 3.333 ~ 584.9. The biggest power of 2 that
    /// doesn't exceed this value is 512. Hence, our conversion formula takes the form:
    ///     ns_time = (tsc_ticks * 512 / 3.333) / 512
    /// Remember, that we want to keep (512 / 3.333) precomputed. Taking that into account:
    ///     ns_time = tsc_ticks * 153 / 512
    /// Ok, let's evaluate how accurate this conversion formula is. There are
    /// 1000000000 * 60 * 60 * 24 * 365 = 31536000000000000 nanoseconds in 1 year. While our
    /// formula gives 105109488000000000 * 153 / 512 = 31409671218750000 nanoseconds. The
    /// difference with the actual value is 126328781250000 nanoseconds which is
    /// 126328781250000 / 1000000000 / 60 / 60 ~ 35 hours.
    /// The error is pretty big. We want to do better.
    /// What if we don't need to measure time periods longer than an hour? How big can "factor"
    /// be in this case? Number of TSC ticks in one hour:
    /// 3.333 * 1000000000 * 60 * 60 = 11998800000000. Dividing the maximum 64-bit value by
    /// this number we get: 18446744073709551615 / 11998800000000 ~ 1537382.4. Thus,
    /// factor <= 1537382.4 * 3.333 ~ 5124095.5. The biggest power of 2 that doesn't exceed
    /// this value is 4194304. Hence, the conversion formula takes the form:
    ///    ns_time = (tsc_ticks * 4194304 / 3.333) / 4194304 = tsc_ticks * 1258417 / 4194304
    /// Let's evaluate its accuracy. There are 1000000000 * 60 * 60 = 3600000000000
    /// nanoseconds in 1 hour. And our formula gives 11998800000000 * 1258417 / 4194304 =
    /// = 3599999880695 nanoseconds. The absolute error is just 119305 nanoseconds (less
    /// than 0.2 milliseconds per hour). Which is really good.
    /// Now we can notice the following:
    ///      tsc_ticks = (tsc_ticks_per_1_hour * number_of_hours) + tsc_ticks_remainder
    /// If we pre-calculate tsc_ticks_per_1_hour than we will be able to extract
    /// number_of_hours from tsc_ticks. Next, we know how many nanoseconds are in 1 hour. Thus,
    /// to complete conversion of tsc_ticks to nanoseconds it remains only to convert
    /// tsc_ticks_remainder to nanoseconds. To do that we can use the procedure described
    /// above. We know that the conversion error will be really small (since
    /// tsc_ticks_remainder represents a time period shorter than 1 hour).
    /// This is the conversion mechanism that we're really going to use. Let's now generalize
    /// and optimize the whole procedure. First of all, we'd like to have a flexible control
    /// over the conversion error. Hence, we don't stick with 1 hour when calculating the
    /// conversion parameters but use a configurable time period. We call this period "time
    /// conversion modulus" (by analogy with modular arithmetic).
    ///      tsc_ticks = (tsc_ticks_per_time_conversion_modulus * number_of_moduli_periods) +
    ///                  + tsc_ticks_remainder
    /// To convert tsc_ticks_remainder to nanoseconds we'll use the already familiar formula:
    ///      ns_per_remainder = (tsc_ticks_remainder * factor / tsc_per_nsec) / factor
    /// tsc_ticks_per_time_conversion_modulus * (factor / tsc_per_nsec) <= UINT64_MAX
    /// factor <= (UINT64_MAX / tsc_ticks_per_time_conversion_modulus) * tsc_per_nsec
    /// We choose the largest "shift" that satisfies the inequality:
    /// 2 ^ shift <= (UINT64_MAX / tsc_ticks_per_time_conversion_modulus) * tsc_per_nsec
    /// Now:
    ///      factor = 2 ^ shift
    /// Next, we precompute the "multiplier":
    ///      mult = factor / tsc_per_nsec
    /// After that the nanosecond worth of tsc_ticks_remainder can be calculated as follows:
    ///      ns_per_remainder = (tsc_ticks_remainder * mult) >> shift
    /// NOTE: in the code below instead of tsc_per_nsec we use (tsc_per_sec / 1000000000).
    ///       E.g. mult = factor * 1000000000 / tsc_per_sec
    /// NOTE: ideally we need the largest "factor" satisfying the following inequality:
    ///       tsc_ticks_per_time_conversion_modulus * (factor / tsc_per_nsec) <= UINT64_MAX.
    ///       But in reality - as you can see above - we choose the largest value satisfying:
    ///       factor <= (UINT64_MAX / tsc_ticks_per_time_conversion_modulus) * tsc_per_nsec
    ///       In integer arithmetic these inequalities are not equivalent. And the largest
    ///       value satisfying the first inequality can be bigger than the largest value
    ///       satisfying the second inequality. Though it's not clear whether this effect
    ///       can affect the final result of our calculations: we need not just the largest
    ///       value satisfying the inequality but the largest power of 2, also tsc_per_nsec is
    ///       not arbitrary but (tsc_per_sec / 1000000000).
    ///       Anyway, we don't care of that for now. Even if we choose "factor" 2 times smaller
    ///       than the maximum allowed, that shouldn't be a problem as long as the value still
    ///       remains "big enough" (which must be true if "time conversion modulus" stays in
    ///       reasonable bounds).
    /// The next problem to solve is extraction of tsc_ticks_remainder and
    /// number_of_moduli_periods from tsc_ticks. Again, we'd like to do that fast. To avoid
    /// division, instead of using tsc_ticks_per_time_conversion_modulus, we will use the
    /// largest power of 2 that doesn't exceed tsc_ticks_per_time_conversion_modulus. We call
    /// this value "TSC modulus". Thus:
    ///      tsc_ticks = (tsc_modulus * number_of_tsc_moduli_periods) + tsc_modulus_remainder
    /// Since tsc_modulus is a power of 2, extraction of number_of_tsc_moduli_periods and
    /// tsc_modulus_remainder from tsc_ticks can be done easily: by doing a bit shift and
    /// applying a bit mask. The power of 2 used to produce tsc_modulus is effectively a bit
    /// length of tsc_modulus_remainder. Thus:
    ///      tsc_modulus = 2 ^ tsc_modulus_remainder_bit_length
    ///      number_of_tsc_moduli_periods = tsc_ticks >> tsc_modulus_remainder_bit_length
    ///      tsc_modulus_remainder = tsc_ticks & (tsc_modulus - 1)
    /// We already know how to convert tsc_modulus_remainder to nanoseconds. To convert TSC
    /// moduli's worth of ticks to nanoseconds we do:
    ///      ns_per_moduli = ns_per_tsc_modulus * number_of_tsc_moduli_periods
    /// Since ns_per_tsc_modulus represents a time period which is not longer than "time
    /// conversion modulus", then we can pre-calculate this value using the same formula as we
    /// use for converting tsc_modulus_remainder to nanoseconds:
    ///     ns_per_tsc_modulus = (tsc_modulus * mult) >> shift
    /// That's it. Now we've fully described our two-stage conversion procedure. In summary,
    /// the procedure is:
    ///     1) convert "tsc moduli's" worth of ticks to nanoseconds using simple
    ///        multiplication
    ///     2) convert the remaining ticks using multiply-shift arithmetic
    ///     3) summ up the results of the above conversions
    /// NOTE: in the code below we use "time conversion modulus" to calculate "mult" and
    ///      "shift". Instead, we could use "TSC modulus" and get slightly better accuracy in
    ///      some cases (because "TSC modulus" corresponds to a time period which is not
    ///      longer than "time conversion modulus"). But we don't do that. We want the
    ///      accuracy to be driven by easily understood "time conversion modulus" which is
    ///      measured in human-friendly seconds.
    ///
    fn from_ratio(tsc_per_sec: u64) -> Result<Self, anyhow::Error> {
        println!("Calculating TSC-to-nanoseconds conversion parameters...");
        
        if u64::MAX / WTMLIB_TIME_CONVERSION_MODULUS < tsc_per_sec {
            bail!("Configured time conversion modulus is too big. TSC worth of this period doesn't fit 64-bit cell");
        }

        let tsc_worth_of_modulus = WTMLIB_TIME_CONVERSION_MODULUS * tsc_per_sec;
        let mult_bound = u64::MAX / tsc_worth_of_modulus;
        // Multiplication here will not produce overflow because tsc_per_sec is smaller than tsc_worth_of_modulus 
        let mut factor_bound = mult_bound * tsc_per_sec / 1000000000;
        let mut shift: u64 = 0;
        
        // Find "factor": the largest power of 2 that does not exceed factor_bound
        while factor_bound > 1 {
            factor_bound >>= 1;
            shift += 1;
        }
        
        let factor = (1 as u64) << shift;
        // Cannot get overflow here. By calculation this value is smaller than mult_bound 
        let mult = factor * 1000000000 / tsc_per_sec;

        println!("Shift: {}, multiplicator: {}", shift, mult);

        // Find the largest power of 2 that doesn't exceed tsc_worth_of_modulus. This number
        // will play a role of "time conversion modulus" but in terms of TSC
        // (WTMLIB_TIME_CONVERSION_MODULUS is measured in seconds)
        let mut tsc_remainder_length = 0;

        while (tsc_worth_of_modulus >> tsc_remainder_length) > 1 {
            tsc_remainder_length+=1;
        }

        let tsc_modulus = (1 as u64) << tsc_remainder_length;

        // Here we could use (tsc_modulus * 1000000000) / tsc_per_sec. But in this case the
        // nanosecond worth of the last tick of every TSC modulus period would be excessively
        // high compared to the worth of any other tick inside the same period. Though, seems
        // that overall accuracy would be slightly better.
        // Current decision is to keep accuracy at the same level for all measurements
        // (instead of "recovering" it a little bit after each TSC modulus period). In this
        // case the nanosecond worth of very similar TSC ranges will always be consistent.
        // Hence, we calculate the nanosecond worth of TSC modulus period using exactly the
        // same formula as we use to calculate the nanosecond worth of TSC remainder
        let nsecs_per_tsc_modulus = (tsc_modulus * mult) >> shift;
        // Calculate a bitmask used to extract TSC remainder. Applying this bitmask is the
        // same as doing (tsc_ticks % tsc_modulus)
        let tsc_remainder_bitmask = tsc_modulus - 1;

        println!("TSC modulus: {}", tsc_modulus);
        println!("Nanoseconds per TSC modulus: {}", nsecs_per_tsc_modulus);
        println!("Bitmask to extract TSC remainder: {}", tsc_remainder_bitmask);

        Ok(Self {
            mult,
            shift,
            nsecs_per_tsc_modulus,
            tsc_remainder_length,
            tsc_remainder_bitmask,
            tsc_ticks_per_sec: tsc_per_sec,
        })
    }

    /// Calculate how TSC changes during a second
    ///
    /// At first, it's measured how TSC changes during the specified period of time.
    /// Then TSC-ticks-per-second is calculated based on the measured value
    fn calc_tsc_count_per_second(time_period_usecs: u64) -> Result<u64, anyhow::Error> {
        use std::time::Instant;
        
        let mut end_time;
        let mut end_tsc_val;
        
        // We first measure the start time and then start TSC value. The end values of
        // time and TSC must be measured in the same order. That's because there exists
        // a time gap between the start measurements. We don't know the value of this gap
        // but we can - at least partially - compensate for the gap. We expect that the
        // gap will be more or less the same each time we collect time and TSC values in
        // the fixed order.
        // Also we ensure that TSC and time values are measured one right after another.
        // There must be no other operations in-between. E.g. we check the return value
        // of 'clock_gettime()' only after the corresponding TSC value is measured
        let start_time = Instant::now();
        let start_tsc_val = get_tsc();
        
        loop {
            end_time = Instant::now();
            end_tsc_val = get_tsc();

            if (end_time - start_time).as_nanos() 
                > (time_period_usecs * 1000) as u128 
            { 
                    break; 
            }
        }

        // Possibly TSC wrap has happened. But we don't guess here, just report
        // the observed inconsistency as an error
        if start_tsc_val.0 >= end_tsc_val.0 {
            bail!(
                "End TSC value ({}) is smaller then or equal to start TSC value ({}). TSC wrap might has happened", 
                start_tsc_val.0, 
                end_tsc_val.0,
            );
        }

        // Well, the difference may be big not only because of TSC inconsistency but also
        // because the elapsed time period is indeed very long. Nevertheless, we assume
        // here that configuration parameters of the library are within "sane" bounds
        if u64::MAX / 1000000000 < end_tsc_val.0 - start_tsc_val.0 {
            bail!("Difference between end and start TSC values is too big (%lu)");
        }

        let res = (end_tsc_val.0 - start_tsc_val.0) * 1000000000 / (end_time - start_time).as_nanos() as u64;
        Ok(res)
    }

    /// Given a series of TSC-per-sec values and using some basic statistics concepts,
    /// calculate a single TSC-per-sec value which would be free from random "noise"
    ///
    /// Relating TSC changes to system time changes is tricky because TSC and system time
    /// cannot be measured simultaneously (at least on the same CPU). Some factors that can
    /// affect the time gap between the measurements are: system call overhead, interrupts,
    /// migration to a different CPU, and context switches. Here we assume that measurement
    /// errors caused by these negative effects have normal distribution. Based on this
    /// assumption, we first filter out statistical ouliers and then caclulate an average of
    /// the remaining values. This method is borrowed from FIO (where it is used exactly in
    /// the same way at the moment of writing).
    ///
    /// There are some reasons to expect that errors of the measurements are indeed random:
    ///    this is how we do measurements (at conceptual level):
    ///       1) measure system time
    ///       2) measure TSC
    ///       3) wait for a specified period of time
    ///       4) measure system time
    ///       5) measure TSC
    ///    I.e. end system time and TSC values are measured in the same order as start
    ///    system time and TSC values. Ideally, we would like to measure system time and
    ///    TSC simultaneously. But in practice there exists a delay between these two
    ///    measurements. In most cases this delay will be caused by the system call
    ///    overhead. We expect that the named overhead will be more or less the same in
    ///    both cases: while measuring start values and also while measuring end values.
    ///    Thus, if we use the ordering shown above, then we can expect that time period
    ///    between start and end TSC measurements will be more or less the same as time
    ///    period between start and end system time measurements. Ideally, it would be
    ///    exactly the same. But in practice, ratio of these two time periods can move to
    ///    both sides from "one": sometimes one period will be longer and sometimes the
    ///    other period will be longer. We expect these deviations from the mean value to
    ///    be more or less random. All other negative effects (like interrupts and CPU
    ///    migrations) can also move the ratio to both sides unpredictably and, thus, are
    ///    also considered random.
    ///
    /// Alternatively, we could take the measurements in the following way:
    ///       1) measure system time
    ///       2) measure TSC
    ///       3) wait for a specified period of time
    ///       4) measure TSC
    ///       5) measure system time
    ///    Compared to the first approach, the ordering of the last two operations is
    ///    reverted here. With this approach, one needs to hold several rounds of
    ///    measurements and identify a round for which
    ///    (TSC_end - TSC_start)/(time_end - time_start) is the biggest. Such a round is
    ///    basically "the best". One can expect that during this round the non-simultaneity
    ///    of TSC and system time measurements was affected only by THE MINIMAL POSSIBLE
    ///    overhead of the system calls. Thus, to complete establishing "TSC<->system time"
    ///    relation one needs to deal somehow with "the minimal possible overhead" of the
    ///    time-measuring system calls. That could be done in - at least - two ways:
    ///       1) wait for a sufficiently long period of time between start and end
    ///          measurements (so that system call overhead becomes negligible compared to
    ///          the overall duration of the experiment)
    ///       2) calculate "the minimal possible overhead" somehow
    ///
    fn calc_golden(samples: &[u64]) -> Result<u64, anyhow::Error> {
        println!("\"Cleaning\" collected TSC-per-second values from random noise");
        
        // Calculate "mean" and "standard deviation" of TSC-per-second observable value
        // We use incremental formulas for computing both. Classical formulas are less
        // stable. E.g. classical formula for calculating "mean" suffers from the
        // necessity to summ up all the data points. That can result in overflow
        // (especially when data set is large). Though, we need to admit that overflow
        // is very unlikely in our case. Because to collect a lot of data points one
        // needs to spend a lot of time measuring time intervals. Which not a very good
        let mut mean = 0.0;
        let mut s = 0.0;
        let mut delta = 0.0;

        for (i, sample) in samples.into_iter().enumerate() {
            delta = *sample as f64 - mean;
            mean += delta / (i as f64 + 1.0);
            s += delta * (*sample as f64 - mean)
        }

        let sigma = 0.0;
        let mut max_sample: u64 = 0;
        let mut min_sample: u64 = u64::MAX;
        let mut num_good_samples = 0;
        let mut average = 0;

        // We use "corrected sample standard deviation" here, and thus, "S" is divided
        // not by the number of samples but by the number of samples minus 1 
        let sigma = if samples.len() > 1 {
            (s / (samples.len() as f64 - 1.0)).sqrt()
        } else {
            s.sqrt()
        };

        // Find minimum and maximum samples
        max_sample = samples.into_iter().max().unwrap().clone();
        min_sample = samples.into_iter().min().unwrap().clone();

        // Filter out statistical outliers and calculate an average 
        for sample in samples {
            if abs_diff(*sample as f64, mean) > sigma {
                continue;
            }

            num_good_samples += 1;

            // Samples can be pretty big (though, it's very-very unlikely). 
            // We don't want to get overflow while calculating their 
            // cumulative summ. That's why we summ up not them but their 
            // distances from the minimum sample */
            // Still, check that we will not get overflow...
            if u64::MAX - average < sample - min_sample {
                bail!("Got overflow while calculating an average of \"good\" samples");
            }

            average += sample - min_sample; 
        }

        average /= num_good_samples;
        // Take into account that the cumulative summ was "shifted" by
        // "num_good_samples * min_sample"
        // Cannot get overflow here (an average cannot be bigger than the maximum sample) 
        average += min_sample;
        println!("Minimum sample: {}, maximum sample: {}", min_sample, max_sample);
        println!("Mean: {}, corrected sample standard deviation: {}", mean, sigma);
        println!("Average \"cleaned\" from statistical noise: {}", average);
        
        Ok(average)
    }

    /// Calculate parameters needed to perform fast and accurate conversion of 
    /// TSC ticks to nanoseconds. Also calculate time (in seconds) remaining 
    /// before the earliest TSC wrap
    pub fn new() -> Result<ConversionParams, anyhow::Error> {
        println!("Calculating TSC-to-nanoseconds conversion parameters...");
        
        // Allocate array to keep tsc-per-second values calculated in different 
        // experiments
        let mut tsc_per_sec = [0u64; WTMLIB_TSC_PER_SEC_SAMPLE_COUNT];
        
        println!("Calculating how TSC changes during a second-long time period");
        for i in 0..WTMLIB_TSC_PER_SEC_SAMPLE_COUNT {
            tsc_per_sec[i] = Self::calc_tsc_count_per_second(
                WTMLIB_TIME_PERIOD_TO_MATCH_WITH_TSC
            )
                .context("Error while calculating TSC worth of a second")?;
            println!(
                "[Measurement {}] TSC ticks per sec: {}", 
                i, 
                tsc_per_sec[i],
            );
        }
        
        let golden = Self::calc_golden(&tsc_per_sec)
            .context("Error while \"cleaning\" TSC-per-second samples from random noise")?;
        
        let conv_params = Self::from_ratio(golden)
            .context("Error while calculating TSC-to-nanoseconds conversion parameters")?;
        
        Ok(conv_params)
    }

    /// Convert TSC ticks to nanoseconds
    /// 
    /// REALLY IMPORTANT: for the conversion to be fast, it must be ensured that the
    /// structure with the conversion parameters always stays in cache
    pub fn convert_to_nanosec(&self, ticks: Timestamp) -> u64 {
        (ticks.0 >> self.tsc_remainder_length) * self.nsecs_per_tsc_modulus
        + ((ticks.0 & self.tsc_remainder_bitmask) * (self.mult) >> self.shift)
    }
}

/// Calculate time (in seconds!) before the earliest TSC wrap
///
/// All available CPUs are considered when calculating the time
pub fn time_before_tsc_wrap(cp: &ConversionParams) -> Result<u64, anyhow::Error> {
unsafe {
    println!("Calculating time before the earliest TSC wrap...");

    let mut ps_state = ProcAndSysState::default();
    let mut max_tsc_val = Timestamp(0);
    let mut curr_tsc_val;
    let secs_before_wrap;
    get_proc_and_system_state(&mut ps_state)
        .context("Couldn't obtain details of the system and process state")?;
    
    let cpu_id = -1;
    let thread_self = std::thread::current().id().as_u64().get();
    let cpu_set_size = libc::CPU_ALLOC_SIZE(ps_state.num_cpus);
    let cpu_set = libc_miss::CPU_ALLOC(ps_state.num_cpus);

    libc_miss::CPU_ZERO_S(cpu_set_size, cpu_set);
    
    for cpu_id in 0..ps_state.num_cpus {
        if libc_miss::CPU_ISSET_S( cpu_id, cpu_set_size, ps_state.initial_cpu_set) == 0 {
            continue; 
        }     

        libc_miss::CPU_SET_S(cpu_id, cpu_set_size, cpu_set);

        if libc::pthread_setaffinity_np(thread_self, cpu_set_size, cpu_set) != 0 {
            libc_miss::CPU_FREE(cpu_set);
            bail!("Couldn't change CPU affinity of the current thread");
        }

        curr_tsc_val = get_tsc();
        println!("TSC on CPU {}: {}", cpu_id, curr_tsc_val.0);

        if curr_tsc_val.0 > max_tsc_val.0 {
            max_tsc_val = curr_tsc_val;
        }

        /* Return CPU mask to the "clean" state */
        libc_miss::CPU_CLR_S( cpu_id, cpu_set_size, cpu_set);
    }
    
    libc_miss::CPU_FREE(cpu_set);

    println!("The maximum TSC value: {}", max_tsc_val.0);
    secs_before_wrap = cp.convert_to_nanosec(Timestamp(u64::MAX - max_tsc_val.0)) / 1000000000;
    println!("Seconds before the maximum TSC will wrap: {}", secs_before_wrap);
    
    // This error is not critical. But we do treat it as critical. See
    // the detailed comment to the identical condition inside
    // "inspect_cpu_switching" function
    restore_initial_proc_state(&ps_state)
        .context("Couldn't restore initial state of the current process")?;

    Ok(secs_before_wrap)
}
}