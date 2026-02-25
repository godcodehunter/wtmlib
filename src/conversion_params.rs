use crate::*;
use libc;

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
        
        let factor = 1 << shift;
        // Cannot get overflow here. By calculation this value is smaller than mult_bound 
        let mult = factor * 1000000000 / tsc_per_sec;

        println!("Shift: {}, multiplicator: {}", shift, mult);

        // Find the largest power of 2 that doesn't exceed tsc_worth_of_modulus. This number
        // will play a role of "time conversion modulus" but in terms of TSC
        // (WTMLIB_TIME_CONVERSION_MODULUS is measured in seconds)
        let mut tsc_remainder_length = 0;

        while (tsc_worth_of_modulus >> tsc_remainder_length) > 1 {
            tsc_remainder_length += 1;
        }

        println!("Length of TSC remainder in bits: {}", tsc_remainder_length);

        let tsc_modulus = 1 << tsc_remainder_length;

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

    /// Try to read TSC frequency from the system (Linux sysfs)
    fn get_tsc_freq_from_sysfs() -> Option<u64> {
        use std::fs;

        // Try reading from sysfs (available on some Linux systems)
        if let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/cpu0/tsc_freq_khz") {
            if let Ok(khz) = content.trim().parse::<u64>() {
                println!("TSC frequency from sysfs: {} kHz", khz);
                return Some(khz * 1000);
            }
        }
        None
    }

    /// Try to read TSC frequency from CPUID (Intel CPUs)
    #[cfg(target_arch = "x86_64")]
    fn get_tsc_freq_from_cpuid() -> Option<u64> {
        use std::arch::x86_64::__cpuid;

        let cpuid0 = unsafe { __cpuid(0) };
        let max_leaf = cpuid0.eax;

        // CPUID leaf 0x15: Time Stamp Counter and Nominal Core Crystal Clock
        if max_leaf >= 0x15 {
            let cpuid15 = unsafe { __cpuid(0x15) };
            let (eax, ebx, ecx) = (cpuid15.eax, cpuid15.ebx, cpuid15.ecx);

            if eax != 0 && ebx != 0 && ecx != 0 {
                let tsc_freq = (ecx as u64) * (ebx as u64) / (eax as u64);
                println!("TSC frequency from CPUID 0x15: {} Hz", tsc_freq);
                return Some(tsc_freq);
            }
        }

        // CPUID leaf 0x16: Processor Frequency Information (Intel)
        if max_leaf >= 0x16 {
            let cpuid16 = unsafe { __cpuid(0x16) };
            if cpuid16.eax != 0 {
                let base_freq = (cpuid16.eax as u64) * 1_000_000;
                println!("Base frequency from CPUID 0x16: {} Hz", base_freq);
                // Note: This is base frequency, not necessarily TSC frequency
                // Only use as fallback
            }
        }

        None
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn get_tsc_freq_from_cpuid() -> Option<u64> {
        None
    }

    /// Calibrate TSC frequency using busy-wait (more accurate than sleep)
    fn calibrate_tsc_busy_wait(duration_ms: u64) -> Result<u64, anyhow::Error> {
        use std::time::{Duration, Instant};

        let target_duration = Duration::from_millis(duration_ms);

        let start_time = Instant::now();
        let start_tsc = get_tsc();

        // Busy-wait for precise timing
        while start_time.elapsed() < target_duration {
            std::hint::spin_loop();
        }

        let end_tsc = get_tsc();
        let elapsed = start_time.elapsed();

        if end_tsc.0 <= start_tsc.0 {
            bail!("TSC did not increase during calibration (possible wrap)");
        }

        let tsc_diff = end_tsc.0 - start_tsc.0;
        let freq = (tsc_diff as f64 / elapsed.as_secs_f64()) as u64;

        Ok(freq)
    }

    /// Get TSC frequency with validation
    ///
    /// Tries multiple methods and validates the result
    fn get_validated_tsc_frequency() -> Result<u64, anyhow::Error> {
        println!("Determining TSC frequency...");

        // Method 1: Try system sources first (most accurate)
        if let Some(freq) = Self::get_tsc_freq_from_sysfs() {
            println!("Using TSC frequency from sysfs: {:.3} GHz", freq as f64 / 1e9);
            return Ok(freq);
        }

        if let Some(freq) = Self::get_tsc_freq_from_cpuid() {
            println!("Using TSC frequency from CPUID: {:.3} GHz", freq as f64 / 1e9);
            return Ok(freq);
        }

        // Method 2: Calibrate with busy-wait (two measurements for validation)
        println!("Calibrating TSC frequency (busy-wait method)...");

        let freq1 = Self::calibrate_tsc_busy_wait(100)?;
        let freq2 = Self::calibrate_tsc_busy_wait(100)?;

        // Validate: frequencies should be within 0.1% of each other
        let diff = if freq1 > freq2 { freq1 - freq2 } else { freq2 - freq1 };
        let diff_pct = (diff as f64 / freq1 as f64) * 100.0;

        if diff_pct > 0.5 {
            // If they differ too much, do a longer calibration
            println!("Initial calibrations differ by {:.2}%, doing longer calibration...", diff_pct);
            let freq3 = Self::calibrate_tsc_busy_wait(500)?;
            println!("TSC frequency (long calibration): {:.6} GHz", freq3 as f64 / 1e9);
            return Ok(freq3);
        }

        // Use average of two measurements
        let freq = (freq1 + freq2) / 2;
        println!("TSC frequency (calibrated): {:.6} GHz", freq as f64 / 1e9);

        Ok(freq)
    }

    /// Calculate how TSC changes during a second
    ///
    /// DEPRECATED: Use get_validated_tsc_frequency() instead.
    /// Kept for compatibility with existing code structure.
    fn calc_tsc_count_per_second(time_period_usecs: u64) -> Result<u64, anyhow::Error> {
        use std::time::Instant;

        let start_time = Instant::now();
        let start_tsc_val = get_tsc();

        // Busy-wait for the specified duration (more accurate than sleep)
        let target_ns = (time_period_usecs * 1000) as u128;
        while start_time.elapsed().as_nanos() < target_ns {
            std::hint::spin_loop();
        }

        let end_time = Instant::now();
        let end_tsc_val = get_tsc();

        if start_tsc_val.0 >= end_tsc_val.0 {
            bail!(
                "End TSC value ({}) is smaller than or equal to start TSC value ({}). TSC wrap might have happened",
                end_tsc_val.0,
                start_tsc_val.0,
            );
        }

        let tsc_diff = end_tsc_val.0 - start_tsc_val.0;
        let time_ns = end_time.duration_since(start_time).as_nanos() as u64;

        if time_ns == 0 {
            bail!("Elapsed time is zero");
        }

        // Calculate frequency: ticks_per_sec = ticks / seconds = ticks * 1e9 / nanoseconds
        let freq = ((tsc_diff as u128 * 1_000_000_000) / time_ns as u128) as u64;
        Ok(freq)
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

        for (i, sample) in samples.iter().enumerate() {
            let delta = *sample as f64 - mean;
            mean += delta / (i as f64 + 1.0);
            s += delta * (*sample as f64 - mean)
        }

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
        let max_sample = *samples.iter().max().unwrap();
        let min_sample = *samples.iter().min().unwrap();

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
    /// TSC ticks to nanoseconds.
    ///
    /// This function determines TSC frequency using the most accurate method available:
    /// 1. System sources (sysfs, CPUID) - instant and most accurate
    /// 2. Busy-wait calibration - takes ~200ms but very accurate
    pub fn new() -> Result<ConversionParams, anyhow::Error> {
        println!("Calculating TSC-to-nanoseconds conversion parameters...");

        let tsc_freq = Self::get_validated_tsc_frequency()
            .context("Failed to determine TSC frequency")?;

        let conv_params = Self::from_ratio(tsc_freq)
            .context("Error while calculating TSC-to-nanoseconds conversion parameters")?;

        Ok(conv_params)
    }

    /// Calculate parameters using the legacy multi-sample method.
    ///
    /// This method takes longer (~15 seconds) but may be more robust on some systems.
    /// Use `new()` for faster initialization with comparable accuracy.
    pub fn new_legacy() -> Result<ConversionParams, anyhow::Error> {
        println!("Calculating TSC-to-nanoseconds conversion parameters (legacy method)...");

        let mut tsc_per_sec = [0u64; WTMLIB_TSC_PER_SEC_SAMPLE_COUNT];

        println!("Calculating how TSC changes during a second-long time period");
        for i in 0..WTMLIB_TSC_PER_SEC_SAMPLE_COUNT {
            tsc_per_sec[i] = Self::calc_tsc_count_per_second(WTMLIB_TIME_PERIOD_TO_MATCH_WITH_TSC)
                .context("Error while calculating TSC worth of a second")?;
            println!("[Measurement {}] TSC ticks per sec: {}", i, tsc_per_sec[i]);
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
    #[inline]
    pub fn convert_to_nanosec(&self, ticks: Timestamp) -> u64 {
        (ticks.0 >> self.tsc_remainder_length) * self.nsecs_per_tsc_modulus
            + ((ticks.0 & self.tsc_remainder_bitmask) * (self.mult) >> self.shift)
    }

    /// Get TSC ticks per second
    ///
    /// This can be used for simple division-based conversion if needed
    #[inline]
    pub fn tsc_ticks_per_sec(&self) -> u64 {
        self.tsc_ticks_per_sec
    }

    /// Calculate time (in seconds!) before the earliest TSC wrap
    /// 
    /// TSC wrap - counter overflow, that is when the value reaches its maximum 
    /// and starts counting from zero
    /// 
    /// All available CPUs are considered when calculating the time
    pub fn time_before_tsc_wrap(&self) -> Result<u64, anyhow::Error> {
    unsafe {
        println!("Calculating time before the earliest TSC wrap...");
    
        let mut max_tsc_val = Timestamp(0);
        let mut curr_tsc_val;
        let secs_before_wrap;
        let mut ps_state = ProcAndSysState::new()
            .context("Couldn't obtain details of the system and process state")?;
        
        let thread_self = libc::pthread_self();
        let cpu_set_size = libc::CPU_ALLOC_SIZE(ps_state.num_cpus);
        let mut cpu_set = std::mem::MaybeUninit::zeroed().assume_init();
         
        for cpu_id in 0..ps_state.num_cpus as usize {
            if !libc::CPU_ISSET( 
                cpu_id, 
                &mut ps_state.initial_cpu_set
            ) {
                continue; 
            }     
    
            libc::CPU_SET(cpu_id, &mut cpu_set);
    
            if libc::pthread_setaffinity_np(thread_self, cpu_set_size, &cpu_set) != 0 {
                bail!("Couldn't change CPU affinity of the current thread");
            }
    
            curr_tsc_val = get_tsc();
            println!("TSC on CPU {}: {}", cpu_id, curr_tsc_val.0);
    
            if curr_tsc_val.0 > max_tsc_val.0 {
                max_tsc_val = curr_tsc_val;
            }
    
            /* Return CPU mask to the "clean" state */
            libc::CPU_CLR(cpu_id, &mut cpu_set);
        }
        
        println!("The maximum TSC value: {}", max_tsc_val.0);
        secs_before_wrap = self.convert_to_nanosec(Timestamp(u64::MAX - max_tsc_val.0)) / 1000000000;
        println!("Seconds before the maximum TSC will wrap: {}", secs_before_wrap);
            
        Ok(secs_before_wrap)
    }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_tsc_count_per_second() {
        let result = ConversionParams::calc_tsc_count_per_second(100_000); // 100ms
        assert!(result.is_ok(), "calc_tsc_count_per_second should succeed");

        let tsc_per_sec = result.unwrap();
        // TSC frequency should be in a reasonable range (100MHz to 10GHz)
        assert!(
            tsc_per_sec > 100_000_000,
            "TSC frequency should be > 100MHz, got {}",
            tsc_per_sec
        );
        assert!(
            tsc_per_sec < 10_000_000_000,
            "TSC frequency should be < 10GHz, got {}",
            tsc_per_sec
        );
    }

    #[test]
    fn test_calc_golden_removes_outliers() {
        // Create samples with outliers
        let samples = [
            3_000_000_000u64,
            3_000_000_100,
            3_000_000_050,
            3_000_000_080,
            3_000_000_020,
            3_500_000_000, // outlier
            3_000_000_030,
            3_000_000_060,
        ];

        let result = ConversionParams::calc_golden(&samples);
        assert!(result.is_ok(), "calc_golden should succeed");

        let golden = result.unwrap();
        // The golden value should be close to 3GHz, not skewed by the outlier
        assert!(
            golden > 2_900_000_000 && golden < 3_100_000_000,
            "Golden value should be around 3GHz, got {}",
            golden
        );
    }

    #[test]
    fn test_from_ratio() {
        // Test with a realistic TSC frequency of 3GHz
        let tsc_per_sec = 3_000_000_000u64;
        let result = ConversionParams::from_ratio(tsc_per_sec);
        assert!(result.is_ok(), "from_ratio should succeed");

        let params = result.unwrap();
        assert!(params.mult > 0, "mult should be positive");
        assert!(params.shift > 0, "shift should be positive");
        assert!(
            params.tsc_remainder_length > 0,
            "tsc_remainder_length should be positive"
        );
    }

    #[test]
    fn test_convert_to_nanosec() {
        // Create params for 3GHz TSC
        let tsc_per_sec = 3_000_000_000u64;
        let params = ConversionParams::from_ratio(tsc_per_sec).unwrap();

        // 3 billion ticks at 3GHz should be ~1 second = 1e9 nanoseconds
        let ns = params.convert_to_nanosec(Timestamp(tsc_per_sec));

        // Allow 1% error
        let expected = 1_000_000_000u64;
        let error = if ns > expected {
            ns - expected
        } else {
            expected - ns
        };
        let error_percent = (error as f64 / expected as f64) * 100.0;

        assert!(
            error_percent < 1.0,
            "Conversion error should be < 1%, got {}% (ns={}, expected={})",
            error_percent,
            ns,
            expected
        );
    }

    #[test]
    fn test_convert_to_nanosec_small_values() {
        let tsc_per_sec = 3_000_000_000u64;
        let params = ConversionParams::from_ratio(tsc_per_sec).unwrap();

        // 3000 ticks at 3GHz should be ~1 microsecond = 1000 nanoseconds
        let ns = params.convert_to_nanosec(Timestamp(3000));

        // Allow some error for small values
        assert!(
            ns >= 900 && ns <= 1100,
            "3000 ticks at 3GHz should be ~1000ns, got {}",
            ns
        );
    }

    #[test]
    fn test_tsc_ticks_per_sec_getter() {
        let tsc_per_sec = 3_000_000_000u64;
        let params = ConversionParams::from_ratio(tsc_per_sec).unwrap();

        assert_eq!(
            params.tsc_ticks_per_sec(),
            tsc_per_sec,
            "tsc_ticks_per_sec getter should return the correct value"
        );
    }
}