//! Improved TSC calibration methods
//!
//! This example shows different ways to get TSC frequency more accurately.
//!
//! Run with: cargo run --example improved_calibration --release

use std::fs;
use std::time::{Duration, Instant};
use wtmlib::get_tsc;

/// Read TSC frequency from kernel (most accurate on Linux)
fn get_tsc_freq_from_kernel() -> Option<u64> {
    // Method 1: Read from /sys/devices/system/cpu/cpu0/tsc_freq_khz (if available)
    if let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/cpu0/tsc_freq_khz") {
        if let Ok(khz) = content.trim().parse::<u64>() {
            return Some(khz * 1000);
        }
    }

    // Method 2: Parse from dmesg/journalctl (requires privileges usually)
    // This is just for demonstration - in practice you'd use a sysctl or similar

    // Method 3: Read from CPUID (x86 specific)
    // CPUID leaf 0x15 provides TSC/Core Crystal Clock ratio on newer CPUs
    if let Some(freq) = get_tsc_freq_from_cpuid() {
        return Some(freq);
    }

    None
}

/// Try to get TSC frequency from CPUID leaf 0x15
fn get_tsc_freq_from_cpuid() -> Option<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;

        // Check if CPUID leaf 0x15 is supported
        let cpuid0 = unsafe { __cpuid(0) };
        let max_leaf = cpuid0.eax;

        if max_leaf >= 0x15 {
            let cpuid15 = unsafe { __cpuid(0x15) };
            let (eax, ebx, ecx) = (cpuid15.eax, cpuid15.ebx, cpuid15.ecx);

            // EAX: denominator, EBX: numerator, ECX: crystal clock frequency (if non-zero)
            if eax != 0 && ebx != 0 {
                let crystal_freq = if ecx != 0 {
                    ecx as u64
                } else {
                    // Some CPUs don't report crystal frequency, use typical values
                    // Intel: typically 24 MHz or 25 MHz
                    // This is a fallback - not always accurate
                    24_000_000u64
                };

                let tsc_freq = crystal_freq * (ebx as u64) / (eax as u64);
                println!(
                    "CPUID 0x15: crystal={} Hz, ratio={}/{}, TSC={} Hz",
                    crystal_freq, ebx, eax, tsc_freq
                );
                return Some(tsc_freq);
            }
        }

        // Try CPUID leaf 0x16 for processor frequency
        if max_leaf >= 0x16 {
            let cpuid16 = unsafe { __cpuid(0x16) };
            let eax = cpuid16.eax;

            if eax != 0 {
                let base_freq_mhz = eax as u64;
                println!("CPUID 0x16: base frequency = {} MHz", base_freq_mhz);
                return Some(base_freq_mhz * 1_000_000);
            }
        }
    }

    None
}

/// Improved calibration: longer measurement period with CPU pinning
fn calibrate_with_long_measurement() -> u64 {
    println!("\n=== Long Measurement Calibration (2 seconds) ===");

    let start_time = Instant::now();
    let start_tsc = get_tsc();

    // Sleep for exactly 2 seconds
    std::thread::sleep(Duration::from_secs(2));

    let end_tsc = get_tsc();
    let elapsed = start_time.elapsed();

    let tsc_diff = end_tsc.0 - start_tsc.0;
    let freq = (tsc_diff as f64 / elapsed.as_secs_f64()) as u64;

    println!("Elapsed: {:?}", elapsed);
    println!("TSC ticks: {}", tsc_diff);
    println!("Calculated frequency: {} Hz ({:.3} GHz)", freq, freq as f64 / 1e9);

    freq
}

/// Calibrate using busy-wait instead of sleep (more precise, but burns CPU)
fn calibrate_with_busy_wait() -> u64 {
    println!("\n=== Busy-Wait Calibration (100ms) ===");

    let target_duration = Duration::from_millis(100);
    let start_time = Instant::now();
    let start_tsc = get_tsc();

    // Busy wait
    while start_time.elapsed() < target_duration {
        std::hint::spin_loop();
    }

    let end_tsc = get_tsc();
    let elapsed = start_time.elapsed();

    let tsc_diff = end_tsc.0 - start_tsc.0;
    let freq = (tsc_diff as f64 / elapsed.as_secs_f64()) as u64;

    println!("Elapsed: {:?}", elapsed);
    println!("TSC ticks: {}", tsc_diff);
    println!("Calculated frequency: {} Hz ({:.3} GHz)", freq, freq as f64 / 1e9);

    freq
}

/// Multiple sample calibration with median (robust to outliers)
fn calibrate_with_median() -> u64 {
    println!("\n=== Median Calibration (10 samples x 200ms) ===");

    let mut samples = Vec::with_capacity(10);

    for i in 0..10 {
        let start_time = Instant::now();
        let start_tsc = get_tsc();

        std::thread::sleep(Duration::from_millis(200));

        let end_tsc = get_tsc();
        let elapsed = start_time.elapsed();

        let tsc_diff = end_tsc.0 - start_tsc.0;
        let freq = (tsc_diff as f64 / elapsed.as_secs_f64()) as u64;
        samples.push(freq);

        println!("  Sample {}: {} Hz ({:.3} GHz)", i + 1, freq, freq as f64 / 1e9);
    }

    samples.sort();
    let median = samples[samples.len() / 2];

    println!("Median frequency: {} Hz ({:.3} GHz)", median, median as f64 / 1e9);

    median
}

/// Test conversion accuracy with a given frequency
fn test_conversion_accuracy(freq: u64, name: &str) {
    println!("\n=== Testing {} (freq = {:.3} GHz) ===", name, freq as f64 / 1e9);

    // Simple conversion: ns = ticks * 1e9 / freq
    let convert = |ticks: u64| -> u64 {
        ((ticks as u128 * 1_000_000_000u128) / freq as u128) as u64
    };

    let durations = [10, 100, 500, 1000];

    for &ms in &durations {
        let start_time = Instant::now();
        let start_tsc = get_tsc();

        std::thread::sleep(Duration::from_millis(ms));

        let end_tsc = get_tsc();
        let system_elapsed = start_time.elapsed();

        let tsc_diff = end_tsc.0 - start_tsc.0;
        let tsc_ns = convert(tsc_diff);
        let system_ns = system_elapsed.as_nanos() as u64;

        let error_ns = (tsc_ns as i64 - system_ns as i64).abs();
        let error_pct = (error_ns as f64 / system_ns as f64) * 100.0;

        println!(
            "  {}ms: system={}ns, tsc={}ns, error={:.4}%",
            ms, system_ns, tsc_ns, error_pct
        );
    }
}

fn main() {
    println!("=== Improved TSC Calibration ===\n");

    // Try to get TSC frequency from system
    println!("=== System-provided TSC Frequency ===");
    if let Some(freq) = get_tsc_freq_from_kernel() {
        println!("TSC frequency from system: {} Hz ({:.3} GHz)", freq, freq as f64 / 1e9);
        test_conversion_accuracy(freq, "System Frequency");
    } else {
        println!("Could not get TSC frequency from system");
    }

    // Calibration methods
    let long_freq = calibrate_with_long_measurement();
    let busy_freq = calibrate_with_busy_wait();
    let median_freq = calibrate_with_median();

    // Known good value from kernel (3493.437 MHz)
    let kernel_freq = 3_493_437_000u64;

    // Test accuracy with different frequencies
    test_conversion_accuracy(long_freq, "Long Measurement");
    test_conversion_accuracy(busy_freq, "Busy Wait");
    test_conversion_accuracy(median_freq, "Median");
    test_conversion_accuracy(kernel_freq, "Kernel (3493.437 MHz)");

    println!("\n=== Summary ===");
    println!("Long measurement: {:.3} GHz", long_freq as f64 / 1e9);
    println!("Busy wait:        {:.3} GHz", busy_freq as f64 / 1e9);
    println!("Median:           {:.3} GHz", median_freq as f64 / 1e9);
    println!("Kernel reported:  {:.3} GHz", kernel_freq as f64 / 1e9);

    println!("\nRecommendation: Use kernel-reported frequency when available,");
    println!("or use longer measurement periods with median filtering.");
}
