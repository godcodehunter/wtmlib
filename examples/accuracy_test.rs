//! Accuracy test for TSC-to-nanosecond conversion
//!
//! This test verifies that wtmlib correctly converts TSC ticks to nanoseconds
//! by comparing against system time.
//!
//! Run with: cargo run --example accuracy_test --release

use std::time::{Duration, Instant};
use wtmlib::{get_tsc, ConversionParams, Timestamp};

fn measure_conversion_accuracy(params: &ConversionParams, duration_ms: u64) -> f64 {
    let system_start = Instant::now();
    let tsc_start = get_tsc();

    std::thread::sleep(Duration::from_millis(duration_ms));

    let tsc_end = get_tsc();
    let system_elapsed = system_start.elapsed();

    let tsc_ticks = tsc_end.0 - tsc_start.0;
    let tsc_ns = params.convert_to_nanosec(Timestamp(tsc_ticks));
    let system_ns = system_elapsed.as_nanos() as u64;

    let error_ns = (tsc_ns as i64 - system_ns as i64).abs() as f64;
    let error_percent = (error_ns / system_ns as f64) * 100.0;

    println!(
        "  Duration: {}ms | System: {:>12}ns | TSC: {:>12}ns | Error: {:>8.0}ns ({:.4}%)",
        duration_ms, system_ns, tsc_ns, error_ns, error_percent
    );

    error_percent
}

fn measure_busy_loop_accuracy(params: &ConversionParams, iterations: u64) -> f64 {
    let system_start = Instant::now();
    let tsc_start = get_tsc();

    // Busy loop - more precise than sleep
    let mut x = 0u64;
    for i in 0..iterations {
        x = x.wrapping_add(i);
    }
    std::hint::black_box(x);

    let tsc_end = get_tsc();
    let system_elapsed = system_start.elapsed();

    let tsc_ticks = tsc_end.0 - tsc_start.0;
    let tsc_ns = params.convert_to_nanosec(Timestamp(tsc_ticks));
    let system_ns = system_elapsed.as_nanos() as u64;

    let error_ns = (tsc_ns as i64 - system_ns as i64).abs() as f64;
    let error_percent = (error_ns / system_ns as f64) * 100.0;

    println!(
        "  Iterations: {:>10} | System: {:>12}ns | TSC: {:>12}ns | Error: {:>8.0}ns ({:.4}%)",
        iterations, system_ns, tsc_ns, error_ns, error_percent
    );

    error_percent
}

fn test_short_intervals(params: &ConversionParams) {
    println!("\n=== Short Interval Measurements ===");
    println!("Testing TSC accuracy for very short intervals:\n");

    let intervals = [10, 100, 1000, 10000, 100000];

    for &iters in &intervals {
        let tsc_start = get_tsc();

        let mut x = 0u64;
        for i in 0..iters {
            x = x.wrapping_add(i);
        }
        std::hint::black_box(x);

        let tsc_end = get_tsc();
        let tsc_ticks = tsc_end.0 - tsc_start.0;
        let tsc_ns = params.convert_to_nanosec(Timestamp(tsc_ticks));

        println!(
            "  {} iterations: {} ticks = {} ns ({:.2} ns/iter)",
            iters,
            tsc_ticks,
            tsc_ns,
            tsc_ns as f64 / iters as f64
        );
    }
}

fn test_tsc_frequency(params: &ConversionParams) {
    println!("\n=== TSC Frequency Verification ===");

    let reported_freq = params.tsc_ticks_per_sec();
    println!("Reported TSC frequency: {} Hz ({:.3} GHz)",
             reported_freq,
             reported_freq as f64 / 1e9);

    // Measure actual frequency over 1 second
    println!("\nMeasuring actual frequency over 1 second...");

    let system_start = Instant::now();
    let tsc_start = get_tsc();

    std::thread::sleep(Duration::from_secs(1));

    let tsc_end = get_tsc();
    let system_elapsed = system_start.elapsed();

    let tsc_ticks = tsc_end.0 - tsc_start.0;
    let measured_freq = (tsc_ticks as f64 / system_elapsed.as_secs_f64()) as u64;

    let freq_error = ((measured_freq as i64 - reported_freq as i64).abs() as f64
        / reported_freq as f64)
        * 100.0;

    println!("Measured TSC frequency:  {} Hz ({:.3} GHz)",
             measured_freq,
             measured_freq as f64 / 1e9);
    println!("Frequency error: {:.4}%", freq_error);
}

fn test_monotonicity() {
    println!("\n=== Monotonicity Test ===");
    println!("Checking that TSC always increases...\n");

    const ITERATIONS: usize = 1_000_000;
    let mut prev = get_tsc();
    let mut violations = 0;
    let mut max_decrease = 0u64;

    for _ in 0..ITERATIONS {
        let curr = get_tsc();
        if curr.0 < prev.0 {
            violations += 1;
            let decrease = prev.0 - curr.0;
            if decrease > max_decrease {
                max_decrease = decrease;
            }
        }
        prev = curr;
    }

    if violations == 0 {
        println!("  PASS: No monotonicity violations in {} reads", ITERATIONS);
    } else {
        println!(
            "  FAIL: {} violations in {} reads (max decrease: {} ticks)",
            violations, ITERATIONS, max_decrease
        );
    }
}

fn test_stability() {
    println!("\n=== Stability Test ===");
    println!("Measuring the same ~10ms interval 10 times:\n");

    let params = ConversionParams::new().expect("Failed to create ConversionParams");
    let mut measurements = Vec::new();

    for i in 1..=10 {
        let tsc_start = get_tsc();
        std::thread::sleep(Duration::from_millis(10));
        let tsc_end = get_tsc();

        let tsc_ticks = tsc_end.0 - tsc_start.0;
        let tsc_ns = params.convert_to_nanosec(Timestamp(tsc_ticks));
        measurements.push(tsc_ns);

        println!("  Run {:2}: {} ns", i, tsc_ns);
    }

    let avg: f64 = measurements.iter().sum::<u64>() as f64 / measurements.len() as f64;
    let variance: f64 = measurements
        .iter()
        .map(|&x| (x as f64 - avg).powi(2))
        .sum::<f64>()
        / measurements.len() as f64;
    let std_dev = variance.sqrt();
    let cv = (std_dev / avg) * 100.0; // Coefficient of variation

    println!("\n  Average: {:.0} ns", avg);
    println!("  Std Dev: {:.0} ns", std_dev);
    println!("  CV:      {:.2}% (lower is better, <5% is good)", cv);
}

fn main() {
    println!("=== TSC Accuracy Test ===\n");

    println!("Initializing ConversionParams (this measures TSC frequency)...\n");
    let params = ConversionParams::new().expect("Failed to create ConversionParams");

    test_tsc_frequency(&params);

    println!("\n=== Sleep-based Accuracy Test ===");
    println!("Comparing TSC measurements against system time:\n");

    let mut total_error = 0.0;
    let durations = [1, 10, 100, 500, 1000];

    for &ms in &durations {
        total_error += measure_conversion_accuracy(&params, ms);
    }
    let avg_error = total_error / durations.len() as f64;
    println!("\n  Average error: {:.4}%", avg_error);

    println!("\n=== Busy-loop Accuracy Test ===");
    println!("(More precise than sleep, no scheduler involvement)\n");

    let iterations = [100_000, 1_000_000, 10_000_000, 100_000_000];
    total_error = 0.0;

    for &iters in &iterations {
        total_error += measure_busy_loop_accuracy(&params, iters);
    }
    let avg_error = total_error / iterations.len() as f64;
    println!("\n  Average error: {:.4}%", avg_error);

    test_short_intervals(&params);
    test_monotonicity();
    test_stability();

    println!("\n=== Summary ===");
    println!("TSC frequency: {:.3} GHz", params.tsc_ticks_per_sec() as f64 / 1e9);
    println!("If errors are <1%, the conversion is working correctly.");
    println!("Higher errors may indicate:");
    println!("  - CPU frequency scaling");
    println!("  - Running in a VM with unstable TSC");
    println!("  - System load affecting measurements");
}
