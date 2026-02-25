//! Integration tests for wtmlib
//!
//! These tests verify the public API works correctly

use wtmlib::{eval_tsc_reliability_cpu_switch, get_tsc, ConversionParams, Timestamp};

#[test]
fn test_get_tsc_basic() {
    let tsc = get_tsc();
    assert!(tsc.0 > 0, "TSC should return a non-zero value");
}

#[test]
fn test_get_tsc_increases_over_time() {
    let start = get_tsc();

    // Do some work to let time pass
    let mut sum = 0u64;
    for i in 0..10000 {
        sum = sum.wrapping_add(i);
    }
    std::hint::black_box(sum);

    let end = get_tsc();
    assert!(
        end.0 > start.0,
        "TSC should increase over time: start={}, end={}",
        start.0,
        end.0
    );
}

#[test]
fn test_conversion_params_new() {
    // This test takes a few seconds as it measures TSC frequency
    let result = ConversionParams::new();
    assert!(
        result.is_ok(),
        "ConversionParams::new() should succeed: {:?}",
        result.err()
    );

    let params = result.unwrap();

    // Check that tsc_ticks_per_sec is reasonable (100MHz to 10GHz)
    let ticks_per_sec = params.tsc_ticks_per_sec();
    assert!(
        ticks_per_sec > 100_000_000,
        "TSC frequency should be > 100MHz, got {}",
        ticks_per_sec
    );
    assert!(
        ticks_per_sec < 10_000_000_000,
        "TSC frequency should be < 10GHz, got {}",
        ticks_per_sec
    );
}

#[test]
fn test_conversion_roundtrip() {
    let params = ConversionParams::new().expect("Failed to create ConversionParams");

    // Measure 100ms of TSC ticks
    let start = get_tsc();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let end = get_tsc();

    let ticks = end.0 - start.0;
    let ns = params.convert_to_nanosec(Timestamp(ticks));

    // Should be approximately 100ms = 100_000_000 ns
    // Allow 20% error due to sleep imprecision
    let expected_ns = 100_000_000u64;
    let error = if ns > expected_ns {
        ns - expected_ns
    } else {
        expected_ns - ns
    };
    let error_percent = (error as f64 / expected_ns as f64) * 100.0;

    assert!(
        error_percent < 20.0,
        "Conversion error should be < 20%, got {}% (measured={}ns, expected={}ns)",
        error_percent,
        ns,
        expected_ns
    );
}

#[test]
fn test_eval_tsc_reliability_cpu_switch() {
    let result = eval_tsc_reliability_cpu_switch();

    match result {
        Ok(estimate) => {
            println!("TSC range length: {}", estimate.tsc_range_length);
            println!("Is monotonic: {}", estimate.is_monotonic);

            // The range should be reasonable (not negative, not huge)
            assert!(
                estimate.tsc_range_length >= 0,
                "TSC range length should be non-negative"
            );
            assert!(
                estimate.tsc_range_length < 1_000_000_000,
                "TSC range length should be reasonable"
            );
        }
        Err(e) => {
            // On some systems (single CPU, VMs), this might fail
            // That's acceptable for testing purposes
            println!("eval_tsc_reliability_cpu_switch failed (may be expected): {}", e);
        }
    }
}

#[test]
fn test_timestamp_operations() {
    let ts1 = Timestamp(1000);
    let ts2 = Timestamp(2000);

    // Test that we can access the inner value
    assert_eq!(ts1.0, 1000);
    assert_eq!(ts2.0, 2000);

    // Test copy
    let ts3 = ts1;
    assert_eq!(ts3.0, ts1.0);
}

#[test]
fn test_multiple_tsc_reads_performance() {
    // Test that TSC reads are fast
    let iterations = 1_000_000;

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(get_tsc());
    }
    let elapsed = start.elapsed();

    let ns_per_read = elapsed.as_nanos() as f64 / iterations as f64;
    println!("Average TSC read time: {:.2} ns", ns_per_read);

    // TSC read should be very fast (< 100ns typically)
    assert!(
        ns_per_read < 1000.0,
        "TSC read should be fast, got {} ns",
        ns_per_read
    );
}
