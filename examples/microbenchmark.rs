//! Microbenchmark to verify TSC measurement accuracy
//!
//! This benchmark measures:
//! 1. TSC read overhead (rdtsc vs rdtscp)
//! 2. Known assembly instruction latencies
//! 3. Measurement consistency
//!
//! Run with: cargo run --example microbenchmark --release

use std::arch::x86_64::{_rdtsc, __rdtscp, _mm_lfence};

/// Read TSC with lfence barrier (prevents reordering)
#[inline(always)]
unsafe fn rdtsc_fenced() -> u64 {
    _mm_lfence();
    let tsc = _rdtsc();
    _mm_lfence();
    tsc
}

/// Read TSC with rdtscp (serializing, returns CPU ID too)
#[inline(always)]
unsafe fn rdtscp_fenced() -> u64 {
    let mut aux: u32 = 0;
    let tsc = __rdtscp(&mut aux);
    _mm_lfence(); // prevent following instructions from being reordered before rdtscp
    tsc
}

/// Measure overhead of rdtsc itself
fn measure_rdtsc_overhead() -> (u64, u64, u64) {
    const ITERATIONS: usize = 10000;
    let mut measurements = Vec::with_capacity(ITERATIONS);

    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtsc_fenced();
            // Nothing here - just measure rdtsc overhead
            let end = rdtsc_fenced();
            measurements.push(end - start);
        }
    }

    measurements.sort();
    let min = measurements[0];
    let median = measurements[ITERATIONS / 2];
    let p99 = measurements[(ITERATIONS * 99) / 100];

    (min, median, p99)
}

/// Measure overhead of rdtscp
fn measure_rdtscp_overhead() -> (u64, u64, u64) {
    const ITERATIONS: usize = 10000;
    let mut measurements = Vec::with_capacity(ITERATIONS);

    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtscp_fenced();
            let end = rdtscp_fenced();
            measurements.push(end - start);
        }
    }

    measurements.sort();
    let min = measurements[0];
    let median = measurements[ITERATIONS / 2];
    let p99 = measurements[(ITERATIONS * 99) / 100];

    (min, median, p99)
}

/// Measure a single NOP instruction
fn measure_nop() -> (u64, u64, u64) {
    const ITERATIONS: usize = 10000;
    let mut measurements = Vec::with_capacity(ITERATIONS);

    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtsc_fenced();
            std::arch::asm!("nop");
            let end = rdtsc_fenced();
            measurements.push(end - start);
        }
    }

    measurements.sort();
    let min = measurements[0];
    let median = measurements[ITERATIONS / 2];
    let p99 = measurements[(ITERATIONS * 99) / 100];

    (min, median, p99)
}

/// Measure 10 NOP instructions
fn measure_10_nops() -> (u64, u64, u64) {
    const ITERATIONS: usize = 10000;
    let mut measurements = Vec::with_capacity(ITERATIONS);

    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtsc_fenced();
            std::arch::asm!(
                "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop",
            );
            let end = rdtsc_fenced();
            measurements.push(end - start);
        }
    }

    measurements.sort();
    let min = measurements[0];
    let median = measurements[ITERATIONS / 2];
    let p99 = measurements[(ITERATIONS * 99) / 100];

    (min, median, p99)
}

/// Measure 100 NOP instructions
fn measure_100_nops() -> (u64, u64, u64) {
    const ITERATIONS: usize = 10000;
    let mut measurements = Vec::with_capacity(ITERATIONS);

    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtsc_fenced();
            std::arch::asm!(
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
                "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
            );
            let end = rdtsc_fenced();
            measurements.push(end - start);
        }
    }

    measurements.sort();
    let min = measurements[0];
    let median = measurements[ITERATIONS / 2];
    let p99 = measurements[(ITERATIONS * 99) / 100];

    (min, median, p99)
}

/// Measure integer ADD instruction with dependency chain
fn measure_add_chain() -> (u64, u64, u64) {
    const ITERATIONS: usize = 10000;
    let mut measurements = Vec::with_capacity(ITERATIONS);

    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtsc_fenced();
            // Dependency chain of 10 ADDs - cannot be parallelized
            std::arch::asm!(
                "add {0}, 1",
                "add {0}, 1",
                "add {0}, 1",
                "add {0}, 1",
                "add {0}, 1",
                "add {0}, 1",
                "add {0}, 1",
                "add {0}, 1",
                "add {0}, 1",
                "add {0}, 1",
                inout(reg) 0u64 => _,
            );
            let end = rdtsc_fenced();
            measurements.push(end - start);
        }
    }

    measurements.sort();
    let min = measurements[0];
    let median = measurements[ITERATIONS / 2];
    let p99 = measurements[(ITERATIONS * 99) / 100];

    (min, median, p99)
}

/// Measure integer MUL instruction (known to take ~3-4 cycles)
fn measure_imul() -> (u64, u64, u64) {
    const ITERATIONS: usize = 10000;
    let mut measurements = Vec::with_capacity(ITERATIONS);

    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtsc_fenced();
            std::arch::asm!(
                "imul {0}, {0}, 7",
                inout(reg) 42u64 => _,
            );
            let end = rdtsc_fenced();
            measurements.push(end - start);
        }
    }

    measurements.sort();
    let min = measurements[0];
    let median = measurements[ITERATIONS / 2];
    let p99 = measurements[(ITERATIONS * 99) / 100];

    (min, median, p99)
}

/// Measure integer DIV instruction (known to take ~20-80 cycles)
fn measure_div() -> (u64, u64, u64) {
    const ITERATIONS: usize = 10000;
    let mut measurements = Vec::with_capacity(ITERATIONS);

    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtsc_fenced();
            std::arch::asm!(
                "xor rdx, rdx",  // Clear high bits for division
                "div {divisor}",
                divisor = in(reg) 7u64,
                inout("rax") 1000000u64 => _,
                out("rdx") _,
            );
            let end = rdtsc_fenced();
            measurements.push(end - start);
        }
    }

    measurements.sort();
    let min = measurements[0];
    let median = measurements[ITERATIONS / 2];
    let p99 = measurements[(ITERATIONS * 99) / 100];

    (min, median, p99)
}

/// Measure memory load latency (L1 cache hit)
fn measure_memory_load() -> (u64, u64, u64) {
    const ITERATIONS: usize = 10000;
    let mut measurements = Vec::with_capacity(ITERATIONS);
    let data: u64 = 42;

    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtsc_fenced();
            let _ = std::ptr::read_volatile(&data);
            let end = rdtsc_fenced();
            measurements.push(end - start);
        }
    }

    measurements.sort();
    let min = measurements[0];
    let median = measurements[ITERATIONS / 2];
    let p99 = measurements[(ITERATIONS * 99) / 100];

    (min, median, p99)
}

/// Test measurement consistency by measuring the same thing multiple times
fn test_consistency() {
    println!("\n=== Consistency Test ===");
    println!("Measuring 100 NOPs 5 times:");

    for i in 1..=5 {
        let (min, median, p99) = measure_100_nops();
        println!(
            "  Run {}: min={:4} median={:4} p99={:4} cycles",
            i, min, median, p99
        );
    }
}

/// Calculate cycles per instruction
fn analyze_nop_scaling() {
    println!("\n=== NOP Scaling Analysis ===");

    let (overhead_min, overhead_median, _) = measure_rdtsc_overhead();
    let (nop1_min, nop1_median, _) = measure_nop();
    let (nop10_min, nop10_median, _) = measure_10_nops();
    let (nop100_min, nop100_median, _) = measure_100_nops();

    println!("Baseline (rdtsc overhead): min={} median={}", overhead_min, overhead_median);
    println!("1 NOP:   min={:4} median={:4}", nop1_min, nop1_median);
    println!("10 NOPs: min={:4} median={:4}", nop10_min, nop10_median);
    println!("100 NOPs: min={:4} median={:4}", nop100_min, nop100_median);

    // Calculate cycles per NOP (subtracting overhead)
    let cycles_per_nop_10 = (nop10_median.saturating_sub(overhead_median)) as f64 / 10.0;
    let cycles_per_nop_100 = (nop100_median.saturating_sub(overhead_median)) as f64 / 100.0;

    println!("\nCycles per NOP (from 10 NOPs):  {:.2}", cycles_per_nop_10);
    println!("Cycles per NOP (from 100 NOPs): {:.2}", cycles_per_nop_100);
    println!("(NOP should be ~0.25 cycles on modern CPUs due to 4-wide execution)");
}

/// Compare wtmlib's get_tsc with raw rdtsc
fn compare_with_wtmlib() {
    println!("\n=== Comparison with wtmlib::get_tsc ===");

    const ITERATIONS: usize = 10000;

    // Measure wtmlib's get_tsc
    let mut wtmlib_measurements = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        unsafe {
            let start = rdtsc_fenced();
            std::hint::black_box(wtmlib::get_tsc());
            let end = rdtsc_fenced();
            wtmlib_measurements.push(end - start);
        }
    }
    wtmlib_measurements.sort();

    // Measure raw rdtsc
    let mut raw_measurements = Vec::with_capacity(ITERATIONS);
    unsafe {
        for _ in 0..ITERATIONS {
            let start = rdtsc_fenced();
            std::hint::black_box(_rdtsc());
            let end = rdtsc_fenced();
            raw_measurements.push(end - start);
        }
    }
    raw_measurements.sort();

    println!(
        "wtmlib::get_tsc: min={:4} median={:4} p99={:4}",
        wtmlib_measurements[0],
        wtmlib_measurements[ITERATIONS / 2],
        wtmlib_measurements[(ITERATIONS * 99) / 100]
    );
    println!(
        "raw _rdtsc:      min={:4} median={:4} p99={:4}",
        raw_measurements[0],
        raw_measurements[ITERATIONS / 2],
        raw_measurements[(ITERATIONS * 99) / 100]
    );
}

fn main() {
    println!("=== TSC Microbenchmark ===\n");
    println!("Measuring TSC accuracy by timing known assembly instructions.");
    println!("All measurements in CPU cycles.\n");

    // Warm up
    for _ in 0..1000 {
        unsafe { std::hint::black_box(_rdtsc()); }
    }

    println!("=== TSC Read Overhead ===");
    let (rdtsc_min, rdtsc_median, rdtsc_p99) = measure_rdtsc_overhead();
    println!(
        "rdtsc + lfence:  min={:4} median={:4} p99={:4} cycles",
        rdtsc_min, rdtsc_median, rdtsc_p99
    );

    let (rdtscp_min, rdtscp_median, rdtscp_p99) = measure_rdtscp_overhead();
    println!(
        "rdtscp + lfence: min={:4} median={:4} p99={:4} cycles",
        rdtscp_min, rdtscp_median, rdtscp_p99
    );

    println!("\n=== Instruction Latencies ===");
    println!("(Overhead included - subtract ~{} cycles for net latency)", rdtsc_median);

    let (min, median, p99) = measure_nop();
    println!("1 NOP:           min={:4} median={:4} p99={:4}", min, median, p99);

    let (min, median, p99) = measure_10_nops();
    println!("10 NOPs:         min={:4} median={:4} p99={:4}", min, median, p99);

    let (min, median, p99) = measure_100_nops();
    println!("100 NOPs:        min={:4} median={:4} p99={:4}", min, median, p99);

    let (min, median, p99) = measure_add_chain();
    println!("10 ADDs (chain): min={:4} median={:4} p99={:4}", min, median, p99);

    let (min, median, p99) = measure_imul();
    println!("IMUL:            min={:4} median={:4} p99={:4}", min, median, p99);

    let (min, median, p99) = measure_div();
    println!("DIV:             min={:4} median={:4} p99={:4}", min, median, p99);

    let (min, median, p99) = measure_memory_load();
    println!("Memory load:     min={:4} median={:4} p99={:4}", min, median, p99);

    analyze_nop_scaling();
    test_consistency();
    compare_with_wtmlib();

    println!("\n=== Expected Values (Modern Intel/AMD) ===");
    println!("NOP:  ~0.25 cycles (4-wide superscalar)");
    println!("ADD:  ~1 cycle latency, ~4/cycle throughput");
    println!("IMUL: ~3 cycles latency");
    println!("DIV:  ~20-80 cycles depending on operand size");
    println!("L1 load: ~4-5 cycles");

    println!("\n=== Conclusion ===");
    let overhead = rdtsc_median;
    println!("TSC measurement overhead: ~{} cycles", overhead);
    println!("Minimum measurable interval: ~{} cycles", overhead);
    println!(
        "For accurate measurements, measure operations taking >>{} cycles",
        overhead * 10
    );
}
