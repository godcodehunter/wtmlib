# wtmlib - Wall-clock Time Measurement Library

Rust port of [WTMLIB](https://github.com/AndreyNevolin/wtmlib) with improvements.

A high-precision, low-overhead library for measuring wall-clock time intervals using the CPU's Time Stamp Counter (TSC).

## Features

- **Nanosecond precision** - Direct TSC reads provide sub-nanosecond resolution
- **Low overhead** - TSC read takes ~20-30 CPU cycles (~10ns)
- **Division-free conversion** - TSC ticks to nanoseconds using fast multiply-shift arithmetic
- **Automatic calibration** - Determines TSC frequency with <0.01% error
- **TSC reliability evaluation** - Detects TSC synchronization issues across CPU cores

## Requirements

- x86_64 architecture (uses `rdtsc` instruction)
- Linux (for CPU affinity and sysfs access)
- CPU with invariant TSC (`constant_tsc` and `nonstop_tsc` flags)

Check your CPU flags:
```bash
grep -E 'constant_tsc|nonstop_tsc' /proc/cpuinfo
```

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
wtmlib = { path = "path/to/wtmlib" }
```

### Basic Example

```rust
use wtmlib::{get_tsc, ConversionParams, Timestamp};

fn main() {
    // Initialize conversion parameters (calibrates TSC frequency)
    let params = ConversionParams::new().expect("Failed to initialize");

    // Measure time interval
    let start = get_tsc();

    // ... code to measure ...

    let end = get_tsc();

    // Convert TSC ticks to nanoseconds
    let ticks = Timestamp(end.0 - start.0);
    let nanoseconds = params.convert_to_nanosec(ticks);

    println!("Elapsed: {} ns", nanoseconds);
}
```

### TSC Reliability Check

Before using TSC for measurements, verify it's reliable on your system:

```rust
use wtmlib::eval_tsc_reliability_cpu_switch;

fn main() {
    match eval_tsc_reliability_cpu_switch() {
        Ok(estimate) => {
            println!("TSC delta range: {} ticks", estimate.delta_range);
            println!("TSC enclosing range: {} ticks", estimate.enclosing_range);
            println!("Monotonic: {}", estimate.monotonic);

            if estimate.monotonic && estimate.delta_range < 1000 {
                println!("TSC is reliable for measurements");
            }
        }
        Err(e) => eprintln!("TSC reliability check failed: {}", e),
    }
}
```

## API Reference

### `get_tsc() -> Timestamp`
Reads the current TSC value. Very fast (~10ns).

### `ConversionParams`
- `new()` - Create with automatic TSC frequency calibration
- `from_ratio(tsc_freq_hz)` - Create with known TSC frequency
- `convert_to_nanosec(ticks)` - Convert TSC ticks to nanoseconds
- `tsc_ticks_per_sec()` - Get calibrated TSC frequency

### `eval_tsc_reliability_cpu_switch() -> Result<CpuSwitchingEstimate>`
Evaluates TSC reliability by measuring TSC values across all CPU cores.

### `eval_tsc_reliability_cop() -> Result<CpuSwitchingEstimate>`
Alternative reliability check using CAS-Ordered Probes method (multi-threaded).

## How It Works

### TSC Reading
The library uses the `rdtsc` x86 instruction to read the CPU's Time Stamp Counter - a 64-bit register that increments at a constant rate on modern CPUs.

### TSC-to-Nanoseconds Conversion
Instead of division (slow), the library uses multiply-shift arithmetic:
```
nanoseconds = (ticks * multiplier) >> shift
```

The multiplier and shift values are precomputed during calibration.

### Calibration
TSC frequency is determined using a hybrid approach:
1. Try reading from Linux sysfs (`/sys/devices/system/cpu/cpu0/tsc_freq_khz`)
2. Try reading from CPUID leaf 0x15 (Intel CPUs)
3. Fall back to busy-wait calibration with cross-validation

This achieves <0.01% frequency error compared to the ~7-8% error from naive sleep-based calibration.

## Examples

Run the examples:

```bash
# Basic accuracy test
cargo run --example accuracy_test --release

# Microbenchmark with assembly instructions
cargo run --example microbenchmark --release

# Compare calibration methods
cargo run --example improved_calibration --release
```

## Notes

> **Time Stamp Counter (TSC)** - A hardware counter present in x86 processors. Similar counters exist on other architectures:
> - Time-stamp counter on x86/x86_64
> - Time base register on PowerPC
> - Interval time counter on Itanium

> For Intel processors, also consider [Intel Processor Trace](https://halobates.de/blog/p/406) for detailed timing analysis.

## References

- [Original WTMLIB (C)](https://github.com/AndreyNevolin/wtmlib)
- [Measuring time: from C to the assembler](https://habr.com/ru/articles/425237/) (Russian)
- [Intel® 64 and IA-32 Architectures SDM](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)

## License

MIT
