#![feature(thread_id_value)]

use anyhow::{Context, Result, bail, Ok};
use eval_tsc_reliability_cop::CpuSwitchingEstimate;
use libc::cpu_set_t;
use crate::config::*;

mod eval_tsc_reliability_cop;
mod conversion_params;
mod config;

#[derive(Debug, Clone, Copy)]
pub struct Timestamp(u64);

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
    ///
    /// Memory allocated for the "initial CPU set" should be deallocated after
    /// use by calling CPU_FREE()
    fn new() -> Result<Self, anyhow::Error> {
        unsafe {
            // Get the number of configured logical CPUs in the system (not all of them
            // may be availabe at the moment; e.g. some of them may be offline)
            let num_cpus = libc::get_nprocs_conf();
            let initial_cpu_id = libc::sched_getcpu();
            if initial_cpu_id < 0 {
                bail!("Couldn't get ID of the current CPU");
            }

            let thread_self = std::thread::current().id().as_u64().get();

            // Get thread's affinity mask. We're going to test TSC values only on the CPUs allowed
            // by the current thread's affinity mask 
            let initial_cpu_set = std::mem::MaybeUninit::zeroed().assume_init();
            
            let cpu_set_size = libc::CPU_ALLOC_SIZE( num_cpus);
            if libc::pthread_getaffinity_np(thread_self, cpu_set_size, initial_cpu_set.as_mut_ptr()) != 0 {
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
    ///
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
            // probability of that is pretty high */
            let thread_self = std::thread::current().id().as_u64().get();
            let cpu_set_size = libc::CPU_ALLOC_SIZE( self.num_cpus);
            let cpu_set = std::mem::MaybeUninit::zeroed().assume_init();
    
            libc::CPU_SET(self.initial_cpu_id, cpu_set);
    
            if libc::pthread_setaffinity_np(thread_self, cpu_set_size, cpu_set) != 0 {
                panic!("Couldn't return the current thread to the initial CPU");
            }
    
            if libc::pthread_setaffinity_np(thread_self, cpu_set_size, &self.initial_cpu_set) != 0 {
                panic!("Couldn't restore CPU affinity of the current thread");
            }
        }
    }
}

/// Maximum size of human-readable error messages returned by the library functions
const WTMLIB_MAX_ERR_MSG_SIZE: usize = 2000;

/// Macro that evaluates to 1 if the library should use non-negative return values. In
/// the opposite case the macro evaluates to -1.
/// The library takes care of its return values' sign to avoid interference with special
/// return values. Only one such 'special value' is relevant for now. Namely,
/// PTHREAD_CANCELED
const WTMLIB_RET_SIGN = ((long)PTHREAD_CANCELED < 0 ? 1 : -1);

/// Return value that indicates generic error
const WTMLIB_RET_GENERIC_ERR = (WTMLIB_RET_SIGN * 1);

/// Return value indicating that major TSC inconsistency was detected
///
/// Getting this return value doesn't necessarily imply that TSC is unreliable and cannot
/// be used as a wall-clock time measure. This type of error in some cases may be caused
/// by TSC wrap (which could has happened on some CPU in the middle of calculations; or it
/// might has happened on one CPU before WTMLIB was called but hasn't yet happened on the
/// other available CPUs; and so on)
const WTMLIB_RET_TSC_INCONSISTENCY = (WTMLIB_RET_SIGN * 2);

/// Return value indicating that configured statistical significance criteria were not met
/// (data collected by the library doesn't contain enough specific patterns)
const WTMLIB_RET_POOR_STAT = (WTMLIB_RET_SIGN * 3);



fn calc_tsc_enclosing_range_cpu_switch() -> Result<(), anyhow::Error> {
    todo!() 
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
fn eval_tsc_monotonicity_cpu_switch() -> Result<(), anyhow::Error> {
    println!("Evaluating TSC monotonicity...");

    // Calculate the number of CPUs available to the current thread
    
    todo!()
}

/// Evaluate reliability of TSC (the required data is collected using "CPU Switching"
/// method - a single thread jumps from one CPU to another and takes all needed
/// measurements)
///
/// Possible return codes:
///      0 - in case of success
///      WTMLIB_RET_TSC_INCONSISTENCY - major TSC inconsistency was detected
///      WTMLIB_RET_GENERIC_ERR - all other errors
///
/// Besides the regular return value the function returns (if the corresponding pointers
/// are non-zero):
///      tsc_range_length - estimated maximum shift between TSC counters running on
///                         different CPUs
///      is_monotonic - whether TSC values measured successively on same or different CPUs
///                     monotonically increase. If the function sets (*is_monotonic) to
///                     "false", that doesn't necessarily imply that TSCs are unreliable.
///                     In rare cases the observed non-monotonicity may be a result of TSC
///                     wrap that occured on one/several CPUs right before or just in the
///                     middle of the computations
///      err_msg - human-readable error message
///
/// Any of the pointer arguments can be zero.
/// In case of non-zero return code, the function doesn't modify memory referenced by
/// tsc_range_length and is_monotonic pointers.
/// err_msg is modified only if the return code is non-zero
///
/// NOTE: if the function sets is_monotonic to "false", that doesn't necessarily imply that
///       TSCs are unreliable. In some cases that can be a result of TSC wrap
pub fn inspect_cpu_switching() -> Result<CpuSwitchingEstimate, anyhow::Error> {
    println!("Evaluating TSC reliability (the required data is collected using \"CPU Switching\" method)...");
    let mut ps_state = ProcAndSysState::new()
        .context("Couldn't obtain details of the system and process state")?;
    
    let tsc_range_length = calc_tsc_enclosing_range_cpu_switch()
        .context("Error while calculating enclosing TSC range")?;
    
    let is_monotonic = eval_tsc_monotonicity_cpu_switch()
        .context("Error while evaluating TSC monotonicity")?;

    Ok(CpuSwitchingEstimate {
        tsc_range_length,
        is_monotonic,
    })
}
















