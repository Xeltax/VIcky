//! The title's clocks and waits while its time is slowed (`VICTORY_TIME_SCALE`, see
//! `victory_core::timescale`): every clock below returns the slowed time, and every wait
//! lasts the span of the title's own time it asks for. Bound only when the scale is set;
//! otherwise the title's imports go to the host's own functions, as before.
//!
//! Where the title reads time (its imports and its code, build 1.0.2.1):
//! * the performance counter: `QueryPerformanceCounter` (285 places in the executable, and
//!   in every DLL it ships) and the C++ library's `_Query_perf_counter` (`steady_clock`):
//!   scaled here;
//! * tick counts (`GetTickCount64`; `GetTickCount` in its network DLL), system times
//!   (`GetSystemTimeAsFileTime`, `GetSystemTimePreciseAsFileTime`, `GetSystemTime`,
//!   `GetLocalTime`, the C++ library's `_Xtime_get_ticks`, the C library's `_time64` and
//!   `_ftime64`) and Media Foundation's `MFGetSystemTime`: scaled here;
//! * GameInput timestamps (`gameinput`), the GPU and system timestamps its command lists
//!   write, and the frame clock its frame waits follow (`victory_gpu`): scaled there;
//! * not scaled: `QueryPerformanceFrequency` (a rate, unchanged on purpose), the audio
//!   device's clock (its audio plays through the host's WASAPI in real time), the thread-pool
//!   timers of its network DLLs and concurrency runtime (host functions), and `rdtsc`: the
//!   executable has none; the OpenMP runtime it ships has four, in spin waits, and nothing
//!   imports that DLL.
//!
//! Waits: every finite time-out the title passes is a span of its own time, converted with
//! `timescale::real_wait_ms` (title ms / f, at most the longer of 10 s and the title's own
//! value; 0 and `INFINITE` unchanged).
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real API would
// trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::time::Duration;

use victory_core::timescale::{self, Source, UNIX_EPOCH_FILE_TIME};
use windows_sys::Win32::Foundation::{FILETIME, SYSTEMTIME};

use crate::{HleModule, hle_export};

/// Whether the title's time is scaled, so whether this module is bound at all.
pub fn active() -> bool {
    timescale::active()
}

/// Takes the start of the title's clocks (before any title code runs), says what is scaled,
/// and from then on writes a line every five seconds with the reads a second by clock and
/// the waits converted.
pub fn start() {
    let asked = std::env::var(timescale::ENV).unwrap_or_default();
    let Some(f) = timescale::factor() else {
        if !asked.is_empty() && asked.trim() != "1" {
            tracing::warn!(
                asked,
                "{} is not a number between 0 and 1: the title's time is not scaled",
                timescale::ENV
            );
        }
        return;
    };
    let base = timescale::start();
    let scaled: Vec<&str> = Source::ALL.iter().map(|s| s.name()).collect();
    tracing::info!(
        "time scale {f} ({}): the title's clocks run at {f} of real time from now on (a real 200 ms frame looks like {:.0} ms to it), \
         from one start (counter {}, {} ticks a second); scaled: {}, and the frame clock of its frame waits; unchanged: \
         QueryPerformanceFrequency; not scaled: the audio device's clock (host WASAPI), the host's thread-pool timers, rdtsc",
        timescale::ENV,
        200.0 * f,
        base.counter,
        base.frequency,
        scaled.join(", "),
    );
    tracing::info!(
        "time scale {f}: a time-out of t ms the title passes (Sleep, SleepEx, WaitForSingleObject(Ex), WaitForMultipleObjects(Ex), \
         SleepConditionVariableCS/SRW, WaitOnAddress, task-queue delays and dispatch, frame waits) lasts t / {f} real ms, at most \
         the longer of {} s and t; 0 and INFINITE unchanged",
        timescale::WAIT_CAP_MS / 1_000
    );
    let _ = std::thread::Builder::new()
        .name("time scale counts".into())
        .spawn(move || lines(f));
}

/// Every five seconds: the reads a second by clock (those read at all), and the waits.
fn lines(f: f64) {
    const EVERY: Duration = Duration::from_secs(5);
    let mut last = std::time::Instant::now();
    loop {
        std::thread::sleep(EVERY);
        let seconds = last.elapsed().as_secs_f64().max(1e-3);
        last = std::time::Instant::now();
        let counts = timescale::take_counts();
        let reads: Vec<String> = Source::ALL
            .iter()
            .zip(counts.reads)
            .filter(|&(_, n)| n > 0)
            .map(|(s, n)| format!("{} {:.0}", s.name(), n as f64 / seconds))
            .collect();
        tracing::info!(
            "time scale {f}: title time reads a second: {}; waits converted a second: {:.1}, adding {:.0} real ms a second, \
             {:.1} a second cut at the cap",
            if reads.is_empty() {
                "none".to_owned()
            } else {
                reads.join(", ")
            },
            counts.waits as f64 / seconds,
            counts.added_ms as f64 / seconds,
            counts.capped as f64 / seconds,
        );
    }
}

// The clocks.

pub extern "system" fn QueryPerformanceCounter(out: *mut i64) -> i32 {
    let Some(now) = timescale::counter() else {
        return host::query_performance_counter(out);
    };
    timescale::count(Source::Counter);
    if out.is_null() {
        return 0;
    }
    // SAFETY: checked non-null; the title passes its out pointer.
    unsafe { out.write(now) };
    1
}

/// The C++ library's counter (`steady_clock`, `high_resolution_clock`).
pub extern "C" fn _Query_perf_counter() -> i64 {
    let Some(now) = timescale::counter() else {
        let mut now = 0;
        host::query_performance_counter(&mut now);
        return now;
    };
    timescale::count(Source::StlCounter);
    now
}

pub extern "system" fn GetTickCount64() -> u64 {
    match timescale::tick_ms() {
        Some(ms) => {
            timescale::count(Source::TickCount64);
            ms
        }
        None => host::get_tick_count64(),
    }
}

pub extern "system" fn GetTickCount() -> u32 {
    match timescale::tick_ms() {
        Some(ms) => {
            timescale::count(Source::TickCount);
            ms as u32
        }
        None => host::get_tick_count64() as u32,
    }
}

/// The system time as the title sees it, counted as a read of `source`.
fn file_time(source: Source) -> u64 {
    match timescale::file_time() {
        Some(time) => {
            timescale::count(source);
            time
        }
        None => host::file_time(),
    }
}

fn write_file_time(out: *mut FILETIME, time: u64) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes its out pointer.
        unsafe {
            out.write(FILETIME {
                dwLowDateTime: time as u32,
                dwHighDateTime: (time >> 32) as u32,
            })
        };
    }
}

pub extern "system" fn GetSystemTimeAsFileTime(out: *mut FILETIME) {
    write_file_time(out, file_time(Source::FileTime));
}

pub extern "system" fn GetSystemTimePreciseAsFileTime(out: *mut FILETIME) {
    write_file_time(out, file_time(Source::FileTimePrecise));
}

pub extern "system" fn GetSystemTime(out: *mut SYSTEMTIME) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes its out pointer.
        unsafe { out.write(host::system_time(file_time(Source::SystemTime))) };
    }
}

pub extern "system" fn GetLocalTime(out: *mut SYSTEMTIME) {
    if !out.is_null() {
        let utc = host::system_time(file_time(Source::LocalTime));
        // SAFETY: checked non-null; the title passes its out pointer.
        unsafe { out.write(host::local_time(&utc)) };
    }
}

/// The C++ library's system clock: 100 ns intervals since 1970.
pub extern "C" fn _Xtime_get_ticks() -> i64 {
    file_time(Source::StlTicks) as i64 - UNIX_EPOCH_FILE_TIME as i64
}

/// Seconds since 1970 at the system time `time`.
fn unix_seconds(time: u64) -> i64 {
    (time as i64 - UNIX_EPOCH_FILE_TIME as i64).div_euclid(10_000_000)
}

/// The C library's `time`.
pub extern "C" fn _time64(out: *mut i64) -> i64 {
    let seconds = unix_seconds(file_time(Source::Time64));
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes its out pointer.
        unsafe { out.write(seconds) };
    }
    seconds
}

/// The C library's `struct __timeb64`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct Timeb {
    pub time: i64,
    pub millitm: u16,
    pub timezone: i16,
    pub dstflag: i16,
}

/// The C library's `_ftime64`: the host fills in the time zone, the time is the title's.
pub extern "C" fn _ftime64(out: *mut Timeb) {
    if out.is_null() {
        return;
    }
    host::ftime64(out);
    if timescale::active() {
        let time = file_time(Source::Ftime64);
        // SAFETY: checked non-null; the title passes its out pointer.
        let out = unsafe { &mut *out };
        out.time = unix_seconds(time);
        out.millitm = ((time - UNIX_EPOCH_FILE_TIME) / 10_000 % 1_000) as u16;
    }
}

/// Media Foundation's clock: the performance counter in 100 ns units.
pub extern "system" fn MFGetSystemTime() -> i64 {
    let Some(now) = timescale::counter() else {
        return host::mf_get_system_time();
    };
    timescale::count(Source::MediaTime);
    (i128::from(now) * 10_000_000 / i128::from(timescale::frequency())) as i64
}

// The waits: the title's time-outs are spans of its own time.

pub extern "system" fn Sleep(ms: u32) {
    host::sleep(timescale::real_wait_ms(ms));
}

pub extern "system" fn SleepEx(ms: u32, alertable: i32) -> u32 {
    host::sleep_ex(timescale::real_wait_ms(ms), alertable)
}

pub extern "system" fn WaitForSingleObject(handle: *mut c_void, ms: u32) -> u32 {
    host::wait_one(handle, timescale::real_wait_ms(ms), 0)
}

pub extern "system" fn WaitForSingleObjectEx(handle: *mut c_void, ms: u32, alertable: i32) -> u32 {
    host::wait_one(handle, timescale::real_wait_ms(ms), alertable)
}

pub extern "system" fn WaitForMultipleObjects(
    count: u32,
    handles: *const *mut c_void,
    all: i32,
    ms: u32,
) -> u32 {
    host::wait_many(count, handles, all, timescale::real_wait_ms(ms), 0)
}

pub extern "system" fn WaitForMultipleObjectsEx(
    count: u32,
    handles: *const *mut c_void,
    all: i32,
    ms: u32,
    alertable: i32,
) -> u32 {
    host::wait_many(count, handles, all, timescale::real_wait_ms(ms), alertable)
}

pub extern "system" fn SleepConditionVariableCS(
    condition: *mut c_void,
    section: *mut c_void,
    ms: u32,
) -> i32 {
    host::sleep_condition_cs(condition, section, timescale::real_wait_ms(ms))
}

pub extern "system" fn SleepConditionVariableSRW(
    condition: *mut c_void,
    lock: *mut c_void,
    ms: u32,
    flags: u32,
) -> i32 {
    host::sleep_condition_srw(condition, lock, timescale::real_wait_ms(ms), flags)
}

pub extern "system" fn WaitOnAddress(
    address: *const c_void,
    compare: *const c_void,
    size: usize,
    ms: u32,
) -> i32 {
    host::wait_on_address(address, compare, size, timescale::real_wait_ms(ms))
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "time",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-profile-*",
            "api-ms-win-core-sysinfo-*",
            "api-ms-win-core-synch-*",
            "msvcp140.dll",
            "ucrtbase.dll",
            "api-ms-win-crt-time-*",
            "mfplat.dll",
        ],
        exports: vec![
            hle_export!(QueryPerformanceCounter),
            hle_export!(_Query_perf_counter),
            hle_export!(GetTickCount64),
            hle_export!(GetTickCount),
            hle_export!(GetSystemTimeAsFileTime),
            hle_export!(GetSystemTimePreciseAsFileTime),
            hle_export!(GetSystemTime),
            hle_export!(GetLocalTime),
            hle_export!(_Xtime_get_ticks),
            hle_export!(_time64),
            hle_export!(_ftime64),
            hle_export!(MFGetSystemTime),
            hle_export!(Sleep),
            hle_export!(SleepEx),
            hle_export!(WaitForSingleObject),
            hle_export!(WaitForSingleObjectEx),
            hle_export!(WaitForMultipleObjects),
            hle_export!(WaitForMultipleObjectsEx),
            hle_export!(SleepConditionVariableCS),
            hle_export!(SleepConditionVariableSRW),
            hle_export!(WaitOnAddress),
        ],
    }
}

mod host {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::{FILETIME, SYSTEMTIME};
    use windows_sys::Win32::System::Performance::QueryPerformanceCounter;
    use windows_sys::Win32::System::SystemInformation::{
        GetSystemTimePreciseAsFileTime, GetTickCount64,
    };
    use windows_sys::Win32::System::Threading::{
        Sleep, SleepConditionVariableCS, SleepConditionVariableSRW, SleepEx,
        WaitForMultipleObjectsEx, WaitForSingleObjectEx, WaitOnAddress,
    };
    use windows_sys::Win32::System::Time::{FileTimeToSystemTime, SystemTimeToTzSpecificLocalTime};

    use super::Timeb;
    use crate::modules::files::host;

    // SAFETY (every function): the title's own arguments, passed on to the host's function
    // of the same name.

    pub(super) fn query_performance_counter(out: *mut i64) -> i32 {
        // SAFETY: see above.
        unsafe { QueryPerformanceCounter(out) }
    }

    pub(super) fn get_tick_count64() -> u64 {
        // SAFETY: no preconditions.
        unsafe { GetTickCount64() }
    }

    pub(super) fn file_time() -> u64 {
        let mut time = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        // SAFETY: a valid out pointer.
        unsafe { GetSystemTimePreciseAsFileTime(&mut time) };
        u64::from(time.dwHighDateTime) << 32 | u64::from(time.dwLowDateTime)
    }

    /// The UTC calendar time of the system time `time`.
    pub(super) fn system_time(time: u64) -> SYSTEMTIME {
        let time = FILETIME {
            dwLowDateTime: time as u32,
            dwHighDateTime: (time >> 32) as u32,
        };
        // SAFETY: plain data, filled in by the call.
        let mut out: SYSTEMTIME = unsafe { std::mem::zeroed() };
        // SAFETY: valid pointers.
        unsafe { FileTimeToSystemTime(&time, &mut out) };
        out
    }

    /// `utc` in the host's time zone.
    pub(super) fn local_time(utc: &SYSTEMTIME) -> SYSTEMTIME {
        let mut out = *utc;
        // SAFETY: valid pointers; a null zone is the host's current one.
        unsafe { SystemTimeToTzSpecificLocalTime(std::ptr::null(), utc, &mut out) };
        out
    }

    pub(super) fn ftime64(out: *mut Timeb) {
        host!("ucrtbase.dll", "_ftime64": extern "C" fn(*mut Timeb))(out)
    }

    pub(super) fn mf_get_system_time() -> i64 {
        host!("mfplat.dll", "MFGetSystemTime": extern "system" fn() -> i64)()
    }

    pub(super) fn sleep(ms: u32) {
        // SAFETY: see above.
        unsafe { Sleep(ms) }
    }

    pub(super) fn sleep_ex(ms: u32, alertable: i32) -> u32 {
        // SAFETY: see above.
        unsafe { SleepEx(ms, alertable) }
    }

    pub(super) fn wait_one(handle: *mut c_void, ms: u32, alertable: i32) -> u32 {
        // SAFETY: see above.
        unsafe { WaitForSingleObjectEx(handle, ms, alertable) }
    }

    pub(super) fn wait_many(
        count: u32,
        handles: *const *mut c_void,
        all: i32,
        ms: u32,
        alertable: i32,
    ) -> u32 {
        // SAFETY: see above.
        unsafe { WaitForMultipleObjectsEx(count, handles, all, ms, alertable) }
    }

    pub(super) fn sleep_condition_cs(condition: *mut c_void, section: *mut c_void, ms: u32) -> i32 {
        // SAFETY: see above.
        unsafe { SleepConditionVariableCS(condition.cast(), section.cast(), ms) }
    }

    pub(super) fn sleep_condition_srw(
        condition: *mut c_void,
        lock: *mut c_void,
        ms: u32,
        flags: u32,
    ) -> i32 {
        // SAFETY: see above.
        unsafe { SleepConditionVariableSRW(condition.cast(), lock.cast(), ms, flags) }
    }

    pub(super) fn wait_on_address(
        address: *const c_void,
        compare: *const c_void,
        size: usize,
        ms: u32,
    ) -> i32 {
        // SAFETY: see above.
        unsafe { WaitOnAddress(address, compare, size, ms) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_export_is_a_host_function_of_a_dll_the_title_imports_it_from() {
        let m = module();
        let mut names: Vec<_> = m.exports.iter().map(|e| e.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), m.exports.len(), "an export named twice");
        // The DLL names the title's images import them from.
        for dll in [
            "KERNEL32.dll",
            "api-ms-win-core-profile-l1-1-0.dll",
            "api-ms-win-core-sysinfo-l1-1-0.dll",
            "api-ms-win-core-sysinfo-l1-2-0.dll",
            "api-ms-win-core-synch-l1-1-0.dll",
            "api-ms-win-core-synch-l1-2-0.dll",
            "api-ms-win-core-synch-l1-2-1.dll",
            "MSVCP140.dll",
            "api-ms-win-crt-time-l1-1-0.dll",
            "MFPlat.DLL",
        ] {
            assert!(m.serves(dll), "{dll}");
        }
        // Each one overrides a host function of the same name (no typo binds nothing);
        // `WaitOnAddress` is in KernelBase only.
        for name in names {
            let dlls = [
                "kernel32.dll",
                "kernelbase.dll",
                "msvcp140.dll",
                "ucrtbase.dll",
                "mfplat.dll",
            ];
            let found = dlls
                .iter()
                .any(|dll| crate::host::function(dll, name).is_some());
            assert!(found, "{name} is no host function");
        }
    }

    /// In a child process of this test program that sets the scale (read once per process):
    /// the title's imports bind ours, and a `Sleep` of its time lasts four times as long.
    #[test]
    fn a_scaled_title_binds_ours_and_sleeps_its_own_time() {
        const NAME: &str =
            "modules::time::tests::a_scaled_title_binds_ours_and_sleeps_its_own_time";
        if std::env::var_os(timescale::ENV).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--test-threads=1"])
                .env(timescale::ENV, "0.25")
                .status()
                .unwrap();
            assert!(status.success(), "the scaled child failed");
            return;
        }
        assert!(active());
        start();
        let ours = [
            (
                "api-ms-win-core-profile-l1-1-0.dll",
                "QueryPerformanceCounter",
                QueryPerformanceCounter as *const () as usize,
            ),
            (
                "api-ms-win-core-sysinfo-l1-1-0.dll",
                "GetTickCount64",
                GetTickCount64 as *const () as usize,
            ),
            (
                "api-ms-win-core-synch-l1-2-0.dll",
                "Sleep",
                Sleep as *const () as usize,
            ),
            (
                "MSVCP140.dll",
                "_Query_perf_counter",
                _Query_perf_counter as *const () as usize,
            ),
            (
                "MFPlat.DLL",
                "MFGetSystemTime",
                MFGetSystemTime as *const () as usize,
            ),
        ];
        for (dll, name, address) in ours {
            assert_eq!(
                crate::registry::lookup(dll, name),
                Some(address),
                "{dll}!{name}"
            );
        }
        let (mut first, mut last) = (0, 0);
        QueryPerformanceCounter(&mut first);
        let ticks = GetTickCount64();
        let real = std::time::Instant::now();
        Sleep(40);
        let real = real.elapsed();
        QueryPerformanceCounter(&mut last);
        assert!(real >= Duration::from_millis(160), "slept {real:?}");
        let title_ms = (last - first) as f64 * 1e3 / timescale::frequency() as f64;
        let quarter = real.as_secs_f64() * 1e3 / 4.0;
        assert!(
            title_ms >= 40.0 && title_ms < quarter + 5.0,
            "{title_ms} ms"
        );
        assert!(GetTickCount64() - ticks >= 39);
        let counts = timescale::take_counts();
        assert!(counts.reads[Source::Counter as usize] >= 2 && counts.waits >= 1);
    }

    #[test]
    fn not_scaled_the_clocks_and_waits_are_the_hosts() {
        // The test process has no scale: every function is the host's.
        assert!(!active());
        let (mut a, mut b) = (0, 0);
        assert_eq!(QueryPerformanceCounter(&mut a), 1);
        assert_eq!(QueryPerformanceCounter(&mut b), 1);
        assert!(b >= a && a > 0);
        assert!(_Query_perf_counter() >= b);
        let ticks = GetTickCount64();
        assert!(ticks.abs_diff(host::get_tick_count64()) < 1_000);
        let mut time = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        GetSystemTimePreciseAsFileTime(&mut time);
        let time = u64::from(time.dwHighDateTime) << 32 | u64::from(time.dwLowDateTime);
        assert!(time.abs_diff(host::file_time()) < 10_000_000);
        // `_Xtime_get_ticks` and `_time64` count from 1970.
        let unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        assert!((_Xtime_get_ticks() / 10_000_000).abs_diff(unix.as_secs() as i64) <= 2);
        let mut seconds = 0;
        assert!(_time64(&mut seconds).abs_diff(unix.as_secs() as i64) <= 2);
        assert!(seconds.abs_diff(unix.as_secs() as i64) <= 2);
        let mut timeb = Timeb::default();
        _ftime64(&mut timeb);
        assert!(timeb.time.abs_diff(unix.as_secs() as i64) <= 2 && timeb.millitm < 1_000);
        assert!(MFGetSystemTime() > 0);
        // A poll of an event nobody sets times out at once (258, WAIT_TIMEOUT).
        // SAFETY: a nameless manual-reset event, not set.
        let event = unsafe {
            windows_sys::Win32::System::Threading::CreateEventW(
                std::ptr::null(),
                1,
                0,
                std::ptr::null(),
            )
        };
        let begun = std::time::Instant::now();
        assert_eq!(WaitForSingleObject(event, 0), 258);
        assert_eq!(WaitForMultipleObjects(1, &event, 1, 5), 258);
        Sleep(1);
        assert!(begun.elapsed() < Duration::from_secs(2));
        // SAFETY: closes the event above.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(event) };
    }

    #[test]
    fn calendar_times_come_from_the_system_time() {
        // 2025-03-04 05:06:07.089 UTC.
        let time = UNIX_EPOCH_FILE_TIME + 1_741_064_767_089 * 10_000;
        let st = host::system_time(time);
        assert_eq!(
            (
                st.wYear,
                st.wMonth,
                st.wDay,
                st.wHour,
                st.wMinute,
                st.wSecond,
                st.wMilliseconds
            ),
            (2025, 3, 4, 5, 6, 7, 89)
        );
        assert_eq!(unix_seconds(time), 1_741_064_767);
        assert_eq!(
            unix_seconds(UNIX_EPOCH_FILE_TIME - 1),
            -1,
            "before 1970 rounds down"
        );
    }
}
