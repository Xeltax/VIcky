//! The title's clock, slowed down for an experiment: `VICTORY_TIME_SCALE=<f>`, 0 < f < 1.
//!
//! The title renders a frame here in about 200 ms where the console takes about 33 ms, and
//! what it drives by frame time (eye adaptation, temporal anti-aliasing, denoisers, motion)
//! may break at such long steps. With a scale `f`, every clock the title reads through our
//! code shows `start + (now - start) * f`: at 0.17 a real 200 ms frame looks like 34 ms to the
//! title, and the game runs in slow motion. Unset, 1, or anything that is not a number in
//! (0, 1): nothing changes, and the title's imports stay bound to the host's own functions.
//!
//! * One start for the whole process ([`start`]): the performance counter, the system time
//!   and the tick count, read back to back.
//! * One reading behind every clock: each value is worked out from the same scaled delta of
//!   the performance counter, so the counter, the tick count, the file time and the
//!   microsecond clocks agree with each other on every thread, and none of them ever goes
//!   back (the system time no longer follows adjustments of the wall clock while scaled).
//! * Frequencies stay what they are: the counter ticks at its own rate, only fewer times.
//! * The waits the title asks for are spans of its own time: see [`real_wait_ms`].
//! * [`count`] keeps how often each clock was read and the waits converted, for a line in
//!   the log (`victory_hle::modules::time`).

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

/// The switch.
pub const ENV: &str = "VICTORY_TIME_SCALE";

/// A wait converted to real time never lasts longer than this, unless the title asked for
/// longer itself: then it waits what it asked.
pub const WAIT_CAP_MS: u32 = 10_000;

/// The wait without end (`INFINITE`), never converted.
pub const INFINITE: u32 = u32::MAX;

/// 100 ns intervals from 1601 (a `FILETIME`'s origin) to 1970.
pub const UNIX_EPOCH_FILE_TIME: u64 = 116_444_736_000_000_000;

/// The scale `text` asks for: `None` unless it is a number in (0, 1).
pub fn parse(text: &str) -> Option<f64> {
    let f: f64 = text.trim().parse().ok()?;
    (f > 0.0 && f < 1.0).then_some(f)
}

/// The scale in force, read from [`ENV`] once: `None` when time is not scaled.
pub fn factor() -> Option<f64> {
    static FACTOR: LazyLock<Option<f64>> = LazyLock::new(|| parse(&std::env::var(ENV).ok()?));
    *FACTOR
}

/// Whether the title's time is scaled.
pub fn active() -> bool {
    factor().is_some()
}

/// The clocks at the start, and the counter's rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Base {
    /// The performance counter.
    pub counter: i64,
    /// Its ticks a second.
    pub frequency: i64,
    /// The system time, in 100 ns intervals since 1601.
    pub file_time: u64,
    /// Milliseconds since the machine started (`GetTickCount64`).
    pub tick_ms: u64,
}

impl Base {
    /// The scaled delta of the counter the title sees when the real counter shows `real`:
    /// `(real - start) * f`, rounded down; 0 before the start.
    pub fn delta(&self, real: i64, f: f64) -> i64 {
        let real = real.saturating_sub(self.counter).max(0);
        // Rounding of a product and the cut to an integer both keep the order, so a later
        // reading never gives a smaller delta.
        (real as f64 * f) as i64
    }

    /// `delta` counter ticks in units of which there are `per_second` a second, rounded down.
    fn units(&self, delta: i64, per_second: u64) -> u128 {
        delta.max(0) as u128 * u128::from(per_second) / self.frequency.max(1) as u128
    }

    pub fn counter_at(&self, delta: i64) -> i64 {
        self.counter + delta
    }

    pub fn file_time_at(&self, delta: i64) -> u64 {
        self.file_time + self.units(delta, 10_000_000) as u64
    }

    pub fn tick_ms_at(&self, delta: i64) -> u64 {
        self.tick_ms + self.units(delta, 1_000) as u64
    }

    /// Nanoseconds since the start.
    pub fn ns_at(&self, delta: i64) -> u128 {
        self.units(delta, 1_000_000_000)
    }
}

static BASE: LazyLock<Base> = LazyLock::new(host::base);

/// Takes the start of every clock now, if it was not taken yet (the first reading takes it
/// otherwise). The runtime calls it before the title runs.
pub fn start() -> Base {
    *BASE
}

/// The scaled delta now, when time is scaled.
fn now() -> Option<(&'static Base, i64)> {
    let f = factor()?;
    let base = &*BASE;
    Some((base, base.delta(host::counter(), f)))
}

/// The performance counter as the title sees it (`QueryPerformanceCounter`).
pub fn counter() -> Option<i64> {
    now().map(|(base, d)| base.counter_at(d))
}

/// The performance counter's ticks a second (unchanged by the scale).
pub fn frequency() -> i64 {
    BASE.frequency
}

/// The system time as the title sees it, in 100 ns intervals since 1601.
pub fn file_time() -> Option<u64> {
    now().map(|(base, d)| base.file_time_at(d))
}

/// Milliseconds since the machine started, as the title sees them (`GetTickCount64`).
pub fn tick_ms() -> Option<u64> {
    now().map(|(base, d)| base.tick_ms_at(d))
}

/// Nanoseconds since the start on the title's clock.
pub fn title_ns() -> Option<u128> {
    now().map(|(base, d)| base.ns_at(d))
}

/// Microseconds since the start on the title's clock.
pub fn title_us() -> Option<u64> {
    title_ns().map(|ns| (ns / 1_000) as u64)
}

/// A title wait of `ms` milliseconds of its own time, in real milliseconds at scale `f`, and
/// whether [`WAIT_CAP_MS`] shortened it. The title means its own time: a `Sleep(16)` is 16 ms
/// of the slowed clock, so 16 / f real ms (95 ms at 0.17, rounded up), and a frame limiter
/// or a poll paced that way sees what it asked for. A long wait would stall its thread for a
/// long real time, so a converted wait ends after at most 10 s, or after the title's own
/// value if that is longer. 0 (a poll) and `INFINITE` stay as they are.
pub fn convert_wait_ms(ms: u32, f: f64) -> (u32, bool) {
    if ms == 0 || ms == INFINITE {
        return (ms, false);
    }
    let real = (f64::from(ms) / f).ceil();
    let cap = ms.max(WAIT_CAP_MS);
    if real > f64::from(cap) {
        (cap, true)
    } else {
        (real as u32, false)
    }
}

/// [`convert_wait_ms`] at the scale in force, counted when converted; `ms` itself when time
/// is not scaled.
pub fn real_wait_ms(ms: u32) -> u32 {
    let Some(f) = factor() else { return ms };
    if ms == 0 || ms == INFINITE {
        return ms;
    }
    let (real, capped) = convert_wait_ms(ms, f);
    count_wait(ms, real, capped);
    real
}

/// [`convert_wait_ms`] for a span of the title's time with any resolution.
pub fn convert_duration(span: Duration, f: f64) -> (Duration, bool) {
    if span.is_zero() {
        return (span, false);
    }
    let real = span.div_f64(f);
    let cap = span.max(Duration::from_millis(WAIT_CAP_MS.into()));
    if real > cap {
        (cap, true)
    } else {
        (real, false)
    }
}

/// [`convert_duration`] at the scale in force, counted when converted; `span` itself when
/// time is not scaled.
pub fn real_duration(span: Duration) -> Duration {
    let Some(f) = factor() else { return span };
    if span.is_zero() {
        return span;
    }
    let (real, capped) = convert_duration(span, f);
    count_wait(span.as_millis() as u32, real.as_millis() as u32, capped);
    real
}

/// The clocks the title reads through our code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Counter,
    StlCounter,
    TickCount,
    TickCount64,
    FileTime,
    FileTimePrecise,
    StlTicks,
    SystemTime,
    LocalTime,
    Time64,
    Ftime64,
    MediaTime,
    InputTimestamp,
    GpuTimestamp,
    SystemTimestamp,
}

impl Source {
    pub const ALL: [Source; 15] = [
        Source::Counter,
        Source::StlCounter,
        Source::TickCount,
        Source::TickCount64,
        Source::FileTime,
        Source::FileTimePrecise,
        Source::StlTicks,
        Source::SystemTime,
        Source::LocalTime,
        Source::Time64,
        Source::Ftime64,
        Source::MediaTime,
        Source::InputTimestamp,
        Source::GpuTimestamp,
        Source::SystemTimestamp,
    ];

    /// As the title knows it.
    pub fn name(self) -> &'static str {
        match self {
            Source::Counter => "QueryPerformanceCounter",
            Source::StlCounter => "_Query_perf_counter",
            Source::TickCount => "GetTickCount",
            Source::TickCount64 => "GetTickCount64",
            Source::FileTime => "GetSystemTimeAsFileTime",
            Source::FileTimePrecise => "GetSystemTimePreciseAsFileTime",
            Source::StlTicks => "_Xtime_get_ticks",
            Source::SystemTime => "GetSystemTime",
            Source::LocalTime => "GetLocalTime",
            Source::Time64 => "_time64",
            Source::Ftime64 => "_ftime64",
            Source::MediaTime => "MFGetSystemTime",
            Source::InputTimestamp => "GameInput timestamps",
            Source::GpuTimestamp => "GPU timestamps",
            Source::SystemTimestamp => "system timestamps written by lists",
        }
    }
}

/// Reads by source since the last [`take_counts`].
static READS: [AtomicU64; Source::ALL.len()] = [const { AtomicU64::new(0) }; Source::ALL.len()];
/// Waits converted, the real milliseconds they added to what the title asked for, and those
/// the cap shortened, since the last [`take_counts`].
static WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// One read of `source` by the title.
pub fn count(source: Source) {
    READS[source as usize].fetch_add(1, Relaxed);
}

fn count_wait(asked_ms: u32, real_ms: u32, capped: bool) {
    WAITS[0].fetch_add(1, Relaxed);
    WAITS[1].fetch_add(u64::from(real_ms.saturating_sub(asked_ms)), Relaxed);
    if capped {
        WAITS[2].fetch_add(1, Relaxed);
    }
}

/// What was counted since the last call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counts {
    /// Reads by source, in the order of [`Source::ALL`].
    pub reads: [u64; Source::ALL.len()],
    pub waits: u64,
    /// Real milliseconds the conversion added to what the title asked for.
    pub added_ms: u64,
    pub capped: u64,
}

/// What was counted since the last call, and counting starts again.
pub fn take_counts() -> Counts {
    Counts {
        reads: std::array::from_fn(|i| READS[i].swap(0, Relaxed)),
        waits: WAITS[0].swap(0, Relaxed),
        added_ms: WAITS[1].swap(0, Relaxed),
        capped: WAITS[2].swap(0, Relaxed),
    }
}

#[cfg(windows)]
mod host {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Performance::{
        QueryPerformanceCounter, QueryPerformanceFrequency,
    };
    use windows_sys::Win32::System::SystemInformation::{
        GetSystemTimePreciseAsFileTime, GetTickCount64,
    };

    pub(super) fn counter() -> i64 {
        let mut now = 0;
        // SAFETY: a valid out pointer.
        unsafe { QueryPerformanceCounter(&mut now) };
        now
    }

    pub(super) fn base() -> super::Base {
        let mut frequency = 0;
        // SAFETY: a valid out pointer.
        unsafe { QueryPerformanceFrequency(&mut frequency) };
        let mut time = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        // Read back to back, the counter last: no clock of the title starts behind another.
        // SAFETY: a valid out pointer; no preconditions.
        let (tick_ms, file_time) = unsafe {
            GetSystemTimePreciseAsFileTime(&mut time);
            (
                GetTickCount64(),
                u64::from(time.dwHighDateTime) << 32 | u64::from(time.dwLowDateTime),
            )
        };
        super::Base {
            counter: counter(),
            frequency: frequency.max(1),
            file_time,
            tick_ms,
        }
    }
}

/// Elsewhere (tests on other hosts): a nanosecond counter from the process's first reading.
#[cfg(not(windows))]
mod host {
    use std::sync::LazyLock;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    static ORIGIN: LazyLock<Instant> = LazyLock::new(Instant::now);

    pub(super) fn counter() -> i64 {
        ORIGIN.elapsed().as_nanos() as i64
    }

    pub(super) fn base() -> super::Base {
        let since = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        super::Base {
            counter: counter(),
            frequency: 1_000_000_000,
            file_time: super::UNIX_EPOCH_FILE_TIME + (since.as_nanos() / 100) as u64,
            tick_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 10 MHz counter (the console's and most PCs') started at an odd value.
    const BASE: Base = Base {
        counter: 123_456_789,
        frequency: 10_000_000,
        file_time: 133_000_000_000_000_000,
        tick_ms: 5_000_000,
    };

    #[test]
    fn only_numbers_between_0_and_1_scale() {
        assert_eq!(parse("0.17"), Some(0.17));
        assert_eq!(parse(" 0.5 "), Some(0.5));
        for off in ["", "1", "1.0", "0", "-0.5", "2", "NaN", "inf", "fast"] {
            assert_eq!(parse(off), None, "{off:?}");
        }
    }

    #[test]
    fn the_title_sees_start_plus_elapsed_times_the_scale() {
        let f = 0.17;
        // A real 200 ms frame (2,000,000 ticks) looks like 34 ms.
        let d = BASE.delta(BASE.counter + 2_000_000, f);
        assert_eq!(d, 340_000);
        assert_eq!(BASE.counter_at(d), BASE.counter + 340_000);
        assert_eq!(BASE.tick_ms_at(d), BASE.tick_ms + 34);
        assert_eq!(BASE.file_time_at(d), BASE.file_time + 340_000);
        assert_eq!(BASE.ns_at(d), 34_000_000);
        // At the start every clock shows its start; before it (another core's earlier
        // reading) too.
        for real in [BASE.counter, BASE.counter - 5] {
            let d = BASE.delta(real, f);
            assert_eq!(d, 0);
            assert_eq!(
                (BASE.counter_at(d), BASE.tick_ms_at(d), BASE.file_time_at(d)),
                (BASE.counter, BASE.tick_ms, BASE.file_time)
            );
        }
    }

    #[test]
    fn every_clock_only_moves_forward() {
        for f in [0.17, 0.5, 1.0 / 3.0, 0.999_999, 1e-6] {
            let mut last = (0, 0, 0, 0);
            // Steps of one tick, then of uneven sizes, up to about a day of counter.
            let mut real = BASE.counter;
            for step in (0..20_000).map(|i| {
                if i < 10_000 {
                    1
                } else {
                    (i * 7_919) % 99_991 + 1
                }
            }) {
                real += step;
                let d = BASE.delta(real, f);
                let now = (
                    BASE.counter_at(d),
                    BASE.tick_ms_at(d),
                    BASE.file_time_at(d),
                    BASE.ns_at(d),
                );
                assert!(
                    now.0 >= last.0 && now.1 >= last.1 && now.2 >= last.2 && now.3 >= last.3,
                    "f {f} at {real}: {now:?} after {last:?}"
                );
                last = now;
            }
            // Over a day the scaled delta stays within a tick of the exact product.
            let day = 864_000_000_000i64;
            let exact = day as f64 * f;
            assert!(
                (BASE.delta(BASE.counter + day, f) as f64 - exact).abs() <= 1.0,
                "f {f}"
            );
        }
    }

    #[test]
    fn counter_ticks_and_file_time_agree() {
        // Any reading: the three clocks show the same span of the title's time, each rounded
        // down to its own unit.
        let f = 0.17;
        for real in [
            1,
            9,
            59,
            12_345,
            10_000_000,
            987_654_321,
            86_400 * 10_000_000,
        ] {
            let d = BASE.delta(BASE.counter + real, f);
            let counter_ns = (BASE.counter_at(d) - BASE.counter) as u128 * 100;
            let file_ns = u128::from(BASE.file_time_at(d) - BASE.file_time) * 100;
            let tick_ns = u128::from(BASE.tick_ms_at(d) - BASE.tick_ms) * 1_000_000;
            assert_eq!(counter_ns, file_ns, "at {real}");
            assert_eq!(counter_ns, BASE.ns_at(d));
            assert!(
                tick_ns <= counter_ns && counter_ns - tick_ns < 1_000_000,
                "at {real}"
            );
        }
        // A counter of another rate (3 MHz): the units still agree to within one of each.
        let odd = Base {
            frequency: 3_000_000,
            ..BASE
        };
        let d = odd.delta(odd.counter + 3_000_000 * 60, 0.25);
        assert_eq!(d, 45_000_000);
        assert_eq!(odd.file_time_at(d) - odd.file_time, 150_000_000);
        assert_eq!(odd.tick_ms_at(d) - odd.tick_ms, 15_000);
        assert_eq!(odd.ns_at(d), 15_000_000_000);
    }

    #[test]
    fn waits_are_the_titles_time_with_a_cap() {
        let f = 0.17;
        assert_eq!(convert_wait_ms(0, f), (0, false), "a poll stays a poll");
        assert_eq!(convert_wait_ms(INFINITE, f), (INFINITE, false));
        assert_eq!(convert_wait_ms(1, f), (6, false));
        assert_eq!(convert_wait_ms(16, f), (95, false));
        assert_eq!(convert_wait_ms(33, f), (195, false));
        // The title's watchdog sleeps 1 s between its looks: 5.9 s now.
        assert_eq!(convert_wait_ms(1_000, f), (5_883, false));
        // Long ones end at 10 s, or at the title's own value when that is longer.
        assert_eq!(convert_wait_ms(5_000, f), (WAIT_CAP_MS, true));
        assert_eq!(convert_wait_ms(30_000, f), (30_000, true));
        assert_eq!(convert_wait_ms(INFINITE - 1, f), (INFINITE - 1, true));
        // Never shorter than asked.
        for ms in [1, 7, 100, 9_999, 10_000, 100_000] {
            for f in [1e-3, 0.17, 0.9] {
                assert!(convert_wait_ms(ms, f).0 >= ms);
            }
        }
        let (span, capped) = convert_duration(Duration::from_micros(33_333), f);
        assert!(!capped && span.as_micros() == 196_076, "{span:?}");
        assert_eq!(
            convert_duration(Duration::from_secs(60), f),
            (Duration::from_secs(60), true)
        );
        assert_eq!(convert_duration(Duration::ZERO, f), (Duration::ZERO, false));
    }

    #[test]
    fn the_host_start_reads_sane_clocks() {
        let base = host::base();
        assert!(base.frequency > 0);
        // After 2001, before 2200.
        assert!(
            base.file_time > 126_000_000_000_000_000 && base.file_time < 190_000_000_000_000_000
        );
        assert!(host::counter() >= base.counter);
    }

    /// The switch itself, in a child process of this test program that sets it (the scale is
    /// read once per process).
    #[test]
    fn a_process_that_asks_sees_its_clocks_slowed() {
        const NAME: &str = "timescale::tests::a_process_that_asks_sees_its_clocks_slowed";
        if std::env::var_os(ENV).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--test-threads=1"])
                .env(ENV, "0.25")
                .status()
                .unwrap();
            assert!(status.success(), "the scaled child failed");
            return;
        }
        assert_eq!(factor(), Some(0.25));
        let base = start();
        let read = || {
            (
                counter().unwrap(),
                tick_ms().unwrap(),
                file_time().unwrap(),
                title_ns().unwrap(),
            )
        };
        let real = std::time::Instant::now();
        let first = read();
        std::thread::sleep(Duration::from_millis(400));
        let last = read();
        let real = real.elapsed().as_secs_f64();
        // A quarter of the real time passed for the title, the same on every clock.
        let seconds = (last.0 - first.0) as f64 / base.frequency as f64;
        assert!(
            (seconds - real * 0.25).abs() < 0.02,
            "{seconds} s in {real} s"
        );
        assert!(((last.2 - first.2) as f64 / 1e7 - seconds).abs() < 1e-3);
        assert!(((last.3 - first.3) as f64 / 1e9 - seconds).abs() < 1e-3);
        assert!(((last.1 - first.1) as f64 / 1e3 - seconds).abs() < 2e-3);
        assert!(first.0 >= base.counter && first.2 >= base.file_time && first.1 >= base.tick_ms);
        // The title's waits last four times as long; polls and endless waits stay.
        assert_eq!(real_wait_ms(16), 64);
        assert_eq!(real_wait_ms(0), 0);
        assert_eq!(real_wait_ms(INFINITE), INFINITE);
        let span = real_duration(Duration::from_millis(10));
        assert!(span.abs_diff(Duration::from_millis(40)) < Duration::from_micros(1));
    }

    #[test]
    fn counts_are_taken_and_start_again() {
        count(Source::Counter);
        count(Source::Counter);
        count(Source::GpuTimestamp);
        count_wait(16, 95, false);
        count_wait(5_000, 10_000, true);
        let counts = take_counts();
        assert!(counts.reads[Source::Counter as usize] >= 2);
        assert!(counts.reads[Source::GpuTimestamp as usize] >= 1);
        assert!(counts.waits >= 2 && counts.added_ms >= 79 + 5_000 && counts.capped >= 1);
        assert_eq!(
            Source::ALL
                .iter()
                .position(|&s| s == Source::SystemTimestamp),
            Some(Source::SystemTimestamp as usize)
        );
    }
}
