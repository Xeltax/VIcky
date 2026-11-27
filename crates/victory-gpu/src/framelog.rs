//! Every frame's length, and what the long ones were made of.
//!
//! The frame-rate line averages over a second or more, and its counters
//! are reset at each line: a single long frame is lost in it, and so is
//! what that frame was made of. Run 222's dips (150 ms frames with some of
//! 250 to 500 ms among them) could not be placed from those lines.
//!
//! Here every counter runs from the start and is never reset. At the end
//! of each present call the totals are read, and their differences since
//! the present before are that frame's. Every five seconds one line gives
//! the frame times (each one, the shortest, the median, the longest), one
//! gives the usual frame's make-up (the mean of the frames that were not
//! long), and each long frame gets a line of its own in the same form, so
//! a long frame is read against a usual one column by column.
//!
//! On by default in diagnostics builds, off in others;
//! `VICTORY_FRAME_LOG=1` or `0` says otherwise. The counters themselves
//! are adds in every build, each thread to a block of its own (VIcky,
//! [`crate::perthread`]: they were adjacent shared atomics).
//!
//! VIcky: the title's turn. The frame is a loop through our submitting
//! thread and the title's own work between two presents, and only the
//! first was measured. Each frame also gives, from the return of the
//! present before to this one's call: the time to the title's first
//! submit call (`queue::execute`, the first hand-over of the frame's
//! lists), from that call to the return of the last one (the burst), and
//! from there to the present call; and the list-call time of the one
//! recording thread that made the most of them (the frame line has only
//! their sum). A line after the usual frame's gives their mean
//! ([`turn_text`]); a long frame's line ends with its own.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

/// Totals since the start (microseconds unless said otherwise).
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum T {
    /// The title's present call, all of it; of it, running the queues'
    /// lines and settling submissions before our copy; and our copy to the
    /// window with the host's present.
    PresentUs,
    PresentLineUs,
    PresentShowUs,
    /// The title's submit calls (`queue::execute`), on every thread.
    ExecuteUs,
    /// The submit thread of ours (`VICTORY_ASYNC_SUBMIT=1`) at work: wall
    /// time running the queues' lines, and its processor time doing so.
    WorkerUs,
    WorkerUserUs,
    WorkerKernelUs,
    /// Submissions sent; preparing them and waiting for the card, on
    /// whichever thread (the frame-rate line's "preparing" and "on the
    /// GPU").
    Submissions,
    PrepareUs,
    CardWaitUs,
    /// The card with nothing of ours queued, and how much of that while
    /// the submitting thread was working.
    CardIdleUs,
    CardIdleBehindUs,
    /// Copies of title memory for submissions (time, bytes), and what
    /// shaders wrote copied back (time, bytes).
    CopyUs,
    CopyBytes,
    CopyBackUs,
    CopyBackBytes,
    /// List calls that record into host lists, on every thread
    /// (nanoseconds, calls). Diagnostics builds only.
    RecordNs,
    RecordCalls,
    /// The runtime changing what title memory shows: calls, the time from
    /// asking for the lock to letting it go, and of that the wait for
    /// readers.
    Remaps,
    RemapUs,
    RemapWaitUs,
    /// Texture refreshes on every thread, and of them the wait for title
    /// memory to hold still; uploads recorded.
    RefreshUs,
    RefreshStillUs,
    Uploads,
    /// Waiting for host pipelines being made.
    PipelineWaitUs,
    /// Writes taken by the guards on view memory.
    GuardFaults,
    /// The title's frame waits: time slept, on every thread.
    PaceUs,
    /// DirectStorage requests finished, their destination bytes, and the
    /// workers' time in them.
    DiskReads,
    DiskBytes,
    DiskUs,
    /// The card's time by kind of list (`crate::cardmeter`), as read a
    /// second and a half after the lists were recorded: our lists that
    /// bring copies in the card's memory up to date and hash their pieces,
    /// our compare lists, and every other list; and how many of each.
    CardSyncUs,
    CardSyncLists,
    CardCheckUs,
    CardCheckLists,
    CardListsUs,
    CardLists,
    /// Looks at which processor a thread is on ([`on_slow_core`]), and how
    /// many found a slow one (an efficiency core of a hybrid processor):
    /// the thread that presents (one look a present), our submit thread
    /// (one a turn), and the recording threads (one every couple of
    /// milliseconds of list calls).
    RenderLooks,
    RenderSlow,
    WorkerLooks,
    WorkerSlow,
    RecordLooks,
    RecordSlow,
    Count,
}

const N: usize = T::Count as usize;
static TOTAL: crate::perthread::Counters<N> = crate::perthread::Counters::new();

/// Adds to a total.
pub fn add(t: T, n: u64) {
    TOTAL.add(t as usize, n);
}

/// Takes from a total (a time that was summed over threads is corrected
/// to the time it took).
pub fn sub(t: T, n: u64) {
    TOTAL.sub(t as usize, n);
}

/// What a thread did inside our calls since its last present: for the
/// thread that presents (the title's render thread).
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum H {
    /// List calls, nanoseconds.
    RecordNs,
    /// Submit calls, microseconds.
    ExecuteUs,
    /// Frame waits, microseconds slept.
    PaceUs,
    Count,
}

const HN: usize = H::Count as usize;

thread_local! {
    static HERE: std::cell::Cell<[u64; HN]> = const { std::cell::Cell::new([0; HN]) };
}

/// Adds to this thread's own count.
pub fn here(h: H, n: u64) {
    HERE.with(|c| {
        let mut v = c.get();
        v[h as usize] += n;
        c.set(v);
    });
}

/// VIcky: microseconds since the first look, plus one (0 is "none").
fn turn_us(at: std::time::Instant) -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    at.saturating_duration_since(*EPOCH.get_or_init(std::time::Instant::now)).as_micros() as u64 + 1
}

/// The frame's hand-overs so far ([`turn_us`]): the earliest call, the
/// latest return (`u64::MAX` and 0 while there is none), how many; when the
/// present was called, and whether the title is inside it now.
static FIRST_HANDOVER: AtomicU64 = AtomicU64::new(u64::MAX);
static LAST_HANDOVER: AtomicU64 = AtomicU64::new(0);
static HANDOVERS: AtomicU64 = AtomicU64::new(0);
static PRESENT_CALLED: AtomicU64 = AtomicU64::new(0);
static PRESENTING: AtomicBool = AtomicBool::new(false);

/// The title's submit call that began at `started` returns. One made while
/// the title is inside its present (by another of its threads) is not the
/// next frame's hand-over and is left out.
pub fn handed_over(started: std::time::Instant) {
    if !on() || PRESENTING.load(Relaxed) {
        return;
    }
    FIRST_HANDOVER.fetch_min(turn_us(started), Relaxed);
    LAST_HANDOVER.fetch_max(turn_us(std::time::Instant::now()), Relaxed);
    HANDOVERS.fetch_add(1, Relaxed);
}

/// The title calls its present (`frame` is called when it returns).
pub fn present_called() {
    if !on() {
        return;
    }
    PRESENT_CALLED.store(turn_us(std::time::Instant::now()), Relaxed);
    PRESENTING.store(true, Relaxed);
}

/// Frames ended so far ([`frame`]), for the list-call time of each thread
/// by frame: the low 16 bits tag [`LONGEST`].
static FRAME_NUMBER: AtomicU64 = AtomicU64::new(0);
/// Each thread's list-call time in the frame it last recorded in: the
/// frame's tag in the top 16 bits, nanoseconds below. The frame's longest
/// is the most of those tagged with it, taken at [`frame`]. (VIcky: one
/// shared word before, compared and swapped at every list call of the
/// thread then leading, and read by all the others at theirs.)
static LONGEST: crate::perthread::Counters<1> = crate::perthread::Counters::new();
const LONGEST_NS: u64 = (1 << 48) - 1;

thread_local! {
    /// This thread's list calls in a frame: the frame's tag, nanoseconds.
    static MINE: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// A list call of `ns` nanoseconds returned on this thread (diagnostics
/// builds, `crate::copystat::recorded`).
pub fn recorded_here(ns: u64) {
    if !on() {
        return;
    }
    let frame = FRAME_NUMBER.load(Relaxed) & 0xFFFF;
    let mine = MINE.with(|m| {
        let (tag, so_far) = m.get();
        let mine = if tag == frame { so_far + ns } else { ns };
        m.set((frame, mine));
        mine.min(LONGEST_NS)
    });
    // (Past the per-thread blocks, threads share one word, kept as before.)
    LONGEST.set_if(0, frame << 48 | mine, |old, new| {
        let (tag, most, frame, mine) = (old >> 48, old & LONGEST_NS, new >> 48, new & LONGEST_NS);
        // A thread still counting a frame that has ended changes nothing.
        let newer = frame.wrapping_sub(tag) & 0xFFFF;
        (newer != 0 && newer < 0x8000) || (newer == 0 && mine > most)
    });
}

/// The most list-call time one thread had in frame `ended` (its tag).
fn longest_in(ended: u64) -> u64 {
    LONGEST.each(0).filter(|v| v >> 48 == ended).map(|v| v & LONGEST_NS).max().unwrap_or(0)
}

/// The title's turn in a frame (microseconds; list calls in nanoseconds).
#[derive(Clone, Copy)]
#[repr(usize)]
enum W {
    /// From the present before's return to this present's call, and of it:
    /// to the first hand-over's call, from there to the last one's return,
    /// from there to the present call. A frame with no hand-over has all
    /// of it in the first.
    TurnUs,
    ToFirstUs,
    BurstUs,
    ToPresentUs,
    /// Submit calls in the burst.
    Handovers,
    /// The one thread with the most list-call time, its nanoseconds.
    LongestRecordNs,
    Count,
}

const WN: usize = W::Count as usize;

/// The frame's turn, the present before having returned at `returned`;
/// the hand-overs are taken.
fn turn(returned: std::time::Instant, longest_ns: u64) -> [f64; WN] {
    let first = FIRST_HANDOVER.swap(u64::MAX, Relaxed);
    let last = LAST_HANDOVER.swap(0, Relaxed);
    let n = HANDOVERS.swap(0, Relaxed);
    let (from, to) = (turn_us(returned), PRESENT_CALLED.load(Relaxed));
    let mut w = [0.0; WN];
    w[W::TurnUs as usize] = to.saturating_sub(from) as f64;
    w[W::Handovers as usize] = n as f64;
    w[W::LongestRecordNs as usize] = longest_ns as f64;
    if n == 0 || first == u64::MAX {
        w[W::ToFirstUs as usize] = w[W::TurnUs as usize];
        return w;
    }
    w[W::ToFirstUs as usize] = first.saturating_sub(from) as f64;
    w[W::BurstUs as usize] = last.saturating_sub(first) as f64;
    w[W::ToPresentUs as usize] = to.saturating_sub(last) as f64;
    w
}

/// Whether the processor this thread runs on right now is a slow one: on a
/// processor with two kinds of cores (the console has eight alike; an Intel
/// hybrid has fast cores and efficiency cores at about half their speed),
/// one of a lower efficiency class than the best. Always false where all
/// cores are alike.
pub fn on_slow_core() -> bool {
    use windows_sys::Win32::System::Threading::GetCurrentProcessorNumber;
    /// By logical processor (of group 0): whether it is slow.
    static SLOW: std::sync::OnceLock<Vec<bool>> = std::sync::OnceLock::new();
    let slow = SLOW.get_or_init(|| {
        let sets = cpu_sets();
        let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
        let mut slow = vec![false; sets.iter().map(|c| c.1 as usize + 1).max().unwrap_or(0)];
        for (_, processor, class) in &sets {
            slow[*processor as usize] = *class < best;
        }
        tracing::info!(
            "processors: {} logical, {} of them slow ones (a lower efficiency class than the best); the frame log counts how \
             often the render, submit and recording threads are found on a slow one",
            slow.len(),
            slow.iter().filter(|&&s| s).count()
        );
        slow
    });
    // SAFETY: a plain query.
    let processor = unsafe { GetCurrentProcessorNumber() } as usize;
    slow.get(processor).copied().unwrap_or(false)
}

/// The system's processor sets of group 0: each one's id, its logical
/// processor and its efficiency class (higher is faster).
pub(crate) fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.LogicalProcessorIndex, set.EfficiencyClass));
        }
        at += info.Size as usize;
    }
    sets
}

/// The slow logical processors of group 0 as a mask (a lower efficiency
/// class than the best): 0 where all cores are alike.
pub fn slow_processors() -> usize {
    let sets = cpu_sets();
    let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
    sets.iter().filter(|c| c.2 < best && c.1 < usize::BITS as u8).fold(0, |mask, c| mask | 1usize << c.1)
}

/// Asks the system to run the calling thread on the fast cores only, where
/// the processor has two kinds (`VICTORY_FAST_CORES=0`: not asked).
///
/// The title keeps its own threads on fast cores (run 229: its render
/// thread in 3,484 looks of 3,484, its recording threads in 104,254 of
/// 104,254), as on the console, whose eight cores are alike. The layer's
/// submit thread was found on a slow one in 1,529 looks of 24,021, and the
/// frame waits for that thread. A set of processors selected for a thread
/// is a preference the scheduler keeps unless nothing of it is free.
pub fn keep_on_fast_cores(what: &str) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_FAST_CORES").is_none_or(|v| v != "0")) {
        return;
    }
    let sets = cpu_sets();
    let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.2 == best).map(|c| c.0).collect();
    // All alike (or nothing known): nothing to ask for.
    if fast.is_empty() || fast.len() == sets.len() {
        return;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), fast.as_ptr(), fast.len() as u32) };
    if ok == 0 {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on the fast cores");
    } else {
        tracing::info!(what, fast = fast.len(), of = sets.len(), "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)");
    }
}

/// Asks the system to run the calling thread on the slow cores only, where
/// the processor has two kinds and at least eight slow logical processors
/// (a helper that must not take a fast core from the title's threads).
/// True if it was asked.
///
/// The title pins its threads as on the console, where disk decompression
/// is the hardware's: measured on a 14700KF, 31 of its threads are tied to
/// logical processors 0-13, the fast cores. The layer decompresses on as
/// many worker threads as there are logical processors (28), free to run
/// anywhere: in the opening cutscene's streaming stretch about ten of them
/// were busy while the title's recording calls took three times as long.
pub fn keep_on_slow_cores(what: &str) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    let sets = cpu_sets();
    let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
    let slow: Vec<u32> = sets.iter().filter(|c| c.2 < best).map(|c| c.0).collect();
    if slow.len() < 8 {
        return false;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), slow.as_ptr(), slow.len() as u32) } != 0;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(what, slow = slow.len(), of = sets.len(), ok, "helper threads are kept on the slow cores: the first one");
    });
    ok
}

/// Whether frames are logged.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_FRAME_LOG") {
        Some(v) => v != "0",
        None => cfg!(feature = "diagnostics"),
    })
}

fn micros(t: windows_sys::Win32::Foundation::FILETIME) -> u64 {
    (u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime)) / 10
}

/// This thread's processor time so far (user, kernel), in microseconds.
pub fn thread_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own thread and four valid out pointers.
    unsafe { GetThreadTimes(GetCurrentThread(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

/// The process's processor time so far (user, kernel), in microseconds.
fn process_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own process and four valid out pointers.
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

/// The process's page faults so far (soft and hard).
fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: plain data the call fills in.
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// The PC's free physical memory, in MiB.
fn free_mb() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    unsafe { GlobalMemoryStatusEx(&mut m) };
    m.ullAvailPhys >> 20
}

/// Values read when a frame ends, all running totals but the last.
const S: usize = 11;
const S_THREAD_USER: usize = 0;
const S_THREAD_KERNEL: usize = 1;
const S_PROCESS_USER: usize = 2;
const S_PROCESS_KERNEL: usize = 3;
const S_PAGE_FAULTS: usize = 4;
const S_DRAWS: usize = 5;
const S_PIPELINES_MADE: usize = 6;
const S_PIPELINES_LOADED: usize = 7;
const S_STAGING_REUSED: usize = 8;
const S_STAGING_MADE: usize = 9;
/// Not a total: the free memory then.
const S_FREE_MB: usize = 10;

fn sample() -> [u64; S] {
    let (tu, tk) = thread_times();
    let (pu, pk) = process_times();
    [
        tu,
        tk,
        pu,
        pk,
        page_faults(),
        crate::draw::drawn(),
        crate::pipeline::frame_log_counts().0,
        crate::pipeline::frame_log_counts().1,
        crate::texture::SPARE_STATS[0].load(Relaxed),
        crate::texture::SPARE_STATS[1].load(Relaxed),
        free_mb(),
    ]
}

/// One frame: its number, when it ended (seconds since the first), its
/// length in microseconds, and the differences of the totals, the sampled
/// values and the presenting thread's own counts over it; and the title's
/// turn in it.
#[derive(Clone)]
struct Frame {
    number: u64,
    at: f64,
    us: f64,
    t: [f64; N],
    s: [f64; S],
    h: [f64; HN],
    w: [f64; WN],
}

struct State {
    first: std::time::Instant,
    /// The last frame's end: when, the totals and samples then, and the
    /// thread that presented.
    last: Option<(std::time::Instant, [u64; N], [u64; S], u32)>,
    number: u64,
    frames: Vec<Frame>,
    since: std::time::Instant,
    /// The lengths of the last frames (microseconds), for the usual one.
    recent: std::collections::VecDeque<f64>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A frame's make-up as text (`f` may be a mean of frames).
fn text(f: &Frame) -> String {
    let t = |x: T| f.t[x as usize];
    let ms = |x: T| f.t[x as usize] / 1e3;
    let s = |i: usize| f.s[i];
    let h = |x: H| f.h[x as usize];
    format!(
        "{:.0} ms | render thread: ran {:.0} ms (user {:.0}, kernel {:.0}); in the present call {:.1} (lines and settling \
         {:.1}, copy and show {:.1}), in submit calls {:.1}, in list calls {:.1}, paced {:.1} | submit thread: at work \
         {:.1} ms (user {:.1}, kernel {:.1}); {:.0} submissions, preparing {:.1}, waiting for the card {:.1}; copies {:.1} \
         ms {:.1} MB; copy-back {:.1} ms {:.2} MB | card: idle {:.1} ms ({:.1} behind us); by its own clock, of the lists of a moment ago: our \
         send-and-hash lists {:.1} ms in {:.0}, our compare lists {:.1} ms in {:.0}, all other lists {:.1} ms in {:.0} | \
         recording, all threads: {:.1} ms in {:.0} calls; all submit calls {:.1} ms | textures: refresh {:.1} ms ({:.1} waiting for memory to hold \
         still), {:.0} uploads, staging {:.0} made {:.0} reused | remapping: {:.0} calls {:.1} ms ({:.1} waiting for \
         readers) | pipelines: {:.0} made, {:.0} loaded, {:.1} ms waited | disk: {:.0} reads {:.1} MB {:.1} worker ms | \
         faults: {:.0} page, {:.0} guard | process: {:.0} user {:.0} kernel ms | pacing, all threads {:.1} ms | {:.0} \
         draws | {:.0} MB free | found on a slow core: the render thread {:.0} of {:.0} looks, the submit thread {:.0} of {:.0}, recording threads {:.0} of {:.0}",
        f.us / 1e3,
        (s(S_THREAD_USER) + s(S_THREAD_KERNEL)) / 1e3,
        s(S_THREAD_USER) / 1e3,
        s(S_THREAD_KERNEL) / 1e3,
        ms(T::PresentUs),
        ms(T::PresentLineUs),
        ms(T::PresentShowUs),
        h(H::ExecuteUs) / 1e3,
        h(H::RecordNs) / 1e6,
        h(H::PaceUs) / 1e3,
        ms(T::WorkerUs),
        ms(T::WorkerUserUs),
        ms(T::WorkerKernelUs),
        t(T::Submissions),
        ms(T::PrepareUs),
        ms(T::CardWaitUs),
        ms(T::CopyUs),
        t(T::CopyBytes) / 1e6,
        ms(T::CopyBackUs),
        t(T::CopyBackBytes) / 1e6,
        ms(T::CardIdleUs),
        ms(T::CardIdleBehindUs),
        ms(T::CardSyncUs),
        t(T::CardSyncLists),
        ms(T::CardCheckUs),
        t(T::CardCheckLists),
        ms(T::CardListsUs),
        t(T::CardLists),
        t(T::RecordNs) / 1e6,
        t(T::RecordCalls),
        ms(T::ExecuteUs),
        ms(T::RefreshUs),
        ms(T::RefreshStillUs),
        t(T::Uploads),
        s(S_STAGING_MADE),
        s(S_STAGING_REUSED),
        t(T::Remaps),
        ms(T::RemapUs),
        ms(T::RemapWaitUs),
        s(S_PIPELINES_MADE),
        s(S_PIPELINES_LOADED),
        ms(T::PipelineWaitUs),
        t(T::DiskReads),
        t(T::DiskBytes) / 1e6,
        ms(T::DiskUs),
        s(S_PAGE_FAULTS),
        t(T::GuardFaults),
        s(S_PROCESS_USER) / 1e3,
        s(S_PROCESS_KERNEL) / 1e3,
        ms(T::PaceUs),
        s(S_DRAWS),
        s(S_FREE_MB),
        t(T::RenderSlow),
        t(T::RenderLooks),
        t(T::WorkerSlow),
        t(T::WorkerLooks),
        t(T::RecordSlow),
        t(T::RecordLooks),
    )
}

/// A frame's turn as text (`f` may be a mean of frames).
fn turn_text(f: &Frame) -> String {
    let ms = |x: W| f.w[x as usize] / 1e3;
    format!(
        "{:.1} ms (present return -> first hand-over {:.1}, burst {:.1} in {:.1} submit calls, last hand-over -> present \
         {:.1}); longest recording thread {:.1} ms (all threads {:.1})",
        ms(W::TurnUs),
        ms(W::ToFirstUs),
        ms(W::BurstUs),
        f.w[W::Handovers as usize],
        ms(W::ToPresentUs),
        f.w[W::LongestRecordNs as usize] / 1e6,
        f.t[T::RecordNs as usize] / 1e6,
    )
}

/// The mean of `frames` (not empty).
fn mean(frames: &[&Frame]) -> Frame {
    let n = frames.len() as f64;
    let mut out = Frame { number: 0, at: 0.0, us: 0.0, t: [0.0; N], s: [0.0; S], h: [0.0; HN], w: [0.0; WN] };
    for f in frames {
        out.us += f.us / n;
        for i in 0..N {
            out.t[i] += f.t[i] / n;
        }
        for i in 0..S {
            out.s[i] += f.s[i] / n;
        }
        for i in 0..HN {
            out.h[i] += f.h[i] / n;
        }
        for i in 0..WN {
            out.w[i] += f.w[i] / n;
        }
    }
    out
}

/// Over how many microseconds a frame is long, the usual one being
/// `usual`: half as long again, and at least 40 ms more.
fn long_over(usual: f64) -> f64 {
    (usual * 1.5).max(usual + 40_000.0)
}

/// The lines of a reading: the frame times, the usual frame, the long ones.
fn lines(frames: &[Frame], usual: f64, seconds: f64) -> Vec<String> {
    let over = long_over(usual);
    let (long, plain): (Vec<&Frame>, Vec<&Frame>) = frames.iter().partition(|f| f.us > over);
    let each: Vec<String> = frames.iter().map(|f| format!("{:.0}", f.us / 1e3)).collect();
    let shortest = frames.iter().map(|f| f.us).fold(f64::INFINITY, f64::min);
    let longest = frames.iter().map(|f| f.us).fold(0.0, f64::max);
    let mut out = vec![format!(
        "frame times: {} frames in {seconds:.1} s: shortest {:.0} ms, median {:.0}, longest {:.0}; {} long (over {:.0} ms, \
         the usual frame of the last {RECENT} being {:.0}); each: {}",
        frames.len(),
        shortest / 1e3,
        median(frames.iter().map(|f| f.us)) / 1e3,
        longest / 1e3,
        long.len(),
        over / 1e3,
        usual / 1e3,
        each.join(" ")
    )];
    if !plain.is_empty() {
        let usual = mean(&plain);
        out.push(format!("usual frame (the mean of {} not long): {}", plain.len(), text(&usual)));
        out.push(format!("title's turn (the mean of {} not long): {}", plain.len(), turn_text(&usual)));
    }
    for f in long.iter().take(LONG_MOST) {
        out.push(format!("long frame {} at {:.1} s: {} | title's turn: {}", f.number, f.at, text(f), turn_text(f)));
    }
    if long.len() > LONG_MOST {
        out.push(format!("long frames: {} more in this reading are not listed", long.len() - LONG_MOST));
    }
    out
}

/// Frames whose lengths give the usual one, and the long frames listed in
/// a reading at most.
const RECENT: usize = 64;
const LONG_MOST: usize = 8;

/// A frame ends: called at the end of the title's present call, on the
/// thread that made it.
pub fn frame() {
    if !on() {
        return;
    }
    crate::cardmeter::collect();
    add(T::RenderLooks, 1);
    if on_slow_core() {
        add(T::RenderSlow, 1);
    }
    let now = std::time::Instant::now();
    let totals: [u64; N] = TOTAL.totals();
    let sampled = sample();
    let here = HERE.with(|c| c.replace([0; HN]));
    // The frame's longest recording thread; list calls from here on count
    // for the next one.
    let ended = FRAME_NUMBER.fetch_add(1, Relaxed) & 0xFFFF;
    let longest_ns = longest_in(ended);
    // SAFETY: a plain query.
    let thread = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = state.get_or_insert_with(|| State {
        first: now,
        last: None,
        number: 0,
        frames: Vec::new(),
        since: now,
        recent: std::collections::VecDeque::new(),
    });
    state.number += 1;
    let last = state.last.replace((now, totals, sampled, thread));
    // The hand-overs are taken at every frame, the first one's too.
    let w = turn(last.as_ref().map_or(now, |l| l.0), longest_ns);
    PRESENTING.store(false, Relaxed);
    if let Some((then, t0, s0, thread0)) = last {
        let mut f = Frame {
            number: state.number,
            at: (now - state.first).as_secs_f64(),
            us: (now - then).as_micros() as f64,
            t: std::array::from_fn(|i| totals[i].wrapping_sub(t0[i]) as i64 as f64),
            s: std::array::from_fn(|i| sampled[i].wrapping_sub(s0[i]) as i64 as f64),
            h: here.map(|v| v as f64),
            w,
        };
        f.s[S_FREE_MB] = sampled[S_FREE_MB] as f64;
        if thread != thread0 {
            // Another thread presented: its processor time is not this frame's.
            (f.s[S_THREAD_USER], f.s[S_THREAD_KERNEL]) = (0.0, 0.0);
        }
        // VIcky: lot 4's probes and `VICTORY_CCD_PLACE`, frame by frame.
        #[cfg(feature = "diagnostics")]
        crate::lot4::frame_ended(f.us, f.t[T::WorkerUs as usize], f.t[T::RecordNs as usize], f.w[W::LongestRecordNs as usize]);
        state.frames.push(f);
    }
    let elapsed = state.since.elapsed();
    if elapsed < std::time::Duration::from_secs(5) || state.frames.is_empty() {
        return;
    }
    let frames = std::mem::take(&mut state.frames);
    state.since = now;
    // The usual frame: the median of the last frames, this reading's with
    // those before it.
    for f in &frames {
        if state.recent.len() == RECENT {
            state.recent.pop_front();
        }
        state.recent.push_back(f.us);
    }
    let usual = median(state.recent.iter().copied());
    for line in lines(&frames, usual, elapsed.as_secs_f64()) {
        tracing::info!("{line}");
    }
    // VIcky (`VICTORY_LOT4_PROBE`): the probes' lines with the reading.
    #[cfg(feature = "diagnostics")]
    crate::lot4::period(frames.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(number: u64, ms: f64) -> Frame {
        let mut f = Frame { number, at: number as f64, us: ms * 1e3, t: [0.0; N], s: [0.0; S], h: [0.0; HN], w: [0.0; WN] };
        f.t[T::PresentUs as usize] = ms * 100.0;
        f.t[T::Uploads as usize] = ms;
        f.w[W::TurnUs as usize] = ms * 100.0;
        f.w[W::BurstUs as usize] = ms * 10.0;
        f
    }

    #[test]
    fn a_long_frame_is_listed_beside_the_usual_one() {
        let frames: Vec<Frame> = [150.0, 160.0, 140.0, 400.0, 150.0].iter().enumerate().map(|(i, &ms)| frame(i as u64, ms)).collect();
        let out = lines(&frames, 150_000.0, 1.0);
        assert_eq!(out.len(), 4, "{out:#?}");
        assert!(out[0].starts_with("frame times: 5 frames in 1.0 s: shortest 140 ms, median 150, longest 400; 1 long (over 225 ms"), "{}", out[0]);
        assert!(out[0].ends_with("each: 150 160 140 400 150"), "{}", out[0]);
        // The usual frame is the mean of the four that are not long.
        assert!(out[1].starts_with("usual frame (the mean of 4 not long): 150 ms |"), "{}", out[1]);
        assert!(out[1].contains("in the present call 15.0 "), "{}", out[1]);
        assert!(out[1].contains(" 150 uploads"), "{}", out[1]);
        // The title's turn of the same frames.
        assert!(out[2].starts_with("title's turn (the mean of 4 not long): 15.0 ms (present return -> first hand-over 0.0, burst 1.5 in"), "{}", out[2]);
        assert!(out[3].starts_with("long frame 3 at 3.0 s: 400 ms |"), "{}", out[3]);
        assert!(out[3].contains("in the present call 40.0 "), "{}", out[3]);
        assert!(out[3].contains(" 400 uploads"), "{}", out[3]);
        assert!(out[3].contains("| title's turn: 40.0 ms (present return -> first hand-over 0.0, burst 4.0 in"), "{}", out[3]);
    }

    /// The turn's parts: to the first hand-over, the burst, to the present.
    #[test]
    fn a_turn_is_cut_at_the_first_and_last_hand_overs() {
        // (The clock's start, before the times the test gives.)
        let _ = turn_us(std::time::Instant::now());
        let returned = std::time::Instant::now();
        let at = |ms: u64| returned + std::time::Duration::from_millis(ms);
        // (Taken as `turn` takes them; the test sets them as the hooks do.)
        FIRST_HANDOVER.store(turn_us(at(30)), Relaxed);
        LAST_HANDOVER.store(turn_us(at(45)), Relaxed);
        HANDOVERS.store(9, Relaxed);
        PRESENT_CALLED.store(turn_us(at(50)), Relaxed);
        let near = |w: &[f64; WN], x: W, ms: f64| (w[x as usize] / 1e3 - ms).abs() < 0.01;
        let w = turn(returned, 7_000_000);
        assert!(near(&w, W::TurnUs, 50.0) && near(&w, W::ToFirstUs, 30.0) && near(&w, W::BurstUs, 15.0) && near(&w, W::ToPresentUs, 5.0), "{w:?}");
        assert_eq!(w[W::Handovers as usize], 9.0);
        assert_eq!(w[W::LongestRecordNs as usize], 7e6);
        // Taken: a frame without hand-overs has its whole turn before the first.
        PRESENT_CALLED.store(turn_us(at(20)), Relaxed);
        let w = turn(returned, 0);
        assert!(near(&w, W::TurnUs, 20.0) && near(&w, W::ToFirstUs, 20.0) && near(&w, W::BurstUs, 0.0), "{w:?}");
    }

    #[test]
    fn long_is_half_as_long_again_and_at_least_40_ms_more() {
        assert_eq!(long_over(150_000.0), 225_000.0);
        assert_eq!(long_over(30_000.0), 70_000.0);
    }

    #[test]
    fn more_long_frames_than_are_listed_are_counted() {
        let mut frames: Vec<Frame> = (0..12).map(|i| frame(i, 500.0)).collect();
        frames.push(frame(12, 100.0));
        let out = lines(&frames, 100_000.0, 5.0);
        assert_eq!(out.iter().filter(|l| l.starts_with("long frame ")).count(), LONG_MOST);
        assert!(out.last().unwrap().starts_with("long frames: 4 more"), "{out:#?}");
    }

    #[test]
    fn totals_and_a_threads_own_counts_add_up() {
        let before = TOTAL.total(T::DiskReads as usize);
        add(T::DiskReads, 3);
        sub(T::DiskReads, 1);
        assert_eq!(TOTAL.total(T::DiskReads as usize).wrapping_sub(before), 2);
        here(H::PaceUs, 5);
        here(H::PaceUs, 6);
        assert_eq!(HERE.with(|c| c.get())[H::PaceUs as usize], 11);
    }
}
