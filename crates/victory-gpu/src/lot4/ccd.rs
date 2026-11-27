//! P7's switch: `VICTORY_CCD_PLACE=1|alt` (off by default; it works without
//! `VICTORY_LOT4_PROBE`). Diagnostics builds only.
//!
//! The 9900X has two CCDs of six cores, each with its own 32 MB L3, and
//! nothing placed our threads by it (the fast and slow core logic does
//! nothing on this CPU). A lock word or a slot record that moves between
//! the CCDs costs far more than within one, and a frame streams over a
//! gigabyte through the caches. With the switch, the threads that share the
//! heap's locks are asked to run on one CCD and the bulk copy threads on
//! the other, as CPU sets (`SetThreadSelectedCpuSets`):
//!
//! - on the CCD that holds most of logical processors 0-15 (where the title
//!   pins its own threads): the submitting thread (`victory-submit`, and any
//!   thread that submits), the heap sync (`victory-heap-sync`), the
//!   parallel mirror's threads (`victory-mirror-*`), the pass over buffer
//!   views (`victory-heap-pass*`) and the title's recording threads (every
//!   thread that makes a list call);
//! - on the other: the wide copy threads (`victory-wide-*`), the relay
//!   (`victory-relay`) and the texture copy-back's writer
//!   (`victory-texture-back`, with `VICTORY_TEXTURE_BACK_ASYNC`).
//!
//! The CCDs are found at run time from the L3 caches
//! (`GetLogicalProcessorInformationEx`, `RelationCache`), not assumed. A
//! thread the title pinned to processors of the other CCD only keeps its
//! pinning (a hard affinity wins over CPU sets); the line counts those.
//! Off again, each thread gets back the CPU sets it had.
//!
//! `alt`: twenty seconds without, twenty with, in turn, from the first
//! frame; each change is said in the log as the other `=alt` switches say
//! theirs (`...(VICTORY_CCD_PLACE=alt): ON` / `: off`, which
//! `xcl-notes\alt-fps.sh` reads), with what the phase before took: frames,
//! the frame, the submit thread at work, the recording threads' sum and
//! the longest one, and the waits for `made` (with the probes on).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// The CCDs: per logical processor of group 0 its CCD, and per CCD its
/// processors (a mask of group 0) and their CPU set ids.
struct Topology {
    ccd_of: Vec<u8>,
    ccds: Vec<(usize, Vec<u32>)>,
}

fn topology() -> &'static Topology {
    static TOPOLOGY: OnceLock<Topology> = OnceLock::new();
    TOPOLOGY.get_or_init(|| {
        let masks = l3_masks();
        let mut ccd_of = vec![0u8; usize::BITS as usize];
        for (i, &mask) in masks.iter().enumerate() {
            for (p, ccd) in ccd_of.iter_mut().enumerate() {
                if mask >> p & 1 != 0 {
                    *ccd = i as u8;
                }
            }
        }
        let sets = crate::framelog::cpu_sets();
        let ccds = masks
            .iter()
            .map(|&mask| {
                let ids = sets.iter().filter(|s| (s.1 as u32) < usize::BITS && mask >> s.1 & 1 != 0).map(|s| s.0).collect();
                (mask, ids)
            })
            .collect();
        Topology { ccd_of, ccds }
    })
}

/// The processors (masks of group 0) that share each L3 cache, lowest
/// first.
fn l3_masks() -> Vec<usize> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationCache, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetLogicalProcessorInformationEx(RelationCache, std::ptr::null_mut(), &mut bytes) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetLogicalProcessorInformationEx(RelationCache, buffer.as_mut_ptr().cast(), &mut bytes) } != 0;
    let mut masks: Vec<usize> = Vec::new();
    let mut at = 0usize;
    while ok && at + size_of::<u32>() * 2 <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()) };
        if info.Size == 0 {
            break;
        }
        if info.Relationship == RelationCache {
            // SAFETY: a cache record.
            let cache = unsafe { &info.Anonymous.Cache };
            if cache.Level == 3 {
                let groups = usize::from(cache.GroupCount.max(1));
                // The record holds `GroupCount` masks from the union on (one
                // where the count is 0, on older systems).
                let first: *const windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY = (&raw const cache.Anonymous).cast();
                for g in 0..groups {
                    // SAFETY: as above.
                    let affinity = unsafe { &*first.add(g) };
                    if affinity.Group == 0 && affinity.Mask != 0 && !masks.contains(&affinity.Mask) {
                        masks.push(affinity.Mask);
                    }
                }
            }
        }
        at += info.Size as usize;
    }
    masks.sort_by_key(|m| m.trailing_zeros());
    masks
}

/// The CCD of the processor this thread runs on now (0 where there is one,
/// or it is not known).
#[inline]
pub(crate) fn current() -> u8 {
    // SAFETY: a plain query.
    let p = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() } as usize;
    topology().ccd_of.get(p).copied().unwrap_or(0)
}

/// 0 off, 1 on, 2 in turn (`VICTORY_CCD_PLACE`), read once; off where
/// there are fewer than two L3 caches.
pub(crate) fn mode() -> u8 {
    #[cfg(not(feature = "diagnostics"))]
    return 0;
    #[cfg(feature = "diagnostics")]
    {
        static MODE: OnceLock<u8> = OnceLock::new();
        *MODE.get_or_init(|| {
            let mode = match std::env::var("VICTORY_CCD_PLACE").as_deref() {
                Ok("1") => 1,
                Ok("alt") => 2,
                _ => return 0,
            };
            let t = topology();
            if t.ccds.len() < 2 {
                tracing::warn!("VICTORY_CCD_PLACE: {} L3 cache(s) found, nothing to place by: off", t.ccds.len());
                return 0;
            }
            let (a, b) = sides();
            tracing::info!(
                "threads placed by CCD (VICTORY_CCD_PLACE={}): {}; the heap's lock sharers (the submitting thread, the heap \
                 sync, the mirror's and the pass's threads, the title's recording threads) on CCD {a}, the bulk copy threads \
                 (wide, relay, texture copy-back writer) on CCD {b}",
                if mode == 1 { "1" } else { "alt: twenty seconds without, twenty with" },
                ccds_text(),
            );
            mode
        })
    }
}

/// The CCDs as text: each one's logical processors.
fn ccds_text() -> String {
    let t = topology();
    let text: Vec<String> = t
        .ccds
        .iter()
        .enumerate()
        .map(|(i, (mask, ids))| format!("CCD {i} (L3) processors {} ({} CPU sets)", processors(*mask), ids.len()))
        .collect();
    text.join(", ")
}

/// A mask as runs of processor numbers ("0-11").
fn processors(mask: usize) -> String {
    let mut runs: Vec<String> = Vec::new();
    let mut p = 0;
    while p < usize::BITS as usize {
        if mask >> p & 1 == 0 {
            p += 1;
            continue;
        }
        let first = p;
        while p < usize::BITS as usize && mask >> p & 1 != 0 {
            p += 1;
        }
        runs.push(if p - 1 == first { first.to_string() } else { format!("{first}-{}", p - 1) });
    }
    runs.join(",")
}

/// The CCD of the lock sharers (the one with most of processors 0-15, the
/// title's), and the bulk copy threads' (the largest of the others).
fn sides() -> (usize, usize) {
    let t = topology();
    let a = (0..t.ccds.len()).max_by_key(|&i| ((t.ccds[i].0 & 0xFFFF).count_ones(), std::cmp::Reverse(i))).unwrap_or(0);
    let b = (0..t.ccds.len()).filter(|&i| i != a).max_by_key(|&i| (t.ccds[i].0.count_ones(), std::cmp::Reverse(i))).unwrap_or(a);
    (a, b)
}

/// Which side a thread is placed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    /// With the heap's lock sharers.
    Locks,
    /// With the bulk copies.
    Bulk,
}

/// The side of one of our named threads, if it is placed.
fn side_of_name(name: &str) -> Option<Side> {
    const LOCKS: [&str; 4] = ["victory-submit", "victory-heap-sync", "victory-mirror-", "victory-heap-pass"];
    const BULK: [&str; 3] = ["victory-wide-", "victory-relay", "victory-texture-back"];
    if LOCKS.iter().any(|p| name.starts_with(p)) {
        Some(Side::Locks)
    } else if BULK.iter().any(|p| name.starts_with(p)) {
        Some(Side::Bulk)
    } else {
        None
    }
}

/// Whether the threads are placed now.
static PLACED: AtomicU8 = AtomicU8::new(0);

/// Threads that registered themselves (the title's recording threads, the
/// submitting ones), and per thread placed, the CPU sets it had before.
struct Threads {
    registered: Vec<u32>,
    before: BTreeMap<u32, Vec<u32>>,
    /// The last placement: threads placed, of them pinned by the title
    /// away from their CCD.
    placed: usize,
    pinned_away: usize,
}

static THREADS: Mutex<Threads> = Mutex::new(Threads { registered: Vec::new(), before: BTreeMap::new(), placed: 0, pinned_away: 0 });

thread_local! {
    static REGISTERED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A list call on this thread (a recording thread of the title).
#[inline]
pub(crate) fn recording() {
    if mode() != 0 && !REGISTERED.get() {
        register();
    }
}

/// A submission on this thread.
#[inline]
pub(crate) fn submitting() {
    recording();
}

#[cold]
fn register() {
    REGISTERED.set(true);
    // SAFETY: a plain query.
    let id = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let mut t = THREADS.lock().unwrap_or_else(|e| e.into_inner());
    t.registered.push(id);
    if PLACED.load(Relaxed) == 1 {
        let mut pinned = false;
        place_one(&mut t, id, Some(Side::Locks), &mut pinned);
    }
}

/// Places (`side`) or gives back its CPU sets (`None`) to thread `id`; true
/// if done. `pinned_away`: set if its affinity holds no processor of its
/// CCD.
fn place_one(t: &mut Threads, id: u32, side: Option<Side>, pinned_away: &mut bool) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
    use windows_sys::Win32::System::Threading::{
        GetThreadGroupAffinity, GetThreadSelectedCpuSets, OpenThread, SetThreadSelectedCpuSets, THREAD_QUERY_LIMITED_INFORMATION,
        THREAD_SET_LIMITED_INFORMATION,
    };
    // SAFETY: a thread of this process, by id; the handle is closed below.
    let h = unsafe { OpenThread(THREAD_QUERY_LIMITED_INFORMATION | THREAD_SET_LIMITED_INFORMATION, 0, id) };
    if h.is_null() {
        return false;
    }
    let topology = topology();
    let (a, b) = sides();
    let ok = match side {
        Some(side) => {
            if !t.before.contains_key(&id) {
                let mut had = [0u32; 64];
                let mut n = 0u32;
                // SAFETY: a buffer of 64 ids and its length.
                let read = unsafe { GetThreadSelectedCpuSets(h, had.as_mut_ptr(), had.len() as u32, &mut n) } != 0;
                t.before.insert(id, if read { had[..n as usize].to_vec() } else { Vec::new() });
            }
            let (mask, ids) = &topology.ccds[if side == Side::Locks { a } else { b }];
            let mut affinity: GROUP_AFFINITY = unsafe { std::mem::zeroed() };
            // SAFETY: the handle and an out structure.
            if unsafe { GetThreadGroupAffinity(h, &mut affinity) } != 0 && affinity.Group == 0 && affinity.Mask & mask == 0 {
                *pinned_away = true;
            }
            // SAFETY: the handle and the system's own set ids.
            unsafe { SetThreadSelectedCpuSets(h, ids.as_ptr(), ids.len() as u32) != 0 }
        }
        None => {
            let had = t.before.remove(&id).unwrap_or_default();
            // SAFETY: the handle and the ids it had (none: no selection).
            unsafe { SetThreadSelectedCpuSets(h, if had.is_empty() { std::ptr::null() } else { had.as_ptr() }, had.len() as u32) != 0 }
        }
    };
    // SAFETY: opened above.
    unsafe { CloseHandle(h) };
    ok
}

/// This process's threads by id, with their names (none for unnamed).
fn named_threads() -> Vec<(u32, Option<String>)> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE, LocalFree};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetThreadDescription, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    let mut out = Vec::new();
    // SAFETY: a snapshot handle, closed below; the entry says its own size;
    // each thread handle is closed, each description freed.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return out;
        }
        let process = GetCurrentProcessId();
        let mut entry: THREADENTRY32 = std::mem::zeroed();
        entry.dwSize = size_of::<THREADENTRY32>() as u32;
        let mut more = Thread32First(snapshot, &mut entry) != 0;
        while more {
            if entry.th32OwnerProcessID == process {
                let id = entry.th32ThreadID;
                let mut name = None;
                let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, id);
                if !h.is_null() {
                    let mut text: windows_sys::core::PWSTR = std::ptr::null_mut();
                    if GetThreadDescription(h, &mut text) >= 0 && !text.is_null() {
                        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
                        let s = String::from_utf16_lossy(std::slice::from_raw_parts(text, len));
                        if !s.is_empty() {
                            name = Some(s);
                        }
                        LocalFree(text.cast());
                    }
                    CloseHandle(h);
                }
                out.push((id, name));
            }
            more = Thread32Next(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    out
}

/// Places every thread there is a side for (`on`), or gives each placed
/// one its CPU sets back.
fn apply(on: bool) {
    let mut t = THREADS.lock().unwrap_or_else(|e| e.into_inner());
    let (mut placed, mut pinned_away) = (0, 0);
    if on {
        let mut targets: Vec<(u32, Side)> = named_threads().into_iter().filter_map(|(id, name)| Some((id, side_of_name(name.as_deref()?)?))).collect();
        for &id in &t.registered {
            if !targets.iter().any(|x| x.0 == id) {
                targets.push((id, Side::Locks));
            }
        }
        for (id, side) in targets {
            let mut away = false;
            if place_one(&mut t, id, Some(side), &mut away) {
                placed += 1;
                pinned_away += usize::from(away);
            }
        }
        PLACED.store(1, Relaxed);
    } else {
        PLACED.store(0, Relaxed);
        let ids: Vec<u32> = t.before.keys().copied().collect();
        for id in ids {
            let mut away = false;
            place_one(&mut t, id, None, &mut away);
        }
    }
    (t.placed, t.pinned_away) = (placed, pinned_away);
}

/// What the phase so far took: frames, and summed over them the frame,
/// the submit thread at work, the recording threads' sum and the longest
/// one's (microseconds); `made` waits at its start (nanoseconds, since the
/// start of the process).
struct Phase {
    started: Option<std::time::Instant>,
    on: bool,
    frames: u64,
    sums: [f64; 4],
    made_wait: u64,
    /// `=1`: when the threads were last placed.
    placed_at: Option<std::time::Instant>,
}

static PHASE: Mutex<Phase> =
    Mutex::new(Phase { started: None, on: false, frames: 0, sums: [0.0; 4], made_wait: 0, placed_at: None });

/// A frame ended (`super::frame_ended`): with `alt`, its counts go to the
/// phase, and the phase changes every twenty seconds; with `1`, the
/// threads are placed at the first frame and again every twenty seconds
/// (for threads made since).
pub(crate) fn frame_ended(us: f64, at_work_us: f64, record_ns: f64, longest_ns: f64) {
    let mode = mode();
    if mode == 0 {
        return;
    }
    let mut p = PHASE.lock().unwrap_or_else(|e| e.into_inner());
    let now = std::time::Instant::now();
    if mode == 1 {
        if p.placed_at.is_none_or(|at| now.duration_since(at).as_secs() >= 20) {
            let first = p.placed_at.is_none();
            p.placed_at = Some(now);
            drop(p);
            apply(true);
            let t = THREADS.lock().unwrap_or_else(|e| e.into_inner());
            if first {
                tracing::info!(
                    "threads placed by CCD (VICTORY_CCD_PLACE=1): {} threads placed, {} of them pinned by the title to \
                     processors of the other CCD only (their pinning wins)",
                    t.placed,
                    t.pinned_away
                );
            }
        }
        return;
    }
    let first = p.started.is_none();
    let started = *p.started.get_or_insert(now);
    p.frames += 1;
    for (sum, v) in p.sums.iter_mut().zip([us, at_work_us, record_ns / 1e3, longest_ns / 1e3]) {
        *sum += v;
    }
    let on = (now.duration_since(started).as_secs() / 20) % 2 == 1;
    // (The first frame says the first phase, off, as the other switches do.)
    if on == p.on && !first {
        return;
    }
    let was = (p.frames, p.sums, p.made_wait);
    p.on = on;
    p.frames = 0;
    p.sums = [0.0; 4];
    let made_wait = super::locks::made_wait_ns();
    p.made_wait = made_wait;
    drop(p);
    apply(on);
    let (placed, away) = {
        let t = THREADS.lock().unwrap_or_else(|e| e.into_inner());
        (t.placed, t.pinned_away)
    };
    let n = was.0.max(1) as f64;
    let made = if super::on() {
        format!("{:.2} ms waited for made a frame", made_wait.saturating_sub(was.2) as f64 / 1e6 / n)
    } else {
        "made's waits not counted (VICTORY_LOT4_PROBE=1 counts them)".into()
    };
    tracing::info!(
        "threads placed by CCD from now (VICTORY_CCD_PLACE=alt): {}; since the change before: {} frames, {:.1} ms a frame, \
         the submit thread at work {:.1} ms, recording {:.1} ms on all threads, the longest recording thread {:.1} ms, {made}",
        if on {
            format!("ON, {placed} threads placed ({away} pinned by the title to the other CCD only)")
        } else {
            "off, every thread where the system puts it".into()
        },
        was.0,
        was.1[0] / 1e3 / n,
        was.1[1] / 1e3 / n,
        was.1[2] / 1e3 / n,
        was.1[3] / 1e3 / n,
    );
}

/// For P7's line: the switch and the CCDs.
pub(crate) fn state_text() -> String {
    let placing = match mode() {
        0 => "CCD placement off (VICTORY_CCD_PLACE=1|alt)".to_owned(),
        m => {
            let t = THREADS.lock().unwrap_or_else(|e| e.into_inner());
            format!(
                "CCD placement {} now {} ({} threads placed, {} pinned away)",
                if m == 1 { "=1" } else { "=alt" },
                if PLACED.load(Relaxed) == 1 { "ON" } else { "off" },
                t.placed,
                t.pinned_away
            )
        }
    };
    format!("{placing}; {}", ccds_text())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_read_as_runs() {
        assert_eq!(processors(0xFFF), "0-11");
        assert_eq!(processors(0xFFF000), "12-23");
        assert_eq!(processors(0b1011), "0-1,3");
    }

    #[test]
    fn named_threads_take_their_side() {
        assert_eq!(side_of_name("victory-submit"), Some(Side::Locks));
        assert_eq!(side_of_name("victory-heap-pass-3"), Some(Side::Locks));
        assert_eq!(side_of_name("victory-mirror-0"), Some(Side::Locks));
        assert_eq!(side_of_name("victory-wide-2"), Some(Side::Bulk));
        assert_eq!(side_of_name("victory-relay"), Some(Side::Bulk));
        assert_eq!(side_of_name("victory-texture-back"), Some(Side::Bulk));
        assert_eq!(side_of_name("victory-late-wait"), None);
    }

    /// The machine's L3 caches cover each processor once; every processor
    /// of the process's group is in one of them.
    #[test]
    fn every_processor_has_one_l3() {
        let masks = l3_masks();
        assert!(!masks.is_empty(), "no L3 cache found");
        let mut all = 0usize;
        for m in &masks {
            assert_eq!(all & m, 0, "two L3 caches share a processor: {masks:x?}");
            all |= m;
        }
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get()).min(usize::BITS as usize);
        assert!(all.count_ones() as usize >= cpus.min(64), "{masks:x?} for {cpus} processors");
        let t = topology();
        assert_eq!(t.ccds.len(), masks.len());
        assert!((current() as usize) < masks.len());
    }
}
