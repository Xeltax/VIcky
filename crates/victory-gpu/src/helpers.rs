//! VIcky: the submitting thread's helpers (`VICTORY_HELPER_PRIORITY`,
//! `VICTORY_WIDE_THREADS`; both off by default).
//!
//! `VICTORY_HELPER_PRIORITY=1|alt`, port of trustno1 80c70b71 (PR #22):
//! the threads the submitting thread waits for run above normal priority,
//! ahead of the title's threads and of the threads that translate lists.
//! On a six-core processor (i5-12400F, RTX 3060) every core was busy, and
//! the helpers, at normal priority, were often not running while the
//! submitting thread waited: off / on in one run, 8.6 / 9.6 fps, the
//! refreshes 23.1 / 17.6 ms a frame, the wait for the heaps' records 6.2 /
//! 2.5 ms. There the submitting thread already ran above normal; here it
//! did not, so the switch raises it too (`victory-submit`, with
//! `VICTORY_ASYNC_SUBMIT=1`) with its helpers: the wide threads
//! (`victory-wide-*`), the heap sync's lane (`victory-heap-sync`), the
//! parallel mirroring's crew (`victory-mirror-*`), the threads of the pass
//! over the heap's buffer views (`victory-heap-pass-*`), the relay
//! (`victory-relay`) and the copy-backs' writer (`victory-texture-back`).
//! Each looks at the switch before each piece of work and asks the system
//! only when it changed, so `alt` works (twenty seconds off, twenty on).
//! Its companion `VICTORY_DSTORAGE_PRIORITY` (victory-hle) lowers the
//! threads that unpack streamed assets.
//!
//! `VICTORY_WIDE_THREADS`, port of foxhound 9f8838b1: the threads kept for
//! wide work ([`crate::shadow`]'s refreshes and snapshots on several
//! threads, the heap sync's spread walk). Ours keep a quarter of the
//! logical processors, 2 to 6 (6 on 24); foxhound's three eighths, 2 to 8
//! (8 on 24). `1`: foxhound's rule; `<n>` of 2 to 32: that many; `alt`:
//! foxhound's rule in the phases with it, ours in the others (the threads
//! past ours wait, taking nothing, in those). The parallel mirroring's crew
//! keeps its size from ours except with `1` or `<n>`.

use std::sync::OnceLock;

use crate::alt::Alt;

pub(crate) static HELPER_PRIORITY: Alt = Alt::new(
    "VICTORY_HELPER_PRIORITY",
    "the submitting thread and the threads it waits for run above normal priority (port of trustno1 80c70b71)",
);

/// A thread's priority under [`HELPER_PRIORITY`]: [`Priority::keep`] before
/// each piece of work asks the system only when the switch's phase changed.
pub(crate) struct Priority {
    applied: Option<bool>,
}

impl Priority {
    pub(crate) fn new() -> Self {
        let mut priority = Self { applied: None };
        priority.keep();
        priority
    }

    pub(crate) fn keep(&mut self) {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_NORMAL,
        };
        if !HELPER_PRIORITY.set() {
            return;
        }
        let on = HELPER_PRIORITY.on();
        if self.applied == Some(on) || self.applied.is_none() && !on {
            self.applied = Some(on);
            return;
        }
        let priority = if on { THREAD_PRIORITY_ABOVE_NORMAL } else { THREAD_PRIORITY_NORMAL };
        // SAFETY: our own thread.
        unsafe { SetThreadPriority(GetCurrentThread(), priority) };
        self.applied = Some(on);
    }
}

/// [`Priority::new`] for a thread that lives for one piece of work.
pub(crate) fn raise() {
    let _ = Priority::new();
}

/// The wide threads' rule (`VICTORY_WIDE_THREADS`, as `1`, `alt` or `<n>`).
static WIDE: Alt = Alt::new(
    "VICTORY_WIDE_THREADS",
    "three eighths of the logical processors, 2 to 8, are kept for wide work (port of foxhound 9f8838b1)",
);

/// Ours: a quarter of the logical processors, 2 to 6.
fn ours() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(2, 6)
}

/// Foxhound's: three eighths of the logical processors, 2 to 8.
fn foxhound() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get() * 3 / 8).clamp(2, 8)
}

/// `VICTORY_WIDE_THREADS=<n>` (2 to 32), if that is what it says.
fn asked() -> Option<usize> {
    static ASKED: OnceLock<Option<usize>> = OnceLock::new();
    *ASKED.get_or_init(|| {
        let n = std::env::var("VICTORY_WIDE_THREADS").ok()?.trim().parse::<usize>().ok().filter(|&n| n >= 2)?.min(32);
        tracing::info!(n, "threads kept for wide work (VICTORY_WIDE_THREADS={n})");
        Some(n)
    })
}

/// The threads to start for wide work: the most any phase uses.
pub(crate) fn wide_kept() -> usize {
    match (asked(), WIDE.mode()) {
        (Some(n), _) => n,
        (None, 0) => ours(),
        (None, _) => foxhound().max(ours()),
    }
}

/// The threads the crew and the like size themselves by: [`wide_kept`],
/// but ours with `alt`.
pub(crate) fn wide_sized() -> usize {
    match (asked(), WIDE.mode()) {
        (None, 2) => ours(),
        _ => wide_kept(),
    }
}

/// Whether wide thread `i` takes work now: all with `1` or `<n>`; with
/// `alt`, the first of ours only in the phases without the switch.
pub(crate) fn wide_active(i: usize) -> bool {
    if asked().is_some() || WIDE.mode() != 2 {
        return true;
    }
    i < if WIDE.on() { foxhound() } else { ours() }
}
