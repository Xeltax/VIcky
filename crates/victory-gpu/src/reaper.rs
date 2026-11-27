//! `VICTORY_REAPER=1`: what the host is done with is let go of on a thread
//! of its own. Host lists of landed flights are reset and kept for reuse
//! ([`crate::list::recycle`]) and retired objects whose submissions have run
//! are dropped. Both are the driver's work (command allocators reset,
//! allocations destroyed, addresses freed). Without the switch they run on
//! the thread that lands or retires them: the submitting thread
//! (`victory-submit`) at the end of each submission sent ahead, the
//! settling thread (`victory-late-wait`), the relay. In a timeline of the
//! world (Series X, RX 7900 XTX) host lists were reset 250 times a frame,
//! 20 ms of threads' time. Only what is already due is sent over: the lag
//! of [`crate::host`]'s `bury` and what `land` hands over are unchanged.
//! Port of Grein a2cf68bb1e via Grimkaaah f2586ef4d7 (GPL-3.0-or-later, as
//! this).
//!
//! Off by default. The frame-rate line counts, per thread, the lists reset
//! and the milliseconds spent resetting them and dropping objects
//! ("reaper:"), with the switch on or off.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::Sender;

/// Threads the counters tell apart: the submitting thread, the thread that
/// settles flights, the relay, the reaper, any other.
const SUBMIT: usize = 0;
const LATE: usize = 1;
const RELAY: usize = 2;
const REAPER: usize = 3;
const OTHER: usize = 4;
const NAMES: [&str; 5] = ["submit", "late-wait", "relay", "reaper", "other"];

/// Per thread, since the last frame line: lists reset, microseconds spent
/// resetting them, objects dropped (batches), microseconds spent dropping.
static LISTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static LIST_US: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static DROPS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static DROP_US: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Whether the work goes to the reaper thread.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REAPER").is_some_and(|v| v == "1"))
}

/// Which of the threads above this is.
fn class() -> usize {
    thread_local! {
        static CLASS: usize = match std::thread::current().name() {
            Some("victory-submit") => SUBMIT,
            Some("victory-late-wait") => LATE,
            Some("victory-relay") => RELAY,
            Some("victory-reaper") => REAPER,
            _ => OTHER,
        };
    }
    CLASS.with(|c| *c)
}

/// Work to do on the reaper thread.
struct Job(Box<dyn FnOnce()>);

// SAFETY: what is sent has run (lists) or is retired (objects): D3D12
// objects may be released and reset on any thread, as the line, which held
// them, and the host's `Retired`; nothing else names them.
unsafe impl Send for Job {}

/// The reaper's queue, `None` when off or when no thread could be made.
fn reaper() -> Option<&'static Sender<Job>> {
    if !on() {
        return None;
    }
    static REAPER: OnceLock<Option<Sender<Job>>> = OnceLock::new();
    REAPER
        .get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            let made = std::thread::Builder::new().name("victory-reaper".into()).spawn(move || {
                for Job(work) in rx {
                    work();
                }
            });
            match made {
                Ok(_) => {
                    tracing::info!("VICTORY_REAPER=1: landed host lists are reset and retired objects dropped on the thread victory-reaper");
                    Some(tx)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no reaper thread: host lists are reset where they land");
                    None
                }
            }
        })
        .as_ref()
}

/// Runs `work`, or has the reaper run it; `work` is counted on the thread
/// that runs it.
fn run(work: impl FnOnce() + 'static) {
    match reaper() {
        Some(tx) => {
            if let Err(std::sync::mpsc::SendError(Job(work))) = tx.send(Job(Box::new(work))) {
                work();
            }
        }
        None => work(),
    }
}

/// [`crate::list::recycle`] of the lists of a flight that has run, on the
/// reaper if there is one.
#[cfg(windows)]
pub(crate) fn recycle(lists: Vec<crate::list::Detached>, ran: Vec<usize>) {
    if lists.is_empty() {
        return;
    }
    run(move || {
        let started = std::time::Instant::now();
        let n = lists.len() as u64;
        crate::list::recycle(lists, &ran);
        let c = class();
        LISTS[c].fetch_add(n, Relaxed);
        LIST_US[c].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    });
}

/// Drops `objects`, retired and due, on the reaper if there is one.
pub(crate) fn dispose<T: 'static>(objects: T) {
    run(move || {
        let started = std::time::Instant::now();
        drop(objects);
        let c = class();
        DROPS[c].fetch_add(1, Relaxed);
        DROP_US[c].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    });
}

/// With the frame line: per thread, over `frames`, the lists reset and the
/// milliseconds a frame spent resetting them and dropping objects.
pub(crate) fn summary(frames: f64) -> String {
    let frames = frames.max(1.0);
    let mut out = format!("reaper {}:", if on() { "on" } else { "off" });
    for (c, name) in NAMES.iter().enumerate() {
        let lists = LISTS[c].swap(0, Relaxed);
        let list_us = LIST_US[c].swap(0, Relaxed);
        let drops = DROPS[c].swap(0, Relaxed);
        let drop_us = DROP_US[c].swap(0, Relaxed);
        if lists == 0 && drops == 0 {
            continue;
        }
        out += &format!(
            " {name} {:.1} lists {:.2} ms resetting + {:.2} ms dropping objects per frame;",
            lists as f64 / frames,
            list_us as f64 / 1e3 / frames,
            drop_us as f64 / 1e3 / frames
        );
    }
    out
}
