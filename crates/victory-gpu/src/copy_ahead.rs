//! `VICTORY_COPY_AHEAD=1`: the host copy of a title buffer is made when the
//! title declares the buffer, on a thread of its own (`victory-ahead`),
//! instead of at the first view a draw names in it (`crate::shadow::find`).
//!
//! In a world run of the source of this port (Series X, RX 7900 XTX) the
//! first frame hung 5.2 s making 3,459 copies (1.1 GB), the translating
//! threads waiting for them; 3,403 of those buffers had been declared 10 s
//! to a minute before, while the loading screen showed. With this the
//! first frame took 3.8 s (3,399 copies made while loading), and one A/B
//! run gave 12.5 against 13.9 fps with the 90th percentile frame longer
//! (50 against 59 ms, "to watch"). Port of Grimkaaah f2586ef4d7 (GPL-3.0-or-later,
//! as this).
//!
//! Off by default. It costs shared memory: the copies of buffers the title
//! declares and never draws from are made too (0.7 GB more in the source's
//! runs). The copy is made by the same code as the lazy one
//! (`shadow::make_copy`: in place if asked for, in the card's memory for
//! big ones), under the lock of the table of buffers, which the title's
//! threads take as well when they declare, release or ask for a buffer.
//! An ask that comes first still makes its own copy; the thread skips
//! buffers that have one or are gone.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, OnceLock};

/// Whether copies are made ahead (`VICTORY_COPY_AHEAD=1`).
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_AHEAD").is_some_and(|v| v == "1"))
}

/// Buffers declared whose copy is to be made: (start, resource).
static QUEUE: Mutex<VecDeque<(u64, usize)>> = Mutex::new(VecDeque::new());
static MOVED: Condvar = Condvar::new();

/// Since the start: copies made ahead, their bytes, the microseconds spent
/// making them, and buffers skipped (copy made by an ask first, or gone).
static MADE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The title declared the buffer `resource` at `start`
/// (`crate::shadow::register`): with the switch, its copy is made soon.
pub(crate) fn declared(resource: usize, start: u64) {
    if !on() {
        return;
    }
    static THREAD: OnceLock<()> = OnceLock::new();
    THREAD.get_or_init(|| {
        match std::thread::Builder::new().name("victory-ahead".into()).spawn(ahead) {
            Ok(_) => tracing::info!(
                "VICTORY_COPY_AHEAD=1: a buffer's host copy is made when the title declares it, on the thread victory-ahead \
                 (more shared memory)"
            ),
            Err(e) => tracing::warn!(error = %e, "no thread to make copies ahead (VICTORY_COPY_AHEAD)"),
        }
    });
    QUEUE.lock().unwrap_or_else(|e| e.into_inner()).push_back((start, resource));
    MOVED.notify_one();
}

/// The thread: the copy of each buffer declared, if it is declared still
/// and has none yet. Says what it made, since the start, each time the
/// queue runs empty after copies.
fn ahead() {
    #[cfg(windows)]
    crate::framelog::keep_on_fast_cores("victory-ahead");
    let mut said = 0;
    loop {
        let next = {
            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
            match queue.pop_front() {
                Some(key) => Some(key),
                None => {
                    let made = MADE[0].load(Relaxed);
                    if made != said {
                        said = made;
                        drop(queue);
                        say();
                        queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                    }
                    if queue.is_empty() {
                        let _queue = MOVED.wait(queue).unwrap_or_else(|e| e.into_inner());
                    }
                    None
                }
            }
        };
        let Some((start, resource)) = next else { continue };
        let started = std::time::Instant::now();
        match crate::shadow::copy_ahead_of(resource, start) {
            Some((len, true)) => {
                MADE[0].fetch_add(1, Relaxed);
                MADE[1].fetch_add(len, Relaxed);
                MADE[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
            }
            _ => {
                MADE[3].fetch_add(1, Relaxed);
            }
        }
    }
}

/// The log line: what the thread made so far.
fn say() {
    tracing::info!(
        "copy ahead: {} host copies of declared buffers made ahead of their first use ({} MB, {:.0} ms making), \
         {} declared buffers skipped (copy made by an ask first, or released) (VICTORY_COPY_AHEAD)",
        MADE[0].load(Relaxed),
        MADE[1].load(Relaxed) >> 20,
        MADE[2].load(Relaxed) as f64 / 1e3,
        MADE[3].load(Relaxed)
    );
}
