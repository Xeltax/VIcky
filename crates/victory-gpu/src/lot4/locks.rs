//! P7 (`VICTORY_LOT4_PROBE=1`): the hot locks' acquisitions, the ones that
//! had to wait and how long, and the ones taken on the other CCD than the
//! lock's last taker (a lock word that moves between the CCDs costs far
//! more than within one). One line a reading, per frame.
//!
//! Each lock site goes through [`mutex`], [`write`] or [`read`] (or counts a
//! `try_*` with [`tried`]): with the probes off that is the plain lock and
//! one load of a flag. On, a lock is tried first; only one that is held
//! elsewhere is timed (so an acquisition that does not wait costs no clock
//! reading). The counts are kept per thread (`crate::perthread`); the CCD
//! of a lock's last taker is one word per kind of lock, written only when
//! it changes (all of a kind's locks share it: `made` of every heap, say).

use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

use crate::perthread::Counters;

/// The locks counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lock {
    /// `HostHeap::made` held alone (its write lock).
    Made,
    /// `HostHeap::made` held together (its read lock: the pass's threads).
    MadeShared,
    /// The pieces of `made` the threads that mirror together take
    /// (`VICTORY_PARALLEL_MIRROR`).
    MadePiece,
    /// The table of views (`crate::descriptor`'s `VIEWS`): read, written.
    Views,
    ViewsWrite,
    /// `crate::shadow`'s `BUFFERS`: the title buffers and their copies,
    /// held alone (every lookup and every change before
    /// `VICTORY_BUFFERS_STRIPES`; with it set, the lookups while it is off
    /// and the rare readers that walk the whole table).
    Buffers,
    /// `HostHeap::placed`: the spans of placed texture views.
    Placed,
    /// `HostHeap::kept`: the resources slots hold.
    Kept,
    /// `HostHeap::buffers`: the title slots holding buffer views.
    HeapBuffers,
    /// `HostHeap::changed`.
    Changed,
    /// `HostHeap::textures`: the queue of textures picked by index.
    Textures,
    /// `HostHeap::next`: the dynamic slots.
    Next,
    /// `crate::watch`'s guards.
    Guards,
    /// `Made::cache`: the made buffer views by record.
    Cache,
    /// VIcky (`VICTORY_BUFFERS_STRIPES`, `crate::stripes`): `BUFFERS` read
    /// on a stripe, and written (its whole lock and, with the switch set,
    /// every stripe).
    BuffersRead,
    BuffersWrite,
    /// `crate::shadow`'s `LOOSE`, the windows of title memory outside
    /// buffers (each lookup `BUFFERS` does not answer): held alone, read on
    /// a stripe, written (as `BUFFERS`).
    Loose,
    LooseRead,
    LooseWrite,
    /// VIcky (`VICTORY_MADE_TURNS`): `made` handed by a thread that makes a
    /// chunk of slots (the heap sync, the pass over buffer views) to a
    /// thread that waits for it, between two slots; the ms are the chunk
    /// maker's, from letting it go to having it again.
    MadeTurn,
}

const LOCKS: usize = 20;
const NAMES: [&str; LOCKS] = [
    "made (alone)",
    "made (shared, the pass's threads)",
    "made's pieces",
    "the table of views (read)",
    "the table of views (written)",
    "BUFFERS",
    "placed",
    "kept",
    "the heap's buffer slots",
    "changed",
    "textures",
    "next (dynamic slots)",
    "guards",
    "the made views' cache",
    "BUFFERS (read, striped)",
    "BUFFERS (written)",
    "windows (LOOSE)",
    "windows (LOOSE, read, striped)",
    "windows (LOOSE, written)",
    "made handed to a waiting thread between a chunk's slots",
];

/// Per lock: taken, of them after a wait, nanoseconds waited, taken on
/// another CCD than its last taker, tries that found it held.
const PER: usize = 5;
static COUNTS: Counters<{ LOCKS * PER }> = Counters::new();
const TAKEN: usize = 0;
const WAITED: usize = 1;
const WAIT_NS: usize = 2;
const CROSS: usize = 3;
const REFUSED: usize = 4;

/// Per lock: the CCD of its last taker, plus one (0: none yet).
static LAST: [AtomicU8; LOCKS] = [const { AtomicU8::new(0) }; LOCKS];

/// `lock` was taken, after `waited_ns` (0: at once).
fn taken(lock: Lock, waited_ns: Option<u64>) {
    let at = lock as usize * PER;
    COUNTS.add(at + TAKEN, 1);
    if let Some(ns) = waited_ns {
        COUNTS.add(at + WAITED, 1);
        COUNTS.add(at + WAIT_NS, ns);
    }
    let ccd = super::ccd::current().wrapping_add(1);
    let last = &LAST[lock as usize];
    let was = last.load(Relaxed);
    if was != ccd {
        if was != 0 {
            COUNTS.add(at + CROSS, 1);
        }
        last.store(ccd, Relaxed);
    }
}

/// A `try_*` of `lock` at a site that does not wait: counted as taken, or
/// as refused.
#[inline]
pub(crate) fn tried(lock: Lock, got: bool) {
    if !super::on() {
        return;
    }
    if got {
        taken(lock, None);
    } else {
        COUNTS.add(lock as usize * PER + REFUSED, 1);
    }
}

/// `lock` was taken after a wait of `ns` nanoseconds, at a site that
/// times its own waits.
#[inline]
pub(crate) fn waited(lock: Lock, ns: u64) {
    if super::on() {
        taken(lock, Some(ns));
    }
}

/// `m` locked (poisoned or not), counted as `lock`.
#[inline]
pub(crate) fn mutex<T>(lock: Lock, m: &Mutex<T>) -> MutexGuard<'_, T> {
    if !super::on() {
        return m.lock().unwrap_or_else(|e| e.into_inner());
    }
    match m.try_lock() {
        Ok(g) => {
            taken(lock, None);
            g
        }
        Err(TryLockError::Poisoned(e)) => {
            taken(lock, None);
            e.into_inner()
        }
        Err(TryLockError::WouldBlock) => {
            let waiting = std::time::Instant::now();
            let g = m.lock().unwrap_or_else(|e| e.into_inner());
            taken(lock, Some(waiting.elapsed().as_nanos() as u64));
            g
        }
    }
}

/// `m`'s write lock (poisoned or not), counted as `lock`.
#[inline]
pub(crate) fn write<T>(lock: Lock, m: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    if !super::on() {
        return m.write().unwrap_or_else(|e| e.into_inner());
    }
    match m.try_write() {
        Ok(g) => {
            taken(lock, None);
            g
        }
        Err(TryLockError::Poisoned(e)) => {
            taken(lock, None);
            e.into_inner()
        }
        Err(TryLockError::WouldBlock) => {
            let waiting = std::time::Instant::now();
            let g = m.write().unwrap_or_else(|e| e.into_inner());
            taken(lock, Some(waiting.elapsed().as_nanos() as u64));
            g
        }
    }
}

/// `m`'s read lock (poisoned or not), counted as `lock`.
#[inline]
pub(crate) fn read<T>(lock: Lock, m: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    if !super::on() {
        return m.read().unwrap_or_else(|e| e.into_inner());
    }
    match m.try_read() {
        Ok(g) => {
            taken(lock, None);
            g
        }
        Err(TryLockError::Poisoned(e)) => {
            taken(lock, None);
            e.into_inner()
        }
        Err(TryLockError::WouldBlock) => {
            let waiting = std::time::Instant::now();
            let g = m.read().unwrap_or_else(|e| e.into_inner());
            taken(lock, Some(waiting.elapsed().as_nanos() as u64));
            g
        }
    }
}

/// Who reads the tables of host copies (`crate::stripes`): per table
/// (`BUFFERS`, the windows), the reads by the title's threads, the heap
/// sync (and the parallel mirror's crew), the pass over buffer views (and
/// its threads), the submitting thread, and our other threads.
const ROLES: usize = 5;
const ROLE_NAMES: [&str; ROLES] =
    ["the title's threads", "the heap sync", "the pass over buffer views", "the submitting thread", "other threads of ours"];
static READ_BY: Counters<{ 2 * ROLES }> = Counters::new();

thread_local! {
    /// This thread's role for [`read_by`], plus one (0: not looked at yet).
    static ROLE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// The role of a thread named `name` ([`ROLE_NAMES`]).
fn role_of(name: Option<&str>) -> usize {
    match name {
        Some(n) if n.starts_with("victory-heap-sync") || n.starts_with("victory-mirror-") => 1,
        Some(n) if n.starts_with("victory-heap-pass") => 2,
        Some(n) if n.starts_with("victory-submit") => 3,
        Some(n) if n.starts_with("victory") => 4,
        _ => 0,
    }
}

/// A read of the table counted as `alone` (`Lock::Buffers` or
/// `Lock::Loose`), whatever the lock it takes, by this thread's role.
#[inline]
pub(crate) fn read_by(alone: Lock) {
    if !super::on() {
        return;
    }
    let table = usize::from(alone == Lock::Loose);
    let role = ROLE.with(|r| {
        if r.get() == 0 {
            r.set(role_of(std::thread::current().name()) as u8 + 1);
        }
        usize::from(r.get() - 1)
    });
    READ_BY.add(table * ROLES + role, 1);
}

/// Nanoseconds waited for `made` (alone, shared, its pieces) since the
/// start, for [`super::ccd`]'s phase lines.
pub(crate) fn made_wait_ns() -> u64 {
    [Lock::Made, Lock::MadeShared, Lock::MadePiece].iter().map(|&l| COUNTS.total(l as usize * PER + WAIT_NS)).sum()
}

/// The reading's line: each lock taken at all, per frame over `frames`.
pub(crate) fn line(frames: f64) {
    let c = COUNTS.take();
    let mut parts = Vec::new();
    for (i, name) in NAMES.iter().enumerate() {
        let at = i * PER;
        let taken = c[at + TAKEN];
        if taken == 0 && c[at + REFUSED] == 0 {
            continue;
        }
        let refused = if c[at + REFUSED] > 0 { format!(", {:.0} tries found it held", c[at + REFUSED] as f64 / frames) } else { String::new() };
        parts.push(format!(
            "{name} {:.0} taken ({:.0} after a wait, {:.2} ms waited; {:.1} % on the other CCD than the last taker){refused}",
            taken as f64 / frames,
            c[at + WAITED] as f64 / frames,
            c[at + WAIT_NS] as f64 / 1e6 / frames,
            super::pct(c[at + CROSS], taken),
        ));
    }
    let (own, most, shared) = crate::perthread::threads();
    tracing::info!(
        "lot4 probe P7: locks per frame: {}; per-thread counters: {own} threads with a block of their own now, {most} at \
         most, {shared} found none free; {}",
        if parts.is_empty() { "none taken".into() } else { parts.join("; ") },
        super::ccd::state_text(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Taken, waited and refused land where the line reads them.
    #[test]
    fn acquisitions_and_waits_are_counted() {
        let m = Mutex::new(0u32);
        let before = COUNTS.totals();
        taken(Lock::Next, None);
        taken(Lock::Next, Some(1500));
        COUNTS.add(Lock::Next as usize * PER + REFUSED, 1);
        let after = COUNTS.totals();
        let at = Lock::Next as usize * PER;
        let d = |i: usize| after[at + i].wrapping_sub(before[at + i]);
        assert_eq!((d(TAKEN), d(WAITED), d(WAIT_NS), d(REFUSED)), (2, 1, 1500, 1));
        // Off (no VICTORY_LOT4_PROBE in tests): the plain lock.
        *mutex(Lock::Next, &m) += 1;
        assert_eq!(*m.lock().unwrap(), 1);
        let rw = RwLock::new(5u32);
        *write(Lock::Made, &rw) += 1;
        assert_eq!(*read(Lock::Made, &rw), 6);
    }

    /// The readers of the tables of copies are told apart by thread name.
    #[test]
    fn readers_by_thread_name() {
        assert_eq!(role_of(Some("victory-heap-sync")), 1);
        assert_eq!(role_of(Some("victory-mirror-3")), 1);
        assert_eq!(role_of(Some("victory-heap-pass-2")), 2);
        assert_eq!(role_of(Some("victory-submit")), 3);
        assert_eq!(role_of(Some("victory-relay")), 4);
        assert_eq!(role_of(Some("worker")), 0);
        assert_eq!(role_of(None), 0);
    }

    /// What a counted lock site costs with the probes off (tests run
    /// without `VICTORY_LOT4_PROBE`), against the plain lock, one thread:
    /// `cargo test -p victory-gpu --profile diag --lib lot4::locks::tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "a benchmark: run alone, with --nocapture"]
    fn bench_off_against_plain() {
        assert!(!super::super::on(), "run without VICTORY_LOT4_PROBE");
        const N: u32 = 20_000_000;
        let m = Mutex::new(0u64);
        let time = |f: &dyn Fn()| {
            let started = std::time::Instant::now();
            for _ in 0..N {
                f();
            }
            started.elapsed().as_nanos() as f64 / f64::from(N)
        };
        for _ in 0..2 {
            let plain = time(&|| *m.lock().unwrap_or_else(|e| e.into_inner()) += 1);
            let counted = time(&|| *mutex(Lock::Next, &m) += 1);
            println!("lot4 locks bench (probes off): plain lock {plain:.2} ns, through the counting helper {counted:.2} ns");
        }
        assert_eq!(*m.lock().unwrap(), 4 * u64::from(N));
    }
}
