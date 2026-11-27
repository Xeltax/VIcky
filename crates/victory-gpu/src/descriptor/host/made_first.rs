//! VIcky (fable-2026-11-27-pf5b.md, lever 2): the pass over buffer views
//! goes first on the heap's slot state (`VICTORY_MADE_PASS_FIRST=1|alt`;
//! off by default).
//!
//! Why: in profile PF5b the pass (`victory-heap-pass`,
//! `VICTORY_HEAP_PASS_BESIDE=1`) takes 102-108 ms from its start to its end
//! and waits 52-55 ms of it for `made` ("120 of 1,168 chunks waited", 0.46
//! ms a wait, about ten of the heap sync's chunks). Every list of the next
//! burst that asked for the heap's buffers waits for the pass at its
//! submission, so the frame is about max(S + 5, P + 42): the submitting
//! thread's work, or the pass's time and 42 ms. With
//! `VICTORY_HEAP_SYNC_NEED` (lever 1) taking the join off the submitting
//! thread, the pass is the wall. The sync lets `made` go between its chunks
//! and takes it back before the pass, asleep in the lock, wakes.
//! `VICTORY_MADE_TURNS` does not help here: by design the chunk makers do
//! not say when they wait for each other (they would hand `made` back and
//! forth slot by slot).
//!
//! With the switch on, two steps:
//! - (a) the pass spins: when its chunk finds `made` held
//!   (`HostHeap::made_counted` on the pass's thread), it says so (a count of
//!   its own beside `Turns::waiting`, [`HostHeap::made_pass_first`]) and
//!   tries again for up to [`PASS_SPIN`] before it sleeps in the lock: a
//!   spinning thread catches the gap between two of the sync's chunks;
//! - (b) the sync hands over: the heap sync's chunk loops
//!   (`HostHeap::mirror_due`, the lean ones of `sync_lean`) look at that count
//!   before each slot, where they look for `VICTORY_MADE_TURNS`' waiters;
//!   when the pass waits, the chunk ends there (its keeps and marks made),
//!   `made` goes, the sync waits up to `TURN_WAIT` for the pass to have taken
//!   it ([`HostHeap::pass_turn_over`]) and then takes it back.
//!   `VICTORY_MADE_PASS_FIRST_HAND=0` leaves (b) out (the pass only spins).
//!
//! The pass's own loops never look at the count and the sync never raises
//! it, so the two cannot hand `made` back and forth. Only the pass's chunks
//! made alone (its one thread) say they wait; the one-slot takers (the
//! title's views and copies, the draws) and the pass's threads holding
//! `made` together (`VICTORY_HEAP_PASS_THREADS`) are as before, and
//! `VICTORY_PARALLEL_MIRROR`'s rounds hand nothing over. No result changes,
//! only who waits for whom: what each slot's making does is the same, and
//! the sync's chunks are cut where the pass comes in, as if smaller there.
//!
//! Alone this lever is expected to lose, as `VICTORY_MADE_TURNS` did: it
//! slows the sync (it now waits for the pass's holds), and without lever 1
//! every sync is joined. With both `alt`, this switch's phases are lever 1's
//! (`crate::alt::Alt::following`); with lever 1 at `1`, its own.
//!
//! A line with the pass's ([`line`]): the pass's wait for `made` and its
//! time from start to end, the submitting thread's waits for the pass and
//! at the sync's join, and the sync's own time, each by this switch's
//! phase; the spins and the hand-overs.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed, Ordering::SeqCst};
use std::time::{Duration, Instant};

use super::{HostHeap, Made, TURN_WAIT, held_alone, made_tried};

/// The switch.
pub(crate) static PASS_FIRST: crate::alt::Alt = crate::alt::Alt::following(
    "VICTORY_MADE_PASS_FIRST",
    "the pass over buffer views goes first on the heap's slot state (PF5b lever 2)",
    &crate::descriptor::at_submit::need::NEED,
);

/// How long the pass spins for `made` before it sleeps in the lock: some
/// five of the sync's chunks.
pub(crate) const PASS_SPIN: Duration = Duration::from_micros(200);

/// Whether the sync's chunks hand `made` to a waiting pass (step (b);
/// `VICTORY_MADE_PASS_FIRST_HAND=0`: the pass only spins, step (a)).
fn hand_set() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_MADE_PASS_FIRST_HAND").is_none_or(|v| v != "0");
        if PASS_FIRST.set() && !on {
            tracing::info!("VICTORY_MADE_PASS_FIRST: the pass only spins, the sync's chunks hand nothing over (VICTORY_MADE_PASS_FIRST_HAND=0)");
        }
        on
    })
}

#[cfg(test)]
thread_local! {
    /// Tests: this thread's switch (on, and with the hand-over), whatever
    /// the environment.
    pub(crate) static TEST_PASS_FIRST: std::cell::Cell<Option<(bool, bool)>> = const { std::cell::Cell::new(None) };
}

/// On the pass's thread, `made` found held: whether it says it waits and
/// spins now.
pub(super) fn spin_now() -> bool {
    #[cfg(test)]
    if let Some((on, _)) = TEST_PASS_FIRST.get() {
        return on;
    }
    PASS_FIRST.on()
}

/// On a chunk maker's thread: whether its chunks hand `made` to a waiting
/// pass now (asked once a call). Never on the pass's own thread.
pub(super) fn hand_now() -> bool {
    #[cfg(test)]
    if let Some((on, hand)) = TEST_PASS_FIRST.get() {
        return on && hand;
    }
    if !PASS_FIRST.set() || crate::draw::heap_pass::parts::counting() {
        return false;
    }
    hand_set() && PASS_FIRST.on()
}

impl HostHeap {
    /// `made` for the pass's chunk, which found it held (`made_counted`):
    /// the pass says it waits ([`HostHeap::pass_wanted`]), tries again for
    /// up to [`PASS_SPIN`], then sleeps in the lock; `began` is when its
    /// wait began.
    pub(super) fn made_pass_first(&self, began: Instant) -> std::sync::RwLockWriteGuard<'_, Made> {
        self.turns.pass_waiting.fetch_add(1, SeqCst);
        let mut got = None;
        while got.is_none() && began.elapsed() < PASS_SPIN {
            for _ in 0..32 {
                std::hint::spin_loop();
            }
            got = made_tried(&self.made);
        }
        let caught = got.is_some();
        let spun = began.elapsed();
        let made = got.unwrap_or_else(|| self.made.write().unwrap_or_else(|e| e.into_inner()));
        // Served before it stops waiting: a sync that handed `made` over
        // sees one or the other.
        self.turns.pass_served.fetch_add(1, SeqCst);
        self.turns.pass_waiting.fetch_sub(1, SeqCst);
        #[cfg(test)]
        PASS_WAITS.fetch_add(1, Relaxed);
        count(K::Waits, 1);
        if caught {
            count(K::Caught, 1);
        }
        count(K::SpunNs, spun.as_nanos() as u64);
        made
    }

    /// Whether a chunk maker holding `made` (with the hand-over `hand`) lets
    /// the waiting pass in now.
    #[inline]
    pub(super) fn pass_wanted(&self, hand: bool) -> bool {
        hand && self.turns.pass_waiting.load(Relaxed) > 0
    }

    /// A chunk of the sync lets `made` go to the waiting pass, waits up to
    /// `TURN_WAIT` for the pass to have it, and takes it back (after the
    /// pass's chunk).
    pub(super) fn pass_turn_over<'a>(&'a self, made: std::sync::RwLockWriteGuard<'a, Made>) -> std::sync::RwLockWriteGuard<'a, Made> {
        let began = Instant::now();
        let served = self.turns.pass_served.load(SeqCst);
        drop(made);
        while self.turns.pass_served.load(SeqCst) == served
            && self.turns.pass_waiting.load(SeqCst) > 0
            && began.elapsed() < TURN_WAIT
        {
            std::hint::spin_loop();
        }
        let made = held_alone(&self.made);
        #[cfg(test)]
        PASS_TURNS.fetch_add(1, Relaxed);
        count(K::Hands, 1);
        count(K::HandNs, began.elapsed().as_nanos() as u64);
        made
    }
}

/// Tests: the pass's waits said, and `made` handed to it, since the start.
#[cfg(test)]
pub(crate) static PASS_WAITS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
pub(crate) static PASS_TURNS: AtomicU64 = AtomicU64::new(0);

/// Since the last line, by [`K`].
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

#[derive(Clone, Copy)]
enum K {
    /// The pass's waits said; of them, those its spin caught; nanoseconds
    /// spun (to the catch, or to the sleep).
    Waits,
    Caught,
    SpunNs,
    /// `made` handed to the pass by the sync's chunks; nanoseconds from the
    /// hand-over to the sync's having it back.
    Hands,
    HandNs,
}

fn count(k: K, n: u64) {
    COUNTS[k as usize].fetch_add(n, Relaxed);
}

/// By phase (ON, off), since the last line, by [`P`].
static PHASES: [[AtomicU64; 9]; 2] = [const { [const { AtomicU64::new(0) }; 9] }; 2];

#[derive(Clone, Copy)]
enum P {
    /// Passes ended, nanoseconds they waited for `made`, microseconds from
    /// their start to their end.
    Passes,
    PassWaitNs,
    PassWallUs,
    /// Microseconds the submitting thread waited for passes still running.
    SubmitWaitUs,
    /// Joins of the heap sync and their microseconds.
    Joins,
    JoinUs,
    /// Syncs ended and their microseconds.
    Syncs,
    SyncUs,
    /// Presents seen at the joins of the phase.
    Frames,
}

/// The current phase's row (0 ON, 1 off); none without the switch.
fn phase() -> Option<usize> {
    PASS_FIRST.set().then(|| usize::from(!PASS_FIRST.on()))
}

fn add(p: P, n: u64) {
    if let Some(row) = phase() {
        PHASES[row][p as usize].fetch_add(n, Relaxed);
    }
}

/// A pass over buffer views ended (`crate::draw::heap_pass::parts::add`):
/// its threads waited `waited_ns` for `made`; `wall` from start to end.
pub(crate) fn pass_ended(waited_ns: u64, wall: Duration) {
    if let Some(row) = phase() {
        let row = &PHASES[row];
        row[P::Passes as usize].fetch_add(1, Relaxed);
        row[P::PassWaitNs as usize].fetch_add(waited_ns, Relaxed);
        row[P::PassWallUs as usize].fetch_add(wall.as_micros() as u64, Relaxed);
    }
}

/// The submitting thread waited `waited` for a pass still running
/// (`crate::draw::heap_pass::Pending::wait`).
pub(crate) fn pass_waited(waited: Duration) {
    add(P::SubmitWaitUs, waited.as_micros() as u64);
}

/// Presents seen at the last join (for the frames of each phase).
static FRAME_SEEN: AtomicU64 = AtomicU64::new(u64::MAX);

/// A submission joined its heap sync after `waited`
/// (`crate::descriptor::at_submit::Started::join`).
pub(crate) fn joined(waited: Duration) {
    let Some(row) = phase() else {
        return;
    };
    let row = &PHASES[row];
    row[P::Joins as usize].fetch_add(1, Relaxed);
    row[P::JoinUs as usize].fetch_add(waited.as_micros() as u64, Relaxed);
    let now = crate::queue::presented();
    let last = FRAME_SEEN.swap(now, Relaxed);
    if last != u64::MAX {
        row[P::Frames as usize].fetch_add(now.saturating_sub(last), Relaxed);
    }
}

/// A heap sync ended after `us` microseconds (`crate::descriptor::sync_held`).
pub(crate) fn synced(us: u64) {
    if let Some(row) = phase() {
        PHASES[row][P::Syncs as usize].fetch_add(1, Relaxed);
        PHASES[row][P::SyncUs as usize].fetch_add(us, Relaxed);
    }
}

/// With the frame line (`crate::draw::heap_pass::line`), over `frames`:
/// nothing without the switch.
pub(crate) fn line(frames: f64) {
    if !PASS_FIRST.set() {
        return;
    }
    let counts: [u64; 5] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed));
    let phases: [[u64; 9]; 2] = std::array::from_fn(|r| std::array::from_fn(|i| PHASES[r][i].swap(0, Relaxed)));
    tracing::info!("{}", said(&counts, &phases, frames, PASS_FIRST.said(), hand_set()));
}

fn said(c: &[u64; 5], p: &[[u64; 9]; 2], frames: f64, mode: &str, hand: bool) -> String {
    let frames = frames.max(1.0);
    let per = |k: K| c[k as usize] as f64 / frames;
    let ms_ns = |k: K| c[k as usize] as f64 / 1e6 / frames;
    let by = |what: &dyn Fn(&[u64; 9]) -> String| format!("ON {}, off {}", what(&p[0]), what(&p[1]));
    let pass_wait = by(&|r| {
        let n = r[P::Passes as usize];
        if n == 0 {
            "no pass".to_owned()
        } else {
            format!("{:.1} ms a pass over {n} passes", r[P::PassWaitNs as usize] as f64 / 1e6 / n as f64)
        }
    });
    let pass_wall = by(&|r| {
        let n = r[P::Passes as usize];
        if n == 0 { "no pass".to_owned() } else { format!("{:.1} ms", r[P::PassWallUs as usize] as f64 / 1000.0 / n as f64) }
    });
    // Per frame of the phase (presents seen at its joins).
    let per_frame = |us: P| {
        by(&move |r: &[u64; 9]| {
            let f = r[P::Frames as usize];
            if f == 0 { "no frame".to_owned() } else { format!("{:.2} ms a frame over {f} frames", r[us as usize] as f64 / 1000.0 / f as f64) }
        })
    };
    format!(
        "the pass over buffer views goes first on the heap's slot state (VICTORY_MADE_PASS_FIRST {mode}, the sync's chunks \
         hand it over: {}): the pass's wait for the lock of the heap's slot state: {pass_wait}; from its start to its end: \
         {pass_wall}; the submitting thread waited for the pass: {}; at the heap sync's join: {}; the heap sync's own: {}; \
         per frame {:.1} waits of the pass said, {:.1} caught by its spin ({:.2} ms spun), {:.1} hand-overs by the sync's \
         chunks ({:.2} ms of the sync's from each hand-over to its having the lock back)",
        if hand { "yes" } else { "no" },
        per_frame(P::SubmitWaitUs),
        per_frame(P::JoinUs),
        per_frame(P::SyncUs),
        per(K::Waits),
        per(K::Caught),
        ms_ns(K::SpunNs),
        per(K::Hands),
        ms_ns(K::HandNs),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line: by phase, per pass and per frame of the phase.
    #[test]
    fn the_line_says_each_phase() {
        let mut c = [0u64; 5];
        c[K::Waits as usize] = 240;
        c[K::Caught as usize] = 200;
        c[K::SpunNs as usize] = 4_000_000;
        c[K::Hands as usize] = 100;
        c[K::HandNs as usize] = 6_000_000;
        let mut p = [[0u64; 9]; 2];
        p[0][P::Passes as usize] = 2;
        p[0][P::PassWaitNs as usize] = 10_000_000;
        p[0][P::PassWallUs as usize] = 120_000;
        p[0][P::Frames as usize] = 2;
        p[0][P::JoinUs as usize] = 6_000;
        p[1][P::Passes as usize] = 1;
        p[1][P::PassWaitNs as usize] = 52_000_000;
        p[1][P::PassWallUs as usize] = 104_000;
        let text = said(&c, &p, 2.0, "alt, ON", true);
        assert!(text.contains("(VICTORY_MADE_PASS_FIRST alt, ON, the sync's chunks hand it over: yes)"), "{text}");
        assert!(text.contains("slot state: ON 5.0 ms a pass over 2 passes, off 52.0 ms a pass over 1 passes"), "{text}");
        assert!(text.contains("from its start to its end: ON 60.0 ms, off 104.0 ms"), "{text}");
        assert!(text.contains("at the heap sync's join: ON 3.00 ms a frame over 2 frames, off no frame"), "{text}");
        assert!(text.contains("per frame 120.0 waits of the pass said, 100.0 caught by its spin (2.00 ms spun), 50.0 hand-overs"), "{text}");
        assert!(text.ends_with("(3.00 ms of the sync's from each hand-over to its having the lock back)"), "{text}");
    }
}
