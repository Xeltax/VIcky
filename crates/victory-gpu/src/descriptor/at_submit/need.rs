//! VIcky (fable-2026-11-27-pf5b.md, lever 1): a submission joins the heap
//! sync only when its lists need what that sync mirrors
//! (`VICTORY_HEAP_SYNC_NEED=count|1|alt|check`; off by default). Used only
//! with `VICTORY_HEAP_SYNC_BESIDE=1`; with `VICTORY_HEAP_SYNC_AT_SUBMIT`
//! set too, this one is used and no sync is started ahead (`super::used`).
//!
//! Why: in profile PF5b (prologue, RT on) the submitting thread is one of
//! the frame's two walls, 137 ms in `Host::submit`, and 31-33 ms of it is
//! the join of the heap sync beside it. The title hands a frame's lists
//! over in a burst (6.4 ms, 17 submit calls), then records the next frame
//! while the submitting thread works through the burst. By place in the
//! burst the syncs mirror 226, 562, 31,700, 17,250, 12,600, 24,560 and
//! 4,200 slots: from the 3rd on, slots written after the burst, the next
//! frame's. This frame's lists do not read them, yet each submission waits
//! for its sync's end (13.6 + 5.7 + 7.9 + 3.9 ms by place).
//!
//! **The rule.** A submission needs a sync that asked for the heaps' pages
//! after its lists' stamps ([`super::need`]: the title's hand-over of each
//! list, raised by the waits its queue passed; for a list of ours, the
//! title's latest call). A sync that has ended and asked after that stamp
//! mirrored every record written before it: syncs follow one another on
//! one lane, each takes every heap page written since the one before took
//! them (`crate::watch`'s journal keeps a page found by any asker for the
//! heap's next take), and it ends only once it has mirrored them, drained
//! the slots `VICTORY_MIRROR_TRY` left to it, and its writes for the card
//! are out. [`State::covered`] is the highest asking stamp of a sync that
//! has ended, raised on the lane's thread at the sync's end after a fence
//! of its own (`beside::Lane::serve` fences only after the job, and a
//! submission that relies on the stamp must find those writes out). A
//! submission whose need is under it reads nothing the sync still running
//! mirrors: that sync took pages written after the covering sync asked,
//! after the hand-over. A submission that settled flights before its sync's
//! place always joins whole, as `VICTORY_HEAP_SYNC_AT_SUBMIT` rules
//! (landings come before the sync without either switch).
//!
//! **What it does**, at the sync's place in `Host::submit` ([`start`]),
//! with what the one sync left running without a join ([`State::running`])
//! became:
//! 1. none left, or it has ended: a sync of its own starts, as without the
//!    switch (the submitting thread waits for its asking, as before);
//! 2. it still runs and an ended sync covers the lists (with skipping on):
//!    it is left running and no sync is started; the join waits for
//!    nothing. Starting one would mean waiting at once for the one left
//!    running to end (one lane, one sync at a time), the wait this lever
//!    takes away;
//! 3. it still runs and asked after the lists' stamps (with skipping on):
//!    it is this submission's sync, joined at the join for what is left of
//!    it (or left running again if covered then);
//! 4. else the submission waits for its end, then starts its own.
//!
//! At the join ([`join`]): if an ended sync covers the lists at that
//! moment, the sync is not waited for but left running; else it is joined
//! as before. `make_waiting` after the join place stays (`crate::host`),
//! so the slots left waiting for a texture's host copy are made as before.
//! Before a sweep ([`before_sweep`]) the one left running ends first: the
//! sweep never meets a sync.
//!
//! The order rules of `super::super::beside`, and how they hold with a sync
//! left running:
//! - The journal (fxceless/victory-fork#410): a sync left running asked at
//!   its own submission's start, before that submission's asking, and asks
//!   nothing after (`sync_held` takes every heap's pages before it says it
//!   asked); a new sync starts only at a submission's sync place, before
//!   that submission's asking, once the one left running has ended
//!   (rules 1 and 4) or not at all (rule 2).
//! - The sweep: as above.
//! - The submission counted as started: a sync left running spans the next
//!   submissions' starts. A host resource a slot lets go of then waits for
//!   the end of the submissions started at that moment
//!   (`crate::host::retire`), and a heap map pool slot given back is handed
//!   out after them (`crate::host::submissions`): later, never earlier.
//! - What the card may meet: as at every later submission of a burst
//!   today, the lists sent may be on the card while a sync mirrors the next
//!   frame's slots. New: also between a submission's refreshes and its
//!   `ExecuteCommandLists`. Such a record was written after the lists that
//!   read its slot were handed over: on the console a race the title does
//!   not run.
//!
//! **Modes.**
//! - `count`: no change of behaviour (every join waits). At each join,
//!   whether an ended sync covered it, and the ms that covered join then
//!   waited: what `1` would save at most. By place in the burst on the
//!   `heap sync by place in the burst` line (diagnostics builds).
//! - `1`: covered joins skipped.
//! - `alt`: twenty seconds as before, twenty with the skips, in turn
//!   (`crate::alt`, the lines `xcl-notes/alt-fps.sh` reads); the off phases
//!   count covered joins as `count` does. A sync left running at the
//!   change is met by rule 4.
//! - `check`: as `1`, and at each join it would skip, the pages of the
//!   heaps written before the lists' stamp must all have been taken by a
//!   sync that has ended ([`super::ledger`]): an independent look at the
//!   kernel's dirty bits and the journal at the stamp, against the pages
//!   each sync took. A violation is counted and logged, and that join waits
//!   whole: no list runs with a page it may read not yet mirrored.
//!
//! The heap sync's line gets a part ([`part`]): joins, covered, skipped;
//! the ms of covered joins that still waited (`count`, the off phases) and
//! of the others; the rules; the submitting thread at the join by phase;
//! with `check`, its counts.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use super::super::beside::{self, Beside, Lane};
use super::{Early, Started, adoptable, ledger};

/// The switch (`1` and `alt`; `count` and `check` are read by [`mode`]).
pub(crate) static NEED: crate::alt::Alt = crate::alt::Alt::new(
    "VICTORY_HEAP_SYNC_NEED",
    "a submission joins the heap sync only when its lists need what that sync mirrors (PF5b lever 1)",
);

/// What the switch says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    /// Every join waits; covered joins counted.
    Count,
    /// Covered joins skipped.
    On,
    /// Twenty seconds each way.
    Alt,
    /// As `On`, each skip checked first ([`super::ledger`]).
    Check,
}

/// The switch's mode, read once.
pub(crate) fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_HEAP_SYNC_NEED").as_deref().map(str::trim) {
            Ok("count") => Mode::Count,
            Ok("check") => Mode::Check,
            _ => match NEED.mode() {
                1 => Mode::On,
                2 => Mode::Alt,
                _ => Mode::Off,
            },
        };
        match mode {
            Mode::Count => tracing::info!(
                "the heap sync's joins a sync that had ended covered are counted, every join waits as before \
                 (VICTORY_HEAP_SYNC_NEED=count)"
            ),
            Mode::Check => tracing::info!(
                "a submission joins the heap sync only when its lists need what that sync mirrors, each skip checked \
                 against the heaps' pages written before the lists' hand-over (VICTORY_HEAP_SYNC_NEED=check)"
            ),
            _ => {}
        }
        mode
    })
}

/// Whether the switch is set at all.
pub(crate) fn set() -> bool {
    mode() != Mode::Off
}

/// Whether covered joins are skipped now (`1`, `check`, `alt`'s ON phases).
fn skipping() -> bool {
    match mode() {
        Mode::On | Mode::Check => true,
        Mode::Alt => NEED.on(),
        Mode::Off | Mode::Count => false,
    }
}

/// The mode for the line.
fn said() -> &'static str {
    match mode() {
        Mode::Count => "count",
        Mode::Check => "check",
        _ => NEED.said(),
    }
}

/// The syncs' state (the one of the run, [`state`]; tests make their own).
pub(super) struct State {
    /// The highest asking stamp of a sync that has ended (0: none yet).
    covered: Arc<AtomicU64>,
    /// The sync left running without a join, if any.
    running: Mutex<Option<Early>>,
    /// `check`: the submission being prepared failed the check at its start
    /// (rule 2 not taken): its join waits whole, not checked again.
    failed: AtomicBool,
}

/// What [`State::begin`] did, at a submission's sync place.
pub(super) enum Begun {
    /// A sync of its own was started (rules 1, 4 and 5); after waiting this
    /// long for the one left running to end (rule 4), if that was the way.
    Own(Early, Option<Duration>),
    /// The one left running is taken as its own (rule 3).
    Adopted(Early),
    /// None started: the one left running still runs, and an ended sync
    /// covers the lists (rule 2).
    Covered,
}

/// What [`State::join`] did.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Joined {
    /// No ended sync covered the lists: joined whole.
    Needed,
    /// Covered, joined all the same: `count`, an off phase, or (with
    /// `failed`) the check found pages not taken by an ended sync.
    Waited { failed: bool },
    /// Covered: left running, not waited for.
    Skipped,
}

impl State {
    pub(super) fn new() -> State {
        State { covered: Arc::new(AtomicU64::new(0)), running: Mutex::new(None), failed: AtomicBool::new(false) }
    }

    fn lock(&self) -> MutexGuard<'_, Option<Early>> {
        self.running.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// [`State::covered`] now.
    pub(super) fn covered(&self) -> u64 {
        self.covered.load(SeqCst)
    }

    /// Gives `sync` to `lane`, its asking stamp kept (taken at its start,
    /// before it asks for any page) and, once it has ended, made the
    /// covered stamp. Not waited for.
    pub(super) fn give(&self, lane: &Lane, sync: impl FnOnce(&dyn Fn()) + Send + 'static) -> Early {
        let asked = Arc::new(AtomicU64::new(0));
        let (at, covered) = (asked.clone(), self.covered.clone());
        let beside = beside::give(lane, move |told| {
            let stamp = super::stamp();
            at.store(stamp, SeqCst);
            // (`check`: the pages it takes are noted as this sync's.)
            let syncing = ledger::syncing(stamp);
            sync(told);
            drop(syncing);
            // What the sync wrote for the card (slots, heap map entries,
            // strides; write-combined) is out before a submission that
            // does not join it relies on it.
            std::sync::atomic::fence(SeqCst);
            covered.fetch_max(stamp, SeqCst);
        });
        Early { beside, asked }
    }

    /// At a submission's sync place: the sync it joins, after what the one
    /// left running became (the rules of the module's comment), for lists
    /// that need `need`, `settled` or not, with skips on (`on`) or not, and
    /// rule 2 checked first (`check`).
    pub(super) fn begin(
        &self,
        lane: &Lane,
        need: u64,
        settled: bool,
        on: bool,
        check: bool,
        sync: impl FnOnce(&dyn Fn()) + Send + 'static,
    ) -> Begun {
        let mut waited = None;
        self.failed.store(false, SeqCst);
        let left = self.lock().take();
        if let Some(early) = left {
            if early.beside.done() {
                // 1. Ended: counted on the sync line as a sync of its own.
                let _ = early.beside.ended();
                count(N::Ended, 1);
            } else if on && adoptable(self.covered(), need, settled) && self.checked(check, need) {
                // 2. Still runs, and covered: left running, none started.
                *self.lock() = Some(early);
                return Begun::Covered;
            } else if on && adoptable(early.asked.load(SeqCst), need, settled) {
                // 3. It asked after what the lists need: theirs.
                return Begun::Adopted(early);
            } else {
                // 4. Its end, then a sync of its own.
                waited = Some(early.beside.ended());
            }
        }
        let early = self.give(lane, sync);
        early.beside.wait_asked();
        Begun::Own(early, waited)
    }

    /// `check`: whether the ledger finds every page the lists need taken by
    /// an ended sync ([`ledger::verify`]); a failure is kept for the join.
    /// True without the check.
    fn checked(&self, check: bool, need: u64) -> bool {
        if !check {
            return true;
        }
        let clean = ledger::verify(need, self.covered()).clean();
        if !clean {
            self.failed.store(true, SeqCst);
        }
        clean
    }

    /// At the join of `early` by lists that need `need`, `settled` or not,
    /// skips on (`on`) or not, checked first (`check`): left running if an
    /// ended sync covers the lists, else joined. The wait, and what was
    /// done.
    pub(super) fn join(&self, early: Early, need: u64, settled: bool, on: bool, check: bool) -> (Duration, Joined) {
        let failed = self.failed.swap(false, SeqCst);
        let covered = self.covered();
        if !adoptable(covered, need, settled) {
            return (early.beside.join(), Joined::Needed);
        }
        if !on {
            return (early.beside.join(), Joined::Waited { failed: false });
        }
        // (Failed at the start already: not checked again.)
        if failed || !self.checked(check, need) {
            self.failed.store(false, SeqCst);
            return (early.beside.join(), Joined::Waited { failed: true });
        }
        let mut running = self.lock();
        if running.is_some() {
            // (Not met: `begin` took the one left running.)
            drop(running);
            return (early.beside.join(), Joined::Waited { failed: false });
        }
        *running = Some(early);
        (Duration::ZERO, Joined::Skipped)
    }

    /// Before a sweep: waits for the one left running to end (it stays left,
    /// and the next submission's start drops it); how long.
    pub(super) fn before_sweep(&self) -> Duration {
        let running = self.lock();
        let Some(early) = running.as_ref() else {
            return Duration::ZERO;
        };
        let waited = early.beside.wait_end();
        drop(running);
        if !waited.is_zero() {
            count(N::Sweeps, 1);
            count(N::SweepUs, waited.as_micros() as u64);
        }
        waited
    }

    /// Tests: whether a sync is left running (ended or not).
    #[cfg(test)]
    pub(super) fn left(&self) -> bool {
        self.lock().is_some()
    }
}

/// The run's syncs.
fn state() -> &'static State {
    static STATE: OnceLock<State> = OnceLock::new();
    STATE.get_or_init(State::new)
}

/// What [`join`] needs of a submission's start.
pub(crate) struct Joining {
    /// The sync's asking stamp (none: no sync was started, rule 2).
    asked: Option<Arc<AtomicU64>>,
    need: u64,
    settled: bool,
    /// Skips on at the submission's start.
    on: bool,
}

/// `super::start` with the switch: the sync of a submission whose lists
/// need `need` (`super::need`), `settled` or not, having waited `before`
/// before its sync's place (a sweep's wait).
pub(super) fn start(need: u64, settled: bool, before: Duration) -> Started {
    let on = skipping();
    let joining = |asked| Some(Joining { asked, need, settled, on });
    let Some(lane) = beside::lane() else {
        // No thread: the sync is done in place, whole.
        return Started { beside: Some(beside::start()), phase: Some(on), before, need: None };
    };
    let state = state();
    let sync = |asked: &dyn Fn()| super::super::sync_heaps_told(Some(asked));
    let (early, before) = match state.begin(lane, need, settled, on, mode() == Mode::Check, sync) {
        Begun::Covered => {
            count(N::NoSync, 1);
            return Started { beside: None, phase: Some(on), before, need: joining(None) };
        }
        Begun::Adopted(early) => {
            count(N::Adopted, 1);
            (early, before)
        }
        Begun::Own(early, waited) => {
            if let Some(waited) = waited {
                count(N::AfterEnd, 1);
                count(N::AfterEndUs, waited.as_micros() as u64);
            }
            (early, before + waited.unwrap_or_default())
        }
    };
    let Early { beside, asked } = early;
    Started { beside: Some(beside), phase: Some(on), before, need: joining(Some(asked)) }
}

/// `super::Started::join` with the switch: the wait, and whether an ended
/// sync covered the lists at the join.
pub(super) fn join(beside: Option<Beside>, joining: Joining) -> (Duration, bool) {
    let Joining { asked, need, settled, on } = joining;
    let (Some(beside), Some(asked)) = (beside, asked) else {
        // Rule 2: none started; covered at the start.
        count(N::Joins, 1);
        count(N::Covered, 1);
        count(N::Skipped, 1);
        return (Duration::ZERO, true);
    };
    let (waited, joined) = state().join(Early { beside, asked }, need, settled, on, mode() == Mode::Check);
    count(N::Joins, 1);
    let us = waited.as_micros() as u64;
    match joined {
        Joined::Needed => count(N::NeededUs, us),
        Joined::Waited { failed } => {
            count(N::Covered, 1);
            count(N::CoveredUs, us);
            if failed {
                count(N::Failed, 1);
            }
        }
        Joined::Skipped => {
            count(N::Covered, 1);
            count(N::Skipped, 1);
        }
    }
    (waited, joined != Joined::Needed)
}

/// `super::before_sweep` with the switch.
pub(super) fn before_sweep() -> Duration {
    state().before_sweep()
}

/// Since the last line, by [`N`].
static COUNTS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

/// The places of [`COUNTS`].
#[derive(Clone, Copy)]
enum N {
    /// Joins (a submission's sync place reached its join, rule 2 included).
    Joins,
    /// Of them, covered by a sync that had ended; of those, skipped; and
    /// those the check found pages for (joined whole).
    Covered,
    Skipped,
    Failed,
    /// Microseconds waited at covered joins all the same, and at the others.
    CoveredUs,
    NeededUs,
    /// Rule 2: no sync started; rule 3: the one left running taken; rule 1:
    /// found it ended; rule 4: waited for its end, and how long.
    NoSync,
    Adopted,
    Ended,
    AfterEnd,
    AfterEndUs,
    /// Sweeps that waited for the one left running, and how long.
    Sweeps,
    SweepUs,
}

fn count(n: N, by: u64) {
    COUNTS[n as usize].fetch_add(by, Relaxed);
}

/// The heap sync line's part with the switch, per frame over `frames`;
/// `phases` is the join by phase (`super::phases`). The counts taken.
pub(super) fn part(frames: f64, phases: &str) -> String {
    let counts: [u64; 13] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed));
    said_part(&counts, frames, said(), phases) + &ledger::part(frames)
}

fn said_part(c: &[u64; 13], frames: f64, mode: &str, phases: &str) -> String {
    let frames = frames.max(1.0);
    let per = |n: N| c[n as usize] as f64 / frames;
    let ms = |n: N| c[n as usize] as f64 / 1000.0 / frames;
    format!(
        " (VICTORY_HEAP_SYNC_NEED {mode}: per frame {:.1} joins, {:.1} covered by a sync that had ended, {:.1} skipped, \
         {:.1} joined whole for the check; {:.2} ms waited at covered joins all the same (what skipping them saves: \
         count, off phases, checks failed) and {:.2} ms at the others; {:.1} submissions started no sync (the one left \
         running was covered), {:.1} took the one left running, {:.1} found it ended, {:.1} waited {:.2} ms for its end, \
         {:.1} sweeps waited {:.2} ms for it; the submitting thread at the join: {phases})",
        per(N::Joins),
        per(N::Covered),
        per(N::Skipped),
        per(N::Failed),
        ms(N::CoveredUs),
        ms(N::NeededUs),
        per(N::NoSync),
        per(N::Adopted),
        per(N::Ended),
        per(N::AfterEnd),
        ms(N::AfterEndUs),
        per(N::Sweeps),
        ms(N::SweepUs),
    )
}

#[cfg(test)]
mod tests {
    use super::super::{need_with, stamp};
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    use std::time::Instant;

    fn lane() -> Arc<Lane> {
        Lane::new("victory-heap-sync-test").expect("a thread")
    }

    /// A sync that says it asked and ends.
    fn quick(asked: &dyn Fn()) {
        asked();
    }

    /// A sync that says it asked, then waits to be let go (`go`), then ends.
    fn held(go: mpsc::Receiver<()>) -> impl FnOnce(&dyn Fn()) + Send + 'static {
        move |asked: &dyn Fn()| {
            asked();
            let _ = go.recv();
        }
    }

    /// Waits until `state`'s covered stamp is at least `at` (a sync's end
    /// raises it on the lane's thread just after its ticket says done).
    fn covered_by(state: &State, at: u64) {
        let began = Instant::now();
        while state.covered() < at {
            assert!(began.elapsed() < Duration::from_secs(10), "the covered stamp never rose");
            std::thread::yield_now();
        }
    }

    /// The needed-slot rule: covered only by a sync that has ended and asked
    /// after the lists' stamp, and never when flights were settled first. A
    /// list of ours (no stamp) needs a sync asked after the title's latest
    /// call.
    #[test]
    fn covered_only_by_an_ended_sync_that_asked_after_the_hand_over() {
        let lane = lane();
        let state = State::new();
        assert_eq!(state.covered(), 0, "nothing covers before any sync ended");
        let handed = stamp();
        // A sync asks after the hand-over and is still running: not covered.
        let (go, wait) = mpsc::channel();
        let early = state.give(&lane, held(wait));
        early.beside.wait_asked();
        let asked = early.asked.load(SeqCst);
        assert!(asked > handed);
        assert!(!adoptable(state.covered(), handed, false), "covered by a sync still running");
        go.send(()).unwrap();
        let _ = early.beside.join();
        covered_by(&state, asked);
        assert_eq!(state.covered(), asked);
        assert!(adoptable(state.covered(), handed, false), "an ended sync asked after the hand-over");
        assert!(!adoptable(state.covered(), handed, true), "flights settled first");
        // A hand-over after that sync's asking: not covered by it.
        let later = stamp();
        assert!(!adoptable(state.covered(), later, false));
        // A list of ours needs a sync asked after the title's latest call.
        assert_eq!(need_with([0, handed].into_iter(), later), later);
        assert!(!adoptable(state.covered(), later, false));
    }

    /// The covered stamp is raised only after the sync's work is done and
    /// never by a sync that panicked.
    #[test]
    fn the_covered_stamp_follows_the_sync_s_end_and_not_a_panic() {
        let lane = lane();
        let state = State::new();
        let wrote = Arc::new(AtomicBool::new(false));
        let early = {
            let wrote = wrote.clone();
            state.give(&lane, move |asked: &dyn Fn()| {
                asked();
                std::thread::sleep(Duration::from_millis(20));
                wrote.store(true, SeqCst);
            })
        };
        early.beside.wait_asked();
        let asked = early.asked.load(SeqCst);
        assert!(state.covered() < asked, "covered before the end");
        let _ = early.beside.join();
        covered_by(&state, asked);
        assert!(wrote.load(SeqCst), "covered before the sync's work was done");
        // A sync that panics covers nothing.
        let early = state.give(&lane, |asked: &dyn Fn()| {
            asked();
            panic!("a sync that fails");
        });
        early.beside.wait_asked();
        let failed = early.asked.load(SeqCst);
        let joined = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| early.beside.join()));
        assert!(joined.is_err());
        // The lane goes on; a later sync covers past it.
        let early = state.give(&lane, quick);
        let _ = early.beside.join();
        let next = early.asked.load(SeqCst);
        covered_by(&state, next);
        assert!(next > failed);
        assert_eq!(state.covered(), next, "the stamp of the sync that panicked was taken");
    }

    /// At the join: skipped only when covered and skipping is on; joined
    /// whole when not covered (`Needed`), in `count` and off phases
    /// (`Waited`), or when a sync is left already.
    #[test]
    fn a_join_is_skipped_only_when_covered_and_on() {
        let lane = lane();
        let state = State::new();
        // A sync that ended covers a first hand-over.
        let handed = stamp();
        let first = state.give(&lane, quick);
        let _ = first.beside.join();
        covered_by(&state, first.asked.load(SeqCst));
        // A hand-over after every sync's asking: needed, whoever ended.
        let own = state.give(&lane, quick);
        own.beside.wait_asked();
        let later = stamp();
        assert_eq!(state.join(own, later, false, true, false).1, Joined::Needed);
        // Covered, skips off (count, off phase): waited for all the same.
        let (go, wait) = mpsc::channel();
        let running = state.give(&lane, held(wait));
        running.beside.wait_asked();
        go.send(()).unwrap();
        assert_eq!(state.join(running, handed, false, false, false).1, Joined::Waited { failed: false });
        assert!(!state.left());
        // Covered, settled first: needed.
        let running = state.give(&lane, quick);
        assert_eq!(state.join(running, handed, true, true, false).1, Joined::Needed);
        // Covered and on: left running, not waited for.
        let (go, wait) = mpsc::channel();
        let running = state.give(&lane, held(wait));
        running.beside.wait_asked();
        let began = Instant::now();
        let (waited, joined) = state.join(running, handed, false, true, false);
        assert_eq!(joined, Joined::Skipped);
        assert_eq!(waited, Duration::ZERO);
        assert!(began.elapsed() < Duration::from_secs(5));
        assert!(state.left(), "the sync was not left running");
        // Its end is waited for before a sweep, and it stays left.
        go.send(()).unwrap();
        let _ = state.before_sweep();
        assert!(state.left());
        // The next start finds it ended (rule 1) and starts its own.
        match state.begin(&lane, stamp(), false, true, false, quick) {
            Begun::Own(early, None) => drop(early.beside.join()),
            _ => panic!("an ended sync left running was not dropped"),
        }
        assert!(!state.left());
    }

    /// The four rules at a submission's start with a sync left running.
    #[test]
    fn a_submission_s_start_meets_the_sync_left_running_by_the_rules() {
        let lane = lane();
        let state = State::new();
        // An ended sync covers the first hand-over.
        let handed = stamp();
        let first = state.give(&lane, quick);
        let _ = first.beside.join();
        covered_by(&state, first.asked.load(SeqCst));
        // A hand-over after that sync's asking, before the next one's.
        let between = stamp();
        assert!(between > state.covered());
        // A sync left running (it asked after both).
        let (go, wait) = mpsc::channel();
        let running = state.give(&lane, held(wait));
        running.beside.wait_asked();
        let asked = running.asked.load(SeqCst);
        assert!(asked > between);
        assert_eq!(state.join(running, handed, false, true, false).1, Joined::Skipped);
        // 2. Covered and on: none started, it is still left.
        assert!(matches!(state.begin(&lane, handed, false, true, false, quick), Begun::Covered));
        assert!(state.left());
        // 3. Not covered (handed over after the covering sync asked) but
        // the one left running asked after: taken as theirs.
        let Begun::Adopted(adopted) = state.begin(&lane, between, false, true, false, quick) else {
            panic!("the one left running was not taken");
        };
        assert!(!state.left());
        assert_eq!(adopted.asked.load(SeqCst), asked);
        // Its join: still not covered (it has not ended), so joined whole
        // once let go (a moment after the join began: not ended at its
        // look).
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            go.send(()).unwrap();
        });
        assert_eq!(state.join(adopted, between, false, true, false).1, Joined::Needed);
        releaser.join().unwrap();
        covered_by(&state, asked);
        // 4. Left running, then a hand-over after its asking: waited for to
        // its end, then a sync of its own.
        let (go, wait) = mpsc::channel();
        let running = state.give(&lane, held(wait));
        running.beside.wait_asked();
        let handed = stamp();
        let joiner = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            go.send(()).unwrap();
        });
        // (Left by hand: as a skipped join leaves it.)
        *state.lock() = Some(running);
        let Begun::Own(own, Some(waited)) = state.begin(&lane, handed, false, true, false, quick) else {
            panic!("the one left running was not waited for");
        };
        assert!(waited >= Duration::from_millis(10), "{waited:?}");
        joiner.join().unwrap();
        assert!(own.asked.load(SeqCst) > handed);
        let _ = own.beside.join();
        // Skips off (an off phase): never rule 2 or 3, always rule 4.
        let (go, wait) = mpsc::channel();
        let running = state.give(&lane, held(wait));
        running.beside.wait_asked();
        *state.lock() = Some(running);
        go.send(()).unwrap();
        match state.begin(&lane, 1, false, false, false, quick) {
            Begun::Own(own, _) => drop(own.beside.join()),
            _ => panic!("off, a sync left running was covered or taken"),
        }
    }

    /// `check`: a page noted at the lists' stamp that no ended sync took
    /// fails the check, at the start (rule 2 not taken) and at the join
    /// (joined whole, `failed`), checked once a submission.
    #[test]
    fn a_failed_check_makes_the_join_wait_whole() {
        let lane = lane();
        let state = State::new();
        // A heap page noted at the hand-over, which no sync takes.
        let handed = stamp();
        super::ledger::note_for_test(handed, vec![usize::MAX / crate::watch::PAGE]);
        let first = state.give(&lane, quick);
        let _ = first.beside.join();
        covered_by(&state, first.asked.load(SeqCst));
        // At the join: covered, but the check fails: joined whole.
        let (go, wait) = mpsc::channel();
        let running = state.give(&lane, held(wait));
        running.beside.wait_asked();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            go.send(()).unwrap();
        });
        assert_eq!(state.join(running, handed, false, true, true).1, Joined::Waited { failed: true });
        releaser.join().unwrap();
        assert!(!state.left());
        // At the start: one left running, covered, the check fails: rule 2
        // not taken (rule 3 here: it asked after the hand-over), and its join
        // waits whole without a second check.
        let (go, wait) = mpsc::channel();
        let running = state.give(&lane, held(wait));
        running.beside.wait_asked();
        *state.lock() = Some(running);
        let Begun::Adopted(adopted) = state.begin(&lane, handed, false, true, true, quick) else {
            panic!("rule 2 taken although the check failed");
        };
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            go.send(()).unwrap();
        });
        assert_eq!(state.join(adopted, handed, false, true, true).1, Joined::Waited { failed: true });
        releaser.join().unwrap();
        // Without the check the same is skipped.
        let running = state.give(&lane, quick);
        running.beside.wait_asked();
        assert_eq!(state.join(running, handed, false, true, false).1, Joined::Skipped);
        let _ = state.before_sweep();
    }

    /// The needed-slot rule end to end, as `Host::submit` walks it: every
    /// record a submission's lists were handed over after is mirrored when
    /// the submission goes on past its join, skipped or not; the sync left
    /// running only mirrors records written after the hand-over. A model of
    /// title slots and host slots, with syncs that copy what changed since
    /// the one before took the slots (one at a time on the lane, as the
    /// journal hands pages out), run over many random bursts.
    #[test]
    fn no_submission_goes_on_with_a_slot_written_before_its_hand_over_unmirrored() {
        const SLOTS: usize = 64;
        let lane = lane();
        let state = State::new();
        // Title slots (a record and the stamp of its write) and host slots
        // (the record mirrored), and the slots written since the last take.
        let title: Arc<Mutex<Vec<(u64, u64)>>> = Arc::new(Mutex::new(vec![(0, 0); SLOTS]));
        let host: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(vec![0; SLOTS]));
        let dirty: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(vec![false; SLOTS]));
        // A sync: takes the dirty slots at its asking, then mirrors them
        // slowly (records as they are then, as the sync reads its pages
        // after asking).
        let sync = |title: Arc<Mutex<Vec<(u64, u64)>>>, host: Arc<Mutex<Vec<u64>>>, dirty: Arc<Mutex<Vec<bool>>>, slow: u64| {
            move |asked: &dyn Fn()| {
                let taken: Vec<usize> = {
                    let mut dirty = dirty.lock().unwrap();
                    let taken = (0..SLOTS).filter(|&i| dirty[i]).collect();
                    dirty.iter_mut().for_each(|d| *d = false);
                    taken
                };
                asked();
                for i in taken {
                    std::thread::sleep(Duration::from_micros(slow));
                    let record = title.lock().unwrap()[i].0;
                    host.lock().unwrap()[i] = record;
                }
            }
        };
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let mut record = 0u64;
        let (mut skipped, mut checked) = (0, 0);
        for _burst in 0..40 {
            // The title writes some slots for this frame, then hands over a
            // few submissions at once (a burst); each has the slots it reads.
            let mut lists: Vec<(u64, Vec<usize>)> = Vec::new();
            for _ in 0..1 + random(4) {
                let reads: Vec<usize> = (0..1 + random(6)).map(|_| random(SLOTS as u64) as usize).collect();
                for &i in &reads {
                    record += 1;
                    title.lock().unwrap()[i] = (record, stamp());
                    dirty.lock().unwrap()[i] = true;
                }
                lists.push((stamp(), reads));
            }
            // The submitting thread works through the burst while the
            // title writes the next frame's slots (other slots: a title
            // does not rewrite slots handed-over lists still read).
            for (need, reads) in lists.iter().cloned() {
                for _ in 0..random(3) {
                    let i = random(SLOTS as u64) as usize;
                    if lists.iter().any(|(_, r)| r.contains(&i)) {
                        continue;
                    }
                    record += 1;
                    title.lock().unwrap()[i] = (record, stamp());
                    dirty.lock().unwrap()[i] = true;
                }
                let make = sync(title.clone(), host.clone(), dirty.clone(), random(300));
                let on = random(5) != 0;
                let begun = state.begin(&lane, need, false, on, false, make);
                let joined = match begun {
                    Begun::Covered => Joined::Skipped,
                    Begun::Adopted(early) | Begun::Own(early, _) => state.join(early, need, false, on, false).1,
                };
                if joined == Joined::Skipped {
                    skipped += 1;
                }
                // Past the join: every slot the lists read shows a record
                // written at or after the one it had at the hand-over.
                let title_now = title.lock().unwrap().clone();
                let host_now = host.lock().unwrap().clone();
                for &i in &reads {
                    checked += 1;
                    let (now, written) = title_now[i];
                    if written < need {
                        assert_eq!(host_now[i], now, "slot {i}: written before the hand-over, not mirrored ({joined:?})");
                    }
                }
            }
        }
        // Leave nothing running.
        let _ = state.before_sweep();
        assert!(skipped > 0, "no join was ever skipped: the rule was not exercised");
        assert!(checked > 0);
    }

    /// What newer titles pass to make a heap (as the descriptor tests do).
    #[repr(C)]
    struct Extended(super::super::super::HeapDesc, super::super::super::Placement);

    /// The needed-slot rule on a real placed, write-watched sampler heap and
    /// the real heap sync, as a burst goes: two lists that read slot 5 are
    /// handed over; the first submission's sync is joined (nothing covers
    /// it yet) and mirrors slot 5; the title then writes slot 6 for the next
    /// frame; the second submission's join is skipped (the first sync, ended,
    /// asked after its hand-over) with slot 5 already shown, and its sync,
    /// left running, mirrors slot 6 by its end; a third list handed over
    /// after a new record in slot 5 is not covered by the sync left running
    /// (it asked before that hand-over): its join waits, and slot 5 shows
    /// the new record. `None` without a host heap (no device).
    fn burst_over_a_real_heap() -> Option<()> {
        use super::super::super::{HEAP_IIDS, HeapDesc, Placement, SAMPLERS, SHADER_VISIBLE, create_heap, create_sampler, heap};
        use crate::object::{S_OK, This, slot};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: a fresh allocation of the test's own.
        let memory = unsafe { VirtualAlloc(std::ptr::null(), crate::watch::PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) };
        assert!(!memory.is_null());
        let desc = HeapDesc { kind: SAMPLERS, count: 64, flags: SHADER_VISIBLE, nodes: 1 };
        let extended = Extended(desc, Placement { address: memory as u64, size: 64 * 16 });
        let mut h: This = std::ptr::null();
        assert_eq!(create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], &mut h), S_OK);
        let mut seen = None;
        if let Some(host) = heap(h).host() {
            let mut elsewhere = [0u64; 6];
            let source = elsewhere.as_mut_ptr() as u64;
            create_sampler(std::ptr::null(), &[21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], source);
            create_sampler(std::ptr::null(), &[20, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], source + 16);
            create_sampler(std::ptr::null(), &[0, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], source + 32);
            let copy = |from: u64, to: usize| {
                // SAFETY: the heap slot and the source both hold 16 bytes.
                unsafe { std::ptr::copy_nonoverlapping(from as *const u8, memory.cast::<u8>().add(to * 16), 16) };
            };
            let key = |slot: usize| host.slots_seen(&[slot as u32])[0].0;
            let lane = lane();
            let state = State::new();
            let sync = |asked: &dyn Fn()| super::super::super::sync_heaps_told(Some(asked));
            // A sync that lingers after its work (not ended when its join
            // looks: the covered stamp is raised only at its end).
            let slow = |asked: &dyn Fn()| {
                super::super::super::sync_heaps_told(Some(asked));
                std::thread::sleep(Duration::from_millis(30));
            };
            // The burst: slot 5 written, two lists handed over.
            copy(source, 5);
            let (first, second) = (stamp(), stamp());
            // 1st submission: nothing covers it yet; joined.
            let Begun::Own(early, _) = state.begin(&lane, first, false, true, false, slow) else {
                panic!("no sync of its own");
            };
            assert_eq!(state.join(early, first, false, true, false).1, Joined::Needed);
            let five = key(5);
            assert_ne!(five, 0, "slot 5 not mirrored by the first submission's sync");
            // The title records the next frame: slot 6.
            copy(source + 16, 6);
            // 2nd submission: its own sync (none left), skipped at the join.
            let Begun::Own(early, _) = state.begin(&lane, second, false, true, false, sync) else {
                panic!("no sync of its own");
            };
            let (waited, joined) = state.join(early, second, false, true, false);
            assert_eq!(joined, Joined::Skipped);
            assert_eq!(waited, Duration::ZERO);
            assert_eq!(key(5), five, "the second list's slot was not shown at its skipped join");
            // A sweep waits for the sync left running: slot 6 is mirrored.
            let _ = state.before_sweep();
            let six = key(6);
            assert_ne!(six, 0, "slot 6 not mirrored by the sync left running");
            // A new record in slot 5, then a third list.
            copy(source + 32, 5);
            let third = stamp();
            let Begun::Own(early, _) = state.begin(&lane, third, false, true, false, slow) else {
                panic!("no sync of its own");
            };
            assert_eq!(state.join(early, third, false, true, false).1, Joined::Needed, "covered by a sync that asked before");
            assert_ne!(key(5), five, "the third list's slot 5 not made from its new record");
            assert_eq!(key(6), six);
            seen = Some(());
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
        seen
    }

    #[test]
    fn a_skipped_join_leaves_its_lists_slots_mirrored_on_a_real_heap() {
        if burst_over_a_real_heap().is_none() {
            eprintln!("no host heap (no device): the real heap was not used");
        }
    }

    /// The line's part: per frame, the covered ms, the rules.
    #[test]
    fn the_line_part_says_the_joins_and_the_rules() {
        let mut c = [0u64; 13];
        c[N::Joins as usize] = 34;
        c[N::Covered as usize] = 20;
        c[N::Skipped as usize] = 16;
        c[N::Failed as usize] = 0;
        c[N::CoveredUs as usize] = 30_000;
        c[N::NeededUs as usize] = 8_000;
        c[N::NoSync as usize] = 4;
        c[N::AfterEnd as usize] = 2;
        c[N::AfterEndUs as usize] = 1_000;
        let text = said_part(&c, 2.0, "alt, ON", "ON 1.00 ms a frame (+0.00 before) over 2 frames, off no submission");
        assert!(text.starts_with(" (VICTORY_HEAP_SYNC_NEED alt, ON: per frame 17.0 joins, 10.0 covered"), "{text}");
        assert!(text.contains("8.0 skipped"), "{text}");
        assert!(text.contains("15.00 ms waited at covered joins all the same"), "{text}");
        assert!(text.contains("4.00 ms at the others"), "{text}");
        assert!(text.contains("2.0 submissions started no sync"), "{text}");
        assert!(text.contains("1.0 waited 0.50 ms for its end"), "{text}");
        assert!(text.ends_with("at the join: ON 1.00 ms a frame (+0.00 before) over 2 frames, off no submission)"), "{text}");
    }
}
