//! VIcky: the next submission's heap sync starts while it waits in line
//! (`VICTORY_HEAP_SYNC_AT_SUBMIT=1|alt`; off by default, and used only with
//! `VICTORY_HEAP_SYNC_BESIDE=1`, [`super::beside`]). Port of Belobug
//! 673be0e + a609a2f (there under `VICTORY_PORT_JOURNAL`: "the heaps'
//! records asked for at the title's submit", a round of its heap-sync
//! thread asked when the title submits and waited for by the lane),
//! adapted to the sync beside instead of Belobug's round-based thread and
//! journal.
//!
//! Why: on the prologue (RTX 4080, RT on) the submitting thread's frame is
//! ~137 ms: 88 ms preparing, a 23 ms wait at the join of the heap sync
//! that runs beside it, a 21 ms wait for the card. The sync of a
//! submission starts when the submission's preparing starts, and the
//! wait at the join is the sync's excess over the preparing (the line by
//! place in the burst, `super::burst`: the 6th and 7th syncs of a frame,
//! and the present's own, whose preparing is short, carry most of it).
//! Meanwhile the sync's lane idles between a submission's sync and the
//! next one's, while the submitting thread takes the snapshots, sends the
//! lists, lands flights and walks the queues' lines.
//!
//! With the switch, when a submission's asking of title memory ends
//! (`crate::watch::ask_once_end`, after its steps) and a submission of the
//! title's waits in a queue's line (or the title is in its present), the
//! next submission's sync is given to the lane at once ([`ahead`]), behind
//! this submission's own: it runs while the submitting thread ends this
//! submission and the next one waits in line. The next submission takes
//! it as its own sync ([`start`], where the sync beside would start), if
//! it asked for the heaps' pages after everything that submission must
//! see; it is then joined where the sync beside is joined. Otherwise the
//! submission waits for its end and starts a sync of its own, as without
//! the switch.
//!
//! What a submission must see ([`need`], with stamps from one counter,
//! [`stamp`]): the title's records written before the title handed its
//! lists over (stamped per host list by `crate::queue`'s execute, the
//! title's own call: `handed`), and, for a part behind a wait that may
//! have been met by the title's own hand, before that wait was passed
//! (Belobug a609a2f: each queue's floor, raised when our thread passes a
//! wait for a fence, or for a memory value found in memory, and given to
//! the parts taken behind it, [`wait_passed`] and [`taken`]; a value wait
//! passed for a write of a flight in the air, or of a part joined before
//! it, is the card's order and raises nothing). Lists of ours (the
//! present's copy, our checks) carry no stamp: theirs is the title's
//! latest call, a present included ([`presenting`]). And not adopted when
//! the submission settled flights before the sync's place (a waited-for
//! submission, or one whose preparing reads what flights write, or the
//! forced sweep's settle): those landings come before the sync without the
//! switch.
//!
//! Why not Belobug's whole design (its sync thread runs numbered rounds
//! asked at any moment, several submissions may share one, and its journal
//! of written pages was cut into 64 parts with locks of their own, some
//! 1,200 lines): here a sync may not ask for the heaps' pages while a
//! submission's asking of title memory stands (fxceless/victory-fork#410,
//! `super::beside`: a page found written would get a number above that
//! asking's, and a host copy holding it would be copied again at every
//! refresh). Our journal (`crate::watch`) has moved far from Belobug's
//! (light refresh, reservations, guards), so its rework does not port;
//! the sync ahead is given only once the submission's asking has ended,
//! and the next submission waits for its asking before its own. Sharing
//! one sync among submissions is `VICTORY_HEAP_SYNC_BURST`, measured on
//! 2026-11-27 (SB1/SB2): the same slots mirrored by fewer syncs that
//! overlap less, the join 28 -> 44 ms; here each submission keeps a sync
//! of its own, only begun earlier. Belobug's `prep_wrote` (our steps
//! writing a heap's memory raise the floor) is not needed: the steps of
//! the submission before are all done when its asking ends, and a
//! submission's own steps come after its sync, with the switch or not.
//! Unlike `VICTORY_HEAP_SYNC_AHEAD` (an extra sync between submissions,
//! stopped at every submission's start: 5.85 ms waited 4.6 times a frame,
//! measured neutral), the sync ahead here is the next submission's own:
//! nothing stops it, and the submission waits for it only where it would
//! wait anyway (the join), before a sweep, or when it cannot take it.
//!
//! The order rules of [`super::beside`], and how they hold:
//!
//! - The sweep never meets a sync: a submission whose sweep is due waits
//!   for a sync ahead to end first ([`before_sweep`]), and may still take
//!   it (sync, then sweep: the order without the sync beside).
//! - The journal: the sync ahead asks only after the asking of the
//!   submission before has ended, and the next submission waits for that
//!   asking before its own ([`start`]).
//! - The submission counted as started: a sync ahead runs while the
//!   submission before counts as started and the one it is for does not
//!   yet: a host resource a slot lets go of then waits for the end of the
//!   one before, and a heap map pool slot given back is handed out after
//!   it, as for a sync in the order without the sync beside (before its
//!   submission counted itself). Nothing of the submission it is for reads
//!   them after its join.
//! - Slots left waiting for a texture's host copy: made by the next
//!   `make_waiting`, at the latest after the join, as before.
//! - What the card may meet: a sync ahead mirrors the records written up
//!   to its asking, which comes after the hand-over of the lists it is
//!   taken for; lists sent before may be on the card while it runs, as
//!   they are while the next submission's own sync runs without the
//!   switch. One moment is new: the
//!   lists of the submission whose asking just ended, between its join
//!   and its `ExecuteCommandLists`, may meet a slot mirrored by the next
//!   submission's sync. Such a record was written after those lists were
//!   handed over: on the console a race the title does not run (it would
//!   rewrite a slot that lists it handed over still read), and the views
//!   the title makes or copies by calls are mirrored at that moment today.
//! - Landings: what flights copy back into title memory lands on the
//!   relay's thread at any moment, with the switch or not; a list that
//!   needs a flight's results first waits for it (a fence or a memory
//!   value: the floors) or settles it before its sync (not taken then).
//!
//! The heap sync's line ([`part`], with the sync beside's) says per frame
//! how many syncs were started ahead, taken, not taken (and why), the ms
//! waited for those not taken and before sweeps, and the submitting
//! thread's ms at the join with the switch ON and off (with `alt`, by
//! phase). `alt`: twenty seconds off, twenty on, in turn, each change said
//! as `(VICTORY_HEAP_SYNC_AT_SUBMIT=alt): ON` / `: off, as usual`
//! (`xcl-notes/alt-fps.sh`); the stamps are taken in both phases, and a
//! sync started ahead in an ON phase may be taken in the off one after.
//!
//! VIcky (fable-2026-11-27-pf5b.md, lever 1): the same stamps serve
//! `VICTORY_HEAP_SYNC_NEED=count|1|alt|check` ([`need`]): a submission's
//! sync starts where it does without either switch (none while the one
//! left running still runs and an ended one covers the lists), and is
//! joined only when no sync that has ended covers its lists. The two
//! switches exclude each other: with both set, `VICTORY_HEAP_SYNC_NEED` is
//! used and no sync is started ahead ([`ahead`]).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use super::beside::{self, Beside, Lane};

pub(crate) mod ledger;
pub(crate) mod need;

/// The switch.
pub(crate) static AT_SUBMIT: crate::alt::Alt = crate::alt::Alt::new(
    "VICTORY_HEAP_SYNC_AT_SUBMIT",
    "the next submission's heap sync starts while it waits in line (port of Belobug 673be0e + a609a2f)",
);

/// Whether the switch, or VIcky's `VICTORY_HEAP_SYNC_NEED` ([`need`]), is
/// set and the sync runs beside: stamps are taken (and syncs started ahead,
/// or joins skipped) only then.
pub(crate) fn used() -> bool {
    static USED: OnceLock<bool> = OnceLock::new();
    *USED.get_or_init(|| {
        let need = need::set();
        if !AT_SUBMIT.set() && !need {
            return false;
        }
        let name = if need { "VICTORY_HEAP_SYNC_NEED" } else { "VICTORY_HEAP_SYNC_AT_SUBMIT" };
        if !beside::on() {
            tracing::warn!(
                "{name} is used only with VICTORY_HEAP_SYNC_BESIDE=1 (with the heap sync on, without levels on demand): \
                 every submission starts its own sync and joins it, as without it"
            );
            return false;
        }
        if need && AT_SUBMIT.set() {
            tracing::warn!(
                "VICTORY_HEAP_SYNC_AT_SUBMIT and VICTORY_HEAP_SYNC_NEED exclude each other: VICTORY_HEAP_SYNC_NEED is used, \
                 and no sync is started ahead"
            );
        }
        true
    })
}

/// The one counter of [`stamp`].
static SEQ: AtomicU64 = AtomicU64::new(0);

/// A new stamp, above every stamp taken before it (never 0). Taken with a
/// full fence: whatever this thread did before taking it happens before
/// anything another thread does after taking a higher one.
pub(crate) fn stamp() -> u64 {
    SEQ.fetch_add(1, SeqCst) + 1
}

/// The title's latest call that hands work over or presents: what a
/// submission of lists of ours must see ([`need`]).
static TITLE_CALL: AtomicU64 = AtomicU64::new(0);

/// The title hands lists over to a queue (`crate::queue`'s execute, before
/// anything is put in a line or sent): each host list's stamp (its
/// `handed`) is raised to now.
pub(crate) fn handed<'a>(lists: impl Iterator<Item = &'a AtomicU64>) {
    if !used() {
        return;
    }
    // (VIcky, `VICTORY_HEAP_SYNC_NEED=check`: the heaps' pages written so
    // far are noted first, [`ledger`].)
    let now = ledger::stamp_noted();
    TITLE_CALL.fetch_max(now, SeqCst);
    for list in lists {
        list.fetch_max(now, SeqCst);
    }
}

/// Presents of the title's in progress ([`presenting`]).
static PRESENTS: AtomicUsize = AtomicUsize::new(0);

/// The title presents (`crate::queue`'s present): a call that the
/// present's own submission must see, and, while the guard lives, a
/// submission to come for [`ahead`].
pub(crate) fn presenting() -> Presenting {
    let on = used();
    if on {
        TITLE_CALL.fetch_max(ledger::stamp_noted(), SeqCst);
        PRESENTS.fetch_add(1, SeqCst);
    }
    Presenting(on)
}

/// See [`presenting`].
pub(crate) struct Presenting(bool);

impl Drop for Presenting {
    fn drop(&mut self) {
        if self.0 {
            PRESENTS.fetch_sub(1, SeqCst);
        }
    }
}

/// A queue's line passed a wait the title's own hand may have met (a
/// fence's, or a memory value found in memory): the parts behind it need
/// a sync asked after now (`floor` is the queue's, raised).
pub(crate) fn wait_passed(floor: &AtomicU64) {
    if used() {
        raise(floor);
    }
}

fn raise(floor: &AtomicU64) {
    floor.fetch_max(ledger::stamp_noted(), SeqCst);
}

/// A part is taken off a queue's line: its host lists' stamps are raised
/// to the queue's `floor` (Belobug a609a2f's floors per queue).
pub(crate) fn taken<'a>(floor: &AtomicU64, lists: impl Iterator<Item = &'a AtomicU64>) {
    if used() {
        apply(floor, lists);
    }
}

fn apply<'a>(floor: &AtomicU64, lists: impl Iterator<Item = &'a AtomicU64>) {
    let floor = floor.load(SeqCst);
    if floor == 0 {
        return;
    }
    for list in lists {
        list.fetch_max(floor, SeqCst);
    }
}

/// What a submission of lists with these stamps needs: a sync that asked
/// after the highest. A list without one (ours) needs one asked after the
/// title's latest call.
fn need(stamps: impl Iterator<Item = u64>) -> u64 {
    need_with(stamps, TITLE_CALL.load(SeqCst))
}

fn need_with(stamps: impl Iterator<Item = u64>, title_call: u64) -> u64 {
    let mut need = 0;
    for stamp in stamps {
        need = need.max(if stamp == 0 { title_call } else { stamp });
    }
    need
}

/// Whether a sync that asked at `asked` serves a submission that needs
/// `need` and settled flights before its sync's place or not (`settled`).
fn adoptable(asked: u64, need: u64, settled: bool) -> bool {
    !settled && asked != 0 && asked > need
}

/// A sync started ahead, not taken yet.
struct Early {
    beside: Beside,
    /// The stamp taken just before it asked for the heaps' pages (0 until
    /// then).
    asked: Arc<AtomicU64>,
}

/// What became of a sync started ahead at a submission's start.
enum Taken {
    /// None was started.
    None,
    /// Taken as the submission's sync.
    Adopted(Beside),
    /// Not taken: ended, after the submission waited this long for it;
    /// `true` if because the submission settled flights first.
    Ended(Duration, bool),
}

/// The next submission's sync, started ahead ([`AHEAD`]; tests make their
/// own).
struct Ahead {
    pending: Mutex<Option<Early>>,
}

impl Ahead {
    const fn new() -> Ahead {
        Ahead { pending: Mutex::new(None) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Early>> {
        self.pending.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Gives `sync` to `lane`, unless a sync ahead is pending; whether it
    /// was given.
    fn give(&self, lane: &Lane, sync: impl FnOnce(&dyn Fn()) + Send + 'static) -> bool {
        let mut pending = self.lock();
        if pending.is_some() {
            return false;
        }
        let asked = Arc::new(AtomicU64::new(0));
        let at = asked.clone();
        let beside = beside::give(lane, move |told| {
            at.store(stamp(), SeqCst);
            sync(told);
        });
        *pending = Some(Early { beside, asked });
        true
    }

    /// Before a sweep: waits for a pending sync ahead to end (it stays
    /// pending); how long.
    fn before_sweep(&self) -> Duration {
        self.lock().as_ref().map_or(Duration::ZERO, |early| early.beside.wait_end())
    }

    /// A submission that needs `need` and `settled` or not takes the sync
    /// started ahead, if any; one it cannot take is waited for to its end.
    fn take(&self, need: u64, settled: bool) -> Taken {
        let Some(early) = self.lock().take() else {
            return Taken::None;
        };
        early.beside.wait_asked();
        if adoptable(early.asked.load(SeqCst), need, settled) {
            return Taken::Adopted(early.beside);
        }
        Taken::Ended(early.beside.ended(), settled)
    }

    /// Tests: waits until the pending sync ahead has asked.
    #[cfg(test)]
    fn wait_asked(&self) {
        if let Some(early) = self.lock().as_ref() {
            early.beside.wait_asked();
        }
    }
}

static AHEAD: Ahead = Ahead::new();

/// Since the last line, by [`C`].
static COUNTS: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];

/// The places of [`COUNTS`].
#[derive(Clone, Copy)]
enum C {
    /// Syncs started ahead.
    Started,
    /// Submissions whose asking ended with nothing in line.
    Nothing,
    /// Syncs ahead taken.
    Adopted,
    /// Not taken: a hand-over or a wait passed after their asking.
    Late,
    /// Not taken: the submission settled flights first.
    Settled,
    /// Microseconds waited for syncs ahead not taken, and before sweeps.
    WaitedUs,
    /// Sweeps that waited for a sync ahead.
    Sweeps,
    /// By phase, ON then off: submissions, microseconds at the join,
    /// microseconds waited before it (sweeps, syncs not taken), frames.
    OnSubmissions,
    OnJoinUs,
    OnBeforeUs,
    OnFrames,
    OffSubmissions,
    OffJoinUs,
    OffBeforeUs,
    OffFrames,
}

fn add(c: C, n: u64) {
    COUNTS[c as usize].fetch_add(n, Relaxed);
}

/// Microseconds this submission waited before its sync's place (a sweep's
/// wait), taken at [`start`].
static BEFORE_US: AtomicU64 = AtomicU64::new(0);

/// The submission being prepared ended its asking of title memory
/// (`crate::host`'s `Host::submit`, after `crate::watch::ask_once_end`,
/// with the sync beside): with the switch on, the next submission's sync
/// is given to the lane now if a submission of the title's waits in a
/// queue's line or the title is in its present
/// (`crate::queue::runs_waiting`).
pub(crate) fn ahead() {
    // (VIcky: not with `VICTORY_HEAP_SYNC_NEED`, which excludes it.)
    if !used() || need::set() || !AT_SUBMIT.on() {
        return;
    }
    if !crate::queue::runs_waiting() && PRESENTS.load(SeqCst) == 0 {
        add(C::Nothing, 1);
        return;
    }
    let Some(lane) = beside::lane() else {
        return;
    };
    if AHEAD.give(lane, |asked| super::sync_heaps_told(Some(asked))) {
        add(C::Started, 1);
    }
}

/// Before the sweep of a submission (`crate::host`'s `Host::submit`): a
/// sync started ahead ends first.
pub(crate) fn before_sweep() {
    if !used() {
        return;
    }
    // VIcky (`VICTORY_HEAP_SYNC_NEED`): a sync left running without a
    // join ends first instead ([`need::before_sweep`]).
    if need::set() {
        let waited = need::before_sweep();
        if !waited.is_zero() {
            BEFORE_US.fetch_add(waited.as_micros() as u64, Relaxed);
        }
        return;
    }
    let waited = AHEAD.before_sweep();
    if !waited.is_zero() {
        add(C::Sweeps, 1);
        add(C::WaitedUs, waited.as_micros() as u64);
        BEFORE_US.fetch_add(waited.as_micros() as u64, Relaxed);
    }
}

/// The heap sync of a submission whose host lists carry `stamps` (their
/// `handed`), where the sync beside starts (`crate::host`'s
/// `Host::submit`, after the sweep and the count of started
/// submissions): the sync started ahead if it serves, else one of its
/// own ([`beside::start`]). `settled`: the submission settled flights
/// before this place.
pub(crate) fn start(stamps: impl Iterator<Item = u64>, settled: bool) -> Started {
    if !used() {
        return Started { beside: Some(beside::start()), phase: None, before: Duration::ZERO, need: None };
    }
    // VIcky (`VICTORY_HEAP_SYNC_NEED`): the sync of its own, or the one
    // left running, or none ([`need::start`]).
    if need::set() {
        let before = Duration::from_micros(BEFORE_US.swap(0, Relaxed));
        return need::start(need(stamps), settled, before);
    }
    let phase = AT_SUBMIT.on();
    let mut before = Duration::from_micros(BEFORE_US.swap(0, Relaxed));
    let beside = match AHEAD.take(need(stamps), settled) {
        Taken::None => beside::start(),
        Taken::Adopted(beside) => {
            add(C::Adopted, 1);
            beside
        }
        Taken::Ended(waited, settled) => {
            add(if settled { C::Settled } else { C::Late }, 1);
            add(C::WaitedUs, waited.as_micros() as u64);
            before += waited;
            beside::start()
        }
    };
    Started { beside: Some(beside), phase: Some(phase), before, need: None }
}

/// A submission's heap sync ([`start`]), joined where the sync beside is.
pub(crate) struct Started {
    /// None: VIcky (`VICTORY_HEAP_SYNC_NEED`), no sync was started, the one
    /// left running still runs and an ended one covers the lists.
    beside: Option<Beside>,
    /// With the switch set: whether it was ON at the submission's start.
    phase: Option<bool>,
    /// What the submission waited for before its sync's place.
    before: Duration,
    /// VIcky (`VICTORY_HEAP_SYNC_NEED`): what the join asks ([`need::join`]).
    need: Option<need::Joining>,
}

impl Started {
    /// [`Beside::join`], counted by phase. VIcky (`VICTORY_HEAP_SYNC_NEED`):
    /// with that switch, skipped when the lists need nothing of the sync
    /// ([`need::join`]). Returns the wait, and whether a sync that had ended
    /// covered the lists at the join (with the switch, `count` included).
    pub(crate) fn join(self) -> (Duration, bool) {
        let (waited, covered) = match (self.beside, self.need) {
            (beside, Some(joining)) => need::join(beside, joining),
            (Some(beside), None) => (beside.join(), false),
            (None, None) => (Duration::ZERO, false),
        };
        if let Some(on) = self.phase {
            note(on, waited, self.before);
        }
        // VIcky (`VICTORY_MADE_PASS_FIRST`): the join by that switch's phase.
        super::host::made_first::joined(waited);
        (waited, covered)
    }
}

/// Presents counted so far by [`note`].
static FRAME_SEEN: AtomicU64 = AtomicU64::new(u64::MAX);

/// A submission joined its sync after `waited`, having waited `before`
/// before it, in an ON phase or not; the presents since the one before go
/// to that phase.
fn note(on: bool, waited: Duration, before: Duration) {
    let [subs, join, earlier, frames] = if on {
        [C::OnSubmissions, C::OnJoinUs, C::OnBeforeUs, C::OnFrames]
    } else {
        [C::OffSubmissions, C::OffJoinUs, C::OffBeforeUs, C::OffFrames]
    };
    add(subs, 1);
    add(join, waited.as_micros() as u64);
    add(earlier, before.as_micros() as u64);
    let now = crate::queue::presented();
    let last = FRAME_SEEN.swap(now, Relaxed);
    if last != u64::MAX {
        add(frames, now.saturating_sub(last));
    }
}

/// The heap sync line's part (empty without the switch), per frame over
/// `frames`; the counts taken.
pub(crate) fn part(frames: f64) -> String {
    if !used() {
        return String::new();
    }
    let counts: [u64; 15] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed));
    // VIcky (`VICTORY_HEAP_SYNC_NEED`): its own part, with the join by phase.
    if need::set() {
        return need::part(frames, &phases(&counts));
    }
    said(&counts, frames, AT_SUBMIT.said())
}

fn said(c: &[u64; 15], frames: f64, mode: &str) -> String {
    let frames = frames.max(1.0);
    let per = |i: C| c[i as usize] as f64 / frames;
    let ms = |i: C| c[i as usize] as f64 / 1000.0 / frames;
    format!(
        " (VICTORY_HEAP_SYNC_AT_SUBMIT {mode}: per frame {:.1} syncs started while the next submission waited in line, \
         {:.1} taken by it, {:.1} not ({:.1} a hand-over or a wait passed after their asking, {:.1} flights settled \
         first), {:.1} asked with nothing in line; {:.2} ms waited for those not taken and before {:.1} sweeps; the \
         submitting thread at the join: {})",
        per(C::Started),
        per(C::Adopted),
        per(C::Late) + per(C::Settled),
        per(C::Late),
        per(C::Settled),
        per(C::Nothing),
        ms(C::WaitedUs),
        per(C::Sweeps),
        phases(c),
    )
}

/// The submitting thread's ms at the join by phase, per frame of each
/// phase: `ON X ms a frame (+Y before) over N frames, off ...`.
fn phases(c: &[u64; 15]) -> String {
    // The join by phase, per frame of that phase (none yet: n/a).
    let phase = |subs: C, join: C, before: C, seen: C| {
        let (n, f) = (c[subs as usize], c[seen as usize]);
        if n == 0 {
            "no submission".to_owned()
        } else if f == 0 {
            format!("{:.2} ms a submission (+{:.2} before) over {n} submissions", c[join as usize] as f64 / 1000.0 / n as f64, c[before as usize] as f64 / 1000.0 / n as f64)
        } else {
            format!(
                "{:.2} ms a frame (+{:.2} before) over {f} frames",
                c[join as usize] as f64 / 1000.0 / f as f64,
                c[before as usize] as f64 / 1000.0 / f as f64
            )
        }
    };
    format!(
        "ON {}, off {}",
        phase(C::OnSubmissions, C::OnJoinUs, C::OnBeforeUs, C::OnFrames),
        phase(C::OffSubmissions, C::OffJoinUs, C::OffBeforeUs, C::OffFrames),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn lane() -> Arc<Lane> {
        Lane::new("victory-heap-sync-test").expect("a thread")
    }

    /// A sync that only says it asked.
    fn quick(asked: &dyn Fn()) {
        asked();
    }

    /// Taken only if it asked after what the submission needs, and the
    /// submission did not settle flights first.
    #[test]
    fn a_sync_ahead_serves_only_a_submission_handed_over_before_its_asking() {
        assert!(adoptable(5, 4, false));
        assert!(!adoptable(4, 4, false), "asked at the hand-over's own stamp");
        assert!(!adoptable(3, 4, false));
        assert!(!adoptable(5, 4, true), "flights settled before the sync's place");
        assert!(!adoptable(0, 0, false), "never asked");
        let lane = lane();
        let ahead = Ahead::new();
        assert!(matches!(ahead.take(0, false), Taken::None));
        // Handed over, then the sync ahead asks: taken.
        let handed = stamp();
        assert!(ahead.give(&lane, quick));
        assert!(!ahead.give(&lane, quick), "one sync ahead at a time");
        assert!(matches!(ahead.take(handed, false), Taken::Adopted(_)));
        // The sync ahead asks, then the hand-over (or a wait passed): not
        // taken; ended.
        assert!(ahead.give(&lane, quick));
        ahead.wait_asked();
        let later = stamp();
        assert!(matches!(ahead.take(later, false), Taken::Ended(_, false)));
        // Settled first: not taken.
        let handed = stamp();
        assert!(ahead.give(&lane, quick));
        assert!(matches!(ahead.take(handed, true), Taken::Ended(_, true)));
    }

    /// A sync ahead that is not taken has ended before the submission's
    /// own begins: they follow one another on the lane, one at a time.
    #[test]
    fn a_sync_ahead_not_taken_ends_before_the_own_sync_asks() {
        let lane = lane();
        let ahead = Ahead::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        {
            let order = order.clone();
            ahead.give(&lane, move |asked| {
                asked();
                std::thread::sleep(Duration::from_millis(20));
                order.lock().unwrap().push("ahead ends");
            });
        }
        ahead.wait_asked();
        let need = stamp();
        let Taken::Ended(waited, false) = ahead.take(need, false) else {
            panic!("taken");
        };
        assert!(waited > Duration::ZERO);
        let own = {
            let order = order.clone();
            beside::begin(&lane, move |asked| {
                order.lock().unwrap().push("own asks");
                asked();
            })
        };
        let _ = own.join();
        assert_eq!(*order.lock().unwrap(), ["ahead ends", "own asks"]);
    }

    /// Before a sweep the sync ahead ends, and is still taken after.
    #[test]
    fn a_sweep_waits_for_the_sync_ahead_which_is_still_taken() {
        let lane = lane();
        let ahead = Ahead::new();
        let ended = Arc::new(AtomicBool::new(false));
        let handed = stamp();
        {
            let ended = ended.clone();
            ahead.give(&lane, move |asked| {
                asked();
                std::thread::sleep(Duration::from_millis(20));
                ended.store(true, SeqCst);
            });
        }
        let _ = ahead.before_sweep();
        assert!(ended.load(SeqCst), "the sweep went on before the sync ahead ended");
        let Taken::Adopted(beside) = ahead.take(handed, false) else {
            panic!("not taken after the sweep");
        };
        let _ = beside.join();
        assert_eq!(ahead.before_sweep(), Duration::ZERO, "nothing pending");
    }

    /// A submission needs the highest stamp of its lists; a list of ours
    /// (no stamp) the title's latest call.
    #[test]
    fn a_submission_needs_its_lists_hand_overs_or_the_title_s_latest_call() {
        assert_eq!(need_with([3, 7, 5].into_iter(), 100), 7);
        assert_eq!(need_with([3, 0].into_iter(), 100), 100);
        assert_eq!(need_with([0].into_iter(), 0), 0);
        assert_eq!(need_with(std::iter::empty(), 100), 0, "nothing to read");
    }

    /// Belobug a609a2f: a wait passed raises its queue's floor, and the
    /// parts taken behind it need a sync asked after the pass; parts taken
    /// before keep theirs.
    #[test]
    fn a_part_behind_a_wait_passed_needs_a_sync_asked_after_the_pass() {
        let floor = AtomicU64::new(0);
        let (before, behind) = (AtomicU64::new(stamp()), AtomicU64::new(stamp()));
        // Nothing passed yet: taken as handed over.
        let was = before.load(SeqCst);
        apply(&floor, [&before].into_iter());
        assert_eq!(before.load(SeqCst), was);
        raise(&floor);
        let passed = floor.load(SeqCst);
        assert!(passed > behind.load(SeqCst));
        apply(&floor, [&behind].into_iter());
        assert_eq!(behind.load(SeqCst), passed);
        assert_eq!(before.load(SeqCst), was, "a part taken before the wait is not raised");
        // A list handed over again later keeps the higher stamp.
        let again = stamp();
        behind.fetch_max(again, SeqCst);
        apply(&floor, [&behind].into_iter());
        assert_eq!(behind.load(SeqCst), again);
    }

    /// The line's part: the join by phase, per frame of each phase.
    #[test]
    fn the_line_part_gives_the_join_by_phase() {
        let mut c = [0u64; 15];
        c[C::Started as usize] = 16;
        c[C::Adopted as usize] = 14;
        c[C::Late as usize] = 1;
        c[C::Settled as usize] = 1;
        c[C::WaitedUs as usize] = 4000;
        c[C::OnSubmissions as usize] = 18;
        c[C::OnJoinUs as usize] = 20_000;
        c[C::OnFrames as usize] = 2;
        c[C::OffSubmissions as usize] = 9;
        c[C::OffJoinUs as usize] = 23_000;
        c[C::OffFrames as usize] = 1;
        let text = said(&c, 2.0, "alt, ON");
        assert!(text.starts_with(" (VICTORY_HEAP_SYNC_AT_SUBMIT alt, ON: per frame 8.0 syncs started"), "{text}");
        assert!(text.contains("7.0 taken by it, 1.0 not (0.5 a hand-over"), "{text}");
        assert!(text.contains("2.00 ms waited for those not taken"), "{text}");
        assert!(text.contains("ON 10.00 ms a frame (+0.00 before) over 2 frames, off 23.00 ms a frame"), "{text}");
        let none = said(&[0; 15], 1.0, "on");
        assert!(none.contains("ON no submission, off no submission"), "{none}");
    }

    /// What newer titles pass to make a heap (as the descriptor tests do).
    #[repr(C)]
    struct Extended(super::super::HeapDesc, super::super::Placement);

    /// The slots a sampler heap's host heap shows after each step of a run
    /// of three submissions, synced the usual way (a sync at each
    /// submission's start) or with syncs started ahead; `None` without a
    /// host heap (no device).
    ///
    /// 1. The title copies a sampler's record into slot 5 as bytes and
    ///    hands the first submission over; its sync (ahead: started after
    ///    the hand-over, taken) is joined.
    /// 2. A sync ahead asks; then the title copies another record into
    ///    slot 6 and hands the second over: the sync ahead is not taken,
    ///    the submission's own sync runs after it.
    /// 3. The title copies the second sampler's record over slot 5 and
    ///    hands the third over; a sync ahead after it is taken.
    fn three_submissions(started_ahead: bool) -> Option<Vec<Vec<(u64, Option<usize>)>>> {
        use super::super::{HEAP_IIDS, HeapDesc, Placement, SAMPLERS, SHADER_VISIBLE, create_heap, create_sampler, heap};
        use crate::object::{S_OK, This, slot};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: a fresh allocation of the test's own.
        let memory = unsafe {
            VirtualAlloc(std::ptr::null(), crate::watch::PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        };
        assert!(!memory.is_null());
        let desc = HeapDesc { kind: SAMPLERS, count: 64, flags: SHADER_VISIBLE, nodes: 1 };
        let extended = Extended(desc, Placement { address: memory as u64, size: 64 * 16 });
        let mut h: This = std::ptr::null();
        assert_eq!(create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], &mut h), S_OK);
        let mut seen = None;
        if let Some(host) = heap(h).host() {
            // Two samplers' records, made by calls elsewhere.
            let mut elsewhere = [0u64; 4];
            let source = elsewhere.as_mut_ptr() as u64;
            create_sampler(std::ptr::null(), &[21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], source);
            create_sampler(std::ptr::null(), &[20, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], source + 16);
            let copy = |from: u64, to: usize| {
                // SAFETY: the heap slot and the source both hold 16 bytes.
                unsafe { std::ptr::copy_nonoverlapping(from as *const u8, memory.cast::<u8>().add(to * 16), 16) };
            };
            let lane = lane();
            let ahead = Ahead::new();
            let sync = |asked: &dyn Fn()| super::super::sync_heaps_told(Some(asked));
            let mut states = Vec::new();
            // 1.
            copy(source, 5);
            let handed = stamp();
            let first = if started_ahead {
                assert!(ahead.give(&lane, sync));
                match ahead.take(handed, false) {
                    Taken::Adopted(beside) => beside,
                    _ => panic!("a sync asked after the hand-over was not taken"),
                }
            } else {
                beside::begin(&lane, sync)
            };
            let _ = first.join();
            states.push(host.slots_seen(&[5, 6]));
            // 2.
            if started_ahead {
                assert!(ahead.give(&lane, sync));
                ahead.wait_asked();
            }
            copy(source, 6);
            let handed = stamp();
            if started_ahead {
                assert!(matches!(ahead.take(handed, false), Taken::Ended(_, false)), "a sync asked before the hand-over was taken");
            }
            let _ = beside::begin(&lane, sync).join();
            states.push(host.slots_seen(&[5, 6]));
            // 3.
            copy(source + 16, 5);
            let handed = stamp();
            let third = if started_ahead {
                assert!(ahead.give(&lane, sync));
                match ahead.take(handed, false) {
                    Taken::Adopted(beside) => beside,
                    _ => panic!("not taken"),
                }
            } else {
                beside::begin(&lane, sync)
            };
            let _ = third.join();
            states.push(host.slots_seen(&[5, 6]));
            seen = Some(states);
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
        seen
    }

    /// The host slots are the same, after each submission's join, with
    /// syncs started ahead as with a sync at each submission's start; and
    /// each submission's join finds what was handed over before it.
    #[test]
    fn syncs_started_ahead_leave_the_slots_as_the_usual_syncs() {
        let Some(usual) = three_submissions(false) else {
            eprintln!("no host heap (no device): the slots were not compared");
            return;
        };
        let Some(ahead) = three_submissions(true) else {
            return;
        };
        let key = |states: &[Vec<(u64, Option<usize>)>], step: usize, slot: usize| states[step][slot].0;
        assert_ne!(key(&usual, 0, 0), 0, "slot 5 not made by the first submission's sync");
        assert_ne!(key(&usual, 1, 1), 0, "slot 6 not made by the second's");
        assert_ne!(key(&usual, 2, 0), key(&usual, 0, 0), "slot 5's new record not made by the third's");
        assert_eq!(key(&usual, 2, 1), key(&usual, 1, 1));
        assert_eq!(usual, ahead);
    }
}
