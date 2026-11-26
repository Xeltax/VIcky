//! VIcky: the value of the signal that begins a frame is published to the
//! title as soon as the submissions before it on its queue have been sent
//! to the card, not when the layer has landed them (the card done, what
//! their shaders wrote copied back) (`VICTORY_FENCE_SENT=1`; off by
//! default, and idle without submissions sent ahead, `crate::host::late`).
//! Every other signal is as before. While such a value is ahead, a byte
//! both the card and the title changed is the title's in the relay's
//! merges ("The title's writes after a value is published").
//!
//! # Why
//!
//! How the title paces itself (the fork's README, "How the title paces
//! itself", and [`super::pace`]): each frame begins with one small
//! submission and a `Signal` on a fence; a title thread then waits for that
//! value by reading the fence in memory (`monitorx` / `mwaitx`; the title
//! never asks for an event). With submissions sent ahead the signal goes
//! behind the flights in the air (`crate::host::late_signal`), so the value
//! is set when the layer has landed the whole frame before it. If the
//! title's recording of the next frame waits for it, one turn of the loop
//! spans two frames: the burst's preparing, the card, the landing and the
//! recording all add up. With the switch the title is told when the burst
//! has been sent, and the card and the landing run beside its recording.
//!
//! # Which signal begins a frame
//!
//! Run RT53c's pace lines (the seconds in which each event is said: 187
//! frames, 29 of them in the world) show one fence signalled about ten
//! times a world frame, its values one apart, and four other fences once or
//! twice a frame. In 25 of the 29 world frames the first signal the title
//! made after a present was on that fence, with the value after the frame
//! before's last one. In the other four another fence's signal came within
//! the log's 0.1 ms of it (in 12 of the 29 frames the second signal follows
//! the first within 0.2 ms: the title signals several fences together), or
//! the frame began before the seconds said. That first signal on the fence
//! was set 0 to 178 ms after the call (48 the median), in 21 of 26 frames
//! within 1 ms of the frame before's last signal on the same fence: in the
//! same landing, behind the whole frame before. In the 158 frames of the
//! menus and loading the fence is signalled twice a frame, and the first
//! signal after a present was on it every time.
//!
//! The title's wait itself cannot be watched on this PC: on an AMD
//! processor `mwaitx` runs natively, and the loader sees it only where it
//! emulates the instruction. So the signal is recognised by its place
//! ([`Begins`]): the first signal after a present on the queue and fence
//! that open the frames, learnt from the first signal after each present
//! (the same queue and fence two frames running to learn them, eight to
//! change them). A frame opened by another fence's signal (the ties above)
//! still has its first signal on the learnt fence taken; a queue and fence
//! that never open two frames running are never taken. Signals after that
//! one on the learnt fence, and every signal on other fences, are as
//! before.
//!
//! # What changes, and what does not
//!
//! - The frame-begin signal still goes behind the flights in the air and
//!   lands with them, in order (`crate::host::sent_signal`); only its value
//!   is published at once ([`super::set_early`]), right after the
//!   submissions before it were sent. With nothing in the air it is set at
//!   once, as before.
//! - A fence published ahead never goes back ([`Ahead`]): a landing below
//!   the published value (the frame before's own signals on that fence,
//!   still behind the flights) is noted and not stored; the landing of the
//!   published value, or of any above it, ends it.
//! - Our queues' waits for fences (a queue's `Wait`: the title's order
//!   between queues) go by the landed value ([`super::landed`]), so they
//!   pass exactly when they did. What the title reads (the fence's memory,
//!   `GetCompletedValue`, events) is the published value: every value up
//!   to it reads as reached, as a fence's value does. A list's wait for a
//!   memory value at a fence's address (none seen) reads memory, so it
//!   would pass at the published value; its part still goes to the card
//!   behind the signal's submission, the order such a wait asks for.
//!
//! # What the title may now do up to a frame early, and what stands against it
//!
//! Told that everything before the signal is done while up to a frame's
//! flights are still on the card or being landed, a title thread may:
//!
//! 1. read results of the frame before's work on the card: what its
//!    shaders wrote (in title memory only once they land), its
//!    bottom-of-pipe writes (timestamps, labels: written at landing), query
//!    results resolved by its lists. It then reads them a frame old.
//!    Nothing in the layer stands against this. The pace line counts what
//!    reached title memory while a published value was ahead (bytes copied
//!    back, bottom-of-pipe writes done): an upper bound of what could have
//!    been read stale (it cannot see the title's reads).
//! 2. write memory the frame before's shaders write (a buffer reused before
//!    its copy-back landed). While a value is published ahead, the relay's
//!    merges keep the title's byte where the card and the title both
//!    changed one, as the console's order has it: "The title's writes after
//!    a value is published", below.
//! 3. write memory the frame before's lists read (constants, upload rings)
//!    before those lists ran. Their copies were brought up when they were
//!    prepared; a later refresh of a copy in the PC's memory may reach a
//!    list still on the card, while copies in the card's memory are
//!    refreshed by copies on the queue, behind it. `crate::host::late`
//!    assumes this for memory a title rewrites without a wait; here it
//!    waited, but early.
//! 4. reset its lists and allocators, release objects, remap memory. Lists
//!    are taken from the title's when they are submitted
//!    (`crate::list::detach`); host objects released while flights run are
//!    kept until they have finished (`crate::host::retire`); page mappings
//!    (`CopyPageMappingsBatchX`) are done at once, so a copy-back that
//!    lands after a remap writes into the new pages.
//!
//! What the layer itself reads stays guarded as before: preparing a
//! submission that reads title memory a flight may write waits for that
//! flight to land (the title's buffer copies, textures: the settles), fills
//! of upload memory that do not wait are checked when the flights land
//! ("fills that read bytes a flight then changed", which must stay 0), and
//! a blocking fence wait of the title settles everything.
//!
//! # The title's writes after a value is published
//!
//! On the console the card's writes before a signal are done when its value
//! is reached, so a byte the title writes after it read the value is
//! written after them, and the title's byte stays. Here the frame before's
//! flights may still be on the card, or being landed, when the title is
//! told. The relay's merge decides per byte (`crate::shadow`'s
//! `merge_piece`) from what the copy holds now (`now`, the card's bytes),
//! the copy's reference (`held`: the copy's bytes when the flight was
//! prepared, with the title's bytes put in since by `merge_under_writes`
//! and what earlier merges took from the card) and title memory as just
//! read (`title`). A byte the card changed (`now` differs from `held`) went
//! into title memory whatever `title` held; every other byte stays the
//! title's.
//!
//! What run RT55b's check counted (about 8,900 bytes a frame) were bytes
//! the card AND the title changed since the reference, to different values:
//! its test was `now != held && title != held && title != now`. A byte only
//! the title changed was never among them, and the merge never puts the
//! card's over one (it takes nothing where `now` equals `held`). The title's
//! own writes since the reference are seen in two ways only: as `title`
//! differing from `held` at the merge, or put into the copy and its
//! reference when a submission prepared in between refreshes the range
//! ("title bytes put in under flights' shader writes", under 5 KB a frame
//! in RT55b). In both the card's byte won where a shader wrote it too
//! (after the putting in, for a copy in the PC's memory; a copy in the
//! card's memory leaves bytes put in out of the merge until the card has
//! them).
//!
//! The rule now, while a value is published ahead ([`keeping`], asked
//! after title memory was read): where the card and the title both changed
//! a byte, the title's stays, and the reference takes the card's (the copy
//! holds it; the next merge must not take it again). The flights merged
//! then are those before the published signal (and any behind it in the
//! same landing, merged with them), so the title has been told they are
//! done:
//!
//! - a title write after the value was published comes after the card's on
//!   the console: the title's byte, as there;
//! - a title write between the flight's preparing and the publishing (the
//!   two cannot be told apart here) is the title writing memory a shader of
//!   its own may be writing, without a wait: a race on the console too, and
//!   the title's byte is one the console could end with;
//! - it is the rule the waited copy-back has always had (`back_piece`, the
//!   title's byte where both changed: a shader's byte computed from older
//!   data once broke a title object).
//!
//! Copies in the card's memory merged with a wait (`card_merge`, without
//! the relay) take the same rule while a value is ahead.
//!
//! The title's bytes put into a copy in the PC's memory while a value is
//! ahead (`merge_under_writes`) go into the copy and its reference, so a
//! shader of the frame before that writes one afterwards would show as a
//! change of the card's alone: they are noted with the stretch's number
//! (`crate::fence::ahead_period`), and a merge in the same stretch keeps
//! the title's byte there too. In a copy in the card's memory they go up on
//! the queue behind the flights, and the merge skips them already. Left as
//! it is: such a byte is the card's for the next submission's shaders if a
//! shader of the frame before writes it after it was put into a copy in the
//! PC's memory (the copy is shared, and only a wait orders the two). A byte
//! another copy's merge put into title memory (two copies of the same
//! memory, both written by shaders) looks like the title's to this merge,
//! and stays as the other copy's.
//!
//! A merge stores its stretch, from the first byte the card changed to the
//! last, as it made it: the title's bytes in between as it read them, so a
//! title write that lands between that read and the store is put back.
//! `VICTORY_FENCE_SENT_KEEP_TITLE=1` ([`keep_exact`]) stores only the
//! 4-byte words the merge took a byte of the card's in (shaders store
//! words): a title byte in a word the card did not write is never stored.
//! Off by default: it is a store per run of such words instead of one per
//! stretch, which may cost time where the card's bytes are scattered.
//!
//! `VICTORY_FENCE_SENT_CHECK=1` counts, while a value is ahead (the pace
//! line): the bytes both changed (kept as the title's: what the check
//! counted before as put under the card's), and of the title writes made
//! between a merge's read and its store (title memory read once more just
//! before the store; a lower bound, a write during the store itself is not
//! seen), the bytes in words the merge took nothing of the card's in that
//! the store put back to an older value (title-only writes lost: 0 with
//! `VICTORY_FENCE_SENT_KEEP_TITLE=1`, which never stores such a word) and
//! those in words it took the card's bytes in (a race on the word, lost to
//! the card's).
//!
//! # The bytes kept as the title's go up at the landing
//!
//! A byte the rule keeps as the title's leaves title memory and the copy's
//! reference apart: the reference took the card's byte, title memory kept
//! the title's (about 10 KB a frame in RT60's check, scattered). The next
//! submission that refreshes the range while a flight may write it finds
//! each such byte again as a title write (`merge_under_writes`) and puts it
//! in. For a copy in the card's memory every run found there is a copy of
//! its own on the queue, 512 a submission at most (`VICTORY_MERGE_RUNS`):
//! past that the flights are settled ("title memory changed under shader
//! writes", 1.5 to 1.8 settles a frame with the rule against about 1.1
//! without it, 35 to 40 ms more of the submitting thread). For a copy in
//! the PC's memory they were only more "changed runs" (9,000 to 14,000 a
//! frame against 700 to 2,900).
//!
//! So the merge that keeps them puts them in itself ([`put_up`]), under the
//! lock `merge_under_writes` takes, with title memory as the merge read it:
//! what `merge_under_writes` does one submission later. In a copy in the
//! card's memory the runs go into the CPU's side and are noted to go up
//! before the next submission (joined over `VICTORY_MERGE_GAP`, as
//! `merge_under_writes` joins them); they lie outside what the merge left
//! out (runs still to go up, and those sent after its fetch), and are left
//! out of later merges themselves until the card has them, so a flight
//! still in the air cannot take the title's byte back. In a copy in the
//! PC's memory they go into the copy and its snapshot and are noted for the
//! stretch, as `merge_under_writes` notes what it puts in while a value is
//! ahead. The next `merge_under_writes` then finds title memory equal to the
//! reference there.
//!
//! What changes besides: the title's byte reaches the card at the next
//! send even if that submission does not refresh the range (before, it
//! waited for a refresh): the rule's own direction, the title wrote after
//! the card. A stretch with more than [`put_up_most`] such runs is left as
//! before. The frame-rate line's "title bytes put in under flights' shader
//! writes" counts the runs put up, the stretches left, and the settles
//! `merge_under_writes` asked for by cause.

use std::sync::{Mutex, OnceLock};

/// Whether the switch is on (`VICTORY_FENCE_SENT=1`), said once.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_FENCE_SENT").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "the signal that begins a frame is published to the title when the submissions before it are sent, and lands as before (VICTORY_FENCE_SENT=1)"
            );
        }
        on
    })
}

/// Diagnostics: whether the relay's merges count, while a value is
/// published ahead, the bytes both the card and the title changed and the
/// title writes their stores met (`VICTORY_FENCE_SENT_CHECK=1`; one more
/// pass over the merged bytes and one more read of each stretch stored,
/// on the landing thread). The module's docs, "The title's writes after a
/// value is published", say what each count is.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var_os("VICTORY_FENCE_SENT_CHECK").is_some_and(|v| v == "1"))
}

/// With the switch, while a value is published ahead of its landing: the
/// number of that stretch of time (`crate::fence::ahead_period`), and the
/// relay's merges keep the title's byte where the card and the title both
/// changed one (`crate::shadow`'s `Both::Title`). None otherwise, at the
/// cost of a look at a counter: the card's byte, as before.
pub(crate) fn keeping() -> Option<u64> {
    if !on() {
        return None;
    }
    super::ahead_period()
}

/// Whether, while a value is published ahead, the relay's merges store into
/// title memory only the 4-byte words they took a byte of the card's in
/// (`VICTORY_FENCE_SENT_KEEP_TITLE=1`, with the switch; off by default: a
/// store per run of such words, not one per stretch, which may cost time
/// where the card's bytes are scattered). Without it a merge stores its
/// whole stretch, the title's bytes as it read them included, and a title
/// write that lands between that read and the store is put back.
pub(crate) fn keep_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let exact = on() && std::env::var_os("VICTORY_FENCE_SENT_KEEP_TITLE").is_some_and(|v| v == "1");
        if exact {
            tracing::info!(
                "while a frame-begin value is published ahead, the relay's merges store only the words the card wrote into title memory (VICTORY_FENCE_SENT_KEEP_TITLE=1)"
            );
        }
        exact
    })
}

/// Which of the relay's merges, while a value is published ahead, put the
/// bytes they keep as the title's up at once ("The bytes kept as the
/// title's go up at the landing", in the module's docs): (those of copies
/// in the card's memory, those of copies in the PC's memory). Both with the
/// switch; `VICTORY_FENCE_SENT_PUT_UP=0` neither (as before: left to the
/// next preparing), `=card` the card's only.
pub(crate) fn put_up() -> (bool, bool) {
    static ON: OnceLock<(bool, bool)> = OnceLock::new();
    *ON.get_or_init(|| {
        if !on() {
            return (false, false);
        }
        let which = match std::env::var("VICTORY_FENCE_SENT_PUT_UP").ok().as_deref().map(str::trim) {
            Some("0") => (false, false),
            Some("card") => (true, false),
            _ => (true, true),
        };
        tracing::info!(
            "while a frame-begin value is published ahead, the relay's merges put the bytes they keep as the title's into the copy's reference at once \
             (copies in the card's memory {}, in the PC's memory {}; at most {} runs a stretch; VICTORY_FENCE_SENT_PUT_UP, VICTORY_FENCE_SENT_PUT_UP_RUNS)",
            which.0,
            which.1,
            put_up_most()
        );
        which
    })
}

/// The most runs of bytes kept as the title's one merged stretch puts up
/// ([`put_up`]); a stretch with more leaves them all to the next preparing,
/// as before (`VICTORY_FENCE_SENT_PUT_UP_RUNS`, 1 to 65536; 64 by default).
/// Each run put up in a copy in the card's memory is a copy of its own on
/// the queue before the next submission.
pub(crate) fn put_up_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_FENCE_SENT_PUT_UP_RUNS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(64, |n| n.clamp(1, 1 << 16))
    })
}

/// What a signal of the title's is to [`Begins`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Begin {
    /// The frame-begin signal: published when sent.
    Yes,
    /// The first signal after a present, not on the learnt queue and fence
    /// (or none is learnt yet): as before.
    First,
    /// Any other: as before.
    No,
}

/// Frames running a queue and fence must open to be learnt when none is.
const LEARN: u32 = 2;
/// And to replace the ones learnt.
const CHANGE: u32 = 8;

/// Which of the title's signals begins a frame (the module's docs say why
/// this way).
#[derive(Debug)]
pub(crate) struct Begins {
    /// A present was made and its frame's begin signal has not come yet.
    armed: bool,
    /// The first signal since the last present has come (true until the
    /// first present).
    opened: bool,
    /// The queue and fence of the first signal after the last presents, and
    /// in how many frames running.
    opener: Option<(usize, usize)>,
    streak: u32,
    /// The queue and fence taken as those the title paces itself on.
    pacing: Option<(usize, usize)>,
}

impl Begins {
    pub(crate) const fn new() -> Self {
        Begins { armed: false, opened: true, opener: None, streak: 0, pacing: None }
    }

    /// The title presented a frame: whether the frame before had no begin
    /// signal taken.
    pub(crate) fn presented(&mut self) -> bool {
        let missed = self.armed;
        self.armed = true;
        self.opened = false;
        missed
    }

    /// A signal of the title's on `queue` for `fence`.
    pub(crate) fn signalled(&mut self, queue: usize, fence: usize) -> Begin {
        let key = (queue, fence);
        let first = !self.opened;
        if first {
            self.opened = true;
            if self.opener == Some(key) {
                self.streak = self.streak.saturating_add(1);
            } else {
                self.opener = Some(key);
                self.streak = 1;
            }
            let needed = if self.pacing.is_none() { LEARN } else { CHANGE };
            if self.pacing != self.opener && self.streak >= needed {
                self.pacing = self.opener;
            }
        }
        if self.armed && self.pacing == Some(key) {
            self.armed = false;
            Begin::Yes
        } else if first {
            Begin::First
        } else {
            Begin::No
        }
    }
}

static BEGINS: Mutex<Begins> = Mutex::new(Begins::new());

fn begins() -> std::sync::MutexGuard<'static, Begins> {
    BEGINS.lock().unwrap_or_else(|e| e.into_inner())
}

/// With the switch: the title presented a frame.
pub(crate) fn presented() {
    if !on() {
        return;
    }
    let missed = begins().presented();
    #[cfg(all(windows, feature = "diagnostics"))]
    if missed {
        super::pace::begin(None);
    }
    #[cfg(not(all(windows, feature = "diagnostics")))]
    let _ = missed;
}

/// With the switch: whether the title's signal on `queue` for `fence` is
/// the frame-begin one. False without the switch, at no cost.
pub(crate) fn signalled(queue: usize, fence: usize) -> bool {
    if !on() {
        return false;
    }
    let begin = begins().signalled(queue, fence);
    #[cfg(all(windows, feature = "diagnostics"))]
    super::pace::begin(Some(begin));
    begin == Begin::Yes
}

/// A fence's value as published to the title and as landed, while a value
/// published ahead of its landing ([`super::set_early`]) has not landed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Ahead {
    /// The value published ahead; 0: none is (a value published is above
    /// the fence's, so never 0).
    published: u64,
    /// While one is: the value the fence's landings have got it to.
    landed: u64,
}

impl Ahead {
    /// `value` published now, the fence being at `now`: whether it is
    /// ahead (false: the fence is there already, nothing is published).
    pub(crate) fn publish(&mut self, now: u64, value: u64) -> bool {
        if value <= now {
            return false;
        }
        if self.published == 0 {
            self.landed = now;
        }
        self.published = value;
        true
    }

    /// A landing of `value`: whether the fence takes it. Below the value
    /// published it is noted, and the published value stays; at or above
    /// it the fence has caught up.
    pub(crate) fn land(&mut self, value: u64) -> bool {
        if self.published == 0 {
            return true;
        }
        if value < self.published {
            self.landed = value;
            return false;
        }
        *self = Ahead::default();
        true
    }

    /// What the landings got the fence to, the fence being at `now`.
    pub(crate) fn landed(&self, now: u64) -> u64 {
        if self.published == 0 { now } else { self.landed }
    }

    pub(crate) fn is_ahead(&self) -> bool {
        self.published != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: usize = 0x10;
    const F: usize = 0x100;
    const G: usize = 0x200;

    #[test]
    fn the_first_signal_after_a_present_on_the_learnt_fence_begins_a_frame() {
        let mut b = Begins::new();
        // Before any present nothing begins a frame.
        assert_eq!(b.signalled(Q, F), Begin::No);
        // The first frame only shows the opener; the second learns it.
        assert!(!b.presented());
        assert_eq!(b.signalled(Q, F), Begin::First);
        assert_eq!(b.signalled(Q, F), Begin::No);
        assert!(b.presented(), "nothing was taken in the first frame");
        assert_eq!(b.signalled(Q, F), Begin::Yes);
        // Later signals of the frame, on that fence and on others, are as before.
        assert_eq!(b.signalled(Q, F), Begin::No);
        assert_eq!(b.signalled(Q, G), Begin::No);
        assert!(!b.presented());
        assert_eq!(b.signalled(Q, F), Begin::Yes);
        assert_eq!(b.signalled(Q, F), Begin::No);
    }

    #[test]
    fn a_frame_opened_by_another_fence_still_takes_the_learnt_one() {
        let mut b = Begins::new();
        for _ in 0..2 {
            b.presented();
            b.signalled(Q, F);
        }
        // Another fence's signal comes first (a tie): not taken; the learnt
        // fence's first one is.
        b.presented();
        assert_eq!(b.signalled(Q, G), Begin::First);
        assert_eq!(b.signalled(Q, F), Begin::Yes);
        assert_eq!(b.signalled(Q, F), Begin::No);
        // A frame with no signal on the learnt fence takes nothing.
        b.presented();
        assert_eq!(b.signalled(Q, G), Begin::First);
        assert!(b.presented(), "that frame had none");
        assert_eq!(b.signalled(Q, F), Begin::Yes);
    }

    #[test]
    fn the_same_fence_on_another_queue_is_not_the_learnt_one() {
        let mut b = Begins::new();
        for _ in 0..2 {
            b.presented();
            b.signalled(Q, F);
        }
        b.presented();
        assert_eq!(b.signalled(Q + 8, F), Begin::First);
        assert_eq!(b.signalled(Q, F), Begin::Yes);
    }

    #[test]
    fn openers_that_alternate_are_never_learnt() {
        let mut b = Begins::new();
        for i in 0..20 {
            b.presented();
            let f = if i % 2 == 0 { F } else { G };
            assert_eq!(b.signalled(Q, f), Begin::First);
        }
    }

    #[test]
    fn the_learnt_fence_changes_only_after_eight_frames_opened_by_another() {
        let mut b = Begins::new();
        for _ in 0..2 {
            b.presented();
            b.signalled(Q, F);
        }
        // Seven frames opened by G, each with F after it: F is still taken.
        for _ in 0..CHANGE - 1 {
            b.presented();
            assert_eq!(b.signalled(Q, G), Begin::First);
            assert_eq!(b.signalled(Q, F), Begin::Yes);
        }
        // The eighth: G is learnt, and taken at once.
        b.presented();
        assert_eq!(b.signalled(Q, G), Begin::Yes);
        assert_eq!(b.signalled(Q, F), Begin::No);
        b.presented();
        assert_eq!(b.signalled(Q, F), Begin::First);
        assert_eq!(b.signalled(Q, G), Begin::Yes);
    }

    #[test]
    fn a_published_value_never_goes_back_and_lands_in_order() {
        let mut a = Ahead::default();
        // The fence is at 26; 27 (the frame-begin value) is published ahead.
        assert!(a.publish(26, 27));
        assert!(a.is_ahead());
        assert_eq!(a.landed(27), 26, "our queues' waits still see the landed value");
        // The frame before's own signals land below it: noted, not taken.
        assert!(!a.land(25));
        assert_eq!(a.landed(27), 25);
        assert!(!a.land(26));
        assert_eq!(a.landed(27), 26);
        // Its own landing ends it.
        assert!(a.land(27));
        assert!(!a.is_ahead());
        assert_eq!(a.landed(27), 27);
        // Nothing ahead: every landing is taken, as before.
        assert!(a.land(28));
        assert!(a.land(3), "without a value ahead a fence takes what it is given");
    }

    #[test]
    fn publishing_what_the_fence_holds_already_does_nothing() {
        let mut a = Ahead::default();
        assert!(!a.publish(30, 30));
        assert!(!a.publish(30, 29));
        assert!(!a.is_ahead());
        assert_eq!(a.landed(30), 30);
    }

    #[test]
    fn a_second_value_published_before_the_first_landed_keeps_the_landed_one() {
        let mut a = Ahead::default();
        assert!(a.publish(26, 27));
        // The next frame's begin value, published while 27 is still behind
        // flights (the fence reads 27).
        assert!(a.publish(27, 37));
        assert_eq!(a.landed(37), 26);
        assert!(!a.land(27), "27 lands below the value published");
        assert_eq!(a.landed(37), 27);
        assert!(!a.land(36));
        // A landing above the published value ends it too.
        assert!(a.land(38));
        assert!(!a.is_ahead());
    }
}
