//! VIcky: the value of the signal that begins a frame is published to the
//! title as soon as the submissions before it on its queue have been sent
//! to the card, not when the layer has landed them (the card done, what
//! their shaders wrote copied back) (`VICTORY_FENCE_SENT=1`; off by
//! default, and idle without submissions sent ahead, `crate::host::late`).
//! Every other signal is as before.
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
//!    its copy-back landed). The relay's merge (on by default) stores only
//!    the bytes the card changed (`crate::shadow`'s `merge_piece`), so the
//!    title's other bytes stay; a byte both wrote ends as the card's, the
//!    title's write lost (without the relay, `back_piece`, as the title's).
//!    `VICTORY_FENCE_SENT_CHECK=1` counts those bytes while a value is
//!    published ahead (the pace line; `VICTORY_MERGE_RACE=1` sees only title
//!    writes between a merge's read and its write). A submission prepared
//!    before the landing puts the title's bytes there into the copies
//!    beside the shaders' writes (late stage 2: "title bytes put in under
//!    flights' shader writes").
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

/// Diagnostics: whether the relay's merges count the title's bytes they put
/// the card's over while a value is published ahead
/// (`VICTORY_FENCE_SENT_CHECK=1`; one more pass over the merged bytes then,
/// on the landing thread).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var_os("VICTORY_FENCE_SENT_CHECK").is_some_and(|v| v == "1"))
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
