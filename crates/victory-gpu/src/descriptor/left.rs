//! VIcky, stage 1 of the structural fps design (`VICTORY_TABLE_BOUND=1|alt`;
//! off by default): the tables draws make afresh mark their title slots
//! wherever they lie in the title's heap, and the counters that decide
//! whether stage 2 (buffer slots left to the draws) is worth doing.
//!
//! The title builds each draw's tables in its view heap by copying records
//! there as bytes, every frame. A draw makes every entry of a table it makes
//! afresh in dynamic slots of its own (`crate::draw::State::tables`), and
//! the title slots' host slots are read only by shaders that pick views by
//! an index they compute. The tree already had marks for such slots
//! (`HostHeap::tabled`, 240 frames): a texture slot a draw read through a
//! table leaves the queue of textures picked by index, a write of it waits
//! for its draw, and the heap sync leaves a new record there to the draws.
//! But the first title slot of a table made afresh was tested against
//! `HostHeap::count`, which with the heap map is the 131,072-slot identity
//! window, and every table made afresh lies past it: the marks were dead
//! ("0 writes of tables' slots" and "0 left to the draws" for a whole boat
//! run, Y2). With the switch the test is against `HostHeap::mirrored`, the
//! title slots the host heap mirrors ([`title_first`]). This is the fix of
//! the dropped commit 7b66261e (`VICTORY_TABLES_LEFT=textures`) on its own.
//!
//! `alt` (twenty seconds off, twenty on, the `=alt` lines
//! `xcl-notes/alt-fps.sh` reads): draws set the marks in both phases; in the
//! off phase nothing acts on them ([`live`]: `HostHeap::tabled_lately`
//! answers no, a read takes no slot out of the queue, a write waits for no
//! draw, the draw makes no title slot in place). At each flip to off the
//! heap sync sweeps every slot with a live mark once: what the sync left
//! there is mirrored as it would have been, and a texture its draw made
//! already goes back in the queue, so the off phase starts from what it
//! would have had (`HostHeap::table_bound_flip`; one slow frame per flip).
//! The tree's own marks (tables read in place, and tables made afresh
//! wholly in the window) are off in the off phase too: they are the dead
//! ones above, so the off phase is main in practice.
//!
//! The counters (one line beside the heap sync line, [`line`]), always on
//! but where said:
//! - where the tables made afresh lie ([`Tally::first`]);
//! - the title slots marked: texture marks, and with the switch buffer
//!   marks (a second array, after myokex c3723f3 as 7b66261 had it; set in
//!   both phases and only counted, nothing reads them yet): every entry of
//!   a table made afresh of up to [`NAMED_ABOVE`] slots, else the entries
//!   the shader names, and every entry read through a table read in place;
//! - the heap sync's new records by rule (the order of avatarmanz's
//!   `sync_heaps`, a9400a6d6): left to the draws by a texture mark; would be
//!   left by a buffer mark of the last [`FRAMES`] frames; would be held by
//!   stage 2's grace (a buffer or placed record elsewhere); no view into a
//!   slot that shows none; no view into a slot that shows one; texture,
//!   buffer and other views made;
//! - the slots of the pass over buffer views, and those carrying a buffer
//!   mark;
//! - window copies let go and made again (`crate::shadow::window_churn`),
//!   and in diagnostics builds with the switch, of the windows alive, those
//!   held only by slots or by the pass's kept ranges
//!   (`crate::shadow::window_holders`).
//!
//! Stage 2's stop criterion, read from the off phases on the boat: the
//! records a buffer mark would leave plus those the grace would hold under
//! 50 % of the records mirrored, and the pass's slots with a buffer mark
//! under 50 % of its slots: stop (the line gives both shares).
//!
//! `VICTORY_TABLE_BOUND_FRESH` ([`fresh`]; on by default with the switch,
//! `=0` off): the texture slots the draws make keep the freshness rules of
//! the slots shaders pick by index (the streamed-texture staleness fixes):
//! only a read leaves a slot to the draws, and memory events still find and
//! make due the slots draws read, whose staged textures are read again
//! after a write behind the guards and held under a DirectStorage write.
//!
//! Stage 2 (`VICTORY_TABLES_LEFT=1|alt|check`, [`mode2`]; off by default;
//! it sets stage 1's bound and marks too, [`set`]): the buffer and constant
//! views in title slots draws read through their tables are left to those
//! draws, which make what they read in dynamic slots of their own. A re-land
//! of the dropped 7b66261e on this tree (after avatarmanz's VIctory-7900XTX
//! 1f1a433676 / a9400a6d6, carried in cool_fox's tree, and myokex c3723f3's
//! buffer marks), with what that commit lacked:
//! - the heap sync leaves a new buffer or constant view record in a slot
//!   with a buffer mark of the last [`FRAMES`] frames, and notes it in the
//!   slot's left bit ([`HeapLeft::bits`]); the pass over buffer views leaves
//!   such slots out and drops the ranges it kept for them;
//! - with `VICTORY_TABLE_BOUND_FRESH` (on by default) only the entries a
//!   draw reads get buffer marks, so only read slots are left;
//! - the hold ([`Holds`]): the host copy a draw's table reads through a
//!   marked slot is held by that slot, in place of the host slot and the
//!   pass's kept range that held it without the switch, until the mark is
//!   [`HOLD_FRAMES`] frames old or the title writes another record there.
//!   Without it the copies were let go once idle and made again (T1:
//!   copies +30 ms while the join fell 15 ms and the pass's wait 25 ms);
//! - `alt`: twenty seconds off, twenty on; at each flip to off every slot
//!   left and not noted since is mirrored as the sync would have, what
//!   waits its grace is mirrored, and the holds go, so the off phase starts
//!   from what it would have had; in ON phases the slots whose mark expired
//!   are healed the same way every [`HEAL_EVERY`] frames;
//! - `check`: everything is mirrored and made as without the switch, and
//!   at each submission every slot its lists read by an index they compute
//!   is checked against what the switch would have left there
//!   ([`check_submission`]); violations are counted and the first named;
//! - one summary per phase, ON against off: records mirrored, the join, the
//!   pass and the submitting thread's wait for it, copies, copies let go and
//!   made again, holds ([`period_line`]).
//!
//! `VICTORY_TABLES_LEFT_GRACE=1` (off by default): a new constant or placed
//! buffer view record in a slot without a mark waits [`GRACE`] frames for
//! a draw's table to claim it before it is mirrored (the design's rule 2).
//! A shader that picks such a buffer by index sees it that much later.
//! `VICTORY_TABLES_LEFT_HOLD=0`: no holds, to tell whether they matter.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use crate::shadow::Shadow;

/// The switch (`VICTORY_TABLE_BOUND=1|alt`).
pub(crate) static TABLE_BOUND: crate::alt::Alt = crate::alt::Alt::new(
    "VICTORY_TABLE_BOUND",
    "tables made afresh mark their title slots wherever they lie in the title's heap, not only in the identity window",
);

#[cfg(test)]
thread_local! {
    /// Tests: (set, live) for this thread, whatever the environment says.
    pub(crate) static TEST_BOUND: Cell<Option<(bool, bool)>> = const { Cell::new(None) };
}

/// Whether the switch is set (`1` or `alt`), or stage 2's
/// (`VICTORY_TABLES_LEFT`, which builds on the bound and the marks): draws
/// test a table made afresh against the title slots the host heap mirrors,
/// and set buffer marks, in both phases.
pub(crate) fn set() -> bool {
    #[cfg(test)]
    if let Some((set, _)) = TEST_BOUND.get() {
        return set;
    }
    TABLE_BOUND.set() || left_set()
}

/// With `alt`, the phase as the last heap sync found it ([`tick`]).
static LIVE: AtomicBool = AtomicBool::new(false);

/// Whether the marks of tables' slots act now: always without the switch
/// and with `1`; with `alt`, in its ON phase.
pub(crate) fn live() -> bool {
    #[cfg(test)]
    if let Some((_, live)) = TEST_BOUND.get() {
        return live;
    }
    TABLE_BOUND.mode() != 2 || LIVE.load(Relaxed)
}

#[cfg(test)]
thread_local! {
    /// Tests: whether `VICTORY_TABLE_BOUND_FRESH` is on for this thread,
    /// whatever the environment says (with the switch set).
    pub(crate) static TEST_FRESH: Cell<Option<bool>> = const { Cell::new(None) };
}

/// VIcky (`VICTORY_TABLE_BOUND_FRESH`; on by default whenever
/// `VICTORY_TABLE_BOUND` is set, `=0` turns it off): the texture slots the
/// draws make keep the freshness rules of the slots shaders pick by index
/// (VICKY-RT.md, stage 1, "Table-made texture slots kept fresh").
///
/// Stage 1 as first measured took a slot a table merely spanned out of
/// every rule for 240 frames, read or not: the heap sync left its new
/// records, its writes waited for a draw, its placed memory left the spans,
/// and the passes over memory mapped, written or waiting skipped it. A slot
/// a draw names is made by that draw, and its texture is kept up to date by
/// the draw's staging buffer, which is read again at every submission that
/// samples it. A slot no draw names (a gap of a table, an entry of a
/// descriptor array the shader indexes from data, a slot picked by index
/// that a table's range lies over) is made by nobody: its texture kept
/// what it showed (a recycled block's last texture, zeros) and the
/// streamed-texture fixes no longer reached it. With this:
/// - only a read leaves a slot to the draws (`HostHeap::tabled_in`);
/// - a slot read through a table keeps its placed memory in the spans, and
///   the passes over memory remapped, written behind the guards, or waiting
///   for levels make it due as any slot (the pass then makes it from fresh
///   bytes, or finds the draws' staging buffer and leaves it to them);
/// - a write behind the guards under a texture draws keep staged has that
///   staging buffer read and uploaded again whole (`crate::texture::restage_in`);
/// - a draw's upload, and the check at submission, wait while a
///   DirectStorage write is under way into the texture with
///   `VICTORY_DS_HOLD=1` (`crate::texture::upload`, `Staging::refresh_timed`);
/// - a staging buffer's memory is hashed at least every eight frames, by its
///   own count, whatever step the frame counter takes.
///
/// Off without `VICTORY_TABLE_BOUND`; with `alt` it acts where the marks
/// act, in the ON phases ([`fresh_live`]), but for the first rule, which
/// keeps the marks the off phase's flip sweeps to the ones that acted.
pub(crate) fn fresh() -> bool {
    if !set() {
        return false;
    }
    #[cfg(test)]
    if let Some(fresh) = TEST_FRESH.get() {
        return fresh;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_TABLE_BOUND_FRESH").is_none_or(|v| v != "0");
        tracing::info!(
            on,
            "texture slots draws make through tables keep the freshness rules of the slots shaders pick by index \
             (VICTORY_TABLE_BOUND_FRESH, on by default with VICTORY_TABLE_BOUND; =0 is stage 1 as first measured)"
        );
        on
    })
}

/// [`fresh`], where the marks act now ([`live`]).
pub(crate) fn fresh_live() -> bool {
    fresh() && live()
}

/// At the start of each heap sync: with `alt`, the phase of now (each change
/// said in the log, `crate::alt::Alt::on`).
pub(crate) fn tick() {
    if TABLE_BOUND.mode() == 2 {
        LIVE.store(TABLE_BOUND.on(), Relaxed);
    }
    // Stage 2 (`VICTORY_TABLES_LEFT=alt`): its phase, and its summary of a
    // phase that ended.
    if mode2() == Mode2::Alt {
        LEFT_LIVE.store(TABLES_LEFT.on(), Relaxed);
    }
    period_tick();
}

/// The first title slot of a table made afresh of `n` slots whose marks are
/// set (`first`: where it lies in the heap, if it does): the table must lie
/// wholly in the host heap's identity window (`window`, `HostHeap::count`),
/// or with the switch in the title slots the host heap mirrors
/// (`mirrored`, `HostHeap::mirrored`).
pub(crate) fn title_first(first: Option<u32>, n: u32, window: u32, mirrored: u32) -> Option<u32> {
    let bound = if set() { mirrored } else { window };
    first.filter(|&first| u64::from(first) + u64::from(n) <= u64::from(bound))
}

/// Frames for which a buffer mark would leave its slot to the draws (stage
/// 2; here only counted).
pub(crate) const FRAMES: u32 = 8;

/// Of a table made afresh longer than this, only the entries the shader
/// names get buffer marks (a big table may lie over slots shaders pick by
/// index).
pub(crate) const NAMED_ABOVE: u32 = 64;

/// A buffer mark up to this many frames ahead of the frame asked about (a
/// draw of a frame begun after the asker read the frame) counts too.
const AHEAD: u32 = 64;

/// The top bit of a mark, kept free as in `HostHeap::tabled`.
const TOP: u32 = 1 << 31;

/// Marks `stamp` as set in `frame`: the frame plus one in 31 bits (written
/// only if it changes: every draw marks its tables' slots).
pub(crate) fn mark(stamp: &AtomicU32, frame: u64) {
    let now = (frame as u32).wrapping_add(1) & !TOP;
    if stamp.load(Relaxed) != now {
        stamp.store(now, Relaxed);
    }
}

/// Whether a buffer mark `stamp` (as [`mark`] sets them; 0 for none) is live
/// at `frame`: set within the last [`FRAMES`] frames, or a little ahead.
pub(crate) fn lately(stamp: u32, frame: u64) -> bool {
    within(stamp, frame, FRAMES)
}

/// Whether a buffer mark `stamp` was set within the last `frames` frames
/// before `frame`, or a little ahead.
pub(crate) fn within(stamp: u32, frame: u64, frames: u32) -> bool {
    let stamp = stamp & !TOP;
    if stamp == 0 {
        return false;
    }
    let now = (frame as u32).wrapping_add(1) & !TOP;
    let since = now.wrapping_sub(stamp) & !TOP;
    let ahead = stamp.wrapping_sub(now) & !TOP;
    since <= frames || ahead <= AHEAD
}

/// Whether entry `k` of a table of `n` entries made afresh gets a buffer
/// mark: every entry of a table of up to [`NAMED_ABOVE`], else those `named`.
pub(crate) fn marked(n: u32, k: u32, named: impl FnOnce(u32) -> bool) -> bool {
    n <= NAMED_ABOVE || named(k)
}

/// What the line counts ([`count`]).
#[derive(Clone, Copy, Debug)]
pub(crate) enum C {
    /// Tables made afresh whose `n` slots from their first title slot lie:
    /// wholly in the identity window; past it in the title slots the host
    /// heap mirrors; past those; not in the heap.
    FirstInWindow,
    FirstPastWindow,
    FirstPastTitle,
    FirstNoIndex,
    /// Texture marks set (tables made afresh and tables read in place);
    /// buffer marks set by tables made afresh, and by entries read in place.
    MarkedTextures,
    MarkedFresh,
    MarkedInPlace,
    /// The heap sync's new records it made, by rule (the left ones are
    /// `super::host::COPIED_TABLED`): a buffer mark would leave it; stage 2's
    /// grace would hold it; no view into a slot that shows none; no view into
    /// a slot that shows one; texture views, buffer views, other views.
    WouldLeave,
    WouldHold,
    NoViewNone,
    NoViewShown,
    TexturesMade,
    BuffersMade,
    OthersMade,
    /// Slots the pass over buffer views was given, and of them those with a
    /// buffer mark.
    PassSlots,
    PassMarked,
    /// VIcky (`VICTORY_TABLE_BOUND_FRESH`): of the texture marks, those of
    /// entries the draw reads (the shader names them, or picks them by a
    /// root constant); the others only lie in a table's range.
    MarkedTexturesRead,
}

const COUNTS: usize = C::MarkedTexturesRead as usize + 1;

static COUNTED: [AtomicU64; COUNTS] = [const { AtomicU64::new(0) }; COUNTS];

#[cfg(test)]
thread_local! {
    /// Tests: with `Some`, what this thread counts is added here too (the
    /// line's counts are shared by the tests running at once).
    pub(crate) static TEST_COUNTS: std::cell::RefCell<Option<[u64; COUNTS]>> = const { std::cell::RefCell::new(None) };
}

/// Counts `n` of `what` for the line.
pub(crate) fn count(what: C, n: u64) {
    count_at(what as usize, n);
}

/// [`count`] of the count at `at`.
fn count_at(at: usize, n: u64) {
    if n > 0 {
        COUNTED[at].fetch_add(n, Relaxed);
        #[cfg(test)]
        TEST_COUNTS.with_borrow_mut(|counts| {
            if let Some(counts) = counts {
                counts[at] += n;
            }
        });
    }
}

/// A draw's counts, added to its thread's when it goes, and those to the
/// line's every [`TALLY_EVERY`] draws: the title's recording threads would
/// otherwise share each count's cache line, table by table.
#[derive(Default)]
pub(crate) struct Tally([u64; COUNTS]);

/// Draws a recording thread's tallies are added up over before they go to
/// the line's counts (a line may get some of the draws of the second
/// before it, and miss some of its own).
const TALLY_EVERY: u32 = 64;

thread_local! {
    /// This thread's tallies not yet added to the line's, and how many.
    static DRAW_TALLY: std::cell::RefCell<([u64; COUNTS], u32)> = const { std::cell::RefCell::new(([0; COUNTS], 0)) };
}

impl Tally {
    /// Counts `n` of `what`.
    pub(crate) fn add(&mut self, what: C, n: u64) {
        self.0[what as usize] += n;
    }

    /// [`count_first`], into the tally.
    pub(crate) fn first(&mut self, first: Option<u32>, n: u32, window: u32, mirrored: u32) {
        self.add(where_first(first, n, window, mirrored), 1);
    }
}

impl Drop for Tally {
    fn drop(&mut self) {
        let due = DRAW_TALLY.with_borrow_mut(|(sum, n)| {
            for (sum, &add) in sum.iter_mut().zip(&self.0) {
                *sum += add;
            }
            *n += 1;
            (*n >= TALLY_EVERY).then(|| {
                *n = 0;
                std::mem::take(sum)
            })
        });
        for (at, n) in due.into_iter().flatten().enumerate() {
            count_at(at, n);
        }
    }
}

/// Where a table made afresh lies (`first`: its first title slot, `None` if
/// it is not in the heap; `n` its slots; `window`, the identity window;
/// `mirrored`, the title slots the host heap mirrors), as [`Tally::first`]
/// counts it.
fn where_first(first: Option<u32>, n: u32, window: u32, mirrored: u32) -> C {
    match first.map(|first| u64::from(first) + u64::from(n)) {
        None => C::FirstNoIndex,
        Some(end) if end <= u64::from(window) => C::FirstInWindow,
        Some(end) if end <= u64::from(mirrored) => C::FirstPastWindow,
        Some(_) => C::FirstPastTitle,
    }
}

thread_local! {
    /// While this thread makes the slots the heap sync found new
    /// ([`SyncMaking`]): their makings are counted by rule.
    static SYNC_MAKING: Cell<bool> = const { Cell::new(false) };
    /// What this thread counted of them ([`count_sync`]), added to the
    /// line's counts when its outermost [`SyncMaking`] goes: some 80,000 a
    /// frame on the boat, on the sync's threads, which would otherwise share
    /// each count's cache line slot by slot.
    static SYNC_TALLY: std::cell::RefCell<[u64; COUNTS]> = const { std::cell::RefCell::new([0; COUNTS]) };
}

/// While alive, the slots this thread makes are the heap sync's, counted
/// by rule (`HostHeap::count_sync_rule`).
pub(crate) struct SyncMaking(bool);

impl SyncMaking {
    pub(crate) fn enter() -> Self {
        SyncMaking(SYNC_MAKING.replace(true))
    }
}

impl Drop for SyncMaking {
    fn drop(&mut self) {
        SYNC_MAKING.set(self.0);
        if !self.0 {
            let tally = SYNC_TALLY.with_borrow_mut(std::mem::take);
            for (at, n) in tally.into_iter().enumerate() {
                count_at(at, n);
            }
        }
    }
}

/// Whether this thread makes the heap sync's slots now ([`SyncMaking`]).
pub(crate) fn in_sync_making() -> bool {
    SYNC_MAKING.get()
}

/// Counts one of the heap sync's slots as `what`, on this thread until its
/// [`SyncMaking`] goes.
pub(crate) fn count_sync(what: C) {
    SYNC_TALLY.with_borrow_mut(|tally| tally[what as usize] += 1);
}

/// The window copies let go and made again at the last line
/// (`crate::shadow::window_churn`, since the start).
static CHURN_SEEN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Diagnostics: the last census of the windows' holders
/// (`crate::shadow::window_holders`) and when it was taken
/// (`crate::host::now_ms`; 0: never).
#[cfg(feature = "diagnostics")]
static CENSUS: std::sync::Mutex<(u64, [u64; 6])> = std::sync::Mutex::new((0, [0; 6]));

/// Diagnostics: a census of the windows' holders at most this often.
#[cfg(feature = "diagnostics")]
const CENSUS_MS: u64 = 5000;

/// Diagnostics, with the switch: the census part of the line; a new census
/// once [`CENSUS_MS`] have passed and the pass over buffer views lets go of
/// its kept ranges for a moment (`crate::draw::heap_pass::kept_copies`).
#[cfg(feature = "diagnostics")]
fn census_part() -> String {
    if !set() {
        return String::new();
    }
    let now = crate::host::now_ms();
    let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    if now >= census.0.saturating_add(CENSUS_MS)
        && let Some(kept) = crate::draw::heap_pass::kept_copies()
    {
        *census = (now, crate::shadow::window_holders(&kept));
    }
    let (at, [alive, listed, slots, ranges, both, idle]) = *census;
    if at == 0 {
        return "; window copies by holder: no census yet (the pass over buffer views held its kept ranges)".to_owned();
    }
    format!(
        "; window copies by holder ({:.0} s ago): {alive} alive, {listed} held by a list or a comparison, of the others \
         {slots} held only by slots, {ranges} only by the pass's kept ranges, {both} by both, {idle} by neither \
         (let go once idle)",
        now.saturating_sub(at) as f64 / 1000.0
    )
}

/// With the heap sync line: per frame over `frames`, the stage 1 counters.
/// `mirrored`: records the heap sync mirrored per frame; `left`: records it
/// left to the draws per frame (`super::host::COPIED_TABLED`). Said whenever
/// the switch is set or a table was made afresh, so that a run without the
/// switch says where its tables lie.
pub(crate) fn line(frames: f64, mirrored: f64, left: f64) {
    let c = COUNTED.each_ref().map(|c| c.swap(0, Relaxed));
    let get = |what: C| c[what as usize];
    let per = |what: C| get(what) as f64 / frames;
    let churn = crate::shadow::window_churn();
    let [went, remade] = [0, 1].map(|i| churn[i].saturating_sub(CHURN_SEEN[i].swap(churn[i], Relaxed)) as f64 / frames);
    let tables = get(C::FirstInWindow) + get(C::FirstPastWindow) + get(C::FirstPastTitle) + get(C::FirstNoIndex);
    if !set() && tables == 0 {
        return;
    }
    let made: u64 = [
        C::WouldLeave,
        C::WouldHold,
        C::NoViewNone,
        C::NoViewShown,
        C::TexturesMade,
        C::BuffersMade,
        C::OthersMade,
    ]
    .iter()
    .map(|&what| get(what))
    .sum();
    let share = |part: u64, of: u64| if of == 0 { 0.0 } else { 100.0 * part as f64 / of as f64 };
    let buffer_marks = if set() {
        format!(
            "{:.0} buffer marks by tables made afresh, {:.0} by entries read in place",
            per(C::MarkedFresh),
            per(C::MarkedInPlace)
        )
    } else {
        "no buffer marks (VICTORY_TABLE_BOUND unset)".to_owned()
    };
    #[cfg(feature = "diagnostics")]
    let census = census_part();
    #[cfg(not(feature = "diagnostics"))]
    let census = String::new();
    // VIcky (`VICTORY_TABLE_BOUND_FRESH`): only the read ones leave a slot
    // to the draws.
    let read = format!(
        "{:.0} of them read{}",
        per(C::MarkedTexturesRead),
        if fresh() { ", the only ones that leave a slot to the draws (FRESH)" } else { "" }
    );
    tracing::info!(
        "tables and the heap sync (VICTORY_TABLE_BOUND {}) per frame: {:.0} tables made afresh, of them {:.0} lie in the \
         identity window, {:.0} reach past it into the title slots the host heap mirrors ({}), {:.1} reach past those, \
         {:.1} lie outside the heap; title slots marked: {:.0} texture marks ({read}), {buffer_marks}; the sync's new records by \
         rule: {left:.0} left to the draws by a texture mark, of the {mirrored:.0} mirrored ({:.0} counted by rule) \
         {:.0} a buffer mark would leave, {:.0} the grace would hold, {:.0} no view into a slot that shows none, {:.0} \
         no view into a slot that shows one, {:.0} texture views, {:.0} buffer views and {:.0} other views made (would \
         leave or hold: {:.0} % of those counted); the pass over buffer views: {:.0} slots, {:.0} of them with a buffer \
         mark ({:.0} %); window copies: {went:.2} let go, {remade:.2} made again{census}",
        TABLE_BOUND.said(),
        tables as f64 / frames,
        per(C::FirstInWindow),
        per(C::FirstPastWindow),
        if set() { "marked" } else { "not marked: VICTORY_TABLE_BOUND unset" },
        per(C::FirstPastTitle),
        per(C::FirstNoIndex),
        per(C::MarkedTextures),
        made as f64 / frames,
        per(C::WouldLeave),
        per(C::WouldHold),
        per(C::NoViewNone),
        per(C::NoViewShown),
        per(C::TexturesMade),
        per(C::BuffersMade),
        per(C::OthersMade),
        share(get(C::WouldLeave) + get(C::WouldHold), made),
        per(C::PassSlots),
        per(C::PassMarked),
        share(get(C::PassMarked), get(C::PassSlots)),
    );
}

// ---------------------------------------------------------------------------
// Stage 2: `VICTORY_TABLES_LEFT=1|alt|check`.
// ---------------------------------------------------------------------------

/// The switch of stage 2. Its `alt` phases are stage 1's when both are
/// `alt` (both stages against neither); else its own.
pub(crate) static TABLES_LEFT: crate::alt::Alt = crate::alt::Alt::following(
    "VICTORY_TABLES_LEFT",
    "the buffer and constant views in the title slots draws read through tables are left to those draws (structural stage 2)",
    &TABLE_BOUND,
);

/// What `VICTORY_TABLES_LEFT` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode2 {
    /// Unset or `0`: as before.
    Off,
    /// `1`: on.
    On,
    /// `alt`: twenty seconds off, twenty on.
    Alt,
    /// `check`: everything mirrored and made as without the switch; what the
    /// switch would have left is modelled and checked at each submission.
    Check,
}

#[cfg(test)]
thread_local! {
    /// Tests: (mode, live, grace, hold) for this thread, whatever the
    /// environment says.
    pub(crate) static TEST_LEFT: Cell<Option<(Mode2, bool, bool, bool)>> = const { Cell::new(None) };
}

/// `VICTORY_TABLES_LEFT=check`, read once.
fn check_asked() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_TABLES_LEFT").is_ok_and(|v| v.trim() == "check");
        if on {
            tracing::info!(
                "VICTORY_TABLES_LEFT=check: everything is mirrored and made as without the switch; what it would have left \
                 is modelled, and each submission checks the slots its lists read by index against it"
            );
        }
        on
    })
}

/// Stage 2's mode.
pub(crate) fn mode2() -> Mode2 {
    #[cfg(test)]
    if let Some((mode, ..)) = TEST_LEFT.get() {
        return mode;
    }
    if check_asked() {
        return Mode2::Check;
    }
    match TABLES_LEFT.mode() {
        1 => Mode2::On,
        2 => Mode2::Alt,
        _ => Mode2::Off,
    }
}

/// Whether stage 2's switch is set at all.
pub(crate) fn left_set() -> bool {
    mode2() != Mode2::Off
}

/// With `alt`, stage 2's phase as the last heap sync found it ([`tick`]).
static LEFT_LIVE: AtomicBool = AtomicBool::new(false);

/// Whether stage 2 leaves slots to the draws now: with `1`, and in the ON
/// phases of `alt`; never with `check` (which mirrors everything).
pub(crate) fn left_live() -> bool {
    #[cfg(test)]
    if let Some((_, live, ..)) = TEST_LEFT.get() {
        return live;
    }
    match mode2() {
        Mode2::On => true,
        Mode2::Alt => LEFT_LIVE.load(Relaxed),
        Mode2::Off | Mode2::Check => false,
    }
}

/// Whether `VICTORY_TABLES_LEFT=check`.
pub(crate) fn checking() -> bool {
    mode2() == Mode2::Check
}

/// Whether the sync and the pass decide by stage 2's rules now: when it
/// leaves slots, and with `check` (which then makes them all the same).
pub(crate) fn acting() -> bool {
    left_live() || checking()
}

/// Whether a table made afresh gives buffer marks only to the entries its
/// draw reads (stage 2 with `VICTORY_TABLE_BOUND_FRESH`: only read slots
/// are left to the draws); else stage 1's rule ([`marked`]).
pub(crate) fn marks_read_only() -> bool {
    left_set() && fresh()
}

/// Frames a new constant or placed buffer record outside the tables waits
/// for a draw's table to claim its slot (`VICTORY_TABLES_LEFT_GRACE=1`).
pub(crate) const GRACE: u64 = 4;

/// Frames a hold lasts after the last buffer mark of its slot (the life of
/// the tree's marks of tables' slots, `TABLED_FRAMES`).
pub(crate) const HOLD_FRAMES: u32 = 240;

/// In ON phases, the slots left whose mark expired are healed, and the
/// holds whose mark is too old let go, once in this many frames.
pub(crate) const HEAL_EVERY: u64 = 64;

/// `VICTORY_TABLES_LEFT_GRACE=1` (off by default): stage 2's rule 2.
pub(crate) fn grace() -> bool {
    #[cfg(test)]
    if let Some((_, _, grace, _)) = TEST_LEFT.get() {
        return grace;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = left_set() && std::env::var_os("VICTORY_TABLES_LEFT_GRACE").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "VICTORY_TABLES_LEFT_GRACE=1: a new constant or placed buffer record outside the tables waits {GRACE} frames \
                 for a draw's table before the heap sync mirrors it"
            );
        }
        on
    })
}

/// The hold, on by default with the switch; `VICTORY_TABLES_LEFT_HOLD=0`
/// turns it off (to tell in one run whether copies let go and made again
/// are its cost).
pub(crate) fn hold_on() -> bool {
    #[cfg(test)]
    if let Some((_, _, _, hold)) = TEST_LEFT.get() {
        return hold;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_TABLES_LEFT_HOLD").is_none_or(|v| v != "0");
        if left_set() {
            tracing::info!(
                on,
                "VICTORY_TABLES_LEFT: the host copies draws read through slots left to them are held by those slots \
                 (VICTORY_TABLES_LEFT_HOLD=0: not)"
            );
        }
        on
    })
}

/// What the heap sync does with a new record ([`decide`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Copied {
    /// Mirrored now, as before.
    Mirror,
    /// Left to the draws that read the slot through a table: its print is
    /// not noted (looked at again when its page changes again), its slot's
    /// left bit set.
    Left,
    /// Waits [`GRACE`] frames ([`Waiting`]); its print is noted.
    Wait,
}

/// The sync's choice for a new record, by stage 2's rules (asked only while
/// [`acting`]): a buffer or constant view (`buffer`) in a slot with a buffer
/// mark of the last [`FRAMES`] frames (`marked`) is left; with the grace, a
/// constant or placed buffer view elsewhere (`held`) waits; the rest is
/// mirrored (textures follow stage 1's marks, records of no view empty the
/// slot). The closures are asked in that order, the cheap one first.
pub(crate) fn decide(marked: impl FnOnce() -> bool, buffer: impl FnOnce() -> bool, held: impl FnOnce() -> bool) -> Copied {
    let marked = marked();
    if marked && buffer() {
        Copied::Left
    } else if !marked && grace() && held() {
        Copied::Wait
    } else {
        Copied::Mirror
    }
}

/// Records that wait their grace ([`Copied::Wait`]), by the frame they were
/// found in, oldest first.
#[derive(Default)]
pub(crate) struct Waiting {
    queue: std::collections::VecDeque<(u64, Vec<u32>)>,
}

impl Waiting {
    /// `slots` were found with new records in `frame`.
    pub(crate) fn push(&mut self, frame: u64, slots: Vec<u32>) {
        if !slots.is_empty() {
            self.queue.push_back((frame, slots));
        }
    }

    /// The slots that have waited [`GRACE`] frames at `frame` (with `all`,
    /// every one), each once and in order; they leave the queue.
    pub(crate) fn due(&mut self, frame: u64, all: bool) -> Vec<u32> {
        let mut out = Vec::new();
        while self.queue.front().is_some_and(|(found, _)| all || found + GRACE <= frame) {
            if let Some((_, slots)) = self.queue.pop_front() {
                out.extend(slots);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The slots waiting (each as often as it was found).
    pub(crate) fn len(&self) -> usize {
        self.queue.iter().map(|(_, slots)| slots.len()).sum()
    }
}

/// Holds alive in every heap (for the summary).
static HOLDS_ALIVE: AtomicI64 = AtomicI64::new(0);

/// Per title slot, the host copy a draw read through it while it was left
/// to the draws, and the record (its key) it was read for. Without the
/// switch the slot's host slot (made by the pass over buffer views) and
/// that pass's kept range held the copy as long as the slot held the
/// record; left to the draws, only the draws' dynamic slots and lists did,
/// until the lists were reset, and the copy was let go once idle
/// (`crate::shadow::let_go_idle`) and made again at the next read. Here it
/// is held until the slot's buffer mark is [`HOLD_FRAMES`] frames old or the
/// title writes another record there: no copy lives longer than without the
/// switch, where the slot held it as long as the record stayed.
///
/// A pointer and a key per title slot, swapped atomically (no lock: every
/// draw takes the holds of the entries it reads): 16 bytes a slot, with the
/// switch only.
pub(crate) struct Holds {
    copies: Vec<AtomicPtr<Shadow>>,
    keys: Vec<AtomicU64>,
}

impl Holds {
    /// The holds of `slots` title slots (none: an empty table).
    pub(crate) fn new(slots: u32) -> Self {
        Holds {
            copies: (0..slots).map(|_| AtomicPtr::new(std::ptr::null_mut())).collect(),
            keys: (0..slots).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }

    /// Title slot `index`, holding the record `key`, holds `copy`. Returns
    /// whether that was new (not the copy and record it held already); the
    /// copy it held before, if any, is given back for retiring.
    pub(crate) fn take(&self, index: u32, key: u64, copy: &Arc<Shadow>) -> (bool, Option<Arc<Shadow>>) {
        let (Some(slot), Some(held_key)) = (self.copies.get(index as usize), self.keys.get(index as usize)) else {
            return (false, None);
        };
        if slot.load(Relaxed).cast_const() == Arc::as_ptr(copy) && held_key.load(Relaxed) == key {
            return (false, None);
        }
        held_key.store(key, Relaxed);
        let new = Arc::into_raw(copy.clone()).cast_mut();
        let old = slot.swap(new, Relaxed);
        if old.is_null() {
            HOLDS_ALIVE.fetch_add(1, Relaxed);
            return (true, None);
        }
        // SAFETY: every pointer in the table came from `Arc::into_raw` and is
        // owned by the table until swapped out, once.
        (true, Some(unsafe { Arc::from_raw(old.cast_const()) }))
    }

    /// The title wrote record `key` at title slot `index`: a hold taken for
    /// another record goes (given back for retiring).
    pub(crate) fn release_other(&self, index: u32, key: u64) -> Option<Arc<Shadow>> {
        let slot = self.copies.get(index as usize)?;
        if slot.load(Relaxed).is_null() || self.keys.get(index as usize).is_some_and(|k| k.load(Relaxed) == key) {
            return None;
        }
        self.drop_at(index as usize)
    }

    /// The hold of slot `at`, out of the table.
    fn drop_at(&self, at: usize) -> Option<Arc<Shadow>> {
        let old = self.copies.get(at)?.swap(std::ptr::null_mut(), Relaxed);
        if old.is_null() {
            return None;
        }
        HOLDS_ALIVE.fetch_sub(1, Relaxed);
        // SAFETY: as in `take`.
        Some(unsafe { Arc::from_raw(old.cast_const()) })
    }

    /// Lets go of the holds whose slot `keep` refuses; returns them.
    pub(crate) fn expire(&self, keep: impl Fn(u32) -> bool) -> Vec<Arc<Shadow>> {
        (0..self.copies.len())
            .filter(|&at| !self.copies[at].load(Relaxed).is_null() && !keep(at as u32))
            .filter_map(|at| self.drop_at(at))
            .collect()
    }

    /// Lets go of every hold; returns them.
    pub(crate) fn clear(&self) -> Vec<Arc<Shadow>> {
        self.expire(|_| false)
    }

    /// The copy title slot `index` holds, by its address (tests).
    #[cfg(test)]
    pub(crate) fn held(&self, index: u32) -> Option<usize> {
        let held = self.copies.get(index as usize)?.load(Relaxed);
        (!held.is_null()).then_some(held as usize)
    }
}

impl Drop for Holds {
    fn drop(&mut self) {
        drop(self.clear());
    }
}

/// Stage 2's state of one view heap (empty without the switch).
pub(crate) struct HeapLeft {
    /// A bit per title slot: the heap sync left the slot's new record to the
    /// draws and has not noted a record there since. The flip to off and the
    /// heal look at these alone.
    pub(crate) bits: Vec<AtomicU64>,
    /// The records waiting their grace.
    pub(crate) waiting: Mutex<Waiting>,
    /// The holds.
    pub(crate) holds: Holds,
    /// `check`: per title slot, the frame (plus one) from which the switch
    /// would have left its host slot showing another view than the title's
    /// record, or none; 0 while it would show the same.
    pub(crate) diverged: Vec<AtomicU32>,
    /// `check`: per title slot, 1 if the heap sync's next making of it is
    /// one the switch would not do (set where it decides, taken where it
    /// makes, on whichever path and thread).
    pub(crate) flags: Vec<AtomicU8>,
    /// Whether stage 2 acted at this heap's last heap sync ([`left_live`]):
    /// a change to false is the flip to off.
    pub(crate) live_seen: AtomicBool,
    /// The frame of the last heal.
    pub(crate) healed: AtomicU64,
}

impl HeapLeft {
    /// Stage 2's state for a heap of `titled` title slots (nothing for a
    /// sampler heap or without the switch).
    pub(crate) fn new(titled: u32, sampler: bool) -> Self {
        let on = !sampler && left_set();
        let check = on && checking();
        let n = if on { titled } else { 0 };
        HeapLeft {
            bits: (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            waiting: Mutex::default(),
            holds: Holds::new(if on && !check { titled } else { 0 }),
            diverged: (0..if check { titled } else { 0 }).map(|_| AtomicU32::new(0)).collect(),
            flags: (0..if check { titled } else { 0 }).map(|_| AtomicU8::new(0)).collect(),
            live_seen: AtomicBool::new(left_live()),
            healed: AtomicU64::new(0),
        }
    }

    /// Whether the heap has stage 2's state.
    pub(crate) fn on(&self) -> bool {
        !self.bits.is_empty()
    }

    /// Notes that the sync left title slot `index`'s record.
    pub(crate) fn bit_set(&self, index: u32) {
        if let Some(word) = self.bits.get(index as usize / 64) {
            let bit = 1u64 << (index % 64);
            if word.load(Relaxed) & bit == 0 {
                word.fetch_or(bit, Relaxed);
            }
        }
    }

    /// Notes that a record of title slot `index` was noted (mirrored).
    pub(crate) fn bit_clear(&self, index: u32) {
        if let Some(word) = self.bits.get(index as usize / 64) {
            let bit = 1u64 << (index % 64);
            if word.load(Relaxed) & bit != 0 {
                word.fetch_and(!bit, Relaxed);
            }
        }
    }

    /// The slots whose bit is set and that `take` accepts, in order; their
    /// bits are cleared.
    pub(crate) fn bits_take(&self, take: impl Fn(u32) -> bool) -> Vec<u32> {
        let mut out = Vec::new();
        for (w, word) in self.bits.iter().enumerate() {
            let mut set = word.load(Relaxed);
            while set != 0 {
                let b = set.trailing_zeros();
                set &= set - 1;
                let index = (w * 64) as u32 + b;
                if take(index) {
                    word.fetch_and(!(1u64 << b), Relaxed);
                    out.push(index);
                }
            }
        }
        out
    }

    /// `check`: the switch's host slot of title slot `index` would differ
    /// from the title's record from `frame` on (kept if it did already).
    pub(crate) fn diverge(&self, index: u32, frame: u64) {
        if let Some(since) = self.diverged.get(index as usize)
            && since.load(Relaxed) == 0
        {
            since.store((frame as u32).wrapping_add(1).max(1), Relaxed);
            DIVERGED_NOW.fetch_add(1, Relaxed);
        }
    }

    /// `check`: title slot `index`'s host slot is as the switch would have it.
    pub(crate) fn undiverge(&self, index: u32) {
        if let Some(since) = self.diverged.get(index as usize)
            && since.load(Relaxed) != 0
            && since.swap(0, Relaxed) != 0
        {
            DIVERGED_NOW.fetch_sub(1, Relaxed);
        }
    }

    /// `check`: since which frame (plus one) the switch's host slot of
    /// `index` would differ; 0 if it would not.
    pub(crate) fn diverged_since(&self, index: u32) -> u32 {
        self.diverged.get(index as usize).map_or(0, |since| since.load(Relaxed))
    }

    /// `check`: the sync's next making of title slot `index` is one the
    /// switch would not do.
    pub(crate) fn flag(&self, index: u32) {
        if let Some(flag) = self.flags.get(index as usize) {
            flag.store(1, Relaxed);
        }
    }

    /// `check`: whether the sync's making of `index` now is one the switch
    /// would not do (the flag taken).
    pub(crate) fn take_flag(&self, index: u32) -> bool {
        self.flags.get(index as usize).is_some_and(|flag| flag.load(Relaxed) != 0 && flag.swap(0, Relaxed) != 0)
    }
}

/// `check`: title slots in every heap whose host slot the switch would have
/// left showing another view than the title's record, or none.
static DIVERGED_NOW: AtomicI64 = AtomicI64::new(0);

thread_local! {
    /// `check`: while this thread makes slots the switch would have left to
    /// the draws (the pass over buffer views' left slots, [`left_making`]).
    static LEFT_MAKING: Cell<bool> = const { Cell::new(false) };
}

/// `check`: `f`, whose makings of slots on this thread are ones the switch
/// would not do (`HostHeap::check_making`).
pub(crate) fn left_making<R>(f: impl FnOnce() -> R) -> R {
    let was = LEFT_MAKING.replace(true);
    let out = f();
    LEFT_MAKING.set(was);
    out
}

/// `check`: whether this thread is in [`left_making`].
pub(crate) fn in_left_making() -> bool {
    LEFT_MAKING.get()
}

/// What stage 2's summary counts, per phase ([`add2`]).
#[derive(Clone, Copy, Debug)]
pub(crate) enum K {
    /// Heap syncs, their microseconds, the records they mirrored (the sync
    /// line's), those left to the draws by a buffer mark, those held by the
    /// grace, and after it mirrored or claimed by a draw's table.
    Syncs,
    SyncUs,
    Mirrored,
    Left,
    Held,
    GraceMirrored,
    GraceClaimed,
    /// Joins of the heap sync beside by the submitting thread, and their
    /// microseconds.
    Joins,
    JoinUs,
    /// Passes over buffer views, their microseconds on their thread (or the
    /// recording thread's in place), their slots, those left to the draws;
    /// the submitting thread's waits for a pass still running, and their
    /// microseconds.
    Passes,
    PassUs,
    PassSlots,
    PassLeft,
    PassWaits,
    PassWaitUs,
    /// Holds taken (new), let go because the title wrote another record
    /// in their slot, let go because their mark was too old, let go at the
    /// flip to off.
    HoldsTaken,
    HoldsReleased,
    HoldsExpired,
    HoldsCleared,
    /// Slots healed in ON phases (left, their mark expired, then mirrored).
    Healed,
}

const K_COUNT: usize = K::Healed as usize + 1;

/// Per phase (0: ON, or the only phase of `1` and `check`; 1: off).
static PHASES: [[AtomicU64; K_COUNT]; 2] = [const { [const { AtomicU64::new(0) }; K_COUNT] }; 2];

/// Per phase, the microseconds of the frame line's copies.
static COPIES_US: [AtomicI64; 2] = [const { AtomicI64::new(0) }; 2];

/// The phase counts go to now.
fn row() -> usize {
    usize::from(mode2() == Mode2::Alt && !left_live())
}

/// Counts `n` of `what` for stage 2's summary (nothing without the switch).
pub(crate) fn add2(what: K, n: u64) {
    if n > 0 && left_set() {
        PHASES[row()][what as usize].fetch_add(n, Relaxed);
    }
}

/// [`add2`] of a duration's microseconds, with one of `count`.
pub(crate) fn add2_timed(count: K, us: K, took: std::time::Duration) {
    if left_set() {
        let row = &PHASES[row()];
        row[count as usize].fetch_add(1, Relaxed);
        row[us as usize].fetch_add(took.as_micros() as u64, Relaxed);
    }
}

/// The frame line's copies took `us` microseconds more (or fewer).
pub(crate) fn copies_us(us: i64) {
    if left_set() {
        COPIES_US[row()].fetch_add(us, Relaxed);
    }
}

/// What the `check` counts, between two summaries.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Ck {
    /// Lists checked at their submission; the slots they read by an index
    /// they compute through tables read in place; of those, the ones whose
    /// host slot the switch would have left showing another view than the
    /// title's record, or none (violations).
    Lists,
    Reads,
    Violations,
    /// Lists whose shaders pick buffers from the whole heap, and the sum
    /// over them of the slots the switch would have left stale at their
    /// submission (a shader of theirs may pick one).
    Picking,
    Exposed,
    /// Slots the pass over buffer views made that the switch leaves to the
    /// draws, of them those the switch would have left stale, and of those
    /// stale for over [`FRAMES`] frames.
    PassLeft,
    PassStale,
    PassStaleLong,
    /// Makings by the heap sync that the switch would not have done.
    SyncLeft,
}

const CK_COUNT: usize = Ck::SyncLeft as usize + 1;

static CHECKS: [AtomicU64; CK_COUNT] = [const { AtomicU64::new(0) }; CK_COUNT];

#[cfg(test)]
thread_local! {
    /// Tests: with `Some`, what this thread counts for the check is added
    /// here too.
    pub(crate) static TEST_CHECKS: std::cell::RefCell<Option<[u64; CK_COUNT]>> = const { std::cell::RefCell::new(None) };
}

/// Counts `n` of `what` for the check.
pub(crate) fn add_check(what: Ck, n: u64) {
    if n > 0 {
        CHECKS[what as usize].fetch_add(n, Relaxed);
        #[cfg(test)]
        TEST_CHECKS.with_borrow_mut(|counts| {
            if let Some(counts) = counts {
                counts[what as usize] += n;
            }
        });
    }
}

/// `check`, at a submission of `runs`, once the heap sync it waits for has
/// ended (after its join, before the lists go): every slot a list read by
/// an index it computes through a table read in place must show what its
/// record names under the switch too (made by the draws or mirrored by the
/// sync); one that the switch would have left showing another view, or
/// none, is a violation. Lists whose shaders pick buffers from the whole
/// heap are counted with the slots the switch would have left stale then.
pub(crate) fn check_submission(runs: &[&crate::host::HostList]) {
    if !checking() {
        return;
    }
    let exposed = DIVERGED_NOW.load(Relaxed).max(0) as u64;
    for list in runs {
        let (mut reads, picks) = list.take_left_reads();
        add_check(Ck::Lists, 1);
        if picks {
            add_check(Ck::Picking, 1);
            add_check(Ck::Exposed, exposed);
        }
        if reads.is_empty() {
            continue;
        }
        reads.sort_unstable();
        reads.dedup();
        for group in reads.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
            let (heap, start) = (group[0].0, group[0].1);
            crate::descriptor::with_heap(heap, start, |heap| {
                if let Some(host) = heap.host() {
                    for &(_, _, slot) in group {
                        host.check_read(slot);
                    }
                }
            });
        }
    }
}

/// `check`: a read of title slot `slot` at a submission found its host slot
/// diverged since `since` (frame plus one): counted, and the first few named.
pub(crate) fn violation(slot: u32, since: u32) {
    add_check(Ck::Violations, 1);
    static SAID: AtomicU32 = AtomicU32::new(0);
    if SAID.fetch_add(1, Relaxed) < 16 {
        let now = (crate::frame::current() as u32).wrapping_add(1);
        tracing::warn!(
            slot,
            frames = now.wrapping_sub(since),
            "VICTORY_TABLES_LEFT=check: a list reads by index a title slot whose host slot the switch would have left \
             showing another view than the title's record, or none"
        );
    }
}

/// Stage 2's state for a line: `off`, `on`, `alt, ON`, `alt, off`, `check`.
pub(crate) fn said2() -> &'static str {
    if mode2() == Mode2::Check { "check" } else { TABLES_LEFT.said() }
}

/// The current period of stage 2's summary: its phase row, when it began
/// (`crate::host::now_ms`), the presents and the window copies let go and
/// made again then.
struct Period {
    row: usize,
    began: u64,
    presented: u64,
    churn: [u64; 2],
}

static PERIOD: Mutex<Option<Period>> = Mutex::new(None);

/// A period of `1` and `check` lasts this long (`alt`: its phase).
const PERIOD_MS: u64 = 20_000;

/// At each heap sync: a period that ended (a flip of `alt`, or 20 s of the
/// others) is summed up ([`period_line`]) and the next begins.
fn period_tick() {
    if !left_set() {
        return;
    }
    let row = row();
    let now = crate::host::now_ms();
    let fresh = || Period { row, began: now, presented: crate::queue::presented(), churn: crate::shadow::window_churn() };
    let mut period = PERIOD.lock().unwrap_or_else(|e| e.into_inner());
    match &*period {
        None => {
            for at in [0, 1] {
                for count in &PHASES[at] {
                    count.store(0, Relaxed);
                }
                COPIES_US[at].store(0, Relaxed);
            }
            *period = Some(fresh());
        }
        Some(p) if p.row != row || (mode2() != Mode2::Alt && now >= p.began + PERIOD_MS) => {
            let ended = period.replace(fresh());
            drop(period);
            if let Some(ended) = ended {
                period_line(&ended, now);
            }
        }
        Some(_) => {}
    }
}

/// The summary of the period `p` that ended at `now`: per frame, by phase.
fn period_line(p: &Period, now: u64) {
    let counts: [u64; K_COUNT] = std::array::from_fn(|i| PHASES[p.row][i].swap(0, Relaxed));
    let copies = COPIES_US[p.row].swap(0, Relaxed).max(0) as f64;
    let checks: [u64; CK_COUNT] = std::array::from_fn(|i| CHECKS[i].swap(0, Relaxed));
    let frames = crate::queue::presented().saturating_sub(p.presented);
    let churn = crate::shadow::window_churn();
    let seconds = now.saturating_sub(p.began) as f64 / 1000.0;
    tracing::info!("{}", said(p.row, &counts, copies, frames, seconds, [churn[0] - p.churn[0], churn[1] - p.churn[1]], &checks));
}

/// The summary's text (see [`period_line`]).
fn said(row: usize, c: &[u64; K_COUNT], copies_us: f64, frames: u64, seconds: f64, churn: [u64; 2], ck: &[u64; CK_COUNT]) -> String {
    let n = frames.max(1) as f64;
    let get = |k: K| c[k as usize];
    let per = |k: K| get(k) as f64 / n;
    let ms = |k: K| get(k) as f64 / 1000.0 / n;
    let phase = match (mode2(), row) {
        (Mode2::Alt, 0) => "the ON phase that ended",
        (Mode2::Alt, _) => "the off phase that ended",
        (Mode2::Check, _) => "check, the last period",
        _ => "on, the last period",
    };
    let check = if mode2() == Mode2::Check {
        let get = |k: Ck| ck[k as usize];
        format!(
            "; check since the last summary: {} lists, {} slots read by an index through tables read in place, {} \
             violations (the switch would have left their host slot showing another view than the title's record, or \
             none); {} lists pick buffers from the whole heap, which then had on average {:.0} slots the switch would \
             have left stale ({} now); the pass made {} slots the switch leaves, {} of them stale under it ({} for over \
             {FRAMES} frames); the sync made {} slots the switch would not have",
            get(Ck::Lists),
            get(Ck::Reads),
            get(Ck::Violations),
            get(Ck::Picking),
            get(Ck::Exposed) as f64 / get(Ck::Picking).max(1) as f64,
            DIVERGED_NOW.load(Relaxed).max(0),
            get(Ck::PassLeft),
            get(Ck::PassStale),
            get(Ck::PassStaleLong),
            get(Ck::SyncLeft),
        )
    } else {
        String::new()
    };
    format!(
        "tables left to the draws (VICTORY_TABLES_LEFT {}), {phase}: {frames} frames in {seconds:.1} s ({:.2} fps); per \
         frame: the heap sync mirrored {:.0} records in {:.1} syncs of {:.2} ms ({:.0} left to the draws by a buffer mark, \
         {:.0} held by the grace, after it {:.0} mirrored and {:.0} claimed by a draw's table); the submitting thread \
         waited {:.2} ms at the sync's join ({:.1} joins); the pass over buffer views {:.2} ms for {:.0} slots ({:.0} \
         left to the draws), and the submitting thread waited {:.2} ms for it ({:.1} waits); copies {:.2} ms (the frame \
         line's); window copies {:.2} let go and {:.2} made again; holds {:.1} taken, {:.1} let go for another record, \
         {:.1} expired, {:.1} let go at the flip, {} alive now (VICTORY_TABLES_LEFT_HOLD {}); {:.1} slots healed{check}",
        said2(),
        frames as f64 / seconds.max(0.001),
        per(K::Mirrored),
        per(K::Syncs),
        ms(K::SyncUs),
        per(K::Left),
        per(K::Held),
        per(K::GraceMirrored),
        per(K::GraceClaimed),
        ms(K::JoinUs),
        per(K::Joins),
        ms(K::PassUs),
        per(K::PassSlots),
        per(K::PassLeft),
        ms(K::PassWaitUs),
        per(K::PassWaits),
        copies_us / 1000.0 / n,
        churn[0] as f64 / n,
        churn[1] as f64 / n,
        per(K::HoldsTaken),
        per(K::HoldsReleased),
        per(K::HoldsExpired),
        per(K::HoldsCleared),
        HOLDS_ALIVE.load(Relaxed).max(0),
        if hold_on() { "on" } else { "0: off" },
        per(K::Healed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table made afresh past the identity window marks its title slots
    /// only with the switch; one past the slots the host heap mirrors never
    /// does; one wholly in the window always does.
    #[test]
    fn the_bound_is_the_mirrored_slots_with_the_switch() {
        for set in [false, true] {
            TEST_BOUND.set(Some((set, true)));
            assert_eq!(title_first(Some(10), 6, 16, 64), Some(10), "in the window");
            assert_eq!(title_first(Some(10), 7, 16, 64), set.then_some(10), "past the window");
            assert_eq!(title_first(Some(60), 4, 16, 64), set.then_some(60));
            assert_eq!(title_first(Some(60), 5, 16, 64), None, "past the mirrored slots");
            assert_eq!(title_first(None, 5, 16, 64), None);
            assert_eq!(title_first(Some(u32::MAX), 2, 16, u32::MAX), None, "no overflow");
        }
        TEST_BOUND.set(None);
    }

    /// A buffer mark is live for [`FRAMES`] frames after the one it was set
    /// in, and no longer; a slot never marked never is; a mark of a frame a
    /// little ahead counts.
    #[test]
    fn buffer_marks_expire_after_eight_frames() {
        assert!(!lately(0, 100), "never marked");
        let stamp = AtomicU32::new(0);
        mark(&stamp, 100);
        let s = stamp.load(Relaxed);
        assert_eq!(s, 101);
        assert!(lately(s, 100));
        assert!(lately(s, 100 + u64::from(FRAMES)));
        assert!(!lately(s, 101 + u64::from(FRAMES)), "expired");
        assert!(!lately(s, 100 + 1_000));
        assert!(lately(s, 99), "a draw of the next frame marked it");
        assert!(lately(s, 100 - u64::from(AHEAD)));
        assert!(!lately(s, 99 - u64::from(AHEAD)));
        // Across the wrap of the 31 bits the mark keeps.
        let wrap = u64::from(TOP) - 2;
        mark(&stamp, wrap);
        assert!(lately(stamp.load(Relaxed), wrap + 3));
        assert!(!lately(stamp.load(Relaxed), wrap + 3 + u64::from(FRAMES)));
    }

    /// Every entry of a table of up to 64 gets a buffer mark; of a longer one
    /// only the entries the shader names.
    #[test]
    fn big_tables_mark_only_the_entries_named() {
        let named = |k: u32| k == 3 || k == 70;
        let small: Vec<u32> = (0..64).filter(|&k| marked(64, k, named)).collect();
        assert_eq!(small, (0..64).collect::<Vec<u32>>());
        let big: Vec<u32> = (0..100).filter(|&k| marked(100, k, named)).collect();
        assert_eq!(big, [3, 70]);
    }

    /// Tables are counted by where they lie: all of the table in the
    /// identity window, in the title slots mirrored, or reaching past them.
    #[test]
    fn tables_are_counted_by_where_they_lie() {
        let at = |first, n| where_first(first, n, 16, 64) as usize;
        assert_eq!(at(Some(10), 6), C::FirstInWindow as usize);
        assert_eq!(at(Some(10), 7), C::FirstPastWindow as usize);
        assert_eq!(at(Some(60), 4), C::FirstPastWindow as usize);
        assert_eq!(at(Some(60), 5), C::FirstPastTitle as usize);
        assert_eq!(at(None, 5), C::FirstNoIndex as usize);
    }

    /// Stage 2's rule: a buffer or constant view in a marked slot is left;
    /// a texture view there is mirrored (stage 1's marks take it); with the
    /// grace, a constant or placed buffer record elsewhere waits; the rest is
    /// mirrored. The closures are asked only as needed, the mark first.
    #[test]
    fn stage_2_leaves_holds_or_mirrors() {
        for grace in [false, true] {
            TEST_LEFT.set(Some((Mode2::On, true, grace, true)));
            assert_eq!(decide(|| true, || true, || panic!("not asked")), Copied::Left);
            assert_eq!(decide(|| true, || false, || panic!("not asked")), Copied::Mirror, "a texture in a marked slot");
            assert_eq!(
                decide(|| false, || panic!("not asked"), || true),
                if grace { Copied::Wait } else { Copied::Mirror }
            );
            assert_eq!(decide(|| false, || panic!("not asked"), || false), Copied::Mirror);
        }
        TEST_LEFT.set(None);
    }

    /// Stage 2's modes: set, live and check as each mode has them; stage 1's
    /// bound and marks with any of them.
    #[test]
    fn stage_2_modes() {
        for (mode, live, set2, acts, check) in [
            (Mode2::Off, false, false, false, false),
            (Mode2::On, true, true, true, false),
            (Mode2::Alt, false, true, false, false),
            (Mode2::Alt, true, true, true, false),
            (Mode2::Check, false, true, true, true),
        ] {
            TEST_LEFT.set(Some((mode, live, false, true)));
            assert_eq!((left_set(), acting(), checking()), (set2, acts, check), "{mode:?} {live}");
            assert_eq!(left_live(), live);
        }
        TEST_LEFT.set(None);
    }

    /// New records wait [`GRACE`] frames and come out each once, in order,
    /// with those of earlier frames; with `all`, every one at once.
    #[test]
    fn waiting_records_come_out_after_four_frames() {
        let mut w = Waiting::default();
        w.push(10, vec![7, 3]);
        w.push(10, Vec::new());
        w.push(11, vec![3, 9]);
        assert_eq!(w.len(), 4);
        assert!(w.due(10 + GRACE - 1, false).is_empty());
        assert_eq!(w.due(10 + GRACE, false), [3, 7]);
        assert_eq!(w.len(), 2);
        assert!(w.due(10 + GRACE, false).is_empty());
        w.push(12, vec![1]);
        assert_eq!(w.due(12, true), [1, 3, 9], "each once, in order, all of them");
        assert_eq!(w.len(), 0);
    }

    /// A slot's left bit is set once, cleared when noted, and taken once.
    #[test]
    fn left_bits_are_taken_once() {
        TEST_LEFT.set(Some((Mode2::On, true, false, true)));
        let left = HeapLeft::new(200, false);
        assert!(left.on() && !left.holds.is_empty() && left.diverged.is_empty());
        for index in [3, 70, 199, 3] {
            left.bit_set(index);
        }
        left.bit_clear(70);
        assert_eq!(left.bits_take(|index| index != 199), [3]);
        assert_eq!(left.bits_take(|_| true), [199]);
        assert!(left.bits_take(|_| true).is_empty());
        assert!(!HeapLeft::new(200, true).on(), "nothing for a sampler heap");
        TEST_LEFT.set(Some((Mode2::Check, false, false, true)));
        let check = HeapLeft::new(200, false);
        assert!(check.holds.is_empty() && check.diverged.len() == 200, "check: the model, no holds");
        check.flag(5);
        assert!(check.take_flag(5) && !check.take_flag(5), "a flag is taken once");
        TEST_LEFT.set(Some((Mode2::Off, false, false, true)));
        assert!(!HeapLeft::new(200, false).on());
        TEST_LEFT.set(None);
    }

    /// The summary says what it counts, whatever the counts.
    #[test]
    fn the_summary_is_said() {
        let text = said(1, &[0; K_COUNT], 0.0, 0, 0.0, [0, 0], &[0; CK_COUNT]);
        assert!(text.starts_with("tables left to the draws"), "{text}");
        let mut counts = [0; K_COUNT];
        counts[K::Mirrored as usize] = 400;
        counts[K::PassWaitUs as usize] = 20_000;
        let text = said(0, &counts, 30_000.0, 10, 2.0, [5, 2], &[0; CK_COUNT]);
        assert!(text.contains("mirrored 40 records") && text.contains("waited 2.00 ms for it"), "{text}");
        assert!(text.contains("copies 3.00 ms") && text.contains("0.50 let go and 0.20 made again"), "{text}");
    }

    /// The sync's makings are counted by rule only inside [`SyncMaking`],
    /// nested or not.
    #[test]
    fn sync_making_nests() {
        assert!(!in_sync_making());
        {
            let _outer = SyncMaking::enter();
            assert!(in_sync_making());
            {
                let _inner = SyncMaking::enter();
                assert!(in_sync_making());
            }
            assert!(in_sync_making(), "the inner one gave back the outer's");
        }
        assert!(!in_sync_making());
    }
}
