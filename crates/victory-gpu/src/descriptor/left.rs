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

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};

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

/// Whether the switch is set (`1` or `alt`): draws test a table made afresh
/// against the title slots the host heap mirrors, and set buffer marks, in
/// both phases.
pub(crate) fn set() -> bool {
    #[cfg(test)]
    if let Some((set, _)) = TEST_BOUND.get() {
        return set;
    }
    TABLE_BOUND.set()
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
    let stamp = stamp & !TOP;
    if stamp == 0 {
        return false;
    }
    let now = (frame as u32).wrapping_add(1) & !TOP;
    let since = now.wrapping_sub(stamp) & !TOP;
    let ahead = stamp.wrapping_sub(now) & !TOP;
    since <= FRAMES || ahead <= AHEAD
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
