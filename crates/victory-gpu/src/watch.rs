//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).
//!
//! With `VICTORY_LIGHT_REFRESH` a block that is read-only by our hand,
//! unwritten since, and whose earlier writes are all in the journal has a
//! bit ([`PUBLISHED`]): a range whose blocks all have theirs is answered
//! from the journal alone, with no lock of the guards and no kernel call
//! ([`light`]).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::BuildHasherDefault;
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

/// Page number → the asking that last found it written, of the pages ever
/// found written: in pieces of [`PIECE`] pages, each made when a page of it
/// is first written. A look at a range of pages is a lookup for each piece
/// it lies in.
///
/// It was a tree of every page ever written (some 80,000-150,000 in a
/// cell): its lookups, under the journal's lock, some 10,000 a frame from
/// the refreshes' looks ([`nothing_new`]) and their askings, walked it
/// twice down to the range's ends; 111 ns a look at a range of a few pages
/// in the tree, 20 ns in pieces (Biskwit/VIctory-reloaded 27b7a9002).
struct Written {
    pieces: HashMap<usize, Box<[u64; PIECE]>, BuildHasherDefault<WordHash>>,
}

/// Pages to a piece of [`Written`]: 4 MiB of title memory.
const PIECE: usize = 64;

impl Written {
    const fn new() -> Written {
        Written {
            pieces: HashMap::with_hasher(BuildHasherDefault::new()),
        }
    }

    /// `page` was found written by asking `now` (not 0); the asking that
    /// found it written before (0: none).
    fn insert(&mut self, page: usize, now: u64) -> u64 {
        let piece = self.pieces.entry(page / PIECE).or_insert_with(|| Box::new([0; PIECE]));
        std::mem::replace(&mut piece[page % PIECE], now)
    }

    /// The asking that last found `page` written, if one did.
    #[cfg(test)]
    fn get(&self, page: usize) -> Option<u64> {
        self.pieces.get(&(page / PIECE)).map(|piece| piece[page % PIECE]).filter(|&w| w != 0)
    }

    /// `page` is not written, as far as the journal knows (the tests').
    #[cfg(test)]
    fn remove(&mut self, page: usize) {
        if let Some(piece) = self.pieces.get_mut(&(page / PIECE)) {
            piece[page % PIECE] = 0;
        }
    }

    /// The pages of `[first, last)` ever found written, in order, each with
    /// the asking that last found it written.
    fn range(&self, first: usize, last: usize) -> impl Iterator<Item = (usize, u64)> + '_ {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::JournalLooks, 1);
        let pieces = if first < last { first / PIECE..(last - 1) / PIECE + 1 } else { 0..0 };
        pieces.filter_map(|n| self.pieces.get(&n).map(|piece| (n, piece))).flat_map(move |(n, piece)| {
            let (from, to) = ((n * PIECE).max(first), ((n + 1) * PIECE).min(last));
            (from..to).filter_map(move |page| {
                let write = piece[page % PIECE];
                (write != 0).then_some((page, write))
            })
        })
    }
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: Written,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Ahead,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

/// Page ranges `[a, b)` asked ahead, by start, each as asked (none merged):
/// whether one of them holds a range is asked at every refresh of a
/// submission (thousands a frame) and at every range of [`ask_once`], where
/// a scan of all of them made the submission's asking quadratic in its
/// ranges. `reach[i]` is the furthest end of `ranges[..=i]`, so "some range
/// with `a <= first` ends at `last` or later", which is "one range holds
/// `[first, last)`", is one binary search (after ScarlettPort 50a9f7c5; the
/// answers are the scan's).
struct Ahead {
    ranges: Vec<(usize, usize)>,
    reach: Vec<usize>,
}

impl Ahead {
    const fn new() -> Ahead {
        Ahead {
            ranges: Vec::new(),
            reach: Vec::new(),
        }
    }

    /// Whether one range asked holds all of `[first, last)`.
    fn holds(&self, first: usize, last: usize) -> bool {
        let i = self.ranges.partition_point(|&(a, _)| a <= first);
        i > 0 && self.reach[i - 1] >= last
    }

    fn push(&mut self, range: (usize, usize)) {
        let i = self.ranges.partition_point(|&r| r <= range);
        self.ranges.insert(i, range);
        self.reach_from(i);
    }

    /// `reach` again from range `i` on.
    fn reach_from(&mut self, i: usize) {
        self.reach.truncate(i);
        let mut far = i.checked_sub(1).map_or(0, |i| self.reach[i]);
        for &(_, b) in &self.ranges[i..] {
            far = far.max(b);
            self.reach.push(far);
        }
    }

    fn clear(&mut self) {
        self.ranges.clear();
        self.reach.clear();
    }

    fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: Written::new(),
    found: Vec::new(),
    ahead: Ahead::new(),
    ahead_asked: 0,
    kept: false,
});

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: impl Iterator<Item = (usize, usize)>) {
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`, and no plan is made (nor any range Vec
    // built: the ranges are not consumed). After ScarlettPort 23870346.
    if JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).kept {
        return;
    }
    let mut pieces: Vec<(usize, usize)> = ranges
        .filter(|r| r.0 < r.1)
        .map(|(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked for while the first look was being made.
    if j.kept {
        return;
    }
    // View memory the guards hold is left to `writes`, which answers it
    // from the guards ([`known_view`]): the kernel's write watch can only
    // fail there. Asked here with the others, a run holding such a range
    // failed, was asked again by halves down to single ranges, failing
    // each time, and the range then went to `writes` all the same (after
    // triippywhale d0581a55). `VICTORY_VIEW_ASK=1` asks as before.
    #[cfg(feature = "diagnostics")]
    let all = pieces.len();
    pieces.retain(|&(first, _)| !known_view(first * PAGE));
    #[cfg(feature = "diagnostics")]
    AHEAD_VIEWS.fetch_add((all - pieces.len()) as u64, std::sync::atomic::Ordering::Relaxed);
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Since the last frame line: ranges of view memory the guards hold that a
/// submission's askings ahead ([`ask_ahead`], [`ask_once`]) left out. Each
/// was a kernel asking that failed, or several, before.
#[cfg(feature = "diagnostics")]
pub(crate) static AHEAD_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A `GetWriteWatch` call over `pages` 4 KiB pages made at `started`, failed
/// or not (the copies line: calls, their microseconds, pages asked).
#[cfg(feature = "diagnostics")]
fn called(started: std::time::Instant, pages: usize) {
    use crate::copystat::{C, add};
    add(C::WatchCalls, 1);
    add(C::WatchCallUs, started.elapsed().as_micros() as u64);
    add(C::WatchPages, pages as u64);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`]. `view` is told the place in `ranges` of each range
/// found to be view memory the guards hold (left to [`writes`]).
pub(crate) fn ask_once(ranges: &[(usize, usize)], mut view: impl FnMut(usize)) {
    if !once() {
        return;
    }
    // The light refresh: a few of the holes known by their bits are asked
    // of Windows again, a submission at a time ([`look_at_holes`]).
    if light_kept() {
        look_at_holes(HOLES_A_LOOK);
    }
    ask_once_end();
    ask_ahead(ranges.iter().copied());
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for (i, &(start, end)) in ranges.iter().enumerate() {
        if start >= end {
            continue;
        }
        // View memory the guards hold is not asked about: the asking can
        // only fail there, one failed kernel call for each such copy at
        // every submission, and `writes` answers it from the guards
        // (after triippywhale cc368c7c).
        if known_view(start) {
            #[cfg(feature = "diagnostics")]
            AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            view(i);
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.holds(first, last) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
    }
}

/// We wrote the 64 KiB page numbered `page` of title memory through a
/// view of our own (`crate::inplace::write_own`): no guard was opened and
/// no kernel write-watch bit set, so the journal is told here. The page,
/// and the same physical page at its other title addresses (`others`,
/// page numbers), count as written at a new asking, which every host copy
/// that holds them takes at its next refresh. Returns the asking that had
/// last found `page` written (0: never) and the new one, for the copy
/// that wrote to note itself current.
pub(crate) fn wrote_own(page: usize, others: &[usize]) -> (u64, u64) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    j.asked += 1;
    let now = j.asked;
    // As in [`wrote`]: an asking that stands for a submission, or for a
    // flush, moves with it, or a copy taking the page at the standing
    // asking would find it written "later" at every refresh.
    if j.kept || !j.ahead.is_empty() {
        j.ahead_asked = now;
    }
    let before = j.written.insert(page, now);
    for &other in others {
        if other != page {
            j.written.insert(other, now);
        }
    }
    (before, now)
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    #[cfg(feature = "diagnostics")]
    let pages = count;
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    #[cfg(feature = "diagnostics")]
    called(started, pages);
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    for k in 0..count {
        let page = j.found[k] / PAGE;
        j.written.insert(page, now);
    }
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    asked_inner(start, end)
}

/// The number of the asking that stands, if `[start, end)` lies in a range
/// asked ahead ([`ask_ahead`], [`ask_once`]): while that number holds, the
/// journal can note no new write in the range (`VICTORY_REFRESH_STAMP`).
pub(crate) fn ahead_now(start: usize, end: usize) -> Option<u64> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead.holds(first, last).then_some(j.ahead_asked)
}

fn asked_inner(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .holds(first, last)
        .then(|| (j.ahead_asked, j.written.range(first, last).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    // The light refresh: its bits kept, and asked or not, as the switch
    // says now.
    let light = if light_kept() { 1 + u8::from(light()) } else { 0 };
    // The counts every 65,536 askings that allow for view memory (in turn
    // they are said at each change; with diagnostics on the copies line).
    #[cfg(not(feature = "diagnostics"))]
    if guarded && light_mode() == 1 {
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if (CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1).is_multiple_of(65_536) {
            let [bits, locked, filled] = light_asks();
            tracing::info!(
                "refreshes of host copies go the light way (VICTORY_LIGHT_REFRESH=1): of the last 65536 askings that allowed for view memory {bits} were answered by the bits of their blocks and {locked} took the guards' lock (the others were of memory the kernel watches); {filled} holes found filled with nobody telling"
            );
        }
    }
    writes_as(start, end, guarded, light)
}

/// [`writes`] with the light refresh as `light` says ([`light`]): 0, as
/// without the switch, no bit is set or asked; 1, the blocks the asking
/// leaves clean get their bits and none is asked (the phase of `alt`
/// without); 2, a range whose blocks all have their bit is answered from
/// the journal.
fn writes_as(start: usize, end: usize, guarded: bool, light: u8) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.holds(first, last) {
        #[cfg(feature = "diagnostics")]
        let collecting = std::time::Instant::now();
        let written = j.written.range(first, last).collect();
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WritesJournal, 1);
            add(C::WritesCollectNs, collecting.elapsed().as_nanos() as u64);
        }
        return Some((j.ahead_asked, written));
    }
    // The light refresh: every block of the range is read-only by our hand
    // and unwritten since an asking noted it, or a hole our notes answer
    // for ([`PUBLISHED`]). The guards would find nothing written and this
    // asking would note nothing: its number and the journal as it is are
    // the answer, with the guards' lock not taken and no kernel call.
    if guarded && light == 2 && published(j, first * PAGE, last * PAGE) {
        light_count(0);
        j.asked += 1;
        return Some((j.asked, j.written.range(first, last).collect()));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // View memory the guards already hold cannot be write-watched: the
    // kernel asking would only fail. Copies of it are flushed many times a
    // frame, each time paying that failed call before the guards answer.
    let known_view = guarded && known_view(first * PAGE);
    #[cfg(feature = "diagnostics")]
    let pages = count;
    // SAFETY: the answer buffer holds `count` entries.
    let failed = known_view
        || unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
    #[cfg(feature = "diagnostics")]
    if !known_view {
        called(started, pages);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        if known_view {
            crate::copystat::C::WatchKnownView
        } else {
            crate::copystat::C::WatchFailed
        },
        failed as u64,
    );
    #[cfg(feature = "diagnostics")]
    if failed && !known_view && guarded {
        failed_asking(first * PAGE, last * PAGE);
    }
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages: Vec<usize> = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded).collect();
    for &page in &pages {
        j.written.insert(page, now);
    }
    // The light refresh: the guards were asked, and what they found is
    // noted. The blocks they made clean are known by their bits from now
    // on (the journal's lock is still held: no other asking came between).
    if failed && light != 0 {
        light_count(1);
        publish(j, &pages, first * PAGE, last * PAGE, holes_known());
    }
    #[cfg(feature = "diagnostics")]
    let collecting = std::time::Instant::now();
    let written = j.written.range(first, last).collect();
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::WritesCollectNs, collecting.elapsed().as_nanos() as u64);
    Some((now, written))
}

/// Diagnostics: one in 2,048 of the askings the kernel's write watch
/// refused for memory the guards did not know (some 200 a frame in the
/// first rooms, each followed by a question to Windows about the memory):
/// what the memory is and what the guards hold of its first block.
#[cfg(feature = "diagnostics")]
fn failed_asking(start: usize, end: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2048 != 0 {
        return;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    let (protected, clean, absent, viewed, aliased) = {
        let g = guards();
        let b = start / BLOCK;
        (g.protected.contains(&b), g.clean.contains(b), g.absent.contains(&b), g.viewed.contains(&b), g.aliased.contains(&b))
    };
    tracing::info!(
        "a failed write-watch asking (one in 2048): {start:#x}+{:#x}; Windows says (answered {n}): base {:#x} region {:#x} state {:#x} type {:#x} protect {:#x} allocation protect {:#x}; \
         the guards of its first block: protected {protected} clean {clean} absent {absent} viewed {viewed} aliased {aliased}; is a view: {}",
        end - start,
        info.BaseAddress as usize,
        info.RegionSize,
        info.State,
        info.Type,
        info.Protect,
        info.AllocationProtect,
        is_view(start, end)
    );
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// `$e`, counted once in the diagnostics counter `$n` and its time added
/// to `$ns` (`crate::copystat`); just `$e` in other builds. For the costs
/// of [`guard`]'s walk, each a call into Windows or a lookup under another
/// lock, which the clock's reading (some 20 ns) does not change much.
macro_rules! timed {
    ($n:ident, $ns:ident, $e:expr) => {{
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let value = $e;
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::$n, 1);
            crate::copystat::add(crate::copystat::C::$ns, started.elapsed().as_nanos() as u64);
        }
        value
    }};
}

/// What one asking's walk of [`guard`] found, block by block, summed there
/// and counted once (diagnostics): blocks reported written; of them, the
/// ones still read-only by our hand; blocks noted absent that are still
/// uncommitted; and with [`hot_frames`], blocks reported for being hot,
/// blocks made hot, hot blocks guarded again to test them, and hot blocks
/// guarded again for being reported too often ([`hot_ratio`]); questions
/// whether memory is uncommitted answered by a run asked about before
/// ([`absent_known`]), and blocks noted absent found committed.
#[derive(Default)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
struct Tally {
    written: u64,
    guarded: u64,
    absent: u64,
    hot: u64,
    made: u64,
    retests: u64,
    cooled: u64,
    saved: u64,
    found: u64,
}

impl Tally {
    /// Adds this walk's numbers to the frame's.
    fn count(&self) {
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::GuardWritten, self.written);
            add(C::GuardWrittenGuarded, self.guarded);
            add(C::GuardAbsentSkipped, self.absent);
            add(C::GuardHotReported, self.hot);
            add(C::GuardHotMade, self.made);
            add(C::GuardHotRetests, self.retests);
            add(C::GuardHotCooled, self.cooled);
            add(C::GuardQueriesSaved, self.saved);
            add(C::GuardAbsentFound, self.found);
        }
    }
}

/// Whether a run of uncommitted blocks Windows was asked about answers for
/// its blocks noted absent at later askings ([`uncommitted_run`]):
/// `VICTORY_GUARD_ABSENT_KNOWN=1` in the same frame, `=forever` until the
/// runtime remaps the run; unset, every asking asks again, as before.
///
/// Run RT75: 36,640 questions a frame (VirtualQuery, `uncommitted_end`),
/// 23 ms of the walk's 32, for 58,458 blocks noted absent walked: the holes
/// of view memory the copies' ranges cover, each asked about again at every
/// asking that walks it. A block noted absent becomes committed when the
/// runtime maps it, and the runtime says so ([`title_memory_remapped`],
/// which lets go of its runs here). Whether nothing else commits such
/// memory cannot be shown: the title's kernel32 imports the HLE does not
/// take (VirtualAlloc, VirtualFree) reach Windows directly, a free gap can
/// be taken by any allocator of the process, and Windows does not tell a
/// placeholder from a reservation. With `=1` such a commit is seen at the
/// frame's first asking that walks the block, as without the switch at the
/// next asking; `=forever` sees none until the runtime remaps the range.
/// Diagnostics count, on the frame line, the blocks noted absent that a
/// question found committed (a commit nobody told us of), and with
/// `VICTORY_GUARD_ABSENT_CHECK=1` one answer from a run in 64 is asked of
/// Windows too and counted wrong if the block is committed ([`check_absent`]).
fn absent_known() -> AbsentKnown {
    #[cfg(test)]
    if let Some(mode) = TEST_ABSENT_KNOWN.get() {
        return mode;
    }
    static MODE: std::sync::OnceLock<AbsentKnown> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_GUARD_ABSENT_KNOWN").as_deref().map(str::trim) {
        Ok("1") => AbsentKnown::Frame,
        Ok("forever") => AbsentKnown::Forever,
        _ => AbsentKnown::Off,
    })
}

/// [`absent_known`]'s modes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AbsentKnown {
    Off,
    Frame,
    Forever,
}

/// [`uncommitted_end`] for the walk, at a block noted absent. With
/// [`absent_known`] the run of wholly uncommitted blocks Windows gave is
/// kept, and a block in it is answered from it, in the same frame or until
/// the runtime remaps it.
fn uncommitted_run(g: &mut Guards, block: usize, tally: &mut Tally) -> usize {
    let mode = absent_known();
    if mode == AbsentKnown::Off {
        return uncommitted_end(block);
    }
    if mode == AbsentKnown::Frame {
        let frame = frame_now();
        if g.uncommitted_frame != frame {
            g.uncommitted_frame = frame;
            g.uncommitted.clear();
        }
    }
    // The runs do not overlap: the one starting last at or before `block`
    // is the only one that can hold it.
    if let Some((_, &end)) = g.uncommitted.range(..=block).next_back()
        && block < end
    {
        tally.saved += 1;
        #[cfg(feature = "diagnostics")]
        check_absent(block);
        return end;
    }
    let end = uncommitted_end(block);
    if end > block {
        // The run before ends at or before `block` (else it answered);
        // the runs inside this one were asked about before it grew.
        let inside: Vec<usize> = g.uncommitted.range(block + 1..end).map(|(&first, _)| first).collect();
        for first in inside {
            g.uncommitted.remove(&first);
        }
        g.uncommitted.insert(block, end);
    }
    end
}

/// The runs of [`uncommitted_run`] that touch blocks `[first, last)` are
/// let go of (the runtime remapped them).
fn forget_uncommitted(g: &mut Guards, first: usize, last: usize) {
    if g.uncommitted.is_empty() || first >= last {
        return;
    }
    let mut gone: Vec<usize> = g.uncommitted.range(first..last).map(|(&start, _)| start).collect();
    if let Some((&start, &end)) = g.uncommitted.range(..first).next_back()
        && end > first
    {
        gone.push(start);
    }
    for start in gone {
        g.uncommitted.remove(&start);
    }
}

/// `VICTORY_GUARD_ABSENT_CHECK`, since the start: answers from a run that
/// were asked of Windows too, and of them the ones where the block was
/// committed.
#[cfg(feature = "diagnostics")]
pub(crate) static ABSENT_CHECKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// With `VICTORY_GUARD_ABSENT_CHECK=1`, one answer from a run in 64: asks
/// Windows too (not counted with the walk's questions) and counts it wrong
/// if `block` is committed now (the first few said in the log).
#[cfg(feature = "diagnostics")]
fn check_absent(block: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    static TURN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !absent_check() || !TURN.fetch_add(1, Relaxed).is_multiple_of(64) {
        return;
    }
    ABSENT_CHECKS[0].fetch_add(1, Relaxed);
    if ask_uncommitted_end(block) <= block && ABSENT_CHECKS[1].fetch_add(1, Relaxed) < 32 {
        tracing::warn!(
            "VICTORY_GUARD_ABSENT_CHECK: block {:#x} was answered uncommitted from a run asked about before, and is committed",
            block * BLOCK
        );
    }
}

/// `VICTORY_GUARD_ABSENT_CHECK=1`.
#[cfg(feature = "diagnostics")]
fn absent_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_ABSENT_CHECK").is_some_and(|v| v == "1"))
}

/// The frame line's words on the questions whether memory is uncommitted,
/// given its counters per frame: answered from runs ([`absent_known`]),
/// blocks noted absent found committed, and the checks.
#[cfg(feature = "diagnostics")]
pub(crate) fn absent_clause(saved: f64, found: f64) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let mode = match absent_known() {
        AbsentKnown::Off => "",
        AbsentKnown::Frame => " (VICTORY_GUARD_ABSENT_KNOWN=1)",
        AbsentKnown::Forever => " (VICTORY_GUARD_ABSENT_KNOWN=forever)",
    };
    let checks = if absent_check() {
        format!(", checks since the start: {} of {} answers wrong", ABSENT_CHECKS[1].load(Relaxed), ABSENT_CHECKS[0].load(Relaxed))
    } else {
        String::new()
    };
    format!(
        " ({saved:.0} more answered by runs asked about before{mode}; {found:.1} blocks noted absent found committed \
         with nothing said{checks})"
    )
}

/// Blocks found written in this many frames in a row are left writable
/// (`VICTORY_GUARD_HOT=<n>`; unset or 0, off): hot.
///
/// In a steady scene the submitting thread spent ~28 ms a frame walking
/// ranges that were not all clean (run RT69), guarding again the blocks
/// the title had written, many of them every frame, and the other views of
/// their pages: a call making each read-only on our thread, then on the
/// title's a fault and a call making it writable again, frame after frame.
/// A hot block is spared both: it stays writable, and every asking that
/// covers it reports it written, so its copies take its 64 KiB at every
/// asking. No write to it can be missed: a block reported at every asking
/// needs no guard.
///
/// The other views of a hot block's physical page (2 to [`MANY_VIEWS`])
/// are made hot with it, writable and reported at every asking: left
/// guarded, a write through the hot one would not open them. A view of the
/// page mapped later ([`views_mapped`]) ends that, and so does guarding a
/// view with the others ([`protect_alias`]): the blocks are guarded again
/// at their next asking. A block remapped is forgotten.
///
/// Once every [`HOT_RETEST`] frames a hot block is guarded again (with its
/// other views): found written in the frame after, it is hot again; if
/// not, it has gone cold. One reported much more often than it had been
/// found written is guarded again at once ([`hot_ratio`]).
/// `VICTORY_GUARD_HOT_CHECK=1` (diagnostics) checks each block made hot or
/// tested again ([`check_hot`]).
fn hot_frames() -> u64 {
    #[cfg(test)]
    if let Some(n) = TEST_HOT.get() {
        return n;
    }
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_GUARD_HOT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// A hot block stays hot while it is reported in a frame at most this many
/// times as often as it was found written in the frame before it went hot
/// (`VICTORY_GUARD_HOT_RATIO=<r>`; 2 if unset, 0 no bound). Past that it is
/// guarded again, and not made hot again for [`HOT_RETEST`] frames.
///
/// Guarded, a block covered by k askings a frame and found written at m of
/// them costs m calls making it read-only and m faults; hot, it costs k
/// copies of its 64 KiB (and more where a texture lies on it, detiled and
/// hashed again). A block the title writes once a frame and that every
/// submission asks about is cheaper guarded.
fn hot_ratio() -> u64 {
    #[cfg(test)]
    if let Some(r) = TEST_HOT_RATIO.get() {
        return r;
    }
    static R: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var("VICTORY_GUARD_HOT_RATIO").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// A hot block is guarded again after this many frames, to see whether it
/// has gone cold; one guarded again by [`hot_ratio`] is not made hot again
/// for as many.
const HOT_RETEST: u64 = 64;

/// A hot block ([`hot_frames`]): the frame it was made hot (or tested
/// again), the most it may be reported in a frame ([`hot_ratio`]), and how
/// often it was in the frame it was last reported in.
#[derive(Clone, Copy)]
struct Hotness {
    since: u64,
    most: u64,
    frame: u64,
    reports: u64,
}

/// A block found written lately ([`hot_frames`]): the last frame it was,
/// in how many frames in a row, and how often in that frame and in the one
/// before it.
#[derive(Clone, Copy)]
struct Streak {
    last: u64,
    frames: u64,
    finds: u64,
    before: u64,
}

/// What [`Hot::keeps`] says of a block found written.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Keep {
    /// Guarded (again) as usual.
    Guard,
    /// Hot: left writable.
    Hot,
    /// To be made hot now, reported at most this many times a frame.
    Make(u64),
}

/// The frames presented so far (in tests, as the test says).
fn frame_now() -> u64 {
    #[cfg(test)]
    if let Some(frame) = TEST_FRAME.get() {
        return frame;
    }
    crate::queue::presented()
}

/// [`hot_frames`] for one asking: the frames in a row that make a block
/// hot (0: off), and this frame.
#[derive(Clone, Copy)]
struct Hot {
    frames: u64,
    frame: u64,
}

impl Hot {
    /// For an asking under the lock: at the first asking of a frame, lets
    /// go of the streaks of blocks not found written in the frame before,
    /// and of the blocks kept cold until this frame.
    fn now(g: &mut Guards) -> Self {
        let frames = hot_frames();
        if frames == 0 {
            return Self { frames, frame: 0 };
        }
        let frame = frame_now();
        if g.pruned != frame {
            g.pruned = frame;
            g.streak.retain(|_, s| s.last.saturating_add(1) >= frame);
            g.cool.retain(|_, &mut until| until > frame);
        }
        Self { frames, frame }
    }

    /// For a block found written: whether it is hot and stays so, is to be
    /// made hot now (the caller guards the run of blocks before it first,
    /// then [`make_hot`]s it), or is guarded (again) as usual. It is noted
    /// found written this frame, also when it was still read-only (written
    /// through another view of its page, guarded at that view's asking):
    /// whichever view is asked first in a frame, each view's blocks count
    /// the frame.
    fn keeps(self, g: &mut Guards, block: usize, tally: &mut Tally) -> Keep {
        if self.frames == 0 {
            return Keep::Guard;
        }
        if let Some(h) = g.hot.get_mut(&block) {
            if self.frame.saturating_sub(h.since) < HOT_RETEST {
                if h.frame != self.frame {
                    h.frame = self.frame;
                    h.reports = 0;
                }
                h.reports += 1;
                if h.reports <= h.most {
                    tally.hot += 1;
                    return Keep::Hot;
                }
                // Asked about much more often than it was written: cheaper
                // guarded, and not made hot again for a while.
                g.hot.remove(&block);
                g.cool.insert(block, self.frame.saturating_add(HOT_RETEST));
                tally.cooled += 1;
                return Keep::Guard;
            }
            // Guarded again, to see whether it has gone cold: found
            // written in the frame after, it is hot again.
            #[cfg(feature = "diagnostics")]
            check_hot(g, block);
            g.hot.remove(&block);
            g.streak.insert(block, Streak { last: self.frame, frames: self.frames - 1, finds: 1, before: 0 });
            tally.retests += 1;
            return Keep::Guard;
        }
        let s = match g.streak.get(&block) {
            Some(&s) if s.last == self.frame => Streak { finds: s.finds + 1, ..s },
            Some(&s) if s.last.saturating_add(1) == self.frame => {
                Streak { last: self.frame, frames: s.frames + 1, finds: 1, before: s.finds }
            }
            _ => Streak { last: self.frame, frames: 1, finds: 1, before: 0 },
        };
        if s.frames < self.frames || g.cool.get(&block).is_some_and(|&until| until > self.frame) {
            g.streak.insert(block, s);
            return Keep::Guard;
        }
        tally.made += 1;
        tally.hot += 1;
        let ratio = hot_ratio();
        Keep::Make(if ratio == 0 { u64::MAX } else { ratio.saturating_mul(s.before.max(1)) })
    }
}

/// Makes `block` hot in `frame` (see [`hot_frames`]), reported at most
/// `most` times a frame, and with it every other view of its physical page
/// (2 to [`MANY_VIEWS`] views): writable, out of the guards' sets, reported
/// written at every asking. `block` is reported at this asking. A view
/// read-only by our hand is made writable (`block` itself too if it was:
/// written through another view, guarded at that view's asking).
fn make_hot(g: &mut Guards, block: usize, frame: u64, most: u64) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let shown = if view_guard() { crate::inplace::pages_of(block * BLOCK) } else { Vec::new() };
    let others = if (2..=MANY_VIEWS).contains(&shown.len()) { shown } else { Vec::new() };
    let group: Vec<usize> = std::iter::once(block).chain(others.iter().map(|at| at / BLOCK)).collect();
    for &b in &group {
        // Not known by its bit once it can be written (the light refresh,
        // [`PUBLISHED`]), nor as a hole.
        PUBLISHED.clear(b);
        forget_hole(g, b);
        if g.protected.remove(&b) {
            g.clean.remove(b);
            let mut old = 0;
            // SAFETY: a block we made read-only.
            unsafe { VirtualProtect((b * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
        }
        g.absent.remove(&b);
        g.aliased.remove(&b);
        g.streak.remove(&b);
        let reports = u64::from(b == block);
        g.hot.insert(b, Hotness { since: frame, most, frame, reports });
    }
    #[cfg(feature = "diagnostics")]
    for &b in &group {
        check_hot(g, b);
    }
}

/// Hot blocks of `[start, end)` are no longer hot: guarded again at their
/// next asking (they are writable and in none of the guards' sets).
fn unhot(g: &mut Guards, start: usize, end: usize) {
    if g.hot.is_empty() {
        return;
    }
    let hot: Vec<usize> = g.hot.range(start / BLOCK..end.div_ceil(BLOCK)).map(|(&b, _)| b).collect();
    for b in hot {
        g.hot.remove(&b);
    }
}

/// `VICTORY_GUARD_HOT_CHECK`, since the start: hot blocks checked, and of
/// them the ones found otherwise than a hot block must be.
#[cfg(feature = "diagnostics")]
pub(crate) static HOT_CHECKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// With `VICTORY_GUARD_HOT_CHECK=1`, for a block made hot or about to be
/// guarded again to test it: whether it is as a hot block must be, in none
/// of the guards' sets (`protected`, `clean` and its bits, `absent`,
/// `aliased`), not read-only, and every other view of its page hot too
/// (else a write through it would not be seen there). Counted, the first
/// few that are not said in the log.
#[cfg(feature = "diagnostics")]
fn check_hot(g: &Guards, block: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOT_CHECK").is_some_and(|v| v == "1")) {
        return;
    }
    HOT_CHECKS[0].fetch_add(1, Relaxed);
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    let read_only = n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READONLY;
    let shown = if view_guard() { crate::inplace::pages_of(block * BLOCK) } else { Vec::new() };
    let cold_views: Vec<usize> = if (2..=MANY_VIEWS).contains(&shown.len()) {
        shown.into_iter().filter(|at| !g.hot.contains_key(&(at / BLOCK))).collect()
    } else {
        Vec::new()
    };
    let sets = [
        g.protected.contains(&block),
        g.clean.contains(block) || (keep_bits() && g.clean.has(block)),
        g.absent.contains(&block),
        g.aliased.contains(&block),
    ];
    if (read_only || !cold_views.is_empty() || sets.iter().any(|&b| b)) && HOT_CHECKS[1].fetch_add(1, Relaxed) < 32 {
        tracing::warn!(
            "VICTORY_GUARD_HOT_CHECK: hot block {:#x} is not as a hot block must be: read-only {read_only}, \
             protected/clean/absent/aliased {sets:?}, other views of its page not hot: {cold_views:x?}",
            block * BLOCK
        );
    }
}

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    });
    if ask {
        return false;
    }
    // The light refresh: a block known by its bit is read-only by our hand
    // or a hole the guards noted, which is what they are asked below.
    if light() && PUBLISHED.all(start / BLOCK, start / BLOCK + 1) {
        return true;
    }
    guards().known(start / BLOCK, summary())
}

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: Clean,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
    /// With [`view_known`]: blocks made read-only at least once since the
    /// runtime last mapped or unmapped them. They are view memory whether
    /// their guard stands now or a write has opened it.
    viewed: BTreeSet<usize>,
    /// With [`hot_frames`]: hot blocks, left writable and reported written
    /// at every asking ([`Hotness`]). None is in `protected`, `clean`,
    /// `absent` or `aliased`, and the other views of a hot block's page are
    /// hot too.
    hot: BTreeMap<usize, Hotness>,
    /// With [`hot_frames`]: blocks found written lately ([`Streak`]; at
    /// each new frame those not found written in the frame before are let
    /// go of).
    streak: BTreeMap<usize, Streak>,
    /// With [`hot_frames`]: blocks guarded again for being reported more
    /// often than they had been found written ([`hot_ratio`]), and the
    /// frame until which they are not made hot again.
    cool: BTreeMap<usize, u64>,
    /// The frame `streak` and `cool` were last pruned at.
    pruned: u64,
    /// With [`absent_known`]: runs of wholly uncommitted blocks Windows was
    /// asked about, first block to end (exclusive), none overlapping; with
    /// `=1`, those of frame `uncommitted_frame`.
    uncommitted: BTreeMap<usize, usize>,
    uncommitted_frame: u64,
    /// With the light refresh and holes answered from runs ([`holes_known`]):
    /// the blocks of `absent` that [`publish`] has looked at. Those that lie
    /// in a placeholder reservation ([`placeholder`]) are known by their
    /// bits ([`PUBLISHED`]) until the runtime tells of them.
    holes: BTreeSet<usize>,
}

/// Whether a block once guarded stays known as view memory while its
/// guard is open (`VICTORY_VIEW_KNOWN=1`; off by default).
///
/// A range whose first block the title wrote since the last check has
/// that block neither read-only nor noted absent, so nothing says it is a
/// view: the kernel's write watch is asked and fails, and then Windows is
/// asked what the memory is (`is_view`), a walk of the page tables to the
/// end of the pages that share the first one's state. In runs 222 and 225
/// that was 338-450 failed askings a frame and 6.5-7.8 ms a frame of the
/// submitting thread in `VirtualQuery`. With the switch the block is
/// remembered from the time it was first made read-only until the runtime
/// maps or unmaps it ([`title_memory_remapped`]).
///
/// `VICTORY_VIEW_KNOWN=alt`: twenty seconds with, twenty without, in turn,
/// each change said in the log, for measuring it in one scene. (Blocks
/// noted in one phase are forgotten at a remapping in either.)
fn view_known() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_VIEW_KNOWN").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "view memory stays known while its guard is open from now (VICTORY_VIEW_KNOWN=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

impl Guards {
    /// Whether `block` is view memory the guards hold: read-only by our
    /// hand, noted absent, or hot (or, with [`view_known`], once guarded). With
    /// `summed` ([`summary`]) a clean block, nearly every one asked about,
    /// is found by its bit first (every clean block is read-only by our
    /// hand), not looked up in the sets.
    fn known(&self, block: usize, summed: bool) -> bool {
        (summed && self.clean.has(block))
            || self.protected.contains(&block)
            || self.absent.contains(&block)
            || self.hot.contains_key(&block)
            || (view_known() && self.viewed.contains(&block))
    }
}

/// Blocks a word of [`Clean`]'s bits stands for: 4 MiB of view memory.
const WORD: usize = 64;

/// [`Guards::clean`]: the blocks read-only by our hand and not written
/// since a copy took them, in a set; and with [`summary`] the same blocks
/// as bits, one per block, 64 to a word.
///
/// In a steady scene the submitting thread asks [`guard`] some 3,600 times
/// a frame which blocks of a range were written, about 13 GB of view
/// memory (200,000 blocks), nearly all of it clean. Each asking read the
/// range's clean blocks off the set to count them, under the lock a title
/// thread's write fault also takes ("guards" 31.7 ms a frame of the
/// submitting thread's ~143). With the bits a range is found all clean
/// with one lookup per 4 MiB, and a range that is not is walked block by
/// block only in the words that are not all clean.
struct Clean {
    /// The blocks, in order.
    set: BTreeSet<usize>,
    /// With [`summary`] (always in tests): word number (block / [`WORD`])
    /// to its bits, bit `n` set if block `word * WORD + n` is clean. A word
    /// with no clean block is left out.
    bits: HashMap<usize, u64, BuildHasherDefault<WordHash>>,
}

impl Clean {
    const fn new() -> Self {
        Self {
            set: BTreeSet::new(),
            bits: HashMap::with_hasher(BuildHasherDefault::new()),
        }
    }

    fn contains(&self, block: usize) -> bool {
        self.set.contains(&block)
    }

    fn insert(&mut self, block: usize) {
        self.set.insert(block);
        if keep_bits() {
            *self.bits.entry(block / WORD).or_insert(0) |= 1 << (block % WORD);
        }
    }

    fn remove(&mut self, block: usize) {
        self.set.remove(&block);
        if keep_bits()
            && let Some(bits) = self.bits.get_mut(&(block / WORD))
        {
            *bits &= !(1 << (block % WORD));
            if *bits == 0 {
                self.bits.remove(&(block / WORD));
            }
        }
    }

    /// Notes blocks `[from, to)` clean.
    fn extend(&mut self, from: usize, to: usize) {
        self.set.extend(from..to);
        if keep_bits() {
            let mut at = from;
            while at < to {
                let end = word_end(at).min(to);
                *self.bits.entry(at / WORD).or_insert(0) |= span(at, end);
                at = end;
            }
        }
    }

    /// The bits of word `word` (bit `n`: block `word * WORD + n` is clean).
    fn word(&self, word: usize) -> u64 {
        self.bits.get(&word).copied().unwrap_or(0)
    }

    /// Whether the bits say `block` is clean.
    fn has(&self, block: usize) -> bool {
        self.word(block / WORD) & (1 << (block % WORD)) != 0
    }

    /// Whether the bits say every block of `[first, last)` is clean.
    fn all(&self, first: usize, last: usize) -> bool {
        let mut at = first;
        while at < last {
            let end = word_end(at).min(last);
            let want = span(at, end);
            if self.word(at / WORD) & want != want {
                return false;
            }
            at = end;
        }
        true
    }

    /// Whether the bits say of every block of `[first, last)` what the set
    /// says: the check of `VICTORY_GUARD_SUMMARY_CHECK`, with the walk of
    /// the set the askings made before the bits.
    #[cfg(any(test, feature = "diagnostics"))]
    fn agrees(&self, first: usize, last: usize) -> bool {
        let mut set = self.set.range(first..last).peekable();
        let mut at = first;
        while at < last {
            let end = word_end(at).min(last);
            let mut from_set = 0u64;
            while let Some(block) = set.next_if(|&&b| b < end) {
                from_set |= 1 << (block % WORD);
            }
            if self.word(at / WORD) & span(at, end) != from_set {
                return false;
            }
            at = end;
        }
        true
    }
}

/// The first block of the word after the one holding `block`.
fn word_end(block: usize) -> usize {
    (block / WORD + 1) * WORD
}

/// The bits of blocks `[from, to)`, which lie in one word.
fn span(from: usize, to: usize) -> u64 {
    let n = to - from;
    let low = if n >= WORD { u64::MAX } else { (1 << n) - 1 };
    low << (from % WORD)
}

/// Hashes a word number with one multiplication: the default hasher costs
/// more than the rest of a lookup of [`Clean`]'s bits.
#[derive(Default)]
struct WordHash(u64);

impl std::hash::Hasher for WordHash {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    fn write_usize(&mut self, n: usize) {
        self.0 = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// Whether [`guard`] answers from [`Clean`]'s bits
/// (`VICTORY_GUARD_SUMMARY=1`; off by default). Off, the bits are not kept
/// and every asking reads its range's clean blocks off the set, as before.
///
/// The answers are the same. A range the bits find all clean is one the
/// set holds every block of. One that is not is walked as before, by the
/// same steps ([`step`]), each block's being clean read off its word's
/// bits instead of the set, and the words found all clean passed over
/// whole: at a clean block the walk does nothing but end a run of written
/// blocks before it. Also, a range whose first block is clean is known as
/// view memory by its bit, and an asking takes the lock once instead of
/// twice (whether the range is a view, and its answer, under one holding).
///
/// `VICTORY_GUARD_SUMMARY_CHECK=1` (diagnostics) also reads the set one
/// asking in [`CHECK_EVERY`] and counts the askings where the bits said
/// otherwise of a block (the "recording per frame" line, since the start:
/// it must stay 0).
fn summary() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_SUMMARY.get() {
        return on;
    }
    summary_switch()
}

/// `VICTORY_GUARD_SUMMARY`, as set.
fn summary_switch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_SUMMARY").is_some_and(|v| v == "1"))
}

/// Whether [`Clean`] keeps its bits: with the switch, and in tests always,
/// so that each test thread can ask either way (`TEST_SUMMARY`).
fn keep_bits() -> bool {
    cfg!(test) || summary_switch()
}

/// [`summary`], for the frame line.
#[cfg(feature = "diagnostics")]
pub(crate) fn guard_summary() -> bool {
    summary()
}

/// The frame line's clause on hot blocks (`VICTORY_GUARD_HOT`), given its
/// counters per frame; empty while the switch is off.
#[cfg(feature = "diagnostics")]
pub(crate) fn hot_clause(made: f64, reported: f64, retests: f64, cooled: f64) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let frames = hot_frames();
    if frames == 0 {
        return String::new();
    }
    let (now, lately, cool) = {
        let g = guards();
        (g.hot.len(), g.streak.len(), g.cool.len())
    };
    let checks = if std::env::var_os("VICTORY_GUARD_HOT_CHECK").is_some_and(|v| v == "1") {
        format!(", checks since the start: {} of {} failed", HOT_CHECKS[1].load(Relaxed), HOT_CHECKS[0].load(Relaxed))
    } else {
        String::new()
    };
    format!(
        "; hot blocks (VICTORY_GUARD_HOT={frames}, ratio {}): {now} now ({lately} found written lately, {cool} kept \
         cold), {made:.1} made hot, {reported:.0} reported written for being hot ({:.1} MB), {retests:.1} guarded \
         again to test, {cooled:.1} for being reported too often{checks}",
        hot_ratio(),
        reported * BLOCK as f64 / 1e6
    )
}

/// One check of `VICTORY_GUARD_SUMMARY_CHECK` in this many askings.
#[cfg(feature = "diagnostics")]
const CHECK_EVERY: u64 = 64;

/// `VICTORY_GUARD_SUMMARY_CHECK`, since the start: askings checked, and
/// of them the ones where the bits differed from the set.
#[cfg(feature = "diagnostics")]
pub(crate) static SUMMARY_CHECKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// With `VICTORY_GUARD_SUMMARY_CHECK=1`, one asking in [`CHECK_EVERY`]:
/// reads the clean blocks of `[first, last)` off the set, as the askings
/// did before the bits, and counts the asking if the bits differ (the
/// first few said in the log).
#[cfg(feature = "diagnostics")]
fn check_summary(g: &Guards, first: usize, last: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static TURN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_SUMMARY_CHECK").is_some_and(|v| v == "1"))
        || !TURN.fetch_add(1, Relaxed).is_multiple_of(CHECK_EVERY)
    {
        return;
    }
    SUMMARY_CHECKS[0].fetch_add(1, Relaxed);
    if !g.clean.agrees(first, last) && SUMMARY_CHECKS[1].fetch_add(1, Relaxed) < 32 {
        tracing::warn!(
            "VICTORY_GUARD_SUMMARY_CHECK: the summary of clean blocks differs from the set over {:#x}..{:#x}: \
             {} of {} blocks clean in the set, all clean by the summary: {}",
            first * BLOCK,
            last * BLOCK,
            g.clean.set.range(first..last).count(),
            last - first,
            g.clean.all(first, last)
        );
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: [`summary`] for this thread, whatever the environment says.
    static TEST_SUMMARY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Tests: [`hot_frames`], [`hot_ratio`] and the frame ([`frame_now`])
    /// for this thread.
    static TEST_HOT: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static TEST_HOT_RATIO: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    /// Tests: [`views_batch`] for this thread.
    static TEST_VIEWS_BATCH: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Tests: [`absent_known`] for this thread, and the questions whether
    /// memory is uncommitted it asked.
    static TEST_ABSENT_KNOWN: std::cell::Cell<Option<AbsentKnown>> = const { std::cell::Cell::new(None) };
    static TEST_QUESTIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static TEST_FRAME: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: Clean::new(),
    absent: BTreeSet::new(),
    aliased: BTreeSet::new(),
    viewed: BTreeSet::new(),
    hot: BTreeMap::new(),
    streak: BTreeMap::new(),
    cool: BTreeMap::new(),
    pruned: 0,
    uncommitted: BTreeMap::new(),
    uncommitted_frame: 0,
    holes: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// [`guards`] for an asking of [`guard`]. With diagnostics a wait for the
/// lock (a title thread's write fault holds it while it opens a block, or
/// another asking while it walks) is counted and timed.
fn guards_waited() -> std::sync::MutexGuard<'static, Guards> {
    #[cfg(feature = "diagnostics")]
    {
        match GUARDS.try_lock() {
            Ok(g) => return g,
            Err(std::sync::TryLockError::Poisoned(e)) => return e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {}
        }
        let waiting = std::time::Instant::now();
        let g = guards();
        crate::copystat::add(crate::copystat::C::GuardWaits, 1);
        crate::copystat::add(crate::copystat::C::GuardWaitNs, waiting.elapsed().as_nanos() as u64);
        g
    }
    #[cfg(not(feature = "diagnostics"))]
    guards()
}

/// Whether the block `page` is read-only by our hand and unwritten since a
/// copy took it (diagnostics of `VICTORY_BACK_OWN_CHECK`).
pub(crate) fn clean(page: usize) -> bool {
    guards().clean.contains(page)
}

/// `VICTORY_LIGHT_REFRESH`: 0 off, 1 on, 2 in turn ([`light`]).
fn light_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_LIGHT_REFRESH").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether the bits of the light refresh are kept ([`PUBLISHED`]): with
/// the switch on, and in both phases of `alt`, so that the phase that asks
/// them finds them true whatever was written in the phase before.
pub(crate) fn light_kept() -> bool {
    light_mode() != 0
}

/// Whether an asking about view memory is answered from the bits of its
/// blocks where they can answer (`VICTORY_LIGHT_REFRESH=1`; off by
/// default). Ported from fxceless/victory-fork 94ed363c (fxceless, after
/// avatarmanz/VIctory-7900XTX 2aa8134b and 74045e46), as reconciled with
/// our guards in `fx-main-forkmerge` 78162a4.
///
/// A frame asks the guards about view memory some 3,200 times to hear,
/// nearly always, that nothing was written; every such asking takes the
/// guards' lock, which the title's threads wait for in their write faults,
/// and walks the range's blocks that are not all clean, the holes of view
/// memory among them (PF1: 96,500 blocks walked a frame, 43,600 of them
/// holes; 16 ms). With the switch a range whose blocks all have their bit
/// ([`PUBLISHED`]) is answered from the journal alone ([`writes`],
/// [`nothing_new`]), and its first block is known for view memory without
/// the lock ([`known_view`]).
///
/// What changes in the answers: nothing for blocks read-only by our hand
/// (a range is known by its bits exactly while the guards would find it
/// all clean). A hole gets a bit only where our notes answer for holes
/// already (`VICTORY_GUARD_ABSENT_KNOWN=1` or `forever`, [`holes_known`]),
/// and only in a placeholder reservation, where pages come to be only
/// through the runtime's mappings, which it tells of
/// ([`title_memory_remapped`]); [`look_at_holes`] asks a few of them of
/// Windows again at each submission. (The fork asks for its own notes kept
/// for good, `VICTORY_ABSENT_ONCE=1 VICTORY_ABSENT_MS=told`; ours are the
/// runs of `VICTORY_GUARD_ABSENT_KNOWN`. With `=1` the guards ask Windows
/// once a frame about a hole; with the bits a placeholder's hole is asked
/// only by that look, as with `=forever`.)
///
/// `VICTORY_LIGHT_REFRESH=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log with the counts of the phase before.
/// The bits are kept in both phases ([`light_kept`]) and asked in one.
pub(crate) fn light() -> bool {
    match light_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                let [bits, locked, filled] = light_asks();
                tracing::info!(
                    "refreshes of host copies go the light way from now (VICTORY_LIGHT_REFRESH=alt): {}; since the change before: {bits} askings of view memory answered by the bits of their blocks, {locked} that took the guards' lock, {filled} holes found filled with nobody telling",
                    if on { "ON" } else { "off, as usual (the bits are kept)" }
                );
            }
            on
        }
    }
}

/// Since they were last taken ([`light_asks`]), with the light refresh
/// kept: askings of view memory answered by the bits of their blocks,
/// askings that went to the guards, and holes known by their bits that
/// Windows said were filled with nobody telling ([`look_at_holes`]).
static LIGHT_ASKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Counts one in [`LIGHT_ASKS`]`[i]` (and on the copies line).
fn light_count(i: usize) {
    LIGHT_ASKS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::C;
        let counter = match i {
            0 => C::LightBits,
            1 => C::LightGuards,
            _ => C::LightFilled,
        };
        crate::copystat::add(counter, 1);
    }
}

/// [`LIGHT_ASKS`], taken.
fn light_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| LIGHT_ASKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The words of a piece of [`Bits`]: the bits of 4 GiB.
const BITS_WORDS: usize = 1 << 10;
/// The pieces of [`Bits`]: 128 TiB, what a process has.
const BITS_PIECES: usize = 1 << 15;

/// A bit per 64 KiB block of the address space, read and cleared with no
/// lock; in pieces made when first needed and never let go.
struct Bits([std::sync::atomic::AtomicPtr<BitsPiece>; BITS_PIECES]);

/// The bits of 4 GiB.
type BitsPiece = [std::sync::atomic::AtomicU64; BITS_WORDS];

impl Bits {
    const fn new() -> Bits {
        Bits([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; BITS_PIECES])
    }

    /// The piece that holds word `word`, if it was ever made.
    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let piece = self.0.get(word / BITS_WORDS)?.load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: null, or a piece made by `set`, which is never freed.
        unsafe { piece.as_ref() }
    }

    /// Sets `block`'s bit (nothing past the address space).
    fn set(&self, block: usize) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let word = block / 64;
        let Some(place) = self.0.get(word / BITS_WORDS) else {
            return;
        };
        if place.load(Acquire).is_null() {
            let made: *mut BitsPiece = Box::into_raw(Box::new([const { std::sync::atomic::AtomicU64::new(0) }; BITS_WORDS]));
            if place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire).is_err() {
                // SAFETY: made above and shown to nobody: another thread's
                // piece is in the place.
                drop(unsafe { Box::from_raw(made) });
            }
        }
        if let Some(piece) = self.piece(word) {
            piece[word % BITS_WORDS].fetch_or(1u64 << (block % 64), SeqCst);
        }
    }

    /// Clears `block`'s bit: ahead of what makes the block writable, and
    /// seen by whoever looks after it.
    fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            piece[word % BITS_WORDS].fetch_and(!(1u64 << (block % 64)), std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Whether the bits of all of `[first, last)` are set (not so of none).
    fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            let Some(piece) = self.piece(word) else {
                return false;
            };
            if piece[word % BITS_WORDS].load(std::sync::atomic::Ordering::SeqCst) & mask != mask {
                return false;
            }
            block = until;
        }
        first < last
    }
}

/// The blocks an asking can answer for from the journal alone. A block has
/// its bit only while both of these hold: it is read-only by our hand and
/// noted clean (`Guards::clean`), or it is a hole our notes answer for that
/// lies in a placeholder reservation (`Guards::holes`); and the journal
/// holds every write to it that an asking found.
///
/// Why that is so. A bit is set in one place, [`publish`]: by the asking
/// that found the block written and had it made clean, after it noted that
/// in the journal and with the journal's lock still held, under the
/// guards' lock, and only if the block is clean at that moment. It is
/// cleared under the guards' lock before the block's write access is given
/// back (`open_here`, which the write fault's handler, [`writing`],
/// [`title_memory_written`] and [`views_mapped`] all go through; and
/// `make_hot`), and when the runtime tells that what the block shows is new
/// ([`title_memory_remapped`]). Clearing does nothing where no bit was ever
/// set, so it is not behind the switch.
///
/// Who asks the bits holds the journal's lock from before the look until
/// it has read the journal ([`published`]), and nobody sets a bit without
/// that lock: a block opened and guarded again between the look and the
/// reading cannot have its bit back in time to be taken for unwritten. A
/// write that comes while the bits are looked at is no different from one
/// that comes just after an asking: its block's bit is gone, and the next
/// asking goes to the guards and finds the block open.
///
/// Our own stores through a view of our own (`crate::inplace::write_own`)
/// open no guard and take no bit: they are noted in the journal
/// ([`wrote_own`]), which is where an asking answered by the bits looks.
static PUBLISHED: Bits = Bits::new();

/// Whether holes of view memory can be known by their bits: our notes
/// answer for them already (`VICTORY_GUARD_ABSENT_KNOWN`, [`absent_known`]),
/// so an asking that walks a hole puts no question to Windows either.
fn holes_known() -> bool {
    absent_known() != AbsentKnown::Off
}

/// An asking of view memory `[start, end)` found the pages `written` (page
/// numbers) written, had the guards make their blocks read-only and clean
/// ([`guard`]) and has noted them in the journal, whose lock it holds
/// still (`_noted`): the blocks of them that are clean now are known by
/// their bits from here on ([`PUBLISHED`]). With `holes` ([`holes_known`]),
/// so are the blocks of the range noted absent and answered from a run of
/// uncommitted blocks that lie in a placeholder reservation; Windows is
/// asked that once for each ([`placeholder`]).
///
/// The blocks an asking leaves clean that have no bit yet are the ones it
/// reports written: every other clean block was left clean by an asking
/// before it, which did this.
fn publish(_noted: &Journal, written: &[usize], start: usize, end: usize, holes: bool) {
    let mut g = guards();
    let g = &mut *g;
    for &page in written {
        let block = page * PAGE / BLOCK;
        // (Not one that a write has opened since the guards were asked,
        // nor one that could not be made read-only.)
        if g.clean.contains(block) {
            PUBLISHED.set(block);
        }
    }
    if !holes {
        return;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The holes of the range not looked at yet that a run answers for. Both
    // sets are read in block order.
    let fresh: Vec<usize> = {
        let mut looked = g.holes.range(first..last).copied().peekable();
        let uncommitted = &g.uncommitted;
        g.absent
            .range(first..last)
            .copied()
            .filter(|&block| {
                while looked.next_if(|&hole| hole < block).is_some() {}
                looked.next_if_eq(&block).is_none()
                    && uncommitted.range(..=block).next_back().is_some_and(|(_, &end)| block < end)
            })
            .collect()
    };
    for block in fresh {
        g.holes.insert(block);
        // (Asked now, not when the block was noted: a view the runtime is
        // mapping there, and has not told of yet, is no placeholder.)
        if placeholder(block) {
            PUBLISHED.set(block);
        }
    }
}

/// Whether every block of `[start, end)` has its bit ([`PUBLISHED`]): what
/// [`writes`] would answer of the range is in the journal. `_held`: the
/// journal's lock, which the caller holds from before this look until it
/// has read the journal (no bit is set meanwhile: [`publish`]).
fn published(_held: &Journal, start: usize, end: usize) -> bool {
    PUBLISHED.all(start / BLOCK, end.div_ceil(BLOCK))
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's question about a process's memory, `(process, address,
    /// class of the answer, answer out, its room, bytes answered out)`: an
    /// NTSTATUS, negative when it cannot answer. It changes nothing.
    fn NtQueryVirtualMemory(
        process: *mut std::ffi::c_void,
        at: *const std::ffi::c_void,
        class: u32,
        out: *mut std::ffi::c_void,
        len: usize,
        returned: *mut usize,
    ) -> i32;
}

/// Whether `block` lies in a placeholder reservation: memory in which pages
/// come to be only by mapping a view, or making an allocation, in its
/// place, which the runtime does and tells of ([`title_memory_remapped`]).
/// An ordinary reservation can be committed by anybody, and free memory
/// taken by anybody, with no word to us: `VirtualQuery` does not tell the
/// three apart, this question does.
fn placeholder(block: usize) -> bool {
    /// `MEMORY_REGION_INFORMATION` (ntddk); only the kinds are read.
    #[repr(C)]
    #[derive(Default)]
    #[allow(dead_code)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    // (`MemoryRegionInformationEx`; class 3 answers with the kind of memory
    // in that word, not the flags.)
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let mut region = Region::default();
    // SAFETY: the answer's room and its size; the kernel checks the address.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    status >= 0 && region.region_type & PLACEHOLDER_RESERVATION != 0
}

/// The note that `block` is uncommitted is gone, or is true no longer: the
/// block is not known by its bit any more ([`publish`]).
fn forget_hole(g: &mut Guards, block: usize) {
    if g.holes.remove(&block) {
        PUBLISHED.clear(block);
    }
}

/// Holes asked of Windows again at each submission ([`look_at_holes`]): a
/// question is a microsecond or so, and the submitting thread is the
/// frame's limit.
const HOLES_A_LOOK: usize = 16;

/// With the light refresh: `most` of the holes known by their bits are
/// asked of Windows again, the next ones each time and round again, in
/// case one was filled with nobody telling ([`title_memory_remapped`]),
/// which this runtime is not known to do. One that was loses its bit and
/// the runs that answered for it, so that the next asking of it goes to
/// Windows, which finds it; counted, and said once. One whose reservation
/// is a placeholder no longer (let go, or made anew) loses its bit.
///
/// Not while the runtime changes a mapping: a view it has mapped and not
/// told of yet is no hole filled behind our back, and forgotten here it
/// could be made read-only before the runtime's word, which forgets the
/// guard and leaves the protection.
fn look_at_holes(most: usize) {
    static FROM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let Some(_still) = title_memory_reading_now() else {
        return;
    };
    let part: Vec<usize> = {
        let g = guards();
        if g.holes.is_empty() {
            return;
        }
        let from = FROM.load(std::sync::atomic::Ordering::Relaxed);
        let part: Vec<usize> = g.holes.range(from..).take(most).copied().collect();
        FROM.store(part.last().map_or(0, |last| last + 1), std::sync::atomic::Ordering::Relaxed);
        part
    };
    for block in part {
        // (Not the ones looked at that lie in no placeholder: they have no
        // bit, and the guards' walk answers for them.)
        if !PUBLISHED.all(block, block + 1) {
            continue;
        }
        let filled = block >= uncommitted_end(block);
        if !filled && placeholder(block) {
            continue;
        }
        let mut g = guards();
        // (Told of since the part was taken: forgotten already.)
        if !g.holes.contains(&block) {
            continue;
        }
        PUBLISHED.clear(block);
        if filled {
            g.holes.remove(&block);
            forget_uncommitted(&mut g, block, block + 1);
            light_count(2);
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::warn!(
                    address = format_args!("{:#x}", block * BLOCK),
                    "a hole of view memory was filled with nobody telling: host copies of it held old bytes until now (VICTORY_LIGHT_REFRESH; said once, counted with the light refresh's askings)"
                );
            });
        }
    }
}

/// Whether nothing of `[start, end)` was written that a copy does not
/// hold, known with nothing asked of the guards or the kernel: `copied` is
/// the copy's note, for each page from page `base` on, of the asking after
/// which it last took the page (0: never), as `crate::shadow` keeps it.
/// True only if [`writes`] would answer from the journal (the range was
/// asked ahead, or its blocks are known by their bits) and the answer would
/// have the copy take no page. `false` if something would have to be
/// asked, and always without the light refresh ([`light`]).
pub(crate) fn nothing_new(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    light() && nothing_noted(start, end, base, copied)
}

/// [`nothing_new`], whatever the switch says.
fn nothing_noted(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    if start >= end {
        return false;
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base)) else {
        return false;
    };
    // A page never taken is taken at the next refresh (so is, at every
    // refresh, a page the copy's refreshes only ever hold in part).
    if taken.contains(&0) {
        return false;
    }
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    // The bits first, then the journal, both under its lock ([`published`]).
    let by_bits = !j.ahead.holds(first, last);
    if by_bits && !published(&j, first * PAGE, last * PAGE) {
        return false;
    }
    let same = j.written.range(first, last).all(|(page, write)| write <= taken[page - first]);
    if same && by_bits {
        light_count(0);
    }
    same
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
/// With [`summary`] the clean blocks are read off [`Clean`]'s bits.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"));
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let summed = summary();
    let mut g = if summed {
        // The question and the answer under one holding of the lock, when
        // the first block says the range is a view (and the handler is in).
        let g = guards_waited();
        let known = !ask && g.known(first, true);
        if known && INSTALLED.is_completed() {
            g
        } else {
            drop(g);
            if !known && !is_view(start, end) {
                return None;
            }
            install();
            guards_waited()
        }
    } else {
        let known = !ask && guards_waited().known(first, false);
        if !known && !is_view(start, end) {
            return None;
        }
        install();
        guards_waited()
    };
    let mut written = Vec::new();
    #[cfg(feature = "diagnostics")]
    let checking = std::time::Instant::now();
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::GuardAsked, 1);
    // Counts an asking found all clean, and the time finding it took.
    #[cfg(feature = "diagnostics")]
    let all_clean = |checking: std::time::Instant| {
        use crate::copystat::{C, add};
        add(C::GuardAllClean, 1);
        add(C::GuardCheckNs, checking.elapsed().as_nanos() as u64);
    };
    if summed {
        #[cfg(feature = "diagnostics")]
        check_summary(&g, first, last);
        // Nothing written, mapped or new since the range was last taken.
        if g.clean.all(first, last) {
            #[cfg(feature = "diagnostics")]
            all_clean(checking);
            return Some(written);
        }
        #[cfg(feature = "diagnostics")]
        let walking = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardCheckNs, (walking - checking).as_nanos() as u64);
        let mut tally = Tally::default();
        let hot = Hot::now(&mut g);
        let _walked = walk_summed(&mut g, first, last, &mut written, &mut tally, hot);
        tally.count();
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::GuardWalked, _walked);
            add(C::GuardWalkNs, walking.elapsed().as_nanos() as u64);
        }
        return Some(written);
    }
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    let clean: Vec<usize> = g.clean.set.range(first..last).copied().collect();
    if clean.len() == last - first {
        // Nothing written, mapped or new since the range was last taken.
        #[cfg(feature = "diagnostics")]
        all_clean(checking);
        return Some(written);
    }
    #[cfg(feature = "diagnostics")]
    let walking = std::time::Instant::now();
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::GuardCheckNs, (walking - checking).as_nanos() as u64);
    let mut run: Option<usize> = None;
    let mut next_clean = 0;
    let mut uncommitted_until = first;
    let mut tally = Tally::default();
    let hot = Hot::now(&mut g);
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let walk = Walk { run: &mut run, uncommitted_until: &mut uncommitted_until, written: &mut written, tally: &mut tally, hot };
        step(&mut g, block, block >= last || is_clean, walk);
    }
    tally.count();
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::GuardWalked, (last - first) as u64);
        add(C::GuardWalkNs, walking.elapsed().as_nanos() as u64);
    }
    Some(written)
}

/// One block of [`guard`]'s walk. `clean`: the block was clean when the
/// asking began, or lies past the range (nothing to open there). A block
/// not clean is reported written, unless it is noted absent and still
/// wholly uncommitted. One still read-only by our hand (written through the
/// runtime) is noted clean again; the others are guarded again in runs,
/// each run ending at the next block that needs no guarding. With
/// [`hot_frames`], a hot block is left writable and ends the run before it
/// ([`Hot::keeps`]).
fn step(g: &mut Guards, block: usize, clean: bool, walk: Walk<'_>) {
    let Walk { run, uncommitted_until, written, tally, hot } = walk;
    let mut open = !clean;
    if open && g.absent.contains(&block) {
        // Ordinary commits need no runtime mapping notification. Reuse
        // this asking's query for the following blocks in the same region
        // (with [`absent_known`], the frame's or an earlier one's).
        if block >= *uncommitted_until {
            *uncommitted_until = uncommitted_run(g, block, tally);
        }
        open = block >= *uncommitted_until;
        if open {
            // Committed with nothing said: a mapping of the runtime's
            // takes its blocks out of `absent` (`title_memory_remapped`).
            g.absent.remove(&block);
            forget_hole(g, block);
            tally.found += 1;
        } else {
            tally.absent += 1;
        }
    }
    if open {
        written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
        tally.written += 1;
        let guarded = g.protected.contains(&block);
        if guarded {
            // Written through the runtime, or through another view of the
            // page guarded at that view's asking: still read-only.
            g.clean.insert(block);
            tally.guarded += 1;
        }
        match hot.keeps(g, block, tally) {
            Keep::Guard if guarded => {}
            Keep::Guard => {
                run.get_or_insert(block);
                return;
            }
            // Left writable.
            Keep::Hot => {}
            Keep::Make(most) => {
                // The run before it first: another view of this page may
                // be in it, and is made hot too.
                if let Some(from) = run.take() {
                    protect(g, from, block);
                }
                make_hot(g, block, hot.frame, most);
                return;
            }
        }
    }
    if let Some(from) = run.take() {
        protect(g, from, block);
    }
}

/// The state one asking's walk carries from block to block ([`step`]):
/// the run of written blocks to guard again, the end of the uncommitted
/// memory last asked about, the pages reported, what was found, and
/// [`hot_frames`] with this frame.
struct Walk<'a> {
    run: &'a mut Option<usize>,
    uncommitted_until: &'a mut usize,
    written: &'a mut Vec<usize>,
    tally: &'a mut Tally,
    hot: Hot,
}

/// [`guard`]'s walk of `[first, last)` with [`summary`]: the words of
/// [`Clean`]'s bits all clean over the range are passed over whole (the
/// walk would only end a run of written blocks at their first block), the
/// others are walked block by block. Returns the blocks walked one by one.
///
/// A word's bits are read when the walk comes to it: what the walk notes
/// clean meanwhile is the block it stands at or ones before it (the runs it
/// guards again end there), so the later blocks read as they were when the
/// asking began, as the set read once at the start gave them.
fn walk_summed(g: &mut Guards, first: usize, last: usize, written: &mut Vec<usize>, tally: &mut Tally, hot: Hot) -> u64 {
    let mut run: Option<usize> = None;
    let mut uncommitted_until = first;
    let mut walked = 0;
    let mut block = first;
    while block < last {
        let end = word_end(block).min(last);
        let bits = g.clean.word(block / WORD);
        let want = span(block, end);
        if bits & want == want {
            if let Some(from) = run.take() {
                protect(g, from, block);
            }
            block = end;
            continue;
        }
        walked += (end - block) as u64;
        while block < end {
            let clean = bits & (1 << (block % WORD)) != 0;
            let walk = Walk { run: &mut run, uncommitted_until: &mut uncommitted_until, written: &mut *written, tally: &mut *tally, hot };
            step(g, block, clean, walk);
            block += 1;
        }
    }
    let walk = Walk { run: &mut run, uncommitted_until: &mut uncommitted_until, written: &mut *written, tally: &mut *tally, hot };
    step(g, last, true, walk);
    walked
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    let (start, end) = (from.saturating_mul(BLOCK), to.saturating_mul(BLOCK));
    if views_batch() {
        // The same pages, found under one taking of the views' lock, and
        // their other views made read-only in runs.
        let shared = timed!(GuardViewLookups, GuardViewLookupNs, crate::inplace::shared_between(start, end, MANY_VIEWS));
        let mut others = Vec::new();
        for (page, shown) in shared {
            let block = page / BLOCK;
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block {
                    others.push(alias);
                }
            }
        }
        protect_aliases(g, others);
        return;
    }
    let views = timed!(GuardViewLookups, GuardViewLookupNs, crate::inplace::views_between(start, end));
    for (base, bytes) in views {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = timed!(GuardViewLookups, GuardViewLookupNs, crate::inplace::pages_of(block * BLOCK));
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            protect_with(g, block, shown);
        }
    }
}

/// [`protect`] for `block`, whose physical page every view in `shown`
/// shows (its own included): the others are made read-only, and all noted
/// guarded together.
fn protect_with(g: &mut Guards, block: usize, shown: Vec<usize>) {
    #[cfg(feature = "diagnostics")]
    ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    for alias in shown {
        let alias = alias / BLOCK;
        g.aliased.insert(alias);
        if alias != block {
            protect_alias(g, alias);
        }
    }
}

/// Whether [`protect`] finds the other views of a run's pages under one
/// taking of the views' lock and makes them read-only in runs
/// (`VICTORY_GUARD_VIEWS_BATCH=1`; off by default).
///
/// Off, the run's views are looked up, then each of their pages under the
/// lock again, its views copied out (`views_between`, `pages_of`), and each
/// other view is made read-only by a call of its own ([`protect_alias`]).
/// With the title's threads on every processor such a call costs some 7 us
/// however few its pages (each processor must forget them), and the other
/// views of a run of pages often lie side by side too (a section mapped
/// twice). On, the same pages are found under one taking of the lock
/// (`crate::inplace::shared_between`), and their other views are made
/// read-only by one call for each run of them ([`protect_aliases`]).
fn views_batch() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_VIEWS_BATCH.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_VIEWS_BATCH").is_some_and(|v| v == "1"))
}

/// [`protect_alias`] for each of `blocks` (other views of pages just
/// guarded), one call for each run of them side by side that is not
/// read-only by our hand already, block by block where a run's call fails.
fn protect_aliases(g: &mut Guards, mut blocks: Vec<usize>) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    blocks.sort_unstable();
    blocks.dedup();
    // Hot views of these pages are guarded with the others again.
    for b in &blocks {
        g.hot.remove(b);
    }
    blocks.retain(|b| !g.protected.contains(b));
    let mut i = 0;
    while i < blocks.len() {
        let mut k = i + 1;
        while k < blocks.len() && blocks[k] == blocks[k - 1] + 1 {
            k += 1;
        }
        let (from, to) = (blocks[i], blocks[k - 1] + 1);
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        let done = to - from > 1
            && timed!(GuardViewProtects, GuardViewProtectNs, unsafe {
                VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READONLY, &mut old)
            }) != 0;
        if done {
            g.protected.extend(from..to);
            if view_known() {
                g.viewed.extend(from..to);
            }
        } else {
            for &b in &blocks[i..k] {
                protect_alias(g, b);
            }
        }
        i = k;
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    // A hot view of the page is guarded with the others again
    // ([`hot_frames`]): reported written at its next asking, as below.
    g.hot.remove(&block);
    if g.protected.contains(&block) {
        return;
    }
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    let done = timed!(GuardViewProtects, GuardViewProtectNs, unsafe {
        VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old)
    });
    if done != 0 {
        g.protected.insert(block);
        if view_known() {
            g.viewed.insert(block);
        }
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        timed!(GuardProtects, GuardProtectNs, unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        })
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardProtected, (to - from) as u64);
        g.protected.extend(from..to);
        g.clean.extend(from, to);
        if view_known() {
            g.viewed.extend(from..to);
        }
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else {
            #[cfg(feature = "diagnostics")]
            let error = std::io::Error::last_os_error();
            if block < uncommitted_end(block) {
                g.absent.insert(block);
            } else {
                // Diagnostics: a committed block that will not be made
                // read-only is reported written at every asking and never
                // known to the guards (one in 1,024 said).
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GuardProtectFailed, 1);
                    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 1024 == 0 {
                        tracing::info!("a block of title memory could not be made read-only (one in 1024): {:#x}: {error}", block * BLOCK);
                    }
                }
            }
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`
/// (counted and timed with the walk's questions).
fn uncommitted_end(block: usize) -> usize {
    #[cfg(test)]
    TEST_QUESTIONS.set(TEST_QUESTIONS.get() + 1);
    timed!(GuardQueries, GuardQueryNs, ask_uncommitted_end(block))
}

/// [`uncommitted_end`], uncounted.
fn ask_uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    // The light refresh: no view of these pages is known by its bit once
    // one of them can be written (who asks the bits takes no lock of ours;
    // after avatarmanz/VIctory-7900XTX 689df313, as in the fork).
    if light_kept() && start < end && !g.aliased.is_empty() {
        for &block in g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)) {
            for alias in crate::inplace::pages_of(block * BLOCK) {
                PUBLISHED.clear(alias / BLOCK);
            }
        }
    }
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for alias in crate::inplace::pages_of(block * BLOCK) {
            if alias / BLOCK != block && open_here(g, alias, alias.saturating_add(BLOCK)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            g.aliased.remove(&(alias / BLOCK));
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(block);
        // Before the block can be written ([`PUBLISHED`]; nothing to do
        // where no bit was ever set).
        PUBLISHED.clear(block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    // The light refresh: all of them told as written before the first can
    // be written (as in [`open`]).
    if light_kept() {
        for &(at, n) in &ranges {
            for &block in g.protected.range(at / BLOCK..at.saturating_add(n).div_ceil(BLOCK)) {
                PUBLISHED.clear(block);
            }
        }
    }
    for &(at, n) in &ranges {
        // A hot block now has another view, which a write through it would
        // not open: guarded again, with its other views, at its next asking.
        unhot(&mut g, at, at.saturating_add(n));
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>, std::time::Instant);

impl Drop for Remapping {
    fn drop(&mut self) {
        // From asking for the lock until it is let go (the frame log).
        crate::framelog::add(crate::framelog::T::RemapUs, self.1.elapsed().as_micros() as u64);
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    let asked = std::time::Instant::now();
    let held = MAPPING.write().unwrap_or_else(|e| e.into_inner());
    crate::framelog::add(crate::framelog::T::Remaps, 1);
    crate::framelog::add(crate::framelog::T::RemapWaitUs, asked.elapsed().as_micros() as u64);
    Remapping(held, asked)
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // Holes found there are no longer known (`VICTORY_SPARSE_KNOWN`).
    crate::host::sparseknown::forget(at, len);
    crate::texback::remapped(at, len);
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range).copied().collect();
    for block in absent {
        g.absent.remove(&block);
        forget_hole(&mut g, block);
    }
    // Back to read-write before the guard is forgotten: the title can write
    // its view as soon as the runtime has given it the word, and a block left
    // read-only with no guard to open it would fault into no handler (the
    // same window `look_at_holes` avoids by taking the mapping lock). What it
    // shows is new ([`PUBLISHED`], cleared there too). After
    // fxceless/victory-fork#1447.
    open_here(&mut g, at, at.saturating_add(len));
    let aliased: Vec<usize> = g.aliased.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
    }
    // Nor hot, nor found written lately, nor kept cold; nor known
    // uncommitted.
    unhot(&mut g, at, at.saturating_add(len));
    let blocks = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    forget_uncommitted(&mut g, blocks.start, blocks.end);
    if !g.streak.is_empty() {
        let streak: Vec<usize> = g.streak.range(blocks.clone()).map(|(&b, _)| b).collect();
        for block in streak {
            g.streak.remove(&block);
        }
    }
    if !g.cool.is_empty() {
        let cool: Vec<usize> = g.cool.range(blocks).map(|(&b, _)| b).collect();
        for block in cool {
            g.cool.remove(&block);
        }
    }
    if view_known() {
        // The runtime calls this for two things only (`xmem`: `map` and
        // `unmap`): a range it has just mapped a section's pages into,
        // and a view it is about to make a placeholder again. Either way
        // what stands there is view memory or a hole in it, which is what
        // `is_view` would answer for a range that begins there; and the
        // write watch's asking can only fail there. The title remaps some
        // 83 times a frame, and each time the blocks were forgotten, so
        // that the next asking about each range went to the kernel twice
        // (run 235: 266 to 565 failed askings a frame with the blocks
        // remembered from their guards alone, no fewer than without).
        g.viewed.extend(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK));
        return;
    }
    // What the range shows is new: no longer known to be a view.
    let viewed: Vec<usize> = g.viewed.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).copied().collect();
    for block in viewed {
        g.viewed.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            crate::framelog::add(crate::framelog::T::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    // SAFETY: the handler only touches our guard state and protections.
    INSTALLED.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Done once the handler of [`install`] is registered.
static INSTALLED: std::sync::Once = std::sync::Once::new();

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod written_tests {
    use super::{BTreeMap, PIECE, Written};

    /// The pieces answer for any range what a tree of the same pages does
    /// (the notes and removals random, in two clusters, one across pieces'
    /// ends; after Biskwit/VIctory-reloaded 27b7a9002's test).
    #[test]
    fn the_pieces_say_what_a_tree_of_the_pages_says() {
        let mut written = Written::new();
        let mut tree = BTreeMap::new();
        let mut seed = 0x9e37_79b9_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as usize
        };
        for now in 1..5000u64 {
            let page = if now % 3 == 0 { 1_000_000 + next() % 300 } else { 10 * PIECE - 150 + next() % 300 };
            assert_eq!(written.insert(page, now), tree.insert(page, now).unwrap_or(0));
            if now % 50 == 0 {
                let gone = page + next() % 3;
                written.remove(gone);
                tree.remove(&gone);
            }
        }
        for _ in 0..2000 {
            let first = [0, 1_000_000 - 50, 10 * PIECE - 200][next() % 3] + next() % 400;
            let last = first + [0, 1, 2, 63, 64, 65, 500][next() % 7];
            let want: Vec<(usize, u64)> = tree.range(first..last).map(|(&p, &w)| (p, w)).collect();
            assert_eq!(written.range(first, last).collect::<Vec<_>>(), want, "{first}..{last}");
            assert_eq!(written.get(first), tree.get(&first).copied());
        }
        assert_eq!(written.range(5, 5).count(), 0);
        assert_eq!(written.range(9, 5).count(), 0);
    }
}

#[cfg(test)]
mod ahead_tests {
    use super::Ahead;

    /// What the journal asked before: one asked range that holds the range.
    fn scanned(ranges: &[(usize, usize)], first: usize, last: usize) -> bool {
        ranges.iter().any(|&(a, b)| a <= first && last <= b)
    }

    /// A xorshift, so the ranges are the same every run.
    fn numbers(mut x: u64) -> impl FnMut(usize) -> usize {
        move |below| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % below as u64) as usize
        }
    }

    /// Ranges overlapping, nested, touching, repeated and empty, pushed one
    /// by one (as [`super::ask_once`] does): the answer is the scan's for
    /// every range, those that span two touching ranges among them (no
    /// single range holds those, and none is merged). After ScarlettPort
    /// 50a9f7c5's test.
    #[test]
    fn holds_answers_as_the_scan_did() {
        let mut next = numbers(0x9e37_79b9_7f4a_7c15);
        for round in 0..200 {
            let count = next(40);
            let ranges: Vec<(usize, usize)> = (0..count)
                .map(|_| {
                    let a = next(300);
                    (a, a + next(60))
                })
                .collect();
            let mut pushed = Ahead::new();
            for (n, &range) in ranges.iter().enumerate() {
                pushed.push(range);
                for _ in 0..20 {
                    let first = next(380);
                    let last = first + next(80);
                    assert_eq!(
                        pushed.holds(first, last),
                        scanned(&ranges[..=n], first, last),
                        "round {round}: [{first}, {last}) after {:?}",
                        &ranges[..=n]
                    );
                }
            }
            for first in 0..380 {
                for last in first..first + 80 {
                    assert_eq!(
                        pushed.holds(first, last),
                        scanned(&ranges, first, last),
                        "round {round}: [{first}, {last}) in {ranges:?}"
                    );
                }
            }
        }
        let mut touching = Ahead::new();
        touching.push((20, 30));
        touching.push((10, 20));
        assert!(touching.holds(10, 20) && touching.holds(20, 30) && touching.holds(12, 12));
        assert!(!touching.holds(15, 25), "two ranges hold it, one does not");
        assert!(!Ahead::new().holds(0, 0));
        let mut cleared = Ahead::new();
        cleared.push((0, 100));
        cleared.clear();
        assert!(cleared.is_empty() && !cleared.holds(1, 2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// The journal forgets the pages of `[start, end)`. The tests share it,
    /// as the runtime's threads do: pages another test wrote or noted
    /// written (`wrote`, `wrote_own`) stay in it after that memory is
    /// freed, and a fresh allocation given the same address was found
    /// written before anything wrote it (in the parallel suite only).
    fn forget_written(start: usize, end: usize) {
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        for page in first..last {
            j.written.remove(page);
        }
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        forget_written(start, end);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// The submission's one asking stands: a further asking ahead makes no
    /// plan and consumes no range (ScarlettPort 23870346).
    #[test]
    fn a_kept_asking_consumes_no_other_range_plan() {
        let _one = one_at_a_time();
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        forget_written(p, p + 4 * PAGE);
        ask_once(&[(p, p + 2 * PAGE), (p + 2 * PAGE, p + 4 * PAGE)], |_| {});
        assert!(JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).kept);
        ask_ahead(std::iter::from_fn(|| panic!("a kept asking must not consume another range plan")));
        ask_once_end();
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A guard made between a view's mapping and the runtime's word about it
    /// is forgotten by that word; the page must get its write access back in
    /// the same step, or the title's next plain write to its own view faults
    /// with no guard to open it (fxceless/victory-fork#1447).
    #[test]
    fn a_guard_forgotten_by_the_remap_gives_its_page_back() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        const SIZE: usize = BLOCK;
        let (section, view, other) = two_views(SIZE);
        assert_eq!(guard(view, view + SIZE), Some(vec![view / PAGE]));
        // The runtime mapped the pages and only now tells the guard.
        title_memory_remapped(view, SIZE);
        crate::inplace::title_memory_view(view, SIZE, section, 0);
        // SAFETY: plain data, filled in by the query about the mapped view.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(view as *const _, &mut info, size_of_val(&info)) };
        assert_eq!(n, size_of_val(&info));
        assert_eq!(info.Protect, PAGE_READWRITE);
        drop_views(section, &[view, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// A small random number source for the tests below (xorshift).
    struct Dice(u64);

    impl Dice {
        /// A number below `n`.
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    /// The summary of clean blocks says of every block what the set says,
    /// through blocks noted clean one at a time and in runs and let go of,
    /// across words and at their edges: whether a range is all clean, each
    /// block, and each word.
    #[test]
    fn the_summary_says_what_the_set_says() {
        let mut dice = Dice(0x2545_F491_4F6C_DD1D);
        let mut clean = Clean::new();
        // Five words and a bit, not starting at a word's edge.
        let base = 0x7_0000 / BLOCK * WORD + 37;
        let size = 5 * WORD + 11;
        for round in 0..20_000 {
            let at = base + dice.below(size);
            match dice.below(8) {
                0..=2 => clean.insert(at),
                3 | 4 => clean.remove(at),
                5 => clean.extend(at, (at + 1 + dice.below(3 * WORD)).min(base + size)),
                6 => {
                    // A range let go of, as a remapping does.
                    for block in at..(at + dice.below(2 * WORD)).min(base + size) {
                        clean.remove(block);
                    }
                }
                _ => {
                    // Everything noted clean, as a first guarding does.
                    if round % 64 == 0 {
                        clean.extend(base, base + size);
                    }
                }
            }
            for _ in 0..4 {
                let first = base - 2 + dice.below(size + 4);
                let last = first + dice.below(base + size + 2 - first.min(base + size));
                let in_set = clean.set.range(first..last).count();
                assert_eq!(clean.all(first, last), in_set == last - first, "{first}..{last} after {round} steps");
                assert!(clean.agrees(first, last), "{first}..{last} after {round} steps");
            }
            if round % 997 == 0 {
                for block in base - WORD..base + size + WORD {
                    assert_eq!(clean.has(block), clean.contains(block), "block {block}");
                }
                assert!(clean.bits.values().all(|&bits| bits != 0), "a word with no clean block is left out");
                assert!(clean.agrees(base - WORD, base + size + WORD));
            }
        }
        // A word whose bits say otherwise is found.
        clean.extend(base, base + size);
        assert!(clean.all(base, base + size));
        clean.set.remove(&(base + WORD));
        assert!(!clean.agrees(base, base + size));
        assert!(clean.agrees(base, base + WORD));
    }

    /// The same steps at two views of their own, one asked about as before
    /// and one with the summary: the same answers each time, and the same
    /// blocks guarded, clean and absent at the end. Pages are committed
    /// (some blocks are never committed whole, some not at all), written
    /// through the handler and by the runtime, and remapped, at random over
    /// ranges that cross the summary's words.
    #[test]
    fn the_summary_answers_as_the_set_does() {
        the_summary_answers_as_the_set(None, AbsentKnown::Off);
    }

    /// The same with the runs of uncommitted blocks kept for the frame
    /// (`VICTORY_GUARD_ABSENT_KNOWN=1`), and kept until remapped with hot
    /// blocks (`=forever`): the pages the test commits with nothing said are
    /// seen alike at both views.
    #[test]
    fn the_summary_answers_as_the_set_does_with_absent_runs_known() {
        let _alone = one_at_a_time();
        the_summary_answers_as_the_set(None, AbsentKnown::Frame);
        the_summary_answers_as_the_set(Some(2), AbsentKnown::Forever);
    }

    /// The same with hot blocks (`VICTORY_GUARD_HOT=2`), the frames going
    /// by: blocks written in two frames in a row (and the blocks that will
    /// not be made read-only, committed in part) go hot, the same at both
    /// views, and are tested again after `HOT_RETEST` frames.
    #[test]
    fn the_summary_answers_as_the_set_does_with_hot_blocks() {
        let _alone = one_at_a_time();
        the_summary_answers_as_the_set(Some(2), AbsentKnown::Off);
    }

    /// The tests with hot blocks or runs of uncommitted blocks kept, one at
    /// a time. Each test numbers its own frames ([`TEST_FRAME`]), and the
    /// first asking of a frame lets go of the streaks of blocks not found
    /// written in the frame before, and of the runs asked about in the frame
    /// before: a test at frame 5,000 would let go of those of a test at
    /// frame 11 (the runtime has one count of frames).
    fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
        static ONE: Mutex<()> = Mutex::new(());
        ONE.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn the_summary_answers_as_the_set(hot: Option<u64>, absent: AbsentKnown) {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            MapViewOfFileEx, SEC_RESERVE, UnmapViewOfFile, VirtualProtect,
        };
        const BLOCKS: usize = 150;
        const SIZE: usize = BLOCKS * BLOCK;
        const PAGES: usize = BLOCK / KERNEL_PAGE;
        let view = |n: usize| {
            // SAFETY: a new reserved pagefile section, mapped; the view
            // holds its own reference once the handle is closed. High up
            // if it can be, away from the other tests' views: one of them
            // unmapped before its guards are told would leave ours read-only
            // and forgotten.
            unsafe {
                let section = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE | SEC_RESERVE,
                    0,
                    SIZE as u32,
                    std::ptr::null(),
                );
                assert!(!section.is_null());
                let apart = usize::from(hot.is_some() || absent != AbsentKnown::Off) * 0x4000_0000;
                let high = (0x2B00_0000_0000 + n * 0x1000_0000 + apart) as *const _;
                let mut view = MapViewOfFileEx(section, FILE_MAP_WRITE, 0, 0, SIZE, high);
                if view.Value.is_null() {
                    view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
                }
                CloseHandle(section);
                assert!(!view.Value.is_null());
                view.Value as usize
            }
        };
        // As before, and with the summary.
        let views = [view(0), view(1)];
        let mut dice = Dice(0x9E37_79B9_7F4A_7C15);
        // The kernel pages committed, numbered from the view's start.
        let mut committed = BTreeSet::new();
        let commit = |committed: &mut BTreeSet<usize>, page: usize| {
            if committed.insert(page) {
                for view in views {
                    let at = view + page * KERNEL_PAGE;
                    // SAFETY: a page of our reserved view.
                    let done = unsafe { VirtualAlloc(at as _, KERNEL_PAGE, MEM_COMMIT, PAGE_READWRITE) };
                    assert_eq!(done as usize, at);
                }
            }
        };
        for block in 0..BLOCKS {
            match dice.below(10) {
                0 => {}
                1 => commit(&mut committed, block * PAGES + dice.below(PAGES)),
                _ => (0..PAGES).for_each(|page| commit(&mut committed, block * PAGES + page)),
            }
        }
        let asked = |views: [usize; 2], from: usize, to: usize| {
            views.map(|view| {
                TEST_SUMMARY.set(Some(view == views[1]));
                let pages = guard(view + from, view + to);
                pages.map(|pages| pages.into_iter().map(|page| page - view / PAGE).collect::<Vec<_>>())
            })
        };
        TEST_HOT.set(hot);
        TEST_ABSENT_KNOWN.set(Some(absent));
        let mut hot_seen = 0;
        for round in 0..4000 {
            if (hot.is_some() || absent != AbsentKnown::Off) && round % 8 == 0 {
                TEST_FRAME.set(Some(100 + round as u64 / 8));
                hot_seen += guards().hot.range(views[0] / BLOCK..views[0] / BLOCK + BLOCKS).count();
            }
            match dice.below(10) {
                0..=4 => {
                    let (from, to) = if dice.below(8) == 0 {
                        (0, SIZE)
                    } else {
                        let from = dice.below(SIZE);
                        (from, from + 1 + dice.below(SIZE - from))
                    };
                    let [before, summed] = asked(views, from, to);
                    assert!(before.is_some());
                    assert_eq!(before, summed, "{from:#x}..{to:#x} after {round} steps");
                }
                5 | 6 => {
                    // The title writes a committed page: through the
                    // handler where its block is guarded.
                    if let Some(&page) = committed.iter().nth(dice.below(committed.len().max(1))) {
                        for view in views {
                            // SAFETY: a committed page of our view.
                            unsafe { ((view + page * KERNEL_PAGE + 8) as *mut u8).write_volatile(round as u8) };
                        }
                    }
                }
                7 => commit(&mut committed, dice.below(BLOCKS * PAGES)),
                8 => {
                    // Inside the views: no other test's block is opened.
                    let at = dice.below(SIZE);
                    let len = (1 + dice.below(4 * BLOCK)).min(SIZE - at);
                    for view in views {
                        writing(view + at, len);
                    }
                }
                _ => {
                    // The runtime maps something new there: read-write.
                    let (first, n) = (dice.below(BLOCKS), 1 + dice.below(3 * WORD));
                    let last = (first + n).min(BLOCKS);
                    for view in views {
                        title_memory_remapped(view + first * BLOCK, (last - first) * BLOCK);
                        for &page in committed.range(first * PAGES..last * PAGES) {
                            let mut old = 0;
                            // SAFETY: a committed page of our view.
                            let done = unsafe {
                                VirtualProtect((view + page * KERNEL_PAGE) as _, KERNEL_PAGE, PAGE_READWRITE, &mut old)
                            };
                            assert_ne!(done, 0);
                        }
                    }
                }
            }
        }
        let [before, summed] = asked(views, 0, SIZE);
        assert_eq!(before, summed);
        {
            let g = guards();
            let held = |view: usize, block: usize| {
                let block = view / BLOCK + block;
                assert_eq!(g.clean.has(block), g.clean.contains(block));
                (
                    g.protected.contains(&block),
                    g.clean.contains(block),
                    g.absent.contains(&block),
                    g.hot.contains_key(&block),
                )
            };
            for block in 0..BLOCKS {
                assert_eq!(held(views[0], block), held(views[1], block), "block {block}");
            }
            for view in views {
                assert!(g.clean.agrees(view / BLOCK, view / BLOCK + BLOCKS));
            }
        }
        if hot.is_some() {
            assert!(hot_seen > 0, "no block went hot");
        }
        TEST_SUMMARY.set(None);
        TEST_HOT.set(None);
        TEST_HOT_RATIO.set(None);
        TEST_ABSENT_KNOWN.set(None);
        TEST_FRAME.set(None);
        for view in views {
            title_memory_remapped(view, SIZE);
            // SAFETY: our view.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        }
    }

    /// The tests of view memory above, asked with the summary.
    #[test]
    fn the_view_tests_with_the_summary() {
        TEST_SUMMARY.set(Some(true));
        kernel_reads_into_guarded_blocks_after_opening();
        writes_in_partly_committed_view_blocks_are_reported();
        a_range_that_begins_in_an_unmapped_block_is_guarded();
        unmapped_blocks_are_reported_once_until_mapped();
        a_write_through_another_view_is_reported_at_the_guarded_one();
        a_view_mapped_beside_a_guarded_one_makes_it_check_again();
        a_write_told_at_one_view_is_reported_at_the_others();
        a_write_is_reported_at_every_checked_view();
        a_page_of_many_views_is_guarded_one_view_at_a_time();
        TEST_SUMMARY.set(None);
    }

    /// The other views of a run's pages found under one taking of the
    /// lock (`VICTORY_GUARD_VIEWS_BATCH`) are those found by the run's views
    /// and then each of their pages: over runs at random across views of
    /// one to four blocks laid side by side and apart, each page shown once
    /// or in two to five views, or in more than `MANY_VIEWS`.
    #[test]
    fn the_other_views_found_at_once_are_those_found_page_by_page() {
        let mut dice = Dice(0x05EE_D0FA_11A5);
        let base = 0x6400_0000_0000usize;
        let section = 0xA11A5;
        // Views of 1-4 blocks, now side by side, now apart; the section's
        // pages are shown by several of them at random.
        let mut at = base;
        let mut laid = Vec::new();
        for _ in 0..200 {
            let blocks = 1 + dice.below(4);
            let offset = dice.below(40) * BLOCK;
            crate::inplace::title_memory_view(at, blocks * BLOCK, section, offset as u64);
            laid.push((at, blocks));
            at += (blocks + dice.below(2) * dice.below(3)) * BLOCK;
        }
        // One page of the section in more views than are guarded together.
        for n in 0..=MANY_VIEWS {
            crate::inplace::title_memory_view(at + n * BLOCK, BLOCK, section, (60 * BLOCK) as u64);
        }
        let end = at + (MANY_VIEWS + 1) * BLOCK;
        let page_by_page = |start: usize, stop: usize| {
            let mut out = Vec::new();
            for (view, bytes) in crate::inplace::views_between(start, stop) {
                for block in (view / BLOCK).max(start / BLOCK)..view.saturating_add(bytes).div_ceil(BLOCK).min(stop / BLOCK) {
                    let shown = crate::inplace::pages_of(block * BLOCK);
                    if (2..=MANY_VIEWS).contains(&shown.len()) {
                        out.push((block * BLOCK, shown));
                    }
                }
            }
            out
        };
        let mut shared_seen = 0;
        for _ in 0..2000 {
            let first = base / BLOCK + dice.below((end - base) / BLOCK);
            let last = first + 1 + dice.below(12);
            let (start, stop) = (first * BLOCK, last * BLOCK);
            let once = crate::inplace::shared_between(start, stop, MANY_VIEWS);
            assert_eq!(once, page_by_page(start, stop), "{start:#x}..{stop:#x}");
            shared_seen += once.len();
        }
        assert!(shared_seen > 0);
        assert!(laid.iter().any(|&(at, _)| crate::inplace::pages_of(at).len() >= 2));
        title_memory_remapped(base, end - base);
    }

    /// The view tests with the other views found at once and guarded in
    /// runs (`VICTORY_GUARD_VIEWS_BATCH`).
    #[test]
    fn the_view_tests_with_the_views_in_runs() {
        let _alone = one_at_a_time();
        TEST_VIEWS_BATCH.set(Some(true));
        a_write_through_another_view_is_reported_at_the_guarded_one();
        a_view_mapped_beside_a_guarded_one_makes_it_check_again();
        a_write_told_at_one_view_is_reported_at_the_others();
        a_write_is_reported_at_every_checked_view();
        a_page_of_many_views_is_guarded_one_view_at_a_time();
        the_other_views_of_a_hot_page_are_hot();
        for summed in [false, true] {
            for ratio in [0, 2] {
                no_write_is_missed(summed, ratio);
            }
        }
        TEST_VIEWS_BATCH.set(None);
    }

    /// With `VICTORY_GUARD_ABSENT_KNOWN=1`, the holes of a view (blocks
    /// noted absent, walked at every asking) are asked about once a frame,
    /// each run of them in one question; with `=forever` not again; a
    /// mapping of the runtime's lets go of the run it touches.
    #[test]
    fn uncommitted_runs_are_asked_about_once_a_frame() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER,
            MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _alone = one_at_a_time();
        // SAFETY: a fresh placeholder of four blocks and a pagefile section
        // of four, as in `unmapped_blocks_are_reported_once_until_mapped`.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (4 * BLOCK) as u32, std::ptr::null());
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree((p + block * BLOCK) as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            title_memory_remapped(p + block * BLOCK, BLOCK);
            view
        };
        TEST_ABSENT_KNOWN.set(Some(AbsentKnown::Frame));
        TEST_FRAME.set(Some(7));
        // Holes: block 0, and blocks 2 and 3.
        let middle = map(1);
        let page = |block: usize| (p + block * BLOCK) / PAGE;
        let ask = || guard(p, p + 4 * BLOCK).expect("view memory");
        let questions = || TEST_QUESTIONS.replace(0);
        let none: Vec<usize> = Vec::new();
        assert_eq!(ask(), (0..4).map(page).collect::<Vec<_>>());
        questions();
        assert_eq!(ask(), none);
        assert_eq!(questions(), 2, "each hole once");
        assert_eq!(ask(), none);
        assert_eq!(questions(), 0, "answered by the frame's runs");
        TEST_FRAME.set(Some(8));
        assert_eq!(ask(), none);
        assert_eq!(questions(), 2, "asked again in the next frame");
        TEST_ABSENT_KNOWN.set(Some(AbsentKnown::Forever));
        for frame in 9..12 {
            TEST_FRAME.set(Some(frame));
            assert_eq!(ask(), none);
            assert_eq!(questions(), 0, "kept until remapped");
        }
        // The runtime maps the first hole: reported, and its run let go of;
        // the other hole is still answered by its run.
        let first = map(0);
        assert!(guards().uncommitted.range(p / BLOCK..p / BLOCK + 2).next().is_none());
        assert_eq!(ask(), [page(0)]);
        assert_eq!(questions(), 0);
        assert_eq!(ask(), none);
        TEST_ABSENT_KNOWN.set(None);
        TEST_FRAME.set(None);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..4 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 4 * BLOCK);
        assert!(guards().uncommitted.range(p / BLOCK..p / BLOCK + 4).next().is_none());
    }

    /// A hole committed with nothing said (as only a caller other than the
    /// runtime's mappings can): with `VICTORY_GUARD_ABSENT_KNOWN=1` it is
    /// seen, and reported written, at the next frame's first asking that
    /// walks it, not at the askings left in its frame; with `=forever` only
    /// once the runtime remaps it.
    #[test]
    fn a_hole_committed_with_nothing_said_is_seen_the_next_frame() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, SEC_RESERVE,
            UnmapViewOfFile,
        };
        let _alone = one_at_a_time();
        const SIZE: usize = 4 * BLOCK;
        // SAFETY: a new reserved pagefile section, mapped; the view holds
        // its own reference once the handle is closed.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            view.Value as usize
        };
        let commit = |block: usize| {
            // SAFETY: a block of our reserved view.
            let done = unsafe { VirtualAlloc((view + block * BLOCK) as _, BLOCK, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(done as usize, view + block * BLOCK);
            // SAFETY: committed just now, read-write.
            unsafe { ((view + block * BLOCK + 8) as *mut u8).write_volatile(1) };
        };
        let page = |block: usize| (view + block * BLOCK) / PAGE;
        let ask = || guard(view, view + SIZE).expect("view memory");
        let none: Vec<usize> = Vec::new();
        commit(0);
        TEST_ABSENT_KNOWN.set(Some(AbsentKnown::Frame));
        TEST_FRAME.set(Some(20));
        // Blocks 1-3 noted absent.
        assert_eq!(ask(), (0..4).map(page).collect::<Vec<_>>());
        assert_eq!(ask(), none);
        commit(2);
        assert_eq!(ask(), none, "the frame's run says uncommitted");
        TEST_FRAME.set(Some(21));
        assert_eq!(ask(), [page(2)], "seen at the next frame");
        assert_eq!(ask(), none);
        TEST_ABSENT_KNOWN.set(Some(AbsentKnown::Forever));
        TEST_FRAME.set(Some(22));
        assert_eq!(ask(), none);
        commit(3);
        for frame in 23..26 {
            TEST_FRAME.set(Some(frame));
            assert_eq!(ask(), none, "not seen with the runs kept");
        }
        title_memory_remapped(view + 3 * BLOCK, BLOCK);
        assert_eq!(ask(), [page(3)], "seen once remapped");
        TEST_ABSENT_KNOWN.set(None);
        TEST_FRAME.set(None);
        title_memory_remapped(view, SIZE);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// The view tests whose holes are mapped by the runtime, with the runs
    /// of uncommitted blocks kept (`VICTORY_GUARD_ABSENT_KNOWN=1`, every
    /// asking in one frame): the runtime's mappings let go of them.
    #[test]
    fn the_view_tests_with_absent_runs_known() {
        let _alone = one_at_a_time();
        TEST_ABSENT_KNOWN.set(Some(AbsentKnown::Frame));
        TEST_FRAME.set(Some(1));
        kernel_reads_into_guarded_blocks_after_opening();
        a_range_that_begins_in_an_unmapped_block_is_guarded();
        unmapped_blocks_are_reported_once_until_mapped();
        a_write_through_another_view_is_reported_at_the_guarded_one();
        a_write_is_reported_at_every_checked_view();
        TEST_ABSENT_KNOWN.set(None);
        TEST_FRAME.set(None);
    }

    /// A pagefile section of `bytes`, mapped once (no title view); its
    /// address.
    fn one_view(bytes: usize) -> usize {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: a fresh pagefile section, mapped; the view holds its own
        // reference once the handle is closed.
        unsafe {
            let section =
                CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, bytes as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            view.Value as usize
        }
    }

    /// With `VICTORY_GUARD_HOT=3`: a block the title writes in three frames
    /// in a row is left writable (its writes fault no more) and reported at
    /// every asking, the blocks beside it guarded as before. After
    /// `HOT_RETEST` frames it is guarded again: written in the frame after,
    /// it is hot again at once; not written, it has gone cold. A remapping
    /// forgets it.
    #[test]
    fn a_block_written_every_frame_goes_hot() {
        let _alone = one_at_a_time();
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        const SIZE: usize = 4 * BLOCK;
        let view = one_view(SIZE);
        TEST_HOT.set(Some(3));
        TEST_HOT_RATIO.set(Some(0));
        let at_frame = |frame: u64| TEST_FRAME.set(Some(frame));
        let page = |block: usize| (view + block * BLOCK) / PAGE;
        let ask = || guard(view, view + SIZE).expect("view memory");
        let write = |block: usize| {
            // SAFETY: a byte of our view, through the handler if guarded.
            unsafe { ((view + block * BLOCK + 8) as *mut u8).write_volatile(1) };
        };
        let none: Vec<usize> = Vec::new();
        // The first asking reports every block, as found written: some
        // frames before the ones counted.
        at_frame(990);
        assert_eq!(ask(), (0..4).map(page).collect::<Vec<_>>());
        for frame in 1001..=1002 {
            at_frame(frame);
            assert!(!writable(view + BLOCK), "guarded");
            write(1);
            assert_eq!(ask(), [page(1)]);
            assert_eq!(ask(), none, "guarded again");
        }
        at_frame(1003);
        write(1);
        assert_eq!(ask(), [page(1)]);
        assert!(writable(view + BLOCK), "hot: left writable");
        assert!(guards().known(view / BLOCK + 1, false), "still known as view memory");
        assert_eq!(ask(), [page(1)], "reported at every asking");
        write(1);
        write(2);
        assert_eq!(ask(), [page(1), page(2)]);
        assert_eq!(ask(), [page(1)], "the block beside it guarded again as before");
        at_frame(1003 + HOT_RETEST - 1);
        assert_eq!(ask(), [page(1)]);
        assert!(writable(view + BLOCK));
        // Tested again: reported once more and guarded.
        at_frame(1003 + HOT_RETEST);
        assert_eq!(ask(), [page(1)]);
        assert!(!writable(view + BLOCK));
        assert_eq!(ask(), none);
        // Written in the frame after: hot again.
        at_frame(1003 + HOT_RETEST + 1);
        write(1);
        assert_eq!(ask(), [page(1)]);
        assert!(writable(view + BLOCK), "hot again");
        // Tested again and not written in the frame after: cold.
        at_frame(1003 + 2 * HOT_RETEST + 1);
        assert_eq!(ask(), [page(1)]);
        at_frame(1003 + 2 * HOT_RETEST + 2);
        assert_eq!(ask(), none);
        at_frame(1003 + 2 * HOT_RETEST + 3);
        write(1);
        assert_eq!(ask(), [page(1)]);
        assert!(!writable(view + BLOCK), "cold: guarded again as any block");
        assert_eq!(ask(), none);
        // Hot once more, then remapped: forgotten.
        for frame in 1003 + 2 * HOT_RETEST + 4..1003 + 2 * HOT_RETEST + 6 {
            at_frame(frame);
            write(1);
            assert_eq!(ask(), [page(1)]);
        }
        assert!(guards().hot.contains_key(&(view / BLOCK + 1)));
        title_memory_remapped(view, SIZE);
        {
            let g = guards();
            assert!(g.hot.range(view / BLOCK..view / BLOCK + 4).next().is_none());
            assert!(g.streak.range(view / BLOCK..view / BLOCK + 4).next().is_none());
        }
        TEST_HOT.set(None);
        TEST_HOT_RATIO.set(None);
        TEST_FRAME.set(None);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// With `VICTORY_GUARD_HOT=2` and `VICTORY_GUARD_HOT_RATIO=2`: a block
    /// found written once a frame goes hot reported at most twice a frame;
    /// asked about a third time, it is guarded again and not made hot for
    /// `HOT_RETEST` frames. One found written three times a frame stays hot
    /// through six askings a frame.
    #[test]
    fn a_hot_block_asked_about_more_often_than_written_is_guarded_again() {
        let _alone = one_at_a_time();
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        const SIZE: usize = 2 * BLOCK;
        let view = one_view(SIZE);
        TEST_HOT.set(Some(2));
        TEST_HOT_RATIO.set(Some(2));
        let at_frame = |frame: u64| TEST_FRAME.set(Some(frame));
        let page = |block: usize| (view + block * BLOCK) / PAGE;
        let ask = || guard(view, view + SIZE).expect("view memory");
        let write = |block: usize| {
            // SAFETY: a byte of our view, through the handler if guarded.
            unsafe { ((view + block * BLOCK + 8) as *mut u8).write_volatile(1) };
        };
        let none: Vec<usize> = Vec::new();
        at_frame(90);
        assert_eq!(ask(), [page(0), page(1)]);
        at_frame(101);
        write(1);
        assert_eq!(ask(), [page(1)]);
        at_frame(102);
        write(1);
        assert_eq!(ask(), [page(1)], "made hot");
        assert!(writable(view + BLOCK));
        assert_eq!(ask(), [page(1)], "reported a second time");
        assert_eq!(ask(), [page(1)], "a third: reported, and guarded again");
        assert!(!writable(view + BLOCK));
        assert_eq!(ask(), none);
        // Written every frame, it is not made hot again for a while.
        for frame in 103..105 {
            at_frame(frame);
            write(1);
            assert_eq!(ask(), [page(1)]);
            assert!(!writable(view + BLOCK), "kept cold");
        }
        at_frame(102 + HOT_RETEST - 1);
        write(1);
        assert_eq!(ask(), [page(1)]);
        at_frame(102 + HOT_RETEST);
        write(1);
        assert_eq!(ask(), [page(1)]);
        assert!(writable(view + BLOCK), "hot again");
        title_memory_remapped(view, SIZE);
        // Found written three times a frame: reported up to six.
        at_frame(500);
        assert_eq!(ask(), [page(0), page(1)]);
        at_frame(511);
        for _ in 0..3 {
            write(0);
            assert_eq!(ask(), [page(0)]);
        }
        at_frame(512);
        write(0);
        assert_eq!(ask(), [page(0)], "made hot");
        for n in 2..=6 {
            assert_eq!(ask(), [page(0)], "asking {n}");
            assert!(writable(view), "hot through asking {n}");
        }
        assert_eq!(ask(), [page(0)]);
        assert!(!writable(view), "guarded again at the seventh");
        TEST_HOT.set(None);
        TEST_HOT_RATIO.set(None);
        TEST_FRAME.set(None);
        title_memory_remapped(view, SIZE);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// With `VICTORY_GUARD_HOT=2`: the page written through one view in two
    /// frames in a row goes hot at both of its views, whichever is asked
    /// first: a write through either is reported at both, at every asking,
    /// with no fault. A third view of the page mapped ends it: the views
    /// are guarded together again.
    #[test]
    fn the_other_views_of_a_hot_page_are_hot_with_it() {
        let _alone = one_at_a_time();
        the_other_views_of_a_hot_page_are_hot();
    }

    fn the_other_views_of_a_hot_page_are_hot() {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        TEST_HOT.set(Some(2));
        TEST_HOT_RATIO.set(Some(0));
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        TEST_FRAME.set(Some(5));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]));
        // Asked first: the first view, then the second.
        for (frame, first) in [(11, [one, two]), (12, [two, one])] {
            TEST_FRAME.set(Some(frame));
            assert!(!writable(one) && !writable(two));
            // SAFETY: the first view, opened by the handler.
            unsafe { ((one + 8) as *mut u8).write_volatile(frame as u8) };
            for view in first {
                assert_eq!(guard(view, view + SIZE), Some(vec![pages(view, 0)]), "frame {frame}");
            }
        }
        assert!(writable(one) && writable(two), "both views of the page hot");
        assert!(!writable(one + BLOCK) && !writable(two + BLOCK), "the next page guarded as before");
        TEST_FRAME.set(Some(13));
        // SAFETY: the second view, writable.
        unsafe { ((two + 8) as *mut u8).write_volatile(1) };
        for _ in 0..2 {
            assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]));
            assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        }
        // A third view: every view of the section's pages is checked again,
        // and guarded together.
        // SAFETY: the section of `two_views`, still open.
        let third = unsafe { MapViewOfFile(section as *mut std::ffi::c_void, FILE_MAP_WRITE, 0, 0, SIZE) }.Value as usize;
        assert_ne!(third, 0);
        crate::inplace::title_memory_view(third, SIZE, section, 0);
        assert!(guards().hot.range(one / BLOCK..one / BLOCK + 2).next().is_none());
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert!(!writable(one) && !writable(two) && !writable(third), "guarded together");
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]));
        assert_eq!(guard(third, third + SIZE), Some(vec![pages(third, 0), pages(third, 1)]));
        TEST_HOT.set(None);
        TEST_HOT_RATIO.set(None);
        TEST_FRAME.set(None);
        drop_views(section, &[one, two, third], SIZE);
    }

    /// Whatever the order of writes through either view of the pages (by
    /// the title, faulting where guarded, and by the runtime, told first),
    /// of askings at either view and of frames, with blocks going hot and
    /// cold (`VICTORY_GUARD_HOT=2`, tested again after `HOT_RETEST` frames)
    /// and views remapped: every block written since an asking at its view
    /// last covered it is reported at the next asking there, and the hot
    /// blocks are as they must be. As before, and with the summary.
    #[test]
    fn no_write_is_missed_with_hot_blocks() {
        let _alone = one_at_a_time();
        for summed in [false, true] {
            for ratio in [0, 2] {
                no_write_is_missed(summed, ratio);
            }
        }
    }

    fn no_write_is_missed(summed: bool, ratio: u64) {
        use windows_sys::Win32::System::Memory::VirtualProtect;
        const BLOCKS: usize = 8;
        const SIZE: usize = BLOCKS * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let views = [one, two];
        TEST_SUMMARY.set(Some(summed));
        TEST_HOT.set(Some(2));
        TEST_HOT_RATIO.set(Some(ratio));
        let mut frame = 5000;
        TEST_FRAME.set(Some(frame));
        let mut dice = Dice(0xC0FF_EE00_1234_5678 ^ u64::from(summed) ^ (ratio << 8));
        // Blocks written since an asking at that view last covered them.
        let mut dirty = [[true; BLOCKS]; 2];
        let (mut hot_seen, mut cold_seen) = (0, 0);
        for round in 0..6000usize {
            match dice.below(16) {
                0..=5 => {
                    let (v, b) = (dice.below(2), dice.below(BLOCKS));
                    // SAFETY: a byte of our view, through the handler if guarded.
                    unsafe { ((views[v] + b * BLOCK + 8 + round % 64) as *mut u8).write_volatile(round as u8) };
                    dirty[0][b] = true;
                    dirty[1][b] = true;
                }
                6..=11 => {
                    let v = dice.below(2);
                    let from = dice.below(BLOCKS);
                    let to = from + 1 + dice.below(BLOCKS - from);
                    let pages = guard(views[v] + from * BLOCK, views[v] + to * BLOCK).expect("view memory");
                    for b in from..to {
                        assert!(
                            !dirty[v][b] || pages.contains(&((views[v] + b * BLOCK) / PAGE)),
                            "view {v} block {b} written and not reported, round {round}, summary {summed}, ratio {ratio}"
                        );
                        dirty[v][b] = false;
                    }
                }
                12 => {
                    // The runtime writes, telling the guards first.
                    let (v, b) = (dice.below(2), dice.below(BLOCKS));
                    let n = (1 + dice.below(2)).min(BLOCKS - b);
                    writing(views[v] + b * BLOCK, n * BLOCK);
                    for k in b..b + n {
                        // SAFETY: a byte of our view, opened by `writing`.
                        unsafe { ((views[v] + k * BLOCK + 16) as *mut u8).write_volatile(round as u8) };
                        dirty[0][k] = true;
                        dirty[1][k] = true;
                    }
                }
                13 | 14 => {
                    frame += 1;
                    TEST_FRAME.set(Some(frame));
                }
                _ => {
                    // Now and then the runtime maps a view anew: forgotten,
                    // writable, its other views checked again.
                    if dice.below(8) == 0 {
                        let v = dice.below(2);
                        title_memory_remapped(views[v], SIZE);
                        let mut old = 0;
                        // SAFETY: our view; only its protection changes.
                        assert_ne!(unsafe { VirtualProtect(views[v] as *const _, SIZE, PAGE_READWRITE, &mut old) }, 0);
                        crate::inplace::title_memory_view(views[v], SIZE, section, 0);
                    }
                }
            }
            let g = guards();
            for b in 0..BLOCKS {
                let [x, y] = views.map(|view| view / BLOCK + b);
                for (block, other) in [(x, y), (y, x)] {
                    if !g.hot.contains_key(&block) {
                        cold_seen += 1;
                        continue;
                    }
                    hot_seen += 1;
                    assert!(g.hot.contains_key(&other), "block {b}: the other view of a hot page is not hot, round {round}");
                    assert!(
                        !g.protected.contains(&block)
                            && !g.clean.contains(block)
                            && !g.clean.has(block)
                            && !g.absent.contains(&block)
                            && !g.aliased.contains(&block),
                        "block {b}: a hot block in the guards' sets, round {round}"
                    );
                    assert!(writable(block * BLOCK), "block {b}: a hot block not writable, round {round}");
                }
            }
        }
        assert!(hot_seen > 0 && cold_seen > 0, "{hot_seen} hot and {cold_seen} cold");
        TEST_SUMMARY.set(None);
        TEST_HOT.set(None);
        TEST_HOT_RATIO.set(None);
        TEST_FRAME.set(None);
        drop_views(section, &views, SIZE);
    }

    /// For timing, about what a steady scene's frame asks: 3,600 ranges
    /// over some 200,000 blocks, 16 of them the size of the title's heap,
    /// in turn in `views` (first blocks) of `blocks` blocks each.
    fn a_frame(dice: &mut Dice, views: &[usize], blocks: usize) -> Vec<(usize, usize)> {
        (0..3600)
            .map(|n| {
                let len = if n < 16 { 2784.min(blocks) } else { 1 + dice.below(86) };
                let at = views[n % views.len()] + dice.below(blocks - len + 1);
                (at, at + len)
            })
            .collect()
    }

    /// Not a test: what finding a frame's ranges all clean costs, the set's
    /// way (before the summary) and the summary's, among 200,000 clean
    /// blocks. `cargo test --profile diag -p victory-gpu --lib -- --ignored
    /// --nocapture costs`.
    #[test]
    #[ignore]
    fn finding_ranges_all_clean_costs() {
        let mut dice = Dice(0x1234_5678_9ABC_DEF1);
        let mut clean = Clean::new();
        // 64 views of 200 MB, 1 GiB apart and not at the words' edges.
        let views: Vec<usize> = (0..64).map(|n| (0x10_0000_0000 / BLOCK) + n * (1 << 14) + n * 37).collect();
        for &view in &views {
            clean.extend(view, view + 3125);
        }
        let ranges = a_frame(&mut dice, &views, 3125);
        for &(first, last) in ranges.iter().step_by(100) {
            clean.remove(first + (last - first) / 2);
        }
        let blocks: usize = ranges.iter().map(|r| r.1 - r.0).sum();
        let set_way = |clean: &Clean| {
            ranges
                .iter()
                .filter(|&&(first, last)| {
                    let found: Vec<usize> = clean.set.range(first..last).copied().collect();
                    std::hint::black_box(found).len() == last - first
                })
                .count()
        };
        let summary_way =
            |clean: &Clean| ranges.iter().filter(|&&(first, last)| std::hint::black_box(clean).all(first, last)).count();
        assert_eq!(set_way(&clean), summary_way(&clean));
        // Warm, and with the caches emptied before each frame (the title
        // runs between the askings): the best of 20 frames each.
        let mut other = vec![0u8; 64 << 20];
        let mut best = [std::time::Duration::MAX; 4];
        for n in 0..40 {
            let cold = usize::from(n % 2 == 1);
            let mut empty = || {
                if cold == 1 {
                    other.iter_mut().for_each(|b| *b = b.wrapping_add(1));
                    std::hint::black_box(&other);
                }
            };
            empty();
            let started = std::time::Instant::now();
            std::hint::black_box(set_way(&clean));
            best[cold * 2] = best[cold * 2].min(started.elapsed());
            empty();
            let started = std::time::Instant::now();
            std::hint::black_box(summary_way(&clean));
            best[cold * 2 + 1] = best[cold * 2 + 1].min(started.elapsed());
        }
        let ms = best.map(|d| d.as_secs_f64() * 1e3);
        println!(
            "finding {} ranges ({blocks} blocks, {} all clean) all clean among {} clean blocks, the best of 20 frames: \
             warm, the set's way {:.3} ms, the summary's {:.3} ms; caches emptied first, the set's way {:.3} ms, \
             the summary's {:.3} ms",
            ranges.len(),
            summary_way(&clean),
            clean.set.len(),
            ms[0],
            ms[1],
            ms[2],
            ms[3]
        );
    }

    /// Not a test: what a frame's askings of [`guard`] cost, as before and
    /// with the summary, at a 256 MB view of our own guarded whole (the
    /// lock taken without waiting; a frame of the title holds some 200,000
    /// clean blocks, this one 4,096). `cargo test --profile diag -p
    /// victory-gpu --lib -- --ignored --nocapture costs`.
    #[test]
    #[ignore]
    fn guarding_a_frame_costs() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        const BLOCKS: usize = 4096;
        const SIZE: usize = BLOCKS * BLOCK;
        // SAFETY: a new pagefile section, mapped; never touched, so no
        // page of it is made.
        let view = unsafe {
            let section =
                CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, SIZE as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            view.Value as usize
        };
        assert_eq!(guard(view, view + SIZE).map(|pages| pages.len()), Some(BLOCKS));
        let mut dice = Dice(0x0BAD_5EED_F00D_CAFE);
        let ranges = a_frame(&mut dice, &[view / BLOCK], BLOCKS);
        let blocks: usize = ranges.iter().map(|r| r.1 - r.0).sum();
        // A frame's askings; before them, `written` blocks written by the
        // title (one in a hundred: their faults are not timed, guarding
        // them again is). The pages reported.
        let frame = |summed: bool, written: usize, dice: &mut Dice| {
            for _ in 0..written {
                // SAFETY: a byte of our view, opened by the handler.
                unsafe { ((view + dice.below(BLOCKS) * BLOCK + 8) as *mut u8).write_volatile(1) };
            }
            TEST_SUMMARY.set(Some(summed));
            let started = std::time::Instant::now();
            let pages: usize = ranges
                .iter()
                .map(|&(first, last)| guard(first * BLOCK, last * BLOCK).map_or(0, |pages| pages.len()))
                .sum();
            (started.elapsed(), pages)
        };
        let mut best = [std::time::Duration::MAX; 4];
        let mut reported = [0; 2];
        for _ in 0..20 {
            for (n, summed) in [false, true].into_iter().enumerate() {
                let (took, pages) = frame(summed, 0, &mut dice);
                assert_eq!(pages, 0, "nothing written");
                best[n] = best[n].min(took);
                let (took, pages) = frame(summed, BLOCKS / 100, &mut dice);
                best[2 + n] = best[2 + n].min(took);
                reported[n] += pages;
            }
        }
        TEST_SUMMARY.set(None);
        let ms = best.map(|d| d.as_secs_f64() * 1e3);
        println!(
            "{} askings of guard ({blocks} blocks) at a view of {BLOCKS} blocks, the best of 20 frames: all clean, \
             as before {:.3} ms, with the summary {:.3} ms; with {} blocks written before each frame, as before \
             {:.3} ms, with the summary {:.3} ms ({} and {} pages reported in all, of blocks picked at random)",
            ranges.len(),
            ms[0],
            ms[1],
            BLOCKS / 100,
            ms[2],
            ms[3],
            reported[0],
            reported[1]
        );
        title_memory_remapped(view, SIZE);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// Not a test: what each step of [`guard`]'s walk costs on this
    /// machine, for reading the frame line's split of "walking": a block
    /// made read-only and writable again, alone and while other threads
    /// read the memory on other processors (as the title's threads do, so
    /// that each change waits for their processors to forget the page);
    /// one call for a run of blocks against one a block; a write fault
    /// taken by our handler; the question whether memory is uncommitted,
    /// at a placeholder and at a reserved view; the lookups of a page's
    /// other views among many views. `cargo test --profile diag -p
    /// victory-gpu --lib -- --ignored --nocapture costs`.
    #[test]
    #[ignore]
    fn each_step_of_a_walk_costs() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS,
            MapViewOfFile, PAGE_NOACCESS, PAGE_READONLY, SEC_RESERVE, UnmapViewOfFile, VirtualAlloc2,
            VirtualProtect,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCKS: usize = 64;
        const SIZE: usize = BLOCKS * BLOCK;
        const REPS: usize = 20;
        let mapped = |bytes: usize, flags: u32| {
            // SAFETY: a new pagefile section, mapped; the view holds its
            // own reference once the handle is closed.
            unsafe {
                let section = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE | flags,
                    (bytes >> 32) as u32,
                    bytes as u32,
                    std::ptr::null(),
                );
                assert!(!section.is_null());
                let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes);
                CloseHandle(section);
                assert!(!view.Value.is_null());
                view.Value as usize
            }
        };
        let unmap = |view: usize| {
            // SAFETY: a view of `mapped`.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        };
        let us = |d: Duration, n: usize| d.as_secs_f64() * 1e6 / n as f64;
        let view = mapped(SIZE, 0);
        for at in (view..view + SIZE).step_by(KERNEL_PAGE) {
            // SAFETY: our view, read-write.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        let set = |at: usize, blocks: usize, how: u32| {
            let mut old = 0;
            // SAFETY: our view; only its protection changes.
            assert_ne!(unsafe { VirtualProtect(at as *const _, blocks * BLOCK, how, &mut old) }, 0);
        };
        // Each block read-only then writable again, one call a block; and
        // runs of 16 blocks, one call a run.
        let changes = || {
            let [mut ro, mut rw, mut run_ro, mut run_rw] = [Duration::ZERO; 4];
            for _ in 0..REPS {
                for b in 0..BLOCKS {
                    let t = Instant::now();
                    set(view + b * BLOCK, 1, PAGE_READONLY);
                    ro += t.elapsed();
                    let t = Instant::now();
                    set(view + b * BLOCK, 1, PAGE_READWRITE);
                    rw += t.elapsed();
                }
                for b in (0..BLOCKS).step_by(16) {
                    let t = Instant::now();
                    set(view + b * BLOCK, 16, PAGE_READONLY);
                    run_ro += t.elapsed();
                    let t = Instant::now();
                    set(view + b * BLOCK, 16, PAGE_READWRITE);
                    run_rw += t.elapsed();
                }
            }
            let n = REPS * BLOCKS;
            [us(ro, n), us(rw, n), us(run_ro, n / 16), us(run_rw, n / 16)]
        };
        // A write fault on each block of a guarded view of its own (not a
        // title view: no other views), and guarding them again, in runs of
        // one block (every other block written) and in one run.
        let faults = || {
            let fv = view;
            assert!(guard(fv, fv + SIZE).is_some());
            let (mut fault, mut single, mut whole) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
            for _ in 0..REPS {
                let t = Instant::now();
                for b in (0..BLOCKS).step_by(2) {
                    // SAFETY: our view, opened by the handler.
                    unsafe { ((fv + b * BLOCK + 8) as *mut u8).write_volatile(2) };
                }
                fault += t.elapsed();
                let t = Instant::now();
                assert_eq!(guard(fv, fv + SIZE).map(|p| p.len()), Some(BLOCKS / 2));
                single += t.elapsed();
                for b in 0..BLOCKS {
                    // SAFETY: as above.
                    unsafe { ((fv + b * BLOCK + 8) as *mut u8).write_volatile(3) };
                }
                let t = Instant::now();
                assert_eq!(guard(fv, fv + SIZE).map(|p| p.len()), Some(BLOCKS));
                whole += t.elapsed();
            }
            title_memory_remapped(fv, SIZE);
            set(fv, BLOCKS, PAGE_READWRITE);
            [us(fault, REPS * BLOCKS / 2), us(single, REPS * BLOCKS / 2), us(whole, REPS * BLOCKS)]
        };
        let quiet = (changes(), faults());
        // Readers on other processors, as the title's threads.
        let stop = Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = (0..std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(2).clamp(1, 16))
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut sum = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        for at in (view..view + SIZE).step_by(KERNEL_PAGE) {
                            // SAFETY: our view, readable throughout.
                            sum = sum.wrapping_add(u64::from(unsafe { (at as *const u8).read_volatile() }));
                        }
                    }
                    sum
                })
            })
            .collect();
        let n_readers = readers.len();
        let busy = (changes(), faults());
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            std::hint::black_box(r.join().unwrap());
        }
        unmap(view);
        // The question whether memory is uncommitted: at a placeholder of
        // 1 GiB, at a reserved view of 256 MiB, and at a reserved view
        // whose next block is committed.
        // SAFETY: a fresh placeholder, released at the end.
        let placeholder = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                1 << 30,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize
        };
        assert_ne!(placeholder, 0);
        let reserved = mapped(256 << 20, SEC_RESERVE);
        let short = mapped(4 * BLOCK, SEC_RESERVE);
        // SAFETY: the second block of our reserved view.
        let done = unsafe { VirtualAlloc((short + BLOCK) as _, BLOCK, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(done as usize, short + BLOCK);
        let query = |at: usize| {
            let t = Instant::now();
            for _ in 0..REPS * 10 {
                std::hint::black_box(uncommitted_end(at / BLOCK));
            }
            us(t.elapsed(), REPS * 10)
        };
        let queries = [query(placeholder), query(reserved), query(short)];
        assert_eq!(uncommitted_end(short / BLOCK), short / BLOCK + 1);
        // SAFETY: our placeholder.
        unsafe { VirtualFree(placeholder as *mut _, 0, MEM_RELEASE) };
        unmap(reserved);
        unmap(short);
        // The lookups among 20,000 views of a block each, high up where
        // nothing is mapped, one page in ten shown twice.
        let base = 0x6100_0000_0000usize;
        const VIEWS: usize = 20_000;
        for n in 0..VIEWS {
            let offset = if n % 10 == 9 { (n - 1) * BLOCK } else { n * BLOCK };
            crate::inplace::title_memory_view(base + n * BLOCK, BLOCK, 0x5EC7, offset as u64);
        }
        let t = Instant::now();
        let mut shown = 0;
        for n in 0..VIEWS {
            shown += crate::inplace::pages_of(base + n * BLOCK).len();
        }
        let pages_of = us(t.elapsed(), VIEWS);
        assert_eq!(shown, VIEWS + VIEWS / 10 * 2);
        let t = Instant::now();
        let mut found = 0;
        for n in (0..VIEWS).step_by(16) {
            found += crate::inplace::views_between(base + n * BLOCK, base + (n + 16) * BLOCK).len();
        }
        let between = us(t.elapsed(), VIEWS / 16);
        assert!(found >= VIEWS);
        title_memory_remapped(base, VIEWS * BLOCK);
        println!(
            "a block made read-only {:.2} us, writable again {:.2} us; 16 blocks in one call {:.2} us and {:.2} us; \
             a write fault taken {:.2} us; a walk guarding blocks again in runs of one {:.2} us a block, in one run \
             {:.2} us a block",
            quiet.0[0], quiet.0[1], quiet.0[2], quiet.0[3], quiet.1[0], quiet.1[1], quiet.1[2]
        );
        println!(
            "the same with {n_readers} threads reading the memory: read-only {:.2} us, writable {:.2} us; 16 blocks \
             {:.2} us and {:.2} us; a fault {:.2} us; guarding again in runs of one {:.2} us a block, in one run {:.2} \
             us a block",
            busy.0[0], busy.0[1], busy.0[2], busy.0[3], busy.1[0], busy.1[1], busy.1[2]
        );
        println!(
            "whether memory is uncommitted: a placeholder of 1 GiB {:.2} us, a reserved view of 256 MiB {:.2} us, \
             a reserved block before a committed one {:.2} us; among {VIEWS} views, a page's other views {:.3} us, \
             the views of 16 blocks {:.3} us",
            queries[0], queries[1], queries[2], pages_of, between
        );
    }

    // The light refresh (`VICTORY_LIGHT_REFRESH`), after the fork's tests
    // of 94ed363c. The switch is not set in tests, so `writes` itself keeps
    // no bits and asks none; these call `writes_as` with what it would
    // pass, and take their turns with the other tests that touch the
    // guards' notes.

    /// Askings answered by the bits, and askings that went to the guards.
    fn light_counts() -> [u64; 2] {
        [0, 1].map(|i| LIGHT_ASKS[i].load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Whether `[start, end)` is known by its bits.
    fn known(start: usize, end: usize) -> bool {
        published(&JOURNAL.lock().unwrap_or_else(|e| e.into_inner()), start, end)
    }

    /// The pages of an answer of [`writes`] written after the asking `since`.
    fn since(since: u64, written: Vec<(usize, u64)>) -> Vec<usize> {
        written.into_iter().filter(|w| w.1 > since).map(|w| w.0).collect()
    }

    /// The asking that makes blocks clean gives them their bits; the next
    /// one is answered by them, and the guards are not asked. A write takes
    /// the written block's bit, through the guarded view or through another
    /// view of the same pages, and the next asking reports it.
    #[test]
    fn a_range_is_known_by_its_bits_until_a_write() {
        let _alone = one_at_a_time();
        const SIZE: usize = 3 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |block: usize| (one + block * BLOCK) / PAGE;
        assert!(!known(one, one + SIZE), "nothing asked yet");
        // The first asking takes every block in, read-only and clean.
        let (first, all) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(all, [(pages(0), first), (pages(1), first), (pages(2), first)]);
        assert!(known(one, one + SIZE), "made clean and noted");
        // The second is answered by the bits.
        let before = light_counts();
        let (second, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(second > first);
        assert!(since(first, written).is_empty(), "nothing written");
        assert_eq!(light_counts(), [before[0] + 1, before[1]], "the guards were not asked");
        // A write to the guarded view faults: the handler takes the bit.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one, one + SIZE));
        assert!(known(one, one + BLOCK) && known(one + 2 * BLOCK, one + SIZE), "the blocks beside it keep theirs");
        let before = light_counts();
        let (third, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(second, written), [pages(1)]);
        assert_eq!(light_counts(), [before[0], before[1] + 1], "the guards were asked");
        assert!(known(one, one + SIZE), "guarded again and noted");
        // A write through the other view of the same pages takes it too.
        // SAFETY: the second view, as above.
        unsafe { ((two + 2 * BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one + 2 * BLOCK, one + SIZE), "written through another view");
        assert!(known(one, one + 2 * BLOCK));
        let (fourth, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(third, written), [pages(2)]);
        // A write of ours that is told of (`writing`) does as well.
        writing(one + 8, 8);
        assert!(!known(one, one + BLOCK));
        let (_, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(fourth, written), [pages(0)]);
        assert!(known(one, one + SIZE));
        // What the runtime maps anew is not known.
        title_memory_remapped(one + BLOCK, BLOCK);
        assert!(!known(one + BLOCK, one + 2 * BLOCK));
        drop_views(section, &[one, two], SIZE);
    }

    /// `VICTORY_LIGHT_REFRESH=alt`: in the phase without, the bits are not
    /// asked, and they are kept: what is written in that phase, asked about
    /// in it or not, is reported by the first asking of the phase with.
    /// With the switch off no bit is ever set.
    #[test]
    fn bits_kept_through_a_phase_that_does_not_ask_them_are_true_in_the_next() {
        let _alone = one_at_a_time();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |block: usize| (one + block * BLOCK) / PAGE;
        // With: taken in, then known by its bits.
        let (first, _) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(known(one, one + SIZE));
        // Without: the bits are there and are not asked.
        let before = light_counts();
        let (second, written) = writes_as(one, one + SIZE, true, 1).expect("a view");
        assert!(since(first, written).is_empty());
        assert_eq!(light_counts(), [before[0], before[1] + 1], "without: the guards answer");
        // Written in the phase without: the fault takes the bit all the
        // same, and that phase's asking gives it back.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one, one + BLOCK));
        let (third, written) = writes_as(one, one + SIZE, true, 1).expect("a view");
        assert_eq!(since(second, written), [pages(0)]);
        assert!(known(one, one + SIZE));
        // Written again in the phase without, through the other view, and
        // not asked about until the phase with: its first asking goes to
        // the guards and reports it.
        // SAFETY: the second view, as above.
        unsafe { ((two + BLOCK + 8) as *mut u8).write_volatile(1) };
        let before = light_counts();
        let (fourth, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(third, written), [pages(1)], "a bit from before the phase without was trusted");
        assert_eq!(light_counts(), [before[0], before[1] + 1]);
        let before = light_counts();
        let (fifth, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(since(fourth, written).is_empty());
        assert_eq!(light_counts(), [before[0] + 1, before[1]]);
        // The switch off: a write still takes the bit, the asking sets none
        // and counts nothing.
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write_volatile(2) };
        let before = light_counts();
        let (_, written) = writes_as(one, one + SIZE, true, 0).expect("a view");
        assert_eq!(since(fifth, written), [pages(0)]);
        assert_eq!(light_counts(), before);
        assert!(!known(one, one + BLOCK), "off: no bit is set");
        drop_views(section, &[one, two], SIZE);
    }

    /// A range with unmapped blocks is not known by its bits while its
    /// holes are asked of Windows at every asking (no runs kept). With runs
    /// (`VICTORY_GUARD_ABSENT_KNOWN`) a hole that lies in a placeholder is;
    /// then nothing is asked until the runtime tells of the block, or the
    /// look at the holes finds it filled. A hole in an ordinary reservation
    /// is never known by a bit.
    #[test]
    fn holes_are_known_by_their_bits_only_with_runs_and_in_a_placeholder() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER,
            MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _alone = one_at_a_time();
        // SAFETY: a fresh placeholder of four blocks and a pagefile section
        // of four, as the runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (4 * BLOCK) as u32, std::ptr::null());
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree((p + block * BLOCK) as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE;
        let (from, to) = (p, p + 4 * BLOCK);
        let first = map(0);
        assert!(placeholder(p / BLOCK + 1) && placeholder(p / BLOCK + 3), "what is left of the placeholder");
        assert!(!placeholder(p / BLOCK), "a view is none");
        TEST_ABSENT_KNOWN.set(Some(AbsentKnown::Off));
        // The first asking takes every block; blocks 1 to 3 are unmapped.
        let (one, all) = writes_as(from, to, true, 2).expect("a view");
        assert_eq!(since(one - 1, all), [pages(0), pages(1), pages(2), pages(3)]);
        assert!(known(p, p + BLOCK) && !known(from, to));
        // No runs: each asking goes to the guards, which ask Windows.
        let before = light_counts();
        let (two, written) = writes_as(from, to, true, 2).expect("a view");
        assert!(since(one, written).is_empty());
        assert_eq!(light_counts(), [before[0], before[1] + 1]);
        assert!(!known(from, to), "a hole no run answers for has no bit");
        // With runs kept, the asking that walks the holes looks at them.
        TEST_ABSENT_KNOWN.set(Some(AbsentKnown::Forever));
        let (three, written) = writes_as(from, to, true, 2).expect("a view");
        assert!(since(two, written).is_empty());
        assert!(known(from, to), "holes in a placeholder, answered by runs");
        let before = light_counts();
        let (four, written) = writes_as(from, to, true, 2).expect("a view");
        assert!(since(three, written).is_empty());
        assert_eq!(light_counts(), [before[0] + 1, before[1]], "answered by the bits, holes and all");
        // Mapped without a word: not seen, as with the runs alone ...
        let second = map(1);
        let (five, written) = writes_as(from, to, true, 2).expect("a view");
        assert!(since(four, written).is_empty(), "nothing is asked about a hole known by its bit");
        // ... until the look at the holes comes by (it goes round, and
        // waits its turn while another test changes a mapping).
        for _ in 0..2000 {
            if !known(p + BLOCK, p + 2 * BLOCK) {
                break;
            }
            look_at_holes(usize::MAX);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!known(p + BLOCK, p + 2 * BLOCK), "found filled");
        assert!(known(p + 2 * BLOCK, to), "the holes beside it are holes still");
        let (six, written) = writes_as(from, to, true, 2).expect("a view");
        assert_eq!(since(five, written), [pages(1)]);
        assert!(known(from, to));
        // Mapped and told, as the runtime does: the bit goes at once.
        let third = map(2);
        title_memory_remapped(p + 2 * BLOCK, BLOCK);
        assert!(!known(p + 2 * BLOCK, p + 3 * BLOCK));
        let (_, written) = writes_as(from, to, true, 2).expect("a view");
        assert_eq!(since(six, written), [pages(2)]);
        // A hole in an ordinary reservation, answered by a run all the same:
        // anybody can commit it with no word to us, so it gets no bit.
        // SAFETY: a fresh reservation of one block, released below.
        let plain = unsafe { VirtualAlloc(std::ptr::null(), BLOCK, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(plain, 0);
        assert!(!placeholder(plain / BLOCK));
        let (seven, all) = writes_as(plain, plain + BLOCK, true, 2).expect("a view");
        assert_eq!(since(seven - 1, all), [plain / PAGE]);
        let (_, written) = writes_as(plain, plain + BLOCK, true, 2).expect("a view");
        assert!(since(seven, written).is_empty(), "a hole");
        assert!(!known(plain, plain + BLOCK));
        TEST_ABSENT_KNOWN.set(None);
        title_memory_remapped(plain, BLOCK);
        // SAFETY: our views, placeholder and reservation.
        unsafe {
            VirtualFree(plain as *mut _, 0, MEM_RELEASE);
            for view in [first, second, third] {
                UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            }
            for block in 0..4 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 4 * BLOCK);
        assert!(!known(p, p + BLOCK) && guards().holes.range(p / BLOCK..p / BLOCK + 4).next().is_none());
    }

    /// A copy has nothing new to take of a range known by its bits when it
    /// took every page of it after the page's last noted write, and that is
    /// known with nothing asked.
    #[test]
    fn a_copy_that_holds_every_noted_write_has_nothing_new() {
        let _alone = one_at_a_time();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let base = one / PAGE;
        assert!(!nothing_noted(one, one + SIZE, base, &[1, 1]), "never asked about: something would have to be asked");
        let (first, _) = writes_as(one, one + SIZE, true, 2).expect("a view");
        let before = light_counts();
        assert!(nothing_noted(one, one + SIZE, base, &[first, first]), "both pages taken at the asking that found them");
        assert_eq!(light_counts(), [before[0] + 1, before[1]]);
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first - 1]), "a page taken before its write was found");
        assert!(!nothing_noted(one, one + SIZE, base, &[first, 0]), "a page never taken");
        assert!(!nothing_noted(one, one + SIZE, base + 1, &[first, first]), "a range the copy's notes do not hold");
        assert!(!nothing_noted(one, one + SIZE, base, &[first]));
        assert!(!nothing_noted(one, one, base, &[first, first]), "no range");
        // Written: the block's bit is gone, and something must be asked.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first]));
        assert!(nothing_noted(one, one + BLOCK, base, &[first, first]), "the block beside it is as it was");
        let (second, _) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first]), "the journal has the write");
        assert!(nothing_noted(one, one + SIZE, base, &[first, second]));
        drop_views(section, &[one, two], SIZE);
    }

    /// A range's bits are all of its blocks', whatever lies beside them.
    #[test]
    fn bits_of_ranges() {
        static BITS: Bits = Bits::new();
        let blocks = [0usize, 1, 63, 64, 65, 127, 128, 200, 65535, 65536, 65537, 1 << 20, (1 << 31) - 1];
        for &b in &blocks {
            assert!(!BITS.all(b, b + 1));
            BITS.set(b);
            assert!(BITS.all(b, b + 1));
        }
        assert!(BITS.all(0, 2) && BITS.all(63, 66) && BITS.all(127, 129) && BITS.all(65535, 65538));
        assert!(!BITS.all(0, 3) && !BITS.all(62, 64) && !BITS.all(65534, 65536) && !BITS.all(5, 5));
        for b in 300..700 {
            BITS.set(b);
        }
        assert!(BITS.all(300, 700) && !BITS.all(299, 700) && !BITS.all(300, 701));
        BITS.clear(511);
        assert!(!BITS.all(300, 700) && BITS.all(300, 511) && BITS.all(512, 700));
        BITS.set(1 << 40);
        BITS.clear(1 << 40);
        assert!(!BITS.all(1 << 40, (1 << 40) + 1), "outside the address space");
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}
