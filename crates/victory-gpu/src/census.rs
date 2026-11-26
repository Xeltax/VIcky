//! `VICTORY_CENSUS=1` (off by default): what of the copy-back of shader
//! writes comes from memory the title reserved write-combining, and who
//! takes those bytes afterwards. Written to the log every
//! `VICTORY_CENSUS_SECONDS` (10). Logging only: nothing here changes what
//! is copied, when, or where.
//!
//! From fork PR #868 (Chipsraine; the per-range classification and the
//! in-flight count by iGoldenNinja): the `XMemVirtualAlloc` reservation each
//! copy in the card's memory lies in, with the flags the title asked for
//! (the runtime drops write-combining, uncached and the GPU access bits;
//! they are kept here); per copy the bytes sent up, the pieces compared and
//! the bytes fetched back; the host resources made per frame by kind.
//!
//! Ours: this tree keeps many copies of the same title memory (a buffer
//! inside a bigger one with a copy of its own, windows over memory outside
//! buffers, copies in place, textures). A byte a shader wrote through one
//! copy reaches the others only through title memory: the write-back puts
//! it there, the write watch tells the others, and they take it again when
//! next asked. So the census notes, per 64 KiB page of write-combining
//! memory (WC), which copy in the card's memory last wrote bytes back
//! there, and counts every later taker of those bytes:
//! - another copy brought up to date over them (a buffer's copy, a window,
//!   a copy in place) or a texture over them: such a page cannot be left on
//!   the card, the other copy would read old bytes;
//! - the same copy taking the page again (the upload echo: the write-back
//!   made title memory newer than the copy's CPU side, and the whole page
//!   goes up again);
//!
//! and, of the WC bytes fetched, how many lie in pages no other live copy
//! covers, and how many in pages no other copy was ever seen taking after
//! a card write. What the title's own CPU reads is not seen: nothing
//! watches reads. Copies in the PC's memory, merged on the CPU, are counted
//! on a line of their own.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Title memory is noted by 64 KiB page, as the card's compare cuts it.
const PAGE: usize = 64 << 10;

/// Whether the census is on (`VICTORY_CENSUS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CENSUS").is_some_and(|v| v == "1"))
}

/// Whether the pages and the reservations are kept.
pub(crate) fn tracking() -> bool {
    on()
}

fn period() -> Duration {
    static S: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    Duration::from_secs(*S.get_or_init(|| {
        std::env::var("VICTORY_CENSUS_SECONDS").ok().and_then(|v| v.trim().parse().ok()).filter(|&s| s > 0).unwrap_or(10)
    }))
}

/// A reservation of `XMemVirtualAlloc`, as the title asked for it.
struct Reservation {
    len: usize,
    kind: u32,
    flags: u64,
    protect: u32,
}

/// The console's write-combining protection bit.
const WRITE_COMBINE: u32 = 0x400;

static RESERVATIONS: Mutex<BTreeMap<usize, Reservation>> = Mutex::new(BTreeMap::new());

fn reservations() -> MutexGuard<'static, BTreeMap<usize, Reservation>> {
    RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The runtime reserved `[at, at + len)` for the title (`XMemVirtualAlloc`
/// at an address of its choice), with the title's own `kind`, XMem `flags`
/// and console `protect` (before the runtime cleans them).
pub fn title_memory_reserved(at: usize, len: usize, kind: u32, flags: u64, protect: u32) {
    if !tracking() || at == 0 || len == 0 {
        return;
    }
    reservations().insert(at, Reservation { len, kind, flags, protect });
}

/// Whether every title range of `ranges` lies wholly inside a single
/// reservation the title asked write-combining. A copy (a window above
/// all) can cover several reservations, so each range is looked at on its
/// own (iGoldenNinja, fork `f9c43fa`).
pub(crate) fn all_wc(ranges: &[(usize, usize)]) -> bool {
    all_wc_in(&reservations(), ranges)
}

fn all_wc_in(reservations: &BTreeMap<usize, Reservation>, ranges: &[(usize, usize)]) -> bool {
    !ranges.is_empty()
        && ranges.iter().all(|&(from, to)| {
            // Seeking from the exclusive end also finds a reservation that
            // begins inside the range. One ending at `to` holds it.
            from < to
                && reservations.range(..to).next_back().is_some_and(|(&base, r)| {
                    base <= from && r.protect & WRITE_COMBINE != 0 && base.checked_add(r.len).is_some_and(|end| to <= end)
                })
        })
}

/// What a later taker of title memory is ([`asked`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Taker {
    /// A title buffer's copy.
    Copy,
    /// A window over memory outside buffers.
    Window,
    /// A copy in place (the card reads title memory itself).
    InPlace,
    /// A texture read from title memory.
    Texture,
}

/// A copy of title memory, as the census names it.
#[derive(Clone, Copy)]
pub(crate) struct Which {
    pub(crate) id: u64,
    pub(crate) start: u64,
    pub(crate) len: u64,
    pub(crate) window: bool,
}

/// What a texture is named by among the takers: its memory's address with
/// the top bit set (copies are numbered from 1 up).
pub(crate) fn texture_taker(at: usize) -> u64 {
    (1 << 63) | at as u64
}

/// The counters, since the start ([`COUNTS`]).
mod n {
    /// Pieces of copies in the card's memory compared, all and WC.
    pub(super) const COMPARED: usize = 0;
    pub(super) const COMPARED_WC: usize = 1;
    /// Bytes fetched back from the card, all and WC.
    pub(super) const FETCHED: usize = 2;
    pub(super) const FETCHED_WC: usize = 3;
    /// Bytes the merges of those stored into title memory, all and WC.
    pub(super) const WRITTEN: usize = 4;
    pub(super) const WRITTEN_WC: usize = 5;
    /// WC bytes fetched in pages one copy covers, at one address.
    pub(super) const WC_ALONE: usize = 6;
    /// WC bytes fetched in pages no other copy took after a card write so
    /// far, at one address.
    pub(super) const WC_UNTAKEN: usize = 7;
    /// WC bytes fetched in pages shown at more than one address.
    pub(super) const WC_MULTI: usize = 8;
    /// Bytes a copy in the card's memory took again of pages it wrote back.
    pub(super) const ECHO: usize = 9;
    /// Bytes of card-written WC pages taken afterwards by others, by kind.
    pub(super) const TAKEN_COPY: usize = 10;
    pub(super) const TAKEN_WINDOW: usize = 11;
    pub(super) const TAKEN_IN_PLACE: usize = 12;
    pub(super) const TAKEN_TEXTURE: usize = 13;
    /// Pages a first other taker took after a card write.
    pub(super) const TAKEN_PAGES: usize = 14;
    /// Pages written back through a copy after another copy wrote them back.
    pub(super) const TWO_WRITERS: usize = 15;
    /// Copies in the PC's memory: declared bytes compared and bytes stored
    /// into title memory, all and WC.
    pub(super) const PC_COMPARED: usize = 16;
    pub(super) const PC_COMPARED_WC: usize = 17;
    pub(super) const PC_WRITTEN: usize = 18;
    pub(super) const PC_WRITTEN_WC: usize = 19;
    /// Bytes sent up to copies in the card's memory.
    pub(super) const SENT: usize = 20;
    pub(super) const COUNT: usize = 21;
}

static COUNTS: [AtomicU64; n::COUNT] = [const { AtomicU64::new(0) }; n::COUNT];

fn count(i: usize, by: u64) {
    if by > 0 {
        COUNTS[i].fetch_add(by, Ordering::Relaxed);
    }
}

/// A page of WC title memory shaders wrote through a copy in the card's
/// memory, and the merge stored into title memory.
#[derive(Debug)]
struct Written {
    /// The copy that wrote it back last.
    owner: u64,
    /// The title bytes written back since its last other taker, as the
    /// smallest stretch that holds them.
    lo: usize,
    hi: usize,
    /// Who took them since (copies' numbers, [`texture_taker`]s).
    takers: Vec<u64>,
    /// Whether the owner took the page again since (the echo).
    echoed: bool,
}

/// The pages noted ([`Written`]), how many live copies cover each page,
/// and the pages another copy took after a card write.
#[derive(Default)]
struct Pages {
    written: BTreeMap<usize, Written>,
    /// Pages noted per owner.
    owners: HashMap<u64, usize>,
    cover: HashMap<usize, u32>,
    taken: HashSet<usize>,
}

/// What [`Pages::took`] found: bytes taken, and pages that had no other
/// taker before.
#[derive(Default, Debug, PartialEq, Eq)]
struct Took {
    bytes: usize,
    pages: usize,
}

impl Pages {
    /// The pages of `[from, to)`.
    fn pages(from: usize, to: usize) -> std::ops::Range<usize> {
        if from >= to { 0..0 } else { from / PAGE..(to - 1) / PAGE + 1 }
    }

    /// A copy of `[start, start + len)` was made (`by` 1) or is gone (`-1`).
    fn cover(&mut self, start: usize, len: usize, made: bool) {
        for page in Self::pages(start, start.saturating_add(len)) {
            let c = self.cover.entry(page).or_insert(0);
            if made {
                *c += 1;
            } else {
                *c = c.saturating_sub(1);
                if *c == 0 {
                    self.cover.remove(&page);
                }
            }
        }
    }

    /// How many live copies cover `page`.
    fn covered(&self, page: usize) -> u32 {
        self.cover.get(&page).copied().unwrap_or(0)
    }

    /// Copy `owner` wrote title bytes `[lo, hi)` (inside one page) back.
    /// True if another copy had written the page back before.
    fn wrote(&mut self, owner: u64, lo: usize, hi: usize) -> bool {
        if lo >= hi {
            return false;
        }
        let page = lo / PAGE;
        let mut two = false;
        match self.written.get_mut(&page) {
            Some(w) => {
                if w.owner != owner {
                    two = true;
                    self.taken.insert(page);
                    if let Some(n) = self.owners.get_mut(&w.owner) {
                        *n -= 1;
                        if *n == 0 {
                            self.owners.remove(&w.owner);
                        }
                    }
                    *self.owners.entry(owner).or_insert(0) += 1;
                    w.owner = owner;
                    (w.lo, w.hi) = (lo, hi);
                } else if w.takers.is_empty() && !w.echoed {
                    // Nobody took the earlier bytes yet: they are all new.
                    (w.lo, w.hi) = (w.lo.min(lo), w.hi.max(hi));
                } else {
                    (w.lo, w.hi) = (lo, hi);
                }
                w.takers.clear();
                w.echoed = false;
            }
            None => {
                self.written.insert(page, Written { owner, lo, hi, takers: Vec::new(), echoed: false });
                *self.owners.entry(owner).or_insert(0) += 1;
            }
        }
        two
    }

    /// `taker` (not `owner` of a page) brought `[from, to)` up to date:
    /// what of the noted pages' written bytes it took for the first time
    /// since their write.
    fn took(&mut self, taker: u64, from: usize, to: usize) -> Took {
        let mut took = Took::default();
        let mut first = Vec::new();
        for (&page, w) in self.written.range_mut(Self::pages(from, to)) {
            if w.owner == taker || w.takers.contains(&taker) {
                continue;
            }
            let (a, b) = (w.lo.max(from), w.hi.min(to));
            if a >= b {
                continue;
            }
            if w.takers.is_empty() {
                took.pages += 1;
                first.push(page);
            }
            w.takers.push(taker);
            took.bytes += b - a;
        }
        self.taken.extend(first);
        took
    }

    /// Copy `id` took title memory `[from, to)` again: the bytes of pages it
    /// wrote back that it takes for the first time since (whole pages' parts,
    /// as they go up again).
    fn recopied(&mut self, id: u64, from: usize, to: usize) -> usize {
        if !self.owners.contains_key(&id) {
            return 0;
        }
        let mut bytes = 0;
        for (&page, w) in self.written.range_mut(Self::pages(from, to)) {
            if w.owner != id || w.echoed {
                continue;
            }
            w.echoed = true;
            bytes += ((page + 1) * PAGE).min(to) - (page * PAGE).max(from);
        }
        bytes
    }
}

static PAGES: LazyLock<Mutex<Pages>> = LazyLock::new(|| Mutex::new(Pages::default()));

fn pages() -> MutexGuard<'static, Pages> {
    PAGES.lock().unwrap_or_else(|e| e.into_inner())
}

/// How many copies own noted pages, and the one when it is one: an ask of
/// that copy has nothing to look for.
static OWNERS: AtomicUsize = AtomicUsize::new(0);
static SOLE_OWNER: AtomicU64 = AtomicU64::new(0);

fn owners_now(p: &Pages) {
    OWNERS.store(p.owners.len(), Ordering::Relaxed);
    SOLE_OWNER.store(p.owners.keys().next().copied().unwrap_or(0), Ordering::Relaxed);
}

/// A copy of title memory `[start, start + len)` was made (any kind: in the
/// card's memory, in the PC's, in place; a buffer's, a window).
pub(crate) fn copy_made(start: u64, len: u64) {
    if tracking() {
        pages().cover(start as usize, len as usize, true);
    }
}

/// That copy is gone.
pub(crate) fn copy_gone(start: u64, len: u64) {
    if tracking() {
        pages().cover(start as usize, len as usize, false);
    }
}

/// `taker` (a copy's number or a [`texture_taker`]) is brought up to date
/// over title memory `[from, to)`, or reads it in place.
pub(crate) fn asked(taker: u64, kind: Taker, from: usize, to: usize) {
    if !tracking() || from >= to {
        return;
    }
    match OWNERS.load(Ordering::Relaxed) {
        0 => return,
        1 if SOLE_OWNER.load(Ordering::Relaxed) == taker => return,
        _ => {}
    }
    let took = pages().took(taker, from, to);
    if took.bytes == 0 {
        return;
    }
    let i = match kind {
        Taker::Copy => n::TAKEN_COPY,
        Taker::Window => n::TAKEN_WINDOW,
        Taker::InPlace => n::TAKEN_IN_PLACE,
        Taker::Texture => n::TAKEN_TEXTURE,
    };
    count(i, took.bytes as u64);
    count(n::TAKEN_PAGES, took.pages as u64);
}

/// Copy `id`, in the card's memory, took title memory `[from, to)` into its
/// CPU side again (and up to the card).
pub(crate) fn recopied(id: u64, from: usize, to: usize) {
    if !tracking() || from >= to || OWNERS.load(Ordering::Relaxed) == 0 {
        return;
    }
    count(n::ECHO, pages().recopied(id, from, to) as u64);
}

/// Counters of one copy in the card's memory, since the start.
#[derive(Default, Clone, Copy)]
struct Row {
    start: u64,
    len: u64,
    window: bool,
    sent: u64,
    sent_ranges: u64,
    compared: u64,
    fetched: u64,
    fetched_pieces: u64,
    written: u64,
}

static ROWS: Mutex<BTreeMap<u64, Row>> = Mutex::new(BTreeMap::new());

fn row(which: Which, f: impl FnOnce(&mut Row)) {
    let mut rows = ROWS.lock().unwrap_or_else(|e| e.into_inner());
    let r = rows.entry(which.id).or_default();
    (r.start, r.len, r.window) = (which.start, which.len, which.window);
    f(r);
}

/// `bytes` of copy `which` were sent up to the card in `ranges` ranges.
pub(crate) fn sent(which: Which, bytes: u64, ranges: u64) {
    if !on() {
        return;
    }
    count(n::SENT, bytes);
    row(which, |r| {
        r.sent += bytes;
        r.sent_ranges += ranges;
    });
    report_if_due();
}

/// `pieces` 64 KiB pieces of copy `which` were compared on the card for the
/// declared title range `declared`.
pub(crate) fn compared(which: Which, declared: (usize, usize), pieces: u64) {
    if !on() {
        return;
    }
    count(n::COMPARED, pieces);
    if all_wc(&[declared]) {
        count(n::COMPARED_WC, pieces);
    }
    row(which, |r| r.compared += pieces);
}

/// Offsets `[lo, hi)` of copy `which` came back from the card and its
/// merge stored `written` bytes of them into title memory. For a piece
/// taken from an in-flight slot, `[lo, hi)` is what of the slot is merged
/// (bytes a newer fetch brought are left out).
pub(crate) fn merged(which: Which, lo: usize, hi: usize, written: usize) {
    if !on() || lo >= hi {
        return;
    }
    let (a, b) = (which.start as usize + lo, which.start as usize + hi);
    let bytes = (hi - lo) as u64;
    count(n::FETCHED, bytes);
    count(n::WRITTEN, written as u64);
    row(which, |r| {
        r.fetched += bytes;
        r.fetched_pieces += 1;
        r.written += written as u64;
    });
    if all_wc(&[(a, b)]) {
        count(n::FETCHED_WC, bytes);
        count(n::WRITTEN_WC, written as u64);
        let multi = crate::inplace::view_census(a, b - a).1 > 0;
        let mut p = pages();
        if multi {
            count(n::WC_MULTI, bytes);
        } else {
            let page = a / PAGE;
            if p.covered(page) <= 1 {
                count(n::WC_ALONE, bytes);
            }
            if !p.taken.contains(&page) {
                count(n::WC_UNTAKEN, bytes);
            }
        }
        if written > 0 {
            if p.wrote(which.id, a, b) {
                count(n::TWO_WRITERS, 1);
            }
            owners_now(&p);
        }
    }
    report_if_due();
}

/// A copy in the PC's memory: shaders' declared title range `[from, to)`
/// was compared and `written` bytes went into title memory.
pub(crate) fn pc_back(from: usize, to: usize, written: usize) {
    if !on() || from >= to {
        return;
    }
    count(n::PC_COMPARED, (to - from) as u64);
    count(n::PC_WRITTEN, written as u64);
    if all_wc(&[(from, to)]) {
        count(n::PC_COMPARED_WC, (to - from) as u64);
        count(n::PC_WRITTEN_WC, written as u64);
    }
}

/// Host resources made, by where: count and bytes ([`made`]).
const KINDS: [&str; 9] =
    ["read-back", "upload", "shared", "uav", "texture", "reserved texture", "depth twin", "placed in title memory", "host list"];
static MADE: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];

/// With the census on: a host resource of kind `kind` ([`KINDS`]) and
/// `bytes` (0 when not known) was asked of the device.
pub(crate) fn made(kind: usize, bytes: u64) {
    if on() && kind < KINDS.len() {
        MADE[kind * 2].fetch_add(1, Ordering::Relaxed);
        MADE[kind * 2 + 1].fetch_add(bytes, Ordering::Relaxed);
    }
}

/// The console protection bits the runtime drops, spelled out.
fn protect_words(p: u32) -> String {
    let mut w = Vec::new();
    if p & WRITE_COMBINE != 0 {
        w.push("WC");
    }
    if p & 0x200 != 0 {
        w.push("uncached");
    }
    if p & 0x100 != 0 {
        w.push("guard");
    }
    let gpu = p & 0x7_f800;
    let mut s = w.join("+");
    if gpu != 0 {
        if !s.is_empty() {
            s.push('+');
        }
        s.push_str(&format!("gpu{gpu:#x}"));
    }
    if s.is_empty() { "-".to_owned() } else { s }
}

/// The last report: when, at which frame, and the counters then.
struct Last {
    at: Instant,
    frame: u64,
    counts: [u64; n::COUNT],
    made: [u64; 18],
    rows: BTreeMap<u64, Row>,
}

static LAST: Mutex<Option<Last>> = Mutex::new(None);

fn report_if_due() {
    let frame = crate::frame::current();
    let now = Instant::now();
    let Ok(mut last) = LAST.try_lock() else {
        return;
    };
    let counts: [u64; n::COUNT] = std::array::from_fn(|i| COUNTS[i].load(Ordering::Relaxed));
    let made: [u64; 18] = std::array::from_fn(|i| MADE[i].load(Ordering::Relaxed));
    let Some(was) = last.as_mut() else {
        let rows = ROWS.lock().unwrap_or_else(|e| e.into_inner()).clone();
        *last = Some(Last { at: now, frame, counts, made, rows });
        return;
    };
    if now.duration_since(was.at) < period() {
        return;
    }
    let rows = ROWS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let frames = frame.saturating_sub(was.frame).max(1) as f64;
    let secs = now.duration_since(was.at).as_secs_f64();
    let d: [f64; n::COUNT] = std::array::from_fn(|i| counts[i].saturating_sub(was.counts[i]) as f64 / frames);
    let mb = |b: f64| b / 1e6;
    let (taken_since_start, covered_pages, noted) = {
        let p = pages();
        (p.taken.len(), p.cover.len(), p.written.len())
    };
    tracing::info!(
        target: "census",
        "census over {secs:.1} s, {frames:.0} frames, copies in the card's memory, per frame: {:.1} MB sent up, {:.0} pieces compared \
         ({:.0} of write-combining memory), {:.1} MB fetched back ({:.1} MB WC), {:.1} MB of it written into title memory ({:.1} MB WC)",
        mb(d[n::SENT]),
        d[n::COMPARED],
        d[n::COMPARED_WC],
        mb(d[n::FETCHED]),
        mb(d[n::FETCHED_WC]),
        mb(d[n::WRITTEN]),
        mb(d[n::WRITTEN_WC]),
    );
    tracing::info!(
        target: "census",
        "census of the WC bytes fetched, per frame: {:.1} MB in pages no other live copy covers, {:.1} MB in pages no other copy took \
         after a card write so far, {:.1} MB in pages shown at more than one address. Taken again after their write-back: by the copy \
         that wrote them (the upload echo) {:.1} MB, by other copies {:.2} MB, windows {:.2} MB, copies in place {:.2} MB, textures \
         {:.2} MB; {:.1} pages a frame taken by another for the first time, {taken_since_start} pages since the start, {:.2} pages a \
         frame written back through two copies in turn ({noted} WC pages noted, {covered_pages} pages under copies). The title's CPU \
         reads are not seen",
        mb(d[n::WC_ALONE]),
        mb(d[n::WC_UNTAKEN]),
        mb(d[n::WC_MULTI]),
        mb(d[n::ECHO]),
        mb(d[n::TAKEN_COPY]),
        mb(d[n::TAKEN_WINDOW]),
        mb(d[n::TAKEN_IN_PLACE]),
        mb(d[n::TAKEN_TEXTURE]),
        d[n::TAKEN_PAGES],
        d[n::TWO_WRITERS],
    );
    tracing::info!(
        target: "census",
        "census of copies in the PC's memory, per frame: {:.1} MB of declared ranges compared ({:.1} MB WC), {:.1} MB written into \
         title memory ({:.1} MB WC)",
        mb(d[n::PC_COMPARED]),
        mb(d[n::PC_COMPARED_WC]),
        mb(d[n::PC_WRITTEN]),
        mb(d[n::PC_WRITTEN_WC]),
    );
    {
        let parts: Vec<String> = KINDS
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let count = made[k * 2].saturating_sub(was.made[k * 2]) as f64 / frames;
                let bytes = made[k * 2 + 1].saturating_sub(was.made[k * 2 + 1]) as f64 / frames;
                format!("{name} {count:.1} ({:.1} MB)", mb(bytes))
            })
            .collect();
        let all: u64 = (0..KINDS.len()).map(|k| made[k * 2].saturating_sub(was.made[k * 2])).sum();
        tracing::info!(target: "census", "census resources made per frame: {:.1} in all; {}", all as f64 / frames, parts.join(", "));
    }
    {
        let reservations = reservations();
        let mut lines: Vec<(f64, String)> = rows
            .iter()
            .map(|(id, now_)| {
                let was_ = was.rows.get(id).copied().unwrap_or_default();
                let per = |a: u64, b: u64| a.saturating_sub(b) as f64 / frames;
                let (sent, fetched) = (per(now_.sent, was_.sent), per(now_.fetched, was_.fetched));
                let start = now_.start as usize;
                let res = reservations.range(..=start).next_back().filter(|(base, r)| *base + r.len > start);
                let (res, words) = match res {
                    Some((base, r)) => (
                        format!("{base:#x}+{}MB kind {:#x} flags {:#x} protect {:#x}", r.len >> 20, r.kind, r.flags, r.protect),
                        protect_words(r.protect),
                    ),
                    None => ("not in a recorded reservation".to_owned(), "?".to_owned()),
                };
                let line = format!(
                    "census copy #{id} {start:#x}+{:.1}MB{}: per frame sent {:.2} MB in {:.1} ranges, compared {:.0} pieces, fetched \
                     {:.2} MB in {:.1} pieces, written {:.2} MB; {words} [{res}]",
                    mb(now_.len as f64),
                    if now_.window { " (window)" } else { "" },
                    mb(sent),
                    per(now_.sent_ranges, was_.sent_ranges),
                    per(now_.compared, was_.compared),
                    mb(fetched),
                    per(now_.fetched_pieces, was_.fetched_pieces),
                    mb(per(now_.written, was_.written)),
                );
                (sent + fetched, line)
            })
            .filter(|(traffic, _)| *traffic > 0.0)
            .collect();
        drop(reservations);
        lines.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, line) in lines.iter().take(24) {
            tracing::info!(target: "census", "{line}");
        }
    }
    *was = Last { at: now, frame, counts, made, rows };
}

#[cfg(test)]
mod tests {
    use super::{PAGE, Pages, Reservation, Took, all_wc_in};
    use std::collections::BTreeMap;

    fn reservation(len: usize, protect: u32) -> Reservation {
        Reservation { len, kind: 0, flags: 0, protect }
    }

    // iGoldenNinja's (fork `f9c43fa`).
    #[test]
    fn a_write_in_the_ordinary_half_of_a_card_window_still_comes_back() {
        const MIB: usize = 1 << 20;
        let window = 8 * MIB;
        let middle = window + 2 * MIB;
        let map = BTreeMap::from([(window, reservation(2 * MIB, 0x404)), (middle, reservation(2 * MIB, 0x04))]);
        let wc = (middle - 65536, middle);
        let ordinary = (middle, middle + 65536);
        assert!(all_wc_in(&map, &[wc]));
        assert!(!all_wc_in(&map, &[ordinary]));
        assert!(!all_wc_in(&map, &[wc, ordinary]));
        assert!(!all_wc_in(&map, &[(wc.0, ordinary.1)]));
    }

    // iGoldenNinja's (fork `f9c43fa`).
    #[test]
    fn a_declaration_must_fit_one_known_reservation() {
        let map = BTreeMap::from([
            (0x10000, reservation(0x10000, 0x404)),
            (0x20000, reservation(0x10000, 0x404)),
            (usize::MAX - 8, reservation(16, 0x404)),
        ]);
        assert!(all_wc_in(&map, &[(0x10000, 0x20000), (0x20000, 0x30000)]));
        assert!(!all_wc_in(&map, &[(0x1ffff, 0x20001)]));
        assert!(!all_wc_in(&map, &[(0x30000, 0x30001)]));
        assert!(!all_wc_in(&map, &[(0x10000, 0x10000)]));
        assert!(!all_wc_in(&map, &[(usize::MAX - 8, 1)]));
        assert!(!all_wc_in(&map, &[(usize::MAX - 8, usize::MAX)]));
    }

    #[test]
    fn another_copy_takes_a_written_page_once_per_write_and_the_owner_never() {
        let mut p = Pages::default();
        let page = 7 * PAGE;
        assert!(!p.wrote(1, page + 16, page + 64));
        // The owner asking is no other taker.
        assert_eq!(p.took(1, page, page + PAGE), Took::default());
        // Another copy over part of the written bytes: those bytes, once.
        assert_eq!(p.took(2, page + 32, page + PAGE), Took { bytes: 32, pages: 1 });
        assert_eq!(p.took(2, page, page + PAGE), Took::default());
        // A third taker counts its bytes, not the page again.
        assert_eq!(p.took(3, page, page + PAGE), Took { bytes: 48, pages: 0 });
        // Asks beside the written bytes take nothing.
        assert_eq!(p.took(4, page + 64, page + PAGE), Took::default());
        assert!(p.taken.contains(&7));
        // A new write: taken again.
        assert!(!p.wrote(1, page, page + 8));
        assert_eq!(p.took(2, page, page + PAGE), Took { bytes: 8, pages: 1 });
    }

    #[test]
    fn writes_nobody_took_yet_add_up_and_a_second_writer_is_told() {
        let mut p = Pages::default();
        let page = 3 * PAGE;
        p.wrote(1, page, page + 8);
        p.wrote(1, page + 100, page + 108);
        assert_eq!(p.took(2, page, page + PAGE).bytes, 108);
        assert!(p.wrote(5, page + 4, page + 12));
        assert_eq!(p.owners.get(&1), None);
        assert_eq!(p.owners.get(&5), Some(&1));
        // The first owner is now another taker.
        assert_eq!(p.took(1, page, page + PAGE).bytes, 8);
    }

    #[test]
    fn the_echo_is_the_owners_page_taken_again_once_per_write() {
        let mut p = Pages::default();
        let page = 9 * PAGE;
        p.wrote(1, page + 10, page + 20);
        assert_eq!(p.recopied(2, page, page + PAGE), 0);
        assert_eq!(p.recopied(1, page, page + PAGE), PAGE);
        assert_eq!(p.recopied(1, page, page + PAGE), 0);
        p.wrote(1, page + 10, page + 20);
        // Part of the page taken: that part goes up again.
        assert_eq!(p.recopied(1, page + PAGE / 2, page + PAGE), PAGE / 2);
    }

    #[test]
    fn copies_cover_their_pages_until_they_go() {
        let mut p = Pages::default();
        p.cover(PAGE - 1, 2, true);
        p.cover(PAGE, 4 * PAGE, true);
        assert_eq!((p.covered(0), p.covered(1), p.covered(4), p.covered(5)), (1, 2, 1, 0));
        p.cover(PAGE - 1, 2, false);
        assert_eq!((p.covered(0), p.covered(1)), (0, 1));
        assert_eq!(Pages::pages(5, 5), 0..0);
    }
}
