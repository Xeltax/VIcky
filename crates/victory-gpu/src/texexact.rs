//! VIcky: streamed textures uploaded exactly, and the audit that says in one
//! run whether they are (VICKY-RT.md, "Streamed textures: exact uploads and
//! the audit"). Both off by default; with every switch unset nothing here
//! runs and no line is printed.
//!
//! The streamed texture staleness switches (`crate::texstale`) lowered the
//! stale slots but left a lottery per texture: in about half of the boat
//! runs a texture picked by index kept wrong bytes for a whole cutscene (a
//! mosaic: a recycled block's last texture; or another colourway). Fable's
//! ranked holes (`xcl-notes` `fable-2026-11-28-texstale2.md`): writes that
//! land with no event for the slot (the spans name only the nearest texture
//! before a write), a hold asked at the look while the bytes are read at the
//! submission, and checks that compare memory now with memory when the
//! upload was recorded instead of with what was uploaded.
//!
//! `VICTORY_TEXTURE_EXACT=1` ([`on`]) in five parts, each its own letter in
//! `VICTORY_TEXTURE_EXACT_PARTS` (all of `abcde` unless said):
//! - **A**: the passes over memory events name every texture whose viewed
//!   levels overlap the range (`descriptor::host` `Spans::slots_exact`), not
//!   only the nearest span before it; textures draws keep staged are read
//!   again after any such event.
//! - **B**: each detiling notes a fingerprint and a full hash of the bytes it
//!   took; the next pass puts them in the slot's checked entry ([`Post`]), so
//!   the looks compare memory with what was uploaded. A read that took
//!   nothing posts [`FAILED_PRINT`].
//! - **C**: a refresh samples the DirectStorage write sequence around its
//!   read ([`read_began`], [`torn`]); a write that overlapped the read makes
//!   the slots there due next frame and the staging read again, at most
//!   [`TORN_MOST`] times in a row for one texture.
//! - **D**: memory mapped and not written since ([`mapped`], [`written`]), or
//!   under a DirectStorage write, holds the uploads of the textures there for
//!   at most [`HOLD_MS`] ([`not_ready`]): they keep their previous host copy.
//! - **E**: a texture quiet for [`QUIET`] frames since its detiling has its
//!   memory hashed whole on the wide threads ([`settle_tick`], about
//!   [`settle_mb`] MB a frame, the newest first) against B's hash; a
//!   difference makes its slots due and its staging read again.
//!
//! `VICTORY_TEXTURE_AUDIT=<frame>,...` ([`audit`]): at those world frames,
//! every texture with a host copy and a current view, its memory against
//! what was last uploaded, in one table.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Mutex, OnceLock};

pub(crate) mod audit;
#[cfg(test)]
pub(crate) mod tests;

/// The parts of `VICTORY_TEXTURE_EXACT`, as bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    A = 1,
    B = 2,
    C = 4,
    D = 8,
    E = 16,
}

/// The parts `VICTORY_TEXTURE_EXACT_PARTS` names (letters a to e, any
/// order, others ignored); unset or empty: all five. Pure, for testing.
pub(crate) fn parse_parts(v: Option<&str>) -> u8 {
    match v.map(str::trim) {
        None | Some("") => 0x1f,
        Some(s) => s.chars().fold(0, |bits, c| {
            bits | match c.to_ascii_lowercase() {
                'a' => 1,
                'b' => 2,
                'c' => 4,
                'd' => 8,
                'e' => 16,
                _ => 0,
            }
        }),
    }
}

/// The letters of `bits`.
fn letters(bits: u8) -> String {
    "abcde".chars().enumerate().filter(|&(i, _)| bits & (1 << i) != 0).map(|(_, c)| c).collect()
}

/// The parts on (0: `VICTORY_TEXTURE_EXACT` is not `1`).
fn parts() -> u8 {
    static PARTS: OnceLock<u8> = OnceLock::new();
    *PARTS.get_or_init(|| {
        if !std::env::var_os("VICTORY_TEXTURE_EXACT").is_some_and(|v| v == "1") {
            return 0;
        }
        let bits = parse_parts(std::env::var("VICTORY_TEXTURE_EXACT_PARTS").ok().as_deref());
        tracing::info!(
            "streamed textures uploaded exactly (VICTORY_TEXTURE_EXACT=1), parts {} (VICTORY_TEXTURE_EXACT_PARTS): A exact \
             events, B prints of the bytes uploaded, C torn reads, D holds of memory not written yet (at most {HOLD_MS} ms), \
             E settle checks after {QUIET} quiet frames ({} MB a frame); counts on the 'texture exact:' line",
            letters(bits),
            settle_mb()
        );
        bits
    })
}

/// Whether `VICTORY_TEXTURE_EXACT=1` with any part.
pub(crate) fn on() -> bool {
    parts() != 0
}

/// Whether part `p` is on.
pub(crate) fn part(p: Part) -> bool {
    parts() & p as u8 != 0
}

/// Whether what each detiling took is kept per texture ([`Record`]): for
/// part E and for the audit.
pub(crate) fn records() -> bool {
    part(Part::E) || audit::on()
}

/// Whether a detiling hashes what it took (records, or part B's posts).
pub(crate) fn hashes() -> bool {
    records() || part(Part::B)
}

/// Whether the DirectStorage writes under way are noted
/// (`crate::texstale::title_memory_write_pending`) for parts C and D, also
/// without `VICTORY_DS_HOLD`.
pub(crate) fn tracks_writes() -> bool {
    part(Part::C) || part(Part::D)
}

/// The first 8 bytes of a full hash.
pub(crate) fn h64(hash: &blake3::Hash) -> u64 {
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(first)
}

// The counts, since the start (the bytes hashed: since the last line).

/// What the parts count ([`count`]).
#[derive(Clone, Copy, Debug)]
pub(crate) enum N {
    /// A: slots the exact spans named for memory events; of them, slots the
    /// old rule missed; slots the old rule named that the exact spans leave
    /// out (their viewed levels end before the range); textures staged for
    /// draws read again after an event.
    ANamed,
    AMissed,
    ALeft,
    ARestaged,
    /// B: posts put in a checked entry; of them, prints that differed from
    /// the print taken when the upload was recorded; reads that took nothing
    /// (posted as [`FAILED_PRINT`]); posts lost (no pass took them).
    BDrained,
    BDiffered,
    BFailed,
    BLost,
    /// C: torn reads caught; re-dues refused after [`TORN_MOST`] in a row.
    CTorn,
    CCapped,
    /// D: looks of the pass held; draw uploads or checks held; holds let go
    /// by the [`HOLD_MS`] bound.
    DPass,
    DDraw,
    DReleased,
    /// E: settle checks; mismatches found; of them, fixed (the next
    /// detiling of that texture checked clean); given up after
    /// [`STRIKES_MOST`]; memory found all zeros (not written yet, looked at
    /// again later); bytes hashed since the last line.
    ECheck,
    EMismatch,
    EFixed,
    EGivenUp,
    EZeros,
    EBytes,
}

static COUNTS: [AtomicU64; 19] = [const { AtomicU64::new(0) }; 19];

/// Counts `by` of `n`.
pub(crate) fn count(n: N, by: u64) {
    if by > 0 {
        COUNTS[n as usize].fetch_add(by, Relaxed);
    }
}

fn counted(n: N) -> u64 {
    COUNTS[n as usize].load(Relaxed)
}

// Part B: the print of the bytes uploaded.

/// The print posted for a pass upload whose read took nothing (a page
/// missing): even, so no fingerprint (always odd) matches it, and the next
/// look uploads the slot again.
pub(crate) const FAILED_PRINT: u64 = 2;

/// What the detiling of a texture the pass uploaded for a slot took, for
/// that slot's checked entry (`descriptor::host` `TextureQueue::exact_print`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Post {
    pub(crate) index: u32,
    /// The queue's write number the slot was uploaded as.
    pub(crate) write: u64,
    /// `crate::texture::fingerprint_of` the bytes, over the slot's ranges.
    pub(crate) print: u64,
    /// [`h64`] of the bytes.
    pub(crate) hash: u64,
    pub(crate) failed: bool,
}

static POSTS: Mutex<Vec<Post>> = Mutex::new(Vec::new());
/// Posts kept until a pass takes them.
const POSTS_MOST: usize = 1 << 16;

/// Part B: a detiling at submission took `post` (on the submitting thread
/// or a wide thread; locks one mutex and calls nothing).
pub(crate) fn post(post: Post) {
    if post.failed {
        count(N::BFailed, 1);
    }
    let mut posts = POSTS.lock().unwrap_or_else(|e| e.into_inner());
    if posts.len() >= POSTS_MOST {
        count(N::BLost, posts.len() as u64);
        posts.clear();
    }
    posts.push(post);
}

/// The posts since the last call (the texture pass).
pub(crate) fn take_posts() -> Vec<Post> {
    std::mem::take(&mut *POSTS.lock().unwrap_or_else(|e| e.into_inner()))
}

// What each detiling took, per texture (parts E and the audit).

/// What the last detiling of a title texture took.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Record {
    /// Its title memory: address and bytes.
    pub(crate) at: u64,
    pub(crate) len: u64,
    /// [`h64`] of the bytes detiled (zeros for pages not mapped).
    pub(crate) hash: u64,
    /// The frame (`crate::frame::current`) of the detiling.
    pub(crate) frame: u64,
    /// Not all of its memory was mapped then.
    pub(crate) partial: bool,
    /// Its host copy's format (DXGI) and size, for the audit's lines.
    pub(crate) format: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// E: the frame from which it waits in [`UNSETTLED`] (0: not there).
    key: u64,
    /// E: a settle check found it differing and no later detiling has been
    /// checked clean since (carried over to the next record), and the checks
    /// in a row that found it differing or all zeros.
    open: bool,
    strikes: u8,
}

impl Record {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(at: u64, len: u64, hash: u64, frame: u64, partial: bool, format: u32, width: u32, height: u32) -> Self {
        Record { at, len, hash, frame, partial, format, width, height, ..Default::default() }
    }
}

static RECORDS: Mutex<Option<HashMap<usize, Record>>> = Mutex::new(None);

/// E: the records to check, by (the frame from which they may be checked,
/// title texture), each with its record's frame and its bytes. Locked alone,
/// or inside [`RECORDS`]' lock, never around it.
static UNSETTLED: Mutex<BTreeMap<(u64, usize), (u64, u64)>> = Mutex::new(BTreeMap::new());
/// The most records waiting for a check (the oldest go past it).
const UNSETTLED_MOST: usize = 1 << 16;

fn records_map() -> std::sync::MutexGuard<'static, Option<HashMap<usize, Record>>> {
    RECORDS.lock().unwrap_or_else(|e| e.into_inner())
}

fn unsettled() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), (u64, u64)>> {
    UNSETTLED.lock().unwrap_or_else(|e| e.into_inner())
}

/// A detiling of title texture `resource` took `rec` (on the submitting
/// thread or a wide thread).
pub(crate) fn detiled(resource: usize, mut rec: Record) {
    let mut map = records_map();
    let map = map.get_or_insert_with(HashMap::new);
    let old = map.get(&resource).copied();
    if let Some(old) = old {
        (rec.open, rec.strikes) = (old.open, old.strikes);
    }
    if part(Part::E) {
        let mut set = unsettled();
        if let Some(old) = old
            && old.key != 0
        {
            set.remove(&(old.key, resource));
        }
        rec.key = rec.frame.max(1);
        set.insert((rec.key, resource), (rec.frame, rec.len));
        while set.len() > UNSETTLED_MOST {
            let Some(((_, gone), _)) = set.pop_first() else {
                break;
            };
            if let Some(r) = map.get_mut(&gone) {
                r.key = 0;
            }
        }
    }
    map.insert(resource, rec);
}

/// The title released `resource`: its record goes.
pub(crate) fn forget(resource: usize) {
    if !records() {
        return;
    }
    let mut map = records_map();
    let Some(old) = map.as_mut().and_then(|m| m.remove(&resource)) else {
        return;
    };
    if old.key != 0 {
        unsettled().remove(&(old.key, resource));
    }
}

/// Every record now (the audit's snapshot).
pub(crate) fn records_snapshot() -> Vec<(usize, Record)> {
    records_map().as_ref().map(|m| m.iter().map(|(&r, &rec)| (r, rec)).collect()).unwrap_or_default()
}

/// The frame of `resource`'s record now.
pub(crate) fn record_frame(resource: usize) -> Option<u64> {
    records_map().as_ref()?.get(&resource).map(|r| r.frame)
}

// Part E: settle checks.

/// Frames since a texture's detiling before it is checked.
pub(crate) const QUIET: u64 = 8;
/// Checks in a row that found a texture differing (or all zeros) before it
/// is left alone until its next detiling.
pub(crate) const STRIKES_MOST: u8 = 16;

/// The bytes the settle checks hash a frame, in MB
/// (`VICTORY_TEXTURE_EXACT_SETTLE_MB`, 8 unless said).
pub(crate) fn settle_mb() -> u64 {
    static MB: OnceLock<u64> = OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_EXACT_SETTLE_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&n| n > 0).unwrap_or(8)
    })
}

/// A settle check is on a wide thread.
static SETTLING: AtomicBool = AtomicBool::new(false);

/// The records quiet since `frame - QUIET` or before to check now, the
/// newest first, until `budget` bytes (at least one): (key, resource,
/// record frame). Pure, for testing.
pub(crate) fn pick_unsettled(set: &BTreeMap<(u64, usize), (u64, u64)>, frame: u64, budget: u64) -> Vec<(u64, usize, u64)> {
    let Some(last) = frame.checked_sub(QUIET) else {
        return Vec::new();
    };
    let mut picked = Vec::new();
    let mut bytes = 0u64;
    for (&(key, resource), &(rec_frame, len)) in set.range(..=(last, usize::MAX)).rev() {
        if bytes >= budget {
            break;
        }
        bytes += len.max(1);
        picked.push((key, resource, rec_frame));
    }
    picked
}

/// What a settle check found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Settled {
    Same,
    Differs,
    Zeros,
    Gone,
}

/// Once a frame (the texture pass): the quiet records to check go to a wide
/// thread, if the last check is done.
pub(crate) fn settle_tick(frame: u64) {
    if !part(Part::E) || SETTLING.load(Relaxed) {
        return;
    }
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    if FRAME.swap(frame, Relaxed) == frame {
        return;
    }
    let picked = {
        let mut set = unsettled();
        let picked = pick_unsettled(&set, frame, settle_mb() << 20);
        for &(key, resource, _) in &picked {
            set.remove(&(key, resource));
        }
        picked
    };
    if picked.is_empty() {
        return;
    }
    let jobs: Vec<(usize, Record)> = {
        let mut map = records_map();
        let Some(map) = map.as_mut() else {
            return;
        };
        picked
            .iter()
            .filter_map(|&(key, resource, rec_frame)| {
                let r = map.get_mut(&resource).filter(|r| r.frame == rec_frame && r.key == key)?;
                r.key = 0;
                Some((resource, *r))
            })
            .collect()
    };
    if jobs.is_empty() {
        return;
    }
    SETTLING.store(true, Relaxed);
    crate::shadow::wide_later(vec![Box::new(move || settle(jobs))]);
}

thread_local! {
    /// Room to read a texture's memory into, for the settle checks and the
    /// audit.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The full hash ([`h64`]) of title memory `[at, at + len)`, pages not
/// mapped read as zeros, and whether it is all zeros. Any thread.
pub(crate) fn memory_hash(at: u64, len: u64) -> (u64, bool) {
    READ.with(|read| {
        let mut read = read.borrow_mut();
        let len = len as usize;
        if read.len() < len {
            read.resize(len, 0);
        }
        crate::host::copy_mapped(read.as_mut_ptr(), at as *const u8, len, false);
        let bytes = &read[..len];
        let hash = h64(&blake3::hash(bytes));
        (hash, bytes.iter().all(|&b| b == 0))
    })
}

/// The settle checks of `jobs` (on a wide thread).
fn settle(jobs: Vec<(usize, Record)>) {
    let mut found: Vec<(usize, Record, Settled)> = Vec::with_capacity(jobs.len());
    for (resource, rec) in jobs {
        if crate::texture::mapped_bytes(rec.at as *const u8, rec.len as usize) == 0 {
            found.push((resource, rec, Settled::Gone));
            continue;
        }
        let (hash, zeros) = memory_hash(rec.at, rec.len);
        count(N::ECheck, 1);
        count(N::EBytes, rec.len);
        let verdict = if hash == rec.hash {
            Settled::Same
        } else if zeros {
            Settled::Zeros
        } else {
            Settled::Differs
        };
        found.push((resource, rec, verdict));
    }
    let now = crate::frame::current();
    let mut redue: Vec<(u64, u64)> = Vec::new();
    {
        let mut map = records_map();
        if let Some(map) = map.as_mut() {
            let mut set = unsettled();
            for (resource, rec, verdict) in found {
                let Some(r) = map.get_mut(&resource).filter(|r| r.frame == rec.frame && r.key == 0) else {
                    continue;
                };
                match verdict {
                    Settled::Gone => {}
                    Settled::Same => {
                        if r.open {
                            r.open = false;
                            count(N::EFixed, 1);
                        }
                        r.strikes = 0;
                    }
                    Settled::Zeros | Settled::Differs => {
                        if verdict == Settled::Zeros {
                            count(N::EZeros, 1);
                        } else {
                            count(N::EMismatch, 1);
                            r.open = true;
                            redue.push((r.at, r.len));
                        }
                        r.strikes = r.strikes.saturating_add(1);
                        if r.strikes > STRIKES_MOST {
                            count(N::EGivenUp, 1);
                        } else {
                            // Looked at again if nothing detiles it anew.
                            r.key = now.max(1);
                            set.insert((r.key, resource), (r.frame, r.len));
                        }
                    }
                }
            }
        }
    }
    for (at, len) in redue {
        crate::descriptor::host::remapped(at as usize, len as usize);
        crate::texture::restage_in(&[(at, at.saturating_add(len))]);
    }
    SETTLING.store(false, Relaxed);
}

// Part C: torn reads.

/// Re-dues of one texture for torn reads in a row before they are refused.
pub(crate) const TORN_MOST: u8 = 16;

/// Bumped when a DirectStorage write begins and when it ends.
static DS_SEQ: AtomicU64 = AtomicU64::new(0);

/// The writes that ended lately, each with the sequence its end took and
/// the title address ranges (start, end) it wrote. Pure, for testing.
#[derive(Default)]
pub(crate) struct WriteLog {
    ended: VecDeque<(u64, Vec<(u64, u64)>)>,
}

/// The most writes [`WriteLog`] keeps (a read lasts milliseconds).
const LOG_MOST: usize = 1024;

impl WriteLog {
    /// A write of `ranges` ended with sequence `seq`.
    pub(crate) fn ended(&mut self, seq: u64, ranges: &[(u64, u64)]) {
        self.ended.push_back((seq, ranges.to_vec()));
        while self.ended.len() > LOG_MOST {
            self.ended.pop_front();
        }
    }

    /// Whether a write that ended after sequence `seq0` wrote into
    /// `[start, end)`.
    pub(crate) fn overlapped_since(&self, seq0: u64, start: u64, end: u64) -> bool {
        for (seq, ranges) in self.ended.iter().rev() {
            if *seq <= seq0 {
                break;
            }
            if ranges.iter().any(|&(s, e)| s < end && start < e) {
                return true;
            }
        }
        false
    }
}

static LOG: Mutex<WriteLog> = Mutex::new(WriteLog { ended: VecDeque::new() });

/// A DirectStorage write begins (`crate::texstale::title_memory_write_pending`).
pub(crate) fn write_began() {
    DS_SEQ.fetch_add(1, SeqCst);
}

/// A DirectStorage write of `ranges` ends: logged before it leaves the
/// writes under way (the caller holds their lock), so that a read that no
/// longer finds it there finds it here.
pub(crate) fn write_ended(ranges: &[(u64, u64)]) {
    if !part(Part::C) {
        DS_SEQ.fetch_add(1, SeqCst);
        return;
    }
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    let seq = DS_SEQ.fetch_add(1, SeqCst) + 1;
    log.ended(seq, ranges);
}

/// Part C: before a refresh reads a texture's memory, the sequence to
/// give [`torn`] after it (none with the part off).
pub(crate) fn read_began() -> Option<u64> {
    part(Part::C).then(|| DS_SEQ.load(SeqCst))
}

/// Part C: whether a read of `[start, end)` that began at sequence `seq0`
/// may have taken part of a write: a DirectStorage write is under way there
/// now, or one there began or ended meanwhile; with part D also memory
/// there mapped and not written yet.
pub(crate) fn torn(seq0: u64, start: u64, end: u64) -> bool {
    if crate::texstale::write_pending_any(start, end) {
        return true;
    }
    if part(Part::D) && pages_held(start, end) {
        return true;
    }
    if DS_SEQ.load(SeqCst) == seq0 {
        return false;
    }
    LOG.lock().unwrap_or_else(|e| e.into_inner()).overlapped_since(seq0, start, end)
}

/// Torn reads in a row by texture address. Pure, for testing.
#[derive(Default)]
pub(crate) struct Strikes {
    by: HashMap<u64, u8>,
}

impl Strikes {
    /// One more torn read of the texture at `at`: whether it is made due
    /// again (false past `most` in a row).
    pub(crate) fn hit(&mut self, at: u64, most: u8) -> bool {
        let n = self.by.entry(at).or_insert(0);
        *n = n.saturating_add(1);
        *n <= most
    }

    /// A clean read of it.
    pub(crate) fn clear(&mut self, at: u64) {
        self.by.remove(&at);
    }

    pub(crate) fn len(&self) -> usize {
        self.by.len()
    }
}

static STRIKES: Mutex<Option<Strikes>> = Mutex::new(None);
static STRIKES_N: AtomicUsize = AtomicUsize::new(0);

/// Part C: a torn read of the texture at `at`: whether to make it due
/// again (counted either way).
pub(crate) fn torn_redue(at: u64) -> bool {
    count(N::CTorn, 1);
    let mut s = STRIKES.lock().unwrap_or_else(|e| e.into_inner());
    let s = s.get_or_insert_with(Strikes::default);
    let again = s.hit(at, TORN_MOST);
    STRIKES_N.store(s.len(), Relaxed);
    if !again {
        count(N::CCapped, 1);
    }
    again
}

/// Part C: a read of the texture at `at` that nothing tore.
pub(crate) fn clean_read(at: u64) {
    if STRIKES_N.load(Relaxed) == 0 {
        return;
    }
    let mut s = STRIKES.lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = s.as_mut() else {
        return;
    };
    s.clear(at);
    STRIKES_N.store(s.len(), Relaxed);
}

// Part D: memory mapped and not written yet.

/// How long after its map memory not written yet holds uploads, and after
/// its start a DirectStorage write under way does.
pub(crate) const HOLD_MS: u64 = 500;

/// Title memory mapped and not written since: byte ranges, apart, each
/// with the millisecond of its map (`crate::host::now_ms`). Pure, for
/// testing.
#[derive(Default)]
pub(crate) struct NotReady {
    set: BTreeMap<u64, (u64, u64)>,
}

impl NotReady {
    /// The starts of the ranges that overlap `[start, end)` (apart and in
    /// order, so their ends rise with their starts).
    fn overlapping(&self, start: u64, end: u64) -> Vec<u64> {
        let mut out = Vec::new();
        for (&s, &(e, _)) in self.set.range(..end).rev() {
            if e <= start {
                break;
            }
            out.push(s);
        }
        out
    }

    /// `[start, end)` is no longer what it was: the ranges are cut there.
    fn cut(&mut self, start: u64, end: u64) {
        for s in self.overlapping(start, end) {
            let Some((e, ms)) = self.set.remove(&s) else {
                continue;
            };
            if s < start {
                self.set.insert(s, (start, ms));
            }
            if e > end {
                self.set.insert(end, (e, ms));
            }
        }
    }

    /// `[start, end)` was mapped at `ms`.
    pub(crate) fn map(&mut self, start: u64, end: u64, ms: u64) {
        if end <= start {
            return;
        }
        self.cut(start, end);
        self.set.insert(start, (end, ms));
    }

    /// `[start, end)` was written.
    pub(crate) fn written(&mut self, start: u64, end: u64) {
        if end > start {
            self.cut(start, end);
        }
    }

    /// The parts of `[start, end)` mapped under [`HOLD_MS`] before `now`
    /// and not written since, and whether a part mapped longer ago lies
    /// there too (its hold is over).
    pub(crate) fn fresh(&self, start: u64, end: u64, now: u64) -> (Vec<(u64, u64)>, bool) {
        let (mut fresh, mut over) = (Vec::new(), false);
        for s in self.overlapping(start, end) {
            let (e, ms) = self.set[&s];
            if now.saturating_sub(ms) < HOLD_MS {
                fresh.push((s.max(start), e.min(end)));
            } else {
                over = true;
            }
        }
        (fresh, over)
    }

    /// Ranges past their hold go.
    pub(crate) fn purge(&mut self, now: u64) {
        self.set.retain(|_, &mut (_, ms)| now.saturating_sub(ms) < HOLD_MS);
    }

    pub(crate) fn len(&self) -> usize {
        self.set.len()
    }
}

static NOT_READY: Mutex<NotReady> = Mutex::new(NotReady { set: BTreeMap::new() });
static NOT_READY_N: AtomicUsize = AtomicUsize::new(0);

fn not_ready_set() -> std::sync::MutexGuard<'static, NotReady> {
    NOT_READY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Part D: the runtime mapped or unmapped title memory `[at, at + len)`
/// (`crate::watch::title_memory_remapped`, with its locks held: locks one
/// mutex and calls nothing). Unmapped memory is told apart at the asking.
pub(crate) fn mapped(at: usize, len: usize) {
    if !part(Part::D) {
        return;
    }
    let now = crate::host::now_ms();
    let mut set = not_ready_set();
    set.map(at as u64, at.saturating_add(len) as u64, now);
    NOT_READY_N.store(set.len(), Relaxed);
}

/// Part D: the bytes of a write the guards do not see are in title memory
/// `[at, at + len)` (`crate::watch::title_memory_written`).
pub(crate) fn written(at: usize, len: usize) {
    if !part(Part::D) || NOT_READY_N.load(Relaxed) == 0 {
        return;
    }
    let mut set = not_ready_set();
    set.written(at as u64, at.saturating_add(len) as u64);
    NOT_READY_N.store(set.len(), Relaxed);
}

/// Whether `[start, end)` holds memory mapped under [`HOLD_MS`] ago and not
/// written since (what is unmapped of it again is left out).
fn pages_held(start: u64, end: u64) -> bool {
    if NOT_READY_N.load(Relaxed) == 0 {
        return false;
    }
    let (fresh, over) = not_ready_set().fresh(start, end, crate::host::now_ms());
    let held = fresh
        .iter()
        .any(|&(s, e)| crate::texture::mapped_bytes(s as *const u8, (e - s) as usize) > 0);
    if over && !held {
        count(N::DReleased, 1);
    }
    held
}

/// Part D: whether a texture in `[start, end)` is held (keeps its previous
/// host copy): memory there was mapped under [`HOLD_MS`] ago and has not
/// been written since, or a DirectStorage write there began under
/// [`HOLD_MS`] ago and is still under way.
pub(crate) fn not_ready(start: u64, end: u64) -> bool {
    if crate::texstale::write_pending_since(start, end)
        .is_some_and(|since| since.elapsed() < std::time::Duration::from_millis(HOLD_MS))
    {
        return true;
    }
    pages_held(start, end)
}

// Once a frame, from the texture pass.

/// The texture pass's turn (`crate::draw` `heap_textures`): part D's ranges
/// past their hold go, part E's checks of the frame go out, and the audit
/// takes its snapshot or ends.
pub(crate) fn tick(heap: &crate::descriptor::Heap, host: &crate::descriptor::host::HostHeap, frame: u64) {
    if part(Part::D) && NOT_READY_N.load(Relaxed) != 0 {
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        if FRAME.swap(frame, Relaxed) != frame {
            let mut set = not_ready_set();
            set.purge(crate::host::now_ms());
            NOT_READY_N.store(set.len(), Relaxed);
        }
    }
    settle_tick(frame);
    audit::tick(heap, host, frame);
}

// The line.

/// The frame of the last line, for the bytes a frame.
static LINE_FRAME: AtomicU64 = AtomicU64::new(0);

/// Every five seconds with the frame line (`crate::texstale::lines`): what
/// the parts did since the start, if `VICTORY_TEXTURE_EXACT=1`.
pub(crate) fn lines() {
    if !on() {
        return;
    }
    let frame = crate::frame::current();
    let frames = frame.saturating_sub(LINE_FRAME.swap(frame, Relaxed)).max(1);
    let hashed = COUNTS[N::EBytes as usize].swap(0, Relaxed);
    tracing::info!(
        "texture exact: parts {} (VICTORY_TEXTURE_EXACT=1), since the start: A {} slots named by memory events through the \
         exact spans, {} of them missed by the old rule, {} named by the old rule left out (their viewed levels end before \
         the range), {} times textures staged for draws read again after an event; B {} prints of the bytes uploaded put \
         in checked entries ({} differed from the print taken when the upload was recorded), {} reads that took nothing, \
         {} posts lost; C {} torn reads caught (a DirectStorage write overlapped the read: due next frame), {} re-dues \
         refused after {TORN_MOST} in a row; D {} holds of the pass and {} of draws (memory mapped and not written yet, \
         or under a write), {} let go by the {HOLD_MS} ms bound; E {} settle checks, {} mismatches found, {} of them \
         fixed, {} given up after {STRIKES_MOST}, {} found all zeros; {:.1} MB a frame hashed since the last line",
        letters(parts()),
        counted(N::ANamed),
        counted(N::AMissed),
        counted(N::ALeft),
        counted(N::ARestaged),
        counted(N::BDrained),
        counted(N::BDiffered),
        counted(N::BFailed),
        counted(N::BLost),
        counted(N::CTorn),
        counted(N::CCapped),
        counted(N::DPass),
        counted(N::DDraw),
        counted(N::DReleased),
        counted(N::ECheck),
        counted(N::EMismatch),
        counted(N::EFixed),
        counted(N::EGivenUp),
        counted(N::EZeros),
        hashed as f64 / f64::from(1u32 << 20) / frames as f64,
    );
}

// Part A's viewed extent.

/// The bytes from a placed texture's start that hold the levels a view
/// starting at level `first` shows, from its console layout (plane 0):
/// `size` in all, `slices` of `slice_size`, the tail block's `alignment`,
/// its first tail level, and each level's offset in a slice. The console
/// lays a slice out with the tail block first and then the levels from the
/// smallest up, level 0 last, so the levels from `first` on end where level
/// `first - 1` starts. A layout that is not so (several slices, levels not
/// in that order, a level past the last) holds all of `size`. Pure, for
/// testing.
pub(crate) fn viewed_end(size: u64, slices: u32, slice_size: u64, alignment: u64, first_tail: u32, offsets: &[u64], first: u32) -> u64 {
    let first = first as usize;
    if first == 0 || slices != 1 || offsets.is_empty() || first >= offsets.len() {
        return size;
    }
    let tail = (first_tail as usize).min(offsets.len());
    // Smallest first: the levels before the tail at falling offsets.
    if tail == 0 || offsets[..tail].windows(2).any(|w| w[0] <= w[1]) {
        return size;
    }
    let end = if first >= tail { alignment.min(slice_size) } else { offsets[first - 1] };
    end.clamp(1, size)
}
