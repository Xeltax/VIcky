//! VIcky: streamed texture staleness (VICKY-RT.md, "Streamed texture
//! staleness").
//!
//! Textures picked by index were uploaded before DirectStorage had written
//! their bytes, and never again: 10,700-16,900 slots a run were "uploaded
//! before the write (left as they were)", and in about 70 % of the runs the
//! clothing of a close-up showed a block mosaic, the wrong colours or black
//! holes. Five switches answer the holes found, each off by default:
//!
//! - `VICTORY_HEAP_RECHECK_MIPS=1` (`crate::texture::recheck_mips`): the
//!   looks after an upload read the start of every mip;
//! - `VICTORY_NO_WAIT_FALLBACK=1` (`crate::texture::no_wait_fallback`): a
//!   refresh that did not wait reads around a missing page and makes the
//!   slot due again;
//! - `VICTORY_RECHECK_PARTLY=1` ([`recheck_partly`]): textures uploaded in
//!   part are asked their fingerprint too;
//! - `VICTORY_DS_HOLD=1` ([`ds_hold`]): a slot is not uploaded while a
//!   DirectStorage write is pending under its memory;
//! - `VICTORY_STALE_SLOTS=1` ([`stale_slots`]): a diagnostic that counts
//!   uploaded slots whose memory no longer matches what they took.
//!
//! (`VICTORY_BCPACK_FIXES=1` is the decoder's, in `victory-hle`; its counts
//! are on a line here too.) The counters and their lines ([`lines`], every
//! five seconds with the frame line) are kept here.
//!
//! Structural stage 1 (`VICTORY_TABLE_BOUND`) has draws make the texture
//! slots they read through tables, with their own staging buffers, out of
//! the queue those switches work on. `VICTORY_TABLE_BOUND_FRESH`
//! (`crate::descriptor::left::fresh`, on by default with it) keeps such slots
//! under the same rules: its counts ([`Fresh`]), the hold of a draw's upload
//! under a DirectStorage write ([`StageHold`], [`stage_held`]), and
//! `VICTORY_STALE_SLOTS`' look at the slots draws make, apart
//! ([`table_made`], [`table_verdict`]) are here too.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::descriptor::View;

fn on(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| v == "1")
}

// `VICTORY_HEAP_RECHECK_MIPS`: what the fingerprints cost.

/// Since the last line: fingerprints taken of textures picked by index
/// (`crate::texture::fingerprint`, with the switch or not), the
/// microseconds they took, and the mip ranges they read besides their
/// three windows.
pub(crate) static PRINTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A fingerprint was taken in `us` microseconds, reading `ranges` mip
/// ranges besides its windows.
pub(crate) fn printed(ranges: u64, us: u64) {
    PRINTS[0].fetch_add(1, Relaxed);
    PRINTS[1].fetch_add(us, Relaxed);
    PRINTS[2].fetch_add(ranges, Relaxed);
}

// `VICTORY_NO_WAIT_FALLBACK`.

/// Since the start, with `VICTORY_NO_WAIT_FALLBACK=1`, refreshes that did not
/// wait and found a page missing: those that took nothing (the staging buffer
/// had been filled before); those read around the missing pages instead;
/// those that found only zeros there; and the times the slots in that memory
/// were made due again (all of them).
pub(crate) static FALLBACK: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

// `VICTORY_RECHECK_PARTLY`.

/// VIcky (`VICTORY_RECHECK_PARTLY=1`; off by default, as before): whether a
/// placed texture uploaded in part (not all of its memory mapped) is looked
/// at with a fingerprint of its memory too.
///
/// Before, such a slot was uploaded again only when more of its memory was
/// mapped. Bytes that land in pages mapped already (a DirectStorage write
/// into a recycled block, which maps nothing) left the host copy with what
/// the pages held before, for as long as the texture stayed at a medium mip.
/// With the switch, the upload notes a fingerprint (every mip read, as with
/// `VICTORY_HEAP_RECHECK_MIPS`: the large mips of such a texture are the
/// ones missing, and the three windows alone see little else), and each
/// look at the slot asks it; changed, the slot is uploaded although no more
/// of it is mapped. At most [`PARTLY_PER_FRAME`] such looks a frame take a
/// fingerprint; the rest look at the mapping alone, as before.
pub(crate) fn recheck_partly() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = on("VICTORY_RECHECK_PARTLY");
        if on {
            tracing::info!(
                "textures picked by index uploaded in part are looked at with a fingerprint of every mip too, at most \
                 {PARTLY_PER_FRAME} a frame (VICTORY_RECHECK_PARTLY=1)"
            );
        }
        on
    })
}

/// Fingerprints [`recheck_partly`]'s looks take at most a frame: each reads
/// up to 48 KiB and 64 mip ranges.
pub(crate) const PARTLY_PER_FRAME: u64 = 256;

/// Since the start, with [`recheck_partly`]: uploads in part that noted a
/// fingerprint; looks that found it the same; looks that found it changed
/// (uploaded again); looks past [`PARTLY_PER_FRAME`] (the mapping alone).
pub(crate) static PARTLY: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Whether one more of [`recheck_partly`]'s looks may take a fingerprint in
/// `frame`.
pub(crate) fn partly_budget(frame: u64) -> bool {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static TAKEN: AtomicU64 = AtomicU64::new(0);
    if FRAME.swap(frame, Relaxed) != frame {
        TAKEN.store(0, Relaxed);
    }
    TAKEN.fetch_add(1, Relaxed) < PARTLY_PER_FRAME
}

// `VICTORY_DS_HOLD`.

/// VIcky (`VICTORY_DS_HOLD=1`; off by default, as before): whether a slot of
/// a placed texture is held back while a DirectStorage write is pending
/// under its memory.
///
/// The title maps pages and has them filled afterwards; the mapping makes
/// the slot due at the next pass, which uploaded what the pages held before
/// the write (the recycled block's last texture, or zeros), and the write
/// that landed then was only counted (`VICTORY_PAGE_WRITES`). With the
/// switch the runtime notes the title addresses of each request's
/// destination from the start of its work to its end
/// ([`title_memory_write_pending`]), and the pass holds a slot whose memory
/// overlaps one for a frame instead of uploading it. Bounded: at most
/// [`DS_HOLD_MOST`] holds for one write of a slot, and a request pending
/// for over [`DS_HOLD_MS`] holds nothing more; then the slot is uploaded as
/// before. Requests still waiting in a queue, before a worker takes them,
/// are not noted.
pub(crate) fn ds_hold() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = on("VICTORY_DS_HOLD");
        if on {
            tracing::info!(
                "textures picked by index are held while a DirectStorage write is under way into their memory, at most \
                 {DS_HOLD_MOST} times a write of the slot and {DS_HOLD_MS} ms a request (VICTORY_DS_HOLD=1)"
            );
        }
        on
    })
}

/// The most times one write of a slot is held ([`ds_hold`]).
pub(crate) const DS_HOLD_MOST: u8 = 16;

/// A request pending for longer holds nothing more ([`ds_hold`]).
pub(crate) const DS_HOLD_MS: u64 = 1000;

/// Since the start, with [`ds_hold`]: requests noted; looks held back;
/// slots uploaded anyway after [`DS_HOLD_MOST`] holds; requests that were
/// pending for over [`DS_HOLD_MS`].
pub(crate) static HOLD: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A request's destination, as title address ranges (start, end).
struct Pending {
    token: u64,
    since: Instant,
    ranges: Vec<(u64, u64)>,
}

static PENDING: Mutex<Vec<Pending>> = Mutex::new(Vec::new());
/// How many [`PENDING`] holds (the pass asks the lock only if any).
static PENDING_N: AtomicUsize = AtomicUsize::new(0);
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

/// A DirectStorage write noted as pending until dropped
/// ([`title_memory_write_pending`]).
pub struct WritePending(u64);

impl Drop for WritePending {
    fn drop(&mut self) {
        if self.0 == 0 {
            return;
        }
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = pending.iter().position(|p| p.token == self.0) {
            let p = pending.swap_remove(i);
            if p.since.elapsed() > Duration::from_millis(DS_HOLD_MS) {
                HOLD[3].fetch_add(1, Relaxed);
            }
        }
        PENDING_N.store(pending.len(), Relaxed);
    }
}

/// VIcky (`VICTORY_DS_HOLD=1`): DirectStorage is about to write the title
/// memory `ranges` gives (address, bytes): the texture pass holds the slots
/// of placed textures there until what this returns is dropped, which is to
/// be once the bytes are there and the GPU side has been told
/// (`title_memory_written`). With the switch off nothing is noted and
/// `ranges` is not called. Called from the runtime's workers; locks one
/// mutex and calls nothing.
pub fn title_memory_write_pending(ranges: impl FnOnce() -> Vec<(usize, usize)>) -> WritePending {
    if !ds_hold() {
        return WritePending(0);
    }
    let ranges: Vec<(u64, u64)> = ranges()
        .into_iter()
        .filter(|&(_, n)| n > 0)
        .map(|(at, n)| (at as u64, at.saturating_add(n) as u64))
        .collect();
    if ranges.is_empty() {
        return WritePending(0);
    }
    let token = NEXT_TOKEN.fetch_add(1, Relaxed);
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    pending.push(Pending { token, since: Instant::now(), ranges });
    PENDING_N.store(pending.len(), Relaxed);
    HOLD[0].fetch_add(1, Relaxed);
    WritePending(token)
}

/// Whether a request noted pending for under [`DS_HOLD_MS`] at `now` writes
/// into `[start, end)`. Pure, for testing.
fn overlaps(pending: &[Pending], start: u64, end: u64, now: Instant) -> bool {
    let most = Duration::from_millis(DS_HOLD_MS);
    pending.iter().any(|p| {
        now.saturating_duration_since(p.since) <= most && p.ranges.iter().any(|&(s, e)| s < end && start < e)
    })
}

/// Whether a DirectStorage write is pending under title memory `[start,
/// end)` ([`ds_hold`]).
pub(crate) fn write_pending(start: u64, end: u64) -> bool {
    if PENDING_N.load(Relaxed) == 0 {
        return false;
    }
    let pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    overlaps(&pending, start, end, Instant::now())
}

/// Requests pending now (for the line).
fn pending_now() -> usize {
    PENDING_N.load(Relaxed)
}

/// Tests: notes `ranges` (start, end) as a pending DirectStorage write
/// whatever the switch says, until what this returns is dropped.
#[cfg(test)]
pub(crate) fn test_pending(ranges: Vec<(u64, u64)>) -> WritePending {
    let token = NEXT_TOKEN.fetch_add(1, Relaxed);
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    pending.push(Pending { token, since: Instant::now(), ranges });
    PENDING_N.store(pending.len(), Relaxed);
    WritePending(token)
}

/// VIcky (`VICTORY_TABLE_BOUND_FRESH` with [`ds_hold`]): the frames a
/// texture the draws keep staged waited for a DirectStorage write into its
/// memory (`crate::texture::upload` and the check at submission): once a
/// frame counts, at most [`DS_HOLD_MOST`] for one write, as a slot of the
/// texture pass is held; a use with no write pending starts the count anew.
/// Shared by the recording threads and the submitting thread: a race moves
/// the count by one, the bound stays.
#[derive(Default)]
pub(crate) struct StageHold {
    frames: std::sync::atomic::AtomicU32,
    /// The frame of the last hold, plus one (0: none).
    last: AtomicU64,
}

impl StageHold {
    /// Whether a use in `frame` waits, a write being `pending` or not
    /// (`most`: [`DS_HOLD_MOST`]).
    pub(crate) fn hold(&self, pending: bool, frame: u64, most: u8) -> bool {
        if !pending {
            if self.frames.load(Relaxed) != 0 {
                self.frames.store(0, Relaxed);
                self.last.store(0, Relaxed);
            }
            return false;
        }
        let frames = self.frames.load(Relaxed);
        let now = frame.wrapping_add(1);
        if frames > 0 && self.last.load(Relaxed) == now {
            // Held in this frame already, by another list or the check.
            return true;
        }
        if frames >= u32::from(most) {
            return false;
        }
        self.frames.store(frames + 1, Relaxed);
        self.last.store(now, Relaxed);
        true
    }
}

/// Whether a draw's use in `frame` of a texture staged for draws, whose
/// memory is `len` bytes at `at`, waits for a DirectStorage write under way
/// there (`hold`: its [`StageHold`]; `on`: `VICTORY_TABLE_BOUND_FRESH` acts
/// now and [`ds_hold`] is set). Counted ([`Fresh::Held`], [`Fresh::HeldOut`]).
pub(crate) fn stage_held(hold: &StageHold, on: bool, at: u64, len: u64, frame: u64) -> bool {
    if !on {
        return false;
    }
    let pending = write_pending(at, at.saturating_add(len));
    let held = hold.hold(pending, frame, DS_HOLD_MOST);
    if pending {
        fresh_count(if held { Fresh::Held } else { Fresh::HeldOut }, 1);
    }
    held
}

// `VICTORY_TABLE_BOUND_FRESH` (`crate::descriptor::left::fresh`).

/// What `VICTORY_TABLE_BOUND_FRESH` counts ([`fresh_count`], since the start).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Fresh {
    /// Slots draws read through tables lately that a pass over memory found,
    /// kept and made due as any slot (stage 1 left them to their draws):
    /// memory mapped, unmapped or about to be written (also a refresh that
    /// missed a page, `VICTORY_NO_WAIT_FALLBACK`); written behind the
    /// guards; a level waiting for the card's memory.
    Remapped,
    PageWritten,
    Waiting,
    /// Textures staged for draws read again whole after a write behind the
    /// guards (`crate::texture::restage_in`).
    Restaged,
    /// With [`ds_hold`]: a draw's upload or check held while a DirectStorage
    /// write was under way into its texture; one that went on after
    /// [`DS_HOLD_MOST`] frames.
    Held,
    HeldOut,
}

/// [`Fresh`]'s counts.
pub(crate) static FRESH: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Counts `n` of `what`.
pub(crate) fn fresh_count(what: Fresh, n: u64) {
    if n > 0 {
        FRESH[what as usize].fetch_add(n, Relaxed);
    }
}

// `VICTORY_STALE_SLOTS`.

/// VIcky (`VICTORY_STALE_SLOTS`; unset or `0`: off): a diagnostic that says
/// in one run how many uploaded textures picked by index no longer match
/// their memory. `1`: 64 slots looked at a frame; a number above 1: that
/// many (at most 4096).
///
/// Each upload of a slot notes a sampled fingerprint of its memory: 256
/// bytes at the start of every mip of every slice (at most 64 mips over all
/// slices) and, for mips of over 64 KiB, at their middle and their end
/// ([`sample_windows`]). Once a frame some slots are looked at in turn: the
/// fingerprint now against the one noted. A slot that differs and is out of
/// the queue keeps bytes no pass will replace; one that differs in the queue
/// is waiting for a pass. Counted per line, with the slots whose upload had
/// a DirectStorage write after it (`HostHeap::slots_written_under`); the
/// slot listing (`VICTORY_LIST_SLOTS`) says it per slot (`stale=[...]`).
/// The noted fingerprint is taken when the upload is recorded; the copy
/// takes the memory at its submission, a moment later.
pub(crate) fn stale_slots() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let n = match std::env::var("VICTORY_STALE_SLOTS").as_deref().map(str::trim) {
            Ok("1") => 64,
            Ok(v) => v.parse::<usize>().map_or(0, |n| n.min(4096)),
            Err(_) => 0,
        };
        if n > 0 {
            tracing::info!(
                "uploaded textures picked by index are looked at against their memory, {n} slots a frame, a sampled \
                 fingerprint of every mip (VICTORY_STALE_SLOTS)"
            );
        }
        n
    })
}

/// Bytes of each window of [`sample_windows`].
const STALE_WINDOW: u64 = 256;
/// A mip of more bytes is also sampled at its middle and its end.
const STALE_LONG: u64 = 64 << 10;
/// The most slots watched (a slot uploaded past it is not).
const WATCH_MOST: usize = 1 << 17;

/// The windows (offset, bytes) of a texture's memory the stale look reads,
/// from its mips' ranges (`Layout::mip_ranges`): [`STALE_WINDOW`] bytes at
/// the start of each, and at the middle (on a 4 KiB line) and the end of
/// each longer than [`STALE_LONG`]. In order, joined where they touch.
/// Pure, for testing.
pub(crate) fn sample_windows(mips: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut windows: Vec<(u64, u64)> = Vec::with_capacity(mips.len() * 3);
    for &(at, len) in mips {
        let mut put = |from: u64| {
            let n = STALE_WINDOW.min(at + len - from);
            if n > 0 {
                windows.push((from, n));
            }
        };
        put(at);
        if len > STALE_LONG {
            put(at + ((len / 2) & !0xfff));
            put(at + len - STALE_WINDOW);
        }
    }
    windows.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(windows.len());
    for (at, n) in windows {
        match out.last_mut() {
            Some(last) if at <= last.0 + last.1 => last.1 = last.1.max(at + n - last.0),
            _ => out.push((at, n)),
        }
    }
    out
}

/// The sampled fingerprint of title memory at `at` (`len` bytes) over
/// `windows` (offsets from `at`; [`sample_windows`]), never 0. Read through
/// the kernel: pages not there count as zeros.
fn sample_print(at: u64, len: u64, windows: &[(u64, u64)]) -> u64 {
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut hasher = blake3::Hasher::new();
    READ.with(|read| {
        let mut read = read.borrow_mut();
        for &(from, n) in windows {
            if from >= len {
                continue;
            }
            let n = n.min(len - from) as usize;
            if read.len() < n {
                read.resize(n, 0);
            }
            read[..n].fill(0);
            crate::host::copy_mapped(read.as_mut_ptr(), (at + from) as *const u8, n, false);
            hasher.update(&read[..n]);
        }
    });
    let mut first = [0u8; 8];
    first.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    u64::from_le_bytes(first) | 1
}

/// [`sample_print`] of `bytes` in hand (a texture's memory as a detiling
/// took it, `crate::texture` `Staging::detile`): the same value as
/// `sample_print` gives for memory that holds those bytes.
pub(crate) fn sample_print_of(bytes: &[u8], windows: &[(u64, u64)]) -> u64 {
    let len = bytes.len() as u64;
    let mut hasher = blake3::Hasher::new();
    for &(from, n) in windows {
        if from >= len {
            continue;
        }
        let n = n.min(len - from) as usize;
        hasher.update(&bytes[from as usize..from as usize + n]);
    }
    let mut first = [0u8; 8];
    first.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    u64::from_le_bytes(first) | 1
}

/// What [`stale_slots`] noted of an uploaded slot.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
struct Watched {
    at: u64,
    len: u64,
    windows: Arc<[(u64, u64)]>,
    print: u64,
    /// The frame of the upload.
    uploaded: u64,
    /// A DirectStorage write landed under it after the upload.
    written: bool,
    /// Its last look found it differing and out of the queue.
    stale_out: bool,
    /// The frame of its last look (0: none yet).
    looked: u64,
}

struct Watch {
    slots: BTreeMap<u32, Watched>,
    /// The slot the next look starts from.
    cursor: u32,
    /// Slots whose last look found them differing and out of the queue.
    stale_out: u64,
}

static WATCH: Mutex<Watch> = Mutex::new(Watch { slots: BTreeMap::new(), cursor: 0, stale_out: 0 });

/// Since the last line, with [`stale_slots`]: slots looked at; of them,
/// differing from what their upload took; of those, out of the queue; in
/// it; with a DirectStorage write after their upload; slots let go (memory
/// unmapped, or the slot now views another texture).
pub(crate) static STALE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

fn watch() -> std::sync::MutexGuard<'static, Watch> {
    WATCH.lock().unwrap_or_else(|e| e.into_inner())
}

/// Title slot `index` was uploaded from title memory `at` (`len` bytes)
/// whose mips lie at `mips` (offsets, bytes; `Layout::mip_ranges`) in
/// `frame`: its sampled fingerprint is noted ([`stale_slots`]).
pub(crate) fn uploaded(index: u32, at: u64, len: u64, mips: &[(u64, u64)], frame: u64) {
    if stale_slots() == 0 || mips.is_empty() {
        return;
    }
    let windows: Arc<[(u64, u64)]> = sample_windows(mips).into();
    let print = sample_print(at, len, &windows);
    let mut w = watch();
    if w.slots.len() >= WATCH_MOST && !w.slots.contains_key(&index) {
        return;
    }
    let entry = Watched { at, len, windows, print, uploaded: frame, written: false, stale_out: false, looked: 0 };
    if let Some(old) = w.slots.insert(index, entry)
        && old.stale_out
    {
        w.stale_out -= 1;
    }
}

/// Title slot `index` is no longer one the pass uploads (a draw reads it
/// through a table): not watched.
pub(crate) fn forget(index: u32) {
    if stale_slots() == 0 {
        return;
    }
    let mut w = watch();
    if let Some(old) = w.slots.remove(&index)
        && old.stale_out
    {
        w.stale_out -= 1;
    }
}

/// A DirectStorage write landed under the textures of title slots `slots`
/// after their upload.
pub(crate) fn written_under(slots: &[u32]) {
    if stale_slots() == 0 || slots.is_empty() {
        return;
    }
    let mut w = watch();
    for slot in slots {
        if let Some(entry) = w.slots.get_mut(slot) {
            entry.written = true;
        }
    }
}

// `VICTORY_STALE_SLOTS` for the slots draws make (`VICTORY_TABLE_BOUND`).

/// What [`stale_slots`] knows of a texture slot a draw made through a table
/// ([`table_made`]): its host copy is the draws' staging buffer's, read
/// again at every submission that samples it, not an upload of the pass.
struct TableWatched {
    at: u64,
    len: u64,
    /// The title texture (`crate::object::This`) the slot's view is of.
    resource: usize,
    windows: Arc<[(u64, u64)]>,
    prints: TablePrints,
    /// Its last look found it stale ([`TableVerdict::Out`] or
    /// [`TableVerdict::Lasting`]).
    stale: bool,
}

/// What the looks at a table-made slot keep ([`table_verdict`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TablePrints {
    /// The sampled fingerprint of what its host copy holds, as last known
    /// from its staging buffer (0: never known).
    host: u64,
    /// That fingerprint when a look first found the memory differing from
    /// it, and the frame of that look ((0, 0): not differing).
    differ: (u64, u64),
}

/// A look's verdict on a table-made slot ([`table_verdict`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TableVerdict {
    /// Its memory is what its host copy took.
    Same,
    /// Nothing known of what its host copy holds yet.
    Unknown,
    /// Differs; its staging buffer's next check, or the pass (it is in the
    /// queue), takes the change.
    Waiting,
    /// Differs from the same host copy for [`TABLE_LASTING`] frames and
    /// more although its staging buffer is checked at every submission
    /// that samples it: those checks miss the change.
    Lasting,
    /// Differs, no staging buffer and not in the queue: nothing uploads it
    /// again.
    Out,
}

/// Frames a table-made slot may differ from its staged host copy before it
/// counts as stale ([`TableVerdict::Lasting`]): past the check's hash turn
/// (8 frames) and a partly mapped texture's read (every 30 frames), with
/// some slack, a change the checks would take is taken.
const TABLE_LASTING: u64 = 40;

/// The verdict of a look at a table-made slot whose memory's sampled
/// fingerprint is `now`, whose staging buffer's last detiling took
/// `staged` (none: no staging buffer, or nothing detiled with the
/// diagnostic on), in `frame`; `queued`: whether the slot is in the queue of
/// textures picked by index. `prints` is what the looks keep. Pure, for
/// testing.
fn table_verdict(prints: &mut TablePrints, now: u64, staged: Option<u64>, frame: u64, queued: bool) -> TableVerdict {
    let host = match staged {
        Some(print) => {
            prints.host = print;
            print
        }
        None => prints.host,
    };
    if host == 0 {
        return TableVerdict::Unknown;
    }
    if now == host {
        prints.differ = (0, 0);
        return TableVerdict::Same;
    }
    if staged.is_none() {
        return if queued { TableVerdict::Waiting } else { TableVerdict::Out };
    }
    if prints.differ.0 != host {
        prints.differ = (host, frame);
        return TableVerdict::Waiting;
    }
    if frame.saturating_sub(prints.differ.1) >= TABLE_LASTING {
        TableVerdict::Lasting
    } else {
        TableVerdict::Waiting
    }
}

struct TableWatch {
    slots: BTreeMap<u32, TableWatched>,
    cursor: u32,
    /// Slots whose last look found them stale.
    stale: u64,
}

static TABLE_WATCH: Mutex<TableWatch> = Mutex::new(TableWatch { slots: BTreeMap::new(), cursor: 0, stale: 0 });

fn table_watch() -> std::sync::MutexGuard<'static, TableWatch> {
    TABLE_WATCH.lock().unwrap_or_else(|e| e.into_inner())
}

/// Since the last line, with [`stale_slots`], of the table-made slots:
/// looked at; differing; of those with no staging buffer and out of the
/// queue; still differing under their staging buffer after
/// [`TABLE_LASTING`] frames; waiting for a check or a pass; let go.
pub(crate) static TABLE_STALE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// A draw made title slot `index` in place from `view` (a texture view it
/// reads through a table made afresh, `crate::draw` `State::tables`, ON
/// phases): watched apart from the pass's slots ([`stale_slots`]), once
/// the draws stage its texture. Nothing with the diagnostic off.
pub(crate) fn table_made(index: u32, view: &View) {
    if stale_slots() == 0 {
        return;
    }
    let View::Placed(false, resource, desc) = view else {
        return;
    };
    let at = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let (Some(len), Some(texture)) = (crate::resource::placed_bytes(resource), crate::resource::placed_known(at, resource))
    else {
        return;
    };
    {
        let w = table_watch();
        if let Some(e) = w.slots.get(&index) {
            if (e.at, e.len, e.resource) == (at, len, texture as usize) {
                return;
            }
        } else if w.slots.len() >= WATCH_MOST {
            return;
        }
    }
    // (A texture the draws do not stage is not theirs to keep: the pass's.)
    let Some(m) = crate::texture::staged_memory(texture, false, true).filter(|m| !m.spans.is_empty()) else {
        return;
    };
    let windows: Arc<[(u64, u64)]> = sample_windows(&m.spans).into();
    let entry = TableWatched { at, len, resource: texture as usize, windows, prints: TablePrints::default(), stale: false };
    let mut w = table_watch();
    if let Some(old) = w.slots.insert(index, entry)
        && old.stale
    {
        w.stale -= 1;
    }
}

/// The off phase of `VICTORY_TABLE_BOUND=alt` begins: no slot is made by a
/// draw, and the ones watched as such are the pass's again.
pub(crate) fn table_watch_clear() {
    if stale_slots() == 0 {
        return;
    }
    let mut w = table_watch();
    w.slots.clear();
    w.stale = 0;
}

/// Looks at up to `n` table-made slots, in turn ([`table_verdict`]).
fn table_sweep(heap: &crate::descriptor::Heap, host: &crate::descriptor::host::HostHeap, frame: u64, n: usize) {
    let batch: Vec<(u32, u64, u64, usize, Arc<[(u64, u64)]>)> = {
        let mut w = table_watch();
        if w.slots.is_empty() {
            return;
        }
        let cursor = w.cursor;
        let batch: Vec<_> = w
            .slots
            .range(cursor..)
            .chain(w.slots.range(..cursor))
            .take(n)
            .map(|(&index, e)| (index, e.at, e.len, e.resource, e.windows.clone()))
            .collect();
        w.cursor = batch.last().map_or(0, |b| b.0.wrapping_add(1));
        batch
    };
    // Per slot: (memory now, staged print, queued), or let go.
    let mut looked: Vec<(u32, u64, usize, Option<(u64, Option<u64>, bool)>)> = Vec::with_capacity(batch.len());
    for (index, at, len, resource, windows) in batch {
        let same_view = match heap.handle(index).and_then(crate::descriptor::view) {
            Some(View::Placed(false, r, desc)) => {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                address == at && crate::resource::placed_bytes(&r) == Some(len)
            }
            _ => false,
        };
        if !same_view || crate::texture::mapped_bytes(at as *const u8, len as usize) == 0 {
            looked.push((index, at, resource, None));
            continue;
        }
        let now = sample_print(at, len, &windows);
        let staged = crate::texture::staged_sample(resource as crate::object::This);
        let queued = staged.is_none() && host.texture_slot_queued(index);
        looked.push((index, at, resource, Some((now, staged, queued))));
    }
    let mut guard = table_watch();
    let w = &mut *guard;
    for (index, at, resource, look) in looked {
        // Only the entry that was looked at (not one made meanwhile).
        let Some(entry) = w.slots.get_mut(&index).filter(|e| (e.at, e.resource) == (at, resource)) else {
            continue;
        };
        let was = entry.stale;
        let Some((now, staged, queued)) = look else {
            TABLE_STALE[5].fetch_add(1, Relaxed);
            w.slots.remove(&index);
            if was {
                w.stale -= 1;
            }
            continue;
        };
        TABLE_STALE[0].fetch_add(1, Relaxed);
        let verdict = table_verdict(&mut entry.prints, now, staged, frame, queued);
        let stale = match verdict {
            TableVerdict::Same | TableVerdict::Unknown => false,
            TableVerdict::Waiting => {
                TABLE_STALE[4].fetch_add(1, Relaxed);
                false
            }
            TableVerdict::Lasting => {
                TABLE_STALE[3].fetch_add(1, Relaxed);
                true
            }
            TableVerdict::Out => {
                TABLE_STALE[2].fetch_add(1, Relaxed);
                true
            }
        };
        if !matches!(verdict, TableVerdict::Same | TableVerdict::Unknown) {
            TABLE_STALE[1].fetch_add(1, Relaxed);
        }
        if stale != was {
            entry.stale = stale;
            if stale {
                w.stale += 1;
            } else {
                w.stale -= 1;
            }
        }
    }
}

/// Looks at up to [`stale_slots`] watched slots, in turn, once a frame:
/// their sampled fingerprint now against the one their upload noted; and
/// as many of the slots draws make ([`table_made`]), apart.
pub(crate) fn sweep(heap: &crate::descriptor::Heap, host: &crate::descriptor::host::HostHeap, frame: u64) {
    let n = stale_slots();
    if n == 0 {
        return;
    }
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    if FRAME.swap(frame, Relaxed) == frame {
        return;
    }
    table_sweep(heap, host, frame, n);
    // Taken under the lock, looked at without it (the looks read memory
    // and ask the queue's lock).
    let batch: Vec<(u32, u64, u64, Arc<[(u64, u64)]>, u64, bool)> = {
        let mut w = watch();
        let cursor = w.cursor;
        let batch: Vec<_> = w
            .slots
            .range(cursor..)
            .chain(w.slots.range(..cursor))
            .take(n)
            .map(|(&index, e)| (index, e.at, e.len, e.windows.clone(), e.print, e.written))
            .collect();
        w.cursor = batch.last().map_or(0, |b| b.0.wrapping_add(1));
        batch
    };
    // Per slot: let go, or (its print, whether it differs and is out of
    // the queue).
    let mut looked: Vec<(u32, u64, Option<bool>)> = Vec::with_capacity(batch.len());
    for (index, at, len, windows, print, written) in batch {
        let same_view = match heap.handle(index).and_then(crate::descriptor::view) {
            Some(View::Placed(false, resource, desc)) => {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                address == at && crate::resource::placed_bytes(&resource) == Some(len)
            }
            _ => false,
        };
        if !same_view || crate::texture::mapped_bytes(at as *const u8, len as usize) == 0 {
            STALE[5].fetch_add(1, Relaxed);
            looked.push((index, print, None));
            continue;
        }
        STALE[0].fetch_add(1, Relaxed);
        let stale = sample_print(at, len, &windows) != print;
        let mut out = false;
        if stale {
            STALE[1].fetch_add(1, Relaxed);
            out = !host.texture_slot_queued(index);
            STALE[if out { 2 } else { 3 }].fetch_add(1, Relaxed);
            if written {
                STALE[4].fetch_add(1, Relaxed);
            }
        }
        looked.push((index, print, Some(out)));
    }
    let mut guard = watch();
    let w = &mut *guard;
    for (index, print, verdict) in looked {
        // Only the upload that was looked at (not one made meanwhile).
        let Some(entry) = w.slots.get_mut(&index).filter(|e| e.print == print) else {
            continue;
        };
        let was = entry.stale_out;
        match verdict {
            None => {
                w.slots.remove(&index);
                if was {
                    w.stale_out -= 1;
                }
            }
            Some(out) => {
                entry.stale_out = out;
                entry.looked = frame;
                match (was, out) {
                    (false, true) => w.stale_out += 1,
                    (true, false) => w.stale_out -= 1,
                    _ => {}
                }
            }
        }
    }
}

/// For the slot listing (`VICTORY_LIST_SLOTS`): title slot `index`'s last
/// upload against its memory now, in a few words.
#[cfg(feature = "diagnostics")]
pub(crate) fn describe(index: u32) -> String {
    if stale_slots() == 0 {
        return "off (VICTORY_STALE_SLOTS)".into();
    }
    let (at, len, windows, print, uploaded, written, looked) = {
        let w = watch();
        let Some(e) = w.slots.get(&index) else {
            return "not watched".into();
        };
        (e.at, e.len, e.windows.clone(), e.print, e.uploaded, e.written, e.looked)
    };
    let stale = sample_print(at, len, &windows) != print;
    format!(
        "stale={} uploaded@{uploaded} ds_write_after_upload={} looked@{looked} windows={}",
        if stale { "yes" } else { "no" },
        if written { "yes" } else { "no" },
        windows.len()
    )
}

// The lines.

/// The lines of the switches that are on (with the frame line, every five
/// seconds); none with all of them off.
pub(crate) fn lines() {
    if crate::texture::recheck_mips() {
        let [n, us, ranges] = [0, 1, 2].map(|i| PRINTS[i].swap(0, Relaxed));
        tracing::info!(
            "textures picked by index, looked at again after their upload with every mip read (VICTORY_HEAP_RECHECK_MIPS=1): \
             since the start {} found as they were, {} changed and uploaded again; since the last line {n} fingerprints \
             reading {ranges} mip ranges besides their windows ({:.1} a fingerprint), {:.1} ms in all",
            crate::draw::HEAP_TEXTURES_RECHECKED[0].load(Relaxed),
            crate::draw::HEAP_TEXTURES_RECHECKED[1].load(Relaxed),
            ranges as f64 / n.max(1) as f64,
            us as f64 / 1e3
        );
    }
    if crate::texture::no_wait_fallback() {
        let [failed, fell, zero, due] = [0, 1, 2, 3].map(|i| FALLBACK[i].load(Relaxed));
        tracing::info!(
            "texture refreshes that did not wait and found a page missing (VICTORY_NO_WAIT_FALLBACK=1), since the start: \
             {fell} read around the missing pages (never filled before), {zero} found only zeros there, {failed} took \
             nothing (filled before); {due} times the slots in that memory were made due again"
        );
    }
    if recheck_partly() {
        let [given, same, changed, over] = [0, 1, 2, 3].map(|i| PARTLY[i].load(Relaxed));
        tracing::info!(
            "textures picked by index uploaded in part, looked at with a fingerprint (VICTORY_RECHECK_PARTLY=1), since the \
             start: {given} uploads noted one, {same} looks found them as they were, {changed} found them changed and \
             uploaded them again, {over} looks past {PARTLY_PER_FRAME} a frame looked at the mapping alone"
        );
    }
    if ds_hold() {
        let [writes, holds, released, outlasted] = [0, 1, 2, 3].map(|i| HOLD[i].load(Relaxed));
        tracing::info!(
            "textures picked by index held while a DirectStorage write is under way into their memory (VICTORY_DS_HOLD=1), \
             since the start: {writes} requests noted, {holds} looks held back, {released} slots uploaded anyway after \
             {DS_HOLD_MOST} holds, {outlasted} requests pending for over {DS_HOLD_MS} ms; {} pending now",
            pending_now()
        );
    }
    if stale_slots() > 0 {
        let [looked, stale, out, queued, written, gone] = [0, 1, 2, 3, 4, 5].map(|i| STALE[i].swap(0, Relaxed));
        let (watched, stale_out) = {
            let w = watch();
            (w.slots.len(), w.stale_out)
        };
        tracing::info!(
            "textures picked by index whose upload no longer matches their memory (VICTORY_STALE_SLOTS, a sampled \
             fingerprint of every mip), since the last line: {looked} slots looked at, {stale} differ ({out} of them out \
             of the queue: nothing uploads them again; {queued} waiting in it; {written} had a DirectStorage write after \
             their upload), {gone} let go (unmapped, or another texture there); now {watched} slots watched, {stale_out} \
             of them differing and out of the queue at their last look"
        );
        if crate::descriptor::left::set() {
            let [looked, differ, out, lasting, waiting, gone] = [0, 1, 2, 3, 4, 5].map(|i| TABLE_STALE[i].swap(0, Relaxed));
            let (watched, stale) = {
                let w = table_watch();
                (w.slots.len(), w.stale)
            };
            tracing::info!(
                "texture slots draws make through tables whose host copy no longer matches their memory (VICTORY_STALE_SLOTS, \
                 apart from the slots above; VICTORY_TABLE_BOUND {}, FRESH {}), since the last line: {looked} looked at, \
                 {differ} differ ({out} of them with no staging buffer and out of the queue: nothing uploads them again; \
                 {lasting} still differing {TABLE_LASTING} frames on under their staging buffer: its checks miss the change; \
                 {waiting} waiting for their staging buffer's next check or a pass), {gone} let go (unmapped, or another \
                 texture there); now {watched} watched, {stale} of them stale at their last look",
                crate::descriptor::left::TABLE_BOUND.said(),
                if crate::descriptor::left::fresh() { "on" } else { "off" },
            );
        }
    }
    if crate::descriptor::left::fresh() {
        let [remapped, written, waiting, restaged, held, held_out] = [0, 1, 2, 3, 4, 5].map(|i| FRESH[i].load(Relaxed));
        tracing::info!(
            "texture slots draws make kept fresh (VICTORY_TABLE_BOUND_FRESH, ON phases), since the start: slots read through \
             tables lately found by the passes over memory and made due as any slot (stage 1 left them to their draws): \
             {remapped} remapped or about to be written (or a refresh that missed a page), {written} written behind the \
             guards, {waiting} waiting for a level; {restaged} textures the draws keep staged read again whole after a write \
             behind the guards; {held} draw uploads or checks held while a DirectStorage write was under way into their \
             texture (VICTORY_DS_HOLD=1), {held_out} that went on after {DS_HOLD_MOST} frames of one write"
        );
    }
    let [streams, kept, carried, first] =
        [0, 1, 2, 3].map(|i| victory_formats::bcpack::FIX_COUNTS[i].load(Relaxed));
    if streams > 0 {
        tracing::info!(
            "BCPack decoder fixes (VICTORY_BCPACK_FIXES=1), since the start: {streams} streams decoded with them; {kept} \
             finished lanes left the next bytes to their peers (Marvisak PR #2648), {carried} refills kept across a \
             fragment's end (PR #2649), {first} BC7 blocks built on the first dictionary entry (cd4402f405)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `VICTORY_DS_HOLD`: a pending request holds what overlaps any of its
    /// ranges, not what only touches them, and holds nothing once it has
    /// been pending for longer than the bound.
    #[test]
    fn a_pending_write_holds_what_it_overlaps_for_a_bounded_time() {
        let now = Instant::now();
        let pending = [
            Pending { token: 1, since: now, ranges: vec![(0x1_0000, 0x2_0000), (0x5_0000, 0x6_0000)] },
            Pending { token: 2, since: now, ranges: vec![(0x9_0000, 0xa_0000)] },
        ];
        assert!(overlaps(&pending, 0x1_8000, 0x1_9000, now));
        assert!(overlaps(&pending, 0x4_0000, 0x5_0001, now));
        assert!(overlaps(&pending, 0x0, 0x10_0000, now));
        assert!(!overlaps(&pending, 0x2_0000, 0x5_0000, now), "touching only");
        assert!(!overlaps(&pending, 0xa_0000, 0xb_0000, now));
        assert!(!overlaps(&[], 0, u64::MAX, now));
        let later = now + Duration::from_millis(DS_HOLD_MS + 1);
        assert!(!overlaps(&pending, 0x1_8000, 0x1_9000, later), "pending too long");
        assert!(overlaps(&pending, 0x1_8000, 0x1_9000, now + Duration::from_millis(DS_HOLD_MS)));
    }

    /// `VICTORY_STALE_SLOTS`: a small mip gets one window at its start, a
    /// long one also at its middle (on a 4 KiB line) and its end; windows
    /// that touch are joined, and none leaves its mip.
    #[test]
    fn the_stale_look_samples_every_mip_and_the_long_ones_thrice() {
        let mips = [(0, 0x80), (0x80, 0x80), (0x100, 0x1000), (0x1_0000, 0x4_0000)];
        let w = sample_windows(&mips);
        assert_eq!(w, [(0, 0x200), (0x1_0000, 0x100), (0x3_0000, 0x100), (0x4_ff00, 0x100)]);
        for (at, n) in sample_windows(&[(0x100, 0x2_0001)]) {
            assert!(at >= 0x100 && at + n <= 0x2_0101, "{at:#x}+{n:#x}");
        }
        assert!(sample_windows(&[]).is_empty());
    }

    /// The sampled fingerprint changes with a byte under a window and not
    /// with one between them; memory past the texture is not read.
    #[test]
    fn the_sampled_fingerprint_sees_its_windows_only() {
        let mut memory = vec![7u8; 0x2_0000];
        let windows = sample_windows(&[(0, 0x1000), (0x1000, 0x1_f000)]);
        let at = memory.as_ptr() as u64;
        let print = sample_print(at, memory.len() as u64, &windows);
        assert_ne!(print, 0);
        memory[0x2000] = 9;
        assert_eq!(sample_print(memory.as_ptr() as u64, memory.len() as u64, &windows), print);
        memory[0x1010] = 9;
        assert_ne!(sample_print(memory.as_ptr() as u64, memory.len() as u64, &windows), print);
        assert_eq!(sample_print(at, 0, &windows), sample_print(at, 0, &[]));
    }

    /// The print of bytes in hand (what a staging buffer's detiling took) is
    /// the print the stale look takes of memory holding them, windows past
    /// the end cut as there.
    #[test]
    fn the_print_of_bytes_in_hand_is_the_print_of_memory_holding_them() {
        let mut memory = vec![0u8; 0x3_0000];
        for (i, b) in memory.iter_mut().enumerate() {
            *b = (i * 7 % 251) as u8;
        }
        let windows = sample_windows(&[(0, 0x1000), (0x1000, 0x2_f000)]);
        let at = memory.as_ptr() as u64;
        assert_eq!(sample_print_of(&memory, &windows), sample_print(at, memory.len() as u64, &windows));
        let short = &memory[..0x1_0080];
        assert_eq!(sample_print_of(short, &windows), sample_print(at, short.len() as u64, &windows));
        assert_ne!(sample_print_of(short, &windows), sample_print_of(&memory, &windows));
    }

    /// `VICTORY_TABLE_BOUND_FRESH` with `VICTORY_DS_HOLD`: a texture staged
    /// for draws is held while a write is under way into its memory (not one
    /// that only touches it), once a frame counts, at most [`DS_HOLD_MOST`]
    /// frames for one write; with the write done it goes on, and the next
    /// write is held anew. Nothing is held while FRESH does not act.
    #[test]
    fn the_ds_hold_applies_to_textures_the_draws_keep() {
        let at = 0x7fe9_0000_0000u64;
        let len = 0x2_0000u64;
        let hold = StageHold::default();
        assert!(!stage_held(&hold, true, at, len, 10), "nothing under way");
        let write = test_pending(vec![(at + 0x1_0000, at + 0x1_1000)]);
        assert!(!stage_held(&hold, false, at, len, 10), "FRESH not acting: never held");
        let touching = StageHold::default();
        assert!(!stage_held(&touching, true, at + len, 0x1000, 10), "a texture that only touches the write");
        for frame in 10..10 + u64::from(DS_HOLD_MOST) {
            assert!(stage_held(&hold, true, at, len, frame), "held at {frame}");
            assert!(stage_held(&hold, true, at, len, frame), "a second use in the frame: held, not counted");
        }
        assert!(!stage_held(&hold, true, at, len, 10 + u64::from(DS_HOLD_MOST)), "gone on after the bound");
        drop(write);
        assert!(!stage_held(&hold, true, at, len, 40), "the write done");
        let next = test_pending(vec![(at, at + 0x100)]);
        assert!(stage_held(&hold, true, at, len, 41), "the next write held anew");
        drop(next);
    }

    /// A slot draws make is judged against what its staging buffer's last
    /// detiling took: the same; differing and waiting for the next check,
    /// still differing from that same host copy [`TABLE_LASTING`] frames on
    /// (its checks miss it), and waiting again once the staging buffer took
    /// something new. Without a staging buffer, against the last host copy
    /// known: in the queue it waits for the pass, out of it nothing uploads
    /// it again. Nothing known yet: no verdict.
    #[test]
    fn table_made_slots_are_judged_against_their_staging_buffer() {
        use TableVerdict::*;
        let mut p = TablePrints::default();
        assert_eq!(table_verdict(&mut p, 5, None, 100, false), Unknown);
        assert_eq!(table_verdict(&mut p, 5, Some(5), 100, false), Same);
        assert_eq!(p.host, 5);
        assert_eq!(table_verdict(&mut p, 7, Some(5), 110, false), Waiting, "first seen differing");
        assert_eq!(table_verdict(&mut p, 7, Some(5), 110 + TABLE_LASTING - 1, false), Waiting);
        assert_eq!(table_verdict(&mut p, 9, Some(5), 110 + TABLE_LASTING, false), Lasting, "the checks missed it");
        assert_eq!(table_verdict(&mut p, 9, Some(8), 200, false), Waiting, "a new detiling, not yet this");
        assert_eq!(table_verdict(&mut p, 9, Some(9), 201, false), Same);
        assert_eq!(p.differ, (0, 0));
        assert_eq!(table_verdict(&mut p, 11, None, 300, true), Waiting, "no staging buffer, in the queue");
        assert_eq!(table_verdict(&mut p, 11, None, 301, false), Out, "nothing uploads it again");
        assert_eq!(table_verdict(&mut p, 9, None, 302, false), Same, "as its last host copy");
    }
}
