//! VIcky: lot 4's probes of the host side (`VICTORY_LOT4_PROBE=1`; off by
//! default), as `fable-2026-11-27-sources.md` section D lays them out. Each
//! says one line with the frame log's reading (every five seconds),
//! `lot4 probe Pn: ...`, per frame over the reading's frames:
//!
//! - P1, the mirror's size: the 64 KiB blocks under live host copies of
//!   buffers and windows (in the card's memory or the PC's; over XMem
//!   private memory, views of physical pages, or host addresses), and the
//!   blocks found written (copied in from title memory) each frame. Counted
//!   on a thread of its own ([`census`]), with P4 and P5.
//! - P2, the slots it would take off the descriptor side: of the slots the
//!   heap sync mirrors and the pass over buffer views makes, the buffer
//!   views (and constant buffer views) whose range lies wholly in P1's
//!   blocks (as the last census found them).
//! - P3, snapshots title memory could stand in for: the snapshot bytes
//!   taken, and at each landing, one 64 KiB block in 16 compared with title
//!   memory: equal, title memory still is the before-image; different, the
//!   block needs a true one (and whether it needed one in the 64 frames
//!   before).
//! - P4 and P5: `crate::inplace::lot4` (each physical page placed at its
//!   first mapping, simulated; the alias census), and the title's
//!   reservations by class ([`reserved`]).
//! - P6: [`faults`] (protects the pages the landings wrote for one frame in
//!   64; `VICTORY_LOT4_PROBE_FAULTS=0` leaves it out).
//! - P7: [`locks`] (acquisitions and waits of the hot locks) and [`ccd`]
//!   (`VICTORY_CCD_PLACE=1|alt`, which works without the probes).
//!
//! Off, each hook is one load of a flag (diagnostics builds; in others the
//! hooks are compiled out and [`on`] is false).

// (Called from diagnostics builds only; the lock helpers are used in all.)
#![cfg_attr(not(feature = "diagnostics"), allow(dead_code))]

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::perthread::Counters;

pub(crate) mod ccd;
pub(crate) mod faults;
pub(crate) mod locks;

/// The probes' block: the console's 64 KiB page.
pub(crate) const BLOCK: usize = 64 << 10;

/// Title addresses below this are the console's; above, the host's.
pub(crate) const HOST_ADDRESSES: u64 = 1 << 40;

/// Whether the probes are on (`VICTORY_LOT4_PROBE=1`, read once).
#[inline]
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::AtomicU8;
        /// 0 not read yet, 1 off, 2 on.
        static ON: AtomicU8 = AtomicU8::new(0);
        match ON.load(Relaxed) {
            1 => false,
            2 => true,
            _ => read_switch(&ON),
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    false
}

#[cfg(feature = "diagnostics")]
#[cold]
fn read_switch(on: &std::sync::atomic::AtomicU8) -> bool {
    let is = std::env::var_os("VICTORY_LOT4_PROBE").is_some_and(|v| v == "1");
    if on.swap(if is { 2 } else { 1 }, Relaxed) == 0 && is {
        tracing::info!(
            "lot 4 probes on (VICTORY_LOT4_PROBE=1): lines 'lot4 probe P1..P7' with the frame log's readings; P6 protects \
             the pages the landings wrote for one frame in 64 (VICTORY_LOT4_PROBE_FAULTS=0: not)"
        );
    }
    is
}

/// Frames ended so far ([`frame_ended`]).
static FRAME: AtomicU64 = AtomicU64::new(0);

/// A frame ended (`crate::framelog::frame`, on the presenting thread):
/// its length, and of it the submit thread's time at work, the recording
/// threads' list calls and the longest one's (microseconds; nanoseconds
/// for the list calls).
pub(crate) fn frame_ended(us: f64, at_work_us: f64, record_ns: f64, longest_ns: f64) {
    ccd::frame_ended(us, at_work_us, record_ns, longest_ns);
    if !on() {
        return;
    }
    let frame = FRAME.fetch_add(1, Relaxed) + 1;
    {
        let mut w = WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        let w = w.get_or_insert_with(Written::default);
        w.blocks += w.frame.len() as u64;
        w.frames += 1;
        let frame_blocks = std::mem::take(&mut w.frame);
        w.period.extend(frame_blocks);
    }
    faults::frame(frame);
}

/// The frame log's reading of `frames` frames is out: the probes' lines.
pub(crate) fn period(frames: usize) {
    if !on() || frames == 0 {
        return;
    }
    let n = frames as f64;
    let written = WRITTEN.lock().unwrap_or_else(|e| e.into_inner()).as_mut().map(Written::take).unwrap_or_default();
    slots_line(n);
    snapshots_line(n);
    faults::line(n);
    locks::line(n);
    census(written);
}

// ---- P1: the blocks under live host copies ----

/// Blocks found written into host copies: this frame's, the reading's, and
/// the sums the line divides.
#[derive(Default)]
struct Written {
    frame: HashSet<u64>,
    period: HashSet<u64>,
    /// Frames, blocks summed over frames, bytes copied, of them into copies
    /// in the card's memory.
    frames: u64,
    blocks: u64,
    bytes: u64,
    card_bytes: u64,
}

/// What a reading's [`Written`] hands to the census line.
#[derive(Default, Clone, Copy)]
struct WrittenTaken {
    frames: u64,
    blocks: u64,
    distinct: u64,
    bytes: u64,
    card_bytes: u64,
}

impl Written {
    fn take(&mut self) -> WrittenTaken {
        let out = WrittenTaken {
            frames: self.frames,
            blocks: self.blocks,
            distinct: self.period.len() as u64,
            bytes: self.bytes,
            card_bytes: self.card_bytes,
        };
        self.period.clear();
        (self.frames, self.blocks, self.bytes, self.card_bytes) = (0, 0, 0, 0);
        out
    }
}

static WRITTEN: Mutex<Option<Written>> = Mutex::new(None);

/// A host copy got `[from, to)` of title memory (`Shadow::copy`: pages
/// found written, or copied whole); `card`: a copy in the card's memory.
pub(crate) fn copied(from: usize, to: usize, card: bool) {
    if !on() || from >= to {
        return;
    }
    let mut w = WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    let w = w.get_or_insert_with(Written::default);
    for block in from / BLOCK..=(to - 1) / BLOCK {
        w.frame.insert(block as u64);
    }
    w.bytes += (to - from) as u64;
    if card {
        w.card_bytes += (to - from) as u64;
    }
}

/// The blocks under live host copies as the last census found them:
/// sorted ranges of whole blocks `[start, end)`, and a number that changes
/// with each census ([`inside`] keeps a copy per thread).
static MIRROR: Mutex<(u64, Option<Arc<Vec<(u64, u64)>>>)> = Mutex::new((0, None));
static MIRROR_AT: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static SEEN: std::cell::RefCell<(u64, Option<Arc<Vec<(u64, u64)>>>)> = const { std::cell::RefCell::new((0, None)) };
}

/// Whether `[at, at + len)` lies wholly in blocks under live host copies.
fn inside(at: u64, len: u64) -> bool {
    let census = MIRROR_AT.load(Relaxed);
    if census == 0 {
        return false;
    }
    SEEN.with_borrow_mut(|seen| {
        if seen.0 != census {
            *seen = MIRROR.lock().unwrap_or_else(|e| e.into_inner()).clone();
        }
        seen.1.as_deref().is_some_and(|ranges| covered(ranges, at, len))
    })
}

/// Whether sorted, disjoint `ranges` hold `[at, at + len)` in one of them.
fn covered(ranges: &[(u64, u64)], at: u64, len: u64) -> bool {
    let end = at.saturating_add(len.max(1));
    let i = ranges.partition_point(|r| r.0 <= at);
    i > 0 && ranges[i - 1].1 >= end
}

/// Whether `at` lies in one of sorted, disjoint `ranges`.
pub(crate) fn covered_by(ranges: &[(u64, u64)], at: u64) -> bool {
    covered(ranges, at, 1)
}

/// `ranges` (start, end) as sorted, disjoint ranges of whole blocks.
fn merged_blocks(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    let b = BLOCK as u64;
    for r in &mut ranges {
        *r = (r.0 / b * b, r.1.div_ceil(b) * b);
    }
    ranges.retain(|r| r.0 < r.1);
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

fn blocks_of(ranges: &[(u64, u64)]) -> u64 {
    ranges.iter().map(|r| (r.1 - r.0) / BLOCK as u64).sum()
}

/// Whether the census thread is at work.
static CENSUS: AtomicBool = AtomicBool::new(false);

/// P1, P4 and P5 on a thread of their own (a walk of every host copy and of
/// the title's views: some milliseconds), each said in its own line when
/// done. P2's blocks are published for the next reading.
fn census(written: WrittenTaken) {
    if CENSUS.swap(true, Relaxed) {
        return;
    }
    let spawned = std::thread::Builder::new().name("victory-lot4-census".into()).spawn(move || {
        census_now(written);
        CENSUS.store(false, Relaxed);
    });
    if spawned.is_err() {
        CENSUS.store(false, Relaxed);
    }
}

fn census_now(written: WrittenTaken) {
    let started = std::time::Instant::now();
    let copies = crate::shadow::lot4_copies();
    let all = merged_blocks(copies.iter().map(|c| (c.0, c.0 + c.1)).collect());
    let card = merged_blocks(copies.iter().filter(|c| c.2).map(|c| (c.0, c.0 + c.1)).collect());
    let pc = merged_blocks(copies.iter().filter(|c| !c.2).map(|c| (c.0, c.0 + c.1)).collect());
    let windows = merged_blocks(copies.iter().filter(|c| c.3).map(|c| (c.0, c.0 + c.1)).collect());
    let host = merged_blocks(all.iter().filter(|r| r.1 > HOST_ADDRESSES).map(|r| (r.0.max(HOST_ADDRESSES), r.1)).collect());
    let viewed = crate::inplace::lot4::view_bytes(&all);
    let total = blocks_of(&all);
    let mb = |blocks: u64| (blocks * BLOCK as u64) as f64 / (1 << 20) as f64;
    let host_blocks = blocks_of(&host);
    let view_blocks = viewed / BLOCK as u64;
    let private = total.saturating_sub(host_blocks + view_blocks);
    let frames = written.frames.max(1) as f64;
    tracing::info!(
        "lot4 probe P1: blocks of 64 KiB under live host copies of buffers and windows: {total} ({:.0} MB): in the card's \
         memory {}, in the PC's {}; over XMem private memory {private} ({:.1} %), views of physical pages {view_blocks} \
         ({:.1} %), host addresses {host_blocks} ({:.1} %); of windows {}; {} copies. Found written (copied in from title \
         memory) per frame: {:.0} blocks ({:.1} MB as blocks; {:.1} MB copied, {:.1} MB of it into card copies), {} \
         distinct over the reading's {} frames",
        mb(total),
        blocks_of(&card),
        blocks_of(&pc),
        pct(private, total),
        pct(view_blocks, total),
        pct(host_blocks, total),
        blocks_of(&windows),
        copies.len(),
        written.blocks as f64 / frames,
        written.blocks as f64 * BLOCK as f64 / 1e6 / frames,
        written.bytes as f64 / 1e6 / frames,
        written.card_bytes as f64 / 1e6 / frames,
        written.distinct,
        written.frames,
    );
    // P2's blocks for the next reading.
    let all = Arc::new(all);
    {
        let mut m = MIRROR.lock().unwrap_or_else(|e| e.into_inner());
        let census = m.0 + 1;
        *m = (census, Some(all.clone()));
        MIRROR_AT.store(census, Relaxed);
    }
    crate::inplace::lot4::placement_line(&copies);
    crate::inplace::lot4::alias_line(&all);
    classes_line();
    tracing::debug!("lot4 census took {} ms", started.elapsed().as_millis());
}

pub(crate) fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 { 0.0 } else { 100.0 * part as f64 / whole as f64 }
}

// ---- P2: the slots the mirror would take off the descriptor side ----

/// Who made a slot.
#[derive(Clone, Copy)]
pub(crate) enum Who {
    /// The heap sync (and the title's own descriptor calls, mirrored then).
    Sync = 0,
    /// The pass over buffer views.
    Pass = 8,
}

/// Per [`Who`]: slots, not a view of ours, buffer views, of them in P1's
/// blocks, constant buffer views, of them in P1's blocks, texture views,
/// other views (null constant buffers, samplers).
static SLOTS: Counters<16> = Counters::new();
const S_ALL: usize = 0;
const S_NONE: usize = 1;
const S_BUFFER: usize = 2;
const S_CONSTANT: usize = 4;
const S_TEXTURE: usize = 6;
const S_OTHER: usize = 7;

thread_local! {
    /// Whether this thread mirrors the title's own descriptor call now
    /// (`HostHeap::mirror_tried`): counted with the sync's slots, but
    /// said apart.
    static IN_CALL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The memory a buffer view reads (address, bytes): its range, or for a
/// view of a whole resource the resource's.
fn buffer_of(view: &crate::descriptor::View) -> Option<(u64, u64)> {
    use crate::descriptor::View;
    crate::descriptor::host::buffer_range(view).or_else(|| match view {
        View::Shader(r, None) | View::Unordered(r, _, None) => {
            crate::resource::buffer_bytes(*r as crate::object::This).map(|(data, len)| (data as u64, len))
        }
        _ => None,
    })
}

/// A slot `who` made from `view` (`None`: no view of ours).
pub(crate) fn slot(who: Who, view: Option<&crate::descriptor::View>) {
    if !on() {
        return;
    }
    use crate::descriptor::View;
    let base = who as usize;
    SLOTS.add(base + S_ALL, 1);
    let Some(view) = view else {
        SLOTS.add(base + S_NONE, 1);
        return;
    };
    let (kind, range) = match view {
        View::Constant(at, size) => (S_CONSTANT, Some((*at, u64::from(*size)))),
        View::NullConstant | View::Sampler(_) | View::ConsoleSampler(_) => (S_OTHER, None),
        _ => match buffer_of(view) {
            Some(range) => (S_BUFFER, Some(range)),
            None => (S_TEXTURE, None),
        },
    };
    SLOTS.add(base + kind, 1);
    if let Some((at, len)) = range
        && inside(at, len)
    {
        SLOTS.add(base + kind + 1, 1);
    }
}

/// A slot the sync made: only off the title's own descriptor calls.
pub(crate) fn sync_slot(view: Option<&crate::descriptor::View>) {
    if on() && !IN_CALL.get() {
        slot(Who::Sync, view);
    }
}

/// `f`, the mirroring of a title's descriptor call (not the sync's).
pub(crate) fn in_call<R>(f: impl FnOnce() -> R) -> R {
    if !on() {
        return f();
    }
    let was = IN_CALL.replace(true);
    let out = f();
    IN_CALL.set(was);
    out
}

fn slots_line(frames: f64) {
    let s = SLOTS.take();
    let part = |base: usize, name: &str| {
        let all = s[base + S_ALL];
        let in_buffers = s[base + S_BUFFER + 1];
        let in_constants = s[base + S_CONSTANT + 1];
        format!(
            "{name} {:.0} slots a frame: buffer views {:.0} ({:.0} in the blocks: {:.1} % of all slots), constant buffer views \
             {:.0} ({:.0} in the blocks), texture views {:.0}, other views {:.0}, no view of ours {:.0}; buffer and constant \
             views in the blocks {:.1} % of all",
            all as f64 / frames,
            s[base + S_BUFFER] as f64 / frames,
            in_buffers as f64 / frames,
            pct(in_buffers, all),
            s[base + S_CONSTANT] as f64 / frames,
            in_constants as f64 / frames,
            s[base + S_TEXTURE] as f64 / frames,
            s[base + S_OTHER] as f64 / frames,
            s[base + S_NONE] as f64 / frames,
            pct(in_buffers + in_constants, all),
        )
    };
    let (sync, pass) = (Who::Sync as usize, Who::Pass as usize);
    let both = s[sync + S_ALL] + s[pass + S_ALL];
    let both_in = s[sync + S_BUFFER + 1] + s[pass + S_BUFFER + 1];
    tracing::info!(
        "lot4 probe P2: slots that are buffer views lying wholly in P1's blocks (the last census's): heap sync and pass \
         together {:.1} % of {:.0} slots a frame (kill under 60 %); {}; {}",
        pct(both_in, both),
        both as f64 / frames,
        part(sync, "the heap sync"),
        part(pass, "the pass over buffer views"),
    );
}

// ---- P3: snapshots title memory could stand in for ----

/// Snapshot bytes and blocks taken; at landings, blocks compared (one in
/// [`SAMPLE`]) and their bytes, of them equal to title memory and their
/// bytes, different, of those with a difference in the 64 frames before
/// and without; blocks with no snapshot; nanoseconds of the compares.
static SNAPS: Counters<12> = Counters::new();
const P_BYTES: usize = 0;
const P_BLOCKS: usize = 1;
const P_SAMPLED: usize = 2;
const P_SAMPLED_BYTES: usize = 3;
const P_EQUAL: usize = 4;
const P_EQUAL_BYTES: usize = 5;
const P_DIFFER: usize = 6;
const P_HISTORY: usize = 7;
const P_FRESH: usize = 8;
const P_NONE: usize = 9;
const P_NS: usize = 10;

/// One block in this many is compared at a landing.
const SAMPLE: u64 = 16;

/// Whether `block` is one of the compared ones (the same blocks every
/// time, so that a block's history is its own).
fn sampled(block: u64) -> bool {
    block.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60 == 0
}

/// A snapshot of `parts` (offsets in the copy, `[from, to)`) of the copy
/// at `start` was taken (`Shadow::writing`).
pub(crate) fn snapped(start: u64, parts: &[(usize, usize)]) {
    if !on() {
        return;
    }
    let (mut bytes, mut blocks) = (0u64, 0u64);
    for &(from, to) in parts {
        if from < to {
            let (a, e) = (start as usize + from, start as usize + to);
            bytes += (to - from) as u64;
            blocks += ((e - 1) / BLOCK - a / BLOCK + 1) as u64;
        }
    }
    SNAPS.add(P_BYTES, bytes);
    SNAPS.add(P_BLOCKS, blocks);
}

/// Per compared block with a difference: the frame of its last one.
static DIFFERED: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);

/// A landing is about to compare the declared ranges `work` (copy, title
/// `[from, to)`) with their snapshots (`crate::shadow::find_all`): before
/// title memory takes what shaders wrote, each sampled block's snapshot is
/// compared with title memory.
pub(crate) fn landed(work: &[(Arc<crate::shadow::Shadow>, usize, usize)]) {
    if !on() {
        return;
    }
    let started = std::time::Instant::now();
    let frame = FRAME.load(Relaxed);
    let mut title = Vec::new();
    for (shadow, from, to) in work {
        let mut at = *from;
        while at < *to {
            let block = (at / BLOCK) as u64;
            let end = ((at / BLOCK + 1) * BLOCK).min(*to);
            if sampled(block) {
                match shadow.lot4_matches(at, end, &mut title) {
                    None => SNAPS.add(P_NONE, 1),
                    Some(equal) => {
                        let n = (end - at) as u64;
                        SNAPS.add(P_SAMPLED, 1);
                        SNAPS.add(P_SAMPLED_BYTES, n);
                        if equal {
                            SNAPS.add(P_EQUAL, 1);
                            SNAPS.add(P_EQUAL_BYTES, n);
                        } else {
                            differed(block, frame);
                        }
                    }
                }
            }
            at = end;
        }
    }
    SNAPS.add(P_NS, started.elapsed().as_nanos() as u64);
}

/// A compared block differs in `frame`: counted once a frame, with or
/// without a difference in the 64 frames before.
fn differed(block: u64, frame: u64) {
    let mut d = DIFFERED.lock().unwrap_or_else(|e| e.into_inner());
    let last = d.get_or_insert_with(HashMap::new).insert(block, frame);
    if last == Some(frame) {
        return;
    }
    SNAPS.add(P_DIFFER, 1);
    let recent = last.is_some_and(|f| frame.saturating_sub(f) <= 64);
    SNAPS.add(if recent { P_HISTORY } else { P_FRESH }, 1);
}

fn snapshots_line(frames: f64) {
    let s = SNAPS.take();
    let scale = SAMPLE as f64 / frames;
    tracing::info!(
        "lot4 probe P3: snapshots per frame {:.1} MB in {:.0} block parts; at the landings, {:.1} sampled blocks a frame \
         (one in {SAMPLE}, {:.2} MB) compared with title memory: {:.1} % of their bytes still equal to the snapshot (title \
         memory could stand in; kill under 80 %), {:.1} blocks a frame differ (about {:.0} a frame over all blocks, which \
         need a true before-image): {:.1} of them (about {:.0}) with a difference in the 64 frames before, {:.1} (about \
         {:.0}) without; {:.1} sampled blocks a frame had no snapshot; the compares took {:.2} ms a frame",
        s[P_BYTES] as f64 / 1e6 / frames,
        s[P_BLOCKS] as f64 / frames,
        s[P_SAMPLED] as f64 / frames,
        s[P_SAMPLED_BYTES] as f64 / 1e6 / frames,
        pct(s[P_EQUAL_BYTES], s[P_SAMPLED_BYTES]),
        s[P_DIFFER] as f64 / frames,
        s[P_DIFFER] as f64 * scale,
        s[P_HISTORY] as f64 / frames,
        s[P_HISTORY] as f64 * scale,
        s[P_FRESH] as f64 / frames,
        s[P_FRESH] as f64 * scale,
        s[P_NONE] as f64 / frames,
        s[P_NS] as f64 / 1e6 / frames,
    );
}

// ---- P5: the title's reservations by class ----

/// Reservations by (page size, write-combined, uncached): how many, bytes.
static CLASSES: Mutex<Option<std::collections::BTreeMap<(u32, bool, bool), (u64, u64)>>> = Mutex::new(None);

/// The title reserved `len` bytes (`XMemVirtualAlloc`, with its own `kind`
/// and console `protect`: `crate::census::title_memory_reserved`).
pub(crate) fn reserved(len: usize, kind: u32, protect: u32) {
    if !on() || len == 0 {
        return;
    }
    const PAGES_2M: u32 = 0x2000_0000;
    const PAGES_64K: u32 = 0x40_0000;
    const WRITE_COMBINE: u32 = 0x400;
    const NO_CACHE: u32 = 0x200;
    let page = if kind & PAGES_2M != 0 {
        2048
    } else if kind & PAGES_64K != 0 {
        64
    } else {
        4
    };
    let mut c = CLASSES.lock().unwrap_or_else(|e| e.into_inner());
    let e = c.get_or_insert_with(Default::default).entry((page, protect & WRITE_COMBINE != 0, protect & NO_CACHE != 0)).or_default();
    e.0 += 1;
    e.1 += len as u64;
}

fn classes_line() {
    let c = CLASSES.lock().unwrap_or_else(|e| e.into_inner());
    let text: Vec<String> = c
        .iter()
        .flatten()
        .map(|(&(page, wc, nc), &(n, bytes))| {
            format!(
                "{page} KiB pages{}{}: {n} reservations {:.0} MB",
                if wc { " write-combined" } else { "" },
                if nc { " uncached" } else { "" },
                bytes as f64 / (1 << 20) as f64
            )
        })
        .collect();
    tracing::info!(
        "lot4 probe P5: the title's reservations (XMemVirtualAlloc at an address of its choice) by class, since the start: {}",
        if text.is_empty() { "none".into() } else { text.join("; ") }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_merge_into_whole_blocks() {
        let b = BLOCK as u64;
        // Rounded out to blocks: (b, 3b), (0, b), (3b, 4b), (4b, 5b); touching
        // ranges join.
        let merged = merged_blocks(vec![(b + 5, 2 * b + 1), (10, 20), (3 * b, 4 * b), (4 * b, 4 * b + 1)]);
        assert_eq!(merged, vec![(0, 5 * b)]);
        assert_eq!(blocks_of(&merged), 5);
        let apart = merged_blocks(vec![(0, 1), (2 * b, 2 * b + 1)]);
        assert_eq!(apart, vec![(0, b), (2 * b, 3 * b)]);
        assert!(!covered(&apart, b, 1));
        assert!(covered(&merged, b, b));
        assert!(covered(&merged, 4 * b + 7, 100));
        assert!(!covered(&merged, 4 * b + 7, b));
        assert!(!covered(&merged, 6 * b, 1));
    }

    #[test]
    fn one_block_in_sixteen_is_sampled() {
        let n = (0..160_000u64).filter(|&b| sampled(b)).count();
        assert!((9_000..11_000).contains(&n), "{n}");
    }

    #[test]
    fn a_difference_counts_once_a_frame_with_its_history() {
        let before = SNAPS.totals();
        differed(0xdead_0001, 1000);
        differed(0xdead_0001, 1000);
        differed(0xdead_0001, 1010);
        differed(0xdead_0001, 1200);
        let after = SNAPS.totals();
        let d = |i: usize| after[i].wrapping_sub(before[i]);
        assert_eq!(d(P_DIFFER), 3);
        assert_eq!(d(P_HISTORY), 1);
        assert_eq!(d(P_FRESH), 2);
    }
}
