//! VIcky (`VICTORY_TEXTURE_AUDIT=<frame>,<frame>,...`; off by default): at
//! each listed world frame, every placed texture with a host copy and a
//! current view, its memory hashed whole against what its last detiling
//! took (`super::Record`, kept while the audit is on), in one table: the
//! mismatches by format, size and state, and the 20 largest.
//!
//! World frames are counted as `VICTORY_CAPTURE_WORLD_ALSO` counts them:
//! presented frames from the first one that ran 500 draws or more.
//!
//! **The decoration bake.** The title bakes its characters' decorations into
//! textures of its own once, with pixel shaders that pick their sources by
//! index (`crate::draw` `BAKE_PIXEL_SHADERS`); a source that is not what its
//! memory holds at that moment stays baked wrong for the run. With the audit
//! on (any value; `VICTORY_TEXTURE_AUDIT=bake` for these alone), each bake
//! draw says after its own pass how many slots of the queue are due, wait
//! for their memory and are uploaded in part ([`bake`]), and the first
//! [`bake_walks`] bake draws (one a frame) take a snapshot for a walk as a
//! listed frame does, whose lines say how many textures differ from their
//! memory then.
//!
//! Three steps, none on the submitting thread:
//! 1. the texture pass of the frame (or the bake draw's) takes a snapshot of
//!    the bookkeeping: the records, the slots whose views describe each
//!    one's memory (with their descriptor handles) and the textures draws
//!    staged lately (read through tables, `VICTORY_TABLES_LEFT` /
//!    `VICTORY_TABLE_BOUND`);
//! 2. a thread of its own (`victory-wide-audit`) keeps the textures that
//!    have a current view (a slot whose view is of that very texture, or
//!    staged for draws lately) and mapped memory, hashes their memory
//!    (`super::memory_hash`, pages not mapped as zeros, as the detiling read
//!    them) and notes the ones that differ; seconds for some 5 GB;
//! 3. the next texture pass reads the queue for the slots of those, and the
//!    records again, says the state of each and writes the lines.
//!
//! States, in this order: **in flight** (an event of the texture, its
//! detiling or a write of a slot of it in the queue, 8 frames or fewer
//! before the snapshot, or after it); **partly mapped** (not all of its
//! memory mapped now); **queued more than 8 frames** (a slot of it in the
//! queue, its last event older); **out of the queue** (nothing will upload
//! it again). Clean: 0 out of the queue and 0 queued more than 8 frames.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use super::Record;
use crate::descriptor::View;

/// What `VICTORY_TEXTURE_AUDIT` says: the listed world frames (at most 64
/// count) and whether anything was set (the bake's reports). Pure, for
/// testing.
pub(crate) fn parse(v: Option<&str>) -> (Vec<u64>, bool) {
    let Some(v) = v.map(str::trim).filter(|v| !v.is_empty() && *v != "0") else {
        return (Vec::new(), false);
    };
    (v.split(',').filter_map(|f| f.trim().parse().ok()).take(64).collect(), true)
}

fn asked() -> &'static (Vec<u64>, bool) {
    static ASKED: OnceLock<(Vec<u64>, bool)> = OnceLock::new();
    ASKED.get_or_init(|| {
        let asked = parse(std::env::var("VICTORY_TEXTURE_AUDIT").ok().as_deref());
        if asked.1 {
            tracing::info!(
                frames = ?asked.0,
                "texture audit at these world frames (presented frames from the first of 500 draws or more), and at the \
                 decoration bake's draws: every texture with a host copy and a current view, its memory against the bytes \
                 last uploaded (VICTORY_TEXTURE_AUDIT; lines 'texture audit at frame N:' and 'texture audit at bake draw N:')"
            );
        }
        asked
    })
}

/// The listed world frames.
pub(crate) fn frames() -> &'static [u64] {
    &asked().0
}

/// Whether the audit is asked for.
pub(crate) fn on() -> bool {
    asked().1
}

/// The bake draws that take a snapshot for a walk, at most one a frame
/// (`VICTORY_TEXTURE_AUDIT_BAKES`, 8 unless said).
pub(crate) fn bake_walks() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_TEXTURE_AUDIT_BAKES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8))
}

/// Frames (`crate::frame::current`) an event may lie before the snapshot
/// and the texture still count as in flight.
pub(crate) const IN_FLIGHT: u64 = 8;

/// Frames since its last use for a texture staged for draws to count as
/// having a current view.
const STAGED_FRAMES: u64 = 16;

/// A mismatch's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum State {
    OutOfQueue,
    QueuedLong,
    InFlight,
    Partly,
}

impl State {
    fn words(self) -> &'static str {
        match self {
            State::OutOfQueue => "out of the queue",
            State::QueuedLong => "queued more than 8 frames",
            State::InFlight => "in flight",
            State::Partly => "partly mapped",
        }
    }
}

/// The state of a texture whose memory differs from its record: `snapshot`
/// is the frame of the bookkeeping, `last_event` its last event's frame
/// (its detiling, or the write of a slot of it in the queue; also one after
/// the snapshot), `queued` whether a slot of it is in the queue, `partly`
/// whether not all of its memory is mapped. Pure, for testing.
pub(crate) fn classify(snapshot: u64, last_event: u64, queued: bool, partly: bool) -> State {
    if last_event.saturating_add(IN_FLIGHT) >= snapshot {
        State::InFlight
    } else if partly {
        State::Partly
    } else if queued {
        State::QueuedLong
    } else {
        State::OutOfQueue
    }
}

/// The size bucket of a texture of `len` bytes: its order and its name.
pub(crate) fn bucket(len: u64) -> (u8, &'static str) {
    match len {
        0..0x1_0000 => (0, "under 64 KiB"),
        0x1_0000..0x4_0000 => (1, "64-256 KiB"),
        0x4_0000..0x10_0000 => (2, "256 KiB-1 MiB"),
        0x10_0000..0x40_0000 => (3, "1-4 MiB"),
        0x40_0000..0x100_0000 => (4, "4-16 MiB"),
        _ => (5, "16 MiB and more"),
    }
}

/// A DXGI format's name, for the common ones.
fn format_name(format: u32) -> String {
    let name = match format {
        2 => "R32G32B32A32_FLOAT",
        10 => "R16G16B16A16_FLOAT",
        24 => "R10G10B10A2_UNORM",
        26 => "R11G11B10_FLOAT",
        28 => "R8G8B8A8_UNORM",
        29 => "R8G8B8A8_UNORM_SRGB",
        34 => "R16G16_FLOAT",
        41 => "R32_FLOAT",
        49 => "R8G8_UNORM",
        54 => "R16_FLOAT",
        56 => "R16_UNORM",
        61 => "R8_UNORM",
        71 => "BC1_UNORM",
        72 => "BC1_UNORM_SRGB",
        74 => "BC2_UNORM",
        75 => "BC2_UNORM_SRGB",
        77 => "BC3_UNORM",
        78 => "BC3_UNORM_SRGB",
        80 => "BC4_UNORM",
        81 => "BC4_SNORM",
        83 => "BC5_UNORM",
        84 => "BC5_SNORM",
        87 => "B8G8R8A8_UNORM",
        91 => "B8G8R8A8_UNORM_SRGB",
        95 => "BC6H_UF16",
        96 => "BC6H_SF16",
        98 => "BC7_UNORM",
        99 => "BC7_UNORM_SRGB",
        _ => return format!("format {format}"),
    };
    format!("{name} ({format})")
}

/// Which audit: a listed world frame, or a bake draw (its number since the
/// start).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    Frame(u64),
    Bake(u64),
}

impl Label {
    /// The start of its lines.
    fn prefix(self) -> String {
        match self {
            Label::Frame(n) => format!("texture audit at frame {n}:"),
            Label::Bake(n) => format!("texture audit at bake draw {n}:"),
        }
    }
}

// When.

/// The first presented frame of 500 draws or more.
static WORLD: AtomicU64 = AtomicU64::new(u64::MAX);
/// The listed frames whose audit was asked for (a bit each).
static TAKEN: AtomicU64 = AtomicU64::new(0);
/// The world frames whose snapshot is due.
static DUE: Mutex<VecDeque<u64>> = Mutex::new(VecDeque::new());
/// An audit is between its snapshot and its lines (claimed by
/// compare-exchange: one at a time).
static RUNNING: AtomicBool = AtomicBool::new(false);

/// At the present of `frame` (`crate::queue`), the draws run on the host so
/// far: the world's first frame is noted, and the audits whose world frame
/// has come are due.
pub(crate) fn presented(frame: u64, drawn: u64) {
    if frames().is_empty() {
        return;
    }
    static LAST: AtomicU64 = AtomicU64::new(0);
    let before = LAST.swap(drawn, Relaxed);
    if drawn.saturating_sub(before) >= 500 && WORLD.compare_exchange(u64::MAX, frame, Relaxed, Relaxed).is_ok() {
        tracing::info!(frame, "texture audit: the world's first frame (500 draws or more) is presented frame {frame}");
    }
    let first = WORLD.load(Relaxed);
    if first == u64::MAX {
        return;
    }
    let after = frame.saturating_sub(first);
    for (i, &f) in frames().iter().enumerate() {
        if f <= after && TAKEN.fetch_or(1 << i, Relaxed) & (1 << i) == 0 {
            DUE.lock().unwrap_or_else(|e| e.into_inner()).push_back(f);
        }
    }
}

/// A draw of the decoration bake (the `n`-th since the start) in `frame`,
/// after its own pass over the textures picked by index (`crate::draw`
/// `early_textures`): the queue then, in a line, and for the first
/// [`bake_walks`] of them (one a frame, none while another audit runs) a
/// snapshot for a walk.
pub(crate) fn bake(heap: &crate::descriptor::Heap, host: &crate::descriptor::host::HostHeap, frame: u64, n: u64) {
    if !on() {
        return;
    }
    static WALKS: AtomicU64 = AtomicU64::new(0);
    static WALKED_IN: AtomicU64 = AtomicU64::new(u64::MAX);
    let [queued, due, waiting, partly, oldest] = host.texture_slots_due(frame, 0).1;
    // (The audit is claimed last: one thread at a time takes it.)
    let walk = WALKED_IN.load(Relaxed) != frame
        && WALKS.load(Relaxed) < bake_walks()
        && RUNNING.compare_exchange(false, true, Relaxed, Relaxed).is_ok();
    if n <= 256 || n.is_multiple_of(64) || walk {
        tracing::info!(
            "{} frame {frame} (presented frame {}), after its pass: {due} slots of textures picked by index due, {waiting} \
             waiting for their memory, {partly} uploaded in part, {queued} in the queue, the oldest due one written {oldest} \
             frames ago; {} uploaded so far{}",
            Label::Bake(n).prefix(),
            crate::queue::presented(),
            crate::draw::HEAP_TEXTURES.load(Relaxed),
            if walk { "; a walk of the textures' memory follows" } else { "" },
        );
    }
    if walk {
        WALKS.fetch_add(1, Relaxed);
        WALKED_IN.store(frame, Relaxed);
        start(heap, host, frame, Label::Bake(n));
    }
}

// The steps.

/// A texture the walk looks at.
struct Item {
    resource: usize,
    rec: Record,
    /// Staged for draws lately.
    staged: bool,
    /// The slots whose view describes memory at its address, with their
    /// descriptor handles.
    slots: Vec<(u32, u64)>,
}

/// A texture whose memory differs from its record.
struct Found {
    resource: usize,
    rec: Record,
    slots: Vec<u32>,
    /// The slot whose view is of it (none: staged for draws only).
    slot: Option<u32>,
    partly: bool,
    zeros: bool,
    /// Memory shown at another address too (`crate::watch::shown_elsewhere`):
    /// the title may write it through that other view.
    aliased: bool,
}

/// What the walk found.
struct Walked {
    label: Label,
    presented: u64,
    frame: u64,
    looked: u64,
    walked: u64,
    bytes: u64,
    staged_only: u64,
    unmapped: u64,
    not_current: u64,
    secs: f64,
    found: Vec<Found>,
}

static DONE: Mutex<Option<Walked>> = Mutex::new(None);

/// The texture pass's turn (`super::tick`): ends an audit whose walk is
/// done, or takes the snapshot of one that is due.
pub(crate) fn tick(heap: &crate::descriptor::Heap, host: &crate::descriptor::host::HostHeap, frame: u64) {
    if !on() {
        return;
    }
    let done = DONE.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(walked) = done {
        finish(host, walked);
        RUNNING.store(false, Relaxed);
    }
    if DUE.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
        || RUNNING.compare_exchange(false, true, Relaxed, Relaxed).is_err()
    {
        return;
    }
    let Some(world) = DUE.lock().unwrap_or_else(|e| e.into_inner()).pop_front() else {
        RUNNING.store(false, Relaxed);
        return;
    };
    start(heap, host, frame, Label::Frame(world));
}

/// Step 1, with [`RUNNING`] set: the snapshot, and the walk's thread.
fn start(heap: &crate::descriptor::Heap, host: &crate::descriptor::host::HostHeap, frame: u64, label: Label) {
    let records = super::records_snapshot();
    let staged = crate::texture::staged_lately(frame, STAGED_FRAMES);
    let starts: Vec<u64> = records.iter().map(|(_, r)| r.at).collect();
    let slots = host.texture_audit_slots(&starts);
    let items: Vec<Item> = records
        .into_iter()
        .zip(slots)
        .map(|((resource, rec), slots)| Item {
            resource,
            rec,
            staged: staged.contains(&resource),
            slots: slots.into_iter().filter_map(|s| Some((s, heap.handle(s)?))).collect(),
        })
        .collect();
    let presented = crate::queue::presented();
    let spawned = std::thread::Builder::new()
        .name("victory-wide-audit".into())
        .spawn(move || walk(items, label, presented, frame));
    if let Err(e) = spawned {
        tracing::warn!(%e, "{} no thread for the walk", label.prefix());
        RUNNING.store(false, Relaxed);
    }
}

/// Whether the descriptor at `handle` views the placed texture `resource`
/// at `at`.
fn views(handle: u64, resource: usize, at: u64) -> bool {
    match crate::descriptor::view(handle) {
        Some(View::Placed(false, desc, view)) => {
            let address = u64::from(view[10]) | u64::from(view[11]) << 32;
            address == at && crate::resource::placed_known(address, &desc).is_some_and(|r| r as usize == resource)
        }
        _ => false,
    }
}

/// Step 2 (its own thread).
fn walk(items: Vec<Item>, label: Label, presented: u64, frame: u64) {
    let began = std::time::Instant::now();
    let mut w = Walked {
        label,
        presented,
        frame,
        looked: items.len() as u64,
        walked: 0,
        bytes: 0,
        staged_only: 0,
        unmapped: 0,
        not_current: 0,
        secs: 0.0,
        found: Vec::new(),
    };
    for item in items {
        let slot = item.slots.iter().find(|&&(_, handle)| views(handle, item.resource, item.rec.at)).map(|&(s, _)| s);
        if slot.is_none() && !item.staged {
            w.not_current += 1;
            continue;
        }
        let (at, len) = (item.rec.at, item.rec.len);
        let mapped = crate::texture::mapped_bytes(at as *const u8, len as usize);
        if mapped == 0 {
            w.unmapped += 1;
            continue;
        }
        let (hash, zeros) = super::memory_hash(at, len);
        w.walked += 1;
        w.bytes += len;
        if slot.is_none() {
            w.staged_only += 1;
        }
        if hash == item.rec.hash {
            continue;
        }
        w.found.push(Found {
            resource: item.resource,
            rec: item.rec,
            slots: item.slots.iter().map(|&(s, _)| s).collect(),
            slot,
            partly: mapped < len,
            zeros,
            aliased: crate::watch::shown_elsewhere(at as usize, len as usize),
        });
    }
    w.secs = began.elapsed().as_secs_f64();
    *DONE.lock().unwrap_or_else(|e| e.into_inner()) = Some(w);
}

/// Step 3: the states, and the lines.
fn finish(host: &crate::descriptor::host::HostHeap, w: Walked) {
    let slots: Vec<u32> = w.found.iter().flat_map(|f| f.slots.iter().copied()).collect();
    let queue: HashMap<u32, u64> = host.texture_audit_queue(&slots);
    let mut states: Vec<(State, &Found)> = Vec::with_capacity(w.found.len());
    for f in &w.found {
        let mut last = f.rec.frame;
        if let Some(now) = super::record_frame(f.resource) {
            last = last.max(now);
        }
        let queued = f.slots.iter().filter_map(|s| queue.get(s).copied()).max();
        if let Some(written) = queued {
            last = last.max(written);
        }
        states.push((classify(w.frame, last, queued.is_some(), f.partly), f));
    }
    let order = [State::OutOfQueue, State::QueuedLong, State::InFlight, State::Partly];
    let [out, long, flight, partly] = order.map(|s| states.iter().filter(|(st, _)| *st == s).count());
    let [a_out, a_long, a_flight, a_partly] = order.map(|s| states.iter().filter(|(st, f)| *st == s && f.aliased).count());
    let partly_out = states
        .iter()
        .filter(|(st, f)| *st == State::Partly && f.slots.iter().all(|s| !queue.contains_key(s)))
        .count();
    let zeros = w.found.iter().filter(|f| f.zeros).count();
    let staged = w.found.iter().filter(|f| f.slot.is_none()).count();
    let prefix = w.label.prefix();
    tracing::info!(
        "{prefix} {} textures with a host copy and a current view walked ({:.0} MB; {} of them read only through tables, \
         staged for draws) in {:.1} s, from the bookkeeping of pass frame {} (presented frame {}); {} differ from the bytes \
         last uploaded: {out} out of the queue, {long} queued more than 8 frames, {flight} in flight (8 frames or fewer since \
         their last event), {partly} partly mapped ({partly_out} of them out of the queue); {staged} of the {} read only \
         through tables, {zeros} with all-zero memory now; shown at another address too: {a_out} out of the queue, {a_long} \
         queued, {a_flight} in flight, {a_partly} partly mapped; {} (clean: 0 out of the queue and 0 queued more than 8 \
         frames); of {} records, {} unmapped now and {} without a current view left out",
        w.walked,
        w.bytes as f64 / f64::from(1u32 << 20),
        w.staged_only,
        w.secs,
        w.frame,
        w.presented,
        w.found.len(),
        w.found.len(),
        if out == 0 && long == 0 { "CLEAN" } else { "NOT CLEAN" },
        w.looked,
        w.unmapped,
        w.not_current,
    );
    // The table: by format and size bucket, a count per state.
    let mut table: BTreeMap<(u32, u8, &'static str), [u64; 4]> = BTreeMap::new();
    for (state, f) in &states {
        let (rank, name) = bucket(f.rec.len);
        let row = table.entry((f.rec.format, rank, name)).or_default();
        row[order.iter().position(|s| s == state).unwrap_or(0)] += 1;
    }
    for ((format, _, size), [o, q, i, p]) in &table {
        tracing::info!(
            "{prefix} by format and size: {}, {size}: {o} out of the queue, {q} queued more than 8 frames, {i} in flight, {p} \
             partly mapped",
            format_name(*format)
        );
    }
    // The 20 largest.
    let mut largest: Vec<&(State, &Found)> = states.iter().collect();
    largest.sort_by(|a, b| b.1.rec.len.cmp(&a.1.rec.len).then(a.0.cmp(&b.0)));
    for (i, (state, f)) in largest.iter().take(20).enumerate() {
        let slot = match f.slot {
            Some(s) => format!("slot {s}"),
            None => "read through tables (staged for draws)".into(),
        };
        tracing::info!(
            "{prefix} largest #{}: {slot}, address {:#x}, {}, {}x{}, {} bytes, {} frames since its upload, {}{}{}{}",
            i + 1,
            f.rec.at,
            format_name(f.rec.format),
            f.rec.width,
            f.rec.height,
            f.rec.len,
            w.frame.saturating_sub(f.rec.frame),
            state.words(),
            if f.rec.partial { ", uploaded in part" } else { "" },
            if f.zeros { ", memory all zeros now" } else { "" },
            if f.aliased { ", shown at another address too" } else { "" },
        );
    }
}
