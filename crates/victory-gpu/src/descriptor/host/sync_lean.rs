//! VIcky: a leaner chunk of the heap sync's mirroring
//! (`VICTORY_HEAP_SYNC_LEAN=1|alt|check`; off by default).
//!
//! Profile PF4 (fable-2026-11-27-pf4.md, lever 2): the heap sync's
//! mirroring (`HostHeap::mirror_due`, with `VICTORY_HEAP_SYNC_THREADS`)
//! costs some 0.66 us a slot, nearly all of it bookkeeping. Of it, per
//! frame: the record read again in place under `made` (5.4 ms, memory the
//! title's recording threads are writing), a fence and a load of the
//! submission counter for each heap map entry written or zeroed (some 5
//! ms), the heap map pool's line looked at for each pool slot taken (2.5
//! ms, diagnostics), and a thread-local borrow for each keep and each mark
//! of a slot (3.35 ms).
//!
//! With the switch the heap sync's walk with `VICTORY_HEAP_SYNC_THREADS`
//! mirrors the slots whose record changed ([`HostHeap::mirror_due_lean`])
//! the same way, in the same order, 64 to a lock of `made` as before, but:
//! - each slot is made from the record the page's copy held, which the
//!   walk compared with the one last seen (`HostHeap::copied_due_with`),
//!   instead of reading it again in place. A record the title wrote again
//!   in between is mirrored by the next sync (its page is marked again);
//!   before, the newer one was mirrored at once and then again;
//! - the chunk's heap map entries (the pool slots taken, given back) are
//!   written at its end, in their order, after one fence (the descriptors
//!   first, then the entries naming them), and the pool slots it gave back
//!   go to the pool after a second fence and one count of the submissions
//!   started (the zeroed entries out before the count). A title slot whose
//!   pool slot is given back has no host slot from then on, as before.
//!   Before a pool slot is taken, what the chunk gave back so far goes to
//!   the pool first, as one by one, and the count of submissions finished
//!   that frees pool slots is read once a chunk (an older count frees no
//!   more than a newer one);
//! - the keeps of the host resources of the chunk's mirrored slots (with
//!   `VICTORY_HEAP_KEPT_SPLIT=1`) and its marks for the pass over buffer
//!   views are gathered in the chunk itself, not through a thread-local,
//!   and made at its end, as `KeptLater` makes them; before the pool is
//!   used up and slots are given back (`HostHeap::reclaim`, which makes its
//!   changes at once), the keeps gathered so far are made;
//! - the switches it reads, the heap map pool's line (diagnostics) and the
//!   first title descriptor that `HostHeap::reclaim` needs, once a chunk.
//!
//! The same slots are made, in the same order, and the host heap and its
//! map end the same (`tests::lean_chunks_make_the_slots_the_usual_way_does`).
//! Other ways into the mirroring (calls, the drains of `VICTORY_MIRROR_TRY`,
//! the walk without `VICTORY_HEAP_SYNC_THREADS`, draws and passes) are as
//! before.
//!
//! `VICTORY_HEAP_SYNC_LEAN=check`: the lean chunks, each slot made from its
//! record read again in place as before; the slots whose record there
//! differs from the page copy's are counted (the heap sync line).
//!
//! `VICTORY_HEAP_SYNC_LEAN=alt`: twenty seconds without, twenty with, in
//! turn from the first heap sync; the phase changes at the start of a
//! heap sync, and each change is said in the log with what the syncs took
//! in the phase before.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering::Relaxed};

use super::*;

/// What the switch asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    On,
    Alt,
    Check,
}

/// The mode, read once (`VICTORY_HEAP_SYNC_LEAN=1|alt|check`).
fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_HEAP_SYNC_LEAN").as_deref() {
            Ok("1") => Mode::On,
            Ok("alt") => Mode::Alt,
            Ok("check") => Mode::Check,
            _ => return Mode::Off,
        };
        tracing::info!(
            "the heap sync mirrors its slots in leaner chunks: each from the record its page's copy held, the heap map's \
             entries, the pool and the keeps once a chunk (VICTORY_HEAP_SYNC_LEAN={}; with VICTORY_HEAP_SYNC_THREADS)",
            match mode {
                Mode::On => "1",
                Mode::Alt => "alt: twenty seconds without, twenty with, in turn",
                _ => "check: each slot from its record read again in place, the records that differ counted",
            }
        );
        mode
    })
}

/// With `alt`, whether the phase now is the one with the switch
/// ([`tick`]).
static PHASE: AtomicBool = AtomicBool::new(false);

/// How the heap sync mirrors now: `None` the usual way; else the lean way,
/// with `true` from the records read in place (`check`).
pub(crate) fn now() -> Option<bool> {
    match mode() {
        Mode::Off => None,
        Mode::On => Some(false),
        Mode::Alt => PHASE.load(Relaxed).then_some(false),
        Mode::Check => Some(true),
    }
}

/// At the start of a heap sync (`super::super::sync_heaps_told`): with
/// `alt`, the phase for now, each change said with what the syncs of the
/// phase before took.
pub(crate) fn tick() {
    if mode() != Mode::Alt {
        return;
    }
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
    static LAST: AtomicU8 = AtomicU8::new(2);
    if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
        PHASE.store(on, Relaxed);
        let [syncs, us, mirrored, lean] = ALT.each_ref().map(|c| c.swap(0, Relaxed));
        tracing::info!(
            "the heap sync mirrors in lean chunks from now (VICTORY_HEAP_SYNC_LEAN=alt): {}; since the change before: \
             {syncs} heap syncs, {:.2} ms a sync, {mirrored} slots mirrored, {lean} of them in lean chunks",
            if on { "ON" } else { "off, each record read again in place as usual" },
            us as f64 / 1000.0 / syncs.max(1) as f64,
        );
    }
}

/// With `alt`, since the last change: heap syncs, their microseconds, the
/// slots they mirrored, and those mirrored in lean chunks.
static ALT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A heap sync took `us` microseconds and mirrored `mirrored` slots.
pub(crate) fn synced(us: u64, mirrored: u64) {
    if mode() == Mode::Alt {
        ALT[0].fetch_add(1, Relaxed);
        ALT[1].fetch_add(us, Relaxed);
        ALT[2].fetch_add(mirrored, Relaxed);
    }
}

/// Since the last sync line: slots mirrored in lean chunks; with `check`,
/// of them those whose record in place differed from the page copy's.
static COUNTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The heap sync line's part, per frame over `frames` (empty without the
/// switch).
pub(crate) fn part(frames: f64) -> String {
    let [lean, differed] = COUNTS.each_ref().map(|c| c.swap(0, Relaxed));
    match mode() {
        Mode::Off => String::new(),
        Mode::Check => format!(
            "; VICTORY_HEAP_SYNC_LEAN=check: {:.0} slots per frame in lean chunks, {:.1} whose record in place differed \
             from the page copy's",
            lean as f64 / frames,
            differed as f64 / frames
        ),
        _ => format!("; VICTORY_HEAP_SYNC_LEAN: {:.0} slots per frame in lean chunks", lean as f64 / frames),
    }
}

/// What a lean chunk leaves for its end, and what it reads once.
pub(super) struct Chunk {
    /// `VICTORY_HEAP_KEPT_SPLIT`: the keeps of mirrored slots and the
    /// marks are gathered here, as `KeptLater` gathers them.
    split: bool,
    /// `made` held alone (false: by the round of `VICTORY_PARALLEL_MIRROR`
    /// for this chunk and the round's others, [`super::par_mirror`]: no
    /// pool slot of another slot is given back, and a stale or pool slot
    /// that comes to hold nothing lets go at once, `par_mirror::keep_now`).
    alone: bool,
    keeps: Vec<(u32, Option<ID3D12Resource>)>,
    marks: Vec<u32>,
    /// Heap map entries to write (entry, value), in order.
    entries: Vec<(u32, u32)>,
    /// Pool slots given back, for the pool once the entries are out.
    given: Vec<u32>,
    /// The submissions finished, read at the chunk's first pool slot taken.
    finished: Option<u64>,
    /// Whether a pool slot was taken (the pool's line).
    taken: bool,
    keep_views: bool,
    skip_same: bool,
}

impl Chunk {
    /// A chunk of `heap`'s, `made` held `alone` or for the round's chunks.
    pub(super) fn new(heap: &HostHeap, alone: bool) -> Self {
        let split = heap.kept.split();
        Chunk {
            split,
            alone,
            keeps: Vec::with_capacity(if split { MIRROR_CHUNK } else { 0 }),
            marks: Vec::with_capacity(if split { MIRROR_CHUNK } else { 0 }),
            entries: Vec::new(),
            given: Vec::new(),
            finished: None,
            taken: false,
            keep_views: keep_views(),
            skip_same: skip_same(),
        }
    }

    /// [`HostHeap::keep`] of host slot `index`.
    fn keep(&mut self, heap: &HostHeap, index: u32, resource: Option<ID3D12Resource>) {
        let _keep = crate::draw::heap_pass::parts::enter(crate::draw::heap_pass::parts::Part::Keep);
        if self.split && heap.kept.low(index) && (self.alone || resource.is_some() || !super::par_mirror::keep_now(heap, index)) {
            self.keeps.push((index, resource));
        } else {
            heap.kept.replace(index, resource, retire_held);
        }
    }

    /// The marks' sink for a note.
    fn marks(&mut self) -> Option<&mut Vec<u32>> {
        self.split.then_some(&mut self.marks)
    }

    /// The heap map entries written, after one fence (the descriptors
    /// first); then the pool slots given back to the pool, after another
    /// and a count of the submissions started (the zeroed entries out
    /// before it).
    fn flush_map(&mut self, heap: &HostHeap, made: &Made) {
        let (Some(map), Some(m)) = (made.map.as_ref(), &heap.map) else {
            self.entries.clear();
            self.given.clear();
            return;
        };
        if !self.entries.is_empty() {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            for (i, entry) in self.entries.drain(..) {
                // SAFETY: entry `i` of the mapped heap map (as `publish`
                // and `unmap` found it).
                unsafe { (m.at as *mut u32).add(i as usize).write_volatile(entry) };
            }
        }
        if !self.given.is_empty() {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            map.quarantine(self.given.drain(..), crate::host::submissions().0);
        }
    }

    /// The keeps gathered so far, made.
    fn flush_keeps(&mut self, heap: &HostHeap) {
        if !self.keeps.is_empty() {
            heap.kept.replace_together(std::mem::take(&mut self.keeps), retire_held);
        }
    }

    /// The chunk's end, `made` still held.
    pub(super) fn end(&mut self, heap: &HostHeap, made: &Made) {
        self.flush_map(heap, made);
        self.flush_keeps(heap);
        if !self.marks.is_empty() {
            let mut changed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Changed, &heap.changed);
            changed.extend(self.marks.drain(..));
        }
        if std::mem::take(&mut self.taken) {
            #[cfg(feature = "diagnostics")]
            note_pool(made);
        }
        self.finished = None;
    }
}

impl HostHeap {
    /// [`HostHeap::mirror_due`] for `due` (title slot, handle), each from
    /// its record in `records` as the page's copy held it (or, with
    /// `check`, as it is in place now, the ones that differ counted).
    pub(crate) fn mirror_due_lean(&self, due: &[(u32, u64)], records: &[[u32; 3]], check: bool) {
        let mut chunk = Chunk::new(self, true);
        let mirrored = self.mirrored();
        // VIcky (`VICTORY_MADE_TURNS`): asked once a call.
        let turns = super::turns_now();
        for (slots, words) in due.chunks(MIRROR_CHUNK).zip(records.chunks(MIRROR_CHUNK)) {
            RETIRE_LATER.set(Some(Vec::new()));
            {
                let mut made = held_alone(&self.made);
                self.note_title_start(slots);
                for (&(index, handle), &record) in slots.iter().zip(words) {
                    // VIcky (`VICTORY_MADE_TURNS`): a thread waits for
                    // `made`: the chunk ends here and goes on once that
                    // thread had it.
                    if self.turn_wanted(turns) {
                        chunk.end(self, &made);
                        made = self.turn_over(made);
                    }
                    if index >= mirrored {
                        // (Counted and said there, as before.)
                        let _ = self.record_at(index, handle, false);
                        continue;
                    }
                    let record = if check { compared(handle, record) } else { record };
                    self.mirror_lean(&made, (index, handle), record, &mut chunk);
                }
                counted(slots.len());
                chunk.end(self, &made);
                drop(made);
            }
            if let Some(held) = RETIRE_LATER.take().filter(|held| !held.is_empty()) {
                crate::host::retire(held);
            }
        }
    }

    /// Where the title's descriptors start, for `reclaim`, as
    /// [`HostHeap::record_at`] notes it: from the chunk's first slot past
    /// the identity window (every slot gives the same).
    pub(super) fn note_title_start(&self, slots: &[(u32, u64)]) {
        if self.map.is_none() || self.title_start.load(Relaxed) != 0 {
            return;
        }
        let mirrored = self.mirrored();
        if let Some(&(index, handle)) = slots.iter().find(|&&(index, _)| index >= self.count && index < mirrored) {
            let start = handle.wrapping_sub(u64::from(index) * TITLE_SLOT_BYTES);
            let _ = self.title_start.compare_exchange(0, start, Relaxed, Relaxed);
        }
    }

    /// [`HostHeap::mirror_made`] of the heap sync (`made` held alone, not
    /// for a draw, marking) for title slot `index` at `handle`, which held
    /// `record`; with the chunk's ways.
    fn mirror_lean(&self, made: &Made, slot: (u32, u64), [tag, i, generation]: [u32; 3], chunk: &mut Chunk) {
        use crate::draw::heap_pass::parts::{Part, enter};
        let key = u64::from(generation) << 32 | u64::from(i);
        let view = {
            let _view = enter(Part::View);
            (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten()
        };
        self.mirror_lean_view(made, slot, key, view, chunk);
    }

    /// [`HostHeap::mirror_lean`] with the record's view (`key` its record)
    /// looked up by the caller (`VICTORY_PARALLEL_MIRROR`, a chunk's at once).
    pub(super) fn mirror_lean_view(&self, made: &Made, (index, handle): (u32, u64), key: u64, view: Option<View>, chunk: &mut Chunk) {
        use crate::draw::heap_pass::parts::{Count, Part, count, enter};
        // VIcky (`VICTORY_LOT4_PROBE`, P2): the sync's slots by kind.
        #[cfg(feature = "diagnostics")]
        crate::lot4::sync_slot(view.as_ref());
        let Some(view) = view else {
            count(Count::NotOurs);
            #[cfg(feature = "diagnostics")]
            note_kind::slot(note_kind::Slot::NotOurs);
            self.empty_lean(made, index, chunk);
            return;
        };
        let host = {
            let _map = enter(Part::Map);
            self.host_slot_lean(made, index, chunk)
        };
        let Some((host, taken)) = host else {
            count(Count::NoHost);
            for stale in self.note_buffer_into(index, Some(&view), false, true, chunk.marks()) {
                if stale != index {
                    self.unmake_lean(made, stale, chunk);
                }
            }
            return;
        };
        if taken {
            count(Count::Taken);
        }
        if !taken && made.key(host) == key {
            count(Count::Same);
            #[cfg(feature = "diagnostics")]
            note_kind::slot(note_kind::Slot::Same);
            return;
        }
        let stale = {
            let _note = enter(Part::Note);
            self.note_buffer_into(index, Some(&view), false, true, chunk.marks())
        };
        for stale in stale {
            if stale != index {
                self.unmake_lean(made, stale, chunk);
            }
        }
        let waiting = view.waiting_texture();
        if let Some(texture) = waiting {
            super::super::wait_for(texture, handle);
        }
        let later = view.is_buffer() || waiting.is_some();
        let old = made.key(host);
        if later
            && chunk.keep_views
            && (!chunk.skip_same || placed_texture_view(&view))
            && old != 0
            && view_at(old as u32, (old >> 32) as u32).is_some_and(|old| same_placed_texture(&old, &view))
        {
            KEPT_VIEWS.fetch_add(1, Relaxed);
            return;
        }
        made.set_key(host, if later { 0 } else { key });
        let kept = if later {
            None
        } else {
            self.make_slot(made, true, &view, key, host, &mut Lookups::each())
        };
        if kept.is_none() && !later {
            count(Count::Unmade);
        }
        if kept.is_none() && !self.sampler {
            let _create = enter(Part::Create);
            null_for(&view, self.cpu(host));
        }
        chunk.keep(self, host, kept.flatten());
        if taken {
            let _map = enter(Part::Map);
            self.publish_lean(made, index, chunk);
        }
    }

    /// [`HostHeap::host_slot`] (held alone) with the chunk's pool slots
    /// given back in the pool before one is taken.
    fn host_slot_lean(&self, made: &Made, title: u32, chunk: &mut Chunk) -> Option<(u32, bool)> {
        if title < self.count {
            return Some((title, false));
        }
        if title >= self.title_count {
            return None;
        }
        crate::draw::heap_pass::parts::count(crate::draw::heap_pass::parts::Count::Mapped);
        let map = made.map.as_ref()?;
        if let Some(host) = map.host(title) {
            return Some((host, false));
        }
        chunk.flush_map(self, made);
        let finished = *chunk.finished.get_or_insert_with(|| crate::host::submissions().1);
        if let Some(host) = map.assign(title, finished) {
            return Some((host, true));
        }
        // VIcky (`VICTORY_PARALLEL_MIRROR`): no other slot's pool slot given
        // back on a round's thread; made once the round's threads are done.
        if !chunk.alone {
            ROOM_WANTED.with_borrow_mut(|wanted| wanted.push(title));
            return None;
        }
        // `reclaim` makes its changes at once: the chunk's so far first.
        chunk.flush_keeps(self);
        if self.reclaim(made) > 0
            && let Some(host) = map.assign(title, crate::host::submissions().1)
        {
            return Some((host, true));
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(title, "heap map pool used up: title slots past the window read the null slot");
        });
        None
    }

    /// [`HostHeap::publish`], the entry left for the chunk's end.
    fn publish_lean(&self, made: &Made, title: u32, chunk: &mut Chunk) {
        let (Some(map), Some(_)) = (&made.map, &self.map) else {
            return;
        };
        let Some(i) = title.checked_sub(map.window()).filter(|&i| i < map.len()) else {
            return;
        };
        chunk.entries.push((i, map.entry(title)));
        chunk.taken = true;
    }

    /// [`HostHeap::unmap`]: the title slot has no pool slot from now on;
    /// its entry is zeroed and the pool slot given back at the chunk's end.
    fn unmap_lean(&self, made: &Made, title: u32, chunk: &mut Chunk) {
        let (Some(map), Some(_)) = (made.map.as_ref(), &self.map) else {
            return;
        };
        let Some(host) = map.host(title).filter(|_| title >= map.window()) else {
            return;
        };
        made.set_key(host, 0);
        chunk.entries.push((title - map.window(), 0));
        if let Some(host) = map.unset(title) {
            chunk.given.push(host);
        }
    }

    /// [`HostHeap::empty_slot`] of the heap sync (not for a draw, marking).
    fn empty_lean(&self, made: &Made, title: u32, chunk: &mut Chunk) {
        use crate::draw::heap_pass::parts::{Part, enter};
        if let Some(host) = made.host(title)
            && made.key(host) != 0
        {
            made.set_key(host, 0);
            {
                let _create = enter(Part::Create);
                if self.sampler {
                    self.null_sampler(host);
                } else {
                    null_view(self.cpu(host));
                }
                chunk.keep(self, host, None);
            }
            let _note = enter(Part::Note);
            let _ = self.note_buffer_into(title, None, false, true, chunk.marks());
        }
        let _map = enter(Part::Map);
        self.unmap_lean(made, title, chunk);
    }

    /// [`HostHeap::unmake`], the keep the chunk's.
    fn unmake_lean(&self, made: &Made, index: u32, chunk: &mut Chunk) {
        // VIcky (`VICTORY_PARALLEL_MIRROR`): as `HostHeap::unmake`.
        if self.par_stale(index) {
            return;
        }
        let Some(host) = made.host(index) else {
            return;
        };
        let Some(key) = made.key_of(host) else {
            return;
        };
        made.set_key(host, 0);
        if key == 0 {
            return;
        }
        match view_at(key as u32, (key >> 32) as u32) {
            Some(view) => null_for(&view, self.cpu(host)),
            None => null_view(self.cpu(host)),
        }
        chunk.keep(self, host, None);
    }
}

/// `n` slots mirrored in lean chunks (also by `VICTORY_PARALLEL_MIRROR`'s
/// rounds, [`super::par_mirror`]).
pub(super) fn counted(n: usize) {
    COUNTS[0].fetch_add(n as u64, Relaxed);
    if mode() == Mode::Alt {
        ALT[3].fetch_add(n as u64, Relaxed);
    }
}

/// `check`: the record at `handle` now, read in place as before; counted
/// if it is not `copied`, the one the page's copy held.
pub(super) fn compared(handle: u64, copied: [u32; 3]) -> [u32; 3] {
    // SAFETY: the handle is a descriptor slot in title memory (as
    // `HostHeap::record_at` reads it).
    let here = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if here != copied {
        COUNTS[1].fetch_add(1, Relaxed);
    }
    here
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state a heap's mirroring leaves: per host slot its record and
    /// whether it holds a resource; per title slot its host slot, its kind
    /// and whether its last note was a plain buffer view's; the heap map's
    /// entries as shaders read them; the marked and the buffer slots, the
    /// texture queue and the spans.
    #[allow(clippy::type_complexity)]
    fn state(
        heap: &HostHeap,
        titles: u32,
    ) -> (Vec<u64>, Vec<bool>, Vec<Option<u32>>, Vec<(u8, bool)>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let made = heap.made.read().unwrap();
        let keys: Vec<u64> = made.keys.iter().map(|k| k.load(Relaxed)).collect();
        let kept = (0..keys.len() as u32).map(|h| heap.kept.with(h, |slot| slot.is_some_and(Option::is_some))).collect();
        let hosts = (0..titles).map(|i| made.host(i)).collect();
        let entries = match (&made.map, &heap.map) {
            // SAFETY: the entries of the mapped heap map.
            (Some(map), Some(m)) => (0..map.len()).map(|i| unsafe { (m.at as *const u32).add(i as usize).read_volatile() }).collect(),
            _ => Vec::new(),
        };
        drop(made);
        let kinds = (0..titles).map(|i| (heap.kind(i), heap.noted_plain(i))).collect();
        let mut changed = heap.buffer_changes();
        changed.sort_unstable();
        let mut queued: Vec<u32> = heap.textures.lock().unwrap().slots.keys().copied().collect();
        let mut spanned: Vec<u32> = heap.placed.lock().unwrap().by_slot.keys().copied().collect();
        // (In the maps' order, which is not the slots'.)
        queued.sort_unstable();
        spanned.sort_unstable();
        (keys, kept, hosts, kinds, entries, changed, heap.buffer_slots(), queued, spanned)
    }

    /// The same records mirrored by the heap sync's chunks the usual way
    /// and the lean way, with and without a heap map (a pool of 8 slots for
    /// 48 title slots, used up and given back), with and without the split
    /// keeps: after every round the two heaps are in the same state.
    #[test]
    fn lean_chunks_make_the_slots_the_usual_way_does() {
        if crate::host::host().is_none() {
            return;
        }
        const TITLES: u32 = 64;
        let base = 0x4c_0000_0000u64;
        let buffer = |at: u64, stride: u32| {
            let mut resource = [0u32; 14];
            (resource[0], resource[4]) = (1, 0x1_0000);
            let mut desc = [0u32; 16];
            (desc[1], desc[2], desc[6], desc[7]) = (1, 5768, 4, stride);
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let texture = |at: u64| {
            let resource = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[2], desc[5]) = (98, 4, 5768, 1);
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let views = [
            buffer(base, 16),
            View::NullConstant,
            buffer(base + 0x1_0000, 32),
            View::Constant(base, 256),
            texture(base + 0x10_0000),
            View::NullConstant,
            buffer(base + 0x2_0000, 16),
            View::Constant(base + 0x100, 256),
            texture(base + 0x20_0000),
        ];
        // Each view's record; a slot given none of them holds nothing of ours.
        let records: Vec<[u32; 3]> = views
            .iter()
            .map(|view| {
                let [i, generation] = super::super::super::words_for(view.clone());
                [VIEW_TAG, i, generation]
            })
            .collect();
        for (map, split) in [(None, false), (None, true), (Some((16, 24)), false), (Some((16, 24)), true)] {
            // Other tests submit to the same host meanwhile, and a heap map
            // hands a pool slot given back out again by the counts of
            // submissions started and finished (`crate::host::submissions`):
            // the two heaps agree only if those counts held still while both
            // were made. A case they moved in is made again (20 tries at
            // most; the last is judged whatever happened).
            for tries in 1..=20 {
                let counts = crate::host::submissions();
                super::super::TEST_FAST.set(Some((0, true, false)));
                super::super::super::kept::TEST_SPLIT.set(Some(split));
                super::super::TEST_MAP.set(map);
                let usual = HostHeap::new(false, TITLES).unwrap();
                let lean = HostHeap::new(false, TITLES).unwrap();
                super::super::TEST_MAP.set(None);
                // The title heap's memory: 32 bytes a slot.
                let mut memory = vec![[0u32; 8]; TITLES as usize];
                let start = memory.as_mut_ptr() as u64;
                let mut rounds = Vec::new();
                for round in 0..8u32 {
                    for s in 0..TITLES {
                        let pick = ((s * 7 + round * 13) % (views.len() as u32 + 3)) as usize;
                        let record = records.get(pick).copied().unwrap_or([0; 3]);
                        memory[s as usize][..3].copy_from_slice(&record);
                    }
                    let due: Vec<(u32, u64)> = (0..TITLES)
                        .filter(|s| (s + round) % 3 != 0)
                        .map(|s| (s, start + u64::from(s) * TITLE_SLOT_BYTES))
                        .collect();
                    let copied: Vec<[u32; 3]> = due
                        .iter()
                        .map(|&(s, _)| {
                            let w = memory[s as usize];
                            [w[0], w[1], w[2]]
                        })
                        .collect();
                    usual.mirror_due(&due);
                    lean.mirror_due_lean(&due, &copied, round % 2 == 1);
                    rounds.push((round, state(&usual, TITLES), state(&lean, TITLES)));
                }
                drop(memory);
                if crate::host::submissions() != counts && tries < 20 {
                    continue;
                }
                for (round, usual, lean) in rounds {
                    assert_eq!(usual, lean, "round {round}, map {map:?}, split {split}, try {tries}");
                }
                break;
            }
        }
        super::super::TEST_FAST.set(None);
        super::super::super::kept::TEST_SPLIT.set(None);
    }
}
