//! VIcky: notes by kind (`VICTORY_HEAP_NOTE_KIND=1|alt|check`; off by
//! default), and the heap sync's slots by kind (diagnostics, always on).
//!
//! Profile PF4 (fable-2026-11-27-pf4.md, lever 1): the notes of what a slot
//! holds (`HostHeap::note_buffer_marking`) were some 20 ms a frame of the
//! heap sync's 64, most of it the locks of the queue of textures picked by
//! index and of the spans of placed textures, and their removals. A note
//! takes the fast way of `VICTORY_HEAP_BUFFERS_SKIP_SAME` only when the
//! slot's last note was a plain buffer view's and this one is too; some 36
//! k buffer-view writes a frame took the full way, because the slot's last
//! note was of another kind.
//!
//! Each title slot of a view heap keeps what its last note left it as
//! (`HostHeap::kinds`, always):
//! - [`PLAIN`]: a plain buffer view's: in the set of buffer slots, in
//!   neither the queue of textures nor the spans;
//! - [`BARE`]: no view, or a view that is neither a buffer's nor a
//!   texture's (a constant buffer view, an unordered view of a texture
//!   object): in none of the three sets (a slot never noted is bare);
//! - [`OTHER`]: a texture's (a placed texture, a texture object): maybe in
//!   the queue and the spans.
//!
//! With the switch a note from [`PLAIN`] or [`BARE`] to [`PLAIN`] or
//! [`BARE`] does only what differs: the stride, the slot into or out of the
//! set of buffer slots, and the mark that it was written since the pass
//! over buffer views last looked. The queue of textures and the spans are
//! not locked: the full note would take the slot out of both, and it is in
//! neither. Bare to bare does nothing (the stride is 0 already). A note
//! from or to a texture's view is a full one, as before.
//!
//! What holds this up: only a note puts a slot in the queue or the spans
//! (`TextureQueue::written`, `Spans::note`), and the passes that make
//! slots of the spans due again (`HostHeap::texture_slots_remapped` and the
//! like) take them from the spans, which hold only slots whose last note
//! was a placed texture's. Those passes read the spans and then fill the
//! queue under two locks one after the other, so a slot noted in between
//! may be queued with a buffer view or none; a full note takes it out
//! again, a short one leaves it there until its next full note (the
//! fast way of `VICTORY_HEAP_BUFFERS_SKIP_SAME` leaves it the same way).
//! The texture pass then finds no texture view there.
//!
//! `VICTORY_HEAP_NOTE_KIND=check`: every such note is made the full way,
//! as without the switch, and before it the three sets are looked at for
//! what the short way would have left otherwise (the slot in the queue, in
//! the spans, or in the buffer slots where its kind says it is not, or not
//! where it says it is); the counts go on the heap sync line.
//!
//! `VICTORY_HEAP_NOTE_KIND=alt`: twenty seconds without, twenty with, in
//! turn from the first heap sync; the phase changes at the start of a
//! heap sync, and each change is said in the log with what the syncs took
//! in the phase before.
//!
//! The heap sync's slots by kind (diagnostics, with any switches): per
//! frame, of the slots the heap sync mirrors, those made from the record
//! they hold already, those whose record is no view of ours, and the notes
//! by the kind the slot's last note left and the kind this one gives
//! (plain, bare, other), on the heap sync line (`super::super::sync_line`).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering::Relaxed};

use super::{HostHeap, View, view_stride};

/// A plain buffer view's note ([`of`]).
pub(crate) const PLAIN: u8 = 1;
/// No view, or one neither of a buffer nor of a texture.
pub(crate) const BARE: u8 = 0;
/// A texture's (or a kind not known).
pub(crate) const OTHER: u8 = 2;

/// The kind a full note of `view` leaves its slot as.
pub(crate) fn of(view: Option<&View>) -> u8 {
    let Some(view) = view else {
        return BARE;
    };
    // As `HostHeap::note_buffer_marking` tells a texture's view: one that
    // may go into the queue, and (placed) into the spans.
    let texture = match view {
        View::Placed(false, resource, _) => resource[0] != 1,
        View::Shader(r, _) => crate::resource::buffer_bytes(*r as crate::object::This).is_none(),
        _ => false,
    };
    if texture {
        OTHER
    } else if view_stride(view).is_some() {
        PLAIN
    } else {
        BARE
    }
}

/// What the switch asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    On,
    Alt,
    Check,
}

#[cfg(test)]
thread_local! {
    /// Tests: the mode for this thread, whatever the environment says.
    pub(crate) static TEST_MODE: std::cell::Cell<Option<Mode>> = const { std::cell::Cell::new(None) };
}

/// The mode, read once (`VICTORY_HEAP_NOTE_KIND=1|alt|check`).
pub(crate) fn mode() -> Mode {
    #[cfg(test)]
    if let Some(mode) = TEST_MODE.get() {
        return mode;
    }
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_HEAP_NOTE_KIND").as_deref() {
            Ok("1") => Mode::On,
            Ok("alt") => Mode::Alt,
            Ok("check") => Mode::Check,
            _ => return Mode::Off,
        };
        tracing::info!(
            "notes of heap slots from and to a plain buffer view or no view do only what differs, without the locks of \
             the texture queue and the placed spans (VICTORY_HEAP_NOTE_KIND={})",
            match mode {
                Mode::On => "1",
                Mode::Alt => "alt: twenty seconds without, twenty with, in turn",
                _ => "check: made the full way, and the sets looked at for what the short way would leave",
            }
        );
        mode
    })
}

/// With `alt`, whether the phase now is the one with the switch
/// ([`tick`]).
static PHASE: AtomicBool = AtomicBool::new(false);

/// Whether a note takes the short way now.
fn short() -> bool {
    match mode() {
        Mode::On => true,
        Mode::Alt => PHASE.load(Relaxed),
        Mode::Off | Mode::Check => false,
    }
}

/// At the start of a heap sync (`super::super::sync_held`): with `alt`,
/// the phase for now, each change said with what the syncs of the phase
/// before took.
pub(crate) fn tick() {
    if mode() != Mode::Alt {
        return;
    }
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
    static LAST: AtomicU8 = AtomicU8::new(2);
    if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
        PHASE.store(on, Relaxed);
        let [syncs, us, mirrored, short] = ALT.each_ref().map(|c| c.swap(0, Relaxed));
        tracing::info!(
            "notes of heap slots by kind from now (VICTORY_HEAP_NOTE_KIND=alt): {}; since the change before: {syncs} heap \
             syncs, {:.2} ms a sync, {mirrored} slots mirrored, {short} notes the short way",
            if on { "ON" } else { "off, every note from or to a bare slot the full way as usual" },
            us as f64 / 1000.0 / syncs.max(1) as f64,
        );
    }
}

/// With `alt`, since the last change: heap syncs, their microseconds, the
/// slots they mirrored, and notes made the short way.
static ALT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A heap sync took `us` microseconds and mirrored `mirrored` slots.
pub(crate) fn synced(us: u64, mirrored: u64) {
    if mode() == Mode::Alt {
        ALT[0].fetch_add(1, Relaxed);
        ALT[1].fetch_add(us, Relaxed);
        ALT[2].fetch_add(mirrored, Relaxed);
    }
}

/// Since the last sync line: notes made the short way; with `check`,
/// notes looked at and those where the short way would have left other
/// sets.
static COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

impl HostHeap {
    /// What title slot `index`'s last note left it as ([`OTHER`] past the
    /// slots kept).
    pub(super) fn kind(&self, index: u32) -> u8 {
        self.kinds.get(index as usize).map_or(OTHER, |k| k.load(Relaxed))
    }

    pub(super) fn set_kind(&self, index: u32, kind: u8) {
        if let Some(k) = self.kinds.get(index as usize) {
            k.store(kind, Relaxed);
        }
    }

    /// With the switch, the note of title slot `index` from kind `was` to
    /// `now` made the short way if both are [`PLAIN`] or [`BARE`]: true if
    /// it was (nothing is stale then). `marks` and `sink` as for
    /// `HostHeap::note_buffer_into`. With `check`, the sets are looked at
    /// and false is returned (the caller makes the full note).
    pub(super) fn note_by_kind(
        &self,
        index: u32,
        view: Option<&View>,
        (was, now): (u8, u8),
        marks: bool,
        sink: Option<&mut Vec<u32>>,
    ) -> bool {
        if was == OTHER || now == OTHER || self.kinds.is_empty() {
            return false;
        }
        if !short() {
            if mode() == Mode::Check {
                self.check_kind(index, was);
            }
            return false;
        }
        if (was, now) != (BARE, BARE) {
            let stride = view.and_then(view_stride);
            self.note_stride(index, stride);
            let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
            let listed = if now == PLAIN {
                buffers.insert(index);
                true
            } else {
                buffers.remove(&index)
            };
            if listed {
                self.mark_into(index, marks, sink);
            }
            drop(buffers);
            self.set_noted(index, !self.noted.is_empty() && now == PLAIN);
            self.set_kind(index, now);
        }
        bump(&COUNTS[0], SHORT);
        if mode() == Mode::Alt {
            bump(&ALT[3], SHORT_ALT);
        }
        true
    }

    /// `VICTORY_HEAP_NOTE_KIND=check`: whether the sets hold title slot
    /// `index` as its kind `was` says (in no queue, in no span, a buffer
    /// slot exactly if [`PLAIN`]); counted, the first differences said.
    fn check_kind(&self, index: u32, was: u8) {
        let queued = self
            .textures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .slots
            .contains_key(&index);
        let spanned = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            placed.by_slot.contains_key(&index) || placed.first.contains_key(&index)
        };
        let listed = self
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&index);
        COUNTS[1].fetch_add(1, Relaxed);
        if queued || spanned || listed != (was == PLAIN) {
            COUNTS[2].fetch_add(1, Relaxed);
            static SAID: AtomicU64 = AtomicU64::new(0);
            if SAID.fetch_add(1, Relaxed) < 32 {
                tracing::warn!(
                    "VICTORY_HEAP_NOTE_KIND=check: the short way would differ: title slot {index}, last noted as {}: \
                     queued as a texture {queued}, in the spans {spanned}, a buffer slot {listed}",
                    if was == PLAIN { "a plain buffer view" } else { "bare" }
                );
            }
        }
    }
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// Diagnostics: whether this thread is in a heap sync
    /// (`super::super::sync_held`), whose slots are counted by kind.
    static SYNCING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Diagnostics: this thread is in a heap sync from now (`true`) or not.
#[cfg(feature = "diagnostics")]
pub(crate) fn syncing(on: bool) {
    SYNCING.set(on);
}

thread_local! {
    /// VIcky (`VICTORY_PARALLEL_MIRROR`): while this thread makes slots with
    /// others, its counts are gathered here ([`KINDS`]' places, then the
    /// short notes and those of `alt`'s phase), and added to the shared ones
    /// once it is done ([`gather`]): the threads took the counters' line in
    /// turn for every slot.
    static GATHER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static GATHERED: [std::cell::Cell<u64>; 13] = const { [const { std::cell::Cell::new(0) }; 13] };
}

/// [`GATHERED`]' places of the short notes, and of those in `alt`'s phase.
const SHORT: usize = 11;
const SHORT_ALT: usize = 12;

/// One more in `shared`, or in this thread's place `at` while it gathers.
fn bump(shared: &AtomicU64, at: usize) {
    if GATHER.get() {
        GATHERED.with(|g| g[at].set(g[at].get() + 1));
    } else {
        shared.fetch_add(1, Relaxed);
    }
}

/// VIcky (`VICTORY_PARALLEL_MIRROR`): this thread's counts gathered from now
/// (`true`), or, when it was gathering, added to the shared ones now; returns
/// whether it was gathering.
pub(crate) fn gather(on: bool) -> bool {
    let was = GATHER.replace(on);
    if was && !on {
        GATHERED.with(|g| {
            let take = |at: usize, shared: &AtomicU64| {
                let n = g[at].replace(0);
                if n > 0 {
                    shared.fetch_add(n, Relaxed);
                }
            };
            #[cfg(feature = "diagnostics")]
            for (at, shared) in KINDS.iter().enumerate() {
                take(at, shared);
            }
            take(SHORT, &COUNTS[0]);
            take(SHORT_ALT, &ALT[3]);
        });
    }
    was
}

/// Diagnostics: whether this thread is in a heap sync now ([`syncing`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn syncing_now() -> bool {
    SYNCING.get()
}

/// Diagnostics, by the slot's kind before and after: the heap sync's
/// notes (`was * 3 + now`), then its slots made from the record they hold
/// already, and those whose record is no view of ours.
#[cfg(feature = "diagnostics")]
static KINDS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

/// What else of a heap sync's slot is counted ([`slot`]).
#[cfg(feature = "diagnostics")]
#[derive(Clone, Copy)]
pub(crate) enum Slot {
    /// Made from the record it holds already: nothing done.
    Same = 9,
    /// Its record is no view of ours (never written, or a released
    /// resource's): emptied, if made.
    NotOurs = 10,
}

/// Diagnostics: a note from kind `was` to `now`, counted if it is a heap
/// sync's.
#[cfg(feature = "diagnostics")]
pub(crate) fn noted(was: u8, now: u8) {
    if SYNCING.get() {
        let at = usize::from(was.min(OTHER)) * 3 + usize::from(now.min(OTHER));
        bump(&KINDS[at], at);
    }
}

/// Diagnostics: a heap sync's slot was `what`.
#[cfg(feature = "diagnostics")]
pub(crate) fn slot(what: Slot) {
    if SYNCING.get() {
        bump(&KINDS[what as usize], what as usize);
    }
}

/// The heap sync line's part, per frame over `frames`: the slots by kind
/// (diagnostics) and the switch's counts (with it).
pub(crate) fn part(frames: f64) -> String {
    let mut out = String::new();
    #[cfg(feature = "diagnostics")]
    {
        let n: [u64; 11] = std::array::from_fn(|i| KINDS[i].swap(0, Relaxed));
        let per = |i: usize| n[i] as f64 / frames;
        out.push_str(&format!(
            "; the sync's slots by kind per frame: {:.0} made from the record they hold, {:.0} no view of ours; notes \
             by the slot's last kind -> this one: plain->plain {:.0}, plain->bare {:.0}, plain->other {:.0}, \
             bare->plain {:.0}, bare->bare {:.0}, bare->other {:.0}, other->plain {:.0}, other->bare {:.0}, \
             other->other {:.0}",
            per(Slot::Same as usize),
            per(Slot::NotOurs as usize),
            per(0),
            per(1),
            per(2),
            per(3),
            per(4),
            per(5),
            per(6),
            per(7),
            per(8),
        ));
    }
    let [short, looked, differed] = COUNTS.each_ref().map(|c| c.swap(0, Relaxed));
    match mode() {
        Mode::Off => {}
        Mode::Check => out.push_str(&format!(
            "; VICTORY_HEAP_NOTE_KIND=check (every note, since the last line): {looked} notes from and to plain or bare \
             looked at, {differed} where the short way would have left other sets"
        )),
        _ => out.push_str(&format!(
            "; VICTORY_HEAP_NOTE_KIND: {:.0} notes per frame the short way",
            short as f64 / frames
        )),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kinds of the views a note tells apart.
    #[test]
    fn views_are_told_by_kind() {
        let base = 0x4a_0000_0000u64;
        let mut buffer = [0u32; 14];
        (buffer[0], buffer[4]) = (1, 0x1_0000);
        let mut desc = [0u32; 16];
        (desc[1], desc[2], desc[6], desc[7]) = (1, 5768, 4, 16);
        (desc[10], desc[11]) = (base as u32, (base >> 32) as u32);
        assert_eq!(of(Some(&View::Placed(false, buffer, desc))), PLAIN);
        assert_eq!(of(None), BARE);
        assert_eq!(of(Some(&View::Constant(base, 256))), BARE);
        assert_eq!(of(Some(&View::NullConstant)), BARE);
        let texture = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
        assert_eq!(of(Some(&View::Placed(false, texture, [0; 16]))), OTHER);
        // An unordered view of placed texture memory goes in no set.
        assert_eq!(of(Some(&View::Placed(true, texture, [0; 16]))), BARE);
        // A view of a resource object that is no buffer (none here) is a
        // texture's.
        assert_eq!(of(Some(&View::Shader(0, None))), OTHER);
        // Plain as the fast way of `VICTORY_HEAP_BUFFERS_SKIP_SAME` tells it.
        for view in [View::Placed(false, buffer, desc), View::Constant(base, 256), View::Placed(false, texture, [0; 16])] {
            assert_eq!(of(Some(&view)) == PLAIN, super::super::plain_buffer(&view));
        }
    }

    /// The sets a note leaves (texture queue, spans, buffer slots), the
    /// stride, the marks and the kinds, the short way and the full way,
    /// over notes between every kind and back.
    #[test]
    fn short_notes_leave_the_sets_as_full_notes_do() {
        if crate::host::host().is_none() {
            return;
        }
        let base = 0x4b_0000_0000u64;
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
        let views: Vec<Option<View>> = vec![
            Some(buffer(base, 16)),
            None,
            Some(buffer(base + 0x1_0000, 32)),
            Some(View::Constant(base, 256)),
            Some(buffer(base, 16)),
            Some(buffer(base, 16)),
            None,
            None,
            Some(texture(base + 0x10_0000)),
            Some(View::Constant(base + 0x100, 256)),
            Some(buffer(base + 0x2_0000, 16)),
            Some(texture(base + 0x20_0000)),
            Some(buffer(base + 0x2_0000, 32)),
            Some(View::NullConstant),
            Some(View::Constant(base, 256)),
            Some(buffer(base, 16)),
        ];
        for skip in [false, true] {
            super::super::TEST_FAST.set(Some((0, skip, false)));
            let full = HostHeap::new(false, 64).unwrap();
            let short = HostHeap::new(false, 64).unwrap();
            let look = |heap: &HostHeap, i: u32| {
                let queued = heap.textures.lock().unwrap().slots.contains_key(&i);
                let spans = heap.placed.lock().unwrap();
                let spanned = spans.by_slot.contains_key(&i) || spans.first.contains_key(&i);
                drop(spans);
                let listed = heap.buffers.lock().unwrap().contains(&i);
                let (_, _, data) = heap.strides.as_ref().unwrap();
                // SAFETY: an entry of the mapped table.
                let stride = unsafe { (*data as *const u32).add(i as usize).read_volatile() };
                (queued, spanned, listed, stride, heap.noted_plain(i), heap.kind(i), heap.buffer_changes())
            };
            for (n, step) in (0..views.len() * 3).enumerate() {
                // Three slots, each its own walk through the views.
                let slot = (step % 3) as u32 + 1;
                let view = views[(step * 5 + slot as usize) % views.len()].as_ref();
                TEST_MODE.set(Some(Mode::Off));
                let stale_full = full.note_buffer(slot, view, false);
                TEST_MODE.set(Some(Mode::On));
                let stale_short = short.note_buffer(slot, view, false);
                TEST_MODE.set(None);
                assert_eq!(stale_full, stale_short, "note {n} (skip {skip})");
                for i in 1..4 {
                    assert_eq!(look(&full, i), look(&short, i), "note {n}, slot {i} (skip {skip})");
                }
            }
        }
        super::super::TEST_FAST.set(None);
    }
}
