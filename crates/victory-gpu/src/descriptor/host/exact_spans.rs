//! VIcky (`VICTORY_TEXTURE_EXACT`, `crate::texexact`): the spans' and the
//! queue's side of the exact uploads.
//!
//! **Part A, exact events.** The passes over memory mapped, unmapped or
//! written ([`HostHeap::texture_slots_remapped`]) asked the spans which
//! slots hold the bytes of a range, and the spans looked only at the
//! nearest span that starts before it ([`Spans::slots_in`],
//! [`Spans::slots_holding`]). `Rule::Start` retires an older span only when
//! a newer one starts within its first 4 KiB, so a recycled block's old
//! inner span stayed, and a write under it named that ghost and not the
//! texture now placed around it: the texture kept the block's last bytes,
//! with nothing to tell of the write (Fable's first hole). With the part,
//! [`Spans::slots_exact`] names every span that holds bytes of the range,
//! however far before it starts. What a span holds is the levels its views
//! show ([`crate::texexact::viewed_end`]): a texture viewed from level 3
//! whose next neighbour sits where its larger levels would be (the title
//! packs them so, `Spans`' note) does not hold its neighbour's bytes; one
//! viewed from level 0 holds all of its span. A span is found through an
//! index of the 1 MiB blocks its full size covers.
//!
//! `Rule::Start` itself is left as it is: retiring a span inside a newer one
//! viewed from level 0 would make null the packed neighbours of a texture
//! the title views from level 0 although they sit inside it (run 188's
//! loss, `Rule::Overlap`). With exact events a ghost span costs an upload of
//! its slots at a write, and hides nothing.
//!
//! **Part B** puts each pass upload's print and hash in the slot's checked
//! entry ([`TextureQueue::exact_print`]); **part D** holds a slot whose
//! memory is not written yet ([`TextureQueue::not_ready`]). The audit's
//! snapshot asks here too.

use std::collections::{BTreeSet, HashMap};

use super::{HostHeap, Spans, TextureQueue};
use crate::texexact::{N, Part, count};

/// The blocks of [`Spans`]' index: 1 MiB.
const COVER_SHIFT: u32 = 20;

/// The most (description, first level) pairs whose viewed bytes are kept.
const VIEWED_MOST: usize = 1 << 16;

impl Spans {
    /// A span `[start, end)` was added: in the index of the blocks it covers.
    pub(super) fn cover_add(&mut self, start: u64, end: u64) {
        if !self.exact || end <= start {
            return;
        }
        for block in (start >> COVER_SHIFT)..=((end - 1) >> COVER_SHIFT) {
            self.cover.entry(block).or_default().push(start);
        }
    }

    /// The span `[start, end)` went.
    pub(super) fn cover_remove(&mut self, start: u64, end: u64) {
        if !self.exact || end <= start {
            return;
        }
        for block in (start >> COVER_SHIFT)..=((end - 1) >> COVER_SHIFT) {
            if let Some(starts) = self.cover.get_mut(&block) {
                if let Some(i) = starts.iter().position(|&s| s == start) {
                    starts.swap_remove(i);
                }
                if starts.is_empty() {
                    self.cover.remove(&block);
                }
            }
        }
    }

    /// Slot `index` views the placed texture of resource description `desc`
    /// from level `first`: the bytes from its start its view shows.
    pub(super) fn view_extent(&mut self, index: u32, desc: &[u32; 14], first: u32) {
        if !self.exact {
            return;
        }
        // (Words 1 and 13 are the layout's filler, what the title's stack
        // held: not part of what the texture is.)
        let mut key = *desc;
        key[1] = 0;
        key[13] = 0;
        let bytes = match self.viewed_sizes.get(&(key, first)) {
            Some(&bytes) => bytes,
            None => {
                let bytes = crate::resource::placed_viewed_bytes(desc, first);
                if self.viewed_sizes.len() >= VIEWED_MOST {
                    self.viewed_sizes.clear();
                }
                self.viewed_sizes.insert((key, first), bytes);
                bytes
            }
        };
        self.set_viewed(index, bytes);
    }

    /// Slot `index`'s view shows `bytes` from its texture's start (none: not
    /// known, all of its span).
    pub(super) fn set_viewed(&mut self, index: u32, bytes: Option<u64>) {
        match bytes {
            Some(bytes) => {
                self.viewed.insert(index, bytes);
            }
            None => {
                self.viewed.remove(&index);
            }
        }
    }

    /// Where what the views of the span at `at` (full end `full`, slots
    /// `slots`) show ends: the most any of them shows, all of the span if
    /// one of them is not known.
    fn held_end(&self, at: u64, full: u64, slots: &BTreeSet<u32>) -> u64 {
        let mut most = 0u64;
        for slot in slots {
            match self.viewed.get(slot) {
                Some(&bytes) => most = most.max(bytes),
                None => return full,
            }
        }
        if slots.is_empty() {
            return full;
        }
        at.saturating_add(most).min(full)
    }

    /// Part A: the slots of every span whose shown levels overlap `[start,
    /// end)`, wherever it starts.
    pub(super) fn slots_exact(&self, start: u64, end: u64) -> Vec<u32> {
        if end <= start {
            return Vec::new();
        }
        let mut starts: Vec<u64> = Vec::new();
        for block in (start >> COVER_SHIFT)..=((end - 1) >> COVER_SHIFT) {
            if let Some(s) = self.cover.get(&block) {
                starts.extend_from_slice(s);
            }
        }
        starts.sort_unstable();
        starts.dedup();
        let mut slots = Vec::new();
        for at in starts {
            if at >= end {
                continue;
            }
            let Some((full, held)) = self.by_start.get(&at) else {
                continue;
            };
            if self.held_end(at, *full, held) > start {
                slots.extend(held.iter().copied());
            }
        }
        slots
    }

    /// Part A for the passes over `ranges` (start, end): the slots the exact
    /// spans name, sorted and once each, and counted against what the old
    /// rule (`old`: [`Spans::slots_in`] or [`Spans::slots_holding`]) names.
    pub(super) fn exact_slots(&self, ranges: &[(u64, u64)], old: impl Fn(&Spans, u64, u64) -> Vec<u32>) -> Vec<u32> {
        let mut exact: Vec<u32> = ranges.iter().flat_map(|&(s, e)| self.slots_exact(s, e)).collect();
        exact.sort_unstable();
        exact.dedup();
        let mut before: Vec<u32> = ranges.iter().flat_map(|&(s, e)| old(self, s, e)).collect();
        before.sort_unstable();
        before.dedup();
        let missed = exact.iter().filter(|s| before.binary_search(s).is_err()).count();
        let left = before.iter().filter(|s| exact.binary_search(s).is_err()).count();
        count(N::ANamed, exact.len() as u64);
        count(N::AMissed, missed as u64);
        count(N::ALeft, left as u64);
        exact
    }
}

impl TextureQueue {
    /// Part B: the detiling of slot `index`'s upload (as write `write`) took
    /// bytes of fingerprint `print` and hash `hash` (`failed`: it took
    /// nothing). If the slot waits for a look with a fingerprint of that
    /// upload, the look compares with these from now on, and comes next
    /// frame when they differ from what it had or the read failed. Returns
    /// 0: no such entry; 1: put there; 2: put there, and the print differed.
    pub(super) fn exact_print(&mut self, index: u32, write: u64, print: u64, hash: u64, failed: bool, frame: u64) -> u8 {
        let Some(w) = self.same(index, write) else {
            return 0;
        };
        if w.checked.2 == 0 {
            return 0;
        }
        let print = if failed { crate::texexact::FAILED_PRINT } else { print };
        let differs = w.checked.2 != print;
        w.checked.2 = print;
        w.exact = hash;
        if differs {
            w.due = w.due.min(frame + 1);
        }
        1 + u8::from(differs)
    }

    /// Part D: slot `index`'s memory is not written yet: not handed out
    /// before frame `due` (the time bound is the memory's).
    pub(super) fn not_ready(&mut self, index: u32, write: u64, due: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = w.due.max(due);
        }
    }
}

impl HostHeap {
    /// Part B: the prints posted since the last pass go to the checked
    /// entries of their slots ([`TextureQueue::exact_print`]).
    pub(crate) fn texture_slots_exact(&self, frame: u64) {
        if !crate::texexact::part(Part::B) {
            return;
        }
        let posts = crate::texexact::take_posts();
        if posts.is_empty() {
            return;
        }
        let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        for p in posts {
            match textures.exact_print(p.index, p.write, p.print, p.hash, p.failed, frame) {
                0 => {}
                1 => count(N::BDrained, 1),
                _ => {
                    count(N::BDrained, 1);
                    count(N::BDiffered, 1);
                }
            }
        }
    }

    /// Part D: see [`TextureQueue::not_ready`].
    pub(crate) fn texture_slot_not_ready(&self, index: u32, write: u64, due: u64) {
        crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures).not_ready(index, write, due);
    }

    /// The audit's snapshot: for each address of `starts`, the slots whose
    /// placed texture view starts there.
    pub(crate) fn texture_audit_slots(&self, starts: &[u64]) -> Vec<Vec<u32>> {
        let placed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*self.placed);
        starts.iter().map(|&at| placed.slots_at(at)).collect()
    }

    /// The audit's end: of `slots`, those in the queue now, with the frame
    /// of their last write.
    pub(crate) fn texture_audit_queue(&self, slots: &[u32]) -> HashMap<u32, u64> {
        let textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        slots.iter().filter_map(|&s| textures.slots.get(&s).map(|w| (s, w.written))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::Spans;

    /// Part A: a write inside an old inner span names the texture placed
    /// around it too, however far before it starts; a neighbour viewed from
    /// a level whose bytes end before the write is left out, one viewed
    /// from level 0 (or not known) is named; a span that ends where the
    /// write starts is not; spans that go leave the index.
    #[test]
    fn exact_spans_name_every_texture_whose_viewed_levels_overlap() {
        let mut spans = Spans { exact: true, ..Default::default() };
        let at = 0x7_0000_0000u64;
        // A 4 MiB texture placed over a recycled block where a 64 KiB one
        // was: the newer one starts 2 MiB before it, so Rule::Start keeps
        // the old span.
        assert!(spans.note(1, at + 0x20_0000, at + 0x21_0000, true).is_empty());
        assert!(spans.note(2, at, at + 0x40_0000, true).is_empty());
        let write = (at + 0x20_4000, at + 0x20_8000);
        // The old rule names the ghost alone.
        assert_eq!(spans.slots_holding(write.0, write.1), [1]);
        assert_eq!(spans.slots_in(write.0, write.1), [1]);
        let mut exact = spans.slots_exact(write.0, write.1);
        exact.sort_unstable();
        assert_eq!(exact, [1, 2]);
        // Viewed from a level whose bytes end before the write: left out.
        spans.set_viewed(2, Some(0x1_0000));
        assert_eq!(spans.slots_exact(write.0, write.1), [1]);
        // Viewed from level 0 (all of it): named again.
        spans.set_viewed(2, Some(0x40_0000));
        let mut exact = spans.slots_exact(write.0, write.1);
        exact.sort_unstable();
        assert_eq!(exact, [1, 2]);
        // Ends exactly where a write starts: not named.
        assert!(spans.note(3, at + 0x40_0000, at + 0x41_0000, true).is_empty());
        assert_eq!(spans.slots_exact(at + 0x40_0000, at + 0x40_1000), [3]);
        // Packed neighbours (run 188's layout): with their views' levels
        // known, a write at the second one's start names it alone.
        let p = at + 0x100_0000;
        assert!(spans.note(10, p, p + 0x2c000, true).is_empty());
        assert!(spans.note(11, p + 0x2000, p + 0x2e000, true).is_empty());
        spans.set_viewed(10, Some(0x2000));
        spans.set_viewed(11, Some(0x2c000));
        assert_eq!(spans.slots_exact(p + 0x2000, p + 0x6000), [11]);
        assert_eq!(spans.slots_exact(p + 0x1000, p + 0x3000), [10, 11]);
        // No bytes: none.
        assert!(spans.slots_exact(p + 0x2000, p + 0x2000).is_empty());
        // Spans that go leave the index.
        spans.forget(2);
        assert_eq!(spans.slots_exact(write.0, write.1), [1]);
        spans.forget(1);
        assert!(spans.slots_exact(write.0, write.1).is_empty());
        spans.forget(3);
        spans.forget(10);
        spans.forget(11);
        assert!(spans.cover.is_empty());
    }

    /// Part A's count against the old rule: what the exact spans add, and
    /// what they leave out.
    #[test]
    fn exact_slots_count_what_the_old_rule_missed() {
        let mut spans = Spans { exact: true, ..Default::default() };
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at + 0x20_0000, at + 0x21_0000, true).is_empty());
        assert!(spans.note(2, at, at + 0x40_0000, true).is_empty());
        let before = crate::texexact::tests::counts();
        let named = spans.exact_slots(&[(at + 0x20_4000, at + 0x20_8000)], Spans::slots_holding);
        assert_eq!(named, [1, 2]);
        let after = crate::texexact::tests::counts();
        assert!(after[0] >= before[0] + 2 && after[1] > before[1]);
    }
}
