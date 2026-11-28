//! VIcky, stage 2 of the structural fps design (`VICTORY_TABLES_LEFT=1|alt|check`,
//! `super::super::left`, VICKY-RT.md "Structural: tables left to the draws",
//! stage 2): the host heap's part.
//!
//! - The heap sync's rule ([`HostHeap::left_found`]): a new buffer or
//!   constant view record in a title slot with a buffer mark of the last 8
//!   frames is left to the draws (its print not noted, its left bit set);
//!   with `VICTORY_TABLES_LEFT_GRACE=1` a new constant or placed buffer
//!   record elsewhere waits 4 frames ([`HostHeap::mirror_waited_in`]). A
//!   new record also lets go of the slot's hold if that was taken for
//!   another record.
//! - The pass over buffer views leaves the marked slots out
//!   ([`HostHeap::leave_to_draws`]).
//! - The holds ([`HostHeap::hold_copy`]): a draw that reads a buffer view
//!   through a marked slot holds the host copy it reads by that slot.
//! - The flip to off ([`HostHeap::tables_left_walk`]): every slot left and
//!   not noted since, and every record waiting its grace, is mirrored as the
//!   sync would have mirrored it, and the holds go. In ON phases the slots
//!   left whose mark expired are healed in the same way.
//! - `check` ([`HostHeap::check_making`], [`HostHeap::check_read`],
//!   [`HostHeap::check_pass`]): everything is made as without the switch,
//!   and a model notes, per title slot, from when the switch would have left
//!   its host slot showing another view than the title's record.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

use super::super::left::{self, Ck, Copied, K};
use super::{HostHeap, Made, key_of_words, record_print};
use crate::shadow::Shadow;

/// Lets go of a hold's copy: dropped if others hold it too (the table of
/// windows does, nearly always), else retired as `crate::shadow::let_go_idle`
/// retires the copies it lets go of.
fn let_go(copy: Arc<Shadow>) {
    if Arc::strong_count(&copy) > 1 {
        drop(copy);
    } else {
        crate::host::retire(copy);
    }
}

impl HostHeap {
    /// The heap sync found a new record `words` at title slot `index` in
    /// `frame` (its print differs from the one last noted; `copied` held):
    /// whether its print is to be noted, and whether it is to be mirrored now
    /// (stage 2's rules, [`left::decide`]). Records found to wait their grace
    /// go to `waits`, holds let go for another record to `retire`. Without
    /// the switch, or while it does not act: noted and mirrored, as before.
    pub(super) fn left_found(
        &self,
        index: u32,
        words: [u32; 3],
        frame: u64,
        waits: &mut Vec<u32>,
        retire: &mut Vec<Arc<Shadow>>,
    ) -> (bool, bool) {
        if !self.left.on() {
            return (true, true);
        }
        // A hold taken for another record goes: the slot no longer names its
        // copy (without the switch, the sync's note made the slot null).
        if let Some(old) = self.left.holds.release_other(index, key_of_words(words)) {
            retire.push(old);
            left::add2(K::HoldsReleased, 1);
        }
        if !left::acting() || super::sync_tables() {
            self.left.bit_clear(index);
            return (true, true);
        }
        let decision = left::decide(
            || self.buffer_marked_lately(index, frame),
            || super::super::record_is_buffer(words),
            || super::super::record_held_kind(words),
        );
        let check = left::checking();
        match decision {
            Copied::Mirror => {
                self.left.bit_clear(index);
                (true, true)
            }
            Copied::Left => {
                left::add2(K::Left, 1);
                if check {
                    // Made as without the switch, as a making it would not do.
                    self.left.flag(index);
                    left::add_check(Ck::SyncLeft, 1);
                    return (true, true);
                }
                self.left.bit_set(index);
                (false, false)
            }
            Copied::Wait => {
                left::add2(K::Held, 1);
                waits.push(index);
                self.left.bit_clear(index);
                if check {
                    self.left.flag(index);
                    left::add_check(Ck::SyncLeft, 1);
                    return (true, true);
                }
                (true, false)
            }
        }
    }

    /// After the sync's look at a page of `frame` (`copied` let go): the
    /// records found to wait their grace are queued, the holds let go retired.
    pub(super) fn left_after(&self, frame: u64, waits: Vec<u32>, retire: Vec<Arc<Shadow>>) {
        if !waits.is_empty() {
            self.left.waiting.lock().unwrap_or_else(|e| e.into_inner()).push(frame, waits);
        }
        for copy in retire {
            let_go(copy);
        }
    }

    /// A draw read title slot `index`, whose record lies at `handle`, through
    /// a table, and its view reads the host copy `copy`: the slot holds that
    /// copy ([`left::Holds`]) until its mark is [`left::HOLD_FRAMES`] frames
    /// old or the title writes another record there. Only while stage 2
    /// leaves slots (the ON phases), with the hold on.
    pub(crate) fn hold_copy(&self, index: u32, handle: u64, copy: &Arc<Shadow>) {
        if self.left.holds.is_empty() || !left::left_live() || !left::hold_on() {
            return;
        }
        let key = key_of_words(super::super::record_words(handle));
        if key == 0 {
            return;
        }
        let (new, old) = self.left.holds.take(index, key, copy);
        if new {
            left::add2(K::HoldsTaken, 1);
        }
        if let Some(old) = old {
            let_go(old);
        }
    }

    /// The pass over buffer views of `frame` was given `slots`: those with a
    /// buffer mark of the last 8 frames are taken out and returned, each once
    /// (in ON phases they are left to the draws, with `check` made all the
    /// same and counted). Nothing while stage 2 does not act.
    pub(crate) fn leave_to_draws(&self, slots: &mut Vec<u32>, frame: u64) -> Vec<u32> {
        left::add2(K::PassSlots, slots.len() as u64);
        if !self.left.on() || self.buffer_marks.is_empty() || !left::acting() {
            return Vec::new();
        }
        let mut out = Vec::new();
        slots.retain(|&index| {
            let marked = self.buffer_marked_lately(index, frame);
            if marked {
                out.push(index);
            }
            !marked
        });
        out.sort_unstable();
        out.dedup();
        left::add2(K::PassLeft, out.len() as u64);
        out
    }

    /// At each heap sync before its pages (`super::super::walk_heap`; title
    /// slot `i` at `start + i * step`): at a flip of `alt` to off, the sweep
    /// ([`HostHeap::left_flip_off`]); else the records whose grace is over,
    /// and in ON phases once in [`left::HEAL_EVERY`] frames the heal of the
    /// slots left whose mark expired and the holds' expiry. Nothing without
    /// the switch. Returns what a flip's sweep did, if one ran.
    pub(crate) fn tables_left_walk(&self, at: (u64, u64)) -> Option<[usize; 4]> {
        if self.sampler || !self.left.on() {
            return None;
        }
        let frame = crate::frame::current();
        let live = left::left_live();
        if self.left.live_seen.swap(live, Relaxed) && !live {
            return Some(self.left_flip_off(at, frame));
        }
        if !left::acting() {
            return None;
        }
        self.mirror_waited_in(frame, at, false);
        if live && frame >= self.left.healed.load(Relaxed) + left::HEAL_EVERY {
            self.left.healed.store(frame, Relaxed);
            self.left_heal(at, frame);
        }
        None
    }

    /// The flip of `alt` to off in `frame`: every slot left to the draws and
    /// not noted since, and every record waiting its grace, is mirrored as
    /// the sync would have mirrored it ([`HostHeap::left_remirror`]), and
    /// every hold goes, so that the off phase starts from what it would have
    /// had. Returns the slots left, the records waiting, those mirrored, and
    /// the holds let go.
    pub(super) fn left_flip_off(&self, at: (u64, u64), frame: u64) -> [usize; 4] {
        let began = std::time::Instant::now();
        let waited = self.left.waiting.lock().unwrap_or_else(|e| e.into_inner()).due(frame, true);
        let left_slots = self.left.bits_take(|_| true);
        let [looked, mirrored, unread] = self.left_remirror(at, &left_slots, &waited);
        let cleared = self.left.holds.clear();
        let holds = cleared.len();
        left::add2(K::HoldsCleared, holds as u64);
        for copy in cleared {
            let_go(copy);
        }
        tracing::info!(
            "VICTORY_TABLES_LEFT=alt, at the flip to off: {} slots left to the draws and {} records waiting their grace \
             swept in {:.1} ms: {looked} read, {mirrored} mirrored as the heap sync would have mirrored them, {unread} on \
             pages that could not be read; {holds} holds let go",
            left_slots.len(),
            waited.len(),
            began.elapsed().as_secs_f64() * 1000.0,
        );
        [left_slots.len(), waited.len(), mirrored, holds]
    }

    /// In ON phases: the slots left to the draws whose buffer mark expired
    /// (and that no texture mark leaves) are mirrored if their record is not
    /// the one last noted: a slot no draw reads any more, whose page the
    /// title does not write again, would otherwise keep the view it showed
    /// before for good. And the holds whose mark is [`left::HOLD_FRAMES`]
    /// frames old go. Returns the slots mirrored and the holds let go.
    pub(super) fn left_heal(&self, at: (u64, u64), frame: u64) -> [usize; 2] {
        let expired = self
            .left
            .bits_take(|index| !self.buffer_marked_lately(index, frame) && !self.tabled_lately(index, frame));
        let [_, mirrored, _] = if expired.is_empty() { [0; 3] } else { self.left_remirror(at, &expired, &[]) };
        left::add2(K::Healed, mirrored as u64);
        let marks = &self.buffer_marks;
        let gone = self.left.holds.expire(|index| {
            marks
                .get(index as usize)
                .is_some_and(|stamp| left::within(stamp.load(Relaxed), frame, left::HOLD_FRAMES))
        });
        let holds = gone.len();
        left::add2(K::HoldsExpired, holds as u64);
        for copy in gone {
            let_go(copy);
        }
        [mirrored, holds]
    }

    /// Mirrors, as the heap sync mirrors, the title slots `left` (left to the
    /// draws: mirrored if their record is not the one last noted) and
    /// `waited` (their record noted when it began to wait: mirrored as it is
    /// now), both in order. The records are read through the kernel, a page
    /// at a time (a page the title unmapped is skipped, as the sync skips
    /// it), and noted as seen. Returns the records read, the slots mirrored,
    /// and the slots on pages that could not be read.
    pub(super) fn left_remirror(&self, (start, step): (u64, u64), left_slots: &[u32], waited: &[u32]) -> [usize; 3] {
        let mirrored_count = self.mirrored();
        let mut slots: Vec<u32> = left_slots.iter().chain(waited).copied().filter(|&i| i < mirrored_count).collect();
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            return [0; 3];
        }
        let page = crate::watch::PAGE as u64;
        let handle_of = |index: u32| start + u64::from(index) * step;
        // (index, handle, the record's words, its print)
        let mut found: Vec<(u32, u64, [u32; 3], u64)> = Vec::with_capacity(slots.len());
        let (mut bytes, mut unread) = (Vec::new(), 0);
        let mut i = 0;
        while i < slots.len() {
            let first = handle_of(slots[i]);
            let j = i + slots[i..].iter().take_while(|&&index| handle_of(index) / page == first / page).count();
            bytes.resize((handle_of(slots[j - 1]) + step - first) as usize, 0);
            if crate::host::read_into(first as *const u8, &mut bytes) {
                for &index in &slots[i..j] {
                    let at = (handle_of(index) - first) as usize;
                    let record = &bytes[at..at + step as usize];
                    found.push((index, handle_of(index), super::words_of(record), record_print(record) | 1));
                }
            } else {
                unread += j - i;
            }
            i = j;
        }
        let mut due: Vec<(u32, u64)> = Vec::with_capacity(found.len());
        let mut retire = Vec::new();
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if copied.is_empty() {
                copied.resize(mirrored_count as usize, 0);
            }
            for &(index, handle, words, print) in &found {
                let Some(last) = copied.get_mut(index as usize) else {
                    continue;
                };
                if *last == print && waited.binary_search(&index).is_err() {
                    continue;
                }
                *last = print;
                if let Some(old) = self.left.holds.release_other(index, key_of_words(words)) {
                    retire.push(old);
                }
                due.push((index, handle));
            }
        }
        for copy in retire {
            let_go(copy);
        }
        {
            // (The sync's makings, counted by rule as its own.)
            let _making = left::SyncMaking::enter();
            self.mirror_due(&due);
        }
        [found.len(), due.len(), unread]
    }

    /// The records whose grace is over at `frame` (with `all`, every one):
    /// those a draw's table claimed meanwhile are left to the draws (their
    /// print forgotten and their left bit set, as for a record left when
    /// found), the others mirrored now as the sync mirrors (if their page
    /// can still be read). With `check` they were mirrored when found; here
    /// the unclaimed ones are taken as made by the switch. Returns the slots
    /// mirrored.
    pub(super) fn mirror_waited_in(&self, frame: u64, (start, step): (u64, u64), all: bool) -> usize {
        let due = {
            let mut waiting = self.left.waiting.lock().unwrap_or_else(|e| e.into_inner());
            if waiting.len() == 0 {
                return 0;
            }
            waiting.due(frame, all)
        };
        if due.is_empty() {
            return 0;
        }
        let (claimed, unclaimed): (Vec<u32>, Vec<u32>) =
            due.into_iter().partition(|&index| self.buffer_marked_lately(index, frame));
        left::add2(K::GraceClaimed, claimed.len() as u64);
        left::add2(K::GraceMirrored, unclaimed.len() as u64);
        if left::checking() {
            for index in unclaimed {
                self.left.undiverge(index);
            }
            return 0;
        }
        if !claimed.is_empty() {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for &index in &claimed {
                if let Some(last) = copied.get_mut(index as usize) {
                    *last = 0;
                }
                self.left.bit_set(index);
            }
        }
        // The title may have unmapped a page of its heap meanwhile: a page's
        // records are read in place only once it was found readable.
        let page = crate::watch::PAGE as u64;
        let (mut page_read, mut page_gone) = (None, None);
        let mut ready: Vec<(u32, u64)> = Vec::with_capacity(unclaimed.len());
        for index in unclaimed.into_iter().filter(|&index| index < self.mirrored()) {
            let handle = start + u64::from(index) * step;
            let pages = (handle / page, (handle + step - 1) / page);
            if page_gone == Some(pages) {
                continue;
            }
            if page_read != Some(pages) {
                if !super::super::mirror_try::readable(handle) || !super::super::mirror_try::readable(handle + step - 4) {
                    page_gone = Some(pages);
                    continue;
                }
                page_read = Some(pages);
            }
            ready.push((index, handle));
        }
        let _making = left::SyncMaking::enter();
        self.mirror_due(&ready);
        ready.len()
    }

    /// `check`: the making of title slot `index` (`made` held) from a record
    /// of key `key` (`view`: whether it names a live view of ours) begins. A
    /// making the switch would also do leaves the host slot as the switch
    /// would have it. One it would not do (the sync's making of a record the
    /// switch leaves or holds, flagged where it decided; the pass's making of
    /// a slot it leaves, [`left::left_making`]) leaves the switch's host slot
    /// showing what it shows now, which differs from the record unless the
    /// slot was made from it already. Nothing without `check`.
    pub(super) fn check_making(&self, made: &Made, index: u32, key: u64, view: bool, for_draw: bool) {
        if self.left.diverged.is_empty() {
            return;
        }
        let left_making = if for_draw { left::in_left_making() } else { self.left.take_flag(index) };
        if !left_making {
            self.left.undiverge(index);
            return;
        }
        let shows = made.host(index).and_then(|host| made.key_of(host)).unwrap_or(0);
        let same = if view { shows == key } else { shows == 0 };
        if !same {
            self.left.diverge(index, crate::frame::current());
        }
    }

    /// `check`, at a submission: a list read title slot `index` by an index
    /// it computes through a table read in place (no draw makes it): counted,
    /// and a violation if the switch would have left its host slot showing
    /// another view than the title's record, or none.
    pub(crate) fn check_read(&self, index: u32) {
        if self.left.diverged.is_empty() {
            return;
        }
        left::add_check(Ck::Reads, 1);
        let since = self.left.diverged_since(index);
        if since != 0 {
            left::violation(index, since);
        }
    }

    /// `check`: the pass over buffer views of `frame` made `slots`, which the
    /// switch leaves to the draws: counted, with those the switch would have
    /// left showing another view than the title's record or none, and of
    /// those the ones that for over [`left::FRAMES`] frames.
    pub(crate) fn check_pass(&self, slots: &[u32], frame: u64) {
        if self.left.diverged.is_empty() {
            return;
        }
        let now = (frame as u32).wrapping_add(1);
        let (mut stale, mut long) = (0, 0);
        for &index in slots {
            let since = self.left.diverged_since(index);
            if since != 0 {
                stale += 1;
                let age = now.wrapping_sub(since);
                if age > left::FRAMES && age < 1 << 30 {
                    long += 1;
                }
            }
        }
        left::add_check(Ck::PassLeft, slots.len() as u64);
        left::add_check(Ck::PassStale, stale);
        left::add_check(Ck::PassStaleLong, long);
    }

    /// Tests: stage 2's state of the heap.
    #[cfg(test)]
    pub(crate) fn left_state(&self) -> &left::HeapLeft {
        &self.left
    }
}
