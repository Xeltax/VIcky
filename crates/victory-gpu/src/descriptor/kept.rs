//! The host resource each slot of a host heap views, held while the slot
//! does (`HostHeap::kept`), and VIcky's split of it in two
//! (`VICTORY_HEAP_KEPT_SPLIT=1`; off by default).
//!
//! Without the switch every slot is behind one lock, as before. Profile
//! PF2 (fable-2026-11-27-pf2.md): the heap sync spun 13.5 ms a frame on
//! that lock in `HostHeap::keep`, which it takes once for each of about
//! 100,000 slots a frame it mirrors, while the title's recording threads
//! take it for every dynamic slot a draw makes a table in
//! (`HostHeap::make_at`) and the pass over buffer views takes it too.
//!
//! The slots the heap mirrors the title's in (below the first dynamic
//! slot: the identity window, the heap map's pool and its slot) and the
//! dynamic slots are different slots: with the switch each part has its own
//! lock ([`Holds`]), and a draw's tables no longer wait for the sync's
//! mirroring or the other way round. The same resources are held in the
//! same slots; a slot's changes are made in the same order, under its
//! part's lock. Lock order as before: the heap's slot state (`made`)
//! before either part; the low part before the high one where both are
//! taken ([`Holds::both`]).
//!
//! With the switch the heap sync's mirroring of a chunk of slots
//! (`HostHeap::mirror_due`, which holds `made` alone for the chunk) also
//! takes the low part's lock once for the chunk, and the lock of the slots
//! marked as written for the pass over buffer views (`HostHeap::changed`)
//! once: the chunk's changes there are noted on its thread and made
//! together, in their order, before `made` is let go. Everything else that
//! changes those slots or marks holds `made` too, so none of them comes
//! between; one that reads a mirrored slot's resource without `made` sees
//! it as before the chunk or after it, as it could have before.

use std::sync::{Mutex, MutexGuard};

/// Whether host heaps made from now on hold their slots' resources in two
/// parts (`VICTORY_HEAP_KEPT_SPLIT=1`), read once.
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_SPLIT.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_HEAP_KEPT_SPLIT").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "host heaps hold their mirrored and their dynamic slots' resources under a lock each, and the heap \
                 sync takes the mirrored part's once a chunk (VICTORY_HEAP_KEPT_SPLIT=1)"
            );
        }
        on
    })
}

#[cfg(test)]
thread_local! {
    /// Tests: the switch for heaps made on this thread, whatever the
    /// environment says.
    pub(crate) static TEST_SPLIT: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// What each slot holds (`T`: a host resource), behind one lock, or with
/// a split behind one for the slots below it and one for the others.
pub(crate) struct Holds<T> {
    low: Mutex<Vec<Option<T>>>,
    high: Mutex<Vec<Option<T>>>,
    /// The first slot of `high`: all the slots without a split.
    split: u32,
    /// Whether there is a split (`high` may still be empty).
    two: bool,
}

/// Both parts of a [`Holds`] locked, the low one first.
pub(crate) struct Both<'a, T> {
    low: MutexGuard<'a, Vec<Option<T>>>,
    high: MutexGuard<'a, Vec<Option<T>>>,
    split: u32,
}

fn lock<T>(part: &Mutex<T>) -> MutexGuard<'_, T> {
    crate::lot4::locks::mutex(crate::lot4::locks::Lock::Kept, part)
}

impl<T> Holds<T> {
    /// `slots` slots holding nothing; with `split`, the ones from there on
    /// behind a lock of their own.
    pub(crate) fn new(slots: u32, split: Option<u32>) -> Self {
        let at = split.map_or(slots, |s| s.min(slots));
        Holds {
            low: Mutex::new((0..at).map(|_| None).collect()),
            high: Mutex::new((at..slots).map(|_| None).collect()),
            split: at,
            two: split.is_some(),
        }
    }

    /// Whether the slots are held in two parts.
    pub(crate) fn split(&self) -> bool {
        self.two
    }

    /// Whether slot `index` is in the low part (all are without a split).
    pub(crate) fn low(&self, index: u32) -> bool {
        index < self.split
    }

    /// Slot `index`'s part and its place there.
    fn part(&self, index: u32) -> (&Mutex<Vec<Option<T>>>, usize) {
        if self.low(index) {
            (&self.low, index as usize)
        } else {
            (&self.high, (index - self.split) as usize)
        }
    }

    /// Slot `index` holds `value` from now on; `old` is given what it held,
    /// under its part's lock. Nothing past the slots.
    pub(crate) fn replace(&self, index: u32, value: Option<T>, old: impl FnOnce(T)) {
        let (part, at) = self.part(index);
        let mut slots = lock(part);
        if let Some(slot) = slots.get_mut(at)
            && let Some(was) = std::mem::replace(slot, value)
        {
            old(was);
        }
    }

    /// [`Holds::replace`] for each of `changes` in order, the ones in the
    /// low part under one lock of it (the others each under its own).
    pub(crate) fn replace_together(&self, changes: Vec<(u32, Option<T>)>, mut old: impl FnMut(T)) {
        if changes.is_empty() {
            return;
        }
        let mut low = lock(&self.low);
        for (index, value) in changes {
            if !self.low(index) {
                self.replace(index, value, &mut old);
                continue;
            }
            if let Some(slot) = low.get_mut(index as usize)
                && let Some(was) = std::mem::replace(slot, value)
            {
                old(was);
            }
        }
    }

    /// `f` given slot `index` (`None` past the slots), under its part's lock.
    pub(crate) fn with<R>(&self, index: u32, f: impl FnOnce(Option<&Option<T>>) -> R) -> R {
        let (part, at) = self.part(index);
        let slots = lock(part);
        f(slots.get(at))
    }

    /// Both parts locked, for several looks.
    pub(crate) fn both(&self) -> Both<'_, T> {
        let low = lock(&self.low);
        let high = lock(&self.high);
        Both { low, high, split: self.split }
    }

    /// The slots.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        lock(&self.low).len() + lock(&self.high).len()
    }
}

impl<T> Both<'_, T> {
    /// Slot `index`; `None` past the slots.
    pub(crate) fn get(&self, index: u32) -> Option<&Option<T>> {
        if index < self.split {
            self.low.get(index as usize)
        } else {
            self.high.get((index - self.split) as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What each slot holds, read one by one.
    fn held(holds: &Holds<u32>) -> Vec<Option<u32>> {
        (0..holds.len() as u32).map(|i| holds.with(i, |s| s.copied().flatten())).collect()
    }

    /// With and without a split, the same changes leave the same slots and
    /// give back the same old values in the same order.
    #[test]
    fn a_split_holds_the_same_as_one_part() {
        let changes = [(0, Some(1)), (5, Some(2)), (9, Some(3)), (5, None), (9, Some(4)), (0, Some(5)), (12, Some(6))];
        let mut seen = Vec::new();
        for split in [None, Some(6), Some(0), Some(10), Some(40)] {
            let holds = Holds::new(10, split);
            let mut olds = Vec::new();
            for &(i, v) in &changes {
                holds.replace(i, v, |old| olds.push((i, old)));
            }
            seen.push((held(&holds), olds));
            let both = holds.both();
            assert_eq!(both.get(9), Some(&Some(4)), "{split:?}");
            assert_eq!(both.get(10), None, "{split:?}: past the slots");
        }
        assert!(seen.windows(2).all(|w| w[0] == w[1]), "{seen:?}");
        assert_eq!(seen[0].0, [Some(5), None, None, None, None, None, None, None, None, Some(4)]);
        assert_eq!(seen[0].1, [(5, 2), (9, 3), (0, 1)]);
    }

    /// Changes made together are those made one by one, in their order.
    #[test]
    fn changes_made_together_are_those_made_one_by_one() {
        let changes = vec![(1, Some(7)), (2, Some(8)), (1, Some(9)), (7, Some(3)), (1, None), (2, Some(4))];
        let (one, together) = (Holds::new(8, Some(5)), Holds::new(8, Some(5)));
        let (mut olds_one, mut olds_together) = (Vec::new(), Vec::new());
        for (i, v) in changes.clone() {
            one.replace(i, v, |old| olds_one.push(old));
        }
        together.replace_together(changes, |old| olds_together.push(old));
        assert_eq!(held(&one), held(&together));
        assert_eq!(olds_one, olds_together);
        assert_eq!(olds_one, [7, 9, 8]);
    }

    /// The point of the split: a slot of the high part changes while the
    /// low part's lock is held elsewhere.
    #[test]
    fn the_high_part_does_not_wait_for_the_low_one() {
        let holds = std::sync::Arc::new(Holds::<u32>::new(8, Some(4)));
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let holder = {
            let holds = holds.clone();
            std::thread::spawn(move || {
                holds.with(1, |_| {
                    held_tx.send(()).unwrap();
                    go_rx.recv().unwrap();
                });
            })
        };
        held_rx.recv().unwrap();
        holds.replace(6, Some(3), |_| {});
        assert_eq!(holds.with(6, |s| s.copied().flatten()), Some(3));
        go_tx.send(()).unwrap();
        holder.join().unwrap();
    }
}
