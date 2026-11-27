//! VIcky: descriptors the title copies (`CopyDescriptorsSimple`,
//! `CopyDescriptors`), and with `=2` the views it makes by calls
//! (`Create*View`), mirrored without waiting for the host heap's slot
//! state (`VICTORY_MIRROR_TRY=1|2|check|check2`; off by default).
//!
//! Profile PF2 (fable-2026-11-27-pf2.md): the title's recording threads
//! waited some 194 ms a frame (about 39 ms each) for `HostHeap::made`, 116
//! ms of it in the mirror of a copied descriptor (`super::mirror`), while
//! the heap sync's mirroring (`HostHeap::mirror_due`) and the pass over
//! buffer views held it. Their own work under it was about 1 ms.
//!
//! With the switch the copy's mirror tries the lock (`HostHeap::mirror_copy`).
//! If it is free, the slot is made at once, as before. If another thread
//! holds it, the slot (title slot and handle) goes on its heap's list
//! ([`Deferred`]), and the first of these to come makes it, through
//! `HostHeap::mirror_due`, the way the sync makes the slots it finds
//! written (`HostHeap::drain_deferred`):
//! - the heap sync of the next submission, ahead of its walk
//!   (`super::sync_held`; every way into the sync goes through there), so
//!   before the lists of that submission run;
//! - the passes over the heap's buffer views and over the textures picked
//!   by index (`crate::draw`), ahead of taking their slots: a slot copied
//!   before the frame's pass is noted for that pass, as when it was made
//!   at the call.
//!
//! A slot is made from its record as it is when it is drained, read in
//! place under the lock, as the sync reads the records it mirrors: if the
//! title wrote the slot again meanwhile, the newer record (the one the
//! console's shaders would read). A slot left twice is made twice, the
//! second time finding it made already.
//!
//! What stays as it was:
//! - Draws. A draw that reads the slot through a table before the drain
//!   makes it itself (`HostHeap::mirror` with `for_draw`), as it makes any
//!   slot whose record changed since it was made; the drain then finds the
//!   slot made from that record and does nothing.
//! - Shaders that pick a slot by index read it when their list runs, after
//!   the sync of the submission that sends it: a copy the title made before
//!   it sent the list is drained by then.
//! - Views made by calls (`Create*View`) are mirrored at the call, but
//!   with `=2`.
//!
//! `VICTORY_MIRROR_TRY=2` (profile PF3, fable-2026-11-27-pf3.md: with `=1`
//! the title's threads still waited some 32 ms a frame for the lock in the
//! mirror of a view made by a call, `super::record`): views made by calls
//! go the same way as copies (`HostHeap::mirror_created`), left on the
//! same list and drained by the same drains. A view made by a call is a
//! record we write in the title's slot at the call, read again at the
//! drain like a copied one; what moves for it is what moves for a copy.
//!
//! What moves, for a slot left on the list: between the call and the
//! drain its host slot shows what it showed before the copy; what its
//! making notes (the queue of textures picked by index, the buffer slots,
//! the slots waiting for a texture's host copy) is noted at the drain, in
//! the frame of the drain (the next submission's, which is the copy's
//! unless a present came between). A buffer view or a placed texture's
//! view copied after the frame's passes waits for the next frame's passes
//! either way.
//!
//! Restrictions, for safety: not used with `VICTORY_HEAP_SYNC=0` (no sync
//! would drain the lists); a heap's list holds at most [`MOST`] slots, past
//! which the copy's mirror waits for the lock, as before; a slot whose
//! record can no longer be read at the drain (the title unmapped that page
//! of its heap) is not made, and counted.
//!
//! `VICTORY_MIRROR_TRY=check`: a slot that finds the lock held is also made
//! at the call, waiting for the lock as before, and then left on the list
//! with the record it was made from; the drain counts the slots whose
//! record differs by then (the title wrote them again in between). The
//! counts go with the heap sync's line ([`part`]). `check2`: the same, with
//! the views made by calls going the way of `=2`.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// What `VICTORY_MIRROR_TRY` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Every copied slot made at the call, waiting for the lock (unset).
    Off,
    /// `1`: made at the call if the lock is free, else left to the next drain.
    Try,
    /// `check`: as `Try`, and made at the call all the same.
    Check,
}

#[cfg(test)]
thread_local! {
    /// Tests: the mode for this thread, whatever the environment says.
    pub(crate) static TEST_MODE: std::cell::Cell<Option<Mode>> = const { std::cell::Cell::new(None) };
    /// Tests: whether views made by calls go the try way too ([`created`]),
    /// whatever the environment says.
    pub(crate) static TEST_CREATED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The mode, read once (`VICTORY_MIRROR_TRY=1|2|check|check2`).
pub(crate) fn mode() -> Mode {
    #[cfg(test)]
    if let Some(mode) = TEST_MODE.get() {
        return mode;
    }
    modes().0
}

/// Whether views the title makes by calls (`Create*View`) go the try way
/// too (`VICTORY_MIRROR_TRY=2` or `check2`; with the mode off, never).
pub(crate) fn created() -> bool {
    #[cfg(test)]
    if let Some(created) = TEST_CREATED.get() {
        return created;
    }
    modes().1
}

/// The mode, and whether views made by calls go its way too, read once.
fn modes() -> (Mode, bool) {
    static MODES: std::sync::OnceLock<(Mode, bool)> = std::sync::OnceLock::new();
    *MODES.get_or_init(|| {
        let (mode, created) = match std::env::var("VICTORY_MIRROR_TRY").as_deref() {
            Ok("1") => (Mode::Try, false),
            Ok("2") => (Mode::Try, true),
            Ok("check") => (Mode::Check, false),
            Ok("check2") => (Mode::Check, true),
            _ => return (Mode::Off, false),
        };
        if !super::sync_on() {
            tracing::warn!(
                "VICTORY_MIRROR_TRY is not used with VICTORY_HEAP_SYNC=0: no heap sync would make the copied slots \
                 left to it"
            );
            return (Mode::Off, false);
        }
        tracing::info!(
            "descriptors the title copies{} are mirrored without waiting for the host heap's lock: one that finds it \
             held is left to the next heap sync or heap pass (VICTORY_MIRROR_TRY={})",
            if created { ", and the views it makes by calls," } else { "" },
            match (mode, created) {
                (Mode::Check, false) => "check: made at the call as well, and compared at the drain",
                (Mode::Check, true) => "check2: made at the call as well, and compared at the drain",
                (_, false) => "1",
                (_, true) => "2",
            }
        );
        (mode, created)
    })
}

/// The most slots a heap's list holds: past it a copied slot is made at
/// the call, waiting for the lock, as before (a list nothing drains for
/// long, with no submission, does not grow without end).
pub(crate) const MOST: usize = 1 << 16;

/// A copied slot left to the next drain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot {
    /// The title slot.
    pub(crate) index: u32,
    /// Its descriptor in title memory.
    pub(crate) handle: u64,
    /// With `check`: the record (`key`) the slot was made from at the
    /// call; else 0.
    pub(crate) key: u64,
}

/// A host heap's copied slots left to the next drain, in the order they
/// were left.
#[derive(Default)]
pub(crate) struct Deferred {
    slots: Mutex<Vec<Slot>>,
    /// Whether `slots` may hold any (set and cleared under its lock): a
    /// drain with nothing left looks at this alone.
    any: AtomicBool,
}

impl Deferred {
    /// Leaves `slot`; false if the list is full ([`MOST`]), and the
    /// caller makes it now.
    pub(crate) fn push(&self, slot: Slot) -> bool {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        if slots.len() >= MOST {
            COUNTS[Count::Full as usize].fetch_add(1, Ordering::Relaxed);
            return false;
        }
        slots.push(slot);
        self.any.store(true, Ordering::Release);
        true
    }

    /// The slots left so far, in order; the list is empty after.
    pub(crate) fn take(&self) -> Vec<Slot> {
        if !self.any.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        self.any.store(false, Ordering::Release);
        std::mem::take(&mut *slots)
    }
}

/// The record key a mirror makes a slot from (`HostHeap::mirror_made`): the
/// view's index and generation; 0 for a record that is not a view of ours.
pub(crate) fn record_key([tag, i, generation]: [u32; 3]) -> u64 {
    if tag != super::VIEW_TAG {
        return 0;
    }
    u64::from(generation) << 32 | u64::from(i)
}

/// Whether the record at `handle` can be read (through the kernel, which
/// fails where a read in place would fault): the title may unmap pages of
/// its heap between the copy and the drain.
pub(crate) fn readable(handle: u64) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut words = [0u32; 3];
    let mut done = 0;
    // SAFETY: the destination is ours; the kernel checks the source.
    unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            handle as *const std::ffi::c_void,
            words.as_mut_ptr().cast(),
            size_of_val(&words),
            &mut done,
        ) != 0
    }
}

/// What the line counts ([`COUNTS`]).
#[derive(Clone, Copy)]
pub(crate) enum Count {
    /// Copied slots made at the call, the lock free.
    MadeAtCall,
    /// Left on a list, the lock held.
    Left,
    /// Made at the call, waiting, because the list was full.
    Full,
    /// Drained by the heap sync.
    DrainedBySync,
    /// Drained by a pass over the heap (`crate::draw`).
    DrainedByPass,
    /// Not made at the drain: the record's page could not be read.
    Gone,
    /// With `check`: slots compared at the drain.
    Compared,
    /// With `check`: of them, those whose record differed from the one
    /// made at the call.
    Differing,
    /// With `=2`: views made by calls, made at the call, the lock free.
    CreatedAtCall,
    /// With `=2`: views made by calls left on a list, the lock held.
    CreatedLeft,
}

/// Since the last line, by [`Count`].
pub(crate) static COUNTS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// Counts `n` of `what`.
pub(crate) fn count(what: Count, n: u64) {
    if n > 0 {
        COUNTS[what as usize].fetch_add(n, Ordering::Relaxed);
    }
}

/// The heap sync line's part (`super::sync_line`), per frame over
/// `frames`; empty with the switch off.
pub(crate) fn part(frames: f64) -> String {
    let n: [u64; 10] = std::array::from_fn(|i| COUNTS[i].swap(0, Ordering::Relaxed));
    let mode = mode();
    if mode == Mode::Off {
        return String::new();
    }
    let per = |what: Count| n[what as usize] as f64 / frames;
    let check = if mode == Mode::Check {
        format!(
            "; check: {:.0} compared at the drain, {:.0} whose record differed from the one made at the call",
            per(Count::Compared),
            per(Count::Differing)
        )
    } else {
        String::new()
    };
    let created = if created() {
        format!(
            "; views made by calls (=2): {:.0} mirrored at the call, {:.0} left as the heap's lock was held",
            per(Count::CreatedAtCall),
            per(Count::CreatedLeft)
        )
    } else {
        String::new()
    };
    format!(
        " (VICTORY_MIRROR_TRY: per frame {:.0} copied slots mirrored at the call, {:.0} left as the heap's lock was \
         held, {:.0} drained by the heap sync, {:.0} by the heap passes, {:.1} not made as their page was gone, {:.1} \
         made waiting as the list was full{created}{check})",
        per(Count::MadeAtCall),
        per(Count::Left),
        per(Count::DrainedBySync),
        per(Count::DrainedByPass),
        per(Count::Gone),
        per(Count::Full),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(index: u32) -> Slot {
        Slot { index, handle: 0x1000 + u64::from(index) * 32, key: u64::from(index) }
    }

    /// The slots come back in the order they were left, once.
    #[test]
    fn a_list_gives_its_slots_back_in_order_once() {
        let list = Deferred::default();
        assert!(list.take().is_empty());
        for i in [5, 3, 9, 3] {
            assert!(list.push(slot(i)));
        }
        assert_eq!(list.take(), [slot(5), slot(3), slot(9), slot(3)]);
        assert!(list.take().is_empty());
        assert!(list.push(slot(1)));
        assert_eq!(list.take(), [slot(1)]);
    }

    /// A full list refuses more; once drained it takes them again.
    #[test]
    fn a_full_list_refuses_until_drained() {
        let list = Deferred::default();
        for i in 0..MOST as u32 {
            assert!(list.push(slot(i)));
        }
        assert!(!list.push(slot(7)));
        assert_eq!(list.take().len(), MOST);
        assert!(list.push(slot(7)));
    }

    /// Slots left from several threads all come back, each thread's in
    /// its own order.
    #[test]
    fn slots_left_from_several_threads_all_come_back() {
        let list = std::sync::Arc::new(Deferred::default());
        let threads: Vec<_> = (0..4u32)
            .map(|t| {
                let list = list.clone();
                std::thread::spawn(move || {
                    for i in 0..1000 {
                        assert!(list.push(slot(t * 1000 + i)));
                    }
                })
            })
            .collect();
        let mut seen = Vec::new();
        for thread in threads {
            seen.extend(list.take());
            thread.join().unwrap();
        }
        seen.extend(list.take());
        assert_eq!(seen.len(), 4000);
        for t in 0..4u32 {
            let own: Vec<u32> = seen.iter().map(|s| s.index).filter(|i| i / 1000 == t).collect();
            assert!(own.windows(2).all(|w| w[0] < w[1]), "thread {t}'s slots out of order");
        }
    }

    /// A record's key is its index and generation; not ours, 0.
    #[test]
    fn record_keys_are_those_a_mirror_makes() {
        assert_eq!(record_key([super::super::VIEW_TAG, 7, 3]), 3 << 32 | 7);
        assert_eq!(record_key([0, 7, 3]), 0);
    }

    /// Memory of ours reads; an address with nothing mapped does not.
    #[test]
    fn readable_says_whether_a_record_reads() {
        let words = [1u32, 2, 3];
        assert!(readable(words.as_ptr() as u64));
        assert!(!readable(0x10));
    }

    /// The line part: empty when off; per frame, with the check's counts
    /// in check mode.
    #[test]
    fn the_line_part_is_per_frame() {
        TEST_MODE.set(Some(Mode::Off));
        let _ = part(1.0);
        assert!(part(1.0).is_empty());
        TEST_MODE.set(Some(Mode::Check));
        TEST_CREATED.set(Some(false));
        count(Count::Left, 10);
        count(Count::Differing, 4);
        let text = part(2.0);
        assert!(text.contains("VICTORY_MIRROR_TRY"), "{text}");
        assert!(text.contains("whose record differed"), "{text}");
        assert!(!text.contains("views made by calls"), "{text}");
        TEST_CREATED.set(Some(true));
        count(Count::CreatedLeft, 6);
        let text = part(2.0);
        assert!(text.contains("views made by calls (=2)"), "{text}");
        TEST_CREATED.set(None);
        TEST_MODE.set(None);
    }
}
