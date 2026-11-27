//! VIcky (`VICTORY_HEAP_SYNC_NEED=check`, [`super::need`]): the check that
//! a join skipped leaves no list with a page of a heap it may read not yet
//! mirrored.
//!
//! What the skip relies on: every heap page written before the lists'
//! stamp has been taken by a sync that has ended (it then mirrored the
//! records changed on it). The check looks at that directly, apart from the
//! stamps' argument:
//! - at each stamp a list may carry (the title's hand-over, a wait passed,
//!   the title's present: `super::handed`, `super::wait_passed`,
//!   `super::presenting`), the pages of the heaps that a sync's take would
//!   give now are noted first ([`stamp_noted`]): the kernel's dirty bits,
//!   read without clearing them, and the journal's pages the heap has not
//!   taken (`crate::watch::Tracked::pending`);
//! - each take of a heap's pages by a sync is noted with the sync's asking
//!   stamp ([`took`], in `super::super::sync_held`);
//! - at a join it would skip ([`verify`]), each page noted at the lists'
//!   stamp must have been taken, after the note, by a sync that has ended
//!   (asking stamp at most the covered stamp: syncs end in turn on one
//!   lane). A page taken only by a sync still running, or by none, is a
//!   violation: counted, the first ones logged, and the join waits whole.
//!
//! The notes and the takes are ordered by one lock ([`LEDGER`]), taken
//! around both, so a take is wholly before or wholly after a note (lock
//! order: the list of heaps, the ledger, the journal). Heaps of view memory
//! (the kernel does not watch them) are counted and left unchecked. Only
//! with the check: without it nothing here takes a lock or asks anything.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::need::{Mode, mode};

/// Notes kept (some 20-40 stamps a frame) and takes kept (a few heaps a
/// sync, some 17 syncs a frame): several frames' worth either way.
const NOTES: usize = 1024;
const TAKES: usize = 4096;

/// The pages a sync's take would give at one stamp.
struct Note {
    stamp: u64,
    /// Its place among notes and takes.
    event: u64,
    /// Heap pages (64 KiB page numbers), in order.
    pages: Vec<usize>,
    /// Heaps the kernel does not watch (view memory): not checked.
    views: u32,
}

/// One heap's take by a sync.
struct Take {
    /// The sync's asking stamp (0: a sync not started by `super::need`).
    sync: u64,
    event: u64,
    pages: Vec<usize>,
}

struct Ledger {
    event: u64,
    notes: VecDeque<Note>,
    takes: VecDeque<Take>,
    /// The event of the last take let go (a note before it may have lost
    /// takes after it: not checked).
    dropped: u64,
}

static LEDGER: Mutex<Ledger> = Mutex::new(Ledger { event: 0, notes: VecDeque::new(), takes: VecDeque::new(), dropped: 0 });

fn ledger() -> std::sync::MutexGuard<'static, Ledger> {
    LEDGER.lock().unwrap_or_else(|e| e.into_inner())
}

fn checking() -> bool {
    mode() == Mode::Check
}

thread_local! {
    /// The asking stamp of the sync this thread runs (the lane's).
    static SYNC: Cell<u64> = const { Cell::new(0) };
}

/// While it lives, this thread runs the sync that asked at a stamp.
pub(super) struct Syncing(u64);

/// This thread runs the sync that asked at `stamp` (`super::need`'s give).
pub(super) fn syncing(stamp: u64) -> Syncing {
    Syncing(SYNC.replace(stamp))
}

impl Drop for Syncing {
    fn drop(&mut self) {
        SYNC.set(self.0);
    }
}

/// A stamp (`super::stamp`); with the check, taken after the heaps' pages
/// a take would give now are noted.
pub(crate) fn stamp_noted() -> u64 {
    if !checking() {
        return super::stamp();
    }
    // Lock order: the list of heaps, the ledger, the journal (as a sync's
    // takes, which hold the list of heaps all along).
    let heaps = super::super::SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let mut ledger = ledger();
    let (pages, views) = pending(&heaps);
    drop(heaps);
    let stamp = super::stamp();
    ledger.event += 1;
    let event = ledger.event;
    note(&mut ledger, Note { stamp, event, pages, views });
    stamp
}

/// Tests: a note of `pages` at `stamp`, as [`stamp_noted`] makes one.
#[cfg(test)]
pub(super) fn note_for_test(stamp: u64, pages: Vec<usize>) {
    let mut ledger = ledger();
    ledger.event += 1;
    let event = ledger.event;
    note(&mut ledger, Note { stamp, event, pages, views: 0 });
}

fn note(ledger: &mut Ledger, note: Note) {
    if ledger.notes.len() == NOTES {
        ledger.notes.pop_front();
    }
    ledger.notes.push_back(note);
}

/// The pages of every shader-visible heap with a host heap that its next
/// take would give (sorted), and the heaps not watched by the kernel.
fn pending(heaps: &[(u64, u64, usize)]) -> (Vec<usize>, u32) {
    let (mut pages, mut views) = (Vec::new(), 0);
    for &(start, end, at) in heaps {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const super::super::Heap) };
        // (The heaps `sync_held` takes.)
        let (Some(Some(_)), Some(written), Some(_)) = (heap.host.get(), &heap.written, super::super::increment(heap.desc.kind))
        else {
            continue;
        };
        match written.pending(start as usize, end as usize) {
            Some(more) => pages.extend(more),
            None => views += 1,
        }
    }
    pages.sort_unstable();
    pages.dedup();
    (pages, views)
}

/// A sync's take of one heap's pages (`take`, the ranges it gave), noted
/// with the check.
pub(crate) fn took(take: impl FnOnce() -> Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if !checking() {
        return take();
    }
    let mut ledger = ledger();
    let ranges = take();
    let pages = pages_of(&ranges);
    ledger.event += 1;
    let event = ledger.event;
    add_take(&mut ledger, Take { sync: SYNC.get(), event, pages });
    ranges
}

fn add_take(ledger: &mut Ledger, take: Take) {
    if ledger.takes.len() == TAKES
        && let Some(gone) = ledger.takes.pop_front()
    {
        ledger.dropped = gone.event;
    }
    ledger.takes.push_back(take);
}

/// The 64 KiB pages of address ranges `[from, to)`, in order.
fn pages_of(ranges: &[(usize, usize)]) -> Vec<usize> {
    let page = crate::watch::PAGE;
    let mut pages: Vec<usize> = ranges
        .iter()
        .filter(|(from, to)| from < to)
        .flat_map(|&(from, to)| from / page..=(to - 1) / page)
        .collect();
    pages.sort_unstable();
    pages.dedup();
    pages
}

/// What [`verify`] found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Verdict {
    /// A note was found for the stamp (else nothing was checked).
    pub(super) checked: bool,
    /// Pages noted at the stamp.
    pub(super) pages: u64,
    /// Of them, taken since only by syncs still running, or by none.
    pub(super) running: u64,
    pub(super) untaken: u64,
    /// Heaps of view memory left unchecked at the stamp.
    pub(super) views: u32,
}

impl Verdict {
    /// No violation (an unchecked stamp counts as none, and is counted).
    pub(super) fn clean(&self) -> bool {
        self.running == 0 && self.untaken == 0
    }
}

/// At a join to be skipped for lists that need `need`, the covered stamp
/// being `covered`: whether every page noted at `need` has been taken since
/// by a sync that has ended. Counted for the line, the first violations
/// logged.
pub(super) fn verify(need: u64, covered: u64) -> Verdict {
    let verdict = {
        let ledger = ledger();
        check(&ledger, need, covered)
    };
    count(&verdict);
    verdict
}

fn check(ledger: &Ledger, need: u64, covered: u64) -> Verdict {
    // Nothing was handed over before: nothing needed.
    if need == 0 {
        return Verdict { checked: true, ..Verdict::default() };
    }
    let Some(note) = ledger.notes.iter().rev().find(|n| n.stamp == need) else {
        return Verdict::default();
    };
    // Takes let go after this note: what was taken is no longer known.
    if ledger.dropped > note.event {
        return Verdict::default();
    }
    let mut verdict = Verdict { checked: true, pages: note.pages.len() as u64, views: note.views, ..Verdict::default() };
    let first = ledger.takes.partition_point(|t| t.event < note.event);
    for &page in &note.pages {
        let mut by_running = false;
        let mut by_ended = false;
        for take in ledger.takes.range(first..) {
            if take.pages.binary_search(&page).is_ok() {
                if take.sync != 0 && take.sync <= covered {
                    by_ended = true;
                    break;
                }
                by_running = true;
            }
        }
        if by_ended {
            continue;
        }
        if by_running {
            verdict.running += 1;
        } else {
            verdict.untaken += 1;
        }
        said(page, need, covered, by_running);
    }
    verdict
}

/// Logs the first violations.
fn said(page: usize, need: u64, covered: u64, by_running: bool) {
    static SAID: AtomicU64 = AtomicU64::new(0);
    if SAID.fetch_add(1, Relaxed) < 16 {
        tracing::warn!(
            page = format_args!("{:#x}", page * crate::watch::PAGE),
            need,
            covered,
            "VICTORY_HEAP_SYNC_NEED=check: a heap page written before the lists' stamp was {} when the join was to be \
             skipped; the join waits whole",
            if by_running { "taken only by a sync still running" } else { "taken by no sync" }
        );
    }
}

/// Since the last line: joins checked, pages, violations (running,
/// untaken), stamps without a note, heaps of view memory left unchecked.
static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

fn count(v: &Verdict) {
    if v.checked {
        COUNTS[0].fetch_add(1, Relaxed);
    } else {
        COUNTS[4].fetch_add(1, Relaxed);
    }
    for (i, n) in [(1, v.pages), (2, v.running), (3, v.untaken), (5, u64::from(v.views))] {
        COUNTS[i].fetch_add(n, Relaxed);
    }
}

/// The heap sync line's part with the check (empty without it), since the
/// last line.
pub(super) fn part(frames: f64) -> String {
    if !checking() {
        return String::new();
    }
    let [checked, pages, running, untaken, unchecked, views] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed));
    said_part([checked, pages, running, untaken, unchecked, views], frames)
}

fn said_part([checked, pages, running, untaken, unchecked, views]: [u64; 6], frames: f64) -> String {
    format!(
        " (VICTORY_HEAP_SYNC_NEED check, since the last line over {frames:.0} frames: {checked} skips checked over {pages} \
         heap pages written before their lists' stamp, {} violations ({running} pages taken only by a sync still \
         running, {untaken} taken by none; those joins waited whole), {unchecked} skips without a note for their \
         stamp, {views} heaps of view memory not checked)",
        running + untaken,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(notes: Vec<Note>, takes: Vec<Take>) -> Ledger {
        Ledger { event: 0, notes: notes.into(), takes: takes.into(), dropped: 0 }
    }

    /// Pages noted at a stamp must be taken after the note by an ended
    /// sync; a take before the note, or by a sync still running, does not
    /// count.
    #[test]
    fn a_page_noted_at_the_stamp_must_be_taken_after_by_an_ended_sync() {
        let note = |stamp, event, pages: &[usize]| Note { stamp, event, pages: pages.to_vec(), views: 0 };
        let take = |sync, event, pages: &[usize]| Take { sync, event, pages: pages.to_vec() };
        // Noted at stamp 10 (event 2): pages 5 and 7.
        let ledger = with(
            vec![note(10, 2, &[5, 7])],
            vec![
                // Before the note: does not count.
                take(8, 1, &[5, 7]),
                // After: sync 12 took 5, sync 14 took 7.
                take(12, 3, &[1, 5]),
                take(14, 4, &[7]),
            ],
        );
        // Both ended (covered 14): clean.
        let v = check(&ledger, 10, 14);
        assert!(v.checked && v.clean(), "{v:?}");
        assert_eq!(v.pages, 2);
        // Sync 14 still runs (covered 12): page 7 taken only by it.
        let v = check(&ledger, 10, 12);
        assert_eq!((v.running, v.untaken), (1, 0), "{v:?}");
        // Neither ended: both.
        let v = check(&ledger, 10, 11);
        assert_eq!((v.running, v.untaken), (2, 0), "{v:?}");
        // A page no take after the note holds.
        let ledger = with(vec![note(10, 2, &[5, 9])], vec![take(8, 1, &[9]), take(12, 3, &[5])]);
        let v = check(&ledger, 10, 20);
        assert_eq!((v.running, v.untaken), (0, 1), "{v:?}");
        // A take by a sync not of ours (stamp 0) never counts as ended.
        let ledger = with(vec![note(10, 2, &[5])], vec![take(0, 3, &[5])]);
        assert_eq!(check(&ledger, 10, 20).running, 1);
        // No note for the stamp: not checked, no violation.
        let v = check(&ledger, 11, 20);
        assert!(!v.checked && v.clean());
        // Nothing needed: clean.
        assert!(check(&ledger, 0, 0).clean());
        // Takes let go after the note: not checked.
        let mut ledger = with(vec![note(10, 2, &[5])], vec![take(12, 4, &[5])]);
        ledger.dropped = 3;
        assert!(!check(&ledger, 10, 20).checked);
    }

    /// The ranges a take gives, as pages.
    #[test]
    fn ranges_become_their_pages() {
        let page = crate::watch::PAGE;
        assert_eq!(pages_of(&[(page, 3 * page), (5 * page + 8, 5 * page + 16), (2 * page, 2 * page)]), [1, 2, 5]);
        assert!(pages_of(&[]).is_empty());
    }

    /// The line's part.
    #[test]
    fn the_line_part_counts_the_checks() {
        let text = said_part([12, 340, 0, 1, 2, 0], 30.0);
        assert!(text.contains("12 skips checked over 340 heap pages"), "{text}");
        assert!(text.contains("1 violations (0 pages taken only by a sync still running, 1 taken by none"), "{text}");
        assert!(text.contains("2 skips without a note"), "{text}");
    }

    /// The ledger end to end over a real write-watched range: a page the
    /// title writes before a stamp is noted at it, and a take after the note
    /// clears it once its sync has ended.
    #[test]
    fn a_written_page_is_noted_and_cleared_by_a_take() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = 4 * crate::watch::PAGE;
        // SAFETY: a fresh allocation of the test's own.
        let memory = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) };
        assert!(!memory.is_null());
        let start = memory as usize;
        let tracked = crate::watch::track(start, len, false).expect("a write-watched range");
        // Nothing written yet (a fresh allocation reports nothing).
        let _ = tracked.take(start, start + len);
        assert_eq!(tracked.pending(start, start + len), Some(Vec::new()));
        // The title writes the second page.
        // SAFETY: inside the allocation.
        unsafe { (memory as *mut u8).add(crate::watch::PAGE + 16).write_volatile(7) };
        let noted = tracked.pending(start, start + len).expect("watched");
        assert_eq!(noted, [start / crate::watch::PAGE + 1]);
        // Peeking does not take: still pending, and the take gives it.
        assert_eq!(tracked.pending(start, start + len).unwrap(), noted);
        let taken = pages_of(&tracked.take(start, start + len));
        assert_eq!(taken, noted);
        assert_eq!(tracked.pending(start, start + len), Some(Vec::new()), "taken, yet still pending");
        // In the ledger: noted at stamp 10, taken by sync 11.
        let ledger = with(
            vec![Note { stamp: 10, event: 1, pages: noted.clone(), views: 0 }],
            vec![Take { sync: 11, event: 2, pages: taken }],
        );
        assert!(check(&ledger, 10, 11).clean());
        assert!(!check(&ledger, 10, 10).clean(), "the take's sync had not ended");
        drop(tracked);
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
    }
}
