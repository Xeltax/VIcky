//! Tests of `VICTORY_TEXTURE_EXACT`'s pure parts and of the audit's
//! classification (`crate::texexact`).

use std::collections::BTreeMap;

use super::audit::{State, bucket, classify, parse};
use super::*;

/// Part A's counts, for `descriptor::host::exact_spans`' test: named,
/// missed by the old rule, left out.
pub(crate) fn counts() -> [u64; 3] {
    [counted(N::ANamed), counted(N::AMissed), counted(N::ALeft)]
}

/// `VICTORY_TEXTURE_EXACT_PARTS`: letters in any order and case, others
/// ignored; unset or empty, all five.
#[test]
fn the_parts_are_read_from_their_letters() {
    assert_eq!(parse_parts(None), 0x1f);
    assert_eq!(parse_parts(Some("")), 0x1f);
    assert_eq!(parse_parts(Some("abcde")), 0x1f);
    assert_eq!(parse_parts(Some("a")), Part::A as u8);
    assert_eq!(parse_parts(Some("Ec")), Part::C as u8 | Part::E as u8);
    assert_eq!(parse_parts(Some("bd,x")), Part::B as u8 | Part::D as u8);
    assert_eq!(parse_parts(Some("x")), 0);
    assert_eq!(letters(Part::B as u8 | Part::D as u8), "bd");
}

/// Part A: the levels from `first` on end where level `first - 1` starts
/// (the tail block first, level 0 last); in the tail, the tail block; a
/// layout not so, or several slices, holds all of it.
#[test]
fn the_viewed_levels_end_where_the_next_larger_one_starts() {
    // 512x256 with 10 levels: tail from level 4 on in the first 4 KiB, level
    // 3 at 0x1000, 2 at 0x2000, 1 at 0x6000, 0 at 0x16000 (0x56000 in all).
    let offsets = [0x16000, 0x6000, 0x2000, 0x1000, 0, 0, 0, 0, 0, 0];
    let end = |first| viewed_end(0x56000, 1, 0x56000, 0x1000, 4, &offsets, first);
    assert_eq!(end(0), 0x56000);
    assert_eq!(end(1), 0x16000);
    assert_eq!(end(2), 0x6000);
    assert_eq!(end(3), 0x2000);
    assert_eq!(end(4), 0x1000);
    assert_eq!(end(9), 0x1000);
    // Past the last level, several slices, or levels in another order: all.
    assert_eq!(end(10), 0x56000);
    assert_eq!(viewed_end(0xac000, 2, 0x56000, 0x1000, 4, &offsets, 2), 0xac000);
    let rising = [0, 0x40000, 0x50000, 0x54000];
    assert_eq!(viewed_end(0x56000, 1, 0x56000, 0x1000, 4, &rising, 2), 0x56000);
}

/// Part C: a read that began at a sequence sees the writes that ended after
/// it and overlap its range, not those that ended before it began, nor
/// those elsewhere; the log keeps its last writes only.
#[test]
fn a_read_sees_the_writes_that_ended_after_it_began() {
    let mut log = WriteLog::default();
    log.ended(3, &[(0x1000, 0x2000)]);
    log.ended(5, &[(0x8000, 0x9000), (0x4000, 0x5000)]);
    // A read of [0x1800, 0x1900) that began at 2: the first write ended at 3.
    assert!(log.overlapped_since(2, 0x1800, 0x1900));
    // Began at 3: it had ended already.
    assert!(!log.overlapped_since(3, 0x1800, 0x1900));
    // The second write's second range, from either sequence before 5.
    assert!(log.overlapped_since(4, 0x4fff, 0x6000));
    assert!(log.overlapped_since(0, 0x4fff, 0x6000));
    // Touching is not overlapping; elsewhere is nothing.
    assert!(!log.overlapped_since(0, 0x5000, 0x6000));
    assert!(!log.overlapped_since(0, 0x2000, 0x4000));
    // Only the last writes are kept.
    for seq in 6..6 + LOG_MOST as u64 {
        log.ended(seq, &[(0x10_0000, 0x10_1000)]);
    }
    assert!(!log.overlapped_since(0, 0x1800, 0x1900));
    assert!(log.overlapped_since(0, 0x10_0000, 0x10_0001));
}

/// Part C: torn reads of one texture in a row are re-dued up to the bound,
/// then refused; a clean read starts the count anew; textures count apart.
#[test]
fn torn_reads_are_re_dued_a_bounded_number_of_times_in_a_row() {
    let mut s = Strikes::default();
    for _ in 0..TORN_MOST {
        assert!(s.hit(0x1000, TORN_MOST));
    }
    assert!(!s.hit(0x1000, TORN_MOST));
    assert!(s.hit(0x2000, TORN_MOST));
    s.clear(0x1000);
    assert!(s.hit(0x1000, TORN_MOST));
    assert_eq!(s.len(), 2);
}

/// Part D: memory mapped holds for under `HOLD_MS` after its map and not
/// past it; what is written of it is held no more, the rest still is; a
/// later map of the same bytes starts their time anew; unmapped parts are
/// left to the caller (the ranges say where).
#[test]
fn memory_not_written_holds_for_a_bounded_time() {
    let mut nr = NotReady::default();
    nr.map(0x10_0000, 0x14_0000, 1_000);
    // A texture inside, at once and just before the bound.
    assert_eq!(nr.fresh(0x11_0000, 0x12_0000, 1_000), (vec![(0x11_0000, 0x12_0000)], false));
    assert_eq!(nr.fresh(0x11_0000, 0x12_0000, 1_000 + HOLD_MS - 1).0.len(), 1);
    // At the bound: no longer held, and said to be over.
    assert_eq!(nr.fresh(0x11_0000, 0x12_0000, 1_000 + HOLD_MS), (vec![], true));
    // Written in the middle: the parts around stay, the middle is free.
    nr.written(0x11_0000, 0x12_0000);
    assert_eq!(nr.fresh(0x11_0000, 0x12_0000, 1_100), (vec![], false));
    assert_eq!(nr.fresh(0x10_0000, 0x14_0000, 1_100).0, vec![(0x12_0000, 0x14_0000), (0x10_0000, 0x11_0000)]);
    assert_eq!(nr.len(), 2);
    // Mapped again later: its time starts anew.
    nr.map(0x13_0000, 0x15_0000, 1_400);
    assert_eq!(nr.fresh(0x13_8000, 0x13_9000, 1_000 + HOLD_MS + 50).0, vec![(0x13_8000, 0x13_9000)]);
    assert_eq!(nr.fresh(0x12_0000, 0x12_1000, 1_000 + HOLD_MS + 50), (vec![], true));
    // Outside every range: nothing.
    assert_eq!(nr.fresh(0x20_0000, 0x21_0000, 1_100), (vec![], false));
    // Past their hold the ranges go.
    nr.purge(1_400 + HOLD_MS);
    assert_eq!(nr.len(), 0);
}

/// Part E: the records quiet for `QUIET` frames, the newest first, until
/// the bytes of a frame (at least one).
#[test]
fn settle_checks_take_the_newest_quiet_records_within_their_bytes() {
    let mut set: BTreeMap<(u64, usize), (u64, u64)> = BTreeMap::new();
    set.insert((10, 1), (10, 4 << 20));
    set.insert((20, 2), (20, 4 << 20));
    set.insert((30, 3), (30, 4 << 20));
    set.insert((95, 4), (95, 4 << 20));
    // Frame 100: 4 is not quiet yet; 3 and 2 fill 8 MB.
    let picked = pick_unsettled(&set, 100, 8 << 20);
    assert_eq!(picked.iter().map(|p| p.1).collect::<Vec<_>>(), [3, 2]);
    // A record larger than the budget is still taken alone.
    set.insert((40, 5), (40, 64 << 20));
    assert_eq!(pick_unsettled(&set, 100, 8 << 20).iter().map(|p| p.1).collect::<Vec<_>>(), [5]);
    // Too early for any.
    assert!(pick_unsettled(&set, 5, 8 << 20).is_empty());
}

/// The audit: in flight when an event lies 8 frames or fewer before the
/// snapshot (or after it), else partly mapped, else queued, else out of the
/// queue; and its switch and buckets.
#[test]
fn the_audit_classifies_mismatches_by_their_last_event_and_queue() {
    assert_eq!(classify(1000, 992, false, false), State::InFlight);
    assert_eq!(classify(1000, 1005, true, true), State::InFlight);
    assert_eq!(classify(1000, 991, false, true), State::Partly);
    assert_eq!(classify(1000, 991, true, true), State::Partly);
    assert_eq!(classify(1000, 991, true, false), State::QueuedLong);
    assert_eq!(classify(1000, 100, false, false), State::OutOfQueue);
    assert_eq!(classify(5, 0, false, false), State::InFlight);
    assert_eq!(parse(None), (vec![], false));
    assert_eq!(parse(Some("0")), (vec![], false));
    assert_eq!(parse(Some("1500, 2000,x,2600")), (vec![1500, 2000, 2600], true));
    assert_eq!(parse(Some("bake")), (vec![], true));
    assert_eq!(bucket(0xffff).0, 0);
    assert_eq!(bucket(0x1_0000).0, 1);
    assert_eq!(bucket(0x40_0000).0, 4);
    assert_eq!(bucket(1 << 30).0, 5);
}

/// Part B: the print of bytes in hand is the fingerprint of memory that
/// holds them, small and large, with and without mip ranges.
#[test]
fn the_print_of_detiled_bytes_matches_the_fingerprint_of_their_memory() {
    for size in [0x8000usize, 0x10_0000] {
        let memory: Vec<u8> = (0..size).map(|i| (i * 7 + i / 4096) as u8).collect();
        for mips in [vec![], vec![(0x4_0000u64, 0x400u64), (0x7_f000, 0x2000)]] {
            let print = crate::texture::fingerprint(memory.as_ptr() as u64, size as u64, &mips);
            assert_eq!(crate::texture::fingerprint_of(&memory, &mips), print);
        }
    }
}
