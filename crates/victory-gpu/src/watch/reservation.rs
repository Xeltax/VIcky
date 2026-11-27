//! `VICTORY_WATCH_RESERVATION`: the submission's one write-watch asking
//! ([`super::ask_once`]) by memory reservation, not by the distance of its
//! ranges. Ported from fxceless/victory-fork PR #1365 (spk-c, after
//! Kelvaroq 1270c42), as reconciled with our journal, guards and light
//! refresh.
//!
//! The ordinary asking joins ranges closer than a fixed gap and, when a
//! call fails (a run across two reservations, or into memory that is not
//! write-watched), asks again by halves. Here the ranges are sorted and
//! deduplicated, Windows is asked once what allocation each group starts
//! in, and a group is one `GetWriteWatch` call within that allocation: it
//! may include the pages between its ranges (they are noted in the journal
//! as any page found written is), and it is allowed a gap as large as the
//! smaller of the two spans it would join. Memory that is not private
//! (views) is left to the guards, as before.
//!
//! The PR's own measure (a synthetic 10,000 fragmented ranges over 100
//! reservations): 1,307 calls down to 100, 4 KiB pages queried 3.9 M down to
//! 0.48 M, asking 15.6 ms down to 8.8 ms; and, in its words, a regression
//! for a single reservation (5.98 ms up to 9.43 ms), so it stays opt-in.
//! Not measured here yet.
//!
//! The mapping may not change while the groups are made (the asking takes
//! the read side of [`super::title_memory_reading_now`]'s permit and, if the
//! runtime holds it, asks the ordinary way), the coverage is forgotten when
//! the runtime maps or unmaps memory, and a write the runtime or the
//! kernel completed under the asking (`title_memory_written`) counts as
//! written after it.
//!
//! `VICTORY_WATCH_RESERVATION=alt`: twenty seconds with, twenty without, in
//! turn, each change said in the log with the counts of the phase before.

use super::*;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

/// 0 off, 1 on, 2 in turn. Off too with `VICTORY_VIEW_ASK=1` or
/// `VICTORY_GUARD_ASK=1`, which ask the kernel about view memory (the
/// groups here never do).
fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        if ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"].iter().any(|name| std::env::var_os(name).is_some_and(|v| v == "1")) {
            return 0;
        }
        match std::env::var("VICTORY_WATCH_RESERVATION").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => 0,
        }
    })
}

/// Since the change before: the submissions' askings made, their
/// microseconds, the `GetWriteWatch` calls they made, the 4 KiB pages those
/// asked about, and the allocation queries.
static PHASE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Whether a submission's asking is made by reservation (`alt`: by the
/// phase, each change said with the counts of the phase before).
pub(super) fn on() -> bool {
    match mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: AtomicU8 = AtomicU8::new(2);
            if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
                let [askings, us, calls, pages, regions] = PHASE.each_ref().map(|n| n.swap(0, Relaxed));
                let per = |n: u64| n as f64 / askings.max(1) as f64;
                tracing::info!(
                    "a submission's write-watch asking goes by reservation from now (VICTORY_WATCH_RESERVATION=alt): {}; \
                     since the change before: {askings} askings, {:.2} ms an asking, {:.1} GetWriteWatch calls, \
                     {:.0} 4 KiB pages asked and {:.1} allocation queries an asking",
                    if on { "ON" } else { "off, as usual" },
                    per(us) / 1e3,
                    per(calls),
                    per(pages),
                    per(regions)
                );
            }
            on
        }
    }
}

/// What a kernel asking of the journal's made: counted for the phases' log
/// (`pages` of 4 KiB).
pub(super) fn count_kernel(pages: usize) {
    if mode() == 2 {
        PHASE[2].fetch_add(1, Relaxed);
        PHASE[3].fetch_add(pages as u64, Relaxed);
    }
}

/// Times a submission's asking until dropped (`alt` only).
pub(super) struct Timed(Option<std::time::Instant>);

pub(super) fn timed() -> Timed {
    Timed((mode() == 2).then(std::time::Instant::now))
}

impl Drop for Timed {
    fn drop(&mut self) {
        if let Some(started) = self.0 {
            PHASE[0].fetch_add(1, Relaxed);
            PHASE[1].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        }
    }
}

/// Windows' answer about the allocation holding page `page`: its first and
/// last page (exclusive) and whether it is private memory (a view is not).
fn region(page: usize) -> Option<(usize, usize, bool)> {
    /// `MEMORY_REGION_INFORMATION` (ntddk), the part that is read.
    #[repr(C)]
    #[derive(Default)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        flags: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    /// `MemoryRegionInformationEx`, whose second word holds the flags of
    /// `WIN32_MEMORY_REGION_INFORMATION` (the first is "private"); class 3
    /// answers with the kind of memory there (as [`super::placeholder`]
    /// found).
    const MEMORY_REGION_INFORMATION: u32 = 7;
    PHASE[4].fetch_add(u64::from(mode() == 2), Relaxed);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::WatchRegions, 1);
    let mut region = Region::default();
    // SAFETY: the answer's room and its size; the kernel checks the address.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            page.checked_mul(PAGE)? as *const _,
            MEMORY_REGION_INFORMATION,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    if status < 0 {
        return None;
    }
    let end = region.allocation_base.checked_add(region.region_size)?;
    Some((region.allocation_base.div_ceil(PAGE), end / PAGE, region.flags & 1 != 0))
}

/// Whether the group `[first, last)` can take `next` with it: the call
/// stays within [`AHEAD_MAX`], and the gap between them within [`AHEAD_GAP`]
/// or the smaller span of the two.
fn can_join(first: usize, last: usize, next: (usize, usize)) -> bool {
    last.max(next.1).saturating_sub(first) <= AHEAD_MAX
        && next.0.saturating_sub(last) <= AHEAD_GAP.max((last - first).min(next.1 - next.0))
}

/// The submission's asking, by reservation, for `ranges` (as
/// [`super::ask_once`] takes them); `view` is told the place of each range
/// found to be view memory the guards hold. False, with nothing done, if the
/// runtime is changing a mapping: the ordinary asking is made instead.
pub(super) fn ask(ranges: &[(usize, usize)], mut view: impl FnMut(usize)) -> bool {
    let Some(_mapping) = title_memory_reading_now() else {
        return false;
    };
    let mut pieces: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (i, &(start, end)) in ranges.iter().enumerate() {
        if start >= end {
            continue;
        }
        // View memory the guards hold is not asked about
        // (as [`super::ask_once`]).
        if known_view(start) {
            #[cfg(feature = "diagnostics")]
            AHEAD_VIEWS.fetch_add(1, Relaxed);
            view(i);
            continue;
        }
        pieces.push((start / PAGE, end.div_ceil(PAGE)));
    }
    pieces.sort_unstable();
    pieces.dedup();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    run(&mut j, &pieces);
    true
}

fn run(j: &mut Journal, mut pieces: &[(usize, usize)]) {
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut allocation = None;
    while let Some(&(first, end)) = pieces.first() {
        // The allocation is asked about only where a group could be made,
        // and kept for the pieces that lie in it.
        if allocation.is_none_or(|(a, b, _)| first < a || first >= b) {
            allocation = if pieces.get(1).is_some_and(|&n| can_join(first, end, n)) { region(first) } else { None };
        }
        let (_, limit, private) = allocation.unwrap_or((first, end, true));
        let mut last = end;
        let mut n = 1;
        while let Some(&next) = pieces.get(n) {
            if last > limit || next.1 > limit || !can_join(first, last, next) {
                break;
            }
            last = last.max(next.1);
            n += 1;
        }
        // A group that fails (or whose allocation is not private, or does
        // not hold it all) is asked piece by piece; what still fails is
        // left to `writes`, as in the ordinary asking.
        if private && last <= limit && last - first <= AHEAD_MAX && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        } else if private && n > 1 {
            for &(a, b) in &pieces[..n] {
                if b - a <= AHEAD_MAX && !j.ahead.holds(a, b) && ask_kernel(j, a, b) {
                    j.ahead.push((a, b));
                }
            }
        }
        pieces = &pieces[n..];
    }
    if j.ahead.is_empty() {
        // Nothing was asked: the number of the asking that stands is new
        // all the same.
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    j.kept = !j.ahead.is_empty();
}

/// Title memory `[at, at + len)` was written by a request that has now
/// completed (a read or a storage request into it), which may have landed
/// after the asking's reset: the pages of it under the asking that stands
/// count as written after it, as [`super::wrote`] does for our own writes.
pub(super) fn completed_write(at: usize, len: usize) {
    if len == 0 || !on() {
        return;
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        complete(&mut j, at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    }
}

/// [`completed_write`] for pages `[first, last)`.
fn complete(j: &mut Journal, first: usize, last: usize) {
    let Journal { ahead, written, asked, ahead_asked, .. } = j;
    // Ranges whose furthest end so far is not past `first` lie before it.
    let from = ahead.reach.partition_point(|&far| far <= first);
    let mut epoch = None;
    for &(a, b) in &ahead.ranges[from..] {
        if a >= last {
            break;
        }
        let (lo, hi) = (a.max(first), b.min(last));
        if lo >= hi {
            continue;
        }
        let now = *epoch.get_or_insert_with(|| {
            *asked += 1;
            *ahead_asked = *asked;
            *asked
        });
        for page in lo..hi {
            written.insert(page, now);
        }
    }
}

/// The runtime maps or unmaps title memory: the coverage of the asking that
/// stands is forgotten (a range under it may now show other pages).
pub(super) fn remapped() {
    if on() {
        ask_once_end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// A write-watched reservation of `pages` 64 KiB pages, freed on drop.
    struct Memory(usize);

    impl Memory {
        fn new(pages: usize) -> Self {
            // SAFETY: a fresh reservation, freed by `drop`.
            let at = unsafe { VirtualAlloc(std::ptr::null(), pages * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) };
            assert!(!at.is_null());
            Self(at as usize)
        }

        /// Commits the first 4 KiB of page `page`; its address.
        fn commit(&self, page: usize) -> usize {
            let at = self.0 + page * PAGE;
            // SAFETY: inside our reservation.
            assert_eq!(unsafe { VirtualAlloc(at as *const _, KERNEL_PAGE, MEM_COMMIT, PAGE_READWRITE) } as usize, at);
            at
        }
    }

    impl Drop for Memory {
        fn drop(&mut self) {
            // SAFETY: our reservation.
            assert_ne!(unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) }, 0);
        }
    }

    fn fresh() -> Journal {
        Journal {
            asked: 0,
            written: Written::new(),
            found: Vec::new(),
            ahead: Ahead::new(),
            ahead_asked: 0,
            kept: false,
        }
    }

    /// Pages between two asked ranges are noted if written and not held;
    /// writes racing the asking from other threads are in the journal at
    /// the next asking at the latest; a page committed again is found.
    #[test]
    fn reservation_gaps_and_racing_writes() {
        let memory = Memory::new(5);
        let a = memory.commit(0);
        let gap = memory.commit(2);
        let b = memory.commit(4);
        let pieces = [(a / PAGE, a / PAGE + 1), (b / PAGE, b / PAGE + 1)];
        let mut j = fresh();
        // SAFETY: committed memory of ours.
        unsafe { (gap as *mut u8).write_volatile(1) };
        run(&mut j, &pieces);
        assert!(j.ahead.holds(a / PAGE, a / PAGE + 1));
        assert!(j.kept);
        assert_eq!(j.written.range(gap / PAGE, gap / PAGE + 1).collect::<Vec<_>>(), [(gap / PAGE, j.asked)], "the gap's write is noted");
        for value in 1..=16u8 {
            let gate = std::sync::Barrier::new(3);
            std::thread::scope(|s| {
                for at in [a, b] {
                    let gate = &gate;
                    s.spawn(move || {
                        gate.wait();
                        // SAFETY: committed memory of ours.
                        unsafe { (at as *mut u8).write_volatile(value) };
                    });
                }
                gate.wait();
                run(&mut j, &pieces);
            });
            let epoch = j.asked;
            run(&mut j, &pieces);
            let last = |page: usize| j.written.range(page, page + 1).map(|(_, w)| w).next().unwrap_or(0);
            assert!(last(a / PAGE) >= epoch && last(b / PAGE) >= epoch, "a write racing the asking is found by the next");
        }
        // SAFETY: committed memory of ours.
        assert_ne!(unsafe { VirtualFree(b as *mut _, KERNEL_PAGE, MEM_DECOMMIT) }, 0);
        assert_eq!(memory.commit(4), b);
        // SAFETY: committed memory of ours.
        unsafe { (b as *mut u8).write_volatile(0x78) };
        run(&mut j, &pieces);
        assert_eq!(j.written.range(b / PAGE, b / PAGE + 1).map(|(_, w)| w).next(), Some(j.asked));
    }

    /// A write completed under the asking that stands is written after it,
    /// for the pages of the asked ranges it lies in and no others.
    #[test]
    fn a_completed_write_counts_as_after_the_asking() {
        let memory = Memory::new(8);
        let (a, b, c) = (memory.commit(0), memory.commit(3), memory.commit(6));
        let pieces = [(a / PAGE, a / PAGE + 1), (b / PAGE, b / PAGE + 1)];
        let mut j = fresh();
        run(&mut j, &pieces);
        let standing = j.ahead_asked;
        // Under the asking (a and b), and past it (c).
        complete(&mut j, a / PAGE, a / PAGE + 1);
        assert!(j.ahead_asked > standing);
        let now = j.ahead_asked;
        assert_eq!(j.written.range(a / PAGE, a / PAGE + 1).collect::<Vec<_>>(), [(a / PAGE, now)]);
        assert_eq!(j.written.range(b / PAGE, b / PAGE + 1).count(), 0, "b was not written to");
        complete(&mut j, c / PAGE, c / PAGE + 1);
        assert_eq!(j.ahead_asked, now, "nothing asked there");
        assert_eq!(j.written.range(c / PAGE, c / PAGE + 1).count(), 0);
    }

    /// Ranges in several reservations are asked in groups that stay within
    /// each: every range is held by the asking, and the journal notes
    /// exactly the pages that were written.
    #[test]
    fn groups_note_what_was_written() {
        let memories: Vec<Memory> = (0..4).map(|_| Memory::new(64)).collect();
        let at: Vec<usize> = memories.iter().flat_map(|m| (0..64).step_by(3).map(|p| m.commit(p)).collect::<Vec<_>>()).collect();
        let pieces: Vec<(usize, usize)> = at.iter().map(|&p| (p / PAGE, p / PAGE + 1)).collect();
        // The first asking drains whatever committing set.
        run(&mut fresh(), &pieces);
        for (n, &p) in at.iter().enumerate() {
            if n % 5 != 2 {
                // SAFETY: committed memory of ours.
                unsafe { (p as *mut u8).write_volatile(7) };
            }
        }
        let mut j = fresh();
        run(&mut j, &pieces);
        assert!(j.ahead.ranges.len() <= memories.len(), "{} calls for {} reservations", j.ahead.ranges.len(), memories.len());
        for (n, &(first, last)) in pieces.iter().enumerate() {
            assert!(j.ahead.holds(first, last), "range {n} is held");
            assert_eq!(j.written.range(first, last).count(), usize::from(n % 5 != 2), "range {n}");
        }
    }
}
