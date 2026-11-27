//! VIcky (`VICTORY_SPARSE_KNOWN=1|alt|check`; off by default): the holes
//! of a host copy's range that `copy_mapped` read around are remembered,
//! per range, and the next copy of the same range reads the runs between
//! them as runs and zeros the holes, instead of failing a copy and finding
//! the holes again page by page.
//!
//! Profile PF2/PF3 (lever E of the PF2 note, lever 3 of PF3): 11.6-12.3
//! ranges of 64 KiB or more a frame have holes, the same ones every frame,
//! and reading around them cost 545-561 kernel reads (172-201 of them
//! failing) and 7.3 ms of the submitting thread in `read_probing`, on top
//! of the failed first copy of each.
//!
//! What a remembered range does at its next copy ([`copy`]):
//! - each run between holes is copied whole (in place with
//!   `VICTORY_COPY_PLAIN=1`, else one kernel read); a run that does not
//!   read (a page went) sends the range the usual way;
//! - each hole's first page is tried with one kernel read (about a
//!   microsecond when it fails); one that reads (memory came) sends the
//!   range the usual way; else the hole is zeroed, as before;
//! - a range not probed for [`REPROBE_FRAMES`] frames goes the usual way
//!   again, at most [`REPROBE_PER_FRAME`] a frame, for changes nobody told
//!   of in the middle of a hole.
//!
//! The usual way records what it found ([`found`], [`whole`]), in every
//! mode, unless the runtime mapped or unmapped memory over the range while
//! it read ([`forget`], from `title_memory_remapped` and new
//! reservations): then the range is forgotten, and asked again at its next
//! copy.
//!
//! `check`: every range goes the usual way, and a remembered one is
//! compared with what was found (counted when they differ). `alt`: twenty
//! seconds the usual way, twenty around remembered holes, in turn, each
//! change said in the log; the holes are learned in both phases.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::PAGE;

/// A remembered range is probed the usual way again after this many frames.
const REPROBE_FRAMES: u64 = 16;
/// At most this many ranges a frame are probed again for their age.
const REPROBE_PER_FRAME: u32 = 2;
/// The most ranges remembered; all are forgotten past it.
const MOST: usize = 4096;
/// Mappings changed lately ([`State::changed`]), kept for ranges being
/// read while they happened.
const CHANGES_KEPT: usize = 512;

/// `VICTORY_SPARSE_KNOWN`: 0 off, 1 on, 2 in turn, 3 check.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_SPARSE_KNOWN").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            Ok("check") => 3,
            _ => return 0,
        };
        tracing::info!(
            "host copies of ranges with holes are read around the holes found the last time, each hole's first page \
             tried, a range probed again every {REPROBE_FRAMES} frames ({REPROBE_PER_FRAME} a frame at most) and \
             forgotten when memory is mapped over it (VICTORY_SPARSE_KNOWN={})",
            match mode {
                1 => "1",
                2 => "alt: twenty seconds the usual way, twenty around remembered holes",
                _ => "check: the usual way, the remembered holes compared with what it finds",
            }
        );
        mode
    })
}

/// Whether this copy reads around remembered holes (`alt`: by the phase,
/// each change said with the counts of the phase before).
fn used() -> bool {
    match mode() {
        1 => true,
        2 => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: AtomicU8 = AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                let [known, known_us, probed, probed_us] = [0, 1, 2, 3].map(|i| PHASE[i].swap(0, Ordering::Relaxed));
                let per = |us: u64, n: u64| us as f64 / n.max(1) as f64;
                tracing::info!(
                    "ranges with holes are read around remembered holes from now (VICTORY_SPARSE_KNOWN=alt): {}; \
                     since the change before: {known} ranges read around remembered holes, {:.1} us a range; \
                     {probed} ranges with holes read the usual way, {:.1} us a range",
                    if on { "ON" } else { "off, the usual way (the holes are still learned)" },
                    per(known_us, known),
                    per(probed_us, probed)
                );
            }
            on
        }
        _ => false,
    }
}

/// A range's holes, as offsets `[from, to)` from its first byte, in order,
/// apart, and the frame it was last probed in.
struct Holes {
    holes: Vec<(usize, usize)>,
    probed: u64,
}

/// The remembered ranges by (first byte, bytes), with the mappings changed
/// lately.
#[derive(Default)]
struct State {
    ranges: HashMap<(usize, usize), Holes>,
    /// Changes counted so far.
    changes: u64,
    /// The last [`CHANGES_KEPT`] changes: (number, at, bytes).
    changed: VecDeque<(u64, usize, usize)>,
    /// The frame of the probes again for age, and how many it had.
    reprobe_frame: u64,
    reprobes: u32,
}

impl State {
    /// Whether a mapping changed over `[at, at + len)` since change
    /// `since` was counted (or too long ago to tell).
    fn changed_since(&self, since: u64, at: usize, len: usize) -> bool {
        if self.changes == since {
            return false;
        }
        if self.changed.front().is_none_or(|c| c.0 > since + 1) {
            return true;
        }
        let end = at.saturating_add(len);
        self.changed.iter().any(|&(n, from, bytes)| n > since && from < end && from.saturating_add(bytes) > at)
    }

    /// Whether a range may be probed again for its age in `frame`.
    fn may_reprobe(&mut self, frame: u64) -> bool {
        if self.reprobe_frame != frame {
            self.reprobe_frame = frame;
            self.reprobes = 0;
        }
        if self.reprobes >= REPROBE_PER_FRAME {
            return false;
        }
        self.reprobes += 1;
        true
    }
}

fn state() -> std::sync::MutexGuard<'static, State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(Mutex::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// Counters since the sparse line before ([`clause`]), by their index.
mod n {
    /// Ranges read around remembered holes, and their microseconds.
    pub(super) const KNOWN: usize = 0;
    pub(super) const KNOWN_US: usize = 1;
    /// Ranges with holes (found now, or remembered) read the usual way,
    /// and their microseconds (from the first copy tried).
    pub(super) const PROBED: usize = 2;
    pub(super) const PROBED_US: usize = 3;
    /// Kernel reads of the remembered ranges (the holes' first pages, and
    /// runs without copies in place), and the failed ones among them.
    pub(super) const READS: usize = 4;
    pub(super) const FAILED: usize = 5;
    /// Remembered ranges sent the usual way: for their age, as a run did
    /// not read, as a hole's first page read.
    pub(super) const AGED: usize = 6;
    pub(super) const RUN_GONE: usize = 7;
    pub(super) const HOLE_READ: usize = 8;
    /// Ranges forgotten as memory was mapped over them, and probes not
    /// remembered as it was while they read.
    pub(super) const FORGOTTEN: usize = 9;
    pub(super) const NOT_KEPT: usize = 10;
    /// `check`: remembered ranges compared with the usual way, and those
    /// that differed.
    pub(super) const CHECKED: usize = 11;
    pub(super) const DIFFERED: usize = 12;
    pub(super) const ALL: usize = 13;
}

static COUNT: [AtomicU64; n::ALL] = [const { AtomicU64::new(0) }; n::ALL];
/// The first four of [`COUNT`], since the last change of `alt`'s phase.
static PHASE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn count(i: usize, by: u64) {
    COUNT[i].fetch_add(by, Ordering::Relaxed);
    if i < 4 {
        PHASE[i].fetch_add(by, Ordering::Relaxed);
    }
}

/// What [`copy`] did with a range.
pub(super) enum Known {
    /// Nothing: the switch is off.
    Off,
    /// Copied around its remembered holes.
    Done,
    /// To read the usual way, and to tell [`found`] or [`whole`] what that
    /// found: whether the range was remembered, the changes counted when it
    /// was looked up, and when its copy began.
    Probe { had: bool, since: u64, started: std::time::Instant },
}

/// A copy of title memory `[src, src + len)` into host memory `dst`, 64 KiB
/// or more: around its remembered holes if it has some and they are used
/// now ([`Known::Done`]), else to be read the usual way.
pub(super) fn copy(dst: *mut u8, src: *const u8, len: usize) -> Known {
    let mode = mode();
    if mode == 0 {
        return Known::Off;
    }
    let started = std::time::Instant::now();
    let (holes, since) = {
        let mut state = state();
        let since = state.changes;
        let probe = |had| Known::Probe { had, since, started };
        if !state.ranges.contains_key(&(src as usize, len)) {
            return probe(false);
        }
        if !used() {
            return probe(true);
        }
        let frame = crate::frame::current();
        let aged = state.ranges.get(&(src as usize, len)).is_some_and(|r| frame.saturating_sub(r.probed) >= REPROBE_FRAMES);
        if aged && state.may_reprobe(frame) {
            count(n::AGED, 1);
            return probe(true);
        }
        let holes = state.ranges.get(&(src as usize, len)).map(|r| r.holes.clone()).unwrap_or_default();
        (holes, since)
    };
    let mut reads = (0, 0);
    let done = around(dst, src, len, &holes, &mut reads);
    count(n::READS, reads.0);
    count(n::FAILED, reads.1);
    match done {
        Ok(()) => {
            count(n::KNOWN, 1);
            count(n::KNOWN_US, started.elapsed().as_micros() as u64);
            Known::Done
        }
        Err(gone) => {
            count(if gone { n::RUN_GONE } else { n::HOLE_READ }, 1);
            Known::Probe { had: true, since, started }
        }
    }
}

/// Copies `[src, src + len)` into `dst` around `holes` (offsets, in order,
/// apart), zeroing them. `Err(true)`: a run did not read; `Err(false)`: a
/// hole's first page did. `reads`: kernel reads made, and the failed ones.
fn around(dst: *mut u8, src: *const u8, len: usize, holes: &[(usize, usize)], reads: &mut (u64, u64)) -> Result<(), bool> {
    let base = src as usize;
    let mut at = 0;
    for &(from, to) in holes.iter().chain(std::iter::once(&(len, len))) {
        if from > at {
            let n = from - at;
            let (s, d) = (src.wrapping_add(at), dst.wrapping_add(at));
            let read = crate::plaincopy::copy_run(d, s, n) || {
                if crate::plaincopy::on() {
                    false
                } else {
                    reads.0 += 1;
                    super::read(s, d, n)
                }
            };
            if !read {
                return Err(true);
            }
        }
        if to > from {
            // The hole's first page: one kernel read.
            let first = ((base + from) / PAGE + 1) * PAGE - base;
            let n = first.min(to) - from;
            reads.0 += 1;
            if super::read(src.wrapping_add(from), dst.wrapping_add(from), n) {
                return Err(false);
            }
            reads.1 += 1;
            super::unmapped(to - from);
            // SAFETY: inside the destination, writable host memory.
            unsafe { dst.wrapping_add(from).write_bytes(0, to - from) };
        }
        at = to;
    }
    Ok(())
}

/// The usual way found holes in `[src, src + len)` (`holes`: first byte and
/// bytes of each, in any order, possibly touching): remembered, unless a
/// mapping changed over the range meanwhile.
pub(super) fn found(known: &Known, src: usize, len: usize, holes: &[(usize, usize)]) {
    let Known::Probe { had, since, started } = *known else {
        return;
    };
    count(n::PROBED, 1);
    count(n::PROBED_US, started.elapsed().as_micros() as u64);
    let holes = offsets(src, len, holes);
    let mut state = state();
    let changed = state.changed_since(since, src, len);
    // (A range forgotten on a mapping meanwhile is not compared.)
    if mode() == 3 && had && !changed {
        count(n::CHECKED, 1);
        if state.ranges.get(&(src, len)).is_none_or(|r| r.holes != holes) {
            count(n::DIFFERED, 1);
        }
    }
    if changed {
        if state.ranges.remove(&(src, len)).is_some() || !holes.is_empty() {
            count(n::NOT_KEPT, 1);
        }
        return;
    }
    if holes.is_empty() {
        state.ranges.remove(&(src, len));
        return;
    }
    if state.ranges.len() >= MOST && !state.ranges.contains_key(&(src, len)) {
        state.ranges.clear();
    }
    let probed = crate::frame::current();
    state.ranges.insert((src, len), Holes { holes, probed });
}

/// The usual way read `[src, src + len)` whole: no holes to remember.
pub(super) fn whole(known: &Known, src: usize, len: usize) {
    if let Known::Probe { had: true, .. } = known {
        found(known, src, len, &[]);
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`, or
/// reserved it anew: ranges over it are forgotten, and probes reading it
/// now are not remembered.
pub(crate) fn forget(at: usize, len: usize) {
    if mode() == 0 {
        return;
    }
    let mut state = state();
    state.changes += 1;
    let change = (state.changes, at, len);
    if state.changed.len() >= CHANGES_KEPT {
        state.changed.pop_front();
    }
    state.changed.push_back(change);
    let end = at.saturating_add(len);
    let before = state.ranges.len();
    state.ranges.retain(|&(from, bytes), _| !(from < end && from.saturating_add(bytes) > at));
    let gone = before - state.ranges.len();
    if gone > 0 {
        count(n::FORGOTTEN, gone as u64);
    }
}

/// Holes given by first byte and bytes as offsets from `src`, inside
/// `[0, len)`, in order, apart (those that touch joined).
fn offsets(src: usize, len: usize, holes: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = holes
        .iter()
        .filter_map(|&(at, n)| {
            let from = at.saturating_sub(src).min(len);
            let to = at.saturating_add(n).saturating_sub(src).min(len);
            (to > from).then_some((from, to))
        })
        .collect();
    out.sort_unstable();
    let mut joined: Vec<(usize, usize)> = Vec::with_capacity(out.len());
    for (from, to) in out {
        match joined.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => joined.push((from, to)),
        }
    }
    joined
}

/// The sparse line's words on remembered holes (empty without the
/// switch), per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn clause(frames: f64) -> String {
    let mode = mode();
    if mode == 0 {
        return String::new();
    }
    let c: Vec<u64> = COUNT.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    let per = |i: usize| c[i] as f64 / frames.max(1.0);
    let known = state().ranges.len();
    let check = if mode == 3 {
        format!("; check: {:.1} compared, {:.2} differed", per(n::CHECKED), per(n::DIFFERED))
    } else {
        String::new()
    };
    format!(
        " (holes remembered, VICTORY_SPARSE_KNOWN={}: {:.1} ranges read around them {:.2} ms with {:.0} kernel reads, \
         {:.0} of them failed; {:.1} ranges with holes read the usual way {:.2} ms, of them {:.2} for their age, \
         {:.2} as a run did not read, {:.2} as a hole's first page read; {known} ranges known, {:.2} forgotten on \
         mappings, {:.2} probes not kept{check})",
        match mode {
            1 => "1",
            2 => "alt",
            _ => "check",
        },
        per(n::KNOWN),
        per(n::KNOWN_US) / 1e3,
        per(n::READS),
        per(n::FAILED),
        per(n::PROBED),
        per(n::PROBED_US) / 1e3,
        per(n::AGED),
        per(n::RUN_GONE),
        per(n::HOLE_READ),
        per(n::FORGOTTEN),
        per(n::NOT_KEPT)
    )
}

#[cfg(test)]
mod tests {
    use super::{PAGE, State, around, offsets};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    /// Holes given by address come back as offsets, clipped to the range,
    /// in order, those that touch or overlap joined.
    #[test]
    fn holes_become_ordered_offsets() {
        let src = 0x10_0000;
        let len = 16 * PAGE;
        let holes = [
            (src + 9 * PAGE, PAGE),
            (src + 3 * PAGE, PAGE),
            (src + 4 * PAGE, PAGE),
            (src - PAGE, 2 * PAGE),
            (src + 15 * PAGE, 4 * PAGE),
            (src + 20 * PAGE, PAGE),
            (src + 3 * PAGE + 16, 64),
        ];
        assert_eq!(
            offsets(src, len, &holes),
            [(0, PAGE), (3 * PAGE, 5 * PAGE), (9 * PAGE, 10 * PAGE), (15 * PAGE, 16 * PAGE)]
        );
        assert!(offsets(src, len, &[]).is_empty());
    }

    /// A change of a mapping over a range read meanwhile is seen, one
    /// elsewhere is not, and one too old to tell counts as seen.
    #[test]
    fn changes_over_a_range_are_seen() {
        let mut state = State::default();
        assert!(!state.changed_since(0, 0x1000, 0x1000));
        state.changes = 2;
        state.changed.extend([(1, 0x8000, 0x1000), (2, 0x2000, 0x10)]);
        assert!(state.changed_since(0, 0x1800, 0x1000), "change 2 overlaps");
        assert!(!state.changed_since(0, 0x1000, 0x1000), "change 2 only touches");
        assert!(!state.changed_since(0, 0x3000, 0x1000));
        assert!(!state.changed_since(2, 0x1800, 0x1000), "nothing since 2");
        state.changed.pop_front();
        assert!(state.changed_since(0, 0x3000, 0x1000), "change 1 is no longer kept");
        // At most this many probes again for their age a frame.
        let tries: Vec<bool> = (0..4).map(|_| state.may_reprobe(7)).collect();
        assert_eq!(tries, [true, true, false, false]);
        assert!(state.may_reprobe(8));
    }

    /// The copy around remembered holes gives the usual bytes (the runs,
    /// zeros in the holes); a hole whose first page came back, or a run
    /// with a page gone, sends the range the usual way.
    #[test]
    fn the_copy_around_known_holes_reads_runs_and_zeros_holes() {
        const PAGES: usize = 12;
        let len = PAGES * PAGE;
        // SAFETY: a private region of ours, written only where committed.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let holes_at = [3, 4, 9];
            let mut expected = vec![0u8; len];
            for page in (0..PAGES).filter(|p| !holes_at.contains(p)) {
                let at = base.add(page * PAGE);
                assert_eq!(VirtualAlloc(at.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE), at.cast());
                for j in 0..PAGE {
                    let b = ((page * 31 + j * 7) % 251 + 1) as u8;
                    at.add(j).write(b);
                    expected[page * PAGE + j] = b;
                }
            }
            let holes = [(3 * PAGE, 5 * PAGE), (9 * PAGE, 10 * PAGE)];
            // What the usual ways record is those holes.
            for probing in [true, false] {
                let mut calls = crate::host::SparseCalls::default();
                let mut dst = vec![0xeeu8; len];
                if probing {
                    crate::host::read_probing(base, dst.as_mut_ptr(), len, 0, &mut calls);
                } else {
                    crate::host::read_sparse(base, dst.as_mut_ptr(), len, &mut calls);
                }
                assert!(dst == expected);
                assert_eq!(offsets(base as usize, len, &calls.holes), holes, "probing {probing}");
            }
            let mut dst = vec![0xeeu8; len];
            let mut reads = (0, 0);
            assert_eq!(around(dst.as_mut_ptr(), base, len, &holes, &mut reads), Ok(()));
            assert!(dst == expected, "the bytes differ from the usual way's");
            // Two holes tried (they fail); the runs by the kernel without
            // copies in place in tests.
            assert_eq!(reads.1, 2);
            // A hole's first page came back.
            let at = base.add(9 * PAGE);
            assert_eq!(VirtualAlloc(at.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE), at.cast());
            assert_eq!(around(dst.as_mut_ptr(), base, len, &holes, &mut reads), Err(false));
            // A page of a run went.
            assert_ne!(VirtualFree(base.add(6 * PAGE).cast(), PAGE, MEM_DECOMMIT), 0);
            let holes = [(3 * PAGE, 5 * PAGE)];
            assert_eq!(around(dst.as_mut_ptr(), base, len, &holes, &mut reads), Err(true));
            VirtualFree(base.cast(), 0, MEM_RELEASE);
        }
    }
}
