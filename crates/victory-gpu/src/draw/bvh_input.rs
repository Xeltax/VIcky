//! VIcky (`VICTORY_BVH_INPUT_CHECK=1`; diagnostics, off by default): a check
//! of what the ray-tracing structure builder's rehydration programs are
//! given. Before each dispatch of the four (4ddd6f9702fab2a7 reads the
//! header, 5e68767498c57eb6 the leaves, caa78b8ca62bc3d6 and
//! 28918b7150a7aad5 the 16-bit and 32-bit boxes), every range their root
//! views read is compared, byte by byte, with title memory: the host copy
//! the runtime binds (a root view reads from its address to the end of the
//! copy) against the title's bytes.
//!
//! Why: the programs match the console exactly on synthetic data, so the
//! bad parent-link addresses of the chapter-1 save's loading screen
//! (`VICTORY_BVH_LINK_STORES=keep`, `VICKY-RT.md`) most likely come from
//! input that is not the title's bytes at that moment: a host copy that is
//! stale (streaming reused the address), or a range that is never
//! refreshed.
//!
//! Two comparisons per range, both of the host copy's bytes:
//!
//! - at recording, as the dispatch is recorded: the refresh the binding
//!   queued has not run, so a difference is expected unless the copy is
//!   already up to date (`refresh_queued` says whether a refresh covering
//!   the range is queued in the list);
//! - at submission, once the list's refresh of the copy has run
//!   (`crate::host`'s `Step::Inspect`): what the card will read, against
//!   title memory at that moment.
//!
//! A range a submission not yet copied back may be writing (`pending`) is
//! newer in the copy than in title memory: a difference there is not a
//! stale copy. The programs rehydrate in place (roots 1 and 2 are the same
//! address), so each dispatch's output is such a range for the next.
//!
//! Logged: the first 32 dispatches with a bad range (a difference at
//! submission, or at recording with no refresh queued), with each bad
//! range's counts; then, at the first dispatch of each frame, a line for
//! the frames since the last (the first 200 lines, then every 100th).
//!
//! shortcut: a root view has no size, so each range is up to 4 MiB; a
//! dispatch reads and compares all of it twice. Fine for a diagnostic run;
//! it costs frames.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::{Root, State};
use crate::host::HostList;
use crate::pipeline::host::HostPipeline;
use crate::shadow::input::{Compared, State as Copy};

/// The programs checked: shader hash and name.
const PROGRAMS: [(&str, &str); 4] = [
    ("4ddd6f9702fab2a7", "DecompressHeader"),
    ("5e68767498c57eb6", "LeafRehydration"),
    ("caa78b8ca62bc3d6", "Box16Rehydration"),
    ("28918b7150a7aad5", "Box32Rehydration"),
];

/// Dispatches logged with details.
const DETAILED: u32 = 32;

/// `VICTORY_BVH_INPUT_CHECK=1`.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_INPUT_CHECK").is_some_and(|v| v == "1"))
}

const TALLIES: usize = 10;
/// Dispatches checked; with a bad range; ranges compared; ranges with a
/// difference at recording; with one at submission; with one at recording
/// and no refresh queued; bytes compared at submission; bytes differing at
/// submission; ranges with a pending overlap; views without a host copy.
static TALLY: [AtomicU64; TALLIES] = [const { AtomicU64::new(0) }; TALLIES];

fn tally(i: usize, n: u64) {
    TALLY[i].fetch_add(n, Ordering::Relaxed);
}

/// One range a dispatch reads.
struct Range {
    views: String,
    address: u64,
    len: u64,
    copy: Copy,
    queued: bool,
    record: Compared,
    submit: Option<Compared>,
}

impl Range {
    /// A difference at submission, or at recording with no refresh queued.
    fn bad(&self) -> bool {
        self.submit.is_some_and(|c| c.differing != 0) || (self.record.differing != 0 && !self.queued)
    }

    fn text(&self) -> String {
        let one = |c: &Compared| match c.first {
            Some(at) => format!("{} of {} bytes differ, first at +{at:#x}", c.differing, c.bytes),
            None => format!("{} bytes equal", c.bytes),
        };
        let kind = match (self.copy.in_place, self.copy.carded) {
            (true, _) => " in place",
            (_, true) => " in the card's memory",
            _ => "",
        };
        let asking = self.copy.asking.map_or_else(|| "not asked ahead".to_owned(), |a| a.to_string());
        format!(
            "{} at {:#x} (copy {}{kind}, {:#x} bytes): at recording {}; at submission {}; refresh_queued={} \
             pending={:#x} pages_never_taken={}/{} taken_asking={}..{} asking_now={asking}",
            self.views,
            self.address,
            self.copy.id,
            self.len,
            one(&self.record),
            self.submit.as_ref().map_or_else(|| "not reached".to_owned(), one),
            self.queued,
            self.copy.pending,
            self.copy.never,
            self.copy.pages,
            self.copy.taken[0],
            self.copy.taken[1],
        )
    }
}

/// One dispatch, shared with its submission-time steps.
struct Dispatch {
    frame: u64,
    hash: &'static str,
    name: &'static str,
    groups: [u32; 3],
    ranges: Vec<Range>,
    /// Submission-time steps still to run.
    left: usize,
}

static DETAILS: AtomicU32 = AtomicU32::new(0);

impl Dispatch {
    /// Every range compared at submission: tallies and the log.
    fn done(&mut self) {
        self.left = self.ranges.len();
        for r in &self.ranges {
            let s = r.submit.unwrap_or_default();
            tally(6, s.bytes);
            tally(7, s.differing);
            tally(4, u64::from(s.differing != 0));
        }
        let bad: Vec<String> = self.ranges.iter().filter(|r| r.bad()).map(Range::text).collect();
        if !bad.is_empty() {
            tally(1, 1);
            if DETAILS.fetch_add(1, Ordering::Relaxed) < DETAILED {
                tracing::info!(
                    frame = self.frame,
                    program = self.hash,
                    name = self.name,
                    groups = ?self.groups,
                    ranges = self.ranges.len(),
                    bad_ranges = bad.len(),
                    detail = %bad.join(" | "),
                    "VICTORY_BVH_INPUT_CHECK: a dispatch of a builder rehydration program reads a range whose host copy differs from title memory (the first 32 such dispatches)"
                );
            }
        }
        for r in &mut self.ranges {
            r.submit = None;
        }
    }
}

impl State {
    /// A dispatch of `p` is bound and about to be recorded in `list`: if it
    /// is one of the builder's rehydration programs and the check is on,
    /// its ranges are compared now, and again at submission.
    pub(super) fn bvh_input_check(&self, list: &HostList, p: &HostPipeline, groups: [u32; 3]) {
        if !on() {
            return;
        }
        let Some(stage) = p.stages.first() else {
            return;
        };
        let Some(&(hash, name)) = PROGRAMS.iter().find(|(h, _)| *h == stage.shader.hash) else {
            return;
        };
        frame_line();
        let roots = self.roots_ref(true);
        let mut ranges: Vec<Range> = Vec::new();
        let mut shadows = Vec::new();
        for (kind, slots) in [('s', &stage.root_srvs), ('u', &stage.root_uavs)] {
            for &(slot, _) in slots {
                let Some(&Root::View(address)) = roots.get(usize::from(slot)) else {
                    continue;
                };
                let views = format!("{kind}{slot}");
                if let Some(r) = ranges.iter_mut().find(|r| r.address == address) {
                    r.views = format!("{}+{views}", r.views);
                    continue;
                }
                let Some(shadow) = crate::shadow::find(address, 1) else {
                    tally(9, 1);
                    continue;
                };
                let len = shadow.left(address);
                ranges.push(Range {
                    views,
                    address,
                    len,
                    copy: shadow.input_state(address, len),
                    queued: list.refresh_queued(&shadow, address as usize, len as usize),
                    record: shadow.input_compare(address, len),
                    submit: None,
                });
                shadows.push(shadow);
            }
        }
        tally(0, 1);
        tally(2, ranges.len() as u64);
        for r in &ranges {
            tally(3, u64::from(r.record.differing != 0));
            tally(5, u64::from(r.record.differing != 0 && !r.queued));
            tally(8, u64::from(r.copy.pending != 0));
        }
        if ranges.is_empty() {
            return;
        }
        let at: Vec<(u64, u64)> = ranges.iter().map(|r| (r.address, r.len)).collect();
        let left = ranges.len();
        let dispatch = Arc::new(Mutex::new(Dispatch {
            frame: crate::frame::current(),
            hash,
            name,
            groups,
            ranges,
            left,
        }));
        for (i, ((address, len), shadow)) in at.into_iter().zip(shadows).enumerate() {
            let (dispatch, held) = (dispatch.clone(), shadow.clone());
            list.defer_inspect(&shadow, move || {
                let compared = held.input_compare(address, len);
                let mut d = dispatch.lock().unwrap_or_else(|e| e.into_inner());
                d.ranges[i].submit = Some(compared);
                d.left -= 1;
                if d.left == 0 {
                    d.done();
                }
            });
        }
    }
}

/// At the first checked dispatch of a frame: a line for the frames since
/// the last. The first 200 lines, then every 100th.
fn frame_line() {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static BEFORE: Mutex<[u64; TALLIES]> = Mutex::new([0; TALLIES]);
    static LINES: AtomicU32 = AtomicU32::new(0);
    let frame = crate::frame::current();
    if FRAME.swap(frame, Ordering::Relaxed) == frame {
        return;
    }
    let mut before = BEFORE.lock().unwrap_or_else(|e| e.into_inner());
    let mut d = [0u64; TALLIES];
    for (i, t) in TALLY.iter().enumerate() {
        let now = t.load(Ordering::Relaxed);
        d[i] = now.saturating_sub(before[i]);
        before[i] = now;
    }
    if d[0] == 0 {
        return;
    }
    let line = LINES.fetch_add(1, Ordering::Relaxed);
    if line >= 200 && !line.is_multiple_of(100) {
        return;
    }
    tracing::info!(
        frame,
        dispatches = d[0],
        dispatches_with_bad_range = d[1],
        ranges = d[2],
        differ_at_recording = d[3],
        differ_at_submission = d[4],
        differ_at_recording_no_refresh_queued = d[5],
        submission_mib_compared = d[6] >> 20,
        submission_bytes_differing = d[7],
        ranges_with_pending_writes = d[8],
        views_without_copy = d[9],
        "VICTORY_BVH_INPUT_CHECK, since the last line: dispatches of the builder's rehydration programs (a bad range differs at submission, or at recording with no refresh queued; submission counts lag a frame)"
    );
}
