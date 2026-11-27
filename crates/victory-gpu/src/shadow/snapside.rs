//! VIcky (`VICTORY_SNAPSHOT_BESIDE=1|alt`; off by default): the snapshots
//! of a submission's shader-written ranges ([`super::take_snapshots`]) are
//! given to the threads kept for wide work where the submitting thread took
//! them, and are done by the time its lists go to the card. They run beside
//! the heap sync's join (`crate::descriptor::beside`), the settle for the
//! card's results and the copies in the card's memory (`Host::card_before`),
//! instead of before them.
//!
//! Profile PF4 (lever L6 of the road to 100 ms a frame): the snapshots were
//! 12.4 ms a frame of the submitting thread, its share of 177-213 MB of
//! copying, just before the join. In a submission that waits at the join
//! they cost nothing; in the others, and before `card_before` (6.7-8 ms,
//! after the join), they count whole.
//!
//! The order, in `crate::host`'s `Host::submit`:
//!
//! - Started where they were taken: the refreshes are done and every range
//!   of the submission is declared ([`super::Shadow::writing`] noted its
//!   snapshot then). From its declaring on, no refresh writes a declared
//!   range of a copy, and the title's own copies write title memory, not
//!   the copy ([`super::wide_back`]).
//! - Done before the submission's lists go to the card (either
//!   `ExecuteCommandLists`: sent ahead or waited for), which write the
//!   copies, and so before anything reads the snapshots: the compare after
//!   the card ([`super::find_all`]) and the copy-back of the submission's
//!   ranges.
//! - What runs meanwhile on the submitting thread: the join (descriptors,
//!   the heap map; no host copy), `make_waiting` (slots of textures), the
//!   settle when the card's results run low, and `card_before`. The settle
//!   lands flights sent ahead: their copy-backs write title memory and, of
//!   the copies and the snapshots, only their own declared ranges.
//!   `Shadow::writing` snapshots only the parts no declared range holds, so
//!   these are other bytes. A snapshot's memory is given back only when no
//!   range of its copy is declared any more (`Shadow::copy_back_done`), and
//!   this submission's ranges stay declared until its own copy-back. The
//!   card's copies go to copies in the card's memory, which take no
//!   snapshot, and the list `card_before` sends writes only those. (With
//!   the relay, `VICTORY_RELAY`, flights' copy-backs already ran on their
//!   own thread while the snapshots were taken.)
//! - The threads: the snapshots wait in a queue of their own (`later` of
//!   the kept threads) that only the kept threads take from, and only when
//!   the shared one is empty. The heap sync's page reads
//!   (`VICTORY_HEAP_SYNC_THREADS`) and the compares of a settle go first,
//!   and no caller of [`super::wide`] (the heap sync's thread among them)
//!   runs a snapshot while it waits for its own jobs. When the submitting
//!   thread comes for the snapshots it takes those not started yet, then
//!   waits for those running ([`Snapping`]'s drop).
//! - A submission that ends early (a panic) waits for its snapshots all the
//!   same, before it lets go of what they copy (the drop).
//!
//! What can cost: the copying shares the memory's speed with the join's
//! sync and the card's copies; the snapshots may end later than on the
//! submitting thread (one thread fewer), which shows as a wait on the
//! frame line's clause.
//!
//! `alt`: twenty seconds on the submitting thread, twenty beside, in turn,
//! each change said in the log with the counts of the phase before.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use super::{Job, SNAPS, copy_jobs, keeps, wide_back, workers};

/// `VICTORY_SNAPSHOT_BESIDE`: 0 off, 1 on, 2 in turn.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_SNAPSHOT_BESIDE").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => return 0,
        };
        if !wide_back() || keeps() {
            tracing::warn!(
                "VICTORY_SNAPSHOT_BESIDE is not used with VICTORY_WIDE_BACK=0 or VICTORY_KEEP_SNAPSHOTS=1: the snapshots \
                 are taken as their ranges are declared"
            );
            return 0;
        }
        tracing::info!(
            "the snapshots of a submission's shader-written ranges are taken on the wide threads, beside the heap \
             sync's join and the copies in the card's memory, and done before its lists go to the card \
             (VICTORY_SNAPSHOT_BESIDE={})",
            if mode == 1 { "1" } else { "alt: twenty seconds on the submitting thread, twenty beside" }
        );
        mode
    })
}

/// Whether this submission's snapshots go beside (`alt`: by the phase,
/// each change said with the counts of the phase before).
fn beside_now() -> bool {
    match mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: AtomicU8 = AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                let [n, thread, beside] = [0, 1, 2].map(|i| PHASE[i].swap(0, Ordering::Relaxed));
                let per = |us: u64| us as f64 / 1000.0 / n.max(1) as f64;
                tracing::info!(
                    "a submission's snapshots are taken beside the join from now (VICTORY_SNAPSHOT_BESIDE=alt): {}; \
                     since the change before: {n} submissions, {:.2} ms a submission of snapshots on the submitting \
                     thread, {:.2} ms a submission of their copying beside the join",
                    if on { "ON" } else { "off, on the submitting thread as usual" },
                    per(thread),
                    per(beside)
                );
            }
            on
        }
    }
}

/// Since the frame line before ([`clause`]): submissions whose snapshots
/// went beside, those of them done when the submitting thread came for
/// them, microseconds of their copying beside (from their start until the
/// submitting thread came, or their end if sooner), and microseconds the
/// submitting thread spent on them then (taking those not started,
/// waiting).
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Since the last change of `alt`'s phase: submissions, microseconds of
/// snapshots on the submitting thread, microseconds of copying beside.
static PHASE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Where the submitting thread took a submission's snapshots, after its
/// refreshes: takes them there as before, or (`VICTORY_SNAPSHOT_BESIDE`)
/// gives them to the kept threads and returns what they are waited for by.
pub(crate) fn take() -> Option<Snapping> {
    let started = Instant::now();
    if !beside_now() {
        super::take_snapshots();
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotUs, us);
        if mode() == 2 {
            PHASE[0].fetch_add(1, Ordering::Relaxed);
            PHASE[1].fetch_add(us, Ordering::Relaxed);
        }
        return None;
    }
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        PHASE[0].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let mut snapping = give(snaps);
    snapping.start_us = started.elapsed().as_micros() as u64;
    Some(snapping)
}

/// Before the submission's lists go to the card: its snapshots given
/// beside are done (taken here if not started, else waited for).
pub(crate) fn done(snapping: Option<Snapping>) {
    drop(snapping);
}

/// Jobs of one [`Snapping`] not done yet, and when the last one ended.
struct Left {
    n: Mutex<(usize, Option<Instant>)>,
    ended: Condvar,
}

impl Left {
    fn one_done(&self) {
        let mut n = self.n.lock().unwrap_or_else(|e| e.into_inner());
        n.0 -= 1;
        if n.0 == 0 {
            n.1 = Some(Instant::now());
            self.ended.notify_all();
        }
    }
}

/// A submission's snapshots given to the kept threads ([`take`]). Dropped
/// ([`done`]), the submitting thread takes the ones not started yet and
/// waits for the others.
pub(crate) struct Snapping {
    left: Arc<Left>,
    /// When the copying could start.
    queued: Instant,
    /// The submitting thread's microseconds to give them.
    start_us: u64,
}

/// `copies` (from, to, bytes) cut as [`super::take_snapshots`] cuts them,
/// queued for the kept threads after their shared work.
fn give(copies: Vec<(usize, usize, usize)>) -> Snapping {
    let jobs = copy_jobs(copies);
    let left = Arc::new(Left {
        n: Mutex::new((jobs.len(), None)),
        ended: Condvar::new(),
    });
    if jobs.is_empty() {
        left.n.lock().unwrap_or_else(|e| e.into_inner()).1 = Some(Instant::now());
    }
    let wrapped: Vec<Job> = jobs
        .into_iter()
        .map(|job| {
            let left = left.clone();
            Box::new(move || {
                job();
                left.one_done();
            }) as Job
        })
        .collect();
    let workers = workers();
    {
        // Under the shared queue's lock, which a kept thread holds from
        // finding both queues empty until it waits: none misses the word.
        let _shared = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        workers.later.lock().unwrap_or_else(|e| e.into_inner()).extend(wrapped);
        workers.more.notify_all();
    }
    Snapping {
        left,
        queued: Instant::now(),
        start_us: 0,
    }
}

impl Drop for Snapping {
    fn drop(&mut self) {
        let came = Instant::now();
        // Only snapshots wait there, one submission's at a time (the
        // submission lock): those not started yet are taken here.
        let workers = workers();
        loop {
            let job = workers.later.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
            match job {
                Some(job) => job(),
                None => break,
            }
        }
        let ended = {
            let mut n = self.left.n.lock().unwrap_or_else(|e| e.into_inner());
            while n.0 > 0 {
                n = self.left.ended.wait(n).unwrap_or_else(|e| e.into_inner());
            }
            n.1.unwrap_or(came)
        };
        let beside = ended.min(came).saturating_duration_since(self.queued).as_micros() as u64;
        let after = came.elapsed().as_micros() as u64;
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(u64::from(ended <= came), Ordering::Relaxed);
        STATS[2].fetch_add(beside, Ordering::Relaxed);
        STATS[3].fetch_add(after, Ordering::Relaxed);
        PHASE[0].fetch_add(1, Ordering::Relaxed);
        PHASE[1].fetch_add(self.start_us + after, Ordering::Relaxed);
        PHASE[2].fetch_add(beside, Ordering::Relaxed);
        // The frame line's snapshots: the submitting thread's part.
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotUs, self.start_us + after);
    }
}

/// The frame line's clause after "snapshots X ms" (empty with the switch
/// off): per frame, the ms of copying beside the join, in how many
/// submissions, how many were done when the submitting thread came for
/// them, and its ms then (part of the snapshots figure).
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn clause(frames: f64) -> String {
    let mode = mode();
    if mode == 0 {
        return String::new();
    }
    let [n, done, beside, after] = [0, 1, 2, 3].map(|i| STATS[i].swap(0, Ordering::Relaxed) as f64 / frames.max(1.0));
    format!(
        " (VICTORY_SNAPSHOT_BESIDE={}: {:.2} ms beside the join, in {:.1} submissions a frame, {:.1} of them done when \
         the submitting thread came for them; {:.2} ms of the snapshots figure is that thread taking those not started \
         and waiting)",
        if mode == 1 { "1" } else { "alt" },
        beside / 1000.0,
        n,
        done,
        after / 1000.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Source and room for `len` bytes, and the copies of `cuts` from one
    /// to the other at the same offsets.
    fn copies(len: usize, cuts: &[usize], seed: usize) -> (Vec<u8>, Vec<u8>, Vec<(usize, usize, usize)>) {
        let src: Vec<u8> = (0..len).map(|i| (i * 13 + seed + i / 4093) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        let list = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        (src, dst, list)
    }

    /// Every byte is copied once the snapping is done, whether the kept
    /// threads or this one took the jobs.
    #[test]
    fn snapshots_given_beside_are_all_copied_when_done() {
        // Small: the suite's tests that depend on timing feel the load.
        let len = 2 * super::super::WIDE_JOB + 777;
        let (src, dst, list) = copies(len, &[0, 3, 4096, super::super::WIDE_JOB + 5, len], 1);
        let snapping = give(list);
        done(Some(snapping));
        assert!(src == dst);
    }

    /// Snapshots given beside wait for no caller of `wide`: a `wide` call
    /// meanwhile ends with its own jobs, and the snapshots are all copied
    /// when done.
    #[test]
    fn a_wide_call_meanwhile_ends_with_its_own_jobs() {
        let len = 3 * super::super::WIDE_JOB;
        let (src, dst, list) = copies(len, &[0, len / 3, len], 7);
        let snapping = give(list);
        let ran = Arc::new(AtomicU64::new(0));
        let jobs: Vec<Job> = (0..12)
            .map(|_| {
                let ran = ran.clone();
                Box::new(move || {
                    ran.fetch_add(1, Ordering::Relaxed);
                }) as Job
            })
            .collect();
        super::super::wide(jobs);
        assert_eq!(ran.load(Ordering::Relaxed), 12);
        drop(snapping);
        assert!(src == dst);
    }

    /// Nothing to copy: done at once.
    #[test]
    fn nothing_given_is_done_at_once() {
        drop(give(Vec::new()));
    }
}
