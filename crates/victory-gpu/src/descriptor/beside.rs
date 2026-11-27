//! VIcky: the heap sync ([`super::sync_heaps`]) on a thread of its own,
//! beside a submission's refreshes (`VICTORY_HEAP_SYNC_BESIDE=1`; off by
//! default, and not used with levels on demand, see below).
//!
//! The submitting thread did, one after the other: the heap sync (some
//! 45 ms a frame with the switches of the cheaper sync, RT40b), the
//! sweep, and then the asking of title memory for the whole submission,
//! the steps of the lists (refreshes of host copies, the title's buffer
//! copies, texture checks), the last refreshes and the snapshots (the
//! frame line's copies and snapshots, some 80 to 95 ms). The sync has to
//! be done only before the lists run. With the switch it starts on its
//! own thread where the refreshes begin, and the submitting thread waits
//! for it after the snapshots: about the larger of the two instead of
//! their sum.
//!
//! The order with the switch, in `crate::host`'s `Host::submit`:
//!
//! 1. the settle of flights at the start, the sweep, `make_waiting`, the
//!    submission counted as started (`STARTED`, `shadow::preparing`), as
//!    before, without the sync before them;
//! 2. the sync given to its thread ([`start`]), which asks for the changed
//!    pages of every heap and says so before it reads the first of them
//!    ([`super::sync_heaps_told`]); the submitting thread waits for that;
//! 3. the submission's asking (`watch::ask_once`), the steps of the lists,
//!    the last refreshes, the snapshots, while the sync reads its pages
//!    and mirrors the records that changed;
//! 4. the join ([`Beside::join`]), and `make_waiting` again;
//! 5. as before: the settle when the card's results run low, the copies
//!    in the card's memory, `ExecuteCommandLists`.
//!
//! VIcky (`VICTORY_SNAPSHOT_BESIDE=1`): the snapshots of 3 are only given
//! to the threads kept for wide work there, and are done just before
//! `ExecuteCommandLists`: they run beside 4 and 5 (`crate::shadow::snapside`
//! says why that order holds).
//!
//! What the sync shares with the rest of a submission, and how the order
//! is kept:
//!
//! - The host heaps behind the title's: their slots, what each slot keeps,
//!   the prints of records seen, the sets of buffer and texture slots, the
//!   spans of placed textures, the stride table, the heap map and its
//!   pool. The sync writes them under their locks, as the title's threads
//!   do at any moment when they make or copy views, and as draws do when
//!   they read tables. Of the steps of 3 none reads them, but with levels
//!   on demand: a texture check's detiling judges a texture's levels by
//!   the views in slots (`texture::Staging::judgement`), and the making of
//!   a slot reads the level such a texture is shown from
//!   (`resource::first_shown`), which those steps change. Levels on demand
//!   exist only with `crate::tiles::wanted`; the switch is then not used,
//!   and the sync runs where it did.
//! - The lists read the slots when they run: the join comes before
//!   `ExecuteCommandLists`. What the sync wrote for the card (slots, heap
//!   map entries, strides: memory the card reads, write-combined) is out
//!   before the join returns: a fence after the sync empties its thread's
//!   write-combining buffers, and the join's lock orders the rest.
//! - Slots left waiting for a texture's host copy (`super::wait_for`) that
//!   another thread made since: the sync hands those textures to
//!   `make_waiting`, which came after it and now comes before it. It runs
//!   again after the join, so they are made before the lists run, as they
//!   were.
//! - The sweep lets go of host copies of title memory, staging buffers and
//!   host textures that no list and no slot holds and that nothing asked
//!   for in seconds. The sync makes slots on host textures
//!   (`resource::host_resource` hands one out without holding it, kept
//!   safe from the sweep by that idle time alone) and lets slots go of what
//!   they held. The two never run at once: the sync starts after the sweep
//!   and is joined in the same submission, under the submission lock,
//!   before the next sweep. Their order is now sweep then sync, where it
//!   was sync then sweep: the order of one submission's sweep and the next
//!   one's sync, which the sync always met. A slot the sync makes on a
//!   host texture the sweep just let go of waits for the texture's next
//!   host copy (`super::wait_for`), as a view the title writes after a
//!   sweep does; what the sync's slots let go of goes at the next sweep,
//!   at most a second later.
//! - The submission counts as started while the sync runs
//!   (`crate::host::submissions`): a host resource a slot lets go of waits
//!   for this submission's end (it was dropped at once when nothing ran),
//!   and a pool slot of the heap map given back is handed out after it.
//!   Later, never earlier; nothing of this submission can read them after
//!   the join.
//! - The journal of written pages (`crate::watch`). The sync asks for its
//!   heaps' pages (a kernel asking that clears them) before the
//!   submission's own asking, as before: step 2 waits for that. Asked
//!   while the submission's asking stands, a page found written would have
//!   a number above that asking's, and a host copy that holds it would
//!   take it again at every refresh of the submission (as in
//!   fxceless/victory-fork#410). The journal is behind one lock, and
//!   every asking notes what it found for all askers, each of which keeps
//!   what it has seen: no write is lost to another asker.
//! - Title memory of the heaps. The title's threads write it at any time;
//!   our steps may too (a title buffer copy, `CopyMemoryToMemoryX`, the
//!   copy-back of a settle). What they wrote while a submission was
//!   prepared was taken by the next submission's sync; now this one may
//!   read it already. A record read while it is being written is read
//!   again at the next sync: its page counts as written after the asking,
//!   and its print differs.
//! - The texture pass's queues of changed memory (`host::remapped`,
//!   `host::page_written`), filled by copy-backs and by the runtime, skip
//!   memory outside every span of a placed texture so far (`SPANNED`),
//!   which the sync widens. A change dropped for that concerns only slots
//!   the sync notes now as written, which are due anyway.
//! - The threads kept for wide work (`shadow::wide_run`): the sync's page
//!   reads (`VICTORY_HEAP_SYNC_THREADS`) and the snapshots share their
//!   queue. Each caller waits for its own jobs and may run another's
//!   meanwhile; no job waits for a caller. The sync itself runs on a
//!   thread of its own ([`Lane`]) and never in that queue: there the
//!   submitting thread, waiting for its snapshots, could take it and run
//!   it whole. Snapshots given beside (`VICTORY_SNAPSHOT_BESIDE`) wait in
//!   a queue of their own, which the kept threads take only when the shared
//!   one is empty and no caller of `wide` takes: the sync's page reads go
//!   first.
//! - The list of heaps (read-locked for the whole sync) and one sync at a
//!   time (`SYNCING`): as before, from the asking to the last slot on one
//!   thread. A heap the title releases waits for the sync's end, as it did.
//! - The locks the submitting thread holds while it waits (steps 2 and 4):
//!   the submission lock alone, which nothing the sync calls takes (on the
//!   submitting thread that would have deadlocked before).
//! - Thread-local state: the retirements of a chunk (`RETIRE_LATER`) are
//!   kept and taken on the thread that mirrors; the tests' own switches per
//!   thread (`TEST_FAST`, `TEST_MAP`) are not seen by the sync's thread, so
//!   tests that set them call the sync in place.
//!
//! What overlaps: the sync's page reads and mirroring with the steps of 3,
//! the bulk of the submitting thread's preparing. Not the sweep, which
//! must not meet it, nor the settle for the card's results and the copies
//! in the card's memory after the join (both short). Only the kernel
//! askings of the heaps' pages stay ahead (one per heap).
//!
//! The sync line (`super::sync_line`) says, with the switch, how long the
//! submitting thread waited for those askings and at the join, against
//! the sync's own time on its thread. The frame line's preparing leaves
//! the wait at the join out, as it left the whole sync out before; the
//! submitting thread counts as at work while it waits, as while it synced.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// Whether the heap sync runs beside the refreshes
/// (`VICTORY_HEAP_SYNC_BESIDE=1`, with the sync on, and without levels on
/// demand).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !std::env::var_os("VICTORY_HEAP_SYNC_BESIDE").is_some_and(|v| v == "1") || !super::sync_on() {
            return false;
        }
        if crate::tiles::wanted() {
            tracing::warn!(
                "VICTORY_HEAP_SYNC_BESIDE=1 is not used with levels on demand (VICTORY_LEVELS_ON_DEMAND=1): \
                 the heap sync stays on the submitting thread, before the sweep"
            );
            return false;
        }
        tracing::info!("the heap sync runs on a thread of its own, beside the refreshes (VICTORY_HEAP_SYNC_BESIDE=1)");
        true
    })
}

/// Since the last sync line: syncs joined, and the microseconds the
/// submitting thread waited for their askings, waited at their joins, and
/// that they ran on their thread.
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The sync line's part (empty while no sync ran beside).
pub(crate) fn part(frames: f64) -> String {
    let [syncs, asking, waited, own] = [0, 1, 2, 3].map(|i| COUNTS[i].swap(0, Relaxed));
    if syncs == 0 {
        return String::new();
    }
    let ms = |us: u64| us as f64 / 1000.0 / frames;
    format!(
        " (VICTORY_HEAP_SYNC_BESIDE: {:.1} syncs a frame on a thread of their own, {:.2} ms of their own; the submitting \
         thread waited {:.2} ms for the askings of the heaps' pages and {:.2} ms at the join after the snapshots, so about \
         {:.2} ms ran beside the refreshes)",
        syncs as f64 / frames,
        ms(own),
        ms(asking),
        ms(waited),
        ms(own.saturating_sub(asking + waited)),
    )
}

/// A sync started beside a submission's refreshes ([`start`]): joined
/// ([`Beside::join`]) before anything reads the host heaps. One dropped
/// without a join is waited for all the same: a submission that ended
/// early does not let go of its lock while its sync runs.
pub(crate) struct Beside {
    ticket: Option<Ticket>,
}

/// Starts the heap sync on its thread, and returns once it has asked for
/// the changed pages of every heap. Without a thread (none could be
/// made) the sync is done here, whole.
pub(crate) fn start() -> Beside {
    static LANE: OnceLock<Option<Arc<Lane>>> = OnceLock::new();
    let lane = LANE.get_or_init(|| {
        let lane = Lane::new("victory-heap-sync");
        if lane.is_none() {
            tracing::warn!("no thread for the heap sync (VICTORY_HEAP_SYNC_BESIDE=1): it runs on the submitting thread");
        }
        lane
    });
    match lane {
        Some(lane) => begin(lane, |asked| super::sync_heaps_told(Some(asked))),
        None => {
            super::sync_heaps();
            Beside { ticket: None }
        }
    }
}

/// Gives `sync` to `lane`, with what it calls once its pages are asked
/// for, and returns when it has called that (or has ended).
pub(super) fn begin(lane: &Lane, sync: impl FnOnce(&dyn Fn()) + Send + 'static) -> Beside {
    let asking = Instant::now();
    let ticket = lane.give(Box::new(move |ticket: &Ticket| sync(&|| ticket.tell(|p| p.asked = true))));
    drop(ticket.wait_until(|p| p.asked || p.done));
    COUNTS[1].fetch_add(asking.elapsed().as_micros() as u64, Relaxed);
    Beside { ticket: Some(ticket) }
}

impl Beside {
    /// Waits for the sync to end, and returns how long this thread waited.
    /// A panic of the sync goes on here, as on the submitting thread before.
    pub(crate) fn join(mut self) -> Duration {
        let Some(ticket) = self.ticket.take() else {
            return Duration::ZERO;
        };
        let waiting = Instant::now();
        let (took, panic) = {
            let mut progress = ticket.wait_until(|p| p.done);
            (progress.took, progress.panic.take())
        };
        let waited = waiting.elapsed();
        COUNTS[0].fetch_add(1, Relaxed);
        COUNTS[2].fetch_add(waited.as_micros() as u64, Relaxed);
        COUNTS[3].fetch_add(took, Relaxed);
        if let Some(panic) = panic {
            std::panic::resume_unwind(panic);
        }
        waited
    }
}

impl Drop for Beside {
    fn drop(&mut self) {
        if let Some(ticket) = self.ticket.take() {
            drop(ticket.wait_until(|p| p.done));
        }
    }
}

/// What a job given to a [`Lane`] has done, for the thread that waits.
#[derive(Default)]
struct Progress {
    /// The job said that its first part is done ([`begin`]'s `asked`).
    asked: bool,
    /// The job ended: its microseconds, and its panic if it had one.
    done: bool,
    took: u64,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

/// One job's [`Progress`], shared by the lane and the waiting thread.
#[derive(Clone, Default)]
pub(crate) struct Ticket(Arc<(Mutex<Progress>, Condvar)>);

impl Ticket {
    /// Changes the progress, and wakes whoever waits.
    fn tell(&self, change: impl FnOnce(&mut Progress)) {
        let (progress, moved) = &*self.0;
        change(&mut progress.lock().unwrap_or_else(|e| e.into_inner()));
        moved.notify_all();
    }

    /// Waits until the progress is `there`.
    fn wait_until(&self, there: impl Fn(&Progress) -> bool) -> MutexGuard<'_, Progress> {
        let (progress, moved) = &*self.0;
        let mut now = progress.lock().unwrap_or_else(|e| e.into_inner());
        while !there(&now) {
            now = moved.wait(now).unwrap_or_else(|e| e.into_inner());
        }
        now
    }
}

pub(crate) type Job = Box<dyn FnOnce(&Ticket) + Send>;

/// A thread kept for one kind of work, which does the jobs given to it one
/// after the other (also the pass over buffer views beside,
/// `crate::draw::heap_pass`, on a lane of its own).
pub(crate) struct Lane {
    jobs: Mutex<VecDeque<(Job, Ticket)>>,
    more: Condvar,
}

impl Lane {
    /// A lane and its thread, named `name`; `None` if no thread could be
    /// made. The thread stays for the run.
    pub(crate) fn new(name: &str) -> Option<Arc<Lane>> {
        let lane = Arc::new(Lane {
            jobs: Mutex::new(VecDeque::new()),
            more: Condvar::new(),
        });
        let kept = lane.clone();
        std::thread::Builder::new().name(name.to_owned()).spawn(move || kept.serve()).ok()?;
        Some(lane)
    }

    /// The lane's thread: each job in turn, timed, its panic kept for the
    /// one who waits for it.
    fn serve(&self) {
        // VIcky (`VICTORY_HELPER_PRIORITY`, `crate::helpers`).
        let mut priority = crate::helpers::Priority::new();
        loop {
            priority.keep();
            let (job, ticket) = {
                let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(next) = jobs.pop_front() {
                        break next;
                    }
                    jobs = self.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                }
            };
            let started = Instant::now();
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&ticket)));
            // What the job wrote into memory the card reads (descriptors,
            // heap map entries, strides; write-combined) leaves this
            // thread's buffers before the waiting thread is told.
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            let took = started.elapsed().as_micros() as u64;
            ticket.tell(|p| {
                p.done = true;
                p.took = took;
                p.panic = ran.err();
            });
        }
    }

    /// Gives `job` to the lane.
    pub(crate) fn give(&self, job: Job) -> Ticket {
        let ticket = Ticket::default();
        self.jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back((job, ticket.clone()));
        self.more.notify_one();
        ticket
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32};
    use std::sync::mpsc;

    fn lane() -> Arc<Lane> {
        Lane::new("victory-heap-sync-test").expect("a thread")
    }

    fn thread_id() -> u32 {
        crate::host::thread_id()
    }

    /// The caller goes on once the job has asked, not before; the job then
    /// runs on the lane's thread until the join, which waits for its end.
    #[test]
    fn the_caller_goes_on_after_the_asking_and_the_join_waits_for_the_end() {
        let lane = lane();
        let before_asking = Arc::new(AtomicBool::new(false));
        let ended = Arc::new(AtomicBool::new(false));
        let on = Arc::new(AtomicU32::new(0));
        let (go, wait) = mpsc::channel::<()>();
        let beside = {
            let (before_asking, ended, on) = (before_asking.clone(), ended.clone(), on.clone());
            begin(&lane, move |asked| {
                on.store(thread_id(), Relaxed);
                before_asking.store(true, Relaxed);
                asked();
                // Held until the caller has looked.
                let _ = wait.recv();
                ended.store(true, Relaxed);
            })
        };
        assert!(before_asking.load(Relaxed), "the caller went on before the asking");
        assert!(!ended.load(Relaxed), "the job ended before it was let go");
        assert_ne!(on.load(Relaxed), thread_id(), "the job ran on the caller's thread");
        go.send(()).expect("the job waits");
        let _ = beside.join();
        assert!(ended.load(Relaxed), "the join came back before the job ended");
    }

    /// A job that ends without saying it has asked lets the caller go all
    /// the same.
    #[test]
    fn a_job_that_never_asks_lets_the_caller_go() {
        let lane = lane();
        let ran = Arc::new(AtomicBool::new(false));
        let beside = {
            let ran = ran.clone();
            begin(&lane, move |_| ran.store(true, Relaxed))
        };
        let _ = beside.join();
        assert!(ran.load(Relaxed));
    }

    /// A panic on the lane comes back at the join, and the lane goes on to
    /// the next job.
    #[test]
    fn a_panic_beside_comes_back_at_the_join() {
        let lane = lane();
        let beside = begin(&lane, |asked| {
            asked();
            panic!("a sync that fails");
        });
        let joined = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| beside.join()));
        assert!(joined.is_err(), "the panic was lost");
        let ran = Arc::new(AtomicBool::new(false));
        let beside = {
            let ran = ran.clone();
            begin(&lane, move |asked| {
                asked();
                ran.store(true, Relaxed);
            })
        };
        let _ = beside.join();
        assert!(ran.load(Relaxed), "the lane stopped after a panic");
    }

    /// One dropped without a join is waited for.
    #[test]
    fn a_sync_beside_dropped_without_a_join_is_waited_for() {
        let lane = lane();
        let ended = Arc::new(AtomicBool::new(false));
        let beside = {
            let ended = ended.clone();
            begin(&lane, move |asked| {
                asked();
                std::thread::sleep(Duration::from_millis(20));
                ended.store(true, Relaxed);
            })
        };
        drop(beside);
        assert!(ended.load(Relaxed));
    }

    /// Jobs on one lane follow one another: the next one asks only after
    /// the one before has ended.
    #[test]
    fn syncs_on_one_lane_follow_one_another() {
        let lane = lane();
        let order = Arc::new(Mutex::new(Vec::new()));
        let first = {
            let order = order.clone();
            begin(&lane, move |asked| {
                asked();
                std::thread::sleep(Duration::from_millis(20));
                order.lock().unwrap().push("first ends");
            })
        };
        let second = {
            let order = order.clone();
            begin(&lane, move |asked| {
                order.lock().unwrap().push("second asks");
                asked();
            })
        };
        let _ = (first.join(), second.join());
        assert_eq!(*order.lock().unwrap(), ["first ends", "second asks"]);
    }

    /// The line's part: empty without syncs; with them, per frame.
    #[test]
    fn the_line_part_is_per_frame_and_taken() {
        let lane = lane();
        // Other tests count too: whatever they left is taken first.
        let _ = part(1.0);
        let beside = begin(&lane, |asked| {
            asked();
            std::thread::sleep(Duration::from_millis(5));
        });
        let _ = beside.join();
        let text = part(2.0);
        assert!(text.contains("VICTORY_HEAP_SYNC_BESIDE"), "{text}");
        assert!(text.contains(" ms of their own"), "{text}");
    }
}
