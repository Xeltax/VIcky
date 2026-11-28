//! VIcky, structural stage 3 (`VICTORY_SUBMIT_LANES=1|alt`; off by default;
//! VICKY-RT.md "Stage 3: preparation on lanes"): two parts of the
//! submitting thread's work go to threads that wait for work anyway.
//!
//! Profile PB1 (boat, `VICTORY_TABLES_LEFT=1`; `xcl-notes`
//! `fable-2026-11-28-wall2.md`): the frame is bound by the submitting
//! thread's own work, 128 ms busy of 167. Of that, the texture refreshes
//! (`crate::texture::Staging::refresh`, 31 ms a frame, 24 of them in
//! ReadProcessMemory) and the last drops of what the sweep lets go of
//! (`resource::let_go_idle`, `shadow::let_go_idle`, `host::retire`: some
//! 6 ms of the card's objects destroyed) run on it, while the wide threads
//! (`victory-wide-N`) are idle 92 % of the time.
//!
//! **The refreshes.** At a list's texture step (`Step::Texture`, once a
//! submission per staging buffer until a step writes title memory) the
//! refresh is given to the wide threads, after their shared work, as the
//! snapshots given beside are (`crate::shadow::snapside`), in batches of
//! [`BATCH`]. Each staging buffer is its own: a refresh reads title memory
//! and the write watch, and writes its staging buffer and its own state
//! under its own lock. The order the picture depends on is kept:
//!
//! - Before a `Run` step, which may write title memory anywhere, every
//!   refresh given is done ([`Lanes::join`]). Before a title buffer copy
//!   (`Step::Copy`, 24 parts of submissions a frame on the boat), the
//!   refreshes whose texture shares a 64 KiB journal page with the copy's
//!   destination are, and all of them if that destination is known to show
//!   pages other addresses show too ([`Lanes::before_copy`]). So a refresh
//!   reads what it would have read in turn, before the write, and the
//!   texture's later steps check it again after the write, as before.
//! - Before a step that reads or writes a staging buffer's state
//!   (`Step::TextureHost`: the upload's "uploaded, not dirty" and the levels
//!   on demand given their memory from what the detiling held), that
//!   buffer's refresh is done ([`Lanes::wait_for`]): the flags end as they
//!   would have, so the next lists record the same uploads.
//! - At the end of the preparing, before the settle when the card's results
//!   run low (its copy-backs write title memory) and before the card's copies
//!   and the lists go (`Host::card_before`), every refresh is done
//!   ([`Lanes::finish`]). Meanwhile the submitting thread goes on: the rest of
//!   the steps, the refreshes of host copies, the snapshots, the heap sync's
//!   join.
//! - A refresh no wide thread has begun by then is run by the submitting
//!   thread itself, so a busy pool never makes it wait for more than the
//!   refreshes already running.
//!
//! The texture table's lock (`texture::STAGING`, which recording threads
//! wait on) is not taken by any of this. A wide thread reads a texture into
//! its own read buffer (`texture::with_read`), so up to one such buffer per
//! wide thread, the size of the largest texture it read through the kernel.
//!
//! **The drops.** In the ON phases, an object `host::retire` would drop at
//! once (no submission running: the sweep's host copies of windows and of
//! textures, and what the heap's slots let go of then) waits instead in a
//! list that the next submission hands to the reaper thread right after its
//! lists went to the card ([`sent`]); the staging buffers the sweep lets go
//! of (`texture::let_go_idle`) and the twins of the host textures it lets
//! go of too, so that nothing is destroyed under the texture table's lock.
//! Nothing names those objects any more (that is why the sweep took them),
//! so their destruction may come later and on another thread. The reaper
//! thread is made for this if `VICTORY_REAPER` is not set.
//!
//! **`alt`**: twenty seconds off, twenty on, with the `... (VICTORY_SUBMIT_LANES=alt): ON`
//! / `off, as usual` lines `alt-fps.sh` reads; the phase is taken at the
//! start of each submission. **The summary**, one a phase (every 20 s with
//! `1`), ON against off ([`said`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::Acquire, Ordering::Relaxed, Ordering::Release};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// The switch.
pub(crate) static SUBMIT_LANES: crate::alt::Alt = crate::alt::Alt::new(
    "VICTORY_SUBMIT_LANES",
    "a submission's texture refreshes run on the wide threads, joined before the card's copies, and the host's last \
     drops on the reaper after the lists go (structural stage 3)",
);

/// Refreshes given are queued for the wide threads this many at a time (and
/// at the end of each list's steps), not one by one.
const BATCH: usize = 4;

const QUEUED: u8 = 0;
const RUNNING: u8 = 1;
const DONE: u8 = 2;

type Work = Box<dyn FnOnce() + Send>;

/// A submission's refreshes given to the lanes and not done yet, and
/// whether the submitting thread waits for one.
#[derive(Default)]
struct Left {
    n: usize,
    waiting: bool,
}

/// What a submission's refreshes given to the lanes share.
#[derive(Default)]
struct Group {
    left: Mutex<Left>,
    ended: Condvar,
    /// Microseconds and refreshes run by the wide threads.
    lane_us: AtomicU64,
    lane_n: AtomicU64,
}

/// One refresh given to the lanes: run once, by a wide thread, or by the
/// submitting thread when it comes for it first.
struct Task {
    state: AtomicU8,
    work: Mutex<Option<Work>>,
    group: Arc<Group>,
}

impl Task {
    /// Runs the task if nobody took it yet (`lane`: on a wide thread, counted
    /// there); returns the microseconds it took, `None` if it was taken.
    fn run(&self, lane: bool) -> Option<u64> {
        if self.state.compare_exchange(QUEUED, RUNNING, Acquire, Relaxed).is_err() {
            return None;
        }
        let work = self.work.lock().unwrap_or_else(|e| e.into_inner()).take();
        let began = Instant::now();
        if let Some(work) = work
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).is_err()
        {
            tracing::error!("a texture refresh given to the wide threads panicked (VICTORY_SUBMIT_LANES)");
        }
        let us = began.elapsed().as_micros() as u64;
        if lane {
            self.group.lane_us.fetch_add(us, Relaxed);
            self.group.lane_n.fetch_add(1, Relaxed);
        }
        self.state.store(DONE, Release);
        let mut left = self.group.left.lock().unwrap_or_else(|e| e.into_inner());
        left.n -= 1;
        if left.waiting {
            self.group.ended.notify_all();
        }
        Some(us)
    }

    fn done(&self) -> bool {
        self.state.load(Acquire) == DONE
    }
}

/// This submission's counts, gone to its phase at [`Lanes::finish`].
#[derive(Default)]
struct Counts {
    refreshes: u64,
    submit_us: u64,
    joins: u64,
    waits: u64,
    waited_us: u64,
}

/// A refresh given: its staging buffer (by address), the journal's pages
/// (`crate::watch::PAGE`, 64 KiB, also the guards' blocks) its texture's
/// memory lies in, first and past the last, and the task.
struct Given {
    key: usize,
    pages: (usize, usize),
    task: Arc<Task>,
}

/// The journal's pages `[at, at + len)` lies in: first, and past the last.
fn pages(at: usize, len: usize) -> (usize, usize) {
    let page = crate::watch::PAGE;
    (at / page, at.saturating_add(len).div_ceil(page))
}

/// A submission's refreshes on the lanes (in `crate::host`'s `Batch`).
#[derive(Default)]
pub(crate) struct Lanes {
    /// This submission gives its refreshes to the wide threads (the phase
    /// at its start).
    on: bool,
    /// The switch is set (`1` or `alt`): what is done is counted.
    set: bool,
    group: Option<Arc<Group>>,
    /// Given since the last join, in order, by staging buffer.
    given: Vec<Given>,
    keys: HashMap<usize, usize>,
    /// Of `given`, those before this are queued for the wide threads.
    queued: usize,
    counts: Counts,
}

impl Lanes {
    /// At the start of a submission's preparing: its phase, said and counted.
    pub(crate) fn begin() -> Self {
        if !SUBMIT_LANES.set() {
            return Self::default();
        }
        let on = SUBMIT_LANES.on();
        LIVE.store(on, Relaxed);
        period_tick(on);
        add(row(), P::Submissions, 1);
        Self::new(on, true)
    }

    fn new(on: bool, set: bool) -> Self {
        let mut lanes = Self::default();
        (lanes.on, lanes.set) = (on, set);
        lanes
    }

    /// The refresh of `staging` at its step: given to the lanes in the ON
    /// phases, else done here, as before.
    pub(crate) fn refresh(&mut self, staging: &Arc<crate::texture::Staging>) {
        if !self.set {
            staging.refresh();
            return;
        }
        if self.on {
            let key = Arc::as_ptr(staging) as *const () as usize;
            let memory = staging.memory();
            let staging = staging.clone();
            self.give(key, memory, Box::new(move || staging.refresh()));
            return;
        }
        let began = Instant::now();
        staging.refresh();
        self.counts.refreshes += 1;
        self.counts.submit_us += began.elapsed().as_micros() as u64;
    }

    /// `work`, the refresh of staging buffer `key` from the title memory
    /// `memory` (address, bytes), given to the lanes.
    fn give(&mut self, key: usize, (at, len): (usize, usize), work: Work) {
        self.counts.refreshes += 1;
        // Given again before a join (a title copy elsewhere made it due
        // again while its refresh ran): after the first one.
        if self.keys.contains_key(&key) {
            self.wait_for(key);
        }
        let group = self.group.get_or_insert_with(Default::default).clone();
        group.left.lock().unwrap_or_else(|e| e.into_inner()).n += 1;
        let task = Arc::new(Task { state: AtomicU8::new(QUEUED), work: Mutex::new(Some(work)), group });
        self.keys.insert(key, self.given.len());
        self.given.push(Given { key, pages: pages(at, len), task });
        if self.given.len() - self.queued >= BATCH {
            self.flush();
        }
    }

    /// Before a title buffer copy into `[dst, dst + len)` (`Step::Copy`): the
    /// refreshes given whose texture memory shares a journal page with it
    /// are done. The others go on: the copy writes none of their bytes, and
    /// neither its note in the write-watch journal (`crate::watch::wrote`)
    /// nor a guard it trips touches their pages, so what they read and what
    /// the journal tells them are what they would have been in turn. A
    /// destination known to show physical pages other addresses show too
    /// (`crate::watch::shown_elsewhere`) may write a texture elsewhere: then
    /// every refresh given is done first.
    pub(crate) fn before_copy(&mut self, dst: usize, len: usize) {
        // (Everything given is done: forgotten, so the lists stay short.)
        if self.given.iter().all(|g| g.task.done()) {
            self.given.clear();
            self.keys.clear();
            self.queued = 0;
            return;
        }
        let (first, past) = pages(dst, len);
        let (mut hit, mut rest) = (Vec::new(), false);
        for g in self.given.iter().filter(|g| !g.task.done()) {
            if g.pages.0 < past && first < g.pages.1 {
                hit.push(g.key);
            } else {
                rest = true;
            }
        }
        if rest && crate::watch::shown_elsewhere(dst, len) {
            self.join();
            return;
        }
        for key in hit {
            self.wait_for(key);
        }
    }

    /// The refreshes given and not queued yet go to the wide threads (at the
    /// end of each list's steps, and every [`BATCH`]).
    pub(crate) fn flush(&mut self) {
        if self.queued >= self.given.len() {
            return;
        }
        let jobs: Vec<Work> = self.given[self.queued..]
            .iter()
            .filter(|g| g.task.state.load(Relaxed) == QUEUED)
            .map(|g| {
                let task = g.task.clone();
                Box::new(move || {
                    let _ = task.run(true);
                }) as Work
            })
            .collect();
        self.queued = self.given.len();
        crate::shadow::wide_later(jobs);
    }

    /// Before a step that reads or writes the state of staging buffer `key`
    /// (`Step::TextureHost`): its refresh given to the lanes is done, here if
    /// no wide thread began it.
    pub(crate) fn wait_for(&mut self, key: usize) {
        let Some(task) = self.keys.get(&key).map(|&at| self.given[at].task.clone()) else {
            return;
        };
        if task.done() {
            return;
        }
        let began = Instant::now();
        if let Some(us) = task.run(false) {
            self.counts.submit_us += us;
            return;
        }
        let mut left = task.group.left.lock().unwrap_or_else(|e| e.into_inner());
        while !task.done() {
            left.waiting = true;
            left = task.group.ended.wait(left).unwrap_or_else(|e| e.into_inner());
        }
        left.waiting = false;
        drop(left);
        self.counts.waits += 1;
        self.counts.waited_us += began.elapsed().as_micros() as u64;
    }

    /// Before a step that writes title memory, and at the end of the
    /// preparing: every refresh given is done. Those no wide thread began
    /// are run here (the last given first: the wide threads take from the
    /// front), the others waited for.
    pub(crate) fn join(&mut self) {
        if self.given.is_empty() {
            return;
        }
        let began = Instant::now();
        let mut ran = 0;
        for task in self.given.iter().rev().map(|g| &g.task) {
            if let Some(us) = task.run(false) {
                ran += us;
            }
        }
        if let Some(group) = &self.group {
            let mut left = group.left.lock().unwrap_or_else(|e| e.into_inner());
            while left.n > 0 {
                left.waiting = true;
                left = group.ended.wait(left).unwrap_or_else(|e| e.into_inner());
            }
            left.waiting = false;
        }
        self.counts.submit_us += ran;
        self.counts.joins += 1;
        self.counts.waited_us += (began.elapsed().as_micros() as u64).saturating_sub(ran);
        self.given.clear();
        self.keys.clear();
        self.queued = 0;
    }

    /// At the end of the preparing, before the settle for the card's results
    /// and `Host::card_before`: [`Lanes::join`], and the submission's counts
    /// go to its phase.
    pub(crate) fn finish(&mut self) {
        self.join();
        if !self.set {
            return;
        }
        let at = usize::from(!self.on);
        let c = std::mem::take(&mut self.counts);
        let (lane_us, lane_n) = self
            .group
            .as_ref()
            .map_or((0, 0), |g| (g.lane_us.swap(0, Relaxed), g.lane_n.swap(0, Relaxed)));
        for (what, n) in [
            (P::Refreshes, c.refreshes),
            (P::OnLanes, lane_n),
            (P::LaneUs, lane_us),
            (P::SubmitUs, c.submit_us),
            (P::Joins, c.joins),
            (P::Waits, c.waits),
            (P::WaitUs, c.waited_us),
        ] {
            add(at, what, n);
        }
    }
}

impl Drop for Lanes {
    /// A submission that ends early (a panic) waits for its refreshes all
    /// the same, before its lists let go of the staging buffers.
    fn drop(&mut self) {
        self.finish();
    }
}

// ---------------------------------------------------------------------------
// The drops.
// ---------------------------------------------------------------------------

/// The phase the last submission started in ([`Lanes::begin`]).
static LIVE: AtomicBool = AtomicBool::new(false);

/// What waits for the next submission's lists to have gone ([`sent`]).
static AFTER_SEND: Mutex<Vec<super::Retired>> = Mutex::new(Vec::new());

/// Whether an object let go of now waits for the reaper (the ON phases).
pub(crate) fn drops_later() -> bool {
    LIVE.load(Relaxed)
}

/// `object`, which nothing names any more: in the ON phases kept until the
/// next submission's lists have gone and then dropped on the reaper, else
/// dropped now.
pub(crate) fn drop_after_send<T: 'static>(object: T) {
    if drops_later() {
        hold(super::Retired(Box::new(object)));
    } else {
        drop(object);
    }
}

/// `host::retire`'s object, which no running submission names, kept for
/// the reaper ([`drop_after_send`]).
pub(super) fn hold(object: super::Retired) {
    AFTER_SEND.lock().unwrap_or_else(|e| e.into_inner()).push(object);
}

/// Right after a submission's lists went to the card: what waits goes to
/// the reaper thread.
pub(crate) fn sent() {
    if !SUBMIT_LANES.set() {
        return;
    }
    let drops = std::mem::take(&mut *AFTER_SEND.lock().unwrap_or_else(|e| e.into_inner()));
    if drops.is_empty() {
        return;
    }
    let at = row();
    add(at, P::Drops, drops.len() as u64);
    crate::reaper::dispose_away(drops, move |us| add(at, P::DropUs, us));
}

// ---------------------------------------------------------------------------
// The counts and the summary.
// ---------------------------------------------------------------------------

/// What the summary counts, per phase.
#[derive(Clone, Copy)]
enum P {
    Submissions,
    /// Texture refreshes, those the wide threads ran, their microseconds
    /// there, and the microseconds of those the submitting thread ran.
    Refreshes,
    OnLanes,
    LaneUs,
    SubmitUs,
    /// Joins with refreshes given, waits for one buffer's refresh, and the
    /// submitting thread's microseconds waiting in both.
    Joins,
    Waits,
    WaitUs,
    /// Uploads lists recorded (`crate::texture::upload`).
    Uploads,
    /// Objects handed to the reaper after the send, and its microseconds.
    Drops,
    DropUs,
    /// The frame line's "preparing".
    PrepareUs,
}

const P_COUNT: usize = P::PrepareUs as usize + 1;

/// Per phase (0: ON, or the only phase of `1`; 1: off).
static PHASES: [[AtomicU64; P_COUNT]; 2] = [const { [const { AtomicU64::new(0) }; P_COUNT] }; 2];

/// The phase counts go to now.
fn row() -> usize {
    usize::from(!LIVE.load(Relaxed))
}

fn add(at: usize, what: P, n: u64) {
    if n > 0 {
        PHASES[at][what as usize].fetch_add(n, Relaxed);
    }
}

/// A list recorded the upload of a texture (nothing without the switch).
pub(crate) fn uploaded() {
    if SUBMIT_LANES.set() {
        add(row(), P::Uploads, 1);
    }
}

/// A submission's preparing took `us` (the frame line's).
pub(crate) fn prepared(us: u64) {
    if SUBMIT_LANES.set() {
        add(row(), P::PrepareUs, us);
    }
}

/// The current period of the summary: its phase row, when it began
/// (`crate::host::now_ms`), the presents then.
struct Period {
    row: usize,
    began: u64,
    presented: u64,
}

static PERIOD: Mutex<Option<Period>> = Mutex::new(None);

/// A period of `1` lasts this long (`alt`: its phase).
const PERIOD_MS: u64 = 20_000;

/// At each submission's start, in phase `on`: a period that ended (a flip
/// of `alt`, or 20 s of `1`) is summed up and the next begins.
fn period_tick(on: bool) {
    let row = usize::from(!on);
    let now = super::now_ms();
    let fresh = || Period { row, began: now, presented: crate::queue::presented() };
    let mut period = PERIOD.lock().unwrap_or_else(|e| e.into_inner());
    match &*period {
        None => {
            for phase in &PHASES {
                for count in phase {
                    count.store(0, Relaxed);
                }
            }
            *period = Some(fresh());
        }
        Some(p) if p.row != row || (SUBMIT_LANES.mode() == 1 && now >= p.began + PERIOD_MS) => {
            let ended = period.replace(fresh());
            drop(period);
            if let Some(p) = ended {
                let counts: [u64; P_COUNT] = std::array::from_fn(|i| PHASES[p.row][i].swap(0, Relaxed));
                let frames = crate::queue::presented().saturating_sub(p.presented);
                let seconds = now.saturating_sub(p.began) as f64 / 1000.0;
                tracing::info!("{}", said(p.row, &counts, frames, seconds));
            }
        }
        Some(_) => {}
    }
}

/// The summary of a period of phase `row`: per frame.
fn said(row: usize, c: &[u64; P_COUNT], frames: u64, seconds: f64) -> String {
    let n = frames.max(1) as f64;
    let per = |p: P| c[p as usize] as f64 / n;
    let ms = |p: P| c[p as usize] as f64 / 1000.0 / n;
    let phase = match (SUBMIT_LANES.mode(), row) {
        (2, 0) => "the ON phase that ended",
        (2, _) => "the off phase that ended",
        _ => "on, the last period",
    };
    format!(
        "a submission's preparation on lanes (VICTORY_SUBMIT_LANES {}), {phase}: {frames} frames in {seconds:.1} s \
         ({:.2} fps); per frame: {:.1} submissions, preparing {:.2} ms (the frame line's); texture refreshes {:.1}, {:.1} \
         of them on the wide threads: {:.2} ms on the submitting thread, {:.2} ms on the wide threads; the submitting \
         thread waited {:.2} ms for the wide threads ({:.1} joins, {:.1} waits for one texture); {:.1} uploads recorded; \
         {:.1} of the host's objects dropped on the reaper after the lists went ({:.2} ms there)",
        SUBMIT_LANES.said(),
        frames as f64 / seconds.max(0.001),
        per(P::Submissions),
        ms(P::PrepareUs),
        per(P::Refreshes),
        per(P::OnLanes),
        ms(P::SubmitUs),
        ms(P::LaneUs),
        ms(P::WaitUs),
        per(P::Joins),
        per(P::Waits),
        per(P::Uploads),
        per(P::Drops),
        ms(P::DropUs),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Refreshes given to the lanes are all done after a join, whoever ran
    /// them; nothing is run twice.
    #[test]
    fn a_join_finds_every_refresh_given_done() {
        let ran: Arc<Vec<AtomicUsize>> = Arc::new((0..40).map(|_| AtomicUsize::new(0)).collect());
        let mut lanes = Lanes::new(true, false);
        for k in 0..40 {
            let ran = ran.clone();
            lanes.give(0x1000 + k * 64, (k << 20, 0x1000), Box::new(move || {
                // Some take a while, so the wide threads and this one share.
                if k.is_multiple_of(7) {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                ran[k].fetch_add(1, Relaxed);
            }));
        }
        lanes.flush();
        lanes.join();
        assert!(ran.iter().all(|r| r.load(Relaxed) == 1), "each refresh once, all before the join returned");
        assert!(lanes.given.is_empty() && lanes.keys.is_empty());
        lanes.join();
    }

    /// Texture `k`'s memory in the tests: two journal pages of its own.
    fn at(k: usize) -> usize {
        0x4000_0000 + k * 2 * crate::watch::PAGE
    }

    /// The order a submission's steps keep, played as `run_deferred` plays
    /// it, each texture's "title memory" a number of its own: a refresh given
    /// before a title copy into its memory reads what was there before it;
    /// one given before a `Run` step (which may write anywhere) reads what
    /// was there before that; one given after reads the write; a buffer's
    /// state step after its refresh sees it done, whoever ran it.
    #[test]
    fn steps_keep_their_order_around_the_lanes() {
        for round in 0..50u64 {
            let r = round * 10;
            let memory: Arc<Vec<AtomicU64>> = Arc::new((0..6).map(|_| AtomicU64::new(r)).collect());
            let staged: Arc<Vec<AtomicU64>> = Arc::new((0..6).map(|_| AtomicU64::new(u64::MAX)).collect());
            let mut lanes = Lanes::new(true, false);
            let refresh = |lanes: &mut Lanes, k: usize| {
                let (memory, staged, slow) = (memory.clone(), staged.clone(), (k + round as usize).is_multiple_of(3));
                lanes.give(k, (at(k), 0x8000), Box::new(move || {
                    if slow {
                        std::thread::yield_now();
                    }
                    staged[k].store(memory[k].load(Relaxed), Relaxed);
                }));
            };
            for k in 0..3 {
                refresh(&mut lanes, k);
            }
            // The state step of texture 1 (an upload's flags).
            lanes.wait_for(1);
            assert_eq!(staged[1].load(Relaxed), r, "round {round}");
            // A title copy into texture 2's memory, then a `Run` step.
            lanes.before_copy(at(2) + 0x100, 64);
            assert_eq!(staged[2].load(Relaxed), r, "round {round}: read before the copy");
            memory[2].store(r + 1, Relaxed);
            lanes.join();
            for m in memory.iter() {
                m.store(r + 2, Relaxed);
            }
            for k in 3..6 {
                refresh(&mut lanes, k);
            }
            lanes.flush();
            lanes.wait_for(5);
            assert_eq!(staged[5].load(Relaxed), r + 2);
            lanes.finish();
            let got: Vec<u64> = staged.iter().map(|s| s.load(Relaxed)).collect();
            assert_eq!(got, [r, r, r, r + 2, r + 2, r + 2], "round {round}");
        }
    }

    /// A title copy waits only for the refreshes whose memory shares a
    /// journal page with its destination; a refresh elsewhere goes on.
    #[test]
    fn a_copy_elsewhere_does_not_wait() {
        let go = Arc::new(AtomicBool::new(false));
        let mut lanes = Lanes::new(true, false);
        let slow = go.clone();
        lanes.give(1, (at(1), 0x1000), Box::new(move || {
            let began = Instant::now();
            while !slow.load(Relaxed) && began.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::yield_now();
            }
        }));
        let quick = Arc::new(AtomicBool::new(false));
        let done = quick.clone();
        lanes.give(2, (at(2), 0x1000), Box::new(move || done.store(true, Relaxed)));
        lanes.flush();
        // Into texture 2's page, which texture 1 does not touch.
        let began = Instant::now();
        lanes.before_copy(at(2) + crate::watch::PAGE - 8, 8);
        assert!(quick.load(Relaxed), "the refresh under the copy is done");
        assert!(began.elapsed() < std::time::Duration::from_secs(2), "the copy waited for a refresh elsewhere");
        assert!(!lanes.given[0].task.done(), "the refresh elsewhere still goes on");
        go.store(true, Relaxed);
        lanes.join();
        assert!(lanes.given.is_empty());
    }

    /// The same buffer given twice before a join runs after the first, not
    /// beside it; a wait for a buffer never given returns at once.
    #[test]
    fn a_buffer_given_twice_runs_in_turn() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut lanes = Lanes::new(true, false);
        for n in 0..2 {
            let log = log.clone();
            lanes.give(7, (at(7), 0x1000), Box::new(move || {
                log.lock().unwrap().push((n, 0));
                std::thread::sleep(std::time::Duration::from_millis(2));
                log.lock().unwrap().push((n, 1));
            }));
            lanes.flush();
        }
        lanes.wait_for(99);
        lanes.join();
        assert_eq!(*log.lock().unwrap(), [(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    /// Off (or unset), nothing is given: the lanes stay empty.
    #[test]
    fn off_nothing_is_given() {
        let lanes = Lanes::new(false, true);
        assert!(lanes.given.is_empty() && lanes.group.is_none());
        let unset = Lanes::default();
        assert!(!unset.on && !unset.set);
    }

    /// The summary says what it counts.
    #[test]
    fn the_summary_is_said() {
        let mut c = [0u64; P_COUNT];
        c[P::Refreshes as usize] = 300;
        c[P::OnLanes as usize] = 250;
        c[P::LaneUs as usize] = 20_000;
        c[P::WaitUs as usize] = 1_000;
        c[P::Drops as usize] = 40;
        let text = said(0, &c, 10, 2.0);
        assert!(text.starts_with("a submission's preparation on lanes"), "{text}");
        assert!(text.contains("texture refreshes 30.0, 25.0 of them on the wide threads"), "{text}");
        assert!(text.contains("2.00 ms on the wide threads") && text.contains("waited 0.10 ms"), "{text}");
        assert!(text.contains("4.0 of the host's objects dropped on the reaper"), "{text}");
    }
}
