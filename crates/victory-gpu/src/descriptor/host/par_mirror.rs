//! VIcky: the heap sync's mirroring on several threads
//! (`VICTORY_PARALLEL_MIRROR=1|alt|check`; off by default). A port of
//! noaclr 589848256d (merged there as 73a8bdf21c), written for VIcky's
//! slot state (the heap map, the cache of made views, the split keeps, the
//! lean chunks and the notes by kind), which that tree does not have.
//!
//! With ray tracing on the frame is the submitting thread's work, and its
//! longest wait is the heap sync's join (some 30 ms a frame): the sync
//! mirrors 93,000 to 97,000 slots a frame at some 0.66 us each, 57 to 64 ms
//! of its own, on one thread (`VICTORY_HEAP_SYNC_THREADS` spreads only the
//! reading of the pages), and holds the heap's slot state (`made`) for all
//! of it, a chunk at a time (fable-2026-11-27-pf4.md, road10.md).
//!
//! With the switch the slots a heap sync found written (every page of the
//! heap together, `super::super::sync_wide`) are made in rounds of
//! [`ROUND`]: the sync holds `made` alone for the round, as it holds it for
//! a chunk without the switch, and the round's slots, cut in up to
//! [`CHUNKS`] chunks in slot order, are made on threads kept for them (as
//! many as `crate::shadow`'s wide threads, [`Crew`]) and the sync's own,
//! each chunk as one chunk of the sync's
//! (`HostHeap::mirror_due`, or with `VICTORY_HEAP_SYNC_LEAN` its lean
//! chunk). Rounds of fewer than [`MIN_SYNC`] slots are made as before.
//!
//! What keeps the chunks' threads apart:
//! - `made` is sharded like noaclr's (slot `i` in piece `i %` [`PIECES`]):
//!   a thread makes a slot with that slot's piece locked ([`Pieces`]), and
//!   no thread holds two pieces. The slots of a sync are distinct, so no
//!   two threads make the same one; what a slot's making changes in `made`
//!   is its own key and heap map entry, and the pool and the cache, each
//!   behind a lock of its own (as for `VICTORY_HEAP_PASS_THREADS`).
//! - The slots a placed texture view makes stale are other slots: one in
//!   the piece of the slot being made is unmade at once, as before; one in
//!   another piece is unmade once the thread has let go of its own piece,
//!   with that slot's piece locked, and only if no thread made it again
//!   meanwhile: it still holds a placed texture view, and no note put it
//!   back in the spans (noaclr's check). No two pieces are ever held
//!   together, so no two threads wait for each other's.
//! - A thread lets go of no other slot's pool slot when the pool is used
//!   up (another thread may be making that slot): those slots are made once
//!   the round's threads are done, `made` still held (`HostHeap::reclaim`,
//!   as `HostHeap::make_room` does for the pass's threads).
//! - The keeps of the slots a thread makes are gathered for its chunk's
//!   end as before (`KeptLater`, the lean chunk), but a slot that comes to
//!   hold nothing (a stale slot unmade, a slot emptied) lets go of its
//!   resource at once: a slot's later making on another thread may be kept
//!   before this thread's chunk ends, and the resource a slot's descriptor
//!   views must never be let go of after it. At worst a slot made null
//!   holds its last resource a little longer.
//!
//! What may differ from one thread: the order in which the round's slots
//! are made (each slot is made by itself, so what is made of each is the
//! same), and with it which pool slot a title slot past the identity
//! window takes, which records the cache keeps, and, of two views in one
//! round that make each other stale, which one stays (the one noted last,
//! as on one thread, but "last" is the threads' order).
//!
//! The pass over buffer views (`crate::draw::heap_pass`) gets [`PASS_THREADS`]
//! threads for [`MIN_PASS`] slots or more when `VICTORY_HEAP_PASS_THREADS`
//! asks for none (noaclr makes `heap_buffers` in 8 chunks too); with the
//! switch set at all, the pass's threads make each slot with its piece
//! locked as above.
//!
//! `VICTORY_PARALLEL_MIRROR=check`: one round in [`CHECK_EVERY`] is made
//! from records read once before it, and then, `made` still held, each of
//! its slots is made again one by one in slot order from the same record
//! (`HostHeap::mirror_made`, one thread). After a round made the way one
//! thread makes it, nothing changes; the slots whose key does are counted
//! (the heap sync line) and the first ones said. Rounds in which a view
//! made another stale are left out (made again, the pair would change on
//! one thread too).
//!
//! `VICTORY_PARALLEL_MIRROR=alt`: twenty seconds without, twenty with, in
//! turn from the first heap sync; the phase changes at the start of a
//! heap sync, and each change is said in the log with what the syncs took
//! in the phase before.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use super::*;

/// What the switch asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    On,
    Alt,
    Check,
}

#[cfg(test)]
thread_local! {
    /// Tests: the mode for this thread, whatever the environment says.
    pub(crate) static TEST_MODE: std::cell::Cell<Option<Mode>> = const { std::cell::Cell::new(None) };
}

/// The mode, read once (`VICTORY_PARALLEL_MIRROR=1|alt|check`).
pub(crate) fn mode() -> Mode {
    #[cfg(test)]
    if let Some(mode) = TEST_MODE.get() {
        return mode;
    }
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_PARALLEL_MIRROR").as_deref() {
            Ok("1") => Mode::On,
            Ok("alt") => Mode::Alt,
            Ok("check") => Mode::Check,
            _ => return Mode::Off,
        };
        tracing::info!(
            "the heap sync makes the slots it finds written on several threads, in chunks under one hold of the heap's \
             slot state, slot i under lock i % {PIECES}; passes over buffer views of {MIN_PASS} slots or more on \
             {PASS_THREADS} threads (VICTORY_PARALLEL_MIRROR={}; port of noaclr 589848256d)",
            match mode {
                Mode::On => "1",
                Mode::Alt => "alt: twenty seconds without, twenty with, in turn",
                _ => "check: one round in 16 made again one by one, the keys that change counted",
            }
        );
        mode
    })
}

/// With `alt`, whether the phase now is the one with the switch
/// ([`tick`]).
static PHASE: AtomicBool = AtomicBool::new(false);

/// Whether the heap sync's slots (and the pass over buffer views) are made
/// on several threads now.
pub(crate) fn now() -> bool {
    match mode() {
        Mode::Off => false,
        Mode::On | Mode::Check => true,
        Mode::Alt => PHASE.load(Relaxed),
    }
}

/// At the start of a heap sync (`super::super::sync_heaps_told`): with
/// `alt`, the phase for now, each change said with what the syncs of the
/// phase before took.
pub(crate) fn tick() {
    if mode() != Mode::Alt {
        return;
    }
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
    static LAST: AtomicU8 = AtomicU8::new(2);
    if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
        PHASE.store(on, Relaxed);
        let [syncs, us, mirrored, wide] = ALT.each_ref().map(|c| c.swap(0, Relaxed));
        tracing::info!(
            "the heap sync makes its slots on several threads from now (VICTORY_PARALLEL_MIRROR=alt): {}; since the \
             change before: {syncs} heap syncs, {:.2} ms a sync, {mirrored} slots mirrored, {wide} of them on several \
             threads",
            if on { "ON" } else { "off, on one thread as usual" },
            us as f64 / 1000.0 / syncs.max(1) as f64,
        );
    }
}

/// With `alt`, since the last change: heap syncs, their microseconds, the
/// slots they mirrored, and those made on several threads.
static ALT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A heap sync took `us` microseconds and mirrored `mirrored` slots.
pub(crate) fn synced(us: u64, mirrored: u64) {
    if mode() == Mode::Alt {
        ALT[0].fetch_add(1, Relaxed);
        ALT[1].fetch_add(us, Relaxed);
        ALT[2].fetch_add(mirrored, Relaxed);
    }
}

/// What [`COUNTS`] holds, since the last sync line.
#[derive(Clone, Copy)]
enum Count {
    /// Slots made on several threads.
    Slots,
    /// Rounds: holds of `made` for them.
    Rounds,
    /// Chunks run.
    Chunks,
    /// Microseconds of the sync's mirroring on several threads, end to end.
    Wall,
    /// Microseconds of its chunks, summed over their threads.
    Threads,
    /// Stale slots of other pieces unmade once the piece was let go of.
    LaterUnmade,
    /// Of those, the ones left as they were: made again meanwhile.
    LaterLeft,
    /// Slots made once the threads were done, the pool used up.
    Room,
    /// Passes over buffer views given [`PASS_THREADS`] threads.
    Passes,
    /// `check`: rounds made again one by one.
    Checked,
    /// `check`: rounds left out (a view made another stale in them).
    CheckOut,
    /// `check`: slots made again one by one.
    CheckSlots,
    /// `check`: of them, those whose key that changed.
    CheckDiffered,
}

static COUNTS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

fn count(what: Count, n: u64) {
    COUNTS[what as usize].fetch_add(n, Relaxed);
}

/// The heap sync line's part, per frame over `frames` (empty without the
/// switch).
pub(crate) fn part(frames: f64) -> String {
    let n: [u64; 13] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed));
    if mode() == Mode::Off {
        return String::new();
    }
    let per = |c: Count| n[c as usize] as f64 / frames;
    let ms = |c: Count| n[c as usize] as f64 / 1000.0 / frames;
    let mut out = format!(
        "; VICTORY_PARALLEL_MIRROR: {:.0} slots per frame made on several threads in {:.1} holds of the slot state, \
         {:.1} chunks run in parallel, {:.2} ms of the sync's mirroring wall time against {:.2} ms summed over its \
         threads, {:.1} stale slots of other pieces unmade after ({:.1} left: made again meanwhile), {:.1} made once \
         the threads were done (pool used up); {:.2} passes over buffer views on {PASS_THREADS} threads",
        per(Count::Slots),
        per(Count::Rounds),
        per(Count::Chunks),
        ms(Count::Wall),
        ms(Count::Threads),
        per(Count::LaterUnmade),
        per(Count::LaterLeft),
        per(Count::Room),
        per(Count::Passes),
    );
    if mode() == Mode::Check {
        let at = |c: Count| n[c as usize];
        out.push_str(&format!(
            "; VICTORY_PARALLEL_MIRROR=check (since the last line): {} rounds made again one by one ({} left out: a \
             view made another stale), {} slots, {} whose key that changed",
            at(Count::Checked),
            at(Count::CheckOut),
            at(Count::CheckSlots),
            at(Count::CheckDiffered),
        ));
    }
    out
}

/// The pieces `made` is sharded in for the threads that make slots
/// together: slot `i` in piece `i % PIECES`.
pub(crate) const PIECES: u32 = 64;

/// A heap sync's slots below this many are made on one thread, as before
/// (noaclr: 128).
pub(crate) const MIN_SYNC: usize = 128;

/// The chunks of a round, at most (noaclr: 8); fewer with fewer wide
/// threads ([`crew_size`]).
pub(crate) const CHUNKS: usize = 8;

/// The slots of one round: one hold of `made`. (Without the switch the
/// sync holds it for a chunk of `MIRROR_CHUNK`; a round of this many on
/// seven threads holds it about as long as some 300.)
pub(crate) const ROUND: usize = 2048;

/// A pass over buffer views of this many slots or more gets
/// [`PASS_THREADS`] threads (noaclr: 256).
pub(crate) const MIN_PASS: usize = 256;

/// The threads of such a pass (noaclr: 8 chunks).
pub(crate) const PASS_THREADS: usize = 8;

/// `check`: one round in this many is made again one by one.
pub(crate) const CHECK_EVERY: u64 = 16;

/// The locks of `made`'s pieces.
pub(crate) struct Pieces(Vec<Mutex<()>>);

impl Pieces {
    pub(crate) fn new() -> Self {
        Pieces((0..PIECES).map(|_| Mutex::new(())).collect())
    }

    /// Slot `index`'s piece locked.
    fn lock(&self, index: u32) -> std::sync::MutexGuard<'_, ()> {
        self.0[(index % PIECES) as usize].lock().unwrap_or_else(|e| e.into_inner())
    }
}

thread_local! {
    /// The heap (its address) whose slots this thread makes together with
    /// other threads (a chunk of a round, a thread of the pass over buffer
    /// views); 0: none.
    static JOB: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// In such a job, the piece of the slot being made, while it is locked.
    static AT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
    /// Stale slots of other pieces, unmade once that piece is let go of.
    static LATER: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether this thread makes `heap`'s slots together with other threads
/// now: a slot that comes to hold nothing lets go of its resource at once
/// (`HostHeap::later`, the lean chunk's keep).
pub(crate) fn in_job(heap: &HostHeap) -> bool {
    JOB.get() == std::ptr::from_ref(heap) as usize
}

/// Whether the threads of the pass over buffer views make each slot with
/// its piece locked: with the switch set at all, in either phase of `alt`
/// (the phase may change while a pass runs).
pub(crate) fn pass_pieces() -> bool {
    mode() != Mode::Off
}

/// The threads for a pass over `slots` buffer views when
/// `VICTORY_HEAP_PASS_THREADS` asks for none: [`PASS_THREADS`] for
/// [`MIN_PASS`] slots or more with the switch on now, else none (0).
pub(crate) fn pass_threads(slots: usize) -> usize {
    if !now() || slots < MIN_PASS {
        return 0;
    }
    count(Count::Passes, 1);
    PASS_THREADS
}

/// What a thread that makes slots with others sets for its job, and gives
/// back after it ([`Job::enter`]).
struct Job {
    was: usize,
    #[cfg(feature = "diagnostics")]
    syncing: Option<bool>,
}

impl Job {
    /// This thread makes `heap`'s slots with other threads from now on;
    /// `syncing`: in a heap sync (the notes by kind count its slots).
    fn enter(heap: &HostHeap, syncing: bool) -> Self {
        let was = JOB.replace(std::ptr::from_ref(heap) as usize);
        #[cfg(not(feature = "diagnostics"))]
        let _ = syncing;
        Job {
            was,
            #[cfg(feature = "diagnostics")]
            syncing: syncing.then(|| {
                let was = note_kind::syncing_now();
                note_kind::syncing(true);
                was
            }),
        }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        JOB.set(self.was);
        #[cfg(feature = "diagnostics")]
        if let Some(was) = self.syncing {
            note_kind::syncing(was);
        }
    }
}

/// Tests: the switches set for the test's own thread, carried to the
/// crew's threads that make its chunks (and given back after).
#[cfg(test)]
#[derive(Clone, Copy)]
struct TestEnv {
    fast: Option<(usize, bool, bool)>,
    kind: Option<note_kind::Mode>,
    mode: Option<Mode>,
}

#[cfg(test)]
impl TestEnv {
    fn here() -> Self {
        TestEnv {
            fast: TEST_FAST.get(),
            kind: note_kind::TEST_MODE.get(),
            mode: TEST_MODE.get(),
        }
    }

    /// Set on this thread; returns what was set before.
    fn set(self) -> Self {
        TestEnv {
            fast: TEST_FAST.replace(self.fast),
            kind: note_kind::TEST_MODE.replace(self.kind),
            mode: TEST_MODE.replace(self.mode),
        }
    }
}

type CrewJob = Box<dyn FnOnce() + Send>;

/// The threads kept for the rounds' chunks beside the sync's own: as many
/// as the wide threads of `crate::shadow` (one fewer than [`CHUNKS`] at
/// most), but their own. A caller of `crate::shadow::wide_run` runs any
/// queued job while it waits, the submitting thread among them with its
/// locks held, and a chunk takes locks of the heap and of the host
/// (`crate::host::retire`, the views it makes).
struct Crew {
    jobs: Mutex<std::collections::VecDeque<CrewJob>>,
    more: std::sync::Condvar,
}

/// The crew's threads.
fn crew_size() -> usize {
    crate::shadow::wide_threads().saturating_sub(1).clamp(1, CHUNKS - 1)
}

fn crew() -> &'static Crew {
    static CREW: OnceLock<Crew> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let crew = CREW.get_or_init(|| Crew {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    STARTED.call_once(|| {
        for i in 0..crew_size() {
            let _ = std::thread::Builder::new().name(format!("victory-mirror-{i}")).spawn(|| {
                let crew = CREW.get().expect("made above");
                loop {
                    let job = {
                        let mut jobs = crew.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                break job;
                            }
                            jobs = crew.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    job();
                }
            });
        }
    });
    crew
}

/// Runs `jobs` on the crew and on this thread, and returns when all of
/// them are done; a job's panic is taken up again here once all are.
fn crew_run(jobs: Vec<CrewJob>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    type Left = (Mutex<(usize, Option<Box<dyn std::any::Any + Send>>)>, std::sync::Condvar);
    let crew = crew();
    let left: Arc<Left> = Arc::new((Mutex::new((jobs.len(), None)), std::sync::Condvar::new()));
    {
        let mut queue = crew.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let left = left.clone();
            queue.push_back(Box::new(move || {
                let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).err();
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                n.0 -= 1;
                if n.1.is_none() {
                    n.1 = panicked;
                }
                if n.0 == 0 {
                    left.1.notify_all();
                }
            }));
        }
    }
    crew.more.notify_all();
    loop {
        let job = crew.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some(job) => job(),
            None => break,
        }
    }
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    while n.0 > 0 {
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
    if let Some(panic) = n.1.take() {
        drop(n);
        std::panic::resume_unwind(panic);
    }
}

/// Slots in rising order (each once), as a heap sync finds them.
fn rising(due: &[(u32, u64)]) -> bool {
    due.windows(2).all(|w| w[0].0 < w[1].0)
}

/// Title slot `index`'s key (0: none, or no host slot).
fn key_of(made: &Made, index: u32) -> u64 {
    made.host(index).and_then(|host| made.key_of(host)).unwrap_or(0)
}

/// What a chunk's thread hands back: the title slots it found no pool slot
/// for, and its microseconds.
type Done = Mutex<Option<(Vec<u32>, u64)>>;

impl HostHeap {
    /// The heap sync's making of `due` (title slot, handle, in rising slot
    /// order; with `lean` (`VICTORY_HEAP_SYNC_LEAN`) each from its record in
    /// `records`): with the switch on now and [`MIN_SYNC`] slots or more,
    /// in rounds on several threads; else as before
    /// ([`HostHeap::mirror_due`], [`HostHeap::mirror_due_lean`]).
    pub(crate) fn mirror_due_par(&self, due: &[(u32, u64)], records: &[[u32; 3]], lean: Option<bool>) {
        if !now() || due.len() < MIN_SYNC || !rising(due) || lean.is_some_and(|_| records.len() != due.len()) {
            match lean {
                Some(check) => self.mirror_due_lean(due, records, check),
                None => self.mirror_due(due),
            }
            return;
        }
        let started = std::time::Instant::now();
        let chunks = crew_size() + 1;
        for (n, round) in due.chunks(ROUND).enumerate() {
            let words = lean.map(|_| &records[n * ROUND..n * ROUND + round.len()]);
            self.mirror_round(round, words, lean, chunks);
        }
        if mode() == Mode::Alt {
            ALT[3].fetch_add(due.len() as u64, Relaxed);
        }
        count(Count::Slots, due.len() as u64);
        count(Count::Wall, started.elapsed().as_micros() as u64);
    }

    /// One round of [`HostHeap::mirror_due_par`]: `slots` (with `lean`, from
    /// `words`) in up to `chunks` chunks on the crew and this thread,
    /// `made` held alone by this thread for all of it.
    fn mirror_round(&self, slots: &[(u32, u64)], words: Option<&[[u32; 3]]>, lean: Option<bool>, chunks: usize) {
        static ROUNDS: AtomicU64 = AtomicU64::new(0);
        let checking = mode() == Mode::Check && ROUNDS.fetch_add(1, Relaxed) % CHECK_EVERY == 0;
        // `check`: the records read once, for both makings.
        let read: Vec<[u32; 3]>;
        let words = match words {
            Some(words) => Some(words),
            None if checking => {
                // SAFETY: the handles are descriptor slots in title memory
                // (as `HostHeap::record_at` reads them).
                read = slots.iter().map(|&(_, handle)| unsafe { (handle as *const [u32; 3]).read_unaligned() }).collect();
                Some(&read[..])
            }
            None => None,
        };
        let made = held_alone(&self.made);
        let stale_before = STALE[0].load(Relaxed);
        let per = slots.len().div_ceil(chunks.max(1));
        let parts = slots.len().div_ceil(per);
        let done: Arc<Vec<Done>> = Arc::new((0..parts).map(|_| Mutex::new(None)).collect());
        let heap_at = std::ptr::from_ref(self) as usize;
        let made_at = std::ptr::from_ref::<Made>(&*made) as usize;
        // With `check` the lean chunk's own check (records read again in
        // place) is not made: both makings take the round's records.
        let lean_check = if checking { lean.map(|_| false) } else { lean };
        #[cfg(test)]
        let env = TestEnv::here();
        let jobs: Vec<Box<dyn FnOnce() + Send>> = (0..parts)
            .map(|c| {
                let done = done.clone();
                let from = c * per;
                let n = per.min(slots.len() - from);
                let slots_at = slots[from..].as_ptr() as usize;
                let words_at = words.map_or(0, |w| w[from..].as_ptr() as usize);
                Box::new(move || {
                    #[cfg(test)]
                    let was = env.set();
                    // SAFETY: the heap, `made` (held by the round's thread)
                    // and the slots and records outlive the round: its thread
                    // waits for every job ([`crew_run`]) before it lets go of
                    // `made` and returns.
                    let (heap, made) = unsafe { (&*(heap_at as *const HostHeap), &*(made_at as *const Made)) };
                    let slots = unsafe { std::slice::from_raw_parts(slots_at as *const (u32, u64), n) };
                    let words = (words_at != 0).then(|| unsafe { std::slice::from_raw_parts(words_at as *const [u32; 3], n) });
                    let began = std::time::Instant::now();
                    let room = heap.mirror_chunk_par(made, slots, words, lean_check);
                    *done[c].lock().unwrap_or_else(|e| e.into_inner()) = Some((room, began.elapsed().as_micros() as u64));
                    #[cfg(test)]
                    was.set();
                }) as Box<dyn FnOnce() + Send>
            })
            .collect();
        crew_run(jobs);
        count(Count::Rounds, 1);
        count(Count::Chunks, parts as u64);
        let mut room = Vec::new();
        for part in done.iter() {
            if let Some((wanted, us)) = part.lock().unwrap_or_else(|e| e.into_inner()).take() {
                room.extend(wanted);
                count(Count::Threads, us);
            }
        }
        // The slots the threads found no pool slot for: made now, as on one
        // thread once the pool slots of slots without a live view are
        // given back (`HostHeap::reclaim`, at most once every two seconds).
        if !room.is_empty() && self.reclaim(&made) > 0 {
            let handles: std::collections::HashMap<u32, (u64, usize)> =
                slots.iter().enumerate().map(|(i, &(index, handle))| (index, (handle, i))).collect();
            RETIRE_LATER.set(Some(Vec::new()));
            for index in room {
                let Some(&(handle, i)) = handles.get(&index) else {
                    continue;
                };
                let Some(read) = self.record_at(index, handle, false) else {
                    continue;
                };
                let record = words.map_or(read, |w| w[i]);
                self.mirror_made(&made, true, (index, handle, false, true), record, &mut Lookups::each());
                count(Count::Room, 1);
            }
            if let Some(held) = RETIRE_LATER.take().filter(|held| !held.is_empty()) {
                crate::host::retire(held);
            }
        }
        if checking && let Some(words) = words {
            if STALE[0].load(Relaxed) != stale_before {
                count(Count::CheckOut, 1);
            } else {
                self.check_round(&made, slots, words);
            }
        }
        drop(made);
    }

    /// `check`: each of the round's `slots` made again one by one in slot
    /// order from its record in `words`, `made` held alone, and the slots
    /// whose key that changes counted.
    fn check_round(&self, made: &Made, slots: &[(u32, u64)], words: &[[u32; 3]]) {
        let before: Vec<u64> = slots.iter().map(|&(index, _)| key_of(made, index)).collect();
        RETIRE_LATER.set(Some(Vec::new()));
        let later = KeptLater::begin(self);
        for (&(index, handle), &record) in slots.iter().zip(words) {
            if self.record_at(index, handle, false).is_some() {
                self.mirror_made(made, true, (index, handle, false, true), record, &mut Lookups::each());
            }
        }
        drop(later);
        if let Some(held) = RETIRE_LATER.take().filter(|held| !held.is_empty()) {
            crate::host::retire(held);
        }
        let mut differed = 0;
        for (&(index, _), &was) in slots.iter().zip(&before) {
            let now = key_of(made, index);
            if now != was {
                differed += 1;
                static SAID: AtomicU64 = AtomicU64::new(0);
                if SAID.fetch_add(1, Relaxed) < 32 {
                    tracing::warn!(
                        "VICTORY_PARALLEL_MIRROR=check: title slot {index} made again one by one changed its key from \
                         {was:#x} to {now:#x}"
                    );
                }
            }
        }
        count(Count::Checked, 1);
        count(Count::CheckSlots, slots.len() as u64);
        count(Count::CheckDiffered, differed);
    }

    /// One chunk of a round on this thread (`made` held by the round's
    /// thread): each slot made as a chunk of the sync's makes it, with its
    /// piece locked. Returns the title slots it found no pool slot for.
    fn mirror_chunk_par(&self, made: &Made, slots: &[(u32, u64)], words: Option<&[[u32; 3]]>, lean: Option<bool>) -> Vec<u32> {
        let _job = Job::enter(self, true);
        // (What this thread gathered before, if it was in a chunk of its
        // own, is given back after.)
        let outer = RETIRE_LATER.replace(Some(Vec::new()));
        match lean {
            Some(check) => {
                let words = words.unwrap_or_default();
                let mirrored = self.mirrored();
                let mut chunk = sync_lean::Chunk::new(self, false);
                self.note_title_start(slots);
                for (i, &(index, handle)) in slots.iter().enumerate() {
                    if index >= mirrored {
                        // (Counted and said there, as before.)
                        let _ = self.record_at(index, handle, false);
                        continue;
                    }
                    let record = if check { sync_lean::compared(handle, words[i]) } else { words[i] };
                    self.in_piece(made, index, || self.mirror_lean(made, (index, handle), record, &mut chunk));
                }
                sync_lean::counted(slots.len());
                chunk.end(self, made);
            }
            None => {
                let later = KeptLater::begin(self);
                for (i, &(index, handle)) in slots.iter().enumerate() {
                    let Some(read) = self.record_at(index, handle, false) else {
                        continue;
                    };
                    let record = words.map_or(read, |w| w[i]);
                    self.in_piece(made, index, || {
                        self.mirror_made(made, false, (index, handle, false, true), record, &mut Lookups::each())
                    });
                }
                drop(later);
            }
        }
        if let Some(held) = RETIRE_LATER.replace(outer).filter(|held| !held.is_empty()) {
            crate::host::retire(held);
        }
        take_room_wanted()
    }

    /// `f`, the making of title slot `index` on a thread that makes slots
    /// with others, with the slot's piece locked; then the stale slots of
    /// other pieces it left ([`HostHeap::par_stale`]) unmade, each with its
    /// own piece locked.
    pub(super) fn in_piece<R>(&self, made: &Made, index: u32, f: impl FnOnce() -> R) -> R {
        let out = {
            let _piece = self.pieces.lock(index);
            AT.set(Some(index % PIECES));
            let out = f();
            AT.set(None);
            out
        };
        let later = LATER.with_borrow_mut(std::mem::take);
        for stale in later {
            let _piece = self.pieces.lock(stale);
            // Made again since (noaclr's check): it holds a view that is
            // no placed texture's (or none), or a note put it back in the
            // spans of placed textures.
            let key = key_of(made, stale);
            let placed = key != 0 && matches!(view_at(key as u32, (key >> 32) as u32), Some(View::Placed(false, ..)));
            if !placed || self.placed.lock().unwrap_or_else(|e| e.into_inner()).by_slot.contains_key(&stale) {
                count(Count::LaterLeft, 1);
                continue;
            }
            // (No piece is noted as held: unmade as usual, its resource
            // let go of at once, `in_job`.)
            self.unmake(made, stale);
            count(Count::LaterUnmade, 1);
        }
        out
    }

    /// For `HostHeap::unmake` (and the lean chunk's): title slot `index`,
    /// made stale by the slot this thread makes with others, if it lies in
    /// another piece than that slot, is left for when that piece is let go
    /// of ([`HostHeap::in_piece`]); true if so (the caller unmakes nothing).
    pub(super) fn par_stale(&self, index: u32) -> bool {
        let Some(piece) = AT.get() else {
            return false;
        };
        if !in_job(self) || index % PIECES == piece {
            return false;
        }
        LATER.with_borrow_mut(|later| later.push(index));
        true
    }

    /// The pass over buffer views's thread (`HostHeap::mirror_shared`):
    /// `f` with this thread noted as making `self`'s slots with others, when
    /// the switch is set ([`pass_pieces`]).
    pub(super) fn pass_job<R>(&self, f: impl FnOnce(bool) -> R) -> R {
        if !pass_pieces() {
            return f(false);
        }
        let _job = Job::enter(self, false);
        f(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A placed buffer view of `at`, `stride` bytes an element.
    fn buffer(at: u64, stride: u32) -> View {
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, 0x1_0000);
        let mut desc = [0u32; 16];
        (desc[1], desc[2], desc[6], desc[7]) = (1, 5768, 4, stride);
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        View::Placed(false, resource, desc)
    }

    /// A placed texture view of `at`, `size` texels a side.
    fn texture(at: u64, size: u32) -> View {
        let resource = [3, 0, 0, 0, size, 0, size, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
        let mut desc = [0u32; 16];
        (desc[0], desc[1], desc[2], desc[5]) = (98, 4, 5768, 1);
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        View::Placed(false, resource, desc)
    }

    /// `view`'s record.
    fn record_of(view: &View) -> [u32; 3] {
        let [i, generation] = super::super::super::words_for(view.clone());
        [VIEW_TAG, i, generation]
    }

    /// The state a heap's mirroring leaves, by title slot (the pool slot
    /// a title slot past the window takes may differ between threads and
    /// one thread): its key and whether its host slot holds a resource, its
    /// kind and last note; the marked and the buffer slots, the texture
    /// queue and the spans.
    #[allow(clippy::type_complexity)]
    fn state(heap: &HostHeap, titles: u32) -> (Vec<(u64, bool)>, Vec<(u8, bool)>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let made = heap.made.read().unwrap();
        let slots = (0..titles)
            .map(|i| {
                let host = made.host(i);
                let held = host.is_some_and(|h| heap.kept.with(h, |slot| slot.is_some_and(Option::is_some)));
                (key_of(&made, i), held)
            })
            .collect();
        drop(made);
        let kinds = (0..titles).map(|i| (heap.kind(i), heap.noted_plain(i))).collect();
        let mut changed = heap.buffer_changes();
        changed.sort_unstable();
        let mut queued: Vec<u32> = heap.textures.lock().unwrap().slots.keys().copied().collect();
        let mut spanned: Vec<u32> = heap.placed.lock().unwrap().by_slot.keys().copied().collect();
        queued.sort_unstable();
        spanned.sort_unstable();
        (slots, kinds, changed, heap.buffer_slots(), queued, spanned)
    }

    /// Sets the switches for this thread (and so the chunks' threads), and
    /// gives them back when it goes.
    struct Switches;

    impl Switches {
        fn set(map: Option<(u32, u32)>, split: bool) -> Self {
            TEST_FAST.set(Some((0, true, false)));
            super::super::super::kept::TEST_SPLIT.set(Some(split));
            TEST_MAP.set(map);
            TEST_MODE.set(Some(Mode::On));
            Switches
        }
    }

    impl Drop for Switches {
        fn drop(&mut self) {
            TEST_FAST.set(None);
            super::super::super::kept::TEST_SPLIT.set(None);
            TEST_MAP.set(None);
            TEST_MODE.set(None);
        }
    }

    /// The same records, of which no view makes another stale, made by
    /// the heap sync's chunks on one thread and in rounds on several, the
    /// usual way and the lean way, with and without a heap map (its pool
    /// big enough) and the split keeps: after every round the heaps are in
    /// the same state, title slot by title slot.
    #[test]
    fn rounds_on_several_threads_make_what_one_thread_makes() {
        if crate::host::host().is_none() {
            return;
        }
        const TITLES: u32 = 1024;
        let base = 0x4d_0000_0000u64;
        let views = [
            buffer(base, 16),
            View::NullConstant,
            buffer(base + 0x1_0000, 32),
            View::Constant(base, 256),
            texture(base + 0x10_0000, 64),
            View::NullConstant,
            buffer(base + 0x2_0000, 16),
            View::Constant(base + 0x100, 256),
            texture(base + 0x20_0000, 64),
        ];
        let records: Vec<[u32; 3]> = views.iter().map(record_of).collect();
        for (map, split) in [(None, false), (None, true), (Some((256, TITLES + 8)), false), (Some((256, TITLES + 8)), true)] {
            for lean in [None, Some(false)] {
                let switches = Switches::set(map, split);
                let one = HostHeap::new(false, TITLES).unwrap();
                let many = HostHeap::new(false, TITLES).unwrap();
                TEST_MAP.set(None);
                let mut memory = vec![[0u32; 8]; TITLES as usize];
                let start = memory.as_mut_ptr() as u64;
                for round in 0..6u32 {
                    for s in 0..TITLES {
                        let pick = ((s * 7 + round * 13) % (views.len() as u32 + 3)) as usize;
                        let record = records.get(pick).copied().unwrap_or([0; 3]);
                        memory[s as usize][..3].copy_from_slice(&record);
                    }
                    let due: Vec<(u32, u64)> = (0..TITLES)
                        .filter(|s| (s + round) % 3 != 0)
                        .map(|s| (s, start + u64::from(s) * TITLE_SLOT_BYTES))
                        .collect();
                    let copied: Vec<[u32; 3]> = due
                        .iter()
                        .map(|&(s, _)| {
                            let w = memory[s as usize];
                            [w[0], w[1], w[2]]
                        })
                        .collect();
                    TEST_MODE.set(Some(Mode::Off));
                    one.mirror_due_par(&due, &copied, lean);
                    TEST_MODE.set(Some(Mode::On));
                    let chunks = COUNTS[Count::Chunks as usize].load(Relaxed);
                    many.mirror_due_par(&due, &copied, lean);
                    assert!(COUNTS[Count::Chunks as usize].load(Relaxed) > chunks + 1, "made in chunks");
                    assert_eq!(
                        state(&one, TITLES),
                        state(&many, TITLES),
                        "round {round}, map {map:?}, split {split}, lean {lean:?}"
                    );
                }
                drop(memory);
                drop(switches);
            }
        }
    }

    /// After noaclr's `sharded_made_concurrent_mirror_and_stale_unmake_is_deadlock_free`:
    /// pairs of texture views of one memory, slot p and slot p + 509 (in
    /// other pieces and other chunks), made on several threads round after
    /// round. A placed texture view is left for its draws by the heap sync
    /// (its key 0), so each round has two parts:
    /// 1. both views of every pair written and made together: each makes
    ///    the other stale (a stale slot of another piece is unmade after,
    ///    no two pieces held). The rounds end (no thread waits for
    ///    another's piece) and one view of each pair stays in the spans, as
    ///    on one thread; a null view in every fourth pair is made;
    /// 2. the second slots get newer views, made (as a draw would), and the
    ///    first slots then views of the same memory, made together: each
    ///    makes its pair's made slot stale, which ends unmade and out of
    ///    the spans.
    #[test]
    fn stale_slots_of_other_pieces_are_unmade_without_deadlock() {
        if crate::host::host().is_none() {
            return;
        }
        const TITLES: u32 = 1024;
        const PAIR: u32 = 509;
        let (tx, rx) = std::sync::mpsc::channel();
        let test = std::thread::spawn(move || {
            let switches = Switches::set(None, true);
            let heap = HostHeap::new(false, TITLES).unwrap();
            let base = 0x4e_0000_0000u64;
            let mut memory = vec![[0u32; 8]; TITLES as usize];
            let start = memory.as_mut_ptr() as u64;
            let handle = |s: u32| start + u64::from(s) * TITLE_SLOT_BYTES;
            let key = |record: [u32; 3]| u64::from(record[2]) << 32 | u64::from(record[1]);
            // Slot s's view: the first of a pair a small texture, the second
            // a large one of the same memory; every fourth pair a null view
            // and nothing; the last slots nothing.
            let view = |s: u32, at: u64| -> Option<View> {
                let (p, second) = if s < PAIR { (s, false) } else { (s - PAIR, true) };
                if p >= PAIR {
                    return None;
                }
                if p % 4 == 3 {
                    return (!second).then_some(View::NullConstant);
                }
                Some(texture(at + u64::from(p) * 0x10_0000, if second { 128 } else { 64 }))
            };
            for round in 0..6u32 {
                let at = base + 0x100_0000 + u64::from(round % 2) * 0x8_0000;
                // (Each view's record made once: a record is a view noted anew.)
                let mut records: Vec<[u32; 3]> =
                    (0..TITLES).map(|s| view(s, at).as_ref().map_or([0; 3], record_of)).collect();
                for (s, record) in records.iter().enumerate() {
                    memory[s][..3].copy_from_slice(record);
                }
                let due: Vec<(u32, u64)> = (0..TITLES).map(|s| (s, handle(s))).collect();
                heap.mirror_due_par(&due, &[], None);
                {
                    let made = heap.made.read().unwrap();
                    let spans = heap.placed.lock().unwrap();
                    for p in 0..PAIR {
                        let (a, b) = (p, p + PAIR);
                        if p % 4 == 3 {
                            assert_eq!([key_of(&made, a), key_of(&made, b)], [key(records[a as usize]), 0], "round {round}, pair {p}");
                            continue;
                        }
                        let spanned = [a, b].map(|s| spans.by_slot.contains_key(&s));
                        assert!(spanned[0] != spanned[1], "round {round}, pair {p}: one view of the memory in the spans, {spanned:?}");
                        assert_eq!([key_of(&made, a), key_of(&made, b)], [0, 0], "round {round}, pair {p}: left to the draws");
                    }
                }
                // 2: the second slots' newer views (other memory), made.
                let at = at + 0x4_0000;
                for s in PAIR..TITLES {
                    records[s as usize] = view(s, at).as_ref().map_or([0; 3], record_of);
                    memory[s as usize][..3].copy_from_slice(&records[s as usize]);
                }
                let seconds: Vec<(u32, u64)> = (PAIR..TITLES).map(|s| (s, handle(s))).collect();
                heap.mirror_due_par(&seconds, &[], None);
                {
                    let made = heap.made.write().unwrap();
                    for s in PAIR..2 * PAIR {
                        if records[s as usize] != [0; 3] {
                            made.set_key(made.host(s).unwrap(), key(records[s as usize]));
                        }
                    }
                }
                for s in 0..PAIR {
                    records[s as usize] = view(s, at).as_ref().map_or([0; 3], record_of);
                    memory[s as usize][..3].copy_from_slice(&records[s as usize]);
                }
                let firsts: Vec<(u32, u64)> = (0..PAIR).map(|s| (s, handle(s))).collect();
                let unmade = COUNTS[Count::LaterUnmade as usize].load(Relaxed);
                heap.mirror_due_par(&firsts, &[], None);
                let pairs = (0..PAIR).filter(|p| p % 4 != 3).count() as u64;
                assert!(
                    COUNTS[Count::LaterUnmade as usize].load(Relaxed) >= unmade + pairs,
                    "round {round}: the stale slots of other pieces unmade after"
                );
                let made = heap.made.read().unwrap();
                let spans = heap.placed.lock().unwrap();
                for p in (0..PAIR).filter(|p| p % 4 != 3) {
                    let (a, b) = (p, p + PAIR);
                    assert!(spans.by_slot.contains_key(&a), "round {round}, pair {p}: the newer view in the spans");
                    assert!(!spans.by_slot.contains_key(&b), "round {round}, pair {p}: the stale view out of the spans");
                    assert_eq!(key_of(&made, b), 0, "round {round}, pair {p}: the stale view unmade");
                }
                for s in 2 * PAIR..TITLES {
                    assert_eq!(key_of(&made, s), 0, "round {round}, slot {s}");
                }
            }
            drop(memory);
            drop(switches);
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_secs(120)).is_ok(),
            "the rounds did not end: threads waited for each other's pieces"
        );
        test.join().unwrap();
    }

    /// noaclr's `parallel_mirror_83k_slots_speedup`: 83,000 slots of
    /// constant views (as many as the title's world entry has its heap
    /// sync and pass make), made by the heap sync's chunks on one thread and
    /// in rounds on several. Run with
    /// `cargo test --release -p victory-gpu --lib par_mirror::tests::rounds_83k -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn rounds_83k() {
        if crate::host::host().is_none() {
            return;
        }
        const SLOTS: u32 = 83_000;
        let _switches = Switches::set(None, true);
        let one = HostHeap::new(false, SLOTS).unwrap();
        let many = HostHeap::new(false, SLOTS).unwrap();
        let base = 0x4f_0000_0000u64;
        let mut memory = vec![[0u32; 8]; SLOTS as usize];
        let start = memory.as_mut_ptr() as u64;
        for (s, slot) in memory.iter_mut().enumerate() {
            slot[..3].copy_from_slice(&record_of(&View::Constant(base + (s as u64 % 256) * 256, 256)));
        }
        let due: Vec<(u32, u64)> = (0..SLOTS).map(|s| (s, start + u64::from(s) * TITLE_SLOT_BYTES)).collect();
        // The crew started first.
        crew_run((0..8).map(|_| Box::new(|| {}) as CrewJob).collect());
        TEST_MODE.set(Some(Mode::Off));
        let t = std::time::Instant::now();
        one.mirror_due_par(&due, &[], None);
        let serial = t.elapsed();
        TEST_MODE.set(Some(Mode::On));
        let t = std::time::Instant::now();
        many.mirror_due_par(&due, &[], None);
        let parallel = t.elapsed();
        eprintln!(
            "par_mirror ({SLOTS} slots): one thread {serial:?}, {} threads {parallel:?} ({:.2}x)",
            crew_size() + 1,
            serial.as_secs_f64() / parallel.as_secs_f64()
        );
        assert!(many.made(SLOTS - 1));
        drop(memory);
    }
}
