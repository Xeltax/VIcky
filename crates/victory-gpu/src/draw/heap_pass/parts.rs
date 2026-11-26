//! VIcky: where the time of the pass over the heap's buffer views goes, and
//! what its slots came to. Always on; a line with the frame line ([`line`]).
//!
//! The thread that runs a pass (or each of its workers, with
//! `VICTORY_HEAP_PASS_THREADS`) counts its slots by what each came to
//! ([`Count`]), and times, in every chunk of slots, its wait for the lock of
//! the heap's slot state (`HostHeap::made`) and how long it then held it
//! ([`waited`], [`held`]). Its own time is split into parts that exclude
//! each other ([`Part`]) with the processor's time stamp counter, in one
//! chunk of slots in [`SAMPLE`]: a stamp costs about as much as some of the
//! parts it times, and every chunk timed would have made the pass several
//! percent longer. A part's share of the time so taken, times the pass's
//! time out of its waits for the lock, is its estimate.
//!
//! The slot state's code is shared with the heap sync and the draws. What it
//! asks here ([`enter`], [`count`]) first reads a flag that is set while a
//! pass runs, and only then, on that thread, the thread's own state.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// The parts of a pass's time. Entering one ends the part the thread was in
/// and starts it; when the guard goes, the part it interrupted goes on
/// ([`In`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    /// What no other part names: the pass's walk of its slots, the chunks
    /// gathered, what each slot came to noted.
    Walk,
    /// The records read in title memory, three words a slot.
    Read,
    /// The view a record names, from the table of views (read-locked).
    View,
    /// The slot's decision: the record its host slot was made from
    /// compared, and the cache of made views asked.
    Decide,
    /// The heap map: the host slot of a title slot past the identity window
    /// found, a pool slot taken or given back, its entry written.
    Map,
    /// The slot noted for the sets shaders pick by index: its stride, its
    /// mark, the sets of buffer and texture slots.
    Note,
    /// Host copies of title memory looked up (`crate::shadow::find` and
    /// `crate::shadow::exact`, behind the locks of their tables), where no
    /// lookup kept for the slot answered.
    Lookup,
    /// The device's calls that make a slot (`CreateShaderResourceView`,
    /// `CreateUnorderedAccessView`, a cached view copied with
    /// `CopyDescriptorsSimple`, null views), and for views of other kinds
    /// than buffers what their making asks before the call.
    Create,
    /// The host resource each slot holds swapped, the old one retired.
    Keep,
    /// Waiting for the lock of the slot state.
    Wait,
    /// The copy and range a slot's view reads, worked out from its lookups.
    Copy,
    /// The ranges kept from pass to pass: each slot's dropped, the new one
    /// kept.
    Ranges,
    /// Diagnostics builds: whether the slot shows the copy its range is on
    /// (`HostHeap::shows`, which takes the slot state's lock and that of
    /// the held resources, once a slot).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    Shows,
}

const PARTS: usize = 13;

/// How the line names each part, in the order of [`Part`].
const NAMES: [&str; PARTS] = [
    "the walk",
    "records read",
    "views named",
    "decisions",
    "heap map",
    "notes",
    "host copies looked up",
    "device calls making slots",
    "held resources swapped",
    "waits for the lock",
    "copies and ranges worked out",
    "kept ranges",
    "the diagnostics' look at the shown buffer",
];

/// What a pass counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Count {
    /// Slots past the title heap or past the host heap's slots: nothing read.
    Past,
    /// Records that name no view of ours (none written, or a released
    /// resource's): the slot made null.
    NotOurs,
    /// Past the identity window with the heap map's pool used up: no host
    /// slot.
    NoHost,
    /// Made from this record already: nothing made.
    Same,
    /// Buffer views copied from the cache of made views.
    Hit,
    /// Buffer views made and kept in the cache.
    Miss,
    /// Views made otherwise (a kind the cache does not keep, or no cache).
    Other,
    /// Views the host could not make: the slot null.
    Unmade,
    /// Views with a copy and range to keep, and without one.
    Range,
    NoRange,
    /// The heap map: host slots of title slots past the identity window
    /// looked up, and pool slots taken.
    Mapped,
    Taken,
    /// Host copies looked up ([`Part::Lookup`]).
    Lookups,
    /// Chunks of slots made under one lock of the slot state; of them,
    /// those that waited for it; and those timed part by part.
    Chunks,
    Waited,
    Timed,
    /// The diagnostics' looks at the shown buffer that waited for the slot
    /// state's lock.
    ShowsWaited,
}

const COUNTS: usize = 17;

/// One chunk of slots in this many is timed part by part.
pub(crate) const SAMPLE: u64 = 8;

/// The thread's state: not in a pass, counting, or counting and timing.
const OFF: u8 = 0;
const COUNTING: u8 = 1;
const TIMING: u8 = 2;

thread_local! {
    static STATE: Cell<u8> = const { Cell::new(OFF) };
    /// The part the thread is in, and the stamp it began at.
    static AT: Cell<(Part, u64)> = const { Cell::new((Part::Walk, 0)) };
    static TICKS: [Cell<u64>; PARTS] = const { [const { Cell::new(0) }; PARTS] };
    static COUNTED: [Cell<u64>; COUNTS] = const { [const { Cell::new(0) }; COUNTS] };
    /// Nanoseconds waited for the slot state's lock, and held.
    static WAITS: [Cell<u64>; 2] = const { [const { Cell::new(0) }; 2] };
}

/// Threads in a pass now: while none is, nothing here reads the thread's
/// state.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// The processor's time stamp counter (elsewhere, nanoseconds).
#[inline]
fn stamp() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: reads the processor's time stamp counter.
        unsafe { std::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

#[inline]
fn state() -> u8 {
    if ACTIVE.load(Relaxed) == 0 {
        return OFF;
    }
    STATE.get()
}

/// Whether this thread runs a pass now (and counts).
#[inline]
pub(crate) fn counting() -> bool {
    state() != OFF
}

/// Counts one of `what`, on a pass's thread.
#[inline]
pub(crate) fn count(what: Count) {
    if state() != OFF {
        COUNTED.with(|c| {
            let c = &c[what as usize];
            c.set(c.get() + 1);
        });
    }
}

/// A pass's thread waited `waited` for the slot state's lock.
pub(crate) fn waited(waited: Duration) {
    if state() != OFF {
        WAITS.with(|w| w[0].set(w[0].get() + waited.as_nanos() as u64));
        count(Count::Waited);
    }
}

/// A pass's thread held the slot state's lock for `held` (a chunk).
pub(crate) fn held(held: Duration) {
    if state() != OFF {
        WAITS.with(|w| w[1].set(w[1].get() + held.as_nanos() as u64));
        count(Count::Chunks);
    }
}

/// While it lives, the thread's time goes to the part it was made for (if
/// the thread times it); then to the part before.
#[must_use]
pub(crate) struct In(Option<Part>);

/// Enters `part` until the guard goes.
#[inline]
pub(crate) fn enter(part: Part) -> In {
    if state() != TIMING {
        return In(None);
    }
    In(Some(switch(part)))
}

impl Drop for In {
    #[inline]
    fn drop(&mut self) {
        if let Some(back) = self.0
            && STATE.get() == TIMING
        {
            switch(back);
        }
    }
}

/// Ends the part the thread is in, begins `to`; returns the one ended.
fn switch(to: Part) -> Part {
    let now = stamp();
    let (from, since) = AT.get();
    TICKS.with(|t| {
        let t = &t[from as usize];
        t.set(t.get() + now.wrapping_sub(since));
    });
    AT.set((to, now));
    from
}

/// What one thread of a pass took ([`Taking::end`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Took {
    /// Its time, and of it, waited for the slot state's lock and held.
    pub(crate) us: u64,
    pub(crate) waited_ns: u64,
    pub(crate) held_ns: u64,
    pub(crate) ticks: [u64; PARTS],
    pub(crate) counts: [u64; COUNTS],
}

impl Took {
    /// The count of `what`.
    #[cfg(test)]
    pub(crate) fn of(&self, what: Count) -> u64 {
        self.counts[what as usize]
    }

    /// The ticks of `part`.
    #[cfg(test)]
    pub(crate) fn ticks_of(&self, part: Part) -> u64 {
        self.ticks[part as usize]
    }
}

/// A pass's thread counting and timing, from [`Taking::begin`] to
/// [`Taking::end`] (or its drop, which throws what it took away).
pub(crate) struct Taking {
    began: Instant,
    chunks: u64,
    ended: bool,
}

impl Taking {
    /// This thread begins counting a pass (or a worker's share of one).
    pub(crate) fn begin() -> Taking {
        TICKS.with(|t| t.iter().for_each(|c| c.set(0)));
        COUNTED.with(|c| c.iter().for_each(|c| c.set(0)));
        WAITS.with(|w| w.iter().for_each(|c| c.set(0)));
        AT.set((Part::Walk, 0));
        STATE.set(COUNTING);
        ACTIVE.fetch_add(1, Relaxed);
        Taking { began: Instant::now(), chunks: 0, ended: false }
    }

    /// A chunk of slots begins: timed part by part if it is the one in
    /// [`SAMPLE`]. Called where no part's guard lives.
    pub(crate) fn chunk(&mut self) {
        let timed = self.chunks.is_multiple_of(SAMPLE);
        self.chunks += 1;
        match (STATE.get(), timed) {
            (TIMING, false) => {
                switch(Part::Walk);
                STATE.set(COUNTING);
            }
            (COUNTING, true) => {
                AT.set((Part::Walk, stamp()));
                STATE.set(TIMING);
            }
            _ => {}
        }
        if timed {
            count(Count::Timed);
        }
    }

    /// Ends the counting; what it took.
    pub(crate) fn end(mut self) -> Took {
        self.ended = true;
        if STATE.get() == TIMING {
            switch(Part::Walk);
        }
        let took = Took {
            us: self.began.elapsed().as_micros() as u64,
            waited_ns: WAITS.with(|w| w[0].get()),
            held_ns: WAITS.with(|w| w[1].get()),
            ticks: TICKS.with(|t| std::array::from_fn(|i| t[i].get())),
            counts: COUNTED.with(|c| std::array::from_fn(|i| c[i].get())),
        };
        stop();
        took
    }
}

impl Drop for Taking {
    fn drop(&mut self) {
        if !self.ended {
            stop();
        }
    }
}

fn stop() {
    STATE.set(OFF);
    ACTIVE.fetch_sub(1, Relaxed);
}

/// Since the last line: passes, their threads, their time on their threads
/// and from start to end, the longest thread's time a pass summed, and the
/// microseconds the workers' results took to be put in order.
static PASSES: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static TICKS_ALL: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static COUNTS_ALL: [AtomicU64; COUNTS] = [const { AtomicU64::new(0) }; COUNTS];
/// Nanoseconds waited and held, since the last line.
static WAITS_ALL: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Adds what the threads of one pass took (`took`), the pass's time from
/// start to end (`wall`), and the time its results took to be put in order
/// (`ordered`, with workers).
pub(crate) fn add(took: &[Took], wall: Duration, ordered: Duration) {
    if took.is_empty() {
        return;
    }
    let longest = took.iter().map(|t| t.us).max().unwrap_or(0);
    for (i, n) in [
        1,
        took.len() as u64,
        took.iter().map(|t| t.us).sum(),
        wall.as_micros() as u64,
        longest,
        ordered.as_micros() as u64,
    ]
    .into_iter()
    .enumerate()
    {
        PASSES[i].fetch_add(n, Relaxed);
    }
    for t in took {
        for (all, n) in TICKS_ALL.iter().zip(t.ticks) {
            all.fetch_add(n, Relaxed);
        }
        for (all, n) in COUNTS_ALL.iter().zip(t.counts) {
            all.fetch_add(n, Relaxed);
        }
        WAITS_ALL[0].fetch_add(t.waited_ns, Relaxed);
        WAITS_ALL[1].fetch_add(t.held_ns, Relaxed);
    }
}

/// The estimate in milliseconds of each part of `us` microseconds of
/// threads' time, of which `waited_ms` waited for the lock: each part's
/// share of the ticks taken (the waits left out), times the rest. `None`
/// when no chunk was timed.
fn estimates(ticks: &[u64; PARTS], us: u64, waited_ms: f64) -> Option<[f64; PARTS]> {
    let wait = Part::Wait as usize;
    let timed: u64 = ticks.iter().enumerate().filter(|&(i, _)| i != wait).map(|(_, t)| t).sum();
    if timed == 0 {
        return None;
    }
    let rest = (us as f64 / 1000.0 - waited_ms).max(0.0);
    Some(std::array::from_fn(|i| if i == wait { waited_ms } else { ticks[i] as f64 / timed as f64 * rest }))
}

/// With the frame line: what the passes did, per pass, since the last line.
/// Nothing while no pass ran.
pub(crate) fn line() {
    let p = PASSES.each_ref().map(|c| c.swap(0, Relaxed));
    let ticks = TICKS_ALL.each_ref().map(|c| c.swap(0, Relaxed));
    let counts = COUNTS_ALL.each_ref().map(|c| c.swap(0, Relaxed));
    let waits = WAITS_ALL.each_ref().map(|c| c.swap(0, Relaxed));
    let passes = p[0];
    if passes == 0 {
        return;
    }
    let per = |n: u64| n as f64 / passes as f64;
    let ms = |us: u64| us as f64 / 1000.0 / passes as f64;
    let [waited_ms, held_ms] = waits.map(|ns| ns as f64 / 1e6 / passes as f64);
    let us = p[2] / passes;
    let parts = match estimates(&ticks, us, waited_ms) {
        Some(est) => {
            let mut order: Vec<usize> = (0..PARTS).filter(|&i| i != Part::Wait as usize).collect();
            order.sort_by(|&a, &b| est[b].total_cmp(&est[a]));
            order.iter().map(|&i| format!("{} {:.1}", NAMES[i], est[i])).collect::<Vec<_>>().join(", ")
        }
        None => "no chunk timed".to_string(),
    };
    let c = |what: Count| per(counts[what as usize]);
    let workers = if p[1] > passes {
        format!(
            "; with VICTORY_HEAP_PASS_THREADS, the busiest thread {:.1} ms a pass and the results put in order {:.1} ms",
            ms(p[4]),
            ms(p[5])
        )
    } else {
        String::new()
    };
    tracing::info!(
        "the pass over buffer views, where its time goes, per pass ({passes} passes since the last line, {:.1} threads a \
         pass): {:.1} ms of its threads' time, {:.1} ms from its start to its end{workers}; {waited_ms:.1} ms waiting for \
         the lock of the heap's slot state ({:.0} of {:.0} chunks of up to 64 slots waited; the lock held {held_ms:.1} ms); \
         the rest, by the processor's counter in one chunk in {SAMPLE} ({:.0} chunks a pass), in ms: {parts}; slots: \
         {:.0} past the heap, {:.0} naming no view of ours, {:.0} without a pool slot, {:.0} made from their record \
         already, {:.0} copied from the cache, {:.0} made and kept in it, {:.0} made otherwise, {:.0} the host could not \
         make; {:.0} with a copy and range to keep, {:.0} without; heap map: {:.0} host slots looked up, {:.0} pool slots \
         taken; {:.0} host copies looked up; {:.0} of the diagnostics' looks at the shown buffer waited for the lock",
        per(p[1]),
        us as f64 / 1000.0,
        ms(p[3]),
        c(Count::Waited),
        c(Count::Chunks),
        c(Count::Timed),
        c(Count::Past),
        c(Count::NotOurs),
        c(Count::NoHost),
        c(Count::Same),
        c(Count::Hit),
        c(Count::Miss),
        c(Count::Other),
        c(Count::Unmade),
        c(Count::Range),
        c(Count::NoRange),
        c(Count::Mapped),
        c(Count::Taken),
        c(Count::Lookups),
        c(Count::ShowsWaited),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spins about `n` microseconds (the counter moves).
    fn spin(n: u64) {
        let began = Instant::now();
        while began.elapsed() < Duration::from_micros(n) {
            std::hint::spin_loop();
        }
    }

    /// Off a pass nothing is counted or timed; in one, counts always, and
    /// time only in the chunks timed, each part's own (a nested part's time
    /// is not its outer part's).
    #[test]
    fn parts_exclude_each_other_and_only_timed_chunks_are_timed() {
        count(Count::Same);
        drop(enter(Part::Copy));
        assert!(!counting(), "this thread runs no pass");
        let mut taking = Taking::begin();
        assert!(counting());
        for chunk in 0..SAMPLE * 2 {
            taking.chunk();
            count(Count::Same);
            let outer = enter(Part::Copy);
            spin(200);
            {
                let _inner = enter(Part::Lookup);
                spin(400);
            }
            spin(200);
            drop(outer);
            // A part entered off the timed chunks is not timed.
            if chunk % SAMPLE != 0 {
                let _r = enter(Part::Ranges);
                spin(100);
            }
        }
        let took = taking.end();
        assert!(!counting());
        assert_eq!(took.of(Count::Same), SAMPLE * 2, "counted in every chunk");
        assert_eq!(took.of(Count::Timed), 2);
        assert_eq!(took.ticks_of(Part::Ranges), 0, "nothing off the timed chunks");
        let (copy, lookup) = (took.ticks_of(Part::Copy), took.ticks_of(Part::Lookup));
        assert!(copy > 0 && lookup > 0);
        // About 800 us of each kind in each: the nested part is its own.
        let ratio = lookup as f64 / copy as f64;
        assert!((0.5..2.0).contains(&ratio), "lookup {lookup}, copy {copy}");
        // Nothing counted after the end.
        count(Count::Same);
        assert_eq!(COUNTED.with(|c| c[Count::Same as usize].get()), SAMPLE * 2);
    }

    /// The estimates share out the time without the waits by the ticks,
    /// the waits as measured.
    #[test]
    fn the_estimates_share_out_the_time_but_the_waits() {
        let mut ticks = [0u64; PARTS];
        ticks[Part::Copy as usize] = 300;
        ticks[Part::Lookup as usize] = 100;
        ticks[Part::Wait as usize] = 10_000;
        let est = estimates(&ticks, 50_000, 10.0).expect("timed");
        assert!((est[Part::Copy as usize] - 30.0).abs() < 1e-9);
        assert!((est[Part::Lookup as usize] - 10.0).abs() < 1e-9);
        assert!((est[Part::Wait as usize] - 10.0).abs() < 1e-9);
        assert_eq!(estimates(&[0; PARTS], 50_000, 0.0), None);
    }
}
