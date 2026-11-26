//! The pass over the view heap's buffer views for shaders that pick them by
//! index ([`super::State::heap_buffers`]), and VIcky's opt-in way to run it
//! on a thread of ours (`VICTORY_HEAP_PASS_BESIDE=1`; off by default).
//!
//! The pass runs once a frame, inside the list call of the first list that
//! binds such a shader. It looks at the slots marked as written since the
//! last pass (every buffer slot each 16th pass): each slot's host slot is
//! made from the title's record if that changed, the host copy its view
//! shows is looked up, and the range the view reads there is kept
//! ([`buffers`]). The spans of all kept ranges are brought up to date when
//! that list is submitted ([`due`]). In RT55c that was 167-203 ms a frame
//! on one of the title's recording threads, which the title pins to two and
//! a half cores: the frame is a loop through that thread's recording and
//! our submitting thread, and the pass was most of the recording.
//!
//! With the switch, the recording thread takes the slots, leaves a step in
//! the list where the refreshes went ([`Pending`]) and returns; a thread of
//! ours ([`lane`]) does the rest, and the submitting thread waits for it
//! before it reads the steps of the lists that need it
//! (`HostList::join_heap_passes`, at the start of `Host::submit`).
//!
//! What the pass reads, and when:
//! - Title memory: only the record of each slot it looks at, three words
//!   read in place when the pass gets to the slot, under the lock of the
//!   heap's slot state (`HostHeap::mirror_chunk`, `HostHeap::mirror_with`).
//!   The memory the views name is read when the list is submitted, by the
//!   refreshes, either way: the pass only says which ranges.
//! - Our own tables: the view a record names (fixed by the record: a view's
//!   entry gets another generation when its resource goes), the host copies
//!   of title memory, and the host heap's slot state.
//!
//! So what moves is when each record is read. In place, the pass read them
//! over its whole run, up to 200 ms after the call, while the title's other
//! recording threads went on writing slots: a slot one of them wrote before
//! the pass got to it was made from the new record; one written after it,
//! from the old record, then made null by that write and made again by the
//! next pass. Beside, every record is still read in place, under the slot
//! lock, when the pass gets to it, and the same holds, now also for what
//! the calling thread writes after its call (which could not come while the
//! pass ran on it). No slot is made from a record older than the one in
//! memory at that moment: the lock orders the making and the title's write,
//! as it always did. The records are not taken at the call to be made later
//! from that copy: a slot the title wrote after the call would then get its
//! old view back after the write had made it null, a step behind the title
//! (or, if such slots were left out, stay null where the pass in place
//! would often have made them). The console reads a slot when its shaders
//! run, after the submission: a record read later is, if anything, closer
//! to that. `VICTORY_HEAP_PASS_BESIDE_CHECK=1` counts how often a slot's
//! view changed between the call and the pass ([`compare`]).
//!
//! The order kept:
//! - The refreshes. The pass's spans become refresh steps of the list in
//!   place of its step, before anything walks the steps of a submission
//!   (the one write-watch asking for all of it, the look for ranges under
//!   flights' writes, the steps themselves). A span the list refreshed
//!   since its last title copy is left out, and a later refresh of a span
//!   put in is dropped up to the next title copy, as `defer_refresh` did
//!   when the pass recorded them ([`splice`]): the list's steps are the
//!   ones the pass in place would have left.
//! - The slots. Every list that asks for the heap's buffers in the frame
//!   waits for the frame's pass at its submission (`HostList::after_heap_pass`),
//!   not only the one with its step, which may come later in the frame's
//!   order: in place, the pass had ended before the call that ran it
//!   returned, and the title sends a frame's lists in a burst once it has
//!   recorded them, so they all ran after it. A list submitted while the
//!   pass runs now waits for it, where before it raced with it.
//! - One pass at a time, in the order of their calls: one thread, one
//!   queue. The ranges kept from pass to pass are that thread's while it
//!   runs ([`RANGES`]); the pass count is taken at the call, and a pass over
//!   every buffer slot clears the ranges when its turn comes.
//! - The heap. The pass holds the list of heaps for reading while it runs,
//!   so the title's release of the heap waits for it, as for the heap sync
//!   (`crate::descriptor::with_heap`); a pass whose heap went before its
//!   turn is not run, and the lists waiting for it go on.
//! - The pass over textures picked by index stays on the recording thread
//!   (it records uploads into the list), now beside the buffer pass instead
//!   of after it. The two share the slot state under its locks, as the
//!   title's threads and draws share it. The one thing the buffer pass did
//!   for it is to note again a slot that had left the buffer views for a
//!   texture view, which the title's write had queued already: at most an
//!   upload is done again a frame later.
//! - The frame for the refreshes of copies that are not write-watched is
//!   the call's.
//!
//! `VICTORY_HEAP_PASS_BESIDE_CHECK=1` (diagnostics): one pass in
//! [`CHECK_EVERY`] is also worked out the way the pass in place would have
//! found it at the call. The records are read at the call, before the pass
//! is handed over; then, while the pass runs, the calling thread finds each
//! one's view and the copy and range its view reads, with the copies looked
//! up anew (as `VICTORY_HEAP_SYNC_CHECK` asks them), making nothing. The
//! pass notes what it found, and the two are compared at its end. A slot
//! whose view differs changed between the call and the pass (the title
//! wrote it or released its view); one with the same view and another copy
//! or range differs, and the first few are logged. A host copy made or let
//! go in between counts as a difference too.
//!
//! A line with the frame line says per frame the time on the recording
//! thread, on our thread and waited for the passes, and the check's counts
//! ([`line`]).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use crate::descriptor::host::{HostHeap, Lookups, PASS_CHUNK, buffer_copy, buffer_copy_with, cache_size};
use crate::descriptor::{Heap, View};
use crate::descriptor::beside::{Lane, Ticket};
use crate::host::HostList;
use crate::shadow::Shadow;

/// A slot's range: the host copy its view shows and the title range it
/// reads there (from, to).
pub(crate) type Range = (Arc<Shadow>, u64, u64);

/// Per title slot, the range its view read when the pass last looked at it.
pub(crate) type Ranges = HashMap<u32, Range>;

/// Per host heap: the passes so far (in place) and the range each slot's
/// buffer view reads, kept from pass to pass. Only the slots written since
/// the last pass are looked at again (83,000 slots a pass otherwise, 0.1 s
/// at best); all of them every 16 passes, because a view of a released
/// resource goes stale without a write, and a slot written before its
/// memory had a host copy is only made then. Beside, the passes are counted
/// at the call ([`CALLS`]) and the count here stays 0.
static RANGES: Mutex<Option<HashMap<usize, (u64, Ranges)>>> = Mutex::new(None);

/// Beside: per host heap, the passes started so far, counted at the call.
static CALLS: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// The host device went while a pass made views: the view that removed it
/// was named once, and later passes no longer ask.
static GONE: AtomicBool = AtomicBool::new(false);

/// What a slot of the pass came to, for the check ([`compare`]): the view
/// its record named (none: no record of ours, or its view went), and the
/// copy (by number) and range it reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Found {
    pub(crate) index: u32,
    pub(crate) view: Option<View>,
    pub(crate) range: Option<(u64, u64, u64)>,
}

/// The pass over the buffer views of `heap` (its host heap `host`) for
/// `slots`, up to the ranges, kept in `ranges`. With `found`, what each
/// slot came to is noted there too (the check).
pub(crate) fn buffers(
    heap: &Heap,
    host: &HostHeap,
    slots: &mut [u32],
    ranges: &mut Ranges,
    mut found: Option<&mut Vec<Found>>,
) {
    // VIcky (`VICTORY_HEAP_SYNC_CACHE`): in slot order (the set of
    // changed slots gives them in no order). Each slot is made by itself,
    // so the order changes nothing of what is made; in order, the slots'
    // state and the heaps written are walked forward, not jumped about
    // (the bench in `descriptor::host::fast_tests`: the pass 20-30%
    // shorter).
    if cache_size() > 0 {
        slots.sort_unstable();
    }
    let mut alive = !GONE.load(Relaxed) && crate::descriptor::host::device_removed().is_none();
    super::HEAP_PASS_PARTS[1].fetch_add(slots.len() as u64, Relaxed);
    // VIcky (`VICTORY_HEAP_SYNC_CACHE`): the copies a slot's making
    // looks up are the ones `buffer_copy` asks for again: each looked
    // up once (`descriptor::host::Lookups`). And the slots are made a
    // chunk at a time, under one lock of the heap's slot state
    // (`HostHeap::mirror_chunk`), ahead of the rest of their work here.
    let shared = cache_size() > 0;
    let mut made_ahead: VecDeque<(Option<View>, Lookups)> = VecDeque::new();
    for (n, &index) in slots.iter().enumerate() {
        ranges.remove(&index);
        let Some(handle) = heap.handle(index) else {
            continue;
        };
        let (view, mut lookups) = if shared {
            if made_ahead.is_empty() {
                // This slot and the next ones that have a handle.
                let ahead: Vec<(u32, u64)> = slots[n..]
                    .iter()
                    .filter_map(|&i| Some((i, heap.handle(i)?)))
                    .take(PASS_CHUNK)
                    .collect();
                let mut lookups: Vec<Lookups> = ahead.iter().map(|_| Lookups::kept()).collect();
                let views = host.mirror_chunk(&ahead, &mut lookups);
                made_ahead.extend(views.into_iter().zip(lookups));
            }
            let Some(made) = made_ahead.pop_front() else {
                continue;
            };
            made
        } else {
            let mut lookups = Lookups::each();
            (host.mirror_with(index, handle, &mut lookups), lookups)
        };
        let Some(view) = view else {
            if let Some(found) = found.as_deref_mut() {
                found.push(Found { index, view: None, range: None });
            }
            continue;
        };
        // Every 64 views while the device lives (a cheap query).
        if alive
            && n % 64 == 63
            && let Some(why) = crate::descriptor::host::device_removed()
        {
            alive = false;
            GONE.store(true, Relaxed);
            let last: Vec<String> = slots[n - 63..=n]
                .iter()
                .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                .map(|s| s.chars().take(200).collect())
                .collect();
            tracing::error!(why, slot = index, "the host device went while buffer views of the heap were made; one of: {last:#?}");
            crate::host::log_removal_context();
        }
        // The copy the slot shows (`buffer_copy`), not the one the
        // view's range alone would find.
        let Some((shadow, address, len)) = buffer_copy_with(&view, &mut lookups) else {
            if let Some(found) = found.as_deref_mut() {
                found.push(Found { index, view: Some(view), range: None });
            }
            continue;
        };
        let (shadow, address, len) = if shared {
            super::HEAP_PASS_FAST[0].fetch_add(1, Relaxed);
            super::HEAP_PASS_FAST[1].fetch_add(lookups.asked(), Relaxed);
            super::shared_copy_checked(&view, (shadow, address, len))
        } else {
            (shadow, address, len)
        };
        #[cfg(feature = "diagnostics")]
        if host.shows(index, &shadow.buffer) == Some(false) {
            super::HEAP_VIEWS_ELSEWHERE.fetch_add(1, Relaxed);
        }
        if let Some(found) = found.as_deref_mut() {
            found.push(Found { index, view: Some(view), range: Some((shadow.id(), address, address + len)) });
        }
        ranges.insert(index, (shadow, address, address + len));
    }
}

/// Per host copy (by its address), the span the kept ranges cover on it.
pub(crate) fn spans(ranges: &Ranges) -> HashMap<usize, Range> {
    let mut spans: HashMap<usize, Range> = HashMap::new();
    for (shadow, from, to) in ranges.values() {
        let key = Arc::as_ptr(shadow) as usize;
        let span = spans.entry(key).or_insert_with(|| (shadow.clone(), *from, *to));
        span.1 = span.1.min(*from);
        span.2 = span.2.max(*to);
    }
    spans
}

/// Of `spans`, the ones to bring up to date for the pass of `frame`, in
/// their order.
pub(crate) fn due(spans: &HashMap<usize, Range>, frame: u64) -> Vec<&Range> {
    // Memory that is not write-watched is copied whole by every refresh.
    // It was once refreshed only every 120 frames (unmapped pages read
    // one by one cost 14 GB a frame in run 11, before they were
    // remembered): streamed geometry then drew from stale buffers, and
    // the garbage ended runs (the card stopped answering 14-90 frames
    // into the opening cutscene; with every frame it ran on). The pass
    // costs some 30 ms a frame this way.
    static WHOLE: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);
    let mut whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
    let whole = whole.get_or_insert_with(Default::default);
    let mut due = Vec::with_capacity(spans.len());
    for span in spans.values() {
        let shadow = &span.0;
        if shadow.unwatched() {
            // By the copy's number: a copy made where one let go was
            // (`shadow::let_go_idle`) has never been refreshed.
            let key = shadow.id() as usize;
            // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
            static EVERY: OnceLock<u64> = OnceLock::new();
            let every = *EVERY.get_or_init(|| {
                std::env::var("VICTORY_UNWATCHED_FRAMES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1)
            });
            match whole.get(&key) {
                Some(&last) if frame < last + every => continue,
                _ => {
                    whole.insert(key, frame);
                }
            }
        }
        due.push(span);
    }
    due
}

/// What one pass made ready, for [`said`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Made {
    /// Slots looked at, views with a range, copies they are on, and the
    /// bytes of the spans on those copies.
    pub(crate) slots: usize,
    pub(crate) views: u32,
    pub(crate) copies: usize,
    pub(crate) bytes: u64,
}

/// The pass as it always ran (without `VICTORY_HEAP_PASS_BESIDE`): on the
/// calling thread, its refreshes recorded into `list` now.
pub(crate) fn in_place(list: &HostList, heap: &Heap, host: &HostHeap, frame: u64) -> Made {
    let mut kept = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let (passes, ranges) = kept
        .get_or_insert_with(Default::default)
        .entry(std::ptr::from_ref(host) as usize)
        .or_insert_with(|| (0, Ranges::new()));
    let changed = host.buffer_changes();
    let mut slots = if *passes % 16 == 0 {
        ranges.clear();
        host.buffer_slots()
    } else {
        changed
    };
    *passes += 1;
    buffers(heap, host, &mut slots, ranges, None);
    let views = ranges.len() as u32;
    let spans = spans(ranges);
    drop(kept);
    let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
    for (shadow, from, to) in due(&spans, frame) {
        shadow.refresh(list, *from, to - from);
    }
    Made { slots: slots.len(), views, copies: spans.len(), bytes }
}

/// Logs what a pass made ready: the first pass, and every growth of the
/// set of views by half. `started`: when the pass began.
pub(crate) fn said(made: Made, started: Instant) {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    let seen = LOGGED.load(Relaxed);
    if u64::from(made.views) > seen + seen / 2 {
        LOGGED.store(u64::from(made.views), Relaxed);
        tracing::info!(
            slots = made.slots,
            views = made.views,
            copies = made.copies,
            mb = made.bytes >> 20,
            ms = started.elapsed().as_millis(),
            "buffer views of the heap made ready for shaders that pick them by index"
        );
    }
}

/// Whether the pass runs on a thread of ours (`VICTORY_HEAP_PASS_BESIDE=1`).
pub(crate) fn beside() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_HEAP_PASS_BESIDE").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "the pass over the heap's buffer views runs on a thread of its own, and the lists that need it wait for it \
                 at their submission (VICTORY_HEAP_PASS_BESIDE=1)"
            );
        }
        on
    })
}

/// Whether one pass in [`CHECK_EVERY`] is checked against what the pass
/// in place would have found at the call (`VICTORY_HEAP_PASS_BESIDE_CHECK=1`,
/// with the pass beside).
fn check_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| beside() && std::env::var_os("VICTORY_HEAP_PASS_BESIDE_CHECK").is_some_and(|v| v == "1"))
}

/// One pass in this many is checked (`VICTORY_HEAP_PASS_BESIDE_CHECK`),
/// between two passes over every buffer slot.
pub(crate) const CHECK_EVERY: u64 = 16;

/// A pass beside, from the call that hands it over to the lists that wait
/// for it: ended once, by the pass or, if the pass never runs, by the
/// dropping of its [`Filler`], so that no list waits for ever.
#[derive(Default)]
pub(crate) struct Pending {
    state: Mutex<Ended>,
    moved: Condvar,
}

/// What a pass beside left.
#[derive(Default)]
struct Ended {
    done: bool,
    /// The spans to bring up to date, for the list with the pass's step.
    refreshes: Vec<Range>,
    /// The pass's panic, given to the one who takes the refreshes.
    panic: Option<Box<dyn std::any::Any + Send>>,
    /// What the check found, if the pass was checked (read by the tests;
    /// the line has it from [`CHECKS`]).
    #[cfg_attr(not(test), allow(dead_code))]
    checked: Option<Checked>,
}

impl Pending {
    fn ended(&self) -> MutexGuard<'_, Ended> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while !state.done {
            state = self.moved.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state
    }

    /// Waits until the pass has ended (counted for the line: a wait that
    /// found it running, and how long).
    pub(crate) fn wait(&self) {
        let began = Instant::now();
        let running = !self.state.lock().unwrap_or_else(|e| e.into_inner()).done;
        drop(self.ended());
        count(C::Waits, 1);
        if running {
            count(C::Late, 1);
            count(C::WaitedUs, began.elapsed().as_micros() as u64);
        }
    }

    /// Once the pass has ended, the spans it left for the list with its
    /// step (taken: a second call gets none). A panic of the pass goes on
    /// here, as it went on on the recording thread before.
    pub(crate) fn take_refreshes(&self) -> Vec<Range> {
        let mut state = self.ended();
        if let Some(panic) = state.panic.take() {
            drop(state);
            std::panic::resume_unwind(panic);
        }
        std::mem::take(&mut state.refreshes)
    }

    /// Ends the pass with what it left, and wakes whoever waits.
    fn end(&self, refreshes: Vec<Range>, panic: Option<Box<dyn std::any::Any + Send>>, checked: Option<Checked>) {
        // What the pass wrote into memory the card reads (descriptors,
        // the stride table; write-combined) leaves this thread's buffers
        // before a waiting thread goes on, as `descriptor::beside` does.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = Ended { done: true, refreshes, panic, checked };
        self.moved.notify_all();
    }

    /// What the check found, once the pass has ended (tests).
    #[cfg(test)]
    fn checked(&self) -> Option<Checked> {
        self.ended().checked
    }
}

/// The pass's side of a [`Pending`]: ends it with what the pass left, or
/// with nothing if it is dropped first (a pass never handed over, or one
/// whose heap went before its turn).
pub(crate) struct Filler(Option<Arc<Pending>>);

impl Filler {
    fn end(mut self, refreshes: Vec<Range>, panic: Option<Box<dyn std::any::Any + Send>>, checked: Option<Checked>) {
        if let Some(pass) = self.0.take() {
            pass.end(refreshes, panic, checked);
        }
    }
}

impl Drop for Filler {
    fn drop(&mut self) {
        if let Some(pass) = self.0.take() {
            pass.end(Vec::new(), None, None);
        }
    }
}

/// A pass for this call to start: its [`Pending`], which the frame's other
/// lists wait for, and the side that ends it.
pub(crate) struct Run {
    pass: Arc<Pending>,
    filler: Filler,
}

impl Run {
    fn new() -> Run {
        let pass = Arc::new(Pending::default());
        Run { filler: Filler(Some(pass.clone())), pass }
    }
}

/// What a list's first ask for the heap's buffers in a frame does beside.
pub(crate) enum Turn {
    /// The frame's pass is started already: the list waits for it.
    Wait(Arc<Pending>),
    /// This list's call starts it ([`start`]).
    Run(Run),
}

/// The frame of the pass last started beside, and that pass: one pass a
/// frame, as the pass in place has it, and the pass the frame's other lists
/// wait for, named under the same lock.
static OF_FRAME: Mutex<Option<(u64, Arc<Pending>)>> = Mutex::new(None);

/// For a list's first ask in `frame`: the frame's pass to wait for, or a
/// new one for this list to start.
pub(crate) fn turn(frame: u64) -> Turn {
    let mut of = OF_FRAME.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((started, pass)) = &*of
        && *started == frame
    {
        return Turn::Wait(pass.clone());
    }
    let run = Run::new();
    *of = Some((frame, run.pass.clone()));
    Turn::Run(run)
}

/// The thread the passes beside run on, made at the first; `None` if none
/// could be made (the passes then run in place, as before, with their
/// steps).
fn lane() -> Option<&'static Arc<Lane>> {
    static LANE: OnceLock<Option<Arc<Lane>>> = OnceLock::new();
    LANE.get_or_init(|| {
        let lane = Lane::new("victory-heap-pass");
        if lane.is_none() {
            tracing::warn!("no thread for the pass over buffer views (VICTORY_HEAP_PASS_BESIDE=1): it runs on the recording thread");
        }
        lane
    })
    .as_ref()
}

/// What a call hands to its pass beside.
struct Call {
    /// The title heap (its address and first descriptor) and its host heap
    /// (its address), as the call had them.
    heap: usize,
    start: u64,
    host: usize,
    slots: Vec<u32>,
    /// A pass over every buffer slot: the kept ranges are cleared first.
    full: bool,
    frame: u64,
    called: Instant,
    /// With the check: what the pass in place would have found at the
    /// call, sent once the calling thread has worked it out.
    at_call: Option<std::sync::mpsc::Receiver<Vec<Found>>>,
    /// Tests: the calling thread's switches of the passes over records
    /// (`descriptor::host::TEST_FAST`), for the lane's thread.
    #[cfg(test)]
    fast: Option<(usize, bool, bool)>,
}

/// The pass beside for the call of `list` that runs the frame's pass
/// (`run`, from [`turn`]): takes the slots to look at, leaves the pass's
/// step in `list` at this point, and hands the pass to [`lane`]. `check`:
/// check this pass whatever the turn (tests; else one in [`CHECK_EVERY`]
/// with the check switch). With the check, the calling thread works out
/// what the pass in place would have found before it returns.
pub(crate) fn start(list: &HostList, heap: &Heap, host: &HostHeap, frame: u64, run: Run, check: bool) {
    let called = Instant::now();
    let key = std::ptr::from_ref(host) as usize;
    let n = {
        let mut calls = CALLS.lock().unwrap_or_else(|e| e.into_inner());
        let passes = calls.get_or_insert_with(Default::default).entry(key).or_insert(0);
        *passes += 1;
        *passes - 1
    };
    let full = n.is_multiple_of(16);
    // As in place: the marks are taken whichever slots the pass looks at.
    let changed = host.buffer_changes();
    let slots = if full { host.buffer_slots() } else { changed };
    let lane = lane();
    // (Not without the lane: the pass would run here, before the answers.)
    let check = lane.is_some() && (check || (check_on() && n % CHECK_EVERY == CHECK_EVERY / 2));
    let records = check.then(|| records_at_call(heap, host, &slots));
    let (answer, at_call) = if check {
        let (answer, at_call) = std::sync::mpsc::sync_channel(1);
        (Some(answer), Some(at_call))
    } else {
        (None, None)
    };
    let Run { pass, filler } = run;
    list.defer_heap_pass(&pass);
    let call = Call {
        heap: std::ptr::from_ref(heap) as usize,
        start: heap.start,
        host: key,
        slots,
        full,
        frame,
        called,
        at_call,
        #[cfg(test)]
        fast: crate::descriptor::host::TEST_FAST.get(),
    };
    give(lane.map(|lane| &**lane), call, filler);
    count(C::Passes, 1);
    count(C::CallUs, called.elapsed().as_micros() as u64);
    if let (Some(records), Some(answer)) = (records, answer) {
        let began = Instant::now();
        // (A pass that ended without its heap no longer listens.)
        let _ = answer.send(answers(records));
        CHECKS[K::CallUs as usize].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
    }
}

/// Runs `call` on `lane` (or here, without one), ending its pass through
/// `filler`.
fn give(lane: Option<&Lane>, call: Call, filler: Filler) {
    let job = move || match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(call))) {
        Ok(Some((refreshes, checked))) => filler.end(refreshes, None, checked),
        // The heap went: nothing to bring up to date.
        Ok(None) => drop(filler),
        Err(panic) => filler.end(Vec::new(), Some(panic), None),
    };
    match lane {
        Some(lane) => {
            let _ = lane.give(Box::new(move |_: &Ticket| job()));
        }
        None => job(),
    }
}

/// The pass of `call`, on the lane's thread: what to bring up to date, and
/// what the check found. `None` if its heap went before its turn.
fn run(call: Call) -> Option<(Vec<Range>, Option<Checked>)> {
    let began = Instant::now();
    count(C::QueuedUs, (began - call.called).as_micros() as u64);
    #[cfg(test)]
    let _fast = TestFast::here(call.fast);
    let Call { heap, start, host, mut slots, full, frame, at_call, .. } = call;
    let (refreshes, found) = crate::descriptor::with_heap(heap, start, |heap| {
        let here = heap.host().filter(|h| std::ptr::from_ref(*h) as usize == host)?;
        let mut kept = RANGES.lock().unwrap_or_else(|e| e.into_inner());
        let (_, ranges) = kept
            .get_or_insert_with(Default::default)
            .entry(host)
            .or_insert_with(|| (0, Ranges::new()));
        if full {
            ranges.clear();
        }
        let mut found = at_call.is_some().then(Vec::new);
        buffers(heap, here, &mut slots, ranges, found.as_mut());
        let views = ranges.len() as u32;
        let spans = spans(ranges);
        drop(kept);
        let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
        let refreshes: Vec<Range> = due(&spans, frame).into_iter().cloned().collect();
        said(Made { slots: slots.len(), views, copies: spans.len(), bytes }, began);
        count(C::Slots, slots.len() as u64);
        Some((refreshes, found))
    })
    .flatten()?;
    count(C::OwnUs, began.elapsed().as_micros() as u64);
    let checked = match (found, at_call) {
        (Some(found), Some(at_call)) => at_call.recv().ok().map(|at_call| compare(&at_call, &found)),
        _ => None,
    };
    if let Some(checked) = checked {
        for (k, n) in [(K::Passes, 1), (K::Slots, checked.slots), (K::Differed, checked.differed), (K::Changed, checked.changed)] {
            CHECKS[k as usize].fetch_add(n, Relaxed);
        }
    }
    Some((refreshes, checked))
}

/// The record of each of `slots` that has a handle, read in place now, as
/// the pass reads it (none past the slots the host heap mirrors).
fn records_at_call(heap: &Heap, host: &HostHeap, slots: &[u32]) -> Vec<(u32, Option<[u32; 3]>)> {
    let mirrored = host.mirrored();
    slots
        .iter()
        .filter_map(|&index| {
            let handle = heap.handle(index)?;
            Some((index, (index < mirrored).then(|| crate::descriptor::record_words(handle))))
        })
        .collect()
}

/// What the pass in place would have found for `records`: each one's view,
/// and the copy and range it reads, the copies looked up anew. Nothing is
/// made.
fn answers(records: Vec<(u32, Option<[u32; 3]>)>) -> Vec<Found> {
    records
        .into_iter()
        .map(|(index, record)| {
            let view = record.and_then(crate::descriptor::record_view);
            let range = view.as_ref().and_then(buffer_copy).map(|(s, a, l)| (s.id(), a, a + l));
            Found { index, view, range }
        })
        .collect()
}

/// What the check found over one pass ([`compare`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Checked {
    /// Slots compared.
    pub(crate) slots: u64,
    /// Of them, the ones with the same view and another copy or range (or
    /// found on one side only).
    pub(crate) differed: u64,
    /// The ones whose view changed between the call and the pass.
    pub(crate) changed: u64,
}

/// Compares what the pass beside found (`beside`) with what the pass in
/// place would have found at the call (`at_call`), slot by slot.
pub(crate) fn compare(at_call: &[Found], beside: &[Found]) -> Checked {
    /// Logs the first few differences.
    fn say(index: u32, then: Option<&Found>, now: Option<&Found>) {
        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if SAID.fetch_add(1, Relaxed) < 16 {
            tracing::warn!(
                slot = index,
                "VICTORY_HEAP_PASS_BESIDE_CHECK: the pass beside found another copy or range than the pass at the call \
                 would have: at the call {:?}, beside {:?}",
                then.map(|f| (&f.view, f.range)),
                now.map(|f| (&f.view, f.range)),
            );
        }
    }
    let mut at: HashMap<u32, &Found> = at_call.iter().map(|f| (f.index, f)).collect();
    let mut checked = Checked::default();
    for now in beside {
        checked.slots += 1;
        match at.remove(&now.index) {
            Some(then) if then.view != now.view => checked.changed += 1,
            Some(then) if then.range == now.range => {}
            then => {
                checked.differed += 1;
                say(now.index, then, Some(now));
            }
        }
    }
    // Slots the call had that the pass did not get to.
    for (index, then) in at {
        checked.slots += 1;
        checked.differed += 1;
        say(index, Some(then), None);
    }
    checked
}

/// How [`splice`] sees a step of a list.
pub(crate) enum Seen<K> {
    /// A title copy, or work that may write title memory: what the list
    /// counts as refreshed starts again after it (`HostList::defer_copy`,
    /// `HostList::defer`).
    Barrier,
    /// A refresh of the range `K` (copy, address, bytes).
    Refresh(K),
    /// Anything else.
    Other,
}

/// Puts `refreshes` (each with its key) in place of the step at `at`, as
/// `HostList::defer_refresh` would have recorded them there: one whose key
/// the steps since the last barrier before `at` refresh is left out (and a
/// key given twice is put in once); and a refresh of a key put in that
/// comes after `at`, up to the next barrier, is dropped, since
/// `defer_refresh` would have found it refreshed. Returns how many were
/// put in.
pub(crate) fn splice<T, K: Eq + std::hash::Hash + Copy>(
    steps: &mut Vec<T>,
    at: usize,
    refreshes: Vec<(K, T)>,
    seen: impl Fn(&T) -> Seen<K>,
) -> usize {
    let since = steps[..at]
        .iter()
        .rposition(|step| matches!(seen(step), Seen::Barrier))
        .map_or(0, |barrier| barrier + 1);
    let mut refreshed: HashSet<K> = steps[since..at]
        .iter()
        .filter_map(|step| match seen(step) {
            Seen::Refresh(key) => Some(key),
            _ => None,
        })
        .collect();
    let mut added = HashSet::new();
    let mut put = Vec::new();
    for (key, step) in refreshes {
        if refreshed.insert(key) {
            added.insert(key);
            put.push(step);
        }
    }
    let n = put.len();
    let after = steps.split_off(at + 1);
    steps.pop();
    steps.extend(put);
    let mut open = true;
    for step in after {
        match seen(&step) {
            Seen::Barrier => open = false,
            Seen::Refresh(key) if open && added.contains(&key) => continue,
            _ => {}
        }
        steps.push(step);
    }
    n
}

/// The line's counts, since the last line ([`COUNTS`]).
#[derive(Clone, Copy)]
enum C {
    /// Passes handed over beside.
    Passes,
    /// Microseconds of their calls on the recording thread (with the
    /// check's reading of the records, without its answers).
    CallUs,
    /// Microseconds of the passes on our thread (without the check).
    OwnUs,
    /// Microseconds from the calls to the passes' starts.
    QueuedUs,
    /// Slots the passes looked at.
    Slots,
    /// Waits of lists for passes, those that found the pass running, and
    /// the microseconds those waited.
    Waits,
    Late,
    WaitedUs,
}

static COUNTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn count(what: C, n: u64) {
    COUNTS[what as usize].fetch_add(n, Relaxed);
}

/// The check's counts, since the last line ([`CHECKS`]).
#[derive(Clone, Copy)]
enum K {
    /// Passes checked; slots compared; of them differing; changed between
    /// the call and the pass ([`Checked`]).
    Passes,
    Slots,
    Differed,
    Changed,
    /// Microseconds of the recording threads for the answers at the call.
    CallUs,
}

static CHECKS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// With the frame line: what the passes beside did per frame over
/// `frames`. Nothing without the switch, or while no pass ran and no list
/// waited for one.
pub(crate) fn line(frames: f64) {
    if !beside() {
        return;
    }
    let c = COUNTS.each_ref().map(|c| c.swap(0, Relaxed));
    let k = CHECKS.each_ref().map(|c| c.swap(0, Relaxed));
    let get = |what: C| c[what as usize];
    let (passes, waits) = (get(C::Passes), get(C::Waits));
    if passes == 0 && waits == 0 {
        return;
    }
    let ms = |us: u64| us as f64 / 1000.0 / frames;
    let check = if check_on() {
        let checked = k[K::Passes as usize];
        format!(
            "; VICTORY_HEAP_PASS_BESIDE_CHECK (one pass in {CHECK_EVERY}, since the last line): {checked} passes, {} slots \
             compared, {} differed (the same view, another copy or range), {} changed between the call and the pass (the \
             title wrote the slot or released its view); {:.1} ms of the recording thread a checked pass for the answers at \
             the call",
            k[K::Slots as usize],
            k[K::Differed as usize],
            k[K::Changed as usize],
            k[K::CallUs as usize] as f64 / 1000.0 / checked.max(1) as f64,
        )
    } else {
        String::new()
    };
    tracing::info!(
        "the pass over buffer views beside (VICTORY_HEAP_PASS_BESIDE=1) per frame: {:.1} passes; {:.2} ms on the title's \
         recording thread to take the slots and hand the pass over (the frame line's \"heap buffers ... ms\" is that and the \
         pass over textures); {:.1} ms on our thread for {:.0} slots, each pass starting {:.1} ms after its call; the \
         submitting thread waited {:.2} ms for them ({} of {} waits of lists found the pass running){check}",
        passes as f64 / frames,
        ms(get(C::CallUs)),
        ms(get(C::OwnUs)),
        get(C::Slots) as f64 / frames,
        get(C::QueuedUs) as f64 / 1000.0 / passes.max(1) as f64,
        ms(get(C::WaitedUs)),
        get(C::Late),
        waits,
    );
}

/// Tests: sets the switches of the passes over records
/// (`descriptor::host::TEST_FAST`) on this thread while it lives, as the
/// calling thread had them.
#[cfg(test)]
struct TestFast(Option<(usize, bool, bool)>);

#[cfg(test)]
impl TestFast {
    fn here(fast: Option<(usize, bool, bool)>) -> TestFast {
        let was = crate::descriptor::host::TEST_FAST.get();
        crate::descriptor::host::TEST_FAST.set(fast);
        TestFast(was)
    }
}

#[cfg(test)]
impl Drop for TestFast {
    fn drop(&mut self) {
        crate::descriptor::host::TEST_FAST.set(self.0);
    }
}

/// Tests: forgets what the passes kept of `host` (a heap made at the
/// address of one an earlier test released would find its ranges).
#[cfg(test)]
fn forget(host: &HostHeap) {
    let key = std::ptr::from_ref(host) as usize;
    if let Some(kept) = RANGES.lock().unwrap().as_mut() {
        kept.remove(&key);
    }
    if let Some(calls) = CALLS.lock().unwrap().as_mut() {
        calls.remove(&key);
    }
}

#[cfg(test)]
mod tests;
