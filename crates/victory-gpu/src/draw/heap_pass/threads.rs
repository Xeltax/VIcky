//! VIcky (`VICTORY_HEAP_PASS_THREADS=<n>`; off by default): the pass over
//! the heap's buffer views ([`super::buffers`]) on `n` threads, the one that
//! runs it and `n - 1` more, made for the pass and gone with it.
//!
//! On one thread the pass took 130-147 ms a frame in the runs of
//! 2026-11-26 (FP0b, FB1: some 78,000 slots a pass, about 1.8 us a slot),
//! and with the pass beside, the frame cannot be shorter than the pass.
//!
//! How:
//! - The pass's slots, in order, are cut in chunks of [`PASS_CHUNK`]. Each
//!   thread takes the next chunk no thread has taken until none is left,
//!   makes its slots (`HostHeap::mirror_shared`) and works out each one's
//!   copy and range, as the pass on one thread does. No two threads make
//!   the same slot.
//! - The heap's slot state is held by the pass's threads together (its read
//!   lock); every other user (a title write, the heap sync, the draws, the
//!   texture pass) takes it alone, as before, and waits for the chunks being
//!   made, as it waited for the pass's chunk. What the making of one slot
//!   changes in it is that slot's own (its key, its heap map entry), and the
//!   heap map's pool and the cache of made views, each behind a lock of its
//!   own (`descriptor::host`'s `Made`).
//! - A thread does not give back the pool slots of other slots to make room
//!   in a used-up pool (another thread may be making them): such slots are
//!   made once all the threads are done (`HostHeap::make_room`).
//! - Then the thread that runs the pass puts the results in the order of the
//!   slots: each slot's kept range dropped and its new one kept, and what
//!   the check of the pass beside notes ([`super::Found`]), as the pass on
//!   one thread leaves them.
//!
//! So the order kept by the pass beside (`super`'s docs) holds as it is: the
//! pass's refreshes are worked out from the kept ranges after this, on the
//! thread that runs the pass, and the lists wait for that pass where they
//! waited.
//!
//! What may differ from the pass on one thread: the order in which the slots
//! are made (each slot is made by itself, so what is made of each is the
//! same), and with it which pool slot a title slot past the identity window
//! takes, which records the cache keeps, and, when the pool is used up and
//! too few pool slots are given back for all that want one, which slots go
//! without (on one thread, the last in slot order); and the moment each
//! record is read (as between the pass in place and the pass beside: under
//! the slot state's lock, when the slot's turn comes).
//! `VICTORY_HEAP_PASS_THREADS_CHECK=1` (diagnostics) runs one pass in
//! [`CHECK_EVERY`] again on one thread at once and compares the two slot by
//! slot ([`check`]).
//!
//! The idea of making a heap's slots on several threads with the slot state
//! in pieces is fxceless's (fork commits 6c17ce9 and c3e643e, per-slot
//! locks and slot sets as bits, `VICTORY_VIEW_THREADS`); this is written
//! for VIcky's slot state (the heap map, the cache of made views), which
//! that fork does not have.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use super::parts::{self, Count, Part, Taking, Took, enter};
use super::{Found, GONE, Range, Ranges};
use crate::descriptor::Heap;
use crate::descriptor::host::{HostHeap, Lookups, PASS_CHUNK, buffer_copy_with, cache_size};

/// At most this many threads a pass.
const MOST: usize = 16;

/// The threads a pass runs on (`VICTORY_HEAP_PASS_THREADS`): 0 or 1, one
/// (the pass as before).
pub(crate) fn threads() -> usize {
    #[cfg(test)]
    if let Some(n) = TEST_THREADS.get() {
        return n;
    }
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("VICTORY_HEAP_PASS_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0)
            .min(MOST);
        if n > 1 {
            tracing::info!(
                threads = n,
                "the pass over the heap's buffer views runs on several threads, with the heap's slot state held by \
                 them together (VICTORY_HEAP_PASS_THREADS)"
            );
        }
        n
    })
}

#[cfg(test)]
thread_local! {
    /// Tests: [`threads`] and [`check_on`] for this thread.
    pub(crate) static TEST_THREADS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    pub(crate) static TEST_CHECK: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The threads for a pass over `slots` slots; `None` for one: the switch
/// off, or fewer than two chunks.
pub(crate) fn wanted(slots: usize) -> Option<usize> {
    let n = threads();
    (n > 1 && slots >= 2 * PASS_CHUNK).then(|| n.min(slots.div_ceil(PASS_CHUNK)))
}

/// Whether one pass in [`CHECK_EVERY`] is run again on one thread and
/// compared (`VICTORY_HEAP_PASS_THREADS_CHECK=1`).
pub(crate) fn check_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_CHECK.get() {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_PASS_THREADS_CHECK").is_some_and(|v| v == "1"))
}

/// One pass on several threads in this many is checked.
pub(crate) const CHECK_EVERY: u64 = 16;

/// Passes on several threads so far (for the check's turn).
static PASSES: AtomicU64 = AtomicU64::new(0);

/// One slot as a thread left it, for the thread that puts the results in
/// order.
struct Slot {
    index: u32,
    /// What the check of the pass beside notes, or this one's (none for a
    /// slot past the title heap, as on one thread).
    found: Option<Found>,
    range: Option<Range>,
}

/// The pass over `slots` of `heap` (its host heap `host`) on `n` threads,
/// as [`super::buffers`] does it on one: the ranges kept in `ranges`, what
/// each slot came to noted in `found` if asked; `marks` as there. `began`:
/// when the pass began, for the line.
#[allow(clippy::too_many_arguments)]
pub(crate) fn buffers(
    heap: &Heap,
    host: &HostHeap,
    slots: &mut [u32],
    ranges: &mut Ranges,
    found: Option<&mut Vec<Found>>,
    marks: bool,
    n: usize,
    began: Instant,
) {
    slots.sort_unstable();
    let check = check_on() && PASSES.fetch_add(1, Relaxed) % CHECK_EVERY == CHECK_EVERY / 2;
    let noting = found.is_some() || check;
    let done: Vec<Mutex<Vec<Slot>>> = (0..slots.len().div_ceil(PASS_CHUNK)).map(|_| Mutex::default()).collect();
    let next = AtomicUsize::new(0);
    let shared = cache_size() > 0;
    #[cfg(test)]
    let fast = crate::descriptor::host::TEST_FAST.get();
    let (heap_at, host_at) = (std::ptr::from_ref(heap) as usize, std::ptr::from_ref(host) as usize);
    let slots_ro: &[u32] = slots;
    // (Read by the threads, changed only once they are done.)
    let kept: &Ranges = ranges;
    let one = || {
        #[cfg(test)]
        let _fast = super::TestFast::here(fast);
        // SAFETY: the heap and its host heap outlive the threads: the
        // caller holds them (the pass beside, the list of heaps for
        // reading: `crate::descriptor::with_heap`) until this returns,
        // which is after every thread is joined.
        let (heap, host) = unsafe { (&*(heap_at as *const Heap), &*(host_at as *const HostHeap)) };
        work(heap, host, (slots_ro, kept), &next, &done, (marks, shared, noting))
    };
    let one = &one;
    let results: Vec<(Took, Vec<(u32, u64)>)> = std::thread::scope(|s| {
        let others: Vec<_> = (1..n)
            .filter_map(|i| {
                std::thread::Builder::new()
                    .name(format!("victory-heap-pass-{i}"))
                    .spawn_scoped(s, one)
                    .ok()
            })
            .collect();
        let mut all = vec![one()];
        for other in others {
            match other.join() {
                Ok(result) => all.push(result),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        all
    });
    let ordering = Instant::now();
    let (took, room): (Vec<Took>, Vec<Vec<(u32, u64)>>) = results.into_iter().unzip();
    let mut room: Vec<(u32, u64)> = room.into_iter().flatten().collect();
    // In the order of the slots, as the pass on one thread comes to them.
    room.sort_unstable();
    if !room.is_empty() {
        let made = host.make_room(&room, marks);
        ROOM[0].fetch_add(room.len() as u64, Relaxed);
        ROOM[1].fetch_add(made as u64, Relaxed);
    }
    let mut noted: Vec<Found> = Vec::new();
    for chunk in done {
        for slot in chunk.into_inner().unwrap_or_else(|e| e.into_inner()) {
            ranges.remove(&slot.index);
            if let Some(f) = slot.found {
                noted.push(f);
            }
            if let Some(range) = slot.range {
                ranges.insert(slot.index, range);
            }
        }
    }
    let ordered = ordering.elapsed();
    parts::add(&took, began.elapsed(), ordered);
    if check {
        self::check(heap, host, slots, &noted, marks);
    }
    if let Some(found) = found {
        found.extend(noted);
    }
}

/// One thread of a pass: takes chunks of `slots` until none is left, each
/// slot's result put in `done` at its chunk. Returns what it counted, and
/// the slots (title slot, handle) it left without a pool slot.
fn work(
    heap: &Heap,
    host: &HostHeap,
    (slots, kept): (&[u32], &Ranges),
    next: &AtomicUsize,
    done: &[Mutex<Vec<Slot>>],
    (marks, shared, noting): (bool, bool, bool),
) -> (Took, Vec<(u32, u64)>) {
    let mut taking = Taking::begin();
    let mut room = Vec::new();
    loop {
        let c = next.fetch_add(1, Relaxed);
        let Some(out) = done.get(c) else {
            break;
        };
        taking.chunk();
        let these = &slots[c * PASS_CHUNK..((c + 1) * PASS_CHUNK).min(slots.len())];
        let ahead: Vec<(u32, u64)> = these.iter().filter_map(|&i| Some((i, heap.handle(i)?))).collect();
        let mut lookups: Vec<Lookups> = ahead.iter().map(|_| if shared { Lookups::kept() } else { Lookups::each() }).collect();
        let views = if ahead.is_empty() { Vec::new() } else { host.mirror_shared(&ahead, &mut lookups, marks) };
        room.extend(crate::descriptor::host::take_room_wanted().into_iter().filter_map(|i| Some((i, heap.handle(i)?))));
        let mut made = views.into_iter().zip(lookups);
        let mut chunk = Vec::with_capacity(these.len());
        #[cfg(feature = "diagnostics")]
        let mut shown: Vec<(u32, std::sync::Arc<crate::shadow::Shadow>)> = Vec::new();
        for &index in these {
            if heap.handle(index).is_none() {
                parts::count(Count::Past);
                chunk.push(Slot { index, found: None, range: None });
                continue;
            }
            let Some((view, mut lookups)) = made.next() else {
                break;
            };
            let Some(view) = view else {
                let found = noting.then(|| Found { index, view: None, range: None });
                chunk.push(Slot { index, found, range: None });
                continue;
            };
            // (No lookup kept: the slot was made from its record already.)
            let made_already = shared && lookups.asked() == 0;
            let copy = {
                let _copy = enter(Part::Copy);
                buffer_copy_with(&view, &mut lookups)
            };
            let Some((shadow, address, len)) = copy else {
                parts::count(Count::NoRange);
                let found = noting.then(|| Found { index, view: Some(view), range: None });
                chunk.push(Slot { index, found, range: None });
                continue;
            };
            parts::count(Count::Range);
            let (shadow, address, len) = if shared {
                crate::draw::HEAP_PASS_FAST[0].fetch_add(1, Relaxed);
                crate::draw::HEAP_PASS_FAST[1].fetch_add(lookups.asked(), Relaxed);
                let _copy = enter(Part::Copy);
                crate::draw::shared_copy_checked(&view, (shadow, address, len))
            } else {
                (shadow, address, len)
            };
            if made_already {
                let again = super::same_range(kept.get(index), (&shadow, address, address + len));
                parts::count(if again { Count::Again } else { Count::Moved });
            }
            #[cfg(feature = "diagnostics")]
            {
                if super::shown() {
                    shown.push((index, shadow.clone()));
                }
            }
            let found = noting.then(|| Found { index, view: Some(view), range: Some((shadow.id(), address, address + len)) });
            chunk.push(Slot { index, found, range: Some((shadow, address, address + len)) });
        }
        // The diagnostics' look at the buffer each slot shows: once a
        // chunk, with the slot state held together (a look a slot took it
        // alone, which would have had every thread wait for it).
        #[cfg(feature = "diagnostics")]
        {
            if !shown.is_empty() {
                let _shows = enter(Part::Shows);
                let pairs: Vec<(u32, &windows::Win32::Graphics::Direct3D12::ID3D12Resource)> =
                    shown.iter().map(|(i, s)| (*i, &s.buffer)).collect();
                crate::draw::HEAP_VIEWS_ELSEWHERE.fetch_add(host.shows_elsewhere(&pairs), Relaxed);
            }
        }
        // Once a chunk while the device lives (a cheap query), as the pass
        // on one thread asks once every 64 views.
        if !GONE.load(Relaxed)
            && let Some(why) = crate::descriptor::host::device_removed()
            && !GONE.swap(true, Relaxed)
        {
            let last: Vec<String> = these
                .iter()
                .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                .map(|s| s.chars().take(200).collect())
                .collect();
            tracing::error!(why, "the host device went while buffer views of the heap were made; one of: {last:#?}");
            crate::host::log_removal_context();
        }
        *out.lock().unwrap_or_else(|e| e.into_inner()) = chunk;
    }
    (taking.end(), room)
}

/// The check's counts and the slots left without a pool slot (the second
/// the ones `HostHeap::make_room` made), since the last line.
static CHECKS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static ROOM: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// What [`check`] found over one pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Checked {
    /// Slots compared.
    pub(crate) slots: u64,
    /// Of them, the ones with the same view and another copy or range, or
    /// that the pass on one thread made otherwise than the threads had left
    /// them (another record, another buffer held).
    pub(crate) differed: u64,
    /// The ones whose view changed between the two passes (the title wrote
    /// the slot or released its view).
    pub(crate) changed: u64,
    /// The slots the pass on one thread made (not found made from their
    /// record already).
    pub(crate) made: u64,
}

/// `VICTORY_HEAP_PASS_THREADS_CHECK`: the pass over `slots` run again at
/// once on this thread, into ranges of its own, and compared slot by slot
/// with what the threads found (`threads`, in slot order) and left: each
/// slot's view, copy and range, the record its host slot was made from and
/// the host buffer it holds. Nothing should differ: a slot the threads left
/// as one thread makes it is found made, and is not made again.
pub(crate) fn check(heap: &Heap, host: &HostHeap, slots: &[u32], threads: &[Found], marks: bool) -> Checked {
    let began = Instant::now();
    let left = host.slots_seen(slots);
    let mut again = slots.to_vec();
    let mut scratch = Ranges::new();
    let mut one = Vec::with_capacity(threads.len());
    let took = super::buffers_taken(heap, host, &mut again, &mut scratch, Some(&mut one), marks);
    let after = host.slots_seen(slots);
    let mut checked = Checked {
        made: [Count::Hit, Count::Miss, Count::Other, Count::Unmade].iter().map(|&c| took.of(c)).sum(),
        ..Checked::default()
    };
    let say = |index: u32, what: String| {
        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if SAID.fetch_add(1, Relaxed) < 16 {
            tracing::warn!(slot = index, "VICTORY_HEAP_PASS_THREADS_CHECK: {what}");
        }
    };
    let mut changed: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut at: HashMap<u32, &Found> = one.iter().map(|f| (f.index, f)).collect();
    for now in threads {
        checked.slots += 1;
        match at.remove(&now.index) {
            Some(then) if then.view != now.view => {
                checked.changed += 1;
                changed.insert(now.index);
            }
            Some(then) if then.range == now.range => {}
            then => {
                checked.differed += 1;
                say(now.index, format!(
                    "the threads found another copy or range than one thread: threads {:?}, one thread {:?}",
                    (&now.view, now.range),
                    then.map(|f| (&f.view, f.range))
                ));
            }
        }
    }
    for (index, then) in at {
        checked.slots += 1;
        checked.differed += 1;
        say(index, format!("one thread found {:?}, the threads nothing", (&then.view, then.range)));
    }
    for ((&index, l), a) in slots.iter().zip(&left).zip(&after) {
        if l != a && !changed.contains(&index) {
            checked.differed += 1;
            say(index, format!(
                "one thread made the slot otherwise than the threads had left it: (record, buffer) {l:?} then {a:?}"
            ));
        }
    }
    for (k, n) in [checked.slots, checked.differed, checked.changed, checked.made].into_iter().enumerate() {
        CHECKS[k + 1].fetch_add(n, Relaxed);
    }
    CHECKS[0].fetch_add(1, Relaxed);
    CHECKS[5].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
    checked
}

/// With the frame line: the slots left without a pool slot, and the
/// check's counts, since the last line. Nothing while neither happened.
pub(crate) fn line() {
    let room = ROOM.each_ref().map(|c| c.swap(0, Relaxed));
    let k = CHECKS.each_ref().map(|c| c.swap(0, Relaxed));
    if room[0] > 0 {
        tracing::info!(
            "the pass over buffer views on several threads (VICTORY_HEAP_PASS_THREADS), since the last line: {} slots its \
             threads left without a pool slot (the heap map's pool used up), {} of them made after them with pool slots \
             given back",
            room[0],
            room[1]
        );
    }
    if k[0] > 0 {
        tracing::info!(
            "VICTORY_HEAP_PASS_THREADS_CHECK (one pass in {CHECK_EVERY} run again on one thread, since the last line): {} \
             passes, {} slots compared, {} differed (the same view, another copy or range, or made otherwise than the \
             threads left it), {} changed between the two (the title wrote the slot or released its view), {} made again \
             by the one thread; {:.1} ms a checked pass for the pass again and the comparison",
            k[0],
            k[1],
            k[2],
            k[3],
            k[4],
            k[5] as f64 / 1000.0 / k[0] as f64,
        );
    }
}
