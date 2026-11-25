//! Fences: a 64-bit value and the events waiting for it. Nothing runs on a
//! GPU yet, so only signals (the title's, later our queues') move one.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::object::{self, E_NOTIMPL, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) mod sent;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
    /// VIcky ([`sent`]): a value published ahead of its landing, and what
    /// landed meanwhile. Taken inside `waiters`' lock where both are.
    ahead: Mutex<sent::Ahead>,
    /// For [`pace`]: when each wait above was asked for.
    #[cfg(all(windows, feature = "diagnostics"))]
    asked: Mutex<Vec<pace::Asked>>,
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
        ahead: Mutex::new(sent::Ahead::default()),
        #[cfg(all(windows, feature = "diagnostics"))]
        asked: Mutex::new(Vec::new()),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
pub(crate) extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        #[cfg(all(windows, feature = "diagnostics"))]
        pace::ready();
        return S_OK;
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    pace::waits(this, &fence.asked, value, fence.value.load(Ordering::Acquire));
    if event.is_null() {
        // With submissions sent to the card and not waited for
        // (`crate::host::late`) the signal may be behind them: they are
        // settled, which sets it.
        #[cfg(windows)]
        if crate::host::late() {
            drop(waiters);
            crate::host::settle_for_fence();
            if fence.value.load(Ordering::Acquire) >= value {
                return S_OK;
            }
            tracing::warn!(value, "blocking fence wait not implemented");
            return E_NOTIMPL;
        }
        // shortcut: a null event means "block until reached"; nothing can
        // reach it while no GPU work runs, so fail instead of hanging.
        tracing::warn!(value, "blocking fence wait not implemented");
        return E_NOTIMPL;
    }
    waiters.push((value, event as usize));
    S_OK
}
/// Moves the fence to `value` and sets the events that waited for it.
///
/// VIcky ([`sent`]): on a fence whose value was published ahead of its
/// landing ([`set_early`]), a value below it is only noted (what our
/// queues' waits go by, [`landed`]) and the fence stays where it is; at or
/// above it the fence takes it as before. Without the switch nothing is
/// ever ahead, and this looks no further than [`AHEAD`].
pub(crate) fn set(this: This, value: u64) {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    // (Read under the lock `set_early` counts in.)
    if AHEAD.load(Ordering::Relaxed) > 0 {
        let mut ahead = fence.ahead.lock().unwrap_or_else(|e| e.into_inner());
        let was = ahead.is_ahead();
        let takes = ahead.land(value);
        if was && !ahead.is_ahead() {
            AHEAD.fetch_sub(1, Ordering::Relaxed);
        }
        if !takes {
            drop(ahead);
            #[cfg(all(windows, feature = "diagnostics"))]
            pace::landed_behind(this, value);
            return;
        }
    }
    fence.value.store(value, Ordering::Release);
    #[cfg(all(windows, feature = "diagnostics"))]
    pace::reached(this, &fence.asked, value);
    wake(&mut waiters, value);
}

/// Sets the events of `waiters` waiting for `value` or less, and lets them go.
fn wake(waiters: &mut Vec<(u64, usize)>, value: u64) {
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
}

/// Fences whose value is published ahead of its landing now ([`set_early`]):
/// counted up before the value is stored and down when a landing reaches
/// it, each under that fence's `waiters` lock.
static AHEAD: AtomicUsize = AtomicUsize::new(0);

/// Whether a fence's value is published ahead of its landing now
/// ([`set_early`]): for the pace meter.
#[cfg(feature = "diagnostics")]
pub(crate) fn ahead() -> bool {
    AHEAD.load(Ordering::Relaxed) > 0
}

/// VIcky ([`sent`]): publishes `value` to the title now (the fence's memory,
/// its events), ahead of the landing of the signal that carries it, which
/// still comes ([`set`]). Whether it went ahead; false if the fence is
/// there already.
pub(crate) fn set_early(this: This, value: u64) -> bool {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    {
        let mut ahead = fence.ahead.lock().unwrap_or_else(|e| e.into_inner());
        let was = ahead.is_ahead();
        if !ahead.publish(fence.value.load(Ordering::Acquire), value) {
            return false;
        }
        if !was {
            AHEAD.fetch_add(1, Ordering::Relaxed);
        }
    }
    fence.value.store(value, Ordering::Release);
    #[cfg(all(windows, feature = "diagnostics"))]
    pace::published(this, &fence.asked, value);
    wake(&mut waiters, value);
    true
}

/// The value the fence's signals have landed at: [`completed`], unless a
/// value was published ahead of its landing ([`set_early`]). Our queues'
/// waits for fences go by it, so that they pass when they did without
/// [`sent`].
pub(crate) fn landed(this: This) -> u64 {
    let fence = fence(this);
    let now = fence.value.load(Ordering::Acquire);
    // A value published is stored after the count went up: read here, it
    // finds the count up (or its landing done).
    if AHEAD.load(Ordering::Relaxed) == 0 {
        return now;
    }
    fence.ahead.lock().unwrap_or_else(|e| e.into_inner()).landed(now)
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    set(this, value);
    // A queue may have waited for it: what it held runs now.
    #[cfg(windows)]
    crate::queue::release_held(false);
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// How the title paces itself on its fences, measured (diagnostics
/// builds): which value it waits for, how long until our line gets the
/// fence there, and how many frames lie between the title's `Signal` on
/// its queue and the fence being set. Run 233 put presents behind the
/// line and the frame did not get shorter: the title waited somewhere
/// else, and this says for what.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) mod pace {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use std::time::Instant;

    use crate::object::This;

    pub(crate) struct Asked {
        value: u64,
        at: Instant,
        frame: u64,
        thread: u32,
    }

    /// The title's `Signal` calls on a queue not yet at the fence:
    /// fence, value, when, frame.
    static QUEUED: Mutex<Vec<(usize, u64, Instant, u64)>> = Mutex::new(Vec::new());
    /// The thread that presents.
    static RENDER: AtomicU32 = AtomicU32::new(0);
    /// Since the last line: waits already met; waits not met; of those,
    /// the render thread's; reached waits, their microseconds, their
    /// frames between; the render thread's reached waits and microseconds;
    /// signals set, their microseconds since the title's call, their
    /// frames between.
    static N: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
    static SAID: AtomicU64 = AtomicU64::new(0);
    static LINE_AT: AtomicU64 = AtomicU64::new(0);

    /// VIcky (`super::sent`), since the last line: frame-begin signals
    /// taken; first signals after a present not taken; presents whose frame
    /// had none taken; values published when sent; their microseconds since
    /// the title's call, and how many of them had the call at hand; flights
    /// and bottom-of-pipe writes still ahead of them in the line; frame-begin
    /// signals set at once (nothing in the air); those that landed before
    /// they could be published; landings of published values and their
    /// microseconds since the publishing; landings below a published value
    /// (not stored); bytes copied back into title memory and bottom-of-pipe
    /// writes done while a published value was ahead of its landing; and
    /// with `VICTORY_FENCE_SENT_CHECK=1` the title's bytes the copy-back put
    /// the card's over meanwhile.
    static SENT: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
    const S_TAKEN: usize = 0;
    const S_FIRST: usize = 1;
    const S_NONE: usize = 2;
    const S_PUBLISHED: usize = 3;
    const S_CALL_US: usize = 4;
    const S_CALLS: usize = 5;
    const S_FLIGHTS: usize = 6;
    const S_WRITES: usize = 7;
    const S_AT_ONCE: usize = 8;
    const S_LANDED_FIRST: usize = 9;
    const S_LANDINGS: usize = 10;
    const S_AHEAD_US: usize = 11;
    const S_BEHIND: usize = 12;
    const S_BACK_BYTES: usize = 13;
    const S_BACK_WRITES: usize = 14;
    const S_LOST: usize = 15;

    /// Values published when sent and not landed yet: fence, value, when.
    static EARLY: Mutex<Vec<(usize, u64, Instant)>> = Mutex::new(Vec::new());

    /// Whether events are said one by one now: four seconds of every two
    /// minutes of the wall clock, the same seconds in which the loader says
    /// the title's `mwaitx` waits (`console_insn`), to lay the two side by side.
    fn detail_now() -> bool {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).is_ok_and(|d| d.as_secs() % 120 < 4)
    }

    fn thread() -> u32 {
        // SAFETY: a plain query.
        unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
    }

    pub(super) fn ready() {
        N[0].fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn waits(fence: This, asked: &Mutex<Vec<Asked>>, value: u64, now: u64) {
        let thread = thread();
        let render = thread == RENDER.load(Ordering::Relaxed);
        N[1].fetch_add(1, Ordering::Relaxed);
        if render {
            N[2].fetch_add(1, Ordering::Relaxed);
        }
        let frame = crate::queue::presented();
        if detail_now() || SAID.fetch_add(1, Ordering::Relaxed) < 48 {
            tracing::info!(
                "pace: the title waits for fence {fence:?} value {value} (it is at {now}) in frame {frame}, on {}",
                if render { "the render thread".to_string() } else { format!("thread {thread}") }
            );
        }
        let mut asked = asked.lock().unwrap_or_else(|e| e.into_inner());
        if asked.len() < 4096 {
            asked.push(Asked { value, at: Instant::now(), frame, thread });
        }
    }

    /// The title called `Signal` for `fence` on a queue.
    pub(crate) fn signal_called(fence: usize, value: u64) {
        let mut queued = QUEUED.lock().unwrap_or_else(|e| e.into_inner());
        if queued.len() < 4096 {
            queued.push((fence, value, Instant::now(), crate::queue::presented()));
        }
    }

    /// The fence was set to `value`: the title's signals up to it have
    /// landed, and its waits up to it are met.
    pub(super) fn reached(fence: This, asked: &Mutex<Vec<Asked>>, value: u64) {
        landed(fence, value);
        met(fence, asked, value);
    }

    /// The title's signals on `fence` up to `value` have landed (the signal
    /// times of the line), and the values published ahead up to it.
    fn landed(fence: This, value: u64) {
        let frame = crate::queue::presented();
        {
            let mut early = EARLY.lock().unwrap_or_else(|e| e.into_inner());
            early.retain(|&(f, v, at)| {
                let done = f == fence as usize && v <= value;
                if done {
                    SENT[S_LANDINGS].fetch_add(1, Ordering::Relaxed);
                    SENT[S_AHEAD_US].fetch_add(at.elapsed().as_micros() as u64, Ordering::Relaxed);
                    if detail_now() || SAID.fetch_add(1, Ordering::Relaxed) < 48 {
                        tracing::info!(
                            "pace: fence {fence:?} lands at {v}, {:.1} ms after it was published when sent (VICTORY_FENCE_SENT=1; frame {frame})",
                            at.elapsed().as_micros() as f64 / 1e3
                        );
                    }
                }
                !done
            });
        }
        {
            let mut queued = QUEUED.lock().unwrap_or_else(|e| e.into_inner());
            queued.retain(|&(f, v, at, called)| {
                let done = f == fence as usize && v <= value;
                if done {
                    N[8].fetch_add(1, Ordering::Relaxed);
                    N[9].fetch_add(at.elapsed().as_micros() as u64, Ordering::Relaxed);
                    N[10].fetch_add(frame.saturating_sub(called), Ordering::Relaxed);
                    if detail_now() || SAID.fetch_add(1, Ordering::Relaxed) < 48 {
                        tracing::info!(
                            "pace: fence {fence:?} is set to {v}, {:.1} ms after the title's Signal for it (called in frame {called}, set in frame {frame})",
                            at.elapsed().as_micros() as f64 / 1e3
                        );
                    }
                }
                !done
            });
        }
    }

    /// The fence reads `value` now: the title's waits up to it are met.
    fn met(fence: This, asked: &Mutex<Vec<Asked>>, value: u64) {
        let frame = crate::queue::presented();
        let render = RENDER.load(Ordering::Relaxed);
        let mut asked = asked.lock().unwrap_or_else(|e| e.into_inner());
        asked.retain(|a| {
            let done = a.value <= value;
            if done {
                let us = a.at.elapsed().as_micros() as u64;
                N[3].fetch_add(1, Ordering::Relaxed);
                N[4].fetch_add(us, Ordering::Relaxed);
                N[5].fetch_add(frame.saturating_sub(a.frame), Ordering::Relaxed);
                if a.thread == render {
                    N[6].fetch_add(1, Ordering::Relaxed);
                    N[7].fetch_add(us, Ordering::Relaxed);
                }
                if detail_now() || SAID.fetch_add(1, Ordering::Relaxed) < 48 {
                    tracing::info!(
                        "pace: fence {fence:?} reached {} {:.1} ms after the wait for it (asked in frame {}, reached in frame {frame})",
                        a.value,
                        us as f64 / 1e3,
                        a.frame
                    );
                }
            }
            !done
        });
    }

    /// VIcky (`super::sent`): what the title's signal was to the
    /// recogniser (`super::sent::Begins`); `None`: a present whose frame
    /// had no frame-begin signal taken.
    pub(crate) fn begin(begin: Option<super::sent::Begin>) {
        let i = match begin {
            Some(super::sent::Begin::Yes) => S_TAKEN,
            Some(super::sent::Begin::First) => S_FIRST,
            Some(super::sent::Begin::No) => return,
            None => S_NONE,
        };
        SENT[i].fetch_add(1, Ordering::Relaxed);
    }

    /// `value` was published ahead of its landing (`super::set_early`): the
    /// title's waits up to it are met now, and its signal's landing is
    /// still to come (the signal times of the line stay those of landings).
    pub(super) fn published(fence: This, asked: &Mutex<Vec<Asked>>, value: u64) {
        let frame = crate::queue::presented();
        SENT[S_PUBLISHED].fetch_add(1, Ordering::Relaxed);
        let called = QUEUED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|&&(f, v, ..)| f == fence as usize && v == value)
            .map(|q| q.2);
        if let Some(at) = called {
            SENT[S_CALLS].fetch_add(1, Ordering::Relaxed);
            SENT[S_CALL_US].fetch_add(at.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        if detail_now() || SAID.fetch_add(1, Ordering::Relaxed) < 48 {
            tracing::info!(
                "pace: fence {fence:?} is published at {value} when sent (the frame-begin signal, VICTORY_FENCE_SENT=1), {} after the title's Signal for it (frame {frame})",
                called.map_or("?".to_string(), |at| format!("{:.1} ms", at.elapsed().as_micros() as f64 / 1e3))
            );
        }
        {
            let mut early = EARLY.lock().unwrap_or_else(|e| e.into_inner());
            if early.len() < 4096 {
                early.push((fence as usize, value, Instant::now()));
            }
        }
        met(fence, asked, value);
    }

    /// A frame-begin signal went behind the flights in the air
    /// (`crate::host::sent_signal`): published ahead (`early`), with
    /// `flights` and `writes` ahead of it in the line, or landed before it
    /// could be.
    pub(crate) fn sent(early: bool, flights: usize, writes: usize) {
        if early {
            SENT[S_FLIGHTS].fetch_add(flights as u64, Ordering::Relaxed);
            SENT[S_WRITES].fetch_add(writes as u64, Ordering::Relaxed);
        } else {
            SENT[S_LANDED_FIRST].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A frame-begin signal was set at once: nothing was in the air.
    pub(crate) fn sent_at_once() {
        SENT[S_AT_ONCE].fetch_add(1, Ordering::Relaxed);
    }

    /// A landing of `value` below the value published on `fence`: noted,
    /// not stored (`super::set`).
    pub(super) fn landed_behind(fence: This, value: u64) {
        SENT[S_BEHIND].fetch_add(1, Ordering::Relaxed);
        landed(fence, value);
    }

    /// While a published value is ahead of its landing (`super::ahead`):
    /// `bytes` copied back into title memory, `writes` bottom-of-pipe
    /// writes done. What the title could have read stale had it read them
    /// in between.
    pub(crate) fn landed_ahead(bytes: u64, writes: u64) {
        SENT[S_BACK_BYTES].fetch_add(bytes, Ordering::Relaxed);
        SENT[S_BACK_WRITES].fetch_add(writes, Ordering::Relaxed);
    }

    /// With `VICTORY_FENCE_SENT_CHECK=1`, while a published value is ahead:
    /// `bytes` the title had written that a relay merge put the card's over
    /// (`crate::shadow`'s `lost_under_published`).
    pub(crate) fn lost_ahead(bytes: u64) {
        SENT[S_LOST].fetch_add(bytes, Ordering::Relaxed);
    }

    /// Once a frame, on the thread that presents: a line every 256 frames.
    pub(crate) fn frame() {
        RENDER.store(thread(), Ordering::Relaxed);
        let frame = crate::queue::presented();
        let last = LINE_AT.load(Ordering::Relaxed);
        if frame < last + 256 {
            return;
        }
        LINE_AT.store(frame, Ordering::Relaxed);
        let n: Vec<f64> = N.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64).collect();
        let s: Vec<f64> = SENT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64).collect();
        let frames = (frame - last).max(1) as f64;
        let per = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
        // VIcky (`super::sent`): only with the switch, so that the line is
        // as before without it.
        let sent = if super::sent::on() {
            format!(
                "; VICTORY_FENCE_SENT=1 (the signal times above are of landings): frame-begin signals {:.2} a frame (first signals after a present not taken {:.2}, \
                 frames without one {:.2}), published when sent {:.2} a frame, {:.1} ms after the title's Signal call and {:.1} ms ahead of their landing, with {:.1} flights \
                 and {:.1} bottom-of-pipe writes still ahead of them; set at once with nothing in the air {:.2}, landed before they could be published {:.2}; \
                 landings below a published value {:.2}; while a published value was ahead: {:.1} KB copied back into title memory and {:.1} bottom-of-pipe writes done{}",
                s[S_TAKEN] / frames,
                s[S_FIRST] / frames,
                s[S_NONE] / frames,
                s[S_PUBLISHED] / frames,
                per(s[S_CALL_US], s[S_CALLS]) / 1e3,
                per(s[S_AHEAD_US], s[S_LANDINGS]) / 1e3,
                per(s[S_FLIGHTS], s[S_PUBLISHED]),
                per(s[S_WRITES], s[S_PUBLISHED]),
                s[S_AT_ONCE] / frames,
                s[S_LANDED_FIRST] / frames,
                s[S_BEHIND] / frames,
                s[S_BACK_BYTES] / frames / 1024.0,
                s[S_BACK_WRITES] / frames,
                if super::sent::check() {
                    format!(", title bytes the copy-back put the card's over {:.1} (VICTORY_FENCE_SENT_CHECK=1)", s[S_LOST] / frames)
                } else {
                    String::new()
                },
            )
        } else {
            String::new()
        };
        tracing::info!(
            "pace, per frame: fence waits already met {:.2}, not met {:.2} (the render thread's {:.2}); a wait is met {:.1} ms and {:.2} frames after it was asked \
             (the render thread's: {:.1} ms, {:.1} ms a frame); the title's queue signals: {:.2} a frame, a fence is set {:.1} ms and {:.2} frames after the title's Signal call{sent}",
            n[0] / frames,
            n[1] / frames,
            n[2] / frames,
            per(n[4], n[3]) / 1e3,
            per(n[5], n[3]),
            per(n[7], n[6]) / 1e3,
            n[7] / frames / 1e3,
            n[8] / frames,
            per(n[9], n[8]) / 1e3,
            per(n[10], n[8]),
        );
    }
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 9, std::ptr::null_mut()), E_NOTIMPL);
        // The title reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    /// VIcky ([`sent`]): a value published ahead is what the title reads and
    /// waits on; the landings below it never move the fence back, and our
    /// queues' waits ([`landed`]) see only what has landed.
    #[cfg(windows)]
    #[test]
    fn a_value_published_ahead_never_goes_back() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 20, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let completed = slot::<extern "system" fn(This) -> u64>(f, 0x40);
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        // SAFETY: plain auto-reset events.
        let (at_27, at_28) = unsafe {
            (CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()), CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()))
        };
        assert_eq!(wait(f, 27, at_27), S_OK);
        assert_eq!(wait(f, 28, at_28), S_OK);
        let is_set = |event| {
            // SAFETY: a valid event.
            unsafe { WaitForSingleObject(event, 0) == WAIT_OBJECT_0 }
        };
        // The frame-begin value 27 is published while 25 and 26 (the frame
        // before's) are still behind flights.
        assert!(set_early(f, 27));
        assert_eq!(completed(f), 27, "the title reads the published value");
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 27, "in memory too");
        assert!(is_set(at_27), "its event is set at once");
        assert_eq!(landed(f), 20, "our queues' waits still see what landed");
        // The landings below it: noted, the fence stays at 27.
        set(f, 25);
        assert_eq!(completed(f), 27);
        assert_eq!(landed(f), 25);
        set(f, 26);
        assert_eq!(completed(f), 27);
        assert_eq!(landed(f), 26);
        assert!(!is_set(at_28));
        // Publishing what the fence holds already does nothing.
        assert!(!set_early(f, 27));
        // Its own landing ends it; the signals after it are as before.
        set(f, 27);
        assert_eq!((completed(f), landed(f)), (27, 27));
        set(f, 28);
        assert_eq!((completed(f), landed(f)), (28, 28));
        assert!(is_set(at_28));
        // SAFETY: ours.
        unsafe {
            CloseHandle(at_27);
            CloseHandle(at_28);
        }
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }
}
