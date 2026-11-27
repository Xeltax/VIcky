//! P6 (`VICTORY_LOT4_PROBE=1`; `VICTORY_LOT4_PROBE_FAULTS=0` leaves it out):
//! lazy write-back, sampled. Would the title read back what shaders wrote
//! before the card writes it again? If it seldom does, title memory could
//! be left stale and fetched on a fault, instead of written at every
//! landing.
//!
//! One frame in [`EVERY`] (half way between P8's, `super::device`): each
//! 64 KiB block a landing wrote into title memory in that frame (`crate::shadow`'s write-back, the texture
//! copy-back) is made no-access right after the write, and stays so through
//! the next frame. A fault on it opens it again (its protection given
//! back), and is counted: a read or a write, by code of the title (or a
//! system library on its behalf) or of the runtime (its image: the
//! functions it implements for the title count there too). At the end the
//! blocks
//! nobody touched are opened. The runtime's own reads and writes through
//! the kernel (`crate::host::copy_mapped`, `read_into`, `write_mapped`, and
//! the kernel writes `crate::watch::writing` hears of) open the blocks of
//! their range first, which a kernel call needs (it fails on a no-access
//! page where code faults); those are counted apart.
//!
//! Only plain private memory is sampled: console addresses, committed
//! read-write private pages (`VirtualQuery`) that are not views of
//! physical pages (whose other views and guards the protections would
//! cross). The cost a lazy fetch would add is estimated from the handler's
//! own time (finding the block and giving its protection back), plus a
//! 64 KiB copy each.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

use super::BLOCK;

/// One frame in this many is sampled.
const EVERY: u64 = 64;

/// 0: no blocks held; 1: the sampled frame (blocks written are protected);
/// 2: the frame after (held blocks stay, no new ones).
static STATE: AtomicU8 = AtomicU8::new(0);

/// A protected block: its address, the protection it had, whether it is
/// open again.
struct Held {
    at: usize,
    old: u32,
    open: bool,
}

/// The held blocks of the last sample (made at the first sampled frame,
/// with room enough that nothing allocates under the lock), sorted by
/// address once the sample ends: a thread that faulted on a block just
/// before another opened it, or just before the sample ended, finds it
/// there and goes on. A leaf lock: nothing is locked, and no title memory
/// touched, while it is held.
static HELD: Mutex<Vec<Held>> = Mutex::new(Vec::new());
const ROOM: usize = 1 << 16;

/// Counts since the last line.
#[derive(Clone, Copy)]
enum N {
    Frames,
    Protected,
    /// Blocks not sampled: views, host addresses, not plain read-write
    /// private memory, or the protection refused.
    Skipped,
    TitleReads,
    TitleWrites,
    OwnReads,
    OwnWrites,
    /// Blocks the runtime opened for its own kernel reads and writes.
    Opened,
    Untouched,
    HandlerNs,
    ProtectNs,
    Full,
    Count,
}

static COUNTS: [AtomicU64; N::Count as usize] = [const { AtomicU64::new(0) }; N::Count as usize];

fn count(n: N, by: u64) {
    COUNTS[n as usize].fetch_add(by, Relaxed);
}

/// Whether P6 runs (with the probes).
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    super::on() && *ON.get_or_init(|| std::env::var_os("VICTORY_LOT4_PROBE_FAULTS").is_none_or(|v| v != "0"))
}

/// The 64-frame stretch (of `crate::frame::current`) last sampled.
static SAMPLED: AtomicU64 = AtomicU64::new(u64::MAX);

/// A frame ended: the sampling's steps. A sample starts half way through a
/// stretch of [`EVERY`] frames of `crate::frame::current`, once a stretch,
/// and not while P8 (`super::device`, which records frames 64k + 1 of the
/// same counter and then pulls a region opened in place) records: its CPU
/// copies would be timed with P6's protections in them.
pub(crate) fn frame(_frame: u64) {
    if !enabled() {
        return;
    }
    let current = crate::frame::current();
    let stretch = current / EVERY;
    let due = current % EVERY >= EVERY / 2 && SAMPLED.load(Relaxed) != stretch && !super::device::recording();
    match STATE.load(Relaxed) {
        0 if due => {
            SAMPLED.store(stretch, Relaxed);
            install();
            {
                let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
                held.clear();
                if held.capacity() < ROOM {
                    held.reserve(ROOM);
                }
            }
            STATE.store(1, Relaxed);
        }
        1 => STATE.store(2, Relaxed),
        2 => {
            let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
            let mut untouched = 0;
            for h in held.iter_mut().filter(|h| !h.open) {
                give_back(h);
                untouched += 1;
            }
            // Kept, open, for the handler's binary search until the next
            // sample.
            held.sort_unstable_by_key(|h| h.at);
            STATE.store(0, Relaxed);
            count(N::Untouched, untouched);
            count(N::Frames, 1);
        }
        _ => {}
    }
}

/// Gives a held block its protection back.
fn give_back(h: &mut Held) {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    let mut was = 0;
    // SAFETY: a block of title memory we made no-access ourselves.
    unsafe { VirtualProtect(h.at as *const _, BLOCK, h.old, &mut was) };
    h.open = true;
}

/// Whether the block at `at` is plain committed read-write private memory.
fn plain_private(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery};
    // SAFETY: plain data the query fills in.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
        return false;
    }
    m.State == MEM_COMMIT
        && m.Type == MEM_PRIVATE
        && m.Protect == PAGE_READWRITE
        && m.BaseAddress as usize <= at
        && m.BaseAddress as usize + m.RegionSize >= at + BLOCK
}

/// A landing wrote `[at, at + len)` of title memory: in the sampled frame,
/// its blocks are protected.
pub(crate) fn written(at: usize, len: usize) {
    if STATE.load(Relaxed) != 1 || len == 0 || !enabled() {
        return;
    }
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, VirtualProtect};
    let started = std::time::Instant::now();
    // The blocks that may be held: looked at before the lock is taken (the
    // view table has a lock of its own).
    let (first, last) = (at / BLOCK, (at + len - 1) / BLOCK);
    let plain: Vec<usize> = (first..=last)
        .map(|block| block * BLOCK)
        .filter(|&b| {
            let ok = (b as u64) < super::HOST_ADDRESSES
                && crate::inplace::cover(b, BLOCK) == crate::inplace::Cover::None
                && plain_private(b);
            if !ok {
                count(N::Skipped, 1);
            }
            ok
        })
        .collect();
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    // (Looked at again under the lock: the end of the hold takes it too.)
    if STATE.load(Relaxed) != 1 {
        return;
    }
    for b in plain {
        if let Some(h) = held.iter_mut().find(|h| h.at == b) {
            // Written again after it was opened: held again.
            if h.open {
                let mut old = 0;
                // SAFETY: as below.
                if unsafe { VirtualProtect(b as *const _, BLOCK, PAGE_NOACCESS, &mut old) } != 0 {
                    h.open = false;
                }
            }
            continue;
        }
        if held.len() >= ROOM {
            count(N::Full, 1);
            break;
        }
        let mut old = 0;
        // SAFETY: committed read-write private title memory (checked
        // above); every reader of ours opens it first, any other access
        // faults into `on_fault`, which opens it.
        if unsafe { VirtualProtect(b as *const _, BLOCK, PAGE_NOACCESS, &mut old) } == 0 {
            count(N::Skipped, 1);
            continue;
        }
        held.push(Held { at: b, old, open: false });
        count(N::Protected, 1);
    }
    count(N::ProtectNs, started.elapsed().as_nanos() as u64);
}

/// The runtime reads or writes `[at, at + len)` of title memory through the
/// kernel (or its own copies): held blocks there are opened first.
#[inline]
pub(crate) fn release(at: usize, len: usize) {
    if STATE.load(Relaxed) == 0 || len == 0 {
        return;
    }
    release_now(at, len);
}

#[cold]
fn release_now(at: usize, len: usize) {
    let end = at.saturating_add(len);
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    let mut opened = 0;
    for h in held.iter_mut().filter(|h| !h.open && h.at < end && h.at + BLOCK > at) {
        give_back(h);
        opened += 1;
    }
    count(N::Opened, opened);
}

/// The runtime's own code: its image's range.
fn own_code() -> (usize, usize) {
    static RANGE: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *RANGE.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
        // SAFETY: this process's own image, mapped for its life; its
        // headers say its size.
        unsafe {
            let base = GetModuleHandleW(std::ptr::null()) as usize;
            if base == 0 {
                return (0, 0);
            }
            let nt = base + (*((base + 0x3c) as *const u32)) as usize;
            // The optional header's SizeOfImage (PE32+), after the
            // signature (4) and the file header (20).
            let size = *((nt + 24 + 56) as *const u32) as usize;
            (base, base + size)
        }
    })
}

/// Whether `address` can now be read (or written).
fn accessible(address: usize, write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READONLY, PAGE_READWRITE,
        VirtualQuery,
    };
    // SAFETY: plain data the query fills in.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(address as *const _, &mut m, size_of_val(&m)) } == 0 || m.State != MEM_COMMIT {
        return false;
    }
    let p = m.Protect & 0xff;
    if write { p == PAGE_READWRITE || p == PAGE_EXECUTE_READWRITE } else { matches!(p, PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE) }
}

/// The handler, first in line, once.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched and its
        // context.
        let (r, context) = unsafe { (&*(*info).ExceptionRecord, &*(*info).ContextRecord) };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let started = std::time::Instant::now();
        let address = r.ExceptionInformation[1];
        let at = address / BLOCK * BLOCK;
        let write = r.ExceptionInformation[0] == 1;
        // (Nothing allocates here: the list has its room, and our code never
        // touches title memory while it holds the lock.)
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        let found = if STATE.load(Relaxed) == 0 {
            held.binary_search_by_key(&at, |h| h.at).ok()
        } else {
            held.iter().position(|h| h.at == at)
        };
        let Some(i) = found else {
            return EXCEPTION_CONTINUE_SEARCH;
        };
        let h = &mut held[i];
        if h.open {
            // Opened by another thread (or the end of the sample) after this
            // one faulted: go on if the access now can.
            drop(held);
            return if accessible(address, write) { EXCEPTION_CONTINUE_EXECUTION } else { EXCEPTION_CONTINUE_SEARCH };
        }
        give_back(h);
        drop(held);
        let (lo, hi) = own_code();
        let own = (lo..hi).contains(&(context.Rip as usize));
        count(
            match (own, write) {
                (false, false) => N::TitleReads,
                (false, true) => N::TitleWrites,
                (true, false) => N::OwnReads,
                (true, true) => N::OwnWrites,
            },
            1,
        );
        count(N::HandlerNs, started.elapsed().as_nanos() as u64);
        EXCEPTION_CONTINUE_EXECUTION
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = own_code();
        // SAFETY: a handler that only looks at its own held blocks.
        unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
    });
}

/// The reading's line (per sampled frame).
pub(crate) fn line(_frames: f64) {
    if !enabled() {
        return;
    }
    let c: [u64; N::Count as usize] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed));
    let sampled = c[N::Frames as usize];
    let per = |n: N| c[n as usize] as f64 / sampled.max(1) as f64;
    let faults = c[N::TitleReads as usize] + c[N::TitleWrites as usize] + c[N::OwnReads as usize] + c[N::OwnWrites as usize];
    let handler_us = if faults > 0 { c[N::HandlerNs as usize] as f64 / 1e3 / faults as f64 } else { 0.0 };
    tracing::info!(
        "lot4 probe P6: lazy write-back sampled in {sampled} frames (one in {EVERY}; the blocks the landings wrote kept \
         no-access through the next frame), per sampled frame: {:.0} blocks held ({:.0} skipped: views, host memory, not \
         plain read-write private memory), title code faulted on {:.1} of them reading and {:.1} writing, the runtime's \
         code on {:.1} reading and {:.1} writing, the runtime's kernel reads and writes opened {:.1}, {:.0} untouched; \
         handler {handler_us:.1} us a fault (about {:.2} ms a sampled frame with a 64 KiB copy each at 10 GB/s); \
         protecting took {:.2} ms a sampled frame{}",
        per(N::Protected),
        per(N::Skipped),
        per(N::TitleReads),
        per(N::TitleWrites),
        per(N::OwnReads),
        per(N::OwnWrites),
        per(N::Opened),
        per(N::Untouched),
        faults as f64 * (handler_us + 6.5) / 1e3 / sampled.max(1) as f64,
        per(N::ProtectNs) / 1e6,
        if c[N::Full as usize] > 0 { format!(" ({} blocks over the list's room)", c[N::Full as usize]) } else { String::new() },
    );
}
