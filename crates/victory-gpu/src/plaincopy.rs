//! Copies of title memory made in place, by this process, with a way back
//! when a page is not there (`VICTORY_COPY_PLAIN=1`; off by default).
//!
//! Every copy of title memory into a host buffer goes through the kernel
//! (`ReadProcessMemory` on our own process, `crate::host::copy_mapped`):
//! the title unmaps and remaps its memory while we read it, a read in
//! place of a page that is gone is an access violation, and the kernel's
//! copy fails instead. It costs what a system call that locks and probes
//! every page costs: run 225's profile has the submitting thread 35 ms a
//! frame inside `ZwReadVirtualMemory` for about 100 MB (3 GB a second),
//! where the processor's own copy moves that much in a third of the time.
//!
//! Here the copy is one `rep movsb` in a function of ours that uses no
//! stack. An exception handler, first in the line, looks at every access
//! violation: one that is a READ and happened at exactly that instruction
//! continues at the function's other exit, which returns 0, and the
//! caller falls back to the kernel's copy as before (it says which page
//! is gone and fills in zeros). A write that faults is not ours to take:
//! the destination is host memory, which is always there; such a fault
//! goes on to the handlers after this one.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".globl victory_copy_guarded",
    ".globl victory_copy_guarded_at",
    ".globl victory_copy_guarded_fault",
    ".globl victory_write_guarded",
    ".globl victory_write_guarded_at",
    ".globl victory_write_guarded_fault",
    "victory_copy_guarded:",
    // rcx: to, rdx: from, r8: bytes. rsi and rdi are the caller's: kept in
    // r9 and r10, which are ours to use, so nothing is on the stack.
    "    mov r9, rsi",
    "    mov r10, rdi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_copy_guarded_at:",
    "    rep movsb",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    mov eax, 1",
    "    ret",
    "victory_copy_guarded_fault:",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    xor eax, eax",
    "    ret",
    "victory_write_guarded:",
    "    mov r9, rsi",
    "    mov r10, rdi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_write_guarded_at:",
    "    rep movsb",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    mov eax, 1",
    "    ret",
    "victory_write_guarded_fault:",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    xor eax, eax",
    "    ret",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn victory_copy_guarded(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_copy_guarded_at: u8;
    static victory_copy_guarded_fault: u8;
    fn victory_write_guarded(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_write_guarded_at: u8;
    static victory_write_guarded_fault: u8;
}

/// Whether copies of title memory into host memory are made in place
/// (`VICTORY_COPY_PLAIN=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = cfg!(target_arch = "x86_64") && std::env::var_os("VICTORY_COPY_PLAIN").is_some_and(|v| v == "1");
        if on {
            install();
            tracing::info!(
                "copies of title memory into host buffers are made in place, with the kernel's copy behind them for a \
                 page that is not there (VICTORY_COPY_PLAIN=1)"
            );
        }
        on
    })
}

/// Copies made in place, their bytes, the copies that met a page that was
/// not there (and went to the kernel's copy), and the copies not tried in
/// place because their source faulted lately ([`RECENT`]) or they were
/// under [`LEAST`] bytes; the fills of under [`LEAST`] bytes made in place
/// ([`copy_short`]); and the writes made in place ([`write`]), their bytes
/// and those that fell back. Summed at the line ([`line`]).
///
/// VIcky (profile PF5b, D1): kept per thread ([`crate::perthread`]). They
/// were nine adjacent shared atomics, added to twice per copy by the
/// submitting thread, the relay, the pass over buffer views and the wide
/// copy threads at once, each add a locked add on a line the others had
/// just taken: 2.0 ms a frame of the submitting thread alone, in every
/// build. The sums are the same, add for add.
static COUNT: crate::perthread::Counters<9> = crate::perthread::Counters::new();
const COPIES: usize = 0;
const COPIED_BYTES: usize = 1;
const GONE: usize = 2;
const LATELY: usize = 3;
const SMALL: usize = 4;
const FILLS: usize = 5;
const WRITES: usize = 6;
const WRITTEN_BYTES: usize = 7;
const WRITES_GONE: usize = 8;

/// The fewest bytes copied in place. A read of a byte or a word is how the
/// layer ASKS whether a page is there (a texture's first page, a record):
/// it is expected to fail, and a failed kernel read costs a microsecond
/// where a fault and its handler cost tens (run 230: 1,000 to 1,300 faults
/// a frame, nearly all of them such askings, took back most of what the
/// copies in place had won).
const LEAST: usize = 256;

/// The 64 KiB blocks of title memory a copy in place faulted in lately:
/// their copies go straight to the kernel's, which reads round the hole.
/// By the block's number, in the place its low bits name (a later block
/// with the same low bits takes the place). Forgotten every two seconds:
/// the title maps the memory in the end.
static RECENT: [std::sync::atomic::AtomicUsize; 1024] = [const { std::sync::atomic::AtomicUsize::new(0) }; 1024];
/// When [`RECENT`] was last emptied (milliseconds of [`now_ms`]).
static EMPTIED: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// The numbers of the first and last 64 KiB block of `[from, from + len)`,
/// never 0 (0 is an empty place).
fn blocks(from: usize, len: usize) -> (usize, usize) {
    ((from >> 16) | 1 << 63, ((from + len.max(1) - 1) >> 16) | 1 << 63)
}

/// Whether a copy from there faulted lately.
fn faulted_lately(from: usize, len: usize) -> bool {
    let now = now_ms();
    let emptied = EMPTIED.load(Relaxed);
    if now.saturating_sub(emptied) >= 2_000 && EMPTIED.compare_exchange(emptied, now, Relaxed, Relaxed).is_ok() {
        for place in &RECENT {
            place.store(0, Relaxed);
        }
        return false;
    }
    let (first, last) = blocks(from, len);
    RECENT[first & 1023].load(Relaxed) == first || RECENT[last & 1023].load(Relaxed) == last
}

/// A copy from there faulted.
fn note_fault(from: usize, len: usize) {
    let (first, last) = blocks(from, len);
    RECENT[first & 1023].store(first, Relaxed);
    RECENT[last & 1023].store(last, Relaxed);
}

/// The handler that turns a read fault inside [`guarded`] into its return
/// of false; put first in the line, once.
#[cfg(target_arch = "x86_64")]
fn install() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE, EXCEPTION_IN_PAGE_ERROR,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched and the
        // context it will continue with.
        let (r, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        // Only at our own copy/write instructions.
        if context.Rip == (&raw const victory_copy_guarded_at) as u64 {
            const READ: usize = 0;
            let read = r.NumberParameters >= 1 && r.ExceptionInformation[0] == READ;
            let gone = (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) && read;
            if gone || r.ExceptionCode == EXCEPTION_GUARD_PAGE {
                context.Rip = (&raw const victory_copy_guarded_fault) as u64;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if context.Rip == (&raw const victory_write_guarded_at) as u64 {
            const WRITE: usize = 1;
            let write = r.NumberParameters >= 1 && r.ExceptionInformation[0] == WRITE;
            let gone = (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) && write;
            if gone || r.ExceptionCode == EXCEPTION_GUARD_PAGE {
                context.Rip = (&raw const victory_write_guarded_fault) as u64;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: a handler that reads the record it is given and nothing else.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    });
}

#[cfg(not(target_arch = "x86_64"))]
fn install() {}

/// Copies `len` bytes from `from` to `to` in place; false if a page of the
/// source was not there (then some of `to` may have been written).
///
/// `to` must be writable memory of `len` bytes that is ours (host memory);
/// `from` may be anything.
#[cfg(target_arch = "x86_64")]
fn guarded(to: *mut u8, from: *const u8, len: usize) -> bool {
    install();
    // SAFETY: the destination is the caller's to write; a source page that
    // is not there ends the copy with 0 instead of a fault ([`install`]).
    unsafe { victory_copy_guarded(to, from, len) != 0 }
}

#[cfg(not(target_arch = "x86_64"))]
fn guarded(_: *mut u8, _: *const u8, _: usize) -> bool {
    false
}

/// Copies `len` bytes from host memory `from` into title memory `to` in place;
/// false if a destination page is not mapped or not writable.
#[cfg(target_arch = "x86_64")]
fn guarded_write(to: *mut u8, from: *const u8, len: usize) -> bool {
    install();
    // SAFETY: `from` is our readable host buffer; any unmapped or read-only
    // destination page ends the write with 0 via [`install`].
    unsafe { victory_write_guarded(to, from, len) != 0 }
}

#[cfg(not(target_arch = "x86_64"))]
fn guarded_write(_: *mut u8, _: *const u8, _: usize) -> bool {
    false
}

fn write_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var_os("VICTORY_WRITE_PLAIN").is_none_or(|v| v != "0"))
}

/// With [`on`] (and `VICTORY_WRITE_PLAIN!=0`): writes host memory at `from`
/// into title memory at `to` in place; false if disabled or a destination page
/// is not writable (caller falls back to `NtWriteVirtualMemory`). Counted in
/// [`COUNT`]: writes made in place, their bytes, and the writes that met an
/// unmapped or read-only destination page (falling back to the kernel's
/// write).
pub(crate) fn write(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !write_on() {
        return false;
    }
    let whole = guarded_write(to, from, len);
    if whole {
        COUNT.add(WRITES, 1);
        COUNT.add(WRITTEN_BYTES, len as u64);
    } else {
        COUNT.add(WRITES_GONE, 1);
    }
    whole
}

/// With [`on`]: copies title memory at `from` into host memory at `to`;
/// false if it is off or a source page was not there (the caller then
/// copies through the kernel, as before).
pub(crate) fn copy(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !on() {
        return false;
    }
    if len < LEAST {
        COUNT.add(SMALL, 1);
        return false;
    }
    if faulted_lately(from as usize, len) {
        COUNT.add(LATELY, 1);
        return false;
    }
    let whole = guarded(to, from, len);
    if whole {
        COUNT.add(COPIES, 1);
        COUNT.add(COPIED_BYTES, len as u64);
    } else {
        COUNT.add(GONE, 1);
        note_fault(from as usize, len);
    }
    whole
}

/// With [`on`]: copies a run of title memory that read the last time it
/// was copied (`crate::host::sparseknown`, the holes beside it remembered)
/// in place, whatever its length and whether its blocks faulted lately
/// (the holes beside it did). False if it is off or a page was not there.
pub(crate) fn copy_run(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !on() {
        return false;
    }
    let whole = guarded(to, from, len);
    if whole {
        COUNT.add(COPIES, 1);
        COUNT.add(COPIED_BYTES, len as u64);
    } else {
        COUNT.add(GONE, 1);
        note_fault(from as usize, len);
    }
    whole
}

/// With [`on`]: copies `len` bytes, fewer than [`LEAST`], of title memory
/// at `from` into host memory at `to`, for a caller that is NOT asking
/// whether the page is there but expects it to be (a list's fill of upload
/// memory: constants the title recorded a draw with). False if it is off,
/// if the source faulted lately, or if a source page was not there; the
/// caller then copies through the kernel, as before.
///
/// Why (fork de4146a0, fxceless): the floor of [`copy`] is there for the
/// askings, which fail. The fills do not: 5,500 to 6,800 of them a frame
/// went to the kernel's copy in profile PF2 (5.4 ms of the submitting
/// thread, "of under 256 bytes" on the line), about 0.8 to 1 us each,
/// where the fork timed the copy in place of 16 to 128 bytes at about 30
/// ns.
pub(crate) fn copy_short(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !on() {
        return false;
    }
    if faulted_lately(from as usize, len) {
        COUNT.add(LATELY, 1);
        return false;
    }
    let whole = guarded(to, from, len);
    if whole {
        COUNT.add(FILLS, 1);
    } else {
        COUNT.add(GONE, 1);
        note_fault(from as usize, len);
    }
    whole
}

/// Whether a list's fills of under [`LEAST`] bytes are copied in place
/// (`VICTORY_FILLS_PLAIN=1`, with `VICTORY_COPY_PLAIN=1`; off by default:
/// through the kernel's copy, as before). `alt`: twenty seconds with,
/// twenty without, in turn, each change said in the log.
///
/// A fill copies constants the title recorded a draw with into upload
/// memory of the list's own (`crate::host`'s `Step::Fill`). [`copy`] leaves
/// copies that short to the kernel because the layer's askings are that
/// short and fail; the fills are there, and each was a system call. The
/// same bytes, read at the same step; a source page that is gone goes to
/// the kernel's copy as before.
fn fills_plain() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_FILLS_PLAIN").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => return 0,
        };
        if !on() {
            tracing::warn!("VICTORY_FILLS_PLAIN needs VICTORY_COPY_PLAIN=1: the fills go through the kernel's copy as before");
            return 0;
        }
        tracing::info!(
            "a list's fills of under {LEAST} bytes are copied in place, the kernel's copy behind them for a page that is \
             not there (VICTORY_FILLS_PLAIN={})",
            if mode == 1 { "1" } else { "alt: twenty seconds with, twenty without" }
        );
        mode
    });
    match mode {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
                tracing::info!(
                    "a list's fills of under {LEAST} bytes are copied in place from now (VICTORY_FILLS_PLAIN=alt): {}",
                    if on { "ON" } else { "off, through the kernel's copy as usual" }
                );
            }
            on
        }
    }
}

/// A list's fill ([`crate::host`]'s `Step::Fill`): `len` bytes of title
/// memory at `src` into upload memory at `dst`. In place if it is short
/// and [`fills_plain`] says so, else (and if a source page was not there)
/// as every copy (`crate::host::copy_mapped`).
pub(crate) fn fill(dst: *mut u8, src: *const u8, len: usize) {
    // (Not with the copies line's timing modes, `VICTORY_COPY_MODE`.)
    #[cfg(feature = "diagnostics")]
    let timing_only = crate::copystat::mode() != 0;
    #[cfg(not(feature = "diagnostics"))]
    let timing_only = false;
    if len > 0 && len < LEAST && !timing_only && fills_plain() && copy_short(dst, src, len) {
        return;
    }
    crate::host::copy_mapped(dst, src, len, false);
}

/// With the frame line: what [`copy`] did per frame over `frames`.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    // (Each count's total since the line before, as the shared counters'
    // swaps to 0 gave.)
    let taken = COUNT.take();
    let [copies, bytes, gone, lately, small, fills] =
        [COPIES, COPIED_BYTES, GONE, LATELY, SMALL, FILLS].map(|i| taken[i] as f64 / frames);
    let [writes, wbytes, wgone] = [WRITES, WRITTEN_BYTES, WRITES_GONE].map(|i| taken[i] as f64 / frames);
    // (Said only with `VICTORY_FILLS_PLAIN`.)
    let fills = if fills_plain() || fills > 0.0 {
        format!("; fills of under {LEAST} bytes made in place (VICTORY_FILLS_PLAIN): {fills:.0}")
    } else {
        String::new()
    };
    tracing::info!(
        "copies of title memory made in place per frame (VICTORY_COPY_PLAIN=1): {copies:.0}, {:.1} MB; {gone:.1} met a page \
         that was not there and went through the kernel's copy; left to the kernel's copy untried: {lately:.0} whose source \
         faulted in the last two seconds, {small:.0} of under {LEAST} bytes; writes back in place: {writes:.0}, {:.1} MB, \
         {wgone:.1} fell back{fills}",
        bytes / 1e6,
        wbytes / 1e6
    );
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const PAGE: usize = 4096;

    #[test]
    fn memory_that_is_there_is_copied() {
        let from: Vec<u8> = (0..100_000u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut to = vec![0u8; from.len()];
        assert!(guarded(to.as_mut_ptr(), from.as_ptr(), from.len()));
        assert_eq!(to, from);
        // Nothing to copy is a copy.
        assert!(guarded(to.as_mut_ptr(), from.as_ptr(), 0));
    }

    /// A fill's few bytes (fork de4146a0): copied whole where they are
    /// there, up to the last byte of a page; a copy that runs into a hole
    /// by one byte ends with false and the thread goes on.
    #[test]
    fn a_short_copy_is_whole_or_ends_without_a_fault() {
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: committed and writable.
        let page = unsafe { std::slice::from_raw_parts_mut(base, PAGE) };
        for (i, b) in page.iter_mut().enumerate() {
            *b = (i * 13 + 5) as u8;
        }
        for len in [1usize, 16, 64, 255] {
            let mut to = vec![0u8; len];
            // The page's last bytes.
            assert!(guarded(to.as_mut_ptr(), base.wrapping_add(PAGE - len), len), "{len}");
            assert_eq!(to, page[PAGE - len..], "{len}");
            // One byte into the hole.
            let mut to = vec![0u8; len];
            assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE - len + 1), len), "{len}");
            assert_eq!(to[..len - 1], page[PAGE - len + 1..], "{len}: the bytes before the hole");
        }
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// Three pages reserved, the first committed: a copy that stays in the
    /// first page is whole; one that runs into the second ends with false,
    /// the bytes before the hole copied, and the thread goes on.
    #[test]
    fn a_page_that_is_not_there_ends_the_copy_without_a_fault() {
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: committed and writable.
        unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0xAB);
        let mut to = vec![0u8; 2 * PAGE];
        assert!(guarded(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        to.fill(0);
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE / 2), PAGE));
        assert!(to[..PAGE / 2].iter().all(|&b| b == 0xAB), "the bytes before the hole");
        assert!(to[PAGE / 2..].iter().all(|&b| b == 0), "nothing after it");
        // A source that is not there at all.
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE), PAGE));
        // Committed but closed to reads.
        // SAFETY: the third page of the reservation.
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_NOACCESS) }.is_null());
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(2 * PAGE), 16));
        // And the registers the caller owns came back: this still works.
        assert!(guarded(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// The counts, kept per thread, reach the line once each whichever
    /// thread added them, each in its own place: what the shared counters'
    /// swaps to 0 gave. (No other test adds to them: they call `guarded`.)
    #[test]
    fn counts_of_every_thread_are_taken_once() {
        let _ = COUNT.take();
        std::thread::scope(|s| {
            for t in 0..6u64 {
                s.spawn(move || {
                    for _ in 0..1_000 {
                        COUNT.add(COPIES, 1);
                        COUNT.add(COPIED_BYTES, 4096 + t);
                        COUNT.add(WRITES_GONE, 1);
                    }
                });
            }
        });
        let taken = COUNT.take();
        let mut wanted = [0u64; 9];
        wanted[COPIES] = 6_000;
        wanted[COPIED_BYTES] = 6_000 * 4096 + 1_000 * (1 + 2 + 3 + 4 + 5);
        wanted[WRITES_GONE] = 6_000;
        assert_eq!(taken, wanted);
        assert_eq!(COUNT.take(), [0; 9], "taken once");
    }

    /// A source that faulted is remembered by its 64 KiB blocks, and only
    /// those.
    #[test]
    fn a_source_that_faulted_is_remembered_by_its_blocks() {
        // (Addresses no other test uses; the table is shared.)
        let at = 0x7654_3210_0000usize;
        assert!(!faulted_lately(at, 4096));
        note_fault(at + 100, 4096);
        assert!(faulted_lately(at, 16));
        assert!(faulted_lately(at + 0xffff, 1));
        // A copy that ends in the block, and one that begins in it.
        assert!(faulted_lately(at - 4096, 8192));
        // The blocks beside it are not remembered.
        assert!(!faulted_lately(at + 0x10000, 4096));
        assert!(!faulted_lately(at - 0x10000, 4096));
        assert_eq!(blocks(0, 0), (1 << 63, 1 << 63));
    }

    /// Many threads at once, half of them faulting: each gets its own answer.
    #[test]
    fn threads_fault_and_copy_side_by_side() {
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(base, 0);
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                std::thread::spawn(move || {
                    let mut to = vec![0u8; PAGE];
                    for _ in 0..2_000 {
                        let from = (base + if t % 2 == 0 { 0 } else { PAGE }) as *const u8;
                        assert_eq!(guarded(to.as_mut_ptr(), from, PAGE), t % 2 == 0);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn a_readonly_or_unmapped_destination_ends_the_write_without_a_fault() {
        use windows_sys::Win32::System::Memory::PAGE_READONLY;
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: page 0 writable, page 1 uncommitted, page 2 read-only.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_READONLY) }.is_null());
        let src = vec![0x5Au8; 2 * PAGE];
        assert!(guarded_write(base, src.as_ptr(), PAGE));
        // SAFETY: committed and readable.
        assert!(unsafe { std::slice::from_raw_parts(base, PAGE) }.iter().all(|&b| b == 0x5A));
        // Spanning into uncommitted page 1 fails cleanly.
        assert!(!guarded_write(base.wrapping_add(PAGE / 2), src.as_ptr(), PAGE));
        // Writing to read-only page 2 fails cleanly.
        assert!(!guarded_write(base.wrapping_add(2 * PAGE), src.as_ptr(), 16));
        // Caller-saved registers intact: valid write still succeeds.
        assert!(guarded_write(base, src.as_ptr(), PAGE));
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }
}
