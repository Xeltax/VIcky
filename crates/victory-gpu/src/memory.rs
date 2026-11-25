//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!   They and the other notes below are in the PC's memory, or with
//!   `VICTORY_NOTES_ON_CARD=1` in the card's and copied back ([`notes`]).
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date.
//!
//! **Which copies shaders write.** Declaring a copy written costs a
//! snapshot of its bytes before the submission and a compare after it
//! (`shadow.rs`), and the prologue numbers every copy there is a number
//! for: with all of them declared by each list that can store, a frame
//! took 3.3 s, 2.8 s of it snapshots and compares (RTX 5080, structure
//! builder on). So shaders mark the copies their stores go to (the
//! **marks**, `victory_shader::emit::MEMORY_MARKS`, read back with the
//! faults once a frame), and a list that can store declares written:
//!
//! - the copies marked in one of the last [`MARK_FRAMES`] frames;
//! - the copies its dispatches name (root views and root constants: where
//!   the builder is told to put a structure);
//! - the copies numbered while it is recorded.
//!
//! shortcut: the first store to a copy none of these holds is not copied
//! back to title memory (its mark brings the copy in from the next frame
//! the marks are read in). That is the loss a page without a copy already
//! has, for a frame or two; the log counts such marks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_MARKS, MEMORY_NOTES, MEMORY_PAGE_MARKS, MEMORY_PAGES, MEMORY_TOUCHED};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

pub(crate) mod notes;
use notes::Notes;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// A copy counts as one shaders write for this many frames after a mark:
/// marks are read once a frame, while lists recorded earlier still run.
const MARK_FRAMES: u64 = 8;
const _: () = assert!(COPIES < MEMORY_MARKS);

/// `VICTORY_MEMORY_NAMED_WRITES=0`: a copy that a dispatch only names (by a
/// root view or a constant that looks like an address) stops being declared
/// written by the name alone once it has been in [`NAMED_FRAMES`] frames
/// without any shader ever marking it. Unset or 1: every name declares its
/// copies written, as before.
///
/// Why: a constant at or above 0x0100_0000 is taken as an address in
/// 256-byte units, and each name maps two 4 MiB windows. Copies so named
/// cost a snapshot before each storing submission and a compare after it
/// even when no shader stores there. The marks (#346) say which copies
/// shaders store to; one that is never marked is one shaders did not write.
///
/// shortcut: a store to such a copy after it stopped being declared is not
/// copied back until its mark is read (the next frame), as for any copy
/// #346 had not declared.
fn named_writes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MEMORY_NAMED_WRITES").is_none_or(|v| v != "0"))
}

/// See [`named_writes`].
const NAMED_FRAMES: u32 = 3;

/// `VICTORY_MEMORY_PAGE_MARKS=1`: a copy declared written because shaders
/// marked it lately is declared written only on the 64 KiB pages they
/// marked lately, not whole (named and new copies: see
/// [`named_write_most`]). Unset: whole copies, as #346 does. The marks are
/// the pages' either way (`MEMORY_PAGE_MARKS`).
///
/// Why: the title's 174 MB heap is one copy. Some shader stores into it
/// every frame, so all of it was declared written: snapshotted and compared
/// whole (365 MB a frame), and a fill of upload memory that reads 480
/// bytes of it waited for the flight that wrote it to land, 77-114 ms a
/// frame with the relay on (RTX 4080, structure builder on).
///
/// shortcut: a store to a page no shader marked within [`MARK_FRAMES`] is
/// not copied back until its mark is read (the next frame), as #346
/// already does for a copy no list declared.
fn page_marks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MEMORY_PAGE_MARKS").is_some_and(|v| v == "1"))
}

/// `VICTORY_NAMED_WRITE_MB=<n>`: a copy a dispatch names (a root view or a
/// constant taken for an address) is declared written from the named
/// address to n MB after it, with the pages shaders marked lately
/// ([`page_marks`]), instead of whole. Unset: whole.
///
/// shortcut: a root view's shader that writes further than n MB past its
/// address is not copied back there.
fn named_write_most() -> Option<u64> {
    static MOST: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_NAMED_WRITE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map(|mb| mb << 20)
    })
}

/// The runs of pages in `[start, start + len)` marked within
/// [`MARK_FRAMES`] of `frame`, as (title address, bytes), cut to the range.
fn marked_runs(marked: &HashMap<u64, u64>, start: u64, len: u64, frame: u64) -> Vec<(u64, u64)> {
    let mut runs: Vec<(u64, u64)> = Vec::new();
    let end = start + len;
    let mut page = start / PAGE;
    while page * PAGE < end {
        if marked.get(&page).is_some_and(|&f| frame.wrapping_sub(f) <= MARK_FRAMES) {
            let (a, e) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
            match runs.last_mut() {
                Some(last) if last.0 + last.1 == a => last.1 += e - a,
                _ => runs.push((a, e - a)),
            }
        }
        page += 1;
    }
    runs
}

/// `runs` and `[at, at + bytes)` joined, in order.
fn with_run(mut runs: Vec<(u64, u64)>, at: u64, bytes: u64) -> Vec<(u64, u64)> {
    if bytes == 0 {
        return runs;
    }
    runs.push((at, bytes));
    runs.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (a, n) in runs {
        match out.last_mut() {
            Some(last) if a <= last.0 + last.1 => last.1 = last.1.max(a + n - last.0),
            _ => out.push((a, n)),
        }
    }
    out
}

/// `VICTORY_MEMORY_REFRESH=touched`: on a list's first dispatch, only the
/// copies a shader touched lately (`MEMORY_TOUCHED`, within
/// [`MARK_FRAMES`]) are brought up to date, with the named, the new and
/// those declared written, and every copy at least once in
/// [`REFRESH_FRAMES`]. Unset: every numbered copy, as before.
///
/// Why: with the structure builder on, all 1,023 copies (4.3 GB) are
/// refreshed on each list's first dispatch, 4 to 5 lists a frame, and each
/// refresh asks the kernel's write watch over them: about 5,700 askings
/// and 28 GB checked a frame against 2,400 and 11 GB without the builder
/// (RTX 4080). Shaders read about a quarter of the copies in a frame.
///
/// shortcut: a copy no shader touched for [`MARK_FRAMES`] frames and not
/// refreshed in the last [`REFRESH_FRAMES`] reads bytes up to that many
/// frames old at its next first touch; it is refreshed from the next
/// frame on.
fn refresh_touched() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_MEMORY_REFRESH").is_ok_and(|v| v.trim() == "touched"))
}

/// See [`refresh_touched`].
const REFRESH_FRAMES: u64 = 8;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    /// The buffer shaders note faults, marks and touches in (`SLOT_FAULTS`)
    /// and its GPU address.
    faults: (ID3D12Resource, u64),
    /// Its mapping, where the notes are read and cleared, when it is in the
    /// PC's memory; `None` when it is in the card's ([`notes::on_card`]):
    /// the copies [`notes`] makes of it are read instead.
    mapped: Option<*mut u32>,
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Per copy: the frame its mark was last found set in (`u64::MAX`:
    /// never), and the frame a list last declared it written in.
    marked: Vec<(u64, u64)>,
    /// Per directory page shaders stored to, the frame its mark
    /// (`MEMORY_PAGE_MARKS`) was last found set in.
    page_marked: HashMap<u64, u64>,
    /// Per copy: the last frame a dispatch's name alone declared it written,
    /// and in how many frames that happened while it was never marked
    /// ([`named_writes`]).
    named: Vec<(u64, u32)>,
    /// Per copy: the frame a shader was last found to have touched it
    /// (`MEMORY_TOUCHED`), and the frame a list last refreshed it
    /// (`u64::MAX`: never).
    touched: Vec<(u64, u64)>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// For the log, per frame (diagnostics): lists that refreshed every copy,
/// copies refreshed and their bytes; copies declared written because they
/// are new, because a dispatch named them, because a shader marked them
/// lately, and their bytes; of the named, those never marked, and their
/// bytes; copies a list's first dispatch did not refresh
/// ([`refresh_touched`]); copies of the card's notes read ([`notes`]), and
/// the frames from their sending to their reading, summed.
#[cfg(feature = "diagnostics")]
static TALLY: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];

#[cfg(feature = "diagnostics")]
fn tally(i: usize, v: u64) {
    TALLY[i].fetch_add(v, std::sync::atomic::Ordering::Relaxed);
}

/// One line for the frame just ended (the first 400 frames that had any).
#[cfg(feature = "diagnostics")]
fn tally_line(frame: u64, numbered: usize) {
    use std::sync::atomic::Ordering;
    let t: Vec<u64> = TALLY.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    if t.iter().all(|&v| v == 0) {
        return;
    }
    static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if SAID.fetch_add(1, Ordering::Relaxed) >= 400 {
        return;
    }
    tracing::info!(
        frame,
        numbered,
        lists_refreshing = t[0],
        refreshed = t[1],
        refreshed_mb = t[2] >> 20,
        declared_new = t[3],
        declared_named = t[4],
        declared_marked = t[5],
        declared_mb = t[6] >> 20,
        named_never_marked = t[7],
        named_mb = t[8] >> 20,
        refresh_skipped = t[9],
        notes_copies_read = t[10],
        notes_frames_late = t[11],
        "title memory by address, last frame: refreshes and copies declared written by reason"
    );
}

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        let (faults, mapped) = match notes::on_card().then(notes::card).flatten() {
            Some(card) => {
                // SAFETY: a live buffer.
                let gpu = unsafe { card.buffer.GetGPUVirtualAddress() };
                ((card.buffer.clone(), gpu), None)
            }
            None => {
                let (buffer, data, gpu) = make(u64::from(MEMORY_NOTES), "faults, marks and touches")?;
                ((buffer, gpu), Some(data))
            }
        };
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults,
            mapped,
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            marked: Vec::new(),
            page_marked: HashMap::new(),
            named: Vec::new(),
            touched: Vec::new(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's. Returns the index in `copies` of the page's copy.
    fn map(&mut self, address: u64) -> Option<usize> {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return None;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        match unsafe { entry.read() } {
            0 => {}
            n => return Some(n as usize - 1),
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return None;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return None;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let shadow = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))?;
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(shadow.clone());
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return None;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
        Some(n)
    }

    /// Everything shaders noted since the notes were last read, as of
    /// `frame`.
    fn take(&mut self, notes: &Notes, frame: u64) {
        self.take_faults(notes);
        self.take_marks(frame, notes);
        self.take_touched(frame, notes);
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self, notes: &Notes) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // A number missed now (the GPU may be writing) is written again
            // next frame.
            let page = notes.take(i);
            if page != 0 {
                pages.push(page);
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// The marks set since they were last read: their copies are ones
    /// shaders write, as of `frame`.
    fn take_marks(&mut self, frame: u64, notes: &Notes) {
        self.marked.resize(self.copies.len(), (u64::MAX, u64::MAX));
        // Marks of copies no list had declared written lately: stores that
        // were not copied back.
        let mut late = 0;
        for n in 0..self.copies.len() {
            // The copy's pages' marks (`MEMORY_PAGE_MARKS`): the copy is
            // marked if any of them is. Each is taken once; the GPU may be
            // writing: a mark set after this read is found next frame.
            let (start, len) = self.copies[n].span();
            let (first, last) = (start / PAGE, (start + len).div_ceil(PAGE).min(u64::from(MEMORY_PAGES)));
            let mut any = false;
            for page in first..last {
                if notes.take(MEMORY_PAGE_MARKS as usize + page as usize) != 0 {
                    self.page_marked.insert(page, frame);
                    any = true;
                }
            }
            if !any {
                continue;
            }
            let (_, declared) = self.marked[n];
            if declared == u64::MAX || frame.wrapping_sub(declared) > MARK_FRAMES {
                late += 1;
            }
            self.marked[n].0 = frame;
        }
        // Pages not marked for long are forgotten.
        if frame % 64 == 0 {
            self.page_marked.retain(|_, &mut f| frame.wrapping_sub(f) <= MARK_FRAMES);
        }
        if late != 0 {
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::info!(
                    copies = late,
                    of = self.copies.len(),
                    "title memory by address: shaders stored to copies no list had declared written (those stores were not copied back); they are declared from now on"
                );
            }
        }
    }

    /// Whether a dispatch naming copy `n` declares it written in `frame`
    /// ([`named_writes`]), and its count of such frames while unmarked.
    fn name_declares(&mut self, n: usize, frame: u64) -> bool {
        if named_writes() {
            return true;
        }
        if self.named.len() <= n {
            self.named.resize(n + 1, (u64::MAX, 0));
        }
        if self.marked.get(n).is_some_and(|&(marked, _)| marked != u64::MAX) {
            return true;
        }
        let (last, frames) = &mut self.named[n];
        if *last != frame {
            if *frames >= NAMED_FRAMES {
                return false;
            }
            *last = frame;
            *frames += 1;
        }
        true
    }

    /// The touches set since they were last read ([`refresh_touched`]).
    fn take_touched(&mut self, frame: u64, notes: &Notes) {
        self.touched.resize(self.copies.len(), (u64::MAX, u64::MAX));
        for n in 0..self.copies.len() {
            // The GPU may be writing: a touch set after this read is found
            // next frame.
            if notes.take(MEMORY_TOUCHED as usize + n + 1) != 0 {
                self.touched[n].0 = frame;
            }
        }
    }

    /// Whether copy `n` is to be refreshed on a list's first dispatch
    /// ([`refresh_touched`]).
    fn refresh_due(&self, n: usize, frame: u64) -> bool {
        let Some(&(touched, refreshed)) = self.touched.get(n) else {
            return true;
        };
        (touched != u64::MAX && frame.wrapping_sub(touched) <= MARK_FRAMES)
            || refreshed == u64::MAX
            || frame.wrapping_sub(refreshed) >= REFRESH_FRAMES
            || self.written_lately(n, frame)
    }

    /// Whether copy `n` is one shaders wrote lately.
    fn written_lately(&self, n: usize, frame: u64) -> bool {
        self.marked.get(n).is_some_and(|&(marked, _)| {
            marked != u64::MAX && frame.wrapping_sub(marked) <= MARK_FRAMES
        })
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it. Their indices go to `named`.
    fn name(&mut self, address: u64, named: &mut Vec<usize>, at: &mut Vec<(usize, u64)>) {
        if let Some(n) = self.map(address) {
            named.push(n);
            at.push((n, address));
        }
        let next = (address & !(CELL - 1)) + CELL;
        if let Some(n) = self.map(next) {
            named.push(n);
            at.push((n, next));
        }
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    // The key of the list's first dispatch by address (`first_read` below).
    const READ: usize = usize::MAX - 0x6d;
    if m.frame != frame {
        #[cfg(feature = "diagnostics")]
        tally_line(m.frame, m.copies.len());
        m.frame = frame;
        m.refused.clear();
        if let Some(at) = m.mapped {
            m.take(&Notes::Shared(at), frame);
        }
    }
    if m.mapped.is_none() && !list.used(READ) {
        // The notes in the card's memory: their last copy once its
        // submission has finished, at the first list's first dispatch after
        // that, which brings every copy up to date: those the faults number
        // now too.
        let taken = notes::take(|copied| m.take(&Notes::copied(copied), frame));
        #[cfg(feature = "diagnostics")]
        if let Some(sent) = taken {
            tally(10, 1);
            tally(11, frame.saturating_sub(sent));
        }
        let _ = taken;
    }
    let numbered = m.copies.len();
    let mut named = Vec::new();
    // And where each named copy was named ([`named_write_most`]).
    let mut named_at: Vec<(usize, u64)> = Vec::new();
    for &address in addresses {
        m.name(address, &mut named, &mut named_at);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32, &mut named, &mut named_at);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8, &mut named, &mut named_at);
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Once per list, and for copies numbered since: the builder alone
    // dispatches hundreds of times a frame.
    // shortcut: a copy another list numbers meanwhile is brought up to date
    // by that list's submission, which may run after this one's. (`READ` is
    // above.)
    const WRITE: usize = usize::MAX - 0x6e;
    // One key per copy below this one, for "this list declared it written".
    const WRITTEN: usize = usize::MAX - 0x1000;
    let first_read = list.first_use(READ);
    let first_write = writes && list.first_use(WRITE);
    m.marked.resize(m.copies.len(), (u64::MAX, u64::MAX));
    m.touched.resize(m.copies.len(), (u64::MAX, u64::MAX));
    // With [`refresh_touched`]: one key per copy below this one, for "this
    // list refreshed it".
    const REFRESHED: usize = usize::MAX - 0x2000;
    let touched_only = refresh_touched();
    // Which copies a list's first dispatch refreshes: all of them, or
    // ([`refresh_touched`]) those due and those named.
    let due: Vec<bool> = if touched_only && first_read {
        (0..m.copies.len()).map(|n| named.contains(&n) || m.refresh_due(n, frame)).collect()
    } else {
        Vec::new()
    };
    // The named copies whose name declares them written ([`named_writes`]).
    let named: Vec<usize> = named.into_iter().filter(|&n| m.name_declares(n, frame)).collect();
    for (n, shadow) in m.copies.iter().enumerate() {
        let (start, len) = shadow.span();
        // Written: see the module. The copies marked lately once per list,
        // the named and the new ones as they come.
        let is_new = n >= numbered;
        let is_named = named.contains(&n);
        let wanted = is_new || is_named || (first_write && m.written_lately(n, frame));
        let declaring = writes && wanted && list.first_use(WRITTEN - n);
        let refresh = if !touched_only {
            first_read || is_new
        } else {
            // Once per list: due on its first dispatch, or new, or about to
            // be declared written (its snapshot is taken after it).
            (is_new || declaring || (first_read && due[n])) && list.first_use(REFRESHED - n)
        };
        if refresh {
            shadow.refresh(list, start, len);
            m.touched[n].1 = frame;
            #[cfg(feature = "diagnostics")]
            {
                tally(1, 1);
                tally(2, len);
            }
        } else if first_read {
            #[cfg(feature = "diagnostics")]
            tally(9, 1);
        }
        if declaring {
            // Not new: only what is known to be written, if the switches
            // ask for it ([`page_marks`], [`named_write_most`]).
            let mut runs = if !is_new && (page_marks() || (is_named && named_write_most().is_some())) {
                marked_runs(&m.page_marked, start, len, frame)
            } else {
                Vec::new()
            };
            let whole = is_new
                || (is_named && named_write_most().is_none())
                || (!is_named && !page_marks());
            if !whole && is_named {
                let most = named_write_most().unwrap_or(u64::MAX);
                for &(i, at) in &named_at {
                    if i == n && at >= start && at < start + len {
                        runs = with_run(runs, at, most.min(start + len - at));
                    }
                }
            }
            if whole || runs.is_empty() {
                shadow.gpu_writes(list, start, len);
            } else {
                for &(at, bytes) in &runs {
                    shadow.gpu_writes(list, at, bytes);
                }
            }
            #[cfg(feature = "diagnostics")]
            {
                tally(if is_new { 3 } else if is_named { 4 } else { 5 }, 1);
                tally(6, len);
                if is_named && !is_new && m.marked[n].0 == u64::MAX {
                    tally(7, 1);
                    tally(8, len);
                }
            }
            m.marked[n].1 = frame;
        }
    }
    #[cfg(feature = "diagnostics")]
    {
        if first_read {
            tally(0, 1);
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.1,
        base,
    })
}

/// For a pass of ours that stands in for a translated program reading and
/// writing title memory by address (`crate::draw`'s `bvh_copy`): how many
/// bytes from `address` on, at most `most`, lie on pages the directory
/// gives to `shadow` at the offsets `shadow` holds them, so that the pass
/// reaches the very bytes the program would; with the number the
/// directory gives the copy (its touch is dword [`MEMORY_TOUCHED`] +
/// number of the notes) and the GPU address of the notes (in the PC's
/// memory, or with `VICTORY_NOTES_ON_CARD=1` in the card's: [`notes`]).
/// None without a directory, or when the page of `address` is not
/// `shadow`'s.
pub(crate) fn run_in(shadow: &Arc<Shadow>, address: u64, most: u64) -> Option<(u64, u32, u64)> {
    let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let m = guard.as_ref()?;
    let (start, len) = shadow.span();
    let end = address.saturating_add(most).min(start + len);
    let (mut page, mut reach, mut number) = (address / PAGE, address, None);
    while reach < end && page < u64::from(MEMORY_PAGES) - 1 {
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { m.directory.1.add(page as usize * 2) };
        // SAFETY: as above; entries are written on this side only.
        let (n, offset) = unsafe { (entry.read(), entry.add(1).read()) };
        let ours = n != 0
            && m.copies.get(n as usize - 1).is_some_and(|c| Arc::ptr_eq(c, shadow))
            && shadow.offset_in(page * PAGE) == Some(u64::from(offset));
        if !ours {
            break;
        }
        number.get_or_insert(n);
        reach = ((page + 1) * PAGE).min(end);
        page += 1;
    }
    Some((reach - address, number?, m.faults.1))
}
