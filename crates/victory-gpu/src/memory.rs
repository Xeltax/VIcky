//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`HostShader::memory`; the contract is
//! `victory_shader::emit`'s `SLOT_MEMORY`, `SLOT_FAULTS` and
//! `USER_MEMORY_BASE`): the console's ray-tracing structure builder and the
//! programs that walk its structures, which lie in plain title memory, in
//! no buffer object.
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view, a table's view and a raw address reach the same host bytes.
//! This module keeps what a shader needs to find them:
//!
//! - the **directory** (`t254`): per 64 KiB page of title address space the
//!   number of the copy that holds the page (0: none) and the page's byte
//!   offset in it; one buffer for all lists. While the GPU may run, entries
//!   only go from empty to filled (offset first, then number); they are
//!   emptied only at submission ([`prepare`]), when no work of ours runs;
//! - per view heap a **run** of `COPIES + 1` dynamic slots, held for the
//!   heap's life: slot 0 a null raw unordered access view (reads zeros,
//!   drops writes), slot `n` a raw unordered access view of copy `n`, the
//!   others null. A copy's view is written into every heap's run before
//!   its number enters the directory, so a shader that finds a number
//!   finds its view in whatever heap it runs with;
//! - per host list the **faults** (`u253`): the page numbers its shaders
//!   found empty, read once the list has run. Those pages get their copies
//!   at the next draw or dispatch that uses title memory, and the list's
//!   writes do not go back into title memory (`HostList::defer_missed`):
//!   what a shader computed from zeros read in place of memory is not the
//!   console's, and written back it broke the title's own data.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next run. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the frame line says whether that happens. The whole
//! answer is the extent of the title's allocation around an address, which
//! the memory module of the layer knows and this one does not ask yet.
//!
//! Every submission with such a list brings every numbered copy up to date
//! ([`prepare`], once per submission until a step writes title memory). A
//! list that can store declares every copy numbered when it records such a
//! draw written (the bytes are compared on the CPU after the run, as for
//! any unordered access view).
//! shortcut: a copy numbered after a list's last such draw was recorded
//! (by another list, from its root arguments or faults) is not declared
//! written by this one: what this list's shaders store through it stays in
//! the host copy. A dirty-page mask written by the shaders would bound both
//! the refresh and the compare (upstream's page bitmap did; the contract
//! has none).
//!
//! A numbered copy the shadow module no longer finds at its range (its
//! buffer released, another made there) is retired at the next submission
//! after a frame change: its directory entries emptied, its views made
//! null, its number free again two frames later. At most [`COPIES`] copies
//! are numbered at once (4 MiB windows mostly: 4 GiB at most).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (a run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// Frames a retired copy's number waits before it names another copy: a
/// submission recorded before the retirement may still be running.
const REUSE_AFTER: u64 = 2;
/// The key of our own view heap ([`heap`]) among the title's.
pub(crate) const OURS: usize = usize::MAX;

/// Our view heap for lists whose shaders use title memory by address with
/// no view heap of the title's bound (dynamic slots only).
pub(crate) fn heap() -> Option<&'static HostHeap> {
    static HEAP: std::sync::OnceLock<Option<HostHeap>> = std::sync::OnceLock::new();
    HEAP.get_or_init(|| HostHeap::new(false, 0)).as_ref()
}

/// Runs `f` on the host heap behind view heap `key` (a title heap's
/// address, or [`OURS`]) while it cannot go away; None if it is gone.
fn with_heap<R>(key: usize, f: impl FnOnce(&HostHeap) -> R) -> Option<R> {
    if key == OURS {
        return heap().map(f);
    }
    crate::descriptor::with_host(key, f)
}

/// A view heap's run of slots.
struct Run {
    /// The host heap it was made in (a heap made later at a released one's
    /// address is another).
    host: usize,
    /// Its first slot: the null view; copy `n`'s view is `n` further.
    base: u32,
}

/// A directory binding belongs to the current title allocation. For in-place
/// allocations the visible buffer is a staging copy so page misses can discard writes.
struct Mapping {
    owner: std::sync::Weak<Shadow>,
    visible: Arc<Shadow>,
}

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]` (None: retired, or never numbered).
    copies: Vec<Option<Mapping>>,
    /// Per copy, when it was numbered ([`Memory::numbered`] then).
    since: Vec<u64>,
    /// Numberings so far: a list declares the copies numbered since it
    /// last declared.
    numbered: u64,
    /// Retired numbers (index into `copies`) and the frame they were
    /// retired in.
    free: Vec<(u32, u64)>,
    /// Per view heap key, its run.
    runs: HashMap<usize, Run>,
    /// Per host list (by address): its faults' GPU address, and the
    /// numberings it has declared written.
    lists: HashMap<usize, (u64, u64)>,
    /// Pages lists' shaders found empty, read after their runs.
    pending: HashSet<u32>,
    /// The frame of the last draw that used title memory, and of the last
    /// check for released copies.
    frame: u64,
    checked: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: HashSet<u64>,
}

// SAFETY: the mapping is of a buffer the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

fn memory() -> std::sync::MutexGuard<'static, Option<Memory>> {
    MEMORY.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a draw binds.
pub(crate) struct Ready {
    /// The directory (`t254`).
    pub(crate) directory: u64,
    /// The list's faults (`u253`).
    pub(crate) faults: u64,
    /// The view heap slot of the null view (`USER_MEMORY_BASE`); copy
    /// `n`'s view is `n` further.
    pub(crate) base: u32,
}

/// Since the last frame line: draws and dispatches using title memory by
/// address (by pipeline), copies numbered and retired, fault pages read,
/// lists whose writes were withheld and the bytes of those writes.
#[derive(Default)]
struct Stats {
    uses: std::collections::BTreeMap<String, u64>,
    numbered: u64,
    retired: u64,
    faults: u64,
    first_fault: Option<u64>,
    withheld_lists: u64,
    withheld_bytes: u64,
}

static STATS: Mutex<Option<Stats>> = Mutex::new(None);

fn stats<R>(f: impl FnOnce(&mut Stats) -> R) -> R {
    f(STATS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// `bytes` of a range shaders wrote through a view did not go back into
/// title memory: their list read memory with no copy (`crate::host`).
pub(crate) fn note_withheld(bytes: u64) {
    stats(|s| s.withheld_bytes += bytes);
}

/// A draw or dispatch of `pipeline` uses title memory by address.
pub(crate) fn note_use(pipeline: &str) {
    stats(|s| *s.uses.entry(pipeline.to_owned()).or_default() += 1);
}

/// Logs what title memory by address did since the last call, if anything:
/// a line per presented frame while shaders use it.
pub(crate) fn log_frame() {
    let s = stats(std::mem::take);
    if s.uses.is_empty() && s.numbered + s.retired + s.faults + s.withheld_bytes == 0 {
        return;
    }
    let live = memory()
        .as_ref()
        .map_or(0, |m| m.copies.iter().flatten().count());
    let mut uses: Vec<(&String, &u64)> = s.uses.iter().collect();
    uses.sort_by(|a, b| b.1.cmp(a.1));
    let uses: Vec<String> = uses.iter().take(32).map(|(p, n)| format!("{n} {p}")).collect();
    tracing::info!(
        "title memory by address this frame: {} draws/dispatches ({}), {} copies numbered ({} live, {} retired), \
         {} fault pages (first {:#x}), {} lists' writes withheld ({:.1} MB)",
        s.uses.values().sum::<u64>(),
        uses.join(", "),
        s.numbered,
        live,
        s.retired,
        s.faults,
        s.first_fault.unwrap_or(0),
        s.withheld_lists,
        s.withheld_bytes as f64 / 1e6,
    );
}

/// The directory pages copy `[first, first + len)` holds whole (the last
/// entry excluded: addresses past the directory read it).
fn whole_pages(first: u64, len: u64) -> std::ops::Range<u64> {
    let last = u64::from(MEMORY_PAGES) - 1;
    let from = first.div_ceil(PAGE);
    let to = ((first + len) / PAGE).min(last);
    from.min(to)..to
}

/// The directory pages `[first, first + len)` touches (the last entry
/// excluded).
fn touched_pages(first: u64, len: u64) -> std::ops::Range<u64> {
    let last = u64::from(MEMORY_PAGES) - 1;
    if len == 0 {
        return 0..0;
    }
    let from = (first / PAGE).min(last);
    let to = ((first + len - 1) / PAGE + 1).min(last);
    from..to
}

/// The page numbers a list's faults hold (dword 0 is the other lanes'
/// sink), each cleared.
fn take_fault_pages(words: &mut [u32]) -> Vec<u32> {
    let mut pages = Vec::new();
    for w in words.iter_mut().skip(1) {
        let page = std::mem::take(w);
        if page != 0 && !pages.contains(&page) {
            pages.push(page);
        }
    }
    pages
}

/// Whether a list's faults hold any page.
fn faulted(words: &[u32]) -> bool {
    words.iter().skip(1).any(|&w| w != 0)
}

/// The title addresses root constants may hold: a low dword and bits 32 to
/// 39 in the next, or one dword in 256-byte units. A constant that only
/// looks like one names memory nothing reads; one that is not the title's
/// memory is refused when its page is asked for.
fn constant_addresses(constants: &[u32]) -> Vec<u64> {
    let mut out = Vec::new();
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            out.push(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            out.push(u64::from(c) << 8);
        }
    }
    out
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let dwords = u64::from(MEMORY_PAGES) * 2;
        let (buffer, data) = host.shared_buffer(dwords * 4)?;
        crate::host::name(&buffer, || "title memory by address: directory".into());
        // SAFETY: a fresh mapping of at least that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: (buffer, data.cast::<u32>(), gpu),
            copies: Vec::new(),
            since: Vec::new(),
            numbered: 0,
            free: Vec::new(),
            runs: HashMap::new(),
            lists: HashMap::new(),
            pending: HashSet::new(),
            frame: u64::MAX,
            checked: u64::MAX,
            refused: HashSet::new(),
        })
    }

    /// Page `page`'s directory entry.
    fn entry(&self, page: u64) -> *mut u32 {
        debug_assert!(page < u64::from(MEMORY_PAGES) - 1);
        // SAFETY: inside the directory's mapping (pages below the last).
        unsafe { self.directory.1.add(page as usize * 2) }
    }

    /// Fills page `page`'s entry if it is empty: the offset first, so that
    /// a shader that sees the number sees the offset too.
    fn fill(&self, page: u64, number: u32, offset: u64) {
        let entry = self.entry(page);
        // SAFETY: inside the directory's mapping.
        unsafe {
            if entry.read_volatile() == 0 {
                entry.add(1).write_volatile(offset as u32);
                entry.write_volatile(number);
            }
        }
    }

    /// Writes copy index `n`'s view (None: a null view) into every heap's
    /// run; runs of heaps that are gone are dropped.
    fn write_view(&mut self, n: u32, copy: Option<&Arc<Shadow>>) {
        self.runs.retain(|&key, run| {
            with_heap(key, |heap| {
                if std::ptr::from_ref(heap) as usize != run.host {
                    return false;
                }
                let slot = run.base + 1 + n;
                match copy {
                    Some(c) => heap.raw_uav_at(slot, &c.buffer, 0, c.span().1),
                    None => heap.null_raw_uav_at(slot),
                }
                true
            })
            .unwrap_or(false)
        });
    }

    /// The number (index into `copies`) of `shadow`, numbering it if it has
    /// none: its views written, then every page it holds whole entered.
    fn number(&mut self, source: &Arc<Shadow>) -> Option<u32> {
        let owner = Arc::downgrade(source);
        if let Some(n) = self
            .copies
            .iter()
            .position(|c| c.as_ref().is_some_and(|c| c.owner.ptr_eq(&owner)))
        {
            return Some(n as u32);
        }
        let frame = crate::frame::current();
        let reusable = self
            .free
            .iter()
            .position(|&(_, retired)| frame >= retired + REUSE_AFTER);
        if reusable.is_none() && self.copies.len() == COPIES as usize {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros"));
            return None;
        }
        let (first, len) = source.span();
        let shadow = if source.in_place() {
            crate::shadow::copied_range(first, len)?
        } else {
            source.clone()
        };
        let n = match reusable {
            Some(k) => self.free.swap_remove(k).0,
            None => {
                self.copies.push(None);
                self.since.push(0);
                self.copies.len() as u32 - 1
            }
        };
        self.copies[n as usize] = Some(Mapping { owner, visible: shadow.clone() });
        self.numbered += 1;
        self.since[n as usize] = self.numbered;
        stats(|s| s.numbered += 1);
        // Views before the directory names the number.
        self.write_view(n, Some(&shadow));
        // Every page the copy holds whole, not only the one asked for: a
        // structure is more than 64 KiB.
        // shortcut: a page of a title buffer inside this one that has a
        // copy of its own is given this copy too, if it had none yet.
        for page in whole_pages(first, len) {
            self.fill(page, n + 1, page * PAGE - first);
        }
        Some(n)
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: inside the directory's mapping.
        if unsafe { self.entry(page).read_volatile() } != 0 || self.refused.contains(&page) {
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // Shaders store through the copies' views.
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            self.refused.insert(page);
            return;
        }
        if let Some(n) = self.number(&shadow) {
            self.fill(page, n + 1, shadow.offset_in(start).unwrap_or(0));
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }

    /// The run of view heap `key`, made on first use: every slot null, then
    /// the views of the copies numbered so far.
    fn run(&mut self, key: usize, heap: &HostHeap) -> Option<u32> {
        let host = std::ptr::from_ref(heap) as usize;
        if let Some(run) = self.runs.get(&key)
            && run.host == host
        {
            return Some(run.base);
        }
        // Held for the heap's life: not given back with a list's slots.
        // shortcut: the run of a released heap's predecessor at this key
        // went with that heap.
        let base = heap.dynamic(COPIES + 1)?;
        for slot in base..=base + COPIES {
            heap.null_raw_uav_at(slot);
        }
        for (n, copy) in self.copies.iter().enumerate() {
            if let Some(c) = copy {
                heap.raw_uav_at(base + 1 + n as u32, &c.visible.buffer, 0, c.visible.span().1);
            }
        }
        self.runs.insert(key, Run { host, base });
        Some(base)
    }

    /// Copies the shadow module no longer finds at their range are retired:
    /// directory entries naming them emptied, views made null, numbers
    /// freed. Only while no work of ours runs (at submission).
    fn retire_released(&mut self) {
        let frame = crate::frame::current();
        let mut gone = Vec::new();
        for (n, copy) in self.copies.iter().enumerate() {
            let Some(c) = copy else { continue };
            let (first, len) = c.visible.span();
            let live = crate::shadow::find(first, len).is_some_and(|s| c.owner.ptr_eq(&Arc::downgrade(&s)));
            if !live {
                gone.push(n as u32);
            }
        }
        for n in gone {
            let Some(c) = self.copies[n as usize].take() else { continue };
            let (first, len) = c.visible.span();
            for page in touched_pages(first, len) {
                let entry = self.entry(page);
                // SAFETY: inside the directory's mapping; nothing of ours
                // runs on the GPU.
                unsafe {
                    if entry.read_volatile() == n + 1 {
                        entry.write_volatile(0);
                        entry.add(1).write_volatile(0);
                    }
                }
            }
            self.write_view(n, None);
            self.free.push((n, frame));
            stats(|s| s.retired += 1);
        }
    }
}

/// Before a draw or dispatch whose shader reads title memory by address,
/// with view heap `key` (`heap` its host heap) bound: pages its shaders
/// found empty before and what its root arguments name get their copies
/// (`addresses`: its root views; `constants`: its root constants, see the
/// module), the heap has the copies' views, and the list has its faults and
/// brings every copy up to date at submission; with `writes`, every copy
/// numbered since the list last declared is declared written. None if the
/// host cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    key: usize,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = memory();
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
    }
    for page in std::mem::take(&mut m.pending) {
        m.map(u64::from(page) * PAGE);
    }
    for &address in addresses {
        m.name(address);
    }
    for address in constant_addresses(constants) {
        m.name(address);
    }
    let base = m.run(key, heap)?;
    let at = std::ptr::from_ref(list) as usize;
    // Once per list (and after each reset): its faults, the directory kept,
    // the copies brought up to date at submission.
    const BIND: usize = usize::MAX - 0x6d;
    if list.first_use(BIND) {
        let host = host()?;
        let bytes = u64::from(MEMORY_FAULTS) * 4;
        let (buffer, data) = host.shared_buffer(bytes)?;
        crate::host::name(&buffer, || "title memory by address: faults".into());
        // SAFETY: a fresh mapping of at least that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        list.keep(&buffer);
        list.keep(&m.directory.0);
        list.defer_memory(key);
        let words = (data as usize, MEMORY_FAULTS as usize);
        list.defer_missed(move || {
            // SAFETY: the list's mapped faults; its run is over.
            let faults = unsafe { std::slice::from_raw_parts(words.0 as *const u32, words.1) };
            let missed = faulted(faults);
            if missed {
                stats(|s| s.withheld_lists += 1);
            }
            missed
        });
        list.defer_after(move || {
            // SAFETY: the list's mapped faults; its run is over (each
            // submission of the list).
            let faults = unsafe { std::slice::from_raw_parts_mut(words.0 as *mut u32, words.1) };
            let pages = take_fault_pages(faults);
            if pages.is_empty() {
                return;
            }
            stats(|s| {
                s.faults += pages.len() as u64;
                s.first_fault.get_or_insert(u64::from(pages[0]) * PAGE);
            });
            if let Some(m) = memory().as_mut() {
                m.pending.extend(pages);
            }
        });
        m.lists.insert(at, (gpu, 0));
    }
    let (faults, declared) = *m.lists.get(&at)?;
    if writes && declared < m.numbered {
        for (copy, &since) in m.copies.iter().zip(&m.since) {
            if let Some(c) = copy
                && since > declared
            {
                let (start, len) = c.visible.span();
                c.visible.gpu_writes(list, start, len);
            }
        }
        m.lists.insert(at, (faults, m.numbered));
    }
    Some(Ready {
        directory: m.directory.2,
        faults,
        base,
    })
}

/// At submission, for a list that uses title memory by address
/// (`HostList::defer_memory`; `key` its view heap): copies released since
/// are retired (once per frame), and `add` takes the range of every
/// numbered copy to bring up to date (the batch does each once per
/// submission until a step writes title memory). Only while no work of
/// ours runs on the GPU.
pub(crate) fn prepare(key: usize, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let _ = key;
    let copies: Vec<Arc<Shadow>> = {
        let mut guard = memory();
        let Some(m) = guard.as_mut() else { return };
        let frame = crate::frame::current();
        if m.checked != frame {
            m.checked = frame;
            m.retire_released();
        }
        m.copies.iter().flatten().map(|m| m.visible.clone()).collect()
    };
    for c in &copies {
        let (start, len) = c.span();
        add(c, start as usize, len as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_copy_enters_the_pages_it_holds_whole() {
        // A 4 MiB window: its 64 pages.
        assert_eq!(whole_pages(0x40_0000, 0x40_0000), 0x40..0x80);
        // A buffer from 0x41_8000 for 0x3_0000: pages 0x42 and 0x43 whole,
        // 0x41 and 0x44 in part.
        assert_eq!(whole_pages(0x41_8000, 0x3_0000), 0x42..0x44);
        assert_eq!(touched_pages(0x41_8000, 0x3_0000), 0x41..0x45);
        // Less than a page, inside one: none whole, one touched.
        assert!(whole_pages(0x41_8000, 0x100).is_empty());
        assert_eq!(touched_pages(0x41_8000, 0x100), 0x41..0x42);
        // The directory's last entry is never filled.
        let last = u64::from(MEMORY_PAGES) - 1;
        assert_eq!(whole_pages((last - 1) * PAGE, 4 * PAGE), last - 1..last);
        assert_eq!(touched_pages(last * PAGE, PAGE), last..last);
        assert_eq!(touched_pages(0, 0), 0..0);
    }

    #[test]
    fn fault_pages_skip_the_sink_and_are_cleared() {
        let mut words = vec![0u32; 8];
        words[0] = 7; // the other lanes' sink
        words[2] = 0x1234;
        words[5] = 0x1234;
        words[6] = 0x40;
        assert!(faulted(&words));
        assert_eq!(take_fault_pages(&mut words), [0x1234, 0x40]);
        assert!(!faulted(&words));
        assert_eq!(words[0], 7);
    }

    #[test]
    fn constants_name_addresses_two_ways() {
        // A low dword and bits 32 to 39; then one in 256-byte units.
        let found = constant_addresses(&[0x4000_0000, 0x12, 0x0200_0000]);
        assert!(found.contains(&0x12_4000_0000));
        assert!(found.contains(&(0x4000_0000 << 8)));
        assert!(found.contains(&(0x0200_0000 << 8)));
        // Small values are no address.
        assert!(constant_addresses(&[0x10, 0x1, 0xffff]).is_empty());
    }
}
