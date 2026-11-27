//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, Weak};
use std::time::Instant;

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let (asked_kind, asked_protect) = (kind, protect);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if base.is_null() {
        // The census keeps what the title asked for (`VICTORY_CENSUS`).
        #[cfg(windows)]
        victory_gpu::title_memory_reserved(at as usize, size, asked_kind, flags, asked_protect);
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section (or several, [`section_pages`]), its pages
// are 64 KiB slices of it, and a mapping is a view of consecutive slices of
// one section in place of a host placeholder.

const PAGE: usize = 64 << 10;

/// The most pages one section holds (`VICTORY_SECTION_MB=<n>`: n MiB); 0
/// (unset, 0 or not a number): one section for each allocation, however
/// big, as before.
///
/// Why (an xperf trace in fork issue #1089, RyanCase94; fork PR #1274;
/// this code is ours): the title asks for about 6.1 GiB of pages at once
/// and maps them 64 KiB at a time. Windows' worker that rebuilds large
/// pages holds a section's lock while it walks that section's views, and
/// every map, commit and unmap of any page of the section waits for it.
/// With one section that is all of the title's physical memory: their
/// remapping time in a cutscene was 32-65 s, and some 11,000 calls a
/// minute took 1 ms or more. With sections of 2 MiB: 2.5 s, and 31 calls.
///
/// What it costs: an allocation's pages lie in several sections, so a run
/// of pages over a section's end is two views, and a DirectStorage
/// destination over one is written through a buffer ([`page_window`]).
fn section_pages() -> usize {
    static PAGES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGES.get_or_init(|| {
        let pages = pages_per_section(std::env::var("VICTORY_SECTION_MB").ok().as_deref());
        if pages > 0 {
            tracing::info!(
                mb = pages * PAGE / MIB,
                pages,
                "physical pages: an allocation is made of sections of at most this many MiB (VICTORY_SECTION_MB)"
            );
        }
        pages
    })
}

/// [`section_pages`] for a value of `VICTORY_SECTION_MB`.
fn pages_per_section(mb: Option<&str>) -> usize {
    mb.and_then(|mb| mb.trim().parse::<usize>().ok())
        .map_or(0, |mb| mb.saturating_mul(MIB / PAGE))
}

struct Section {
    handle: HANDLE,
    /// A section of placement ([`place_on`]): the title address its first
    /// byte stands for (its window's first address). `None`: an
    /// allocation's own section.
    window: Option<usize>,
    /// Whether the GPU side is told when it closes: sections that come and
    /// go with the title's allocations (`VICTORY_SECTION_MB`), and with the
    /// placement on (an allocation's section closes as soon as all its
    /// pages are placed elsewhere; a section of placement when no page or
    /// view is left in it).
    told: bool,
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        // No title view shows the section any more (views hold it). Its
        // handle's number may come round again for a new section, which
        // with small sections is common: the GPU side lets its own view of
        // this one go first. One section per allocation leaves it as it
        // was.
        if self.told {
            victory_gpu::title_memory_section_closed(self.handle as usize);
        }
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

// VIcky: physical pages placed at their first mapping (`VICTORY_PAGE_PLACE`,
// roadmap B2; lot 4's probe P4 simulated it, `victory_gpu::inplace::lot4`).

/// `VICTORY_PAGE_PLACE=1` (off by default): each physical page gets its
/// place in a section when the title maps it for the first time, at the
/// offset its address names, instead of the place its allocation gave it.
///
/// Why: a page's place is fixed today when the title allocates it
/// ([`make_pages`]), and the title maps its pages in an order of its own,
/// so a buffer it maps at consecutive addresses shows pages from scattered
/// places. One host heap can only be opened on one run of one section (a
/// heap over two views side by side is refused, `E_INVALIDARG`, measured on
/// the software adapter), so `VICTORY_INPLACE_CONSTANTS` turns down the
/// constants heap as "view memory from several places" (174 MB in 88 runs
/// in the boat save's probe run with `VICTORY_SECTION_MB=2`). P4 simulated
/// this placement: 113,446 pages placed at their address and none
/// elsewhere, and both title buffers of 64 MB or more (238 MB) one run.
///
/// How: the title's address space is cut into windows, each a section of
/// its own made when a page is first placed in it ([`window_of`]): below
/// 4 GiB one window of 4 GiB, so that everything the title maps there (its
/// 32-bit graphics memory, where `VICTORY_INPLACE_CONSTANTS` looks for the
/// constants heap) is one section and a buffer mapped in order one run of
/// it; above, windows of `VICTORY_SECTION_MB` MiB if set (its loading gain
/// kept: Windows' large-page worker holds a section's lock while it walks
/// that section's views), else 4 GiB. A page first mapped at `at` takes the
/// place `at` of its window's section, and keeps it wherever and however
/// often it is mapped afterwards (a second address is an alias of that
/// place, as before of its allocation's).
///
/// - A place is one page's: if another page has the place first (mapped
///   there before, still allocated or still shown), the page keeps the
///   place its allocation gave it, as without the switch (counted).
/// - A page written before its first mapping (a DirectStorage destination
///   not mapped yet) keeps its allocation's place too, with its bytes
///   (counted).
/// - A freed page's place is given back when no view shows it any more; a
///   page placed there later reads zero at its first mapping (zeroed then,
///   as a page of a new section reads).
///
/// What it costs: off, a look at whether any page waits for its place, in
/// the mapping and freeing calls. On, a few table lookups per page at its
/// first mapping, a section made per window used (some thousands of 2 MiB
/// ones with `VICTORY_SECTION_MB=2`, each a `CreateFileMapping` of a
/// reserved section: no memory until mapped and committed, as now), and a
/// zeroing of 64 KiB for a place given again. An allocation's own sections
/// are still made: they hold the pages that are not placed. The GPU side
/// says the line [`placement_line`] (`victory_gpu::title_memory_placement`)
/// every ten seconds with the constants heap's state in place.
fn place_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = place_mode(std::env::var("VICTORY_PAGE_PLACE").ok().as_deref());
        if on {
            tracing::info!(
                high_mb = place_high() / MIB,
                "physical pages are placed at their first mapping (VICTORY_PAGE_PLACE=1): each takes the place its address names in a \
                 section of placement (below 4 GiB one of 4096 MB; above, windows of high_mb MB), so pages mapped at consecutive \
                 addresses are one run of a section"
            );
            victory_gpu::title_memory_placement(placement_line);
        }
        on
    })
}

/// [`place_on`] for a value of `VICTORY_PAGE_PLACE`: only `1` is on.
fn place_mode(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// Title addresses below this are its 32-bit memory: one window.
const LOW_WINDOW: usize = 1 << 32;

/// The bytes of a window of placement above 4 GiB: `VICTORY_SECTION_MB`'s
/// sections if set, else 4 GiB.
fn place_high() -> usize {
    match section_pages() {
        0 => LOW_WINDOW,
        per => per * PAGE,
    }
}

/// The window of placement that holds title address `at`, as (first
/// address, bytes): below 4 GiB `[0, 4 GiB)`, above it windows of `high`
/// bytes (a multiple of [`PAGE`]) from 0 on.
fn window_of(at: usize, high: usize) -> (usize, usize) {
    if at < LOW_WINDOW {
        (0, LOW_WINDOW)
    } else {
        (at / high * high, high)
    }
}

/// The placement's state, in the page table ([`Physical`]).
#[derive(Default)]
struct Place {
    /// Pages that may still be placed (allocated with the switch on, never
    /// mapped nor written yet), with the bytes of their windows above
    /// 4 GiB ([`place_high`]; tests give smaller ones).
    unplaced: HashMap<usize, usize>,
    /// Sections of placement by window (first address, bytes).
    windows: HashMap<(usize, usize), Window>,
    /// Places taken, by address: the page that has it, or `None` once that
    /// page is freed while a view still shows it (the place is given back
    /// when the last such view goes).
    taken: BTreeMap<usize, Option<usize>>,
    /// Places given again whose page has not been mapped since: zeroed at
    /// its first mapping, whichever view it is.
    zero: HashSet<usize>,
}

/// A window's section, held weakly (it goes when no page or view holds
/// it), and the places of it ever given (a bit each).
struct Window {
    section: Weak<Section>,
    given: Vec<u64>,
}

/// The placement's counts since the start ([`placement_line`]): pages placed
/// at their address; pages kept at their allocation's place because their
/// address's place was taken first, because they were written before their
/// first mapping, because no section of placement could be made; places
/// given back, given again (zeroed); sections of placement made, their
/// bytes.
static PLACED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// The pages of `pages` mapped at `base` for the first time take the
/// places their addresses name ([`place_on`]). Before the call's views are
/// mapped, after what was there was unmapped (a place given back by that
/// can be taken again here).
fn place_first(physical: &mut Physical, base: usize, pages: &[usize]) {
    for (i, page) in pages.iter().enumerate() {
        let Some(high) = physical.place.unplaced.remove(page) else {
            continue;
        };
        let at = base + i * PAGE;
        if physical.place.taken.contains_key(&at) {
            PLACED[1].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let (first, bytes) = window_of(at, high);
        let index = (at - first) / PAGE;
        let Some((section, again)) = window_section(&mut physical.place, first, bytes, index)
        else {
            PLACED[3].fetch_add(1, Ordering::Relaxed);
            continue;
        };
        if again {
            physical.place.zero.insert(at);
            PLACED[5].fetch_add(1, Ordering::Relaxed);
        }
        physical.place.taken.insert(at, Some(*page));
        // The allocation's place goes: never mapped nor written, it holds
        // nothing (its section closes once all its pages have gone).
        physical.pages.insert(*page, Slice { section, index });
        PLACED[0].fetch_add(1, Ordering::Relaxed);
    }
}

/// The section of the window `[first, first + bytes)`, made if it has none
/// (or its last went), and whether its place `index` was given before
/// (now marked given). `None` if the host refuses the section.
fn window_section(
    place: &mut Place,
    first: usize,
    bytes: usize,
    index: usize,
) -> Option<(Arc<Section>, bool)> {
    let key = (first, bytes);
    let section = match place.windows.get(&key).and_then(|w| w.section.upgrade()) {
        Some(section) => section,
        None => {
            // As an allocation's ([`make_pages`]): reserved, its pages
            // committed when mapped.
            // SAFETY: plain section creation.
            let handle = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
                    (bytes as u64 >> 32) as u32,
                    bytes as u32,
                    std::ptr::null(),
                )
            };
            if handle.is_null() {
                if PLACED[3].load(Ordering::Relaxed) < 8 {
                    tracing::warn!(
                        first = format_args!("{first:#x}"),
                        mb = bytes / MIB,
                        error = %std::io::Error::last_os_error(),
                        "no section of placement: the page keeps its allocation's place (VICTORY_PAGE_PLACE)"
                    );
                }
                return None;
            }
            PLACED[6].fetch_add(1, Ordering::Relaxed);
            PLACED[7].fetch_add(bytes as u64, Ordering::Relaxed);
            MADE[0].fetch_add(1, Ordering::Relaxed);
            MADE[1].fetch_add(bytes as u64, Ordering::Relaxed);
            victory_gpu::title_memory_section(handle as usize, bytes as u64, on_wine());
            let section = Arc::new(Section {
                handle,
                window: Some(first),
                told: true,
            });
            // A new section's pages read zero: no place of it given yet.
            let given = vec![0; (bytes / PAGE).div_ceil(64)];
            place.windows.insert(
                key,
                Window {
                    section: Arc::downgrade(&section),
                    given,
                },
            );
            section
        }
    };
    let given = &mut place.windows.get_mut(&key)?.given;
    let bit = 1u64 << (index % 64);
    let again = given[index / 64] & bit != 0;
    given[index / 64] |= bit;
    Some((section, again))
}

/// A placed page's place `at` lost a holder: the page was freed (`freed`),
/// or the last view that showed it went. Given back when both are so.
fn place_released(place: &mut Place, at: usize, freed: bool, shown: bool) {
    match place.taken.get_mut(&at) {
        Some(owner) if freed && shown => *owner = None,
        Some(owner) if (freed || owner.is_none()) && !shown => {
            place.taken.remove(&at);
            place.zero.remove(&at);
            PLACED[4].fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

/// The placement's line: what is placed now (bytes, in how many runs of
/// consecutive addresses, the largest), and the counts since the start.
/// The GPU side says it with the constants heap's state in place, from its
/// submitting thread: `None` (said next time) while a mapping call holds
/// the page table, which that thread does not wait for.
fn placement_line() -> Option<String> {
    let (mut live, mut freed, mut runs) = (0u64, 0u64, 0u64);
    let (mut largest, mut run, mut next) = ((0usize, 0usize), (0usize, 0usize), usize::MAX);
    {
        let physical = match PHYSICAL.try_lock() {
            Ok(held) => held,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        for (&at, owner) in &physical.place.taken {
            if owner.is_none() {
                freed += 1;
                continue;
            }
            live += 1;
            if at != next {
                runs += 1;
                run = (at, 0);
            }
            run.1 += PAGE;
            largest = largest.max((run.1, run.0));
            next = at + PAGE;
        }
    }
    let count = |i: usize| PLACED[i].load(Ordering::Relaxed);
    Some(format!(
        "physical pages placed at their first mapping (VICTORY_PAGE_PLACE=1): {} MB placed now in {runs} runs of consecutive \
         addresses (the largest {} MB at {:#x}), {freed} places of freed pages still shown; since the start {} pages placed at \
         their address, {} kept at their allocation's place ({} their place taken first, {} written before their first mapping, \
         {} no section of placement), {} places given back, {} given again (zeroed); {} sections of placement made, {} MB \
         (below 4 GiB one of 4096 MB, above windows of {} MB)",
        (live * PAGE as u64) >> 20,
        largest.0 >> 20,
        largest.1,
        count(0),
        count(1) + count(2) + count(3),
        count(1),
        count(2),
        count(3),
        count(4),
        count(5),
        count(6),
        count(7) >> 20,
        place_high() / MIB,
    ))
}

/// What the title's mapping calls cost, since the start (for
/// `VICTORY_SECTION_MB`'s A/B): by [`Call`], the calls, their microseconds
/// from entry to return, those of 1 ms or more, and the longest.
static CALLS: [[AtomicU64; 4]; 3] = [const { [const { AtomicU64::new(0) }; 4] }; 3];
/// Of those calls, the time in Windows' own calls, where the section lock
/// is waited for (a placeholder split with a view mapped in it, a commit,
/// an unmap): microseconds, and those single calls of 1 ms or more.
static SYSTEM: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Sections made for physical pages, and their bytes.
static MADE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// DirectStorage destinations turned down by [`page_window`] because their
/// pages are more than one run (of several sections, allocations, or out
/// of order), so written through a buffer; and their bytes.
static BUFFERED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The title's mapping calls, as [`CALLS`] counts them.
#[derive(Clone, Copy)]
enum Call {
    /// `XMemMapPhysicalPages` with pages (each page of a scatter call too).
    Map,
    /// The same without (an unmap).
    Unmap,
    /// The GPU's page-mapping copies ([`map_alias`]), one a call.
    Alias,
}

/// Times a mapping call; made first in it, so dropped last, after the
/// call's locks are let go. Says the totals now and then.
struct Timed(Call, Instant);

impl Timed {
    fn new(call: Call) -> Timed {
        Timed(call, Instant::now())
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        let us = self.1.elapsed().as_micros() as u64;
        let at = &CALLS[self.0 as usize];
        at[0].fetch_add(1, Ordering::Relaxed);
        at[1].fetch_add(us, Ordering::Relaxed);
        if us >= 1000 {
            at[2].fetch_add(1, Ordering::Relaxed);
        }
        at[3].fetch_max(us, Ordering::Relaxed);
        say_costs();
    }
}

/// Runs one of Windows' calls on a title mapping, timed ([`SYSTEM`]).
fn system<R>(call: impl FnOnce() -> R) -> R {
    let started = Instant::now();
    let result = call();
    let us = started.elapsed().as_micros() as u64;
    SYSTEM[0].fetch_add(us, Ordering::Relaxed);
    if us >= 1000 {
        SYSTEM[1].fetch_add(1, Ordering::Relaxed);
    }
    result
}

/// The totals line, at most every ten seconds (from the end of a mapping
/// call, so while the title maps).
fn say_costs() {
    static FIRST: LazyLock<Instant> = LazyLock::new(Instant::now);
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = FIRST.elapsed().as_millis() as u64;
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 10_000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    tracing::info!("{}", costs_line());
}

fn costs_line() -> String {
    let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let ms = |us: u64| us as f64 / 1e3;
    let call = |call: Call| {
        let [n, us, slow, longest] = &CALLS[call as usize];
        format!(
            "{} in {:.1} ms ({} of 1 ms or more, the longest {:.1} ms)",
            load(n),
            ms(load(us)),
            load(slow),
            ms(load(longest))
        )
    };
    format!(
        "physical pages, since the start: maps {}; unmaps {}; page-mapping copies {}; in Windows' calls {:.1} ms \
         ({} of 1 ms or more); {} sections made, {} MB ({}); {} DirectStorage destinations of more than one run of \
         pages written through a buffer, {:.1} MB",
        call(Call::Map),
        call(Call::Unmap),
        call(Call::Alias),
        ms(load(&SYSTEM[0])),
        load(&SYSTEM[1]),
        load(&MADE[0]),
        load(&MADE[1]) >> 20,
        match section_pages() {
            0 => "one section for each allocation".to_string(),
            pages => format!("VICTORY_SECTION_MB={}", pages * PAGE / MIB),
        },
        load(&BUFFERED[0]),
        load(&BUFFERED[1]) as f64 / 1e6,
    )
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    /// The key in `Physical::shown` of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }

    /// The place ([`place_on`]) of the page `k` pages after this one: the
    /// title address it stands for, if it is a section of placement's.
    fn place(&self, k: usize) -> Option<usize> {
        self.section
            .window
            .map(|first| first + (self.index + k) * PAGE)
    }
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Where each page shows: by its section and its index there, the
    /// address of that page in every view that shows it now ([`shown_at`];
    /// as upstream's c75fdba keeps it). The title maps pages where the
    /// card reads them and has them filled afterwards by DirectStorage,
    /// which writes through views of our own.
    shown: HashMap<(usize, usize), Vec<usize>>,
    /// Pages placed at their first mapping ([`place_on`]).
    place: Place,
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    if n.checked_mul(PAGE).is_none_or(|size| size == 0) {
        return 0;
    }
    let numbers = match make_pages(n, section_pages(), place_on().then(place_high)) {
        Ok(numbers) => numbers,
        Err(error) => {
            tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
            // SAFETY: checked non-null.
            unsafe { count.write(0) };
            return 0;
        }
    };
    tracing::debug!(
        n,
        sections = n.div_ceil(match section_pages() {
            0 => n,
            per => per,
        }),
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    for (index, number) in numbers.into_iter().enumerate() {
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Makes `n` pages (`n` > 0, `n * PAGE` in range) in sections of `per`
/// pages at most (0: all in one), and enters them in the page table: their
/// numbers, in order, as one allocation's. The host's error, with nothing
/// made, if it refuses a section. With `place` (the bytes of a window of
/// placement above 4 GiB, [`place_on`]) the pages take their places at
/// their first mapping; the sections made here hold them until then, and
/// for good those that are not placed.
fn make_pages(n: usize, per: usize, place: Option<usize>) -> std::io::Result<Vec<usize>> {
    let per = if per == 0 { n } else { per };
    let mut sections: Vec<(Arc<Section>, usize)> = Vec::with_capacity(n.div_ceil(per));
    let mut made = 0;
    while made < n {
        let here = per.min(n - made);
        let size = here * PAGE;
        // A pagefile-backed section of `size` bytes, its pages committed
        // when first mapped or written (`commit`): the title asks for 6.1
        // GiB at once, which a busy host often cannot commit (error 1450).
        // SAFETY: plain section creation.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                // Under Wine a committed section is a sparse file, free
                // until touched, and committing a reserved one page by page
                // costs a wineserver request each that scans every range
                // committed so far: commit it whole there.
                PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
                (size as u64 >> 32) as u32,
                size as u32,
                std::ptr::null(),
            )
        };
        if handle.is_null() {
            // Read before the sections made so far are closed (when
            // `sections` goes).
            return Err(std::io::Error::last_os_error());
        }
        sections.push((
            Arc::new(Section {
                handle,
                window: None,
                told: section_pages() > 0 || place.is_some(),
            }),
            here,
        ));
        made += here;
    }
    for (section, here) in &sections {
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add((here * PAGE) as u64, Ordering::Relaxed);
        // For the GPU side's in-place census: only a committed section can
        // be opened as a host heap.
        victory_gpu::title_memory_section(section.handle as usize, (here * PAGE) as u64, on_wine());
    }
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    let mut numbers = Vec::with_capacity(n);
    for (section, here) in sections {
        for index in 0..here {
            let number = first + numbers.len() * FRAMES;
            let slice = Slice {
                section: section.clone(),
                index,
            };
            physical.pages.insert(number, slice);
            if let Some(high) = place {
                physical.place.unplaced.insert(number, high);
            }
            numbers.push(number);
        }
    }
    Ok(numbers)
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        let page = unsafe { *pages.add(i) };
        let Some(slice) = physical.pages.remove(&page) else {
            continue;
        };
        // The placement's (`place_on`; nothing to do without it): a page
        // never mapped is placed no more; a placed page's place is given
        // back, now or when the last view that shows it goes.
        if !physical.place.unplaced.is_empty() {
            physical.place.unplaced.remove(&page);
        }
        if let Some(at) = slice.place(0) {
            let shown = physical.shown.contains_key(&slice.key(0));
            place_released(&mut physical.place, at, true, shown);
        }
    }
    1
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len };
        commit(at.Value as usize, len).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early. `write`: the caller writes
/// the bytes, so pages not mapped yet keep the places they have
/// ([`place_on`]: their bytes are there).
fn runs(pages: &[usize], offset: usize, len: usize, write: bool) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let mut physical = physical();
    if write && !physical.place.unplaced.is_empty() {
        for page in pages {
            if physical.place.unplaced.remove(page).is_some() {
                PLACED[2].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    // A place given again whose first mapping failed is zeroed before it
    // is written, not after (by `map`), which would wipe the bytes.
    if write && !physical.place.zero.is_empty() {
        for page in pages {
            let Some(slice) = physical.pages.get(page).cloned() else {
                continue;
            };
            if slice.place(0).is_some_and(|at| physical.place.zero.remove(&at)) {
                View::map(&slice, 1)?.bytes().fill(0);
            }
        }
    }
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len, false).is_some()
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len(), true) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one section (how titles usually hand them out; an
/// allocation is one section unless [`section_pages`] cuts it); `None`
/// otherwise (pages of several sections or allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len, true)?;
    let [(slice, n)] = runs.as_slice() else {
        if runs.len() > 1 {
            BUFFERED[0].fetch_add(1, Ordering::Relaxed);
            BUFFERED[1].fetch_add(len as u64, Ordering::Relaxed);
        }
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        within: offset % PAGE,
        len,
    })
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it (the GPU side's guards fault only on a write at the
/// guarded address itself). Empty if no view shows them. Byte for byte,
/// not by whole pages: the title packs several small textures into one
/// page and fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let physical = physical();
        for (i, page) in touched.iter().enumerate() {
            let Some(addresses) = physical.pages.get(page).and_then(|slice| physical.shown.get(&slice.key(0))) else {
                continue;
            };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            pieces.extend(addresses.iter().map(|&at| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// Bytes `[offset, offset + len)` of `pages` have been written through a
/// view of our own: the GPU side is told at every title address that
/// shows them (`victory_gpu::title_memory_written`; what it does with it
/// is `VICTORY_PAGE_WRITES`' to say, and it counts either way). Call when
/// the bytes are there, and without the page table's lock ([`shown_at`]
/// takes and leaves it).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    for (at, bytes) in shown_at(pages, offset, len) {
        victory_gpu::title_memory_written(at, bytes, true);
    }
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let _timed = Timed::new(if pages.is_null() { Call::Unmap } else { Call::Map });
    let base = address as usize;
    let mut physical = physical();
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`).
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, base, n * PAGE);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    // Pages mapped for the first time take their places (`place_on`; none
    // wait for one without it).
    if !physical.place.unplaced.is_empty() {
        place_first(&mut physical, base, pages);
    }
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section (a run of
        // an allocation's pages ends at its section's end).
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(&mut physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let _timed = Timed::new(Call::Alias);
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let mut physical = physical();
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, to, len);
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let at = from + k;
        let Some(slice) = physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
        else {
            return false;
        };
        if !map(&mut physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = system(|| unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    });
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    if !system(|| commit(base, len)) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    // A place given again ([`place_on`]) reads zero at its page's first
    // mapping, as a page of a new section does; nothing has seen the view
    // yet.
    if !physical.place.zero.is_empty() {
        for k in 0..len.div_ceil(PAGE) {
            if slice
                .place(k)
                .is_some_and(|at| physical.place.zero.remove(&at))
            {
                // SAFETY: a committed read-write page of the view just
                // mapped.
                unsafe { ((base + k * PAGE) as *mut u8).write_bytes(0, PAGE) };
            }
        }
    }
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    // What the range shows now, for the GPU side's in-place census.
    victory_gpu::title_memory_view(
        base,
        len,
        slice.section.handle as usize,
        (slice.index * PAGE) as u64,
    );
    for k in 0..len.div_ceil(PAGE) {
        physical.shown.entry(slice.key(k)).or_default().push(base + k * PAGE);
    }
    physical.views.insert(base, (len, slice));
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    // Places ([`place_on`]) whose last view went here: the place and the
    // page's key in `shown`.
    let mut unshown: Vec<(usize, (usize, usize))> = Vec::new();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        for k in 0..size.div_ceil(PAGE) {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                    if let Some(place) = slice.place(k) {
                        unshown.push((place, key));
                    }
                }
            }
        }
        victory_gpu::title_memory_remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        system(|| unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        });
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
    // A freed page's place is given back when no view shows it any more,
    // the parts of views mapped again above included.
    for (place, key) in unshown {
        if !physical.shown.contains_key(&key) {
            place_released(&mut physical.place, place, false, false);
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every
    // 1,024th: "used" is all the process has committed, the layer's own
    // memory and the card's included (27 GB in run 60's gameplay against a
    // limit of 13.5 GB). A title that budgets its streaming from these two
    // figures believes it is far over its limit.
    // shortcut: the figure is not yet the title's own memory alone (its
    // heaps and the pages it mapped); count those if this line shows the
    // title asking.
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading();
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len, false).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// `VICTORY_SECTION_MB` in whole MiB; anything else is one section an
    /// allocation, as is a size of at least the allocation's.
    #[test]
    fn section_sizes_come_from_the_switch() {
        assert_eq!(pages_per_section(None), 0);
        assert_eq!(pages_per_section(Some("")), 0);
        assert_eq!(pages_per_section(Some("0")), 0);
        assert_eq!(pages_per_section(Some("two")), 0);
        assert_eq!(pages_per_section(Some("-2")), 0);
        assert_eq!(pages_per_section(Some("1")), 16);
        assert_eq!(pages_per_section(Some(" 2 ")), 32);
        for per in [0, 7, 9] {
            let pages = make_pages(7, per, None).expect("a section");
            let groups = runs(&pages, 0, 7 * PAGE, false).expect("known pages");
            assert_eq!(groups.len(), 1, "sections of {per} pages");
            assert_eq!((groups[0].0.index, groups[0].1), (0, 7));
            assert_eq!(XMemFreePhysicalPages(7, pages.as_ptr()), 1);
        }
    }

    /// An allocation cut into sections (`VICTORY_SECTION_MB`): five pages
    /// in sections of two. Its numbers are one allocation's; runs, windows
    /// and views end at a section's end; bytes written over the bounds,
    /// through views of ours or the title's, land in the right pages; and
    /// what an unmapping leaves, and a page-mapping copy, show their own
    /// section's pages.
    #[test]
    fn an_allocation_in_several_sections_is_mapped_and_written_across_their_bounds() {
        let pages = make_pages(5, 2, None).expect("sections");
        assert!(pages.windows(2).all(|w| w[1] == w[0] + FRAMES), "numbered as one allocation");
        let groups = runs(&pages, 0, 5 * PAGE, false).expect("known pages");
        assert_eq!(groups.iter().map(|&(_, n)| n).collect::<Vec<_>>(), [2, 2, 1]);
        assert!(groups.iter().all(|(slice, _)| slice.index == 0), "each run from its section's start");
        let sections: std::collections::HashSet<usize> =
            groups.iter().map(|(slice, _)| slice.section.handle as usize).collect();
        assert_eq!(sections.len(), 3);
        // Inside one section in place; over a bound through a buffer.
        assert!(page_window(&pages, 2 * PAGE + 7, PAGE).is_some());
        let buffered = BUFFERED[0].load(Ordering::Relaxed);
        assert!(page_window(&pages, PAGE, 2 * PAGE).is_none());
        assert!(BUFFERED[0].load(Ordering::Relaxed) > buffered);
        let data: Vec<u8> = (0..3 * PAGE + 100).map(|i| (i % 251) as u8 ^ 0x5a).collect();
        let offset = PAGE / 2;
        assert!(write_physical(&pages, offset, &data));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + data.len()], data[..]);
        assert!(seen[offset + data.len()..].iter().all(|&b| b == 0), "after");

        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        let (a, b) = (reserve(5), reserve(2));
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        let views_in = |p: *mut c_void, n: usize| {
            let start = p as usize;
            physical().views.range(start..start + n * PAGE).map(|(_, (len, _))| len / PAGE).collect::<Vec<_>>()
        };
        // All five in one call: a view per section.
        assert_eq!(XMemMapPhysicalPages(a, 5, pages.as_ptr()), a);
        assert_eq!(views_in(a, 5), [2, 2, 1]);
        // SAFETY: all five pages are mapped read-write at `a`.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(a.cast::<u8>(), 5 * PAGE), &seen[..]);
            // The title writes over both bounds.
            at(a, 2).cast::<u8>().sub(1).write_bytes(0xee, 2);
            at(a, 4).cast::<u8>().sub(1).write_bytes(0xdd, 2);
        }
        assert_eq!(page_bytes(pages[1])[PAGE - 1], 0xee);
        assert_eq!(page_bytes(pages[2])[0], 0xee);
        assert_eq!(page_bytes(pages[3])[PAGE - 1], 0xdd);
        assert_eq!(page_bytes(pages[4])[0], 0xdd);
        // Where the pages show, over a bound: one run of addresses.
        assert_eq!(shown_at(&pages, PAGE, 3 * PAGE), vec![(at(a, 1) as usize, 3 * PAGE)]);
        // The second section's second page unmapped: its first stays, itself.
        assert_eq!(XMemMapPhysicalPages(at(a, 3), 1, std::ptr::null()), at(a, 3));
        assert_eq!(views_in(a, 5), [2, 1, 1]);
        assert_eq!(query(at(a, 3)).host.State, MEM_RESERVE);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 0xee);
        // The GPU's page-mapping copies of the third section's page and of
        // the second's first.
        assert!(map_alias(b as u64, at(a, 4) as u64, PAGE as u64));
        assert!(map_alias(at(b, 1) as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: both mapped read-write.
        unsafe {
            assert_eq!(b.cast::<u8>().read(), 0xdd);
            assert_eq!(at(b, 1).cast::<u8>().read(), 0xee);
        }
        for (p, n) in [(a, 5), (b, 2)] {
            assert_eq!(XMemMapPhysicalPages(p, n, std::ptr::null()), p);
        }
        assert!(shown_at(&pages, 0, 5 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(5, pages.as_ptr()), 1);
    }

    /// The title's call cuts an allocation as the switch says (run the
    /// tests with `VICTORY_SECTION_MB=1` too: sections of 16 pages, and the
    /// GPU side told of each one closed as the pages go).
    #[test]
    fn the_title_call_cuts_by_the_switch() {
        let pages = allocate(33);
        let per = match section_pages() {
            0 => 33,
            per => per,
        };
        let expected: Vec<usize> = (0..33).step_by(per).map(|first| per.min(33 - first)).collect();
        let groups = runs(&pages, 0, 33 * PAGE, false).expect("known pages");
        assert_eq!(groups.iter().map(|&(_, n)| n).collect::<Vec<_>>(), expected);
        // The last byte of one section and the first of the next (or of
        // the same one).
        let a = XMemVirtualAlloc(std::ptr::null_mut(), 33 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 33, pages.as_ptr()), a);
        let bound = (per * PAGE).min(32 * PAGE);
        assert!(write_physical(&pages, bound - 1, &[17, 29]));
        // SAFETY: mapped read-write.
        unsafe {
            assert_eq!(a.cast::<u8>().add(bound - 1).read(), 17);
            assert_eq!(a.cast::<u8>().add(bound).read(), 29);
        }
        assert_eq!(XMemMapPhysicalPages(a, 33, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(33, pages.as_ptr()), 1);
    }

    /// The A/B line's counts: each kind of call, Windows' calls inside
    /// them, the sections made. (Other tests map at the same time, so the
    /// counts only grow by at least as much.)
    #[test]
    fn mapping_calls_and_sections_are_counted() {
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let calls: Vec<u64> = CALLS.iter().map(|c| load(&c[0])).collect();
        let (made, system) = (load(&MADE[0]), load(&SYSTEM[0]));
        let pages = make_pages(3, 2, None).expect("sections");
        assert!(load(&MADE[0]) >= made + 2);
        let a = XMemVirtualAlloc(std::ptr::null_mut(), 4 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!a.is_null());
        let fourth = a.wrapping_byte_add(3 * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        assert!(map_alias(fourth as u64, a as u64, PAGE as u64));
        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        for (call, before) in [Call::Map, Call::Unmap, Call::Alias].into_iter().zip(calls) {
            assert!(load(&CALLS[call as usize][0]) > before);
        }
        // Three views mapped and unmapped: Windows' calls took some time,
        // or at least were timed.
        assert!(load(&SYSTEM[0]) >= system);
        let line = costs_line();
        let how = if section_pages() == 0 { "(one section for each allocation)" } else { "(VICTORY_SECTION_MB=" };
        for part in ["maps ", "; unmaps ", "; page-mapping copies ", " sections made, ", how] {
            assert!(line.contains(part), "{line}");
        }
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    // VIcky: physical pages placed at their first mapping (`place_on`). The
    // tests make placeable pages themselves (`make_pages` with a window
    // size), so the switch itself stays off and other tests see nothing.

    /// Pages placeable in windows of `window` pages above 4 GiB.
    fn placeable(n: usize, window: usize) -> Vec<usize> {
        make_pages(n, 0, Some(window * PAGE)).expect("pages")
    }

    /// A placeholder of `n` pages above 4 GiB whose first address is a
    /// multiple of `align` pages (a window's first address).
    fn reserve_aligned(n: usize, align: usize) -> *mut c_void {
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: LOW_WINDOW as *mut c_void,
            HighestEndingAddress: ((1usize << 40) - 1) as *mut c_void,
            Alignment: align * PAGE,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // SAFETY: a reservation of the test's own; the parameter lives
        // through the call.
        let at = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null_mut(),
                n * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                &mut parameter,
                1,
            )
        };
        assert!(!at.is_null(), "{}", std::io::Error::last_os_error());
        assert_eq!(at as usize % (align * PAGE), 0);
        at
    }

    /// A placeholder of `n` pages anywhere, as titles reserve them.
    fn reserve(n: usize) -> *mut c_void {
        let at = XMemVirtualAlloc(
            std::ptr::null_mut(),
            n * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!at.is_null());
        at
    }

    /// The views in `[p, p + n pages)`, by their pages.
    fn views_in(p: *mut c_void, n: usize) -> Vec<usize> {
        let start = p as usize;
        physical()
            .views
            .range(start..start + n * PAGE)
            .map(|(_, (len, _))| len / PAGE)
            .collect()
    }

    /// Where a page lives now: its section's window (if a section of
    /// placement's) and its index there.
    fn place_of(page: usize) -> (Option<usize>, usize) {
        let physical = physical();
        let slice = physical.pages.get(&page).expect("known page");
        (slice.section.window, slice.index)
    }

    fn placed_count(i: usize) -> u64 {
        PLACED[i].load(Ordering::Relaxed)
    }

    /// The places taken in `[p, p + n pages)`.
    fn held(p: *mut c_void, n: usize) -> usize {
        physical()
            .place
            .taken
            .range(p as usize..p as usize + n * PAGE)
            .count()
    }

    /// The switch, and the windows: below 4 GiB one of 4 GiB, above windows
    /// of the size given from 0 on.
    #[test]
    fn placement_switch_and_windows() {
        assert!(!place_mode(None));
        assert!(!place_mode(Some("0")));
        assert!(!place_mode(Some("yes")));
        assert!(place_mode(Some("1")));
        assert!(place_mode(Some(" 1 ")));
        let two = 2 * MIB;
        assert_eq!(window_of(0x3de0_0000, two), (0, 1 << 32));
        assert_eq!(window_of(0, two), (0, 1 << 32));
        assert_eq!(window_of((1 << 32) - PAGE, two), (0, 1 << 32));
        assert_eq!(window_of(1 << 32, two), (1 << 32, two));
        assert_eq!(window_of((1 << 32) + 3 * MIB, two), ((1 << 32) + two, two));
        assert_eq!(window_of((5 << 32) + 7, 1 << 32), (5 << 32, 1 << 32));
        assert_eq!(
            window_of(0x2_85c2_0000, two),
            (0x2_85c2_0000 / two * two, two)
        );
    }

    /// Pages of one allocation mapped in another order at consecutive
    /// addresses: without placement one view each (scattered places), with
    /// it one view of one run of their window's section, at the offsets
    /// their addresses name. Mapped at a second address in the same order,
    /// one view again (an alias of the same places); a page keeps its place
    /// wherever it is mapped later; freed and unmapped, the places are given
    /// back.
    #[test]
    fn pages_first_mapped_at_consecutive_addresses_are_one_run() {
        let order = |p: &[usize]| [p[2], p[0], p[3], p[1]];
        // Without (the title's call, the switch off): four views.
        if !place_on() {
            let plain = allocate(4);
            let c = reserve(4);
            assert_eq!(XMemMapPhysicalPages(c, 4, order(&plain).as_ptr()), c);
            assert_eq!(views_in(c, 4), [1, 1, 1, 1]);
            assert_eq!(XMemMapPhysicalPages(c, 4, std::ptr::null()), c);
            assert_eq!(XMemFreePhysicalPages(4, plain.as_ptr()), 1);
        }

        let pages = placeable(4, 4);
        let placed = placed_count(0);
        let (a, b) = (reserve_aligned(4, 4), reserve(4));
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        let mapped = order(&pages);
        assert_eq!(XMemMapPhysicalPages(a, 4, mapped.as_ptr()), a);
        assert_eq!(views_in(a, 4), [4], "one view: one run of one section");
        assert!(placed_count(0) >= placed + 4);
        for (k, &page) in mapped.iter().enumerate() {
            assert_eq!(
                place_of(page),
                (Some(a as usize), k),
                "page {k} at its address"
            );
        }
        let groups = runs(&mapped, 0, 4 * PAGE, false).expect("known pages");
        assert_eq!(groups.len(), 1);
        // The title's bytes are the pages': through our own view of them.
        // SAFETY: mapped read-write.
        unsafe { at(a, 1).cast::<u8>().write(0x5c) };
        assert_eq!(page_bytes(pages[0])[0], 0x5c);
        // The same pages at a second address, in the same order: one view.
        assert_eq!(XMemMapPhysicalPages(b, 4, mapped.as_ptr()), b);
        assert_eq!(views_in(b, 4), [4]);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().read() }, 0x5c);
        // Unmapped and mapped elsewhere alone: the place stays.
        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert_eq!(
            XMemMapPhysicalPages(at(a, 3), 1, [pages[0]].as_ptr()),
            at(a, 3)
        );
        assert_eq!(place_of(pages[0]), (Some(a as usize), 1));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(a, 3).cast::<u8>().read() }, 0x5c);
        // Freed while shown: the places are held, then given back.
        let back = placed_count(4);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
        assert_eq!(held(a, 4), 4, "still shown at b (and a + 3 pages)");
        for (p, n) in [(at(a, 3), 1), (b, 4)] {
            assert_eq!(XMemMapPhysicalPages(p, n, std::ptr::null()), p);
        }
        assert_eq!(held(a, 4), 0, "no view left: given back");
        assert!(placed_count(4) >= back + 4);
    }

    /// A place another page took first: the page keeps its allocation's
    /// place, as without the switch, and is counted; the first page keeps
    /// its place while it lives, mapped or not.
    #[test]
    fn a_place_taken_first_keeps_the_allocations_place() {
        let pages = placeable(2, 4);
        let a = reserve_aligned(4, 4);
        let second = a.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        assert_eq!(place_of(pages[0]), (Some(a as usize), 0));
        // Unmapped: still its place (the page lives).
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        let taken = placed_count(1);
        assert_eq!(XMemMapPhysicalPages(a, 1, pages[1..].as_ptr()), a);
        assert_eq!(place_of(pages[1]), (None, 1), "its allocation's place");
        assert!(placed_count(1) > taken);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(3) };
        assert_eq!(page_bytes(pages[1])[0], 3);
        // The first page mapped next to it shows its own place.
        assert_eq!(XMemMapPhysicalPages(second, 1, pages.as_ptr()), second);
        assert_eq!(place_of(pages[0]), (Some(a as usize), 0));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { second.cast::<u8>().read() }, 0);
        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        assert_eq!(held(a, 4), 0, "freed, shown nowhere: given back");
    }

    /// A freed page's place is given back when its last view goes; a page
    /// placed there later reads zero, though the section kept the bytes
    /// (another page holds it), and the place is counted as given again.
    #[test]
    fn freed_places_are_given_back_and_read_zero_when_given_again() {
        let pages = placeable(2, 4);
        let a = reserve_aligned(4, 4);
        let alias = reserve(1);
        let second = a.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(alias, 1, pages.as_ptr()), alias);
        // SAFETY: both mapped read-write.
        unsafe {
            a.cast::<u8>().write_bytes(0xab, PAGE);
            second.cast::<u8>().write(0x11);
        }
        // Freed while shown at two addresses: held.
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
        assert_eq!(physical().place.taken.get(&(a as usize)), Some(&None));
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(
            physical().place.taken.get(&(a as usize)),
            Some(&None),
            "still shown at the alias"
        );
        // SAFETY: still mapped.
        assert_eq!(unsafe { alias.cast::<u8>().read() }, 0xab);
        assert_eq!(XMemMapPhysicalPages(alias, 1, std::ptr::null()), alias);
        assert!(
            !physical().place.taken.contains_key(&(a as usize)),
            "given back"
        );
        // A new page first mapped there takes the place, zeroed.
        let again = placed_count(5);
        let newer = placeable(1, 4);
        assert_eq!(XMemMapPhysicalPages(a, 1, newer.as_ptr()), a);
        assert_eq!(place_of(newer[0]), (Some(a as usize), 0));
        assert!(placed_count(5) > again);
        // SAFETY: mapped read-write.
        unsafe {
            assert!(
                std::slice::from_raw_parts(a.cast::<u8>(), PAGE)
                    .iter()
                    .all(|&b| b == 0),
                "a fresh page reads zero"
            );
            assert_eq!(second.cast::<u8>().read(), 0x11, "its neighbour untouched");
        }
        // Freed while shown nowhere: given back at once.
        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(1, newer.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, pages[1..].as_ptr()), 1);
        assert_eq!(held(a, 4), 0);
    }

    /// A place given again whose first mapping failed (simulated: placed,
    /// not mapped) is zeroed before a DirectStorage write lands in it, not
    /// after: the write's bytes stay, and the old page's do not show.
    #[test]
    fn a_place_given_again_and_written_before_its_mapping_keeps_the_bytes() {
        let pages = placeable(2, 4);
        let a = reserve_aligned(4, 4);
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write_bytes(0xab, PAGE) };
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
        let newer = placeable(1, 4);
        place_first(&mut physical(), a as usize, &newer);
        assert!(physical().place.zero.contains(&(a as usize)), "given again, to be zeroed");
        assert!(write_physical(&newer, 1, &[5]));
        let bytes = page_bytes(newer[0]);
        assert_eq!(bytes[1], 5);
        assert!(bytes.iter().enumerate().all(|(i, &b)| i == 1 || b == 0), "zeroed first");
        assert_eq!(XMemMapPhysicalPages(a, 1, newer.as_ptr()), a);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { a.cast::<u8>().add(1).read() }, 5, "not zeroed again");
        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(1, newer.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, pages[1..].as_ptr()), 1);
    }

    /// A page written before its first mapping (a DirectStorage destination
    /// not mapped yet) keeps its allocation's place and its bytes; the
    /// page beside it that was not written is placed.
    #[test]
    fn pages_written_before_their_first_mapping_keep_their_place() {
        let pages = placeable(2, 4);
        let written = placed_count(2);
        assert!(write_physical(&pages[..1], 5, &[7]));
        assert!(placed_count(2) > written);
        let a = reserve_aligned(4, 4);
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        assert_eq!(place_of(pages[0]), (None, 0));
        assert_eq!(place_of(pages[1]), (Some(a as usize), 1));
        assert_eq!(views_in(a, 2), [1, 1]);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { a.cast::<u8>().add(5).read() }, 7);
        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// Windows of two pages: four pages mapped over two windows are one run
    /// of each window's section, two views; DirectStorage writes over the
    /// bound land in both.
    #[test]
    fn a_run_over_a_window_bound_is_one_run_per_window() {
        let pages = placeable(4, 2);
        let a = reserve_aligned(4, 2);
        let mapped = [pages[3], pages[2], pages[1], pages[0]];
        assert_eq!(XMemMapPhysicalPages(a, 4, mapped.as_ptr()), a);
        assert_eq!(views_in(a, 4), [2, 2]);
        assert_eq!(place_of(pages[3]), (Some(a as usize), 0));
        assert_eq!(place_of(pages[0]), (Some(a as usize + 2 * PAGE), 1));
        assert!(write_physical(&mapped, 2 * PAGE - 1, &[0x21, 0x22]));
        // SAFETY: mapped read-write.
        unsafe {
            assert_eq!(a.cast::<u8>().add(2 * PAGE - 1).read(), 0x21);
            assert_eq!(a.cast::<u8>().add(2 * PAGE).read(), 0x22);
        }
        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// Below 4 GiB everything is one section: two pages mapped apart, in
    /// two calls, are views of the same section at the offsets their
    /// addresses name, so a buffer mapped there in order is one run of it.
    #[test]
    fn the_32_bit_memory_is_one_section_of_placement() {
        let pages = placeable(2, 4);
        let low = || {
            let p = XMemVirtualAlloc(
                std::ptr::null_mut(),
                PAGE,
                MEM_RESERVE | PAGES_64K,
                GRAPHICS | ADDRESS_32BIT,
                PAGE_READWRITE,
            );
            assert!(!p.is_null() && (p as usize) + PAGE <= LOW_WINDOW, "{p:p}");
            p
        };
        let (a, b) = (low(), low());
        assert_eq!(XMemMapPhysicalPages(b, 1, pages[1..].as_ptr()), b);
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        assert_eq!(place_of(pages[0]), (Some(0), a as usize / PAGE));
        assert_eq!(place_of(pages[1]), (Some(0), b as usize / PAGE));
        let same = {
            let physical = physical();
            Arc::ptr_eq(
                &physical.pages[&pages[0]].section,
                &physical.pages[&pages[1]].section,
            )
        };
        assert!(same, "one section below 4 GiB");
        for p in [a, b] {
            assert_eq!(XMemMapPhysicalPages(p, 1, std::ptr::null()), p);
        }
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// The placement's line names what is placed now and the counts.
    #[test]
    fn the_placement_line_says_what_is_placed() {
        let pages = placeable(3, 4);
        let a = reserve_aligned(4, 4);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let line = (0..1000)
            .find_map(|_| placement_line())
            .expect("the page table free once");
        for part in [
            "MB placed now in ",
            " runs of consecutive addresses (the largest ",
            " pages placed at their address, ",
            " sections of placement made, ",
        ] {
            assert!(line.contains(part), "{line}");
        }
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }
}
