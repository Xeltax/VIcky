//! Lot 4's device probes P0 and P8 (`VICTORY_LOT4_PROBE=1`; off by default;
//! the host side's P1-P7 are in [`super`]). They decide
//! whether lot 4 builds a mirror of title memory keyed by physical page:
//! the title's 64 KiB frames become the tiles of one reserved buffer, each
//! mapped at every offset the title maps it (`UpdateTileMappings`), so that
//! a page written once is current at all its addresses and a buffer view
//! is an offset into the mirror. Both probes only measure: nothing the
//! title or the host's submissions use changes.
//!
//! - **P0** ([`p0`]), once, right after the host device is made: what this
//!   card and its driver allow. Here, questions only: the tiers, the
//!   Recreate-At-GPUVA tier (asked, nothing required), the memory budgets,
//!   reserved buffers of 2 to 16 GiB made and let go. Every tile mapping
//!   runs in a child process of this one ([`child`]), on the same adapter:
//!   what `UpdateTileMappings` costs for 1 to 16,384 ranges, one tile of a
//!   heap in the card's memory mapped at several offsets of the largest
//!   buffer, read by copies and written by a compute shader through them,
//!   and tiles from heaps in the PC's memory, a heap opened over the
//!   process's own memory (`OpenExistingHeapFromAddress1`) among them. A
//!   mapping into an opened heap removed the device before
//!   (`crate::inplace`'s notes), the software adapter lost its device to
//!   the aliasing test, and a device removed here would end the run.
//! - **P8** ([`tick`]), every [`EVERY`] frames: one frame's CPU copies of
//!   title memory into host copies are recorded; of those refreshed from
//!   private, write-watched title memory, the one that copied the most is
//!   opened in place (`crate::inplace::place_part`, the in-place path's own
//!   call) and its runs are pulled again by the card's copy engine, into a
//!   buffer of ours of the kind the CPU copied into. CPU and GPU times side
//!   by side, per sample and as running medians.
//!
//! Every line starts with `lot4 probe P0:` or `lot4 probe P8:` (VICKY-RT.md,
//! "Lot 4 probes", says how to read them). Off, the probes cost one load of
//! a flag per refresh and per run copied, and one `OnceLock` read a sweep.

use std::collections::HashMap;
use std::ffi::c_void;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;
use windows_sys::Win32::System::Memory as mem;

use crate::inplace::{self, Rig};

/// The card's tile and the console's page: 64 KiB.
const TILE: u64 = 64 << 10;
const GIB: u64 = 1 << 30;
/// P8 samples one frame in this many.
pub(crate) const EVERY: u64 = 64;
/// Tiles of P0's heap in the card's memory (4 MiB).
const HEAP_TILES: u32 = 64;
/// The range counts P0 times `UpdateTileMappings` with, and how often each.
const RANGES: [u32; 4] = [1, 64, 1024, 16_384];
const REPEATS: usize = 5;
/// The reserved buffer sizes P0 tries, in GiB.
const RESERVED_GIB: [u64; 4] = [2, 4, 8, 16];
/// How long a child of P0 may take.
const CHILD_LIMIT: Duration = Duration::from_secs(15);
/// The child's mode, adapter (a LUID in hex) and the largest reserved
/// buffer the parent made (bytes), in its environment.
const CHILD: &str = "VICTORY_LOT4_CHILD";
const CHILD_ADAPTER: &str = "VICTORY_LOT4_CHILD_ADAPTER";
const CHILD_BYTES: &str = "VICTORY_LOT4_CHILD_BYTES";
/// What P0's children try, each in a process of its own: one tile of a
/// heap in the card's memory at two offsets of a 4 MiB reserved buffer;
/// in the largest reserved buffer, the cost of `UpdateTileMappings` and
/// one tile at offsets up to its end; tiles of a heap in the PC's memory
/// the CPU maps (write-back, as the host copies are), of one the CPU
/// cannot map, and of a heap opened over the process's own memory, with a
/// reserved buffer made cross-adapter (as the opened heap is) and without.
const MODES: [&str; 6] = ["card-heap", "card-heap-big", "pc-heap-cpu", "pc-heap", "opened-cross", "opened"];
/// P8: a host copy's runs pulled lie within this many bytes of its first.
const PULL_SPAN: u64 = 64 << 20;
/// P8: the host copies tried, most bytes copied first, before giving up.
const TRIES: usize = 4;

/// `VICTORY_LOT4_PROBE=1`; any other value or unset: off.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch_on(std::env::var("VICTORY_LOT4_PROBE").ok().as_deref()))
}

pub(super) fn switch_on(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// `text` on one line (Windows' error messages hold line breaks).
fn flat(text: &str) -> String {
    text.replace(['\r', '\n'], " ").trim_end().to_owned()
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "NO" }
}

/// 64 KiB, or `len` bytes, that differ from those of any other seed.
fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_mul(29).wrapping_add(seed) ^ ((i >> 8) as u8)).collect()
}

fn gpu_va(resource: &ID3D12Resource) -> u64 {
    // SAFETY: a plain query of a live resource.
    unsafe { resource.GetGPUVirtualAddress() }
}

fn query<T>(device: &ID3D12Device, feature: D3D12_FEATURE, data: &mut T) -> windows::core::Result<()> {
    // SAFETY: a plain query into a structure of its own size.
    unsafe { device.CheckFeatureSupport(feature, (data as *mut T).cast(), size_of::<T>() as u32) }
}

fn luid_text(luid: LUID) -> String {
    format!("{:08x}{:08x}", luid.HighPart as u32, luid.LowPart)
}

fn luid_parse(text: &str) -> Option<LUID> {
    let v = u64::from_str_radix(text.trim(), 16).ok()?;
    Some(LUID { LowPart: v as u32, HighPart: (v >> 32) as i32 })
}

// P0: the device.

/// What P0 found. The lines are the whole account; the fields feed the
/// verdict.
#[derive(Default)]
pub(crate) struct P0 {
    pub(crate) lines: Vec<String>,
    tier: i32,
    /// The largest reserved buffer made, in bytes (0: none).
    reserved: u64,
    /// One heap tile mapped at several offsets: copies read the same bytes
    /// through each, and a shader's store through one is read through
    /// another. `None`: not tried.
    alias_copy: Option<bool>,
    alias_shader: Option<bool>,
    /// `UpdateTileMappings`: ranges, median CPU microseconds, median
    /// microseconds more until the queue had done it.
    cost: Vec<(u32, f64, f64)>,
    /// Per child mode: whether tiles of that heap worked (`None`: no
    /// answer).
    system: Vec<(&'static str, Option<bool>)>,
    recreate: Option<i32>,
    /// The PC's memory the system lets the card use for this process.
    shared_budget: Option<u64>,
}

/// P0, logged: called once by `crate::host` right after the device is
/// made, with the system's memory budgets for the process. Every failure
/// is a line; nothing here ends the run.
pub(crate) fn p0(device: &ID3D12Device, budgets: Option<[[u64; 4]; 2]>) {
    let started = Instant::now();
    tracing::info!(
        "lot4 probe P0: VICTORY_LOT4_PROBE=1: the device probe now, once; P8 (a copy-queue pull of one region) one frame in {EVERY}"
    );
    let child = Child::this_exe();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe(device, budgets, child.as_ref()))) {
        Ok(p) => {
            for line in &p.lines {
                tracing::info!("lot4 probe P0: {}", flat(line));
            }
        }
        Err(_) => tracing::warn!("lot4 probe P0: the probe stopped on a panic; its lines are lost, the run goes on"),
    }
    tracing::info!("lot4 probe P0: took {:.0} ms", ms(started.elapsed()));
}

/// P0's work on `device`: questions here, every tile mapping in a child
/// that `child` starts (`None`: no mapping is tried).
pub(crate) fn probe(device: &ID3D12Device, budgets: Option<[[u64; 4]; 2]>, child: Option<&Child>) -> P0 {
    let mut p = P0::default();
    caps(device, &mut p);
    budget_lines(budgets, &mut p);
    reserved_buffers(device, &mut p);
    // SAFETY: a plain query.
    let adapter = luid_text(unsafe { device.GetAdapterLuid() });
    let Some(child) = child else {
        p.lines.push("tile mappings: not tried (not running as victory.exe: no child process to try them in)".to_owned());
        verdict(&mut p);
        return p;
    };
    for mode in MODES {
        let (lines, ended) = child.run(mode, &adapter, p.reserved);
        let mut works = None;
        for line in lines {
            if let Some(cost) = line.strip_prefix("COST ") {
                p.cost.extend(parse_cost(cost));
                continue;
            }
            if let Some(alias) = line.strip_prefix("ALIAS ") {
                (p.alias_copy, p.alias_shader) = parse_alias(alias);
                continue;
            }
            if let Some(verdict) = line.strip_prefix("VERDICT ") {
                works = Some(verdict.contains(" WORKS"));
            }
            p.lines.push(format!("[{mode}] {line}"));
        }
        p.system.push((mode, works));
        p.lines.push(format!("[{mode}] the child {ended}"));
    }
    verdict(&mut p);
    p
}

/// A child's `COST <ranges> <CPU us> <queue us>`.
fn parse_cost(text: &str) -> Option<(u32, f64, f64)> {
    let mut words = text.split_whitespace();
    Some((words.next()?.parse().ok()?, words.next()?.parse().ok()?, words.next()?.parse().ok()?))
}

/// A child's `ALIAS copy=yes|no shader=yes|no`.
fn parse_alias(text: &str) -> (Option<bool>, Option<bool>) {
    let field = |name: &str| text.split_whitespace().find_map(|w| w.strip_prefix(name)).map(|v| v == "yes");
    (field("copy="), field("shader="))
}

/// The device's answers: tiers, address bits, Recreate At GPUVA, upload
/// heaps, copy-queue timestamps, the runtime and the adapter.
fn caps(device: &ID3D12Device, p: &mut P0) {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    let asked = query(device, D3D12_FEATURE_D3D12_OPTIONS, &mut o);
    p.tier = o.TiledResourcesTier.0;
    let mut va = D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT::default();
    let va_ok = query(device, D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &mut va).is_ok();
    p.lines.push(format!(
        "tiled resources tier {} (reserved buffers need 1 or more; 2 reads unmapped tiles as zeros){}; resource heap tier {}; GPU address bits \
         per resource {}, per process {}",
        o.TiledResourcesTier.0,
        if asked.is_ok() { "" } else { " (the query failed)" },
        o.ResourceHeapTier.0,
        if va_ok { va.MaxGPUVirtualAddressBitsPerResource.to_string() } else { "?".to_owned() },
        if va_ok { va.MaxGPUVirtualAddressBitsPerProcess.to_string() } else { "?".to_owned() },
    ));
    let mut o20 = D3D12_FEATURE_DATA_D3D12_OPTIONS20::default();
    let recreate = match query(device, D3D12_FEATURE_D3D12_OPTIONS20, &mut o20) {
        Ok(()) => {
            p.recreate = Some(o20.RecreateAtTier.0);
            format!(
                "tier {} (0 none, 1 heaps and buffers made again at a chosen GPU address; the runtime asks developer mode for it, not asked \
                 here); compute-only write watch {}",
                o20.RecreateAtTier.0,
                o20.ComputeOnlyWriteWatchSupported.as_bool()
            )
        }
        Err(e) => {
            // The runtime may know the options' older, shorter form.
            #[repr(C)]
            #[derive(Default)]
            struct Older {
                compute_only_write_watch: windows::core::BOOL,
            }
            let mut older = Older::default();
            match query(device, D3D12_FEATURE_D3D12_OPTIONS20, &mut older) {
                Ok(()) => format!(
                    "not known to this runtime (its options 20 lack the tier); compute-only write watch {}",
                    older.compute_only_write_watch.as_bool()
                ),
                Err(_) => format!("not known to this runtime (options 20: {})", e.message().trim()),
            }
        }
    };
    p.lines.push(format!("Recreate At GPUVA: {recreate}"));
    let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    let upload = match query(device, D3D12_FEATURE_D3D12_OPTIONS16, &mut o16) {
        Ok(()) => o16.GPUUploadHeapSupported.as_bool().to_string(),
        Err(_) => "not known to this runtime".to_owned(),
    };
    let mut o3 = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
    let stamps = query(device, D3D12_FEATURE_D3D12_OPTIONS3, &mut o3).is_ok() && o3.CopyQueueTimestampQueriesSupported.as_bool();
    p.lines.push(format!("GPU upload heaps {upload}; timestamps on copy queues {stamps} (P8's GPU time needs them)"));
    p.lines.push(format!("the D3D12 runtime: {}; {}", runtime(device), adapter(device)));
}

/// Which D3D12 runtime the process loaded (the Agility SDK's
/// `D3D12Core.dll` beside the exe, or the system's) and its SDK version
/// where the runtime says it.
fn runtime(device: &ID3D12Device) -> String {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
    let name: Vec<u16> = "D3D12Core.dll\0".encode_utf16().collect();
    // SAFETY: a NUL-terminated name; a module of this process.
    let module = unsafe { GetModuleHandleW(name.as_ptr()) };
    let core = if module.is_null() {
        "D3D12Core.dll not loaded".to_owned()
    } else {
        let mut path = [0u16; 520];
        // SAFETY: a loaded module and a buffer of its stated length.
        let n = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
        format!("D3D12Core.dll from {}", String::from_utf16_lossy(&path[..n.min(path.len())]))
    };
    let sdk = match device.cast::<ID3D12DeviceConfiguration>() {
        // SAFETY: a plain query.
        Ok(config) => format!("SDK version {}", unsafe { config.GetDesc() }.SDKVersion),
        Err(_) => "no device configuration interface (an older runtime)".to_owned(),
    };
    format!("{core}, {sdk}")
}

/// The adapter's name, memory and user-mode driver version.
fn adapter(device: &ID3D12Device) -> String {
    // SAFETY: plain DXGI queries on the device's own adapter.
    unsafe {
        let luid = device.GetAdapterLuid();
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return "no DXGI factory".to_owned();
        };
        let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter1>(luid) else {
            return "the adapter not found by its LUID".to_owned();
        };
        let Ok(desc) = adapter.GetDesc1() else {
            return "no adapter description".to_owned();
        };
        let name = String::from_utf16_lossy(&desc.Description);
        let driver = adapter
            .CheckInterfaceSupport(&IDXGIDevice::IID)
            .map(|v| format!("{}.{}.{}.{}", (v >> 48) & 0xffff, (v >> 32) & 0xffff, (v >> 16) & 0xffff, v & 0xffff))
            .unwrap_or_else(|_| "?".to_owned());
        format!(
            "adapter {} ({} MB of its own, {} MB of the PC's shared), driver {driver}",
            name.trim_end_matches('\0').trim(),
            desc.DedicatedVideoMemory >> 20,
            desc.SharedSystemMemory >> 20
        )
    }
}

/// `QueryVideoMemoryInfo`'s two groups: for each, budget, use, what could
/// still be reserved, what is reserved.
fn budget_lines(budgets: Option<[[u64; 4]; 2]>, p: &mut P0) {
    let Some([own, shared]) = budgets else {
        p.lines.push("QueryVideoMemoryInfo: no answer (no adapter for the device)".to_owned());
        return;
    };
    p.shared_budget = Some(shared[0]);
    let group = |g: [u64; 4]| {
        format!(
            "budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved",
            g[0] >> 20,
            g[1] >> 20,
            g[2] >> 20,
            g[3] >> 20
        )
    };
    p.lines.push(format!(
        "QueryVideoMemoryInfo, at the start: the card's own memory (local): {}; the PC's memory shared with the card (non-local): {}",
        group(own),
        group(shared)
    ));
}

fn buffer_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    inplace::buffer_desc(bytes, flags)
}

/// A reserved buffer of `bytes`, which shaders may also write.
fn reserved_buffer(device: &ID3D12Device, bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> windows::core::Result<ID3D12Resource> {
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; a buffer has no clear value.
    unsafe { device.CreateReservedResource(&buffer_desc(bytes, flags), D3D12_RESOURCE_STATE_COMMON, None, &mut made)? };
    made.ok_or_else(windows::core::Error::empty)
}

fn refused(e: &windows::core::Error) -> String {
    format!("REFUSED {:#010x} ({})", e.code().0, e.message().trim())
}

/// Reserved buffers of [`RESERVED_GIB`], made and let go at once (a
/// refusal is an error code, nothing more): the largest made is the one
/// the child `card-heap-big` maps.
fn reserved_buffers(device: &ID3D12Device, p: &mut P0) {
    let mut said = Vec::new();
    for gib in RESERVED_GIB {
        let bytes = gib * GIB;
        match reserved_buffer(device, bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
            Ok(buffer) => {
                said.push(format!("{gib} GiB ok (GPU address {:#x})", gpu_va(&buffer)));
                p.reserved = bytes;
            }
            Err(e) => said.push(format!("{gib} GiB {}", refused(&e))),
        }
    }
    p.lines.push(format!(
        "reserved buffers (CreateReservedResource: address space only, no memory until tiles are mapped): {}; reserved-resource support for \
         buffers {}",
        said.join(", "),
        yes(p.reserved > 0)
    ));
}

/// Arrays for one `UpdateTileMappings` that maps, for each pair
/// (resource tile, heap tile), one tile of the resource to one of the
/// heap: as many regions and ranges as pairs, one tile each.
struct Mapping {
    coords: Vec<D3D12_TILED_RESOURCE_COORDINATE>,
    sizes: Vec<D3D12_TILE_REGION_SIZE>,
    flags: Vec<D3D12_TILE_RANGE_FLAGS>,
    starts: Vec<u32>,
    counts: Vec<u32>,
}

impl Mapping {
    fn new(pairs: &[(u32, u32)]) -> Mapping {
        Mapping {
            coords: pairs.iter().map(|&(t, _)| D3D12_TILED_RESOURCE_COORDINATE { X: t, Y: 0, Z: 0, Subresource: 0 }).collect(),
            sizes: vec![D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() }; pairs.len()],
            flags: vec![D3D12_TILE_RANGE_FLAG_NONE; pairs.len()],
            starts: pairs.iter().map(|&(_, h)| h).collect(),
            counts: vec![1; pairs.len()],
        }
    }

    /// Puts the mapping on `queue` (the call returns before the queue has
    /// done it).
    fn send(&self, queue: &ID3D12CommandQueue, buffer: &ID3D12Resource, heap: &ID3D12Heap) {
        let n = self.coords.len() as u32;
        // SAFETY: a live reserved buffer and heap; five arrays of `n`
        // entries that live through the call; each region and range is one
        // tile, inside the buffer and the heap (the callers' pairs).
        unsafe {
            queue.UpdateTileMappings(
                buffer,
                n,
                Some(self.coords.as_ptr()),
                Some(self.sizes.as_ptr()),
                heap,
                n,
                Some(self.flags.as_ptr()),
                Some(self.starts.as_ptr()),
                Some(self.counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }
}

/// `n` pairs (resource tile, heap tile) spread over a buffer of `tiles`
/// tiles, each heap tile of [`HEAP_TILES`] used at several offsets (as the
/// mirror would map a page the title shows at several addresses).
fn spread(n: u32, tiles: u64) -> Vec<(u32, u32)> {
    let step = (tiles / u64::from(n.max(1))).max(1);
    (0..n).map(|i| ((u64::from(i) * step).min(tiles.saturating_sub(1)) as u32, (i.wrapping_mul(7)) % HEAP_TILES)).collect()
}

/// A heap of `tiles` tiles in the card's memory, for buffers.
fn card_heap(device: &ID3D12Device, tiles: u32) -> windows::core::Result<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: u64::from(tiles) * TILE,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
    };
    let mut made: Option<ID3D12Heap> = None;
    // SAFETY: plain creation; the out pointer is ours.
    unsafe { create_heap(device, &desc, &mut made)? };
    made.ok_or_else(windows::core::Error::empty)
}

/// `CreateHeap` into `made`.
unsafe fn create_heap(device: &ID3D12Device, desc: &D3D12_HEAP_DESC, made: &mut Option<ID3D12Heap>) -> windows::core::Result<()> {
    // SAFETY: the caller's description and out pointer.
    unsafe { device.CreateHeap::<ID3D12Heap>(desc, made) }
}

/// Tile 3 of `heap` (in the card's memory) mapped at several offsets of
/// the reserved buffer (`tiles` tiles): copies through one offset read
/// through the others, and a compute shader's store through one read
/// through another. In a child only ([`child_big`]).
fn alias(rig: &mut Rig, buffer: &ID3D12Resource, tiles: u64, heap: &ID3D12Heap, p: &mut P0) {
    let low = 4 * GIB / TILE;
    let mut at: Vec<u64> = vec![0, 1, tiles / 2, tiles - 1];
    if tiles > low {
        at.extend([low - 1, low]);
    }
    at.sort_unstable();
    at.dedup();
    let pairs: Vec<(u32, u32)> = at.iter().map(|&t| (t as u32, 3)).collect();
    let mapping = Mapping::new(&pairs);
    let started = Instant::now();
    mapping.send(&rig.queue, buffer, heap);
    let call = us(started.elapsed());
    let offsets: Vec<String> = at.iter().map(|&t| format!("{:#x}", t * TILE)).collect();
    if let Err(e) = rig.settle() {
        p.lines.push(format!("one heap tile mapped at {} offsets: after the mapping the device says {e}", at.len()));
        (p.alias_copy, p.alias_shader) = (Some(false), Some(false));
        return;
    }
    let last = *at.last().unwrap_or(&0);
    let a = pattern(TILE as usize, 0x11);
    let copy = rig.store(buffer, 0, &a).and_then(|()| {
        let mut all = true;
        for &t in &at[1..] {
            all &= rig.fetch(buffer, t * TILE, TILE as usize)? == a;
        }
        Ok(all)
    });
    let b = pattern(TILE as usize, 0x5d);
    let back = rig.store(buffer, last * TILE, &b).and_then(|()| Ok(rig.fetch(buffer, 0, TILE as usize)? == b));
    let words: Vec<u8> = [40u32, 2, 0].iter().flat_map(|w| w.to_le_bytes()).collect();
    let second = at.get(1).copied().unwrap_or(0);
    let shaded = rig
        .store(buffer, second * TILE, &words)
        .and_then(|()| rig.shade(gpu_va(buffer) + last * TILE))
        .and_then(|()| rig.fetch(buffer, 0, 12));
    let shader = matches!(&shaded, Ok(got) if got.get(8..12) == Some(&42u32.to_le_bytes()[..]));
    let told = |r: &windows::core::Result<bool>| match r {
        Ok(b) => yes(*b).to_owned(),
        Err(e) => format!("failed ({e})"),
    };
    p.alias_copy = Some(matches!(copy, Ok(true)) && matches!(back, Ok(true)));
    p.alias_shader = Some(shader);
    p.lines.push(format!(
        "one tile of a heap in the card's memory mapped at {} offsets of the {} GiB reserved buffer ({}) by one UpdateTileMappings of {call:.0} us: \
         64 KiB copied in through offset 0 read the same through every other offset {}; copied in through the last, read the same through \
         offset 0 {}; a compute shader's store through the last offset (root unordered view), read through offset 0 {}",
        at.len(),
        tiles * TILE / GIB,
        offsets.join(", "),
        told(&copy),
        told(&back),
        match &shaded {
            Ok(_) => yes(shader).to_owned(),
            Err(e) => format!("failed ({e})"),
        }
    ));
}

/// CPU time of `UpdateTileMappings` for [`RANGES`] ranges, and the time
/// until the queue had done it; then the cost of unmapping all at once.
/// In a child only ([`child_big`]).
fn tile_cost(rig: &mut Rig, buffer: &ID3D12Resource, tiles: u64, heap: &ID3D12Heap, p: &mut P0) {
    let mut said = Vec::new();
    for n in RANGES {
        if u64::from(n) > tiles {
            said.push(format!("{n}: more ranges than the buffer has tiles"));
            continue;
        }
        let mapping = Mapping::new(&spread(n, tiles));
        let (mut cpu, mut queue) = (Vec::new(), Vec::new());
        let mut failed = None;
        for _ in 0..REPEATS {
            let started = Instant::now();
            mapping.send(&rig.queue, buffer, heap);
            let call = started.elapsed();
            if let Err(e) = rig.settle() {
                failed = Some(e);
                break;
            }
            cpu.push(us(call));
            queue.push(us(started.elapsed()) - us(call));
        }
        if let Some(e) = failed {
            said.push(format!("{n}: after the call the device says {e}"));
            break;
        }
        let best = cpu.iter().copied().fold(f64::MAX, f64::min);
        p.cost.push((n, median(&cpu), median(&queue)));
        said.push(format!(
            "{n}: {:.0} us CPU (best {best:.0}, {:.2} us a range), then {:.0} us until the queue had done it",
            median(&cpu),
            median(&cpu) / f64::from(n),
            median(&queue)
        ));
    }
    // All tiles unmapped by one range.
    let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
    let size = D3D12_TILE_REGION_SIZE { NumTiles: tiles as u32, ..Default::default() };
    let flag = D3D12_TILE_RANGE_FLAG_NULL;
    let (first, count) = (0u32, tiles as u32);
    let started = Instant::now();
    // SAFETY: one region of the whole buffer and one range of as many
    // tiles that maps them to nothing (the heap is not read for it).
    unsafe {
        rig.queue.UpdateTileMappings(
            buffer,
            1,
            Some(&raw const coord),
            Some(&raw const size),
            heap,
            1,
            Some(&raw const flag),
            Some(&raw const first),
            Some(&raw const count),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }
    let call = us(started.elapsed());
    let settled = rig.settle();
    p.lines.push(format!(
        "UpdateTileMappings on a direct queue, N ranges of one tile each (N regions of one tile spread over the {} GiB buffer; {HEAP_TILES} heap \
         tiles each mapped at several offsets), median of {REPEATS}: {}; all {tiles} tiles unmapped by one range: {call:.0} us CPU{}",
        tiles * TILE / GIB,
        said.join("; "),
        match settled {
            Ok(()) => String::new(),
            Err(e) => format!(", then the device says {e}"),
        }
    ));
}

/// The VERDICT line: what kills the mirror, what limits it.
fn verdict(p: &mut P0) {
    let mut dead = Vec::new();
    let mut limits = Vec::new();
    if p.tier < 1 {
        dead.push("no tiled resources".to_owned());
    }
    if p.reserved == 0 {
        dead.push("no reserved buffer".to_owned());
    } else if p.reserved < 4 * GIB {
        limits.push(format!("reserved buffers only up to {} GiB: arenas under 4 GiB", p.reserved / GIB));
    }
    let works = |mode: &str| p.system.iter().find(|s| s.0 == mode).and_then(|s| s.1);
    if works("card-heap") == Some(false) {
        dead.push("one tile of a heap in the card's memory does not read the same at two offsets of a 4 MiB buffer".to_owned());
    }
    match (p.alias_copy, p.alias_shader) {
        (Some(true), Some(true)) => {}
        _ if p.system.is_empty() => {}
        _ if works("card-heap") == Some(true) => limits.push(format!(
            "one tile at several offsets of the {} GiB buffer failed where a 4 MiB buffer works: smaller arenas",
            p.reserved / GIB
        )),
        _ => dead.push("one tile at several offsets of the large buffer does not read the same".to_owned()),
    }
    // Fable's threshold: 4,096 ranges over about 20 ms.
    let at_4096 = p.cost.iter().find(|c| c.0 == 16_384).map(|c| c.1 / 4.0).or_else(|| p.cost.iter().find(|c| c.0 == 1024).map(|c| c.1 * 4.0));
    if let Some(micro) = at_4096
        && micro > 20_000.0
    {
        dead.push(format!("UpdateTileMappings about {:.1} ms for 4,096 ranges (over 20)", micro / 1e3));
    }
    let word = |w: Option<bool>| match w {
        Some(true) => "works",
        Some(false) => "FAILS",
        None => "no answer",
    };
    let pc = ["pc-heap-cpu", "pc-heap"].iter().any(|m| works(m) == Some(true));
    let opened = ["opened-cross", "opened"].iter().any(|m| works(m) == Some(true));
    if !p.system.is_empty() && !pc && !opened {
        limits.push("no tiles in the PC's memory: the mirror only as large as the card's free memory".to_owned());
    } else if !p.system.is_empty() && !opened {
        limits.push("tiles in the PC's memory only from heaps of ours, not over title memory: every page still copied in".to_owned());
    }
    if let Some(budget) = p.shared_budget
        && budget < 12 << 30
    {
        limits.push(format!("the PC's memory budget for the card {} MB (under 12 GB)", budget >> 20));
    }
    let cost = p.cost.iter().map(|c| format!("{} ranges {:.0} us", c.0, c.1)).collect::<Vec<_>>().join(", ");
    let alive = if !dead.is_empty() {
        format!("DEAD: {}", dead.join("; "))
    } else if p.system.is_empty() {
        "UNKNOWN (no child process: no mapping tried)".to_owned()
    } else {
        "ALIVE".to_owned()
    };
    p.lines.push(format!(
        "VERDICT: reserved buffer of 4 GiB or more {}; one tile at two offsets of a 4 MiB buffer {}; at several offsets of the largest: copies {}, \
         shader store {}; UpdateTileMappings CPU {}; tiles in the PC's memory: CPU-mapped heap {}, heap without CPU access {}, opened over host \
         memory {} (cross-adapter buffer) / {} (plain); Recreate At GPUVA tier {} -> the mirror {alive}{}",
        yes(p.reserved >= 4 * GIB),
        word(works("card-heap")),
        p.alias_copy.map_or("not tried", |b| if b { "same" } else { "DIFFER" }),
        p.alias_shader.map_or("not tried", |b| if b { "seen" } else { "NOT seen" }),
        if cost.is_empty() { "not measured".to_owned() } else { cost },
        word(works("pc-heap-cpu")),
        word(works("pc-heap")),
        word(works("opened-cross")),
        word(works("opened")),
        p.recreate.map_or("unknown".to_owned(), |t| t.to_string()),
        if limits.is_empty() { String::new() } else { format!("; limited: {}", limits.join("; ")) },
    ));
}

// P0's children: tile mappings that might remove their device.

/// How a child of this process is started: the program and the arguments
/// before the mode, which goes in its environment.
pub(crate) struct Child {
    exe: PathBuf,
    args: Vec<String>,
}

impl Child {
    /// This program, if it is `victory.exe` (whose `main` hands a child to
    /// [`child`] before anything else).
    fn this_exe() -> Option<Child> {
        let exe = std::env::current_exe().ok()?;
        let name = exe.file_stem()?.to_string_lossy().to_ascii_lowercase();
        (name == "victory").then(|| Child { exe, args: Vec::new() })
    }

    /// Runs `mode` on the adapter with LUID `adapter` (`reserved`: the
    /// largest reserved buffer made, in bytes), [`CHILD_LIMIT`] at most:
    /// the lines it said, and how it ended.
    fn run(&self, mode: &str, adapter: &str, reserved: u64) -> (Vec<String>, String) {
        let started = Instant::now();
        let spawned = std::process::Command::new(&self.exe)
            .args(&self.args)
            .env(CHILD, mode)
            .env(CHILD_ADAPTER, adapter)
            .env(CHILD_BYTES, reserved.to_string())
            .env_remove("VICTORY_LOT4_PROBE")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn();
        let mut process = match spawned {
            Ok(process) => process,
            Err(e) => return (Vec::new(), format!("could not be started: {e}")),
        };
        let mut out = process.stdout.take();
        let reader = std::thread::spawn(move || {
            let mut text = String::new();
            if let Some(out) = out.as_mut() {
                let _ = out.read_to_string(&mut text);
            }
            text
        });
        let ended = loop {
            match process.try_wait() {
                Ok(Some(status)) => break format!("ended ({status}) after {:.0} ms", ms(started.elapsed())),
                Ok(None) if started.elapsed() < CHILD_LIMIT => std::thread::sleep(Duration::from_millis(10)),
                Ok(None) => {
                    let _ = process.kill();
                    let _ = process.wait();
                    break format!("was stopped after {} s", CHILD_LIMIT.as_secs());
                }
                Err(e) => break format!("was lost: {e}"),
            }
        };
        let text = reader.join().unwrap_or_default();
        (child_lines(&text), ended)
    }
}

/// The lines a child said, out of whatever else its output holds (a test
/// harness writes the test's name before its first line, on the same line).
fn child_lines(text: &str) -> Vec<String> {
    text.lines().filter_map(|l| l.trim_end().split_once("lot4 child: ")).map(|(_, said)| said.to_owned()).collect()
}

/// In a child started by P0 ([`CHILD`] set): tries its mode, says what
/// happened on standard output and returns true; false in any other
/// process. `victory.exe`'s `main` calls it first.
pub fn child() -> bool {
    let Ok(mode) = std::env::var(CHILD) else {
        return false;
    };
    let say = |line: String| {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "lot4 child: {}", flat(&line));
        let _ = out.flush();
    };
    let bytes = std::env::var(CHILD_BYTES).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
    match child_device() {
        Ok(device) if mode.trim() == "card-heap-big" => child_big(&device, bytes, &say),
        Ok(device) => child_probe(&device, mode.trim(), &say),
        Err(e) => say(format!("VERDICT {mode} FAILS: no device on the adapter ({e})")),
    }
    true
}

/// The child `card-heap-big`: in a reserved buffer of `bytes` (the
/// largest the parent made), what `UpdateTileMappings` costs, then one
/// tile at offsets up to the buffer's end ([`tile_cost`], [`alias`]); also
/// `COST` and `ALIAS` lines for the parent's verdict.
fn child_big(device: &ID3D12Device, bytes: u64, say: &dyn Fn(String)) {
    const MODE: &str = "card-heap-big";
    let fail = |why: String| say(format!("VERDICT {MODE} FAILS: {why}"));
    if bytes < TILE {
        return fail("the parent made no reserved buffer".to_owned());
    }
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => return fail(format!("no queue ({e})")),
    };
    let buffer = match reserved_buffer(device, bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
        Ok(buffer) => buffer,
        Err(e) => return fail(format!("the {} GiB reserved buffer: {}", bytes / GIB, refused(&e))),
    };
    let heap = match card_heap(device, HEAP_TILES) {
        Ok(heap) => heap,
        Err(e) => return fail(format!("a heap in the card's memory: {}", refused(&e))),
    };
    let tiles = bytes / TILE;
    let mut p = P0::default();
    tile_cost(&mut rig, &buffer, tiles, &heap, &mut p);
    for c in &p.cost {
        say(format!("COST {} {:.1} {:.1}", c.0, c.1, c.2));
    }
    alias(&mut rig, &buffer, tiles, &heap, &mut p);
    for line in p.lines.drain(..) {
        say(line);
    }
    let word = |b: bool| if b { "yes" } else { "no" };
    if let (Some(copy), Some(shader)) = (p.alias_copy, p.alias_shader) {
        say(format!("ALIAS copy={} shader={}", word(copy), word(shader)));
    }
    match standing(device) {
        Err(why) => fail(why),
        Ok(()) if p.alias_copy == Some(true) && p.alias_shader == Some(true) => {
            say(format!("VERDICT {MODE} WORKS: one tile reads and writes the same at every offset; the device stands"))
        }
        Ok(()) => fail("one tile does not read the same at every offset".to_owned()),
    }
}

fn child_device() -> windows::core::Result<ID3D12Device> {
    let text = std::env::var(CHILD_ADAPTER).unwrap_or_default();
    let luid = luid_parse(&text).ok_or_else(windows::core::Error::empty)?;
    // SAFETY: plain DXGI queries and device creation.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumAdapterByLuid(luid)?;
        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        device.ok_or_else(windows::core::Error::empty)
    }
}

/// What the device says after a step, as words; `Err` once it is removed.
fn standing(device: &ID3D12Device) -> Result<(), String> {
    // SAFETY: a plain query.
    match unsafe { device.GetDeviceRemovedReason() } {
        Ok(()) => Ok(()),
        Err(e) => Err(format!("the device was REMOVED ({:#010x}: {})", e.code().0, e.message().trim())),
    }
}

/// The heap a child maps tiles from, and where the CPU writes its bytes.
struct Source {
    heap: ID3D12Heap,
    /// The CPU's address of the heap's tile 2, if the CPU has one.
    cpu: Option<*mut u8>,
    /// A buffer placed in the heap to give the CPU its address.
    _placed: Option<ID3D12Resource>,
    /// Memory of ours the heap was opened over: released by the caller
    /// once the heap is gone (a child that stops early leaves it to the
    /// end of its process).
    memory: Option<*mut c_void>,
}

/// The heap `mode` names, `Err` with why not.
fn source(device: &ID3D12Device, mode: &str) -> Result<Source, String> {
    const TILES: u64 = 16;
    match mode {
        "card-heap" => {
            let heap = card_heap(device, TILES as u32).map_err(|e| format!("the heap: {}", refused(&e)))?;
            Ok(Source { heap, cpu: None, _placed: None, memory: None })
        }
        "pc-heap-cpu" | "pc-heap" => {
            let cpu = mode == "pc-heap-cpu";
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: TILES * TILE,
                Properties: D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_CUSTOM,
                    CPUPageProperty: if cpu { D3D12_CPU_PAGE_PROPERTY_WRITE_BACK } else { D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE },
                    MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                    ..Default::default()
                },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            };
            let mut made: Option<ID3D12Heap> = None;
            // SAFETY: plain creation; the out pointer is ours.
            unsafe { create_heap(device, &desc, &mut made) }.map_err(|e| format!("the heap: {}", refused(&e)))?;
            let heap = made.ok_or("the heap: none made")?;
            if !cpu {
                return Ok(Source { heap, cpu: None, _placed: None, memory: None });
            }
            let mut placed: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a buffer over the heap's tile 2.
            unsafe {
                device.CreatePlacedResource(&heap, 2 * TILE, &buffer_desc(TILE, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, None, &mut placed)
            }
            .map_err(|e| format!("a buffer placed in the heap for the CPU: {}", refused(&e)))?;
            let placed = placed.ok_or("a buffer placed in the heap: none made")?;
            let mut data = std::ptr::null_mut();
            // SAFETY: a buffer in a heap the CPU maps; mapped for good.
            unsafe { placed.Map(0, None, Some(&mut data)) }.map_err(|e| format!("the placed buffer mapped: {}", refused(&e)))?;
            Ok(Source { heap, cpu: Some(data.cast()), _placed: Some(placed), memory: None })
        }
        "opened-cross" | "opened" => {
            let bytes = (TILES * TILE) as usize;
            // SAFETY: memory of our own, made the way `xmem` makes the
            // title's private memory.
            let base = unsafe {
                mem::VirtualAlloc(std::ptr::null(), bytes, mem::MEM_RESERVE | mem::MEM_COMMIT | 0x20_0000, mem::PAGE_READWRITE)
            };
            if base.is_null() {
                return Err(format!("no memory to open: {}", std::io::Error::last_os_error()));
            }
            match inplace::open(device, inplace::From::Part(base as usize, bytes)) {
                Ok(heap) => Ok(Source {
                    heap,
                    // SAFETY: inside the allocation.
                    cpu: Some(unsafe { base.cast::<u8>().add(2 * TILE as usize) }),
                    _placed: None,
                    memory: Some(base),
                }),
                Err(e) => {
                    // SAFETY: our own allocation, nothing opened on it.
                    unsafe { mem::VirtualFree(base, 0, mem::MEM_RELEASE) };
                    Err(format!("the heap opened over host memory (OpenExistingHeapFromAddress1): {}", refused(&e)))
                }
            }
        }
        _ => Err(format!("no such mode {mode:?}")),
    }
}

/// A child's test: heap tile 2 mapped at tiles 0 and 9 of a reserved
/// buffer, bytes written into it (by the CPU where it can), read through
/// both offsets; a GPU write through one, seen by the CPU and through the
/// other. The device is asked after each step. Ends with `VERDICT <mode>
/// WORKS` or `VERDICT <mode> FAILS: why`.
fn child_probe(device: &ID3D12Device, mode: &str, say: &dyn Fn(String)) {
    let fail = |why: String| say(format!("VERDICT {mode} FAILS: {why}"));
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => return fail(format!("no queue ({e})")),
    };
    let mut source = match source(device, mode) {
        Ok(source) => source,
        Err(why) => return fail(why),
    };
    say(inplace::describe(&source.heap));
    let flags = if mode == "opened-cross" { D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER } else { D3D12_RESOURCE_FLAG_NONE };
    let buffer = match reserved_buffer(device, 64 * TILE, flags) {
        Ok(buffer) => buffer,
        Err(e) => return fail(format!("the reserved buffer{}: {}", if mode == "opened-cross" { " (cross-adapter)" } else { "" }, refused(&e))),
    };
    let started = Instant::now();
    Mapping::new(&[(0, 2), (9, 2)]).send(&rig.queue, &buffer, &source.heap);
    let call = us(started.elapsed());
    let settled = rig.settle();
    if let Err(why) = standing(device) {
        return fail(format!("{why} by UpdateTileMappings of the heap's tile 2 at the buffer's tiles 0 and 9 ({call:.0} us)"));
    }
    if let Err(e) = settled {
        return fail(format!("the queue did not finish the mapping ({e})"));
    }
    say(format!("heap tile 2 mapped at the buffer's tiles 0 and 9 in {call:.0} us; the device stands"));
    let a = pattern(TILE as usize, 0x21);
    let how = match source.cpu {
        Some(at) => {
            // SAFETY: the heap's tile 2 as the CPU sees it, 64 KiB.
            unsafe { std::ptr::copy_nonoverlapping(a.as_ptr(), at, a.len()) };
            "by the CPU into the heap's memory".to_owned()
        }
        None => match rig.store(&buffer, 0, &a) {
            Ok(()) => "by a GPU copy through offset 0".to_owned(),
            Err(e) => return fail(format!("a GPU copy through offset 0: {e}; {}", standing(device).err().unwrap_or_default())),
        },
    };
    let read = |rig: &mut Rig, tile: u64| rig.fetch(&buffer, tile * TILE, TILE as usize);
    let (first, ninth) = (read(&mut rig, 0), read(&mut rig, 9));
    let same = matches!(&first, Ok(got) if *got == a) && matches!(&ninth, Ok(got) if *got == a);
    if let Err(why) = standing(device) {
        return fail(format!("{why} by GPU reads through the mapping"));
    }
    say(format!(
        "64 KiB written {how}: read the same through tile 0 {}, through tile 9 {}",
        matches!(&first, Ok(got) if *got == a),
        matches!(&ninth, Ok(got) if *got == a)
    ));
    // Write watching over the opened memory: what the GPU writes there.
    if let Some(memory) = source.memory {
        // SAFETY: our own write-watched allocation.
        unsafe { mem::ResetWriteWatch(memory, (16 * TILE) as usize) };
    }
    let b = pattern(TILE as usize, 0x6b);
    let stored = rig.store(&buffer, 9 * TILE, &b);
    let cpu_sees = source.cpu.map(|at| {
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts(at.cast_const(), b.len()) == b.as_slice() }
    });
    let through = read(&mut rig, 0);
    let seen_through = matches!(&through, Ok(got) if *got == b);
    let watched = source.memory.map(|memory| {
        let (mut found, mut count, mut granularity) = ([0usize; 256], 256usize, 0u32);
        // SAFETY: our own write-watched allocation and arrays of the given size.
        let asked = unsafe { mem::GetWriteWatch(0, memory, (16 * TILE) as usize, found.as_mut_ptr().cast(), &mut count, &mut granularity) };
        if asked == 0 { count } else { usize::MAX }
    });
    if let Err(why) = standing(device) {
        return fail(format!("{why} by a GPU write through the mapping"));
    }
    say(format!(
        "a GPU copy written through tile 9 ({}): the CPU sees it in the heap's memory {}; read through tile 0 {seen_through}{}",
        if stored.is_ok() { "done" } else { "FAILED" },
        cpu_sees.map_or("(no CPU access)".to_owned(), |b| b.to_string()),
        match watched {
            Some(usize::MAX) => "; write watching could not be asked".to_owned(),
            Some(n) => format!("; write watching saw {n} pages written of the 16 of 4 KiB the copy wrote"),
            None => String::new(),
        }
    ));
    drop(buffer);
    let memory = source.memory.take();
    drop(source);
    if let Some(memory) = memory {
        // SAFETY: our own allocation; the heap over it is gone.
        unsafe { mem::VirtualFree(memory, 0, mem::MEM_RELEASE) };
    }
    if same && seen_through && stored.is_ok() && cpu_sees != Some(false) {
        say(format!("VERDICT {mode} WORKS: tiles of this heap read and write the same bytes through both offsets"));
    } else {
        fail("the bytes differ through the mapping".to_owned());
    }
}

// P8: a copy-queue pull of one region, against today's CPU copy.

/// Whether this frame's copies are being recorded: the hooks' only load.
static RECORDING: AtomicBool = AtomicBool::new(false);
/// The frame recorded (`crate::frame::current`).
static RECORDED_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);

/// A host copy's CPU copies in the recorded frame.
#[derive(Default, Clone)]
struct Copied {
    /// The title memory it copies: start and bytes.
    start: u64,
    len: u64,
    /// In the card's memory: the CPU copied into its CPU side.
    carded: bool,
    /// The write watch (or the guards) answered for it.
    watched: bool,
    /// Title addresses `[from, to)` copied, and the nanoseconds each took.
    runs: Vec<(usize, usize, u64)>,
    /// Microseconds of the refreshes that copied (with their asking).
    refresh_us: u64,
}

impl Copied {
    fn bytes(&self) -> u64 {
        self.runs.iter().map(|r| (r.1 - r.0) as u64).sum()
    }
}

static COPIES: Mutex<Option<HashMap<u64, Copied>>> = Mutex::new(None);

/// Whether the copies are being recorded ([`copied`], [`refreshed`]).
#[inline]
pub(crate) fn recording() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

/// `crate::shadow`: host copy `id` (title memory `span`) copied title
/// memory `[from, to)` in `ns` nanoseconds of CPU.
pub(crate) fn copied(id: u64, span: (u64, u64), carded: bool, watched: bool, from: usize, to: usize, ns: u64) {
    if crate::frame::current() != RECORDED_FRAME.load(Ordering::Relaxed) {
        return;
    }
    let mut all = COPIES.lock().unwrap_or_else(|e| e.into_inner());
    let c = all
        .get_or_insert_with(HashMap::new)
        .entry(id)
        .or_insert_with(|| Copied { start: span.0, len: span.1, carded, ..Default::default() });
    c.watched |= watched;
    c.runs.push((from, to, ns));
}

/// `crate::host`'s refresh of host copy `id` took `us` microseconds.
pub(crate) fn refreshed(id: u64, us: u64) {
    if crate::frame::current() != RECORDED_FRAME.load(Ordering::Relaxed) {
        return;
    }
    let mut all = COPIES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(c) = all.as_mut().and_then(|all| all.get_mut(&id)) {
        c.refresh_us += us;
    }
}

/// One P8 sample's numbers, for the running medians.
#[derive(Clone, Copy)]
struct Sample {
    /// Today's CPU copy of the runs, and the refreshes that made them.
    copy_ms: f64,
    refresh_ms: f64,
    /// The pull: CPU to record and submit, GPU by timestamps.
    pull_ms: f64,
    gpu_ms: Option<f64>,
    /// The pull's CPU ms a run.
    per_run_ms: f64,
}

#[derive(Default)]
struct State {
    /// The frame being recorded.
    armed: Option<u64>,
    /// The next frame to record from.
    next: u64,
    samples: Vec<Sample>,
    /// The bytes were compared once.
    verified: bool,
    /// The copy queue of ours, made at the first pull (`Err`: refused).
    pull: Option<Result<Pull, String>>,
    /// Samples that found no region, said so (the first few, then fewer).
    none: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// A pull is on the card: the next sample waits for it.
static IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// Called by `crate::host`'s sweep on the thread that submits, before a
/// submission: arms the recording of the next frame's copies every
/// [`EVERY`] frames, and once that frame is over takes its sample.
pub(crate) fn tick() {
    if !on() {
        return;
    }
    let frame = crate::frame::current();
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = state.get_or_insert_with(|| State { next: EVERY, ..Default::default() });
    match state.armed {
        Some(armed) if frame > armed => {
            RECORDING.store(false, Ordering::Relaxed);
            RECORDED_FRAME.store(u64::MAX, Ordering::Relaxed);
            state.armed = None;
            state.next = armed + EVERY;
            let copies = COPIES.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sample(state, armed, &copies))).is_err() {
                tracing::warn!("lot4 probe P8: the sample of frame {armed} stopped on a panic; the run goes on");
            }
        }
        Some(_) => {}
        None if frame >= state.next && !IN_FLIGHT.load(Ordering::Acquire) => {
            // What a copy noted after the last sample's frame was taken.
            *COPIES.lock().unwrap_or_else(|e| e.into_inner()) = None;
            state.armed = Some(frame + 1);
            RECORDED_FRAME.store(frame + 1, Ordering::Relaxed);
            RECORDING.store(true, Ordering::Relaxed);
        }
        None => {}
    }
}

/// The frame's copies, summed: (runs, bytes, CPU ns) from write-watched
/// memory, and from memory not watched.
fn sums(copies: &HashMap<u64, Copied>) -> [(u64, u64, u64); 2] {
    let mut out = [(0, 0, 0); 2];
    for c in copies.values() {
        let o = &mut out[usize::from(!c.watched)];
        o.0 += c.runs.len() as u64;
        o.1 += c.bytes();
        o.2 += c.runs.iter().map(|r| r.2).sum::<u64>();
    }
    out
}

/// The host copies a pull may take, most bytes copied first: those whose
/// memory the write watch (or the guards) answered for, that copied.
fn candidates(copies: &HashMap<u64, Copied>) -> Vec<&Copied> {
    let mut order: Vec<&Copied> = copies.values().filter(|c| c.watched && !c.runs.is_empty()).collect();
    order.sort_by(|a, b| b.bytes().cmp(&a.bytes()).then(a.start.cmp(&b.start)));
    order
}

/// The runs of `c` within [`PULL_SPAN`] of its first, in address order,
/// and the 64 KiB pages that hold them.
fn within(c: &Copied) -> (Vec<(usize, usize, u64)>, (u64, u64)) {
    let mut runs = c.runs.clone();
    runs.sort_by_key(|r| r.0);
    let first = runs.first().map_or(0, |r| r.0 as u64);
    runs.retain(|r| (r.1 as u64) <= first + PULL_SPAN);
    let end = runs.iter().map(|r| r.1 as u64).max().unwrap_or(first);
    (runs, inplace::pages_of_range(first, end - first))
}

/// The sample of `frame`'s `copies`, on the thread that submits.
fn sample(state: &mut State, frame: u64, copies: &HashMap<u64, Copied>) {
    let [watched, unwatched] = sums(copies);
    let mb = |b: u64| b as f64 / 1e6;
    let frame_line = format!(
        "the frame's CPU copies into host copies: {} runs, {:.2} MB, {:.2} ms from write-watched memory; {} runs, {:.2} MB, {:.2} ms from memory \
         not watched{}",
        watched.0,
        mb(watched.1),
        watched.2 as f64 / 1e6,
        unwatched.0,
        mb(unwatched.1),
        unwatched.2 as f64 / 1e6,
        if watched.0 > 2000 { " (over 2,000 runs a frame: one copy-queue region each would not do)" } else { "" }
    );
    let Some(host) = crate::host::host() else {
        return;
    };
    let (mut views, mut not_whole) = (0, 0);
    let mut chosen = None;
    let mut refusal = None;
    for c in candidates(copies).into_iter().take(TRIES) {
        if inplace::cover(c.start as usize, c.len.max(1) as usize) != inplace::Cover::None {
            views += 1;
            continue;
        }
        let (runs, (at, end)) = within(c);
        let started = Instant::now();
        match inplace::place_part(&host.device, at, end - at) {
            Ok(placed) => {
                chosen = Some((c, runs, at, placed, started.elapsed()));
                break;
            }
            Err(None) => not_whole += 1,
            Err(Some(e)) => {
                refusal = Some(e);
                break;
            }
        }
    }
    let Some((c, runs, at, placed, open)) = chosen else {
        state.none += 1;
        if state.none <= 4 || state.none.is_power_of_two() {
            tracing::info!(
                "lot4 probe P8: frame {frame}: no region to pull: of the host copies refreshed from write-watched memory (most bytes first, {TRIES} \
                 tried), {views} are view memory (not private), {not_whole} not private memory committed whole{}; {frame_line}",
                refusal.map_or(String::new(), |e| format!(", and the host refused to open the next: {}", flat(&e.to_string())))
            );
        }
        return;
    };
    let pull = match state.pull.get_or_insert_with(|| Pull::new(&host.device).map_err(|e| flat(&e.to_string()))) {
        Ok(pull) => pull,
        Err(why) => {
            tracing::info!("lot4 probe P8: no copy queue of ours ({why}); {frame_line}");
            return;
        }
    };
    let bytes: u64 = runs.iter().map(|r| (r.1 - r.0) as u64).sum();
    if let Err(e) = pull.room(&host.device, bytes) {
        tracing::info!("lot4 probe P8: no destination buffer of {bytes} bytes ({}); {frame_line}", flat(&e.to_string()));
        return;
    }
    let spans: Vec<(usize, usize)> = runs.iter().map(|r| (r.0, r.1)).collect();
    // The title's bytes just before, for the one comparison.
    let before = (!state.verified).then(|| title_bytes(&spans));
    let sent = match pull.send(&placed.buffer, at, &spans) {
        Ok(sent) => sent,
        Err(e) => {
            tracing::info!("lot4 probe P8: the pull was not sent ({}); {frame_line}", flat(&e.to_string()));
            return;
        }
    };
    let copy_ms = runs.iter().map(|r| r.2).sum::<u64>() as f64 / 1e6;
    let Some(landing) = pull.landing(sent.0) else {
        return;
    };
    let job = Job {
        frame,
        start: c.start,
        len: c.len,
        carded: c.carded,
        runs: spans,
        bytes,
        copy_ms,
        refresh_ms: c.refresh_us as f64 / 1e3,
        open_ms: ms(open),
        frame_line,
        frame_runs: watched.0,
        before,
        placed,
        landing,
        pull_ms: ms(sent.1),
        submitted: Instant::now(),
    };
    // Cleared once the pull has landed; a pull that never lands leaves it
    // set, and P8 stops (its list may still be on the card).
    IN_FLIGHT.store(true, Ordering::Release);
    hand(job);
}

/// The thread that waits for the pulls ([`Job::land`]), made at the first.
static WAITER: OnceLock<Option<Mutex<std::sync::mpsc::Sender<Job>>>> = OnceLock::new();

/// Gives `job` to the waiting thread; without one, waits for it here and
/// leaves its line out (the caller holds the state's lock).
fn hand(job: Job) {
    let waiter = WAITER.get_or_init(|| {
        let (jobs, taken) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("victory-lot4-p8".to_owned())
            .spawn(move || {
                for job in taken {
                    if job.land() {
                        IN_FLIGHT.store(false, Ordering::Release);
                    }
                }
            })
            .ok()
            .map(|_| Mutex::new(jobs))
    });
    let job = match waiter {
        Some(jobs) => match jobs.lock().unwrap_or_else(|e| e.into_inner()).send(job) {
            Ok(()) => return,
            Err(std::sync::mpsc::SendError(job)) => job,
        },
        None => job,
    };
    tracing::info!("lot4 probe P8: no thread to wait for the pull: waited for here, the sample left out");
    match job.wait() {
        Ok(()) => {
            drop(job);
            IN_FLIGHT.store(false, Ordering::Release);
        }
        // Its list may still read through the heap: kept for good.
        Err(_) => std::mem::forget(job),
    }
}

/// The title's bytes of `spans`, one after the other (read as the
/// refresh check reads them: no copy counter counts them).
fn title_bytes(spans: &[(usize, usize)]) -> Vec<u8> {
    let mut out = vec![0u8; spans.iter().map(|s| s.1 - s.0).sum()];
    let mut at = 0;
    for &(from, to) in spans {
        crate::host::read_into(from as *const u8, &mut out[at..at + (to - from)]);
        at += to - from;
    }
    out
}

/// A pointer that crosses to the thread that waits.
struct Ptr(*mut u8);
// SAFETY: it points into a mapped buffer the job holds, read only there.
unsafe impl Send for Ptr {}

/// What the waiting thread needs to see a pull land.
struct Landing {
    fence: ID3D12Fence,
    value: u64,
    /// The timestamps' read-back buffer and the copy queue's ticks a ms.
    stamps: Option<(ID3D12Resource, f64)>,
    /// The destination, held while `data` (its mapping) is read.
    _dest: ID3D12Resource,
    data: Ptr,
}

/// A pull sent, on its way to its line.
struct Job {
    frame: u64,
    start: u64,
    len: u64,
    carded: bool,
    runs: Vec<(usize, usize)>,
    bytes: u64,
    copy_ms: f64,
    refresh_ms: f64,
    open_ms: f64,
    frame_line: String,
    frame_runs: u64,
    before: Option<Vec<u8>>,
    /// The heap over the title's memory: held until the pull has landed.
    placed: inplace::Placed,
    landing: Landing,
    pull_ms: f64,
    submitted: Instant,
}

impl Job {
    /// Waits (five seconds at most) until the card has done the pull.
    fn wait(&self) -> Result<(), String> {
        let l = &self.landing;
        // SAFETY: an event of our own, closed below.
        let event = unsafe { CreateEventW(None, false, false, None) }.map_err(|e| e.to_string())?;
        // SAFETY: our fence and event.
        unsafe {
            let set = l.fence.SetEventOnCompletion(l.value, event).map_err(|e| e.to_string());
            let r = if set.is_ok() { WaitForSingleObject(event, 5_000).0 } else { 0 };
            let _ = windows::Win32::Foundation::CloseHandle(event);
            set?;
            if r == 0 { Ok(()) } else { Err("not done after 5 s".to_owned()) }
        }
    }

    /// Waits for the pull, reads its timestamps, compares the bytes once,
    /// lets the heap go and says the sample's line. False if it did not
    /// land.
    fn land(self) -> bool {
        let waited = self.wait();
        let done_ms = ms(self.submitted.elapsed());
        if let Err(why) = waited {
            tracing::warn!("lot4 probe P8: frame {}: the pull did not land ({}); P8 stops here", self.frame, flat(&why));
            // Its list may still read through the heap: kept for good.
            std::mem::forget(self);
            return false;
        }
        let l = &self.landing;
        let gpu_ms = l.stamps.as_ref().and_then(|(readback, per_ms)| {
            let mut data = std::ptr::null_mut();
            // SAFETY: the read-back buffer of two timestamps, written by
            // the pull that has landed.
            unsafe {
                readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut data)).ok()?;
                let t = data.cast::<[u64; 2]>().read();
                readback.Unmap(0, Some(&D3D12_RANGE::default()));
                (t[1] >= t[0] && *per_ms > 0.0).then(|| (t[1] - t[0]) as f64 / per_ms)
            }
        });
        let verified = self.before.as_ref().map(|before| {
            // SAFETY: the destination holds the runs' bytes one after the
            // other, `before.len()` of them, landed.
            let pulled = unsafe { std::slice::from_raw_parts(l.data.0.cast_const(), before.len()) };
            let after = title_bytes(&self.runs);
            let equal = (0..before.len()).filter(|&i| pulled[i] == before[i] || pulled[i] == after[i]).count();
            let exact = pulled == after.as_slice();
            (equal, before.len(), exact)
        });
        drop(self.placed);
        let per_run_ms = self.pull_ms / self.runs.len().max(1) as f64;
        let sample = Sample { copy_ms: self.copy_ms, refresh_ms: self.refresh_ms, pull_ms: self.pull_ms, gpu_ms, per_run_ms };
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(state) = state.as_mut() else {
            return true;
        };
        state.samples.push(sample);
        if verified.is_some() {
            state.verified = true;
        }
        let s = &state.samples;
        let pick = |f: fn(&Sample) -> f64| median(&s.iter().map(f).collect::<Vec<_>>());
        let gpus: Vec<f64> = s.iter().filter_map(|s| s.gpu_ms).collect();
        let (copy, pull) = (pick(|s| s.copy_ms), pick(|s| s.pull_ms));
        let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { f64::INFINITY };
        let per_run = pick(|s| s.per_run_ms);
        tracing::info!(
            "lot4 probe P8: sample {} (frame {}): a host copy of title memory at {:#x}+{:#x} ({}; private, write-watched, opened in place), {} runs, \
             {:.3} MB: (a) today's CPU copy {:.3} ms (all its refreshes that copied, with their write-watch asking, {:.3} ms); (b) a copy-queue pull from a heap over \
             that memory {:.3} ms CPU to record and submit, GPU {} by timestamps, landed {:.2} ms after the submit; the heap opened in {:.2} ms (once \
             a region if kept); (a)/(b) {:.1}x | running medians over {} samples: (a) {:.3} ms, refresh {:.3} ms, (b) {:.3} ms CPU, GPU {}, (a)/(b) \
             {:.1}x -> {} | at (b)'s median CPU a run, the frame's {} runs from write-watched memory would take {:.2} ms to pull | {}",
            s.len(),
            self.frame,
            self.start,
            self.len,
            if self.carded { "in the card's memory: the CPU copies into its CPU side" } else { "a copy in the PC's memory" },
            self.runs.len(),
            self.bytes as f64 / 1e6,
            self.copy_ms,
            self.refresh_ms,
            self.pull_ms,
            gpu_ms.map_or("n/a".to_owned(), |g| format!("{g:.3} ms")),
            done_ms,
            self.open_ms,
            ratio(self.copy_ms, self.pull_ms),
            s.len(),
            copy,
            pick(|s| s.refresh_ms),
            pull,
            if gpus.is_empty() { "n/a".to_owned() } else { format!("{:.3} ms", median(&gpus)) },
            ratio(copy, pull),
            if ratio(copy, pull) >= 2.0 { "PASS (2x cheaper in CPU ms or more)" } else { "DEAD (not 2x cheaper in CPU ms)" },
            self.frame_runs,
            per_run * self.frame_runs as f64,
            self.frame_line
        );
        if let Some((equal, all, exact)) = verified {
            tracing::info!(
                "lot4 probe P8: verify (once): {equal} of {all} bytes pulled equal the title's bytes read just before or just after the pull; equal to \
                 those after exactly: {exact} (the title may write meanwhile)"
            );
        }
        true
    }
}

/// P8's copy queue, list, fence, timestamps and destination.
struct Pull {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: u64,
    /// Two timestamps, their read-back buffer and the queue's ticks a ms.
    stamps: Option<(ID3D12QueryHeap, ID3D12Resource, f64)>,
    /// A buffer of the kind the CPU copies into (the PC's memory, write
    /// back, which the CPU maps), mapped, and its bytes.
    dest: Option<(ID3D12Resource, *mut u8, u64)>,
}

// SAFETY: D3D12 objects are free-threaded; the mapped pointer is used by
// whoever holds the state's lock, or the job that holds the buffer.
unsafe impl Send for Pull {}

impl Pull {
    fn new(device: &ID3D12Device) -> windows::core::Result<Pull> {
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let mut o3 = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            let stamps = if query(device, D3D12_FEATURE_D3D12_OPTIONS3, &mut o3).is_ok() && o3.CopyQueueTimestampQueriesSupported.as_bool() {
                let mut heap: Option<ID3D12QueryHeap> = None;
                let made = device.CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP, Count: 2, NodeMask: 0 },
                    &mut heap,
                );
                let readback = inplace_committed(device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_STATE_COPY_DEST);
                let ticks = queue.GetTimestampFrequency();
                match (made, heap, readback, ticks) {
                    (Ok(()), Some(heap), Ok(readback), Ok(ticks)) if ticks > 0 => Some((heap, readback, ticks as f64 / 1e3)),
                    _ => None,
                }
            } else {
                None
            };
            Ok(Pull { queue, allocator, list, fence, done: 0, stamps, dest: None })
        }
    }

    /// A destination of `bytes` or more.
    fn room(&mut self, device: &ID3D12Device, bytes: u64) -> windows::core::Result<()> {
        if self.dest.as_ref().is_some_and(|d| d.2 >= bytes) {
            return Ok(());
        }
        let size = bytes.max(1).next_multiple_of(TILE << 4);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer of ours.
        unsafe {
            device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)?;
            let buffer = buffer.ok_or_else(windows::core::Error::empty)?;
            buffer.Map(0, None, Some(&mut data))?;
            self.dest = Some((buffer, data.cast(), size));
        }
        Ok(())
    }

    /// Records and submits one `CopyBufferRegion` per span `[from, to)`
    /// of title memory, read from `source` (a buffer placed over the
    /// 64 KiB pages from title address `at`), into the destination one
    /// after the other, between two timestamps: the fence value to wait
    /// for, and the CPU time from the list's reset to the signal.
    fn send(&mut self, source: &ID3D12Resource, at: u64, spans: &[(usize, usize)]) -> windows::core::Result<(u64, Duration)> {
        let dest = self.dest.as_ref().map(|d| d.0.clone()).ok_or_else(windows::core::Error::empty)?;
        let started = Instant::now();
        // SAFETY: our own allocator and list (the last pull has landed:
        // `IN_FLIGHT`); live buffers; each span lies in the source's pages
        // and the destination holds them all (`room`).
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            if let Some((heap, _, _)) = &self.stamps {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            }
            let mut to_at = 0u64;
            for &(from, to) in spans {
                let n = (to - from) as u64;
                self.list.CopyBufferRegion(&dest, to_at, source, from as u64 - at, n);
                to_at += n;
            }
            if let Some((heap, readback, _)) = &self.stamps {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, readback, 0);
            }
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.done += 1;
            self.queue.Signal(&self.fence, self.done)?;
        }
        Ok((self.done, started.elapsed()))
    }

    /// What the waiting thread needs for the pull signalled at `value`
    /// (`None` before a destination is made).
    fn landing(&self, value: u64) -> Option<Landing> {
        let (dest, data) = self.dest.as_ref().map(|d| (d.0.clone(), d.1))?;
        Some(Landing { fence: self.fence.clone(), value, stamps: self.stamps.as_ref().map(|s| (s.1.clone(), s.2)), _dest: dest, data: Ptr(data) })
    }
}

/// A committed buffer of `size` bytes in a heap of `kind`.
fn inplace_committed(device: &ID3D12Device, kind: D3D12_HEAP_TYPE, size: u64, state: D3D12_RESOURCE_STATES) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE), state, None, &mut buffer)? };
    buffer.ok_or_else(windows::core::Error::empty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_one_and_nothing_else() {
        assert!(switch_on(Some("1")));
        assert!(switch_on(Some(" 1 ")));
        assert!(!switch_on(None));
        assert!(!switch_on(Some("0")));
        assert!(!switch_on(Some("yes")));
        assert!(!switch_on(Some("2")));
    }

    #[test]
    fn a_message_with_line_breaks_is_one_line() {
        assert_eq!(flat("driver state suspect,\r\nsee the log\r\n"), "driver state suspect,  see the log");
    }

    #[test]
    fn medians_of_odd_even_and_no_samples() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[]), 0.0);
    }

    /// The ranges P0 times: as many distinct buffer tiles as ranges, inside
    /// the buffer, and every heap tile mapped at several of them.
    #[test]
    fn ranges_spread_over_the_buffer_and_share_heap_tiles() {
        for (n, tiles) in [(16_384u32, 65_536u64), (1024, 65_536), (64, 32_768), (1, 64), (16_384, 16_384)] {
            let pairs = spread(n, tiles);
            assert_eq!(pairs.len(), n as usize);
            let mut at: Vec<u32> = pairs.iter().map(|p| p.0).collect();
            at.sort_unstable();
            at.dedup();
            assert_eq!(at.len(), n as usize, "distinct buffer tiles");
            assert!(pairs.iter().all(|p| u64::from(p.0) < tiles && p.1 < HEAP_TILES));
        }
        let pairs = spread(16_384, 65_536);
        let zero = pairs.iter().filter(|p| p.1 == 0).count();
        assert_eq!(zero, 16_384 / HEAP_TILES as usize, "heap tile 0 at as many offsets as the others");
    }

    #[test]
    fn a_luid_goes_to_the_child_and_back() {
        let luid = LUID { LowPart: 0x89ab_cdef, HighPart: -2 };
        let back = luid_parse(&luid_text(luid)).expect("parsed");
        assert_eq!((back.LowPart, back.HighPart), (luid.LowPart, luid.HighPart));
        assert!(luid_parse("not hex").is_none());
    }

    #[test]
    fn a_childs_cost_and_alias_lines_are_read() {
        assert_eq!(parse_cost("16384 14023.5 2101.0"), Some((16_384, 14_023.5, 2_101.0)));
        assert_eq!(parse_cost("16384 x"), None);
        assert_eq!(parse_alias("copy=yes shader=no"), (Some(true), Some(false)));
        assert_eq!(parse_alias("shader=yes"), (None, Some(true)));
    }

    #[test]
    fn a_childs_lines_are_picked_out_of_its_output() {
        let text = "running 1 test\ntest lot4::device::tests::child_entry ... lot4 child: heap of 1024 KiB\r\nnoise\nlot4 child: VERDICT pc-heap WORKS: yes\nok\n";
        assert_eq!(child_lines(text), vec!["heap of 1024 KiB".to_owned(), "VERDICT pc-heap WORKS: yes".to_owned()]);
    }

    fn p0_alive() -> P0 {
        P0 {
            tier: 3,
            reserved: 16 * GIB,
            alias_copy: Some(true),
            alias_shader: Some(true),
            cost: vec![(1, 20.0, 50.0), (1024, 900.0, 300.0), (16_384, 14_000.0, 2_000.0)],
            system: MODES.iter().map(|&m| (m, Some(true))).collect(),
            recreate: Some(0),
            shared_budget: Some(16 << 30),
            ..Default::default()
        }
    }

    /// The verdict names what kills the mirror and what only limits it.
    #[test]
    fn the_verdict_says_what_kills_the_mirror() {
        let last = |p: P0| {
            let mut p = p;
            verdict(&mut p);
            p.lines.pop().unwrap()
        };
        let alive = last(p0_alive());
        assert!(alive.contains("the mirror ALIVE") && !alive.contains("limited"), "{alive}");
        let differ = last(P0 { alias_copy: Some(false), ..p0_alive() });
        assert!(differ.contains("ALIVE; limited: one tile at several offsets of the 16 GiB buffer failed where a 4 MiB buffer works"), "{differ}");
        let small = |system: Vec<(&'static str, Option<bool>)>| P0 { alias_copy: Some(false), system, ..p0_alive() };
        let neither = last(small(vec![("card-heap", Some(false)), ("card-heap-big", Some(false))]));
        assert!(neither.contains("DEAD: one tile of a heap in the card's memory does not read the same at two offsets"), "{neither}");
        assert!(neither.contains("one tile at several offsets of the large buffer does not read the same"), "{neither}");
        let untried = last(P0 { system: Vec::new(), alias_copy: None, alias_shader: None, ..p0_alive() });
        assert!(untried.contains("the mirror UNKNOWN"), "{untried}");
        let none = last(P0 { reserved: 0, ..p0_alive() });
        assert!(none.contains("DEAD: no reserved buffer"), "{none}");
        let slow = last(P0 { cost: vec![(16_384, 90_000.0, 0.0)], ..p0_alive() });
        assert!(slow.contains("22.5 ms for 4,096 ranges"), "{slow}");
        let small = last(P0 { reserved: 2 * GIB, ..p0_alive() });
        assert!(small.contains("ALIVE; limited: reserved buffers only up to 2 GiB"), "{small}");
        let vram = last(P0 { system: MODES.iter().map(|&m| (m, Some(m.starts_with("card")))).collect(), ..p0_alive() });
        assert!(vram.contains("ALIVE; limited: no tiles in the PC's memory"), "{vram}");
        let ours = last(P0 { system: vec![("card-heap", Some(true)), ("pc-heap-cpu", Some(true)), ("opened", Some(false))], ..p0_alive() });
        assert!(ours.contains("only from heaps of ours"), "{ours}");
    }

    fn copied_of(start: u64, watched: bool, runs: &[(usize, usize)]) -> Copied {
        Copied { start, len: 1 << 20, watched, runs: runs.iter().map(|&(a, b)| (a, b, 1000)).collect(), ..Default::default() }
    }

    /// P8 takes watched copies only, most bytes first.
    #[test]
    fn the_pull_takes_the_watched_copy_that_copied_most() {
        let mut copies = HashMap::new();
        copies.insert(1, copied_of(0x1000_0000, true, &[(0x1000_0000, 0x1001_0000)]));
        copies.insert(2, copied_of(0x2000_0000, false, &[(0x2000_0000, 0x2100_0000)]));
        copies.insert(3, copied_of(0x3000_0000, true, &[(0x3000_0000, 0x3002_0000), (0x3004_0000, 0x3004_1000)]));
        copies.insert(4, copied_of(0x4000_0000, true, &[]));
        let order: Vec<u64> = candidates(&copies).iter().map(|c| c.start).collect();
        assert_eq!(order, vec![0x3000_0000, 0x1000_0000]);
        let [w, u] = sums(&copies);
        assert_eq!((w.0, w.1, w.2), (3, 0x1_0000 + 0x2_0000 + 0x1000, 3000));
        assert_eq!((u.0, u.1), (1, 0x100_0000));
    }

    /// The runs pulled lie within the span from the first, in address
    /// order, and the pages around them start on a 64 KiB line.
    #[test]
    fn the_runs_pulled_lie_within_the_span() {
        let far = 0x5000_0000 + PULL_SPAN as usize + 0x1000;
        let c = copied_of(0x5000_0000, true, &[(0x5000_2000, 0x5000_3000), (far, far + 0x100), (0x5000_0100, 0x5000_0200)]);
        let (runs, (at, end)) = within(&c);
        assert_eq!(runs.iter().map(|r| (r.0, r.1)).collect::<Vec<_>>(), vec![(0x5000_0100, 0x5000_0200), (0x5000_2000, 0x5000_3000)]);
        assert_eq!((at, end), (0x5000_0000, 0x5001_0000));
    }

    /// The child's entry for the tests below (they start this test binary
    /// with only this test and the child's mode in the environment).
    #[test]
    fn child_entry() {
        if std::env::var_os(CHILD).is_some() {
            assert!(child());
        }
    }

    /// The software adapter's device (this process's one; a card is never
    /// opened by the tests).
    fn warp() -> Option<ID3D12Device> {
        let adapter = crate::host::warp_adapter().ok()?;
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain device creation on the software adapter.
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }.ok()?;
        device
    }

    fn test_child() -> Child {
        Child {
            exe: std::env::current_exe().expect("the test binary"),
            args: vec!["--exact".to_owned(), "lot4::device::tests::child_entry".to_owned(), "--nocapture".to_owned(), "--test-threads=1".to_owned()],
        }
    }

    /// P0 whole on the software adapter: every child ends and most give a
    /// verdict; one tile reads the same at two offsets of a small buffer;
    /// and whatever a child did to its device (the software adapter
    /// removes it for the large buffer's test and the opened heap's
    /// mapping), this process's stands. (On the card a run with
    /// `VICTORY_LOT4_PROBE=1` is the measurement.)
    #[test]
    fn warp_p0_maps_tiles_in_children_and_this_device_stands() {
        let Some(device) = warp() else { return };
        let p = probe(&device, None, Some(&test_child()));
        for line in &p.lines {
            eprintln!("lot4 probe P0: {line}");
        }
        // SAFETY: a plain query.
        assert!(unsafe { device.GetDeviceRemovedReason() }.is_ok(), "this process's device stands");
        let text = p.lines.join("\n");
        assert!(text.contains("VERDICT: reserved buffer"), "{text}");
        assert!(text.contains("QueryVideoMemoryInfo: no answer"), "{text}");
        if p.tier >= 1 {
            let small = p.system.iter().find(|s| s.0 == "card-heap").and_then(|s| s.1);
            assert_eq!(small, Some(true), "one tile at two offsets of a small buffer: {text}");
        }
        assert_eq!(p.system.len(), MODES.len());
        for mode in MODES {
            assert!(text.contains(&format!("[{mode}] the child ")), "how the child of {mode} ended: {text}");
        }
        // A child that ends in a crash says no verdict; the protocol is
        // shown by those that do (the heaps of ours, at least, answer).
        assert!(p.system.iter().any(|s| s.1.is_some()), "at least one child's verdict: {text}");
    }

    /// P8's pull on the software adapter: runs of private, write-watched
    /// memory opened in place, pulled by a copy queue into our buffer, land
    /// with the memory's bytes, one after the other.
    #[test]
    fn warp_p8_pulls_the_runs_title_memory_holds() {
        let Some(device) = warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            return;
        }
        let size = 8 * TILE as usize;
        // SAFETY: an allocation of the test's own, made as `xmem` makes
        // the title's private memory.
        let base = unsafe { mem::VirtualAlloc(std::ptr::null(), size, mem::MEM_RESERVE | mem::MEM_COMMIT | 0x20_0000, mem::PAGE_READWRITE) };
        assert!(!base.is_null());
        let bytes = pattern(size, 0x37);
        // SAFETY: inside the allocation.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.cast::<u8>(), size) };
        let b = base as usize;
        let spans = vec![(b + 0x1000 + 3, b + 0x3000), (b + 2 * TILE as usize + 100, b + 5 * TILE as usize), (b + 7 * TILE as usize, b + size)];
        let (at, end) = inplace::pages_of_range(spans[0].0 as u64, (spans[2].1 - spans[0].0) as u64);
        let placed = inplace::place_part(&device, at, end - at).unwrap_or_else(|e| panic!("not opened: {e:?}"));
        let mut pull = Pull::new(&device).expect("a copy queue");
        let total: u64 = spans.iter().map(|s| (s.1 - s.0) as u64).sum();
        pull.room(&device, total).expect("a destination");
        let (value, cpu) = pull.send(&placed.buffer, at, &spans).expect("sent");
        let landing = pull.landing(value).expect("a destination");
        // SAFETY: our fence and an event of our own.
        unsafe {
            let event = CreateEventW(None, false, false, None).unwrap();
            landing.fence.SetEventOnCompletion(value, event).unwrap();
            assert_eq!(WaitForSingleObject(event, 20_000).0, 0, "landed");
            let _ = windows::Win32::Foundation::CloseHandle(event);
        }
        let want = title_bytes(&spans);
        // SAFETY: the destination holds the spans' bytes, landed.
        let got = unsafe { std::slice::from_raw_parts(landing.data.0.cast_const(), want.len()) };
        assert!(got == want.as_slice(), "the pulled bytes are the memory's");
        eprintln!("pull of {} spans, {total} bytes: {:.3} ms CPU; timestamps on the copy queue {}", spans.len(), ms(cpu), pull.stamps.is_some());
        drop((landing, placed, pull));
        // SAFETY: the test's own allocation, nothing opened on it now.
        unsafe { mem::VirtualFree(base, 0, mem::MEM_RELEASE) };
    }
}
