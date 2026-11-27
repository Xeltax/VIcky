//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; unset or `1`: the four water formats found by
//! the upstream trace; `0`: none), a list in which shaders write one copies it
//! into a read-back buffer of ours when it closes, and once the list has run
//! the texels are tiled into the texture's title memory in the console's
//! layout, as the console's GPU would have left them.
//!
//! VIcky: on by default, as in TaynanGT's fork (`23925a4aa2`). Mind the
//! value: in Geckomacabre/vx-layer `eee5667d` `=1` was an empty list (off),
//! and in fxceless/victory-fork `f5c03f3b0f` unset was off.
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.
//!
//! The read-back buffers are kept and used again ([`read_back_buffer`]):
//! every list that copies back made one and let it go at its reset, about
//! 770 a second (fxceless/victory-fork issue #1340: `copied=965888` in a
//! 21-minute run), and on Linux with NVIDIA's open kernel module that churn
//! ran the driver out of DMA mappings. After fork PR #1359 (fxceless).
//!
//! A thread that lands copy-backs keeps its buffer for the title's bytes
//! and the tiling geometry of the textures it tiled ([`Scratch`]), after
//! TaynanGT f0019f2e20 (Taynan Piza): a buffer of the texture's
//! size (up to 512 KiB) was made, and the runs of its layout found again,
//! at every landing.
//!
//! Two opt-in switches, each `1` or `alt` (twenty seconds off, twenty on,
//! in turn, as `VICTORY_LIGHT_REFRESH=alt`; each change said in the log):
//!
//! - `VICTORY_TEXTURE_BACK_LEAN`: a copy that a later copy of the same
//!   memory in the same flight supersedes is not landed (nothing tells the
//!   title anything between the read-backs of one flight's lists), and
//!   texels title memory holds already are not written again (the write
//!   opens the guards over them and makes the textures placed there due
//!   again). What the title can read, and when, is the same.
//! - `VICTORY_TEXTURE_BACK_AHEAD`: the memory copy-backs write no longer
//!   holds back a submission sent ahead whose preparing reads it (a texture
//!   of the title's in the same memory, a buffer copy, a fill): its
//!   flights are not waited for. What the title's CPU reads is the same:
//!   a copy-back still lands before anything behind its flight is told to
//!   the title. What changes is what such a preparing takes: the texels of
//!   the last landing, older by the flights in the air (about a frame; on
//!   the console a pass after the compute pass reads what it wrote), and,
//!   if it reads while a landing writes, a mix until the next landing.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_KINDS: std::cell::Cell<Option<&'static [(u64, u32, u32)]>> =
        const { std::cell::Cell::new(None) };
}

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    #[cfg(test)]
    if let Some(kinds) = TEST_KINDS.get() {
        return kinds;
    }
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let kinds = parse_kinds(std::env::var("VICTORY_TEXTURE_BACK").ok().as_deref());
        if !kinds.is_empty() {
            tracing::info!(?kinds, "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)");
        } else {
            tracing::info!("no texture shaders write is copied back into title memory (VICTORY_TEXTURE_BACK=0)");
        }
        kinds
    })
}

/// The kinds `VICTORY_TEXTURE_BACK` asks for (`None`: unset): unset or `1`,
/// the four water kinds; `0` (or anything that names no kind), none.
fn parse_kinds(spec: Option<&str>) -> Vec<(u64, u32, u32)> {
    const DEFAULT: &str = "256x256f41,256x256f16,64x64f16,128x128f36";
    let list = match spec.map(str::trim) {
        None | Some("1") => DEFAULT,
        Some(spec) => spec,
    };
    list.split(',')
        .filter_map(|k| {
            let (size, format) = k.trim().split_once('f')?;
            let (w, h) = size.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
        })
        .collect()
}

/// An opt-in switch: unset or `0` off, `1` on, `alt` twenty seconds off
/// and twenty on, in turn from the first asking, each change said in the
/// log as `<what> from now (<name>=alt): ON` (or `off, as usual`).
struct Switch {
    name: &'static str,
    what: &'static str,
    mode: OnceLock<u8>,
    epoch: OnceLock<std::time::Instant>,
    last: AtomicU8,
}

impl Switch {
    const fn new(name: &'static str, what: &'static str) -> Self {
        Self { name, what, mode: OnceLock::new(), epoch: OnceLock::new(), last: AtomicU8::new(2) }
    }

    fn mode(&self) -> u8 {
        *self.mode.get_or_init(|| {
            let mode = match std::env::var(self.name).as_deref().map(str::trim) {
                Ok("1") => 1,
                Ok("alt") => 2,
                _ => 0,
            };
            if mode == 1 {
                tracing::info!("{} ({}=1)", self.what, self.name);
            }
            mode
        })
    }

    fn on(&self) -> bool {
        match self.mode() {
            0 => false,
            1 => true,
            _ => {
                let on = (self.epoch.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
                if self.last.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                    tracing::info!("{} from now ({}=alt): {}", self.what, self.name, if on { "ON" } else { "off, as usual" });
                }
                on
            }
        }
    }

    /// Its state, for the frame-rate line.
    fn said(&self) -> &'static str {
        match (self.mode(), self.last.load(Ordering::Relaxed)) {
            (0, _) => "off",
            (1, _) => "on",
            (_, 1) => "alt, ON",
            _ => "alt, off",
        }
    }
}

static LEAN: Switch = Switch::new(
    "VICTORY_TEXTURE_BACK_LEAN",
    "texture copy-backs land lean: a copy a later one of the same memory in the same flight supersedes is not landed, texels title memory holds already are not written again",
);
static AHEAD: Switch = Switch::new(
    "VICTORY_TEXTURE_BACK_AHEAD",
    "submissions that read memory texture copy-backs in the air write go ahead without waiting for them: they read the last landing's texels",
);

/// Whether copy-backs land lean (`VICTORY_TEXTURE_BACK_LEAN`).
pub(crate) fn lean() -> bool {
    LEAN.on()
}

/// Whether the memory copy-backs in the air write holds back a submission
/// whose preparing reads it (unless `VICTORY_TEXTURE_BACK_AHEAD`).
pub(crate) fn holding() -> bool {
    !AHEAD.on()
}

/// Since the last frame-rate line ([`summary`]): copies recorded and their
/// bytes; copies landed into title memory and their bytes; copies left out
/// as superseded and as unchanged (`VICTORY_TEXTURE_BACK_LEAN`);
/// microseconds landing; settles for a submission that reads memory
/// copy-backs in the air write, and the microseconds they waited;
/// such submissions that went ahead (`VICTORY_TEXTURE_BACK_AHEAD`).
pub(crate) static STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
pub(crate) const SUPERSEDED: usize = 4;
pub(crate) const SETTLES: usize = 7;
pub(crate) const WAITED_US: usize = 8;
pub(crate) const WENT_AHEAD: usize = 9;

/// Whether a copy-back was ever recorded (the frame-rate line says
/// nothing of them before).
static EVER: AtomicBool = AtomicBool::new(false);

/// The copy-backs' part of the frame-rate line, per frame since the line
/// before; empty before the first copy-back.
pub(crate) fn summary(frames: f64) -> String {
    let frames = frames.max(1.0);
    let [copied, copied_bytes, landed, bytes, superseded, unchanged, landing_us, settles, waited_us, ahead] =
        std::array::from_fn(|i| STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
    if !EVER.load(Ordering::Relaxed) {
        return String::new();
    }
    format!(
        " texture copy-back: {copied:.1} copied {:.1} MB, {landed:.1} landed {:.1} MB in {:.2} ms (lean {}: {superseded:.1} \
         superseded, {unchanged:.1} unchanged), {settles:.2} settles for a read of its memory, {:.1} ms waiting for the card \
         (ahead {}: {ahead:.2} such reads went ahead);",
        copied_bytes / 1e6,
        bytes / 1e6,
        landing_us / 1e3,
        LEAN.said(),
        waited_us / 1e3,
        AHEAD.said()
    )
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket { owner, at, end, valid: AtomicBool::new(true) });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| {
        let Some(ticket) = weak.upgrade() else { return false };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Release);
        }
        true
    });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy independently of the title's resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let _mapping = crate::watch::title_memory_reading();
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self { copy, layout, ticket })
    }

    pub(crate) fn owner(&self) -> usize { self.ticket.owner }

    /// Whether the ticket is still valid: the texture's memory has not been
    /// reused or remapped since it was registered. A stale target is
    /// re-registered so the read-back tracks the memory the texture is in
    /// now, not where it was when it was last written.
    pub(crate) fn fresh(&self) -> bool { self.ticket.valid.load(Ordering::Acquire) }
}

/// A read-back buffer nothing holds.
struct Spare(ID3D12Resource);

// SAFETY: a D3D12 buffer (free-threaded), touched by nobody while it is
// spare.
unsafe impl Send for Spare {}

/// Spare read-back buffers by size (a power of two), and their bytes in all.
static SPARE: Mutex<(BTreeMap<u64, Vec<Spare>>, u64)> = Mutex::new((BTreeMap::new(), 0));
/// The most bytes kept spare unless `VICTORY_READBACK_POOL_MB` says, and
/// the largest buffer that is pooled.
const SPARE_MOST: (u64, u64) = (256 << 20, 16 << 20);
/// Read-back buffers taken from the spare ones, and made, since the start.
static SPARE_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether read-back buffers are kept and used again (on unless
/// `VICTORY_READBACK_POOL=0`: made for each copy and let go, as before).
fn pool_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_READBACK_POOL").is_none_or(|v| v != "0"))
}

/// The most bytes of read-back buffers kept spare (`VICTORY_READBACK_POOL_MB`).
fn spare_most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_READBACK_POOL_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map_or(SPARE_MOST.0, |mb| mb << 20)
    })
}

/// The size a buffer of `total` bytes is pooled at; 0: not pooled.
fn pool_size(total: u64, on: bool) -> u64 {
    let size = total.max(64 << 10).next_power_of_two();
    if on && size <= SPARE_MOST.1 { size } else { 0 }
}

/// A read-back buffer of at least `total` bytes, in COPY_DEST (where a
/// read-back buffer always stays), and its pool size (0: not pooled).
/// Nothing needs clearing: [`land`] reads only the footprints the list's
/// copies wrote.
fn read_back_buffer(host: &crate::host::Host, total: u64) -> Option<(ID3D12Resource, u64)> {
    let size = pool_size(total, pool_on());
    if size != 0 {
        let spare = {
            let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
            let taken = all.0.get_mut(&size).and_then(Vec::pop);
            if taken.is_some() {
                all.1 -= size;
            }
            taken
        };
        if let Some(Spare(buffer)) = spare {
            SPARE_STATS[0].fetch_add(1, Ordering::Relaxed);
            return Some((buffer, size));
        }
    }
    let made = if size != 0 { size } else { total };
    let buffer = host.read_back_buffer(made)?;
    SPARE_STATS[1].fetch_add(1, Ordering::Relaxed);
    Some((buffer, size))
}

/// The readback resources and the allocation the pixels belong to.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    /// The buffer's pool size ([`read_back_buffer`]); 0: its own, released
    /// when this is dropped.
    pooled: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

/// A list keeps its copies back until it is reset, which is after its last
/// submission has run (also when a closed list is submitted more than once:
/// every run lands from the same buffer), or until the list itself goes,
/// unrun or run: the buffer is free for another list then.
impl Drop for Back {
    fn drop(&mut self) {
        if self.pooled != 0 {
            let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
            if all.1 + self.pooled <= spare_most() {
                all.1 += self.pooled;
                all.0.entry(self.pooled).or_default().push(Spare(self.buffer.clone()));
            }
        }
    }
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }

    /// Whether its allocation is still the one it was recorded for.
    pub(crate) fn fresh(&self) -> bool {
        self.ticket.valid.load(Ordering::Acquire)
    }
}

/// Records a host copy from a texture already in COPY_SOURCE.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture kept through the command list's state tracking.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // The last footprint's advertised extent can omit row-end padding.
    // The tiler reads a pitched slice, so the mapped buffer must include it.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let (buffer, pooled) = read_back_buffer(host, total)?;
    EVER.store(true, Ordering::Relaxed);
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(total, Ordering::Relaxed);
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back { buffer, pooled, footprints, rows, mips, layout: target.layout, ticket: target.ticket })
}

/// What a thread that lands copy-backs keeps from one landing to the next:
/// the title's bytes being tiled, the same as read
/// (`VICTORY_TEXTURE_BACK_LEAN`), and the tiling geometry of the mips it
/// tiled. After TaynanGT f0019f2e20 (there one is handed down by
/// each caller; here each thread that lands keeps its own).
#[derive(Default)]
struct Scratch {
    tiled: Vec<u8>,
    read: Vec<u8>,
    tiling: victory_formats::texture::TilingCache,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// Once the list has run, tile its texels back into the same title allocation.
pub(crate) fn land(back: &Back) {
    let began = std::time::Instant::now();
    SCRATCH.with_borrow_mut(|scratch| land_with(back, scratch));
    STATS[6].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
}

fn land_with(back: &Back, scratch: &mut Scratch) {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    let _mapping = crate::watch::title_memory_reading();
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(skipped, "skipping texture readbacks whose title allocation changed");
        }
        return;
    }
    let (at, end) = back.range();
    let size = end - at;
    let Scratch { tiled, read, tiling } = scratch;
    if tiled.len() < size {
        tiled.resize(size, 0);
    }
    let tiled = &mut tiled[..size];
    // Keep title bytes the subresources do not cover.
    if !crate::host::read_into(at as *const u8, tiled) {
        return;
    }
    let lean = lean();
    if lean {
        read.clear();
        read.extend_from_slice(tiled);
    }
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * rows as usize * f.Depth as usize;
        // SAFETY: the subresource's part of the mapped buffer.
        let linear = unsafe { std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len) };
        if let Err(e) = tiling.tile(&back.layout, tiled, (mip, slice), linear, (pitch, pitch * rows as usize)) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled"));
            // SAFETY: mapped above.
            unsafe { back.buffer.Unmap(0, None) };
            return;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, None) };
    if lean && read[..] == tiled[..] {
        STATS[5].fetch_add(1, Ordering::Relaxed);
        return;
    }
    crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
    STATS[2].fetch_add(1, Ordering::Relaxed);
    STATS[3].fetch_add(size as u64, Ordering::Relaxed);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(
            copied,
            bytes = size,
            buffers_reused = SPARE_STATS[0].load(Ordering::Relaxed),
            buffers_made = SPARE_STATS[1].load(Ordering::Relaxed),
            "texture readback landed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On unless `0`; `1` is the four water kinds, not an empty list (as it
    /// was in vx-layer `eee5667d`).
    #[test]
    fn the_switch_is_on_unless_zero() {
        let water: Vec<(u64, u32, u32)> = vec![(256, 256, 41), (256, 256, 16), (64, 64, 16), (128, 128, 36)];
        assert_eq!(parse_kinds(None), water);
        assert_eq!(parse_kinds(Some("1")), water);
        assert_eq!(parse_kinds(Some(" 1 ")), water);
        assert!(parse_kinds(Some("0")).is_empty());
        assert!(parse_kinds(Some("")).is_empty());
        assert_eq!(parse_kinds(Some("64x64f16, 128x128f36")), water[2..]);
    }

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }

    #[test]
    fn readback_buffers_pool_by_power_of_two_up_to_16_mib() {
        // The water kinds: a 256x256 R32 map with its levels, and smaller.
        assert_eq!(pool_size(1, true), 64 << 10);
        assert_eq!(pool_size(64 << 10, true), 64 << 10);
        assert_eq!(pool_size((256 * 256 * 4) * 4 / 3, true), 512 << 10);
        assert_eq!(pool_size(16 << 20, true), 16 << 20);
        assert_eq!(pool_size((16 << 20) + 1, true), 0);
        assert_eq!(pool_size(1, false), 0);
    }
}
