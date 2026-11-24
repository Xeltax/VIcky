//! Presented frames written to PNG files, to see what reaches the screen:
//! with `VICTORY_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size. Only in builds with diagnostics.

use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| std::env::var_os("VICTORY_CAPTURE").map(PathBuf::from))
        .as_ref()
}

/// Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated): saved
/// too, and their draws traced at debug level.
pub(crate) fn listed(n: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let list = std::env::var("VICTORY_CAPTURE_FRAMES").unwrap_or_default();
        list.split(',')
            .filter_map(|f| f.trim().parse().ok())
            .collect()
    })
    .contains(&n)
        || timed(n)
}

/// combo (local): `VICTORY_CAPTURE_AT_SECS=150,200`: the first frame built after each of
/// those seconds of the run is listed too (menus take a different number of frames each run).
fn timed(n: u64) -> bool {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static LEFT: std::sync::Mutex<Option<Vec<f64>>> = std::sync::Mutex::new(None);
    static CHOSEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
    let start = *START.get_or_init(std::time::Instant::now);
    let mut chosen = CHOSEN.lock().unwrap_or_else(|e| e.into_inner());
    if chosen.contains(&n) {
        return true;
    }
    let mut left = LEFT.lock().unwrap_or_else(|e| e.into_inner());
    let left = left.get_or_insert_with(|| {
        std::env::var("VICTORY_CAPTURE_AT_SECS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect()
    });
    let now = start.elapsed().as_secs_f64();
    if let Some(i) = left.iter().position(|&t| now >= t) {
        left.remove(i);
        // (asked first at a present, too late for that frame's targets: the next frame's are
        // noted from its first draw)
        chosen.extend([n, n + 1]);
        tracing::info!(frame = n, secs = now, "combo: frame listed for capture");
        return true;
    }
    false
}

/// A plane selected for this frame, with the same resource ID as its target dump.
pub(crate) fn present_source(
    frame: u64,
    plane: usize,
    composition_index: usize,
    resource: crate::object::This,
    desc: &D3D12_RESOURCE_DESC,
) {
    tracing::info!(
        frame,
        plane,
        composition_index,
        ?resource,
        width = desc.Width,
        height = desc.Height,
        format = desc.Format.0,
        "present source"
    );
}

/// Whether frame `n` is saved.
pub(crate) fn wanted(n: u64) -> bool {
    folder().is_some() && (n.is_power_of_two() || listed(n))
}

/// Saves 8-bit RGBA or BGRA rows (`pitch` bytes apart) as frame `n`.
pub(crate) fn save(n: u64, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    save_as(&format!("frame-{n}"), width, height, pitch, bgra, rows);
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, at half size.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    let Some(folder) = folder() else { return };
    let (w, h) = ((width / 2).max(1), (height / 2).max(1));
    // Filter byte 0 per row, then RGB of every other pixel of every other row.
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        let row = &rows[(2 * y).min(height as usize - 1) * pitch..];
        for x in 0..w as usize {
            let p = &row[(2 * x).min(width as usize - 1) * 4..][..3];
            if bgra {
                raw.extend([p[2], p[1], p[0]]);
            } else {
                raw.extend_from_slice(p);
            }
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    chunk(
        &mut png,
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3),
    );
    chunk(&mut png, b"IEND", &[]);
    let path = folder.join(format!("{name}.png"));
    match std::fs::write(&path, png) {
        Ok(()) => tracing::info!(path = %path.display(), "frame saved"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame not saved"),
    }
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Records a copy of `texture` (in the copy-source state) into a
/// readback buffer.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(
    frame: u64,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let f = footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "frame not saved: not 8-bit color");
            return;
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (every submission is waited for); the
    // mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let len = f.RowPitch as usize * f.Height as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
        save(frame, f.Width, f.Height, f.RowPitch as usize, bgra, rows);
        buffer.Unmap(0, None);
    }
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`
/// (what each holds at the end of the frame; formats normalized to show
/// structure: see [`Scale`]). Each that is not 8-bit colour is logged first
/// as `target values`, with its largest finite value and how many of its
/// channel values are non-zero.
/// The host copies are held: the title may release a target before the
/// frame is presented.
static TARGETS: std::sync::Mutex<Vec<(usize, std::sync::Arc<crate::resource::HostTexture>)>> =
    std::sync::Mutex::new(Vec::new());

fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && std::env::var_os("VICTORY_DUMP_TARGETS").is_some())
}

/// Notes a title texture the frame being built renders to or writes.
pub(crate) fn note_target(resource: crate::object::This) {
    if resource.is_null() || !dump_on() || !listed(crate::queue::presented() + 1) {
        return;
    }
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    if !targets.iter().any(|&(r, _)| r == resource as usize)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        targets.push((resource as usize, copy));
    }
}

/// At the present of frame `frame`: saves the noted targets.
pub(crate) fn dump_targets(frame: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let targets = std::mem::take(&mut *TARGETS.lock().unwrap_or_else(|e| e.into_inner()));
    let Some(host) = crate::host::host() else { return };
    for (i, (resource, copy)) in targets.iter().enumerate() {
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            continue;
        }
        // Every submission has run; the dump's list moves it there and back.
        let state = copy.state.load(Relaxed);
        dump_one(host, texture, state, frame, i, *resource as crate::object::This, &d);
    }
}

/// With `VICTORY_DUMP_TARGETS` set (and `VICTORY_CAPTURE`, as for targets),
/// buffers too: the title memory ranges
/// the listed frames bind for unordered access (a compute's output that
/// later passes read), in first-binding order. Each is logged as
/// `uav buffer <i>` when first bound: with the draw trace on, after the
/// draw or dispatch that binds it. That is not always the writer: a draw
/// can still be dropped after binding, later binders of the same range are
/// not logged again, and a note can come without its trace line:
/// `ExecuteIndirect` has none, and a command whose trace check and binding
/// fall on either side of a present or of a listing by time can be noted
/// for a listed frame its trace line did not go to. A range is one
/// host copy, address and length: a buffer released and made again at the
/// same address within the frame is noted twice. At the frame's present each range is summarized in the log
/// (`uav buffer contents`); a range of at most [`RAW_MAX`] bytes is also
/// saved whole as `frame-<n>-buf<i>-<address>-<len>.bin` (a larger one gets
/// no file). At most [`NOTED_MAX`] ranges per frame are noted.
///
/// What is read is the range's host copy, the bytes the GPU side sees, at
/// the present. Title memory can differ where the title's CPU wrote after
/// the last copy-back, and work of the next frame that other threads
/// already submit can change the copy while it is read.
///
/// shortcut: ranges are noted when commands are recorded, so a list
/// recorded in another frame and submitted again in a listed one is not
/// seen; and they are the ranges declared for copy-back
/// (`Shadow::gpu_writes`), so a buffer a shader reaches only through the
/// resource heap, and not through its tables, is not seen either. A frame
/// listed by time is noted only from the moment it is listed; the frame
/// after it is usually whole, but can miss ranges bound just before.
static BUFFERS: std::sync::Mutex<Notes<NotedBuffer>> = std::sync::Mutex::new(Notes::new());

/// A range a listed frame's shaders write: its host copy, address, length.
type NotedBuffer = (std::sync::Arc<crate::shadow::Shadow>, u64, u64);

/// Ranges noted per frame at most (the world binds a few hundred).
const NOTED_MAX: usize = 4096;
/// Largest range saved as a file (the world also binds views of its whole
/// heap, about 174 MB, which are only summarized).
const RAW_MAX: u64 = 4 << 20;
/// Bytes read at a time to summarize a range.
const PIECE: usize = 1 << 20;

/// Notes by frame. `newest` is the latest frame dumped and `taken` the
/// frames dumped within [`STALE`] frames of it. A note for a dumped frame
/// (its recorder read the frame number before that frame's present took the
/// notes), or for a frame `STALE` or more frames older than `newest`, is
/// refused rather than kept and never dumped. Presents may overlap, so each
/// frame is taken on its own; when a frame is taken, notes `STALE` or more
/// frames older than `newest` are dropped (a present that late finds none).
/// Notes of a frame never presented as listed stay until a listed present
/// at least `STALE` frames later.
struct Notes<T> {
    newest: u64,
    taken: Vec<u64>,
    notes: Vec<(u64, T)>,
}

/// The age, in frames, at which a note is too old to keep.
const STALE: u64 = 64;

impl<T> Notes<T> {
    const fn new() -> Notes<T> {
        Notes {
            newest: 0,
            taken: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Whether `frame` is old enough, next to the newest frame dumped, for
    /// its notes to go.
    fn stale(&self, frame: u64) -> bool {
        is_stale(frame, self.newest)
    }

    /// Adds `note` to `frame`'s notes; its index there, or `None` if the
    /// frame was already taken or is stale, `same` finds an equal note of
    /// that frame, or it has [`NOTED_MAX`] notes.
    fn add(&mut self, frame: u64, note: T, same: impl Fn(&T) -> bool) -> Option<usize> {
        if self.taken.contains(&frame) || self.stale(frame) {
            return None;
        }
        let mut this_frame = self.notes.iter().filter(|n| n.0 == frame);
        let i = this_frame.clone().count();
        if i >= NOTED_MAX || this_frame.any(|n| same(&n.1)) {
            return None;
        }
        self.notes.push((frame, note));
        Some(i)
    }

    /// `frame`'s notes, in order; other frames' stay, unless stale.
    fn take(&mut self, frame: u64) -> Vec<T> {
        self.newest = self.newest.max(frame);
        self.taken.push(frame);
        let newest = self.newest;
        let stale = |f: u64| is_stale(f, newest);
        self.taken.retain(|&f| !stale(f));
        let (now, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.notes)
            .into_iter()
            .partition(|n| n.0 == frame);
        self.notes = rest.into_iter().filter(|n| !stale(n.0)).collect();
        now.into_iter().map(|n| n.1).collect()
    }
}

/// Whether `frame` is [`STALE`] or more frames older than `newest`.
fn is_stale(frame: u64, newest: u64) -> bool {
    frame < newest && newest - frame >= STALE
}

/// Notes a range of title memory the frame being built binds for unordered
/// access.
pub(crate) fn note_buffer(shadow: &std::sync::Arc<crate::shadow::Shadow>, address: u64, len: u64) {
    let frame = crate::queue::presented() + 1;
    if !dump_on() || !listed(frame) {
        return;
    }
    let mut buffers = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    let same =
        |(s, a, l): &NotedBuffer| std::sync::Arc::ptr_eq(s, shadow) && (*a, *l) == (address, len);
    if let Some(i) = buffers.add(frame, (shadow.clone(), address, len), same) {
        tracing::info!(i, address = format_args!("{address:#x}"), len, "uav buffer");
    }
}

/// At the present of frame `frame`: summarizes the ranges noted for it.
/// Notes for the next frame, which other threads may already be recording,
/// stay.
pub(crate) fn dump_buffers(frame: u64) {
    let this_frame = BUFFERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take(frame);
    for (i, (shadow, address, len)) in this_frame.into_iter().enumerate() {
        let mut sum = Summing::new();
        let mut raw = (len <= RAW_MAX).then(Vec::new);
        let read = shadow.read_pieces(address, len, PIECE, |piece| {
            sum.add(piece);
            if let Some(raw) = &mut raw {
                raw.extend_from_slice(piece);
            }
        });
        if !read {
            tracing::warn!(
                frame,
                i,
                address = format_args!("{address:#x}"),
                "uav buffer not read"
            );
            continue;
        }
        let s = sum.finish();
        tracing::info!(
            frame,
            i,
            address = format_args!("{address:#x}"),
            len = s.bytes,
            dwords = s.dwords,
            zero = s.zero,
            nonfinite = s.nonfinite,
            min = s.min,
            max = s.max,
            mean = s.mean,
            "uav buffer contents{}",
            if s.all_zero { " (all zero)" } else { "" }
        );
        let (Some(raw), Some(folder)) = (raw, folder()) else {
            continue;
        };
        let name = format!("frame-{frame}-buf{i:03}-{address:#x}-{}.bin", raw.len());
        let path = folder.join(name);
        if let Err(e) = std::fs::write(&path, &raw) {
            tracing::warn!(error = %e, path = %path.display(), "buffer not saved");
        }
    }
}

/// A buffer's contents in a few numbers, its dwords read as `f32` (the
/// usual element of the buffers lighting reads).
#[derive(Debug, Default, PartialEq)]
struct Stats {
    bytes: usize,
    /// Every byte is 0, a trailing partial dword included.
    all_zero: bool,
    dwords: usize,
    /// Dwords that are 0.
    zero: usize,
    /// Dwords that are NaN or infinite as floats.
    nonfinite: usize,
    /// Over the finite floats (0 when there are none).
    min: f32,
    max: f32,
    mean: f64,
}

/// [`Stats`] built from pieces of a range, read in order; every piece but
/// the last must be a whole number of dwords.
struct Summing {
    s: Stats,
    finite: usize,
    sum: f64,
    min: f32,
    max: f32,
    nonzero: bool,
}

impl Summing {
    fn new() -> Summing {
        Summing {
            s: Stats::default(),
            finite: 0,
            sum: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            nonzero: false,
        }
    }

    fn add(&mut self, bytes: &[u8]) {
        self.s.bytes += bytes.len();
        self.nonzero |= bytes.iter().any(|&b| b != 0);
        for d in bytes.as_chunks::<4>().0 {
            let bits = u32::from_le_bytes(*d);
            let v = f32::from_bits(bits);
            self.s.dwords += 1;
            self.s.zero += usize::from(bits == 0);
            if v.is_finite() {
                self.finite += 1;
                self.sum += f64::from(v);
                self.min = self.min.min(v);
                self.max = self.max.max(v);
            } else {
                self.s.nonfinite += 1;
            }
        }
    }

    fn finish(self) -> Stats {
        let mut s = self.s;
        s.all_zero = !self.nonzero;
        if self.finite > 0 {
            (s.min, s.max, s.mean) = (self.min, self.max, self.sum / self.finite as f64);
        }
        s
    }
}

impl Stats {
    #[cfg(test)]
    fn of(bytes: &[u8]) -> Stats {
        let mut sum = Summing::new();
        sum.add(bytes);
        sum.finish()
    }
}

fn dump_one(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    frame: u64,
    i: usize,
    resource: crate::object::This,
    d: &D3D12_RESOURCE_DESC,
) {
    let Ok(list) = host.new_list() else { return };
    let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
    let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: an open list of ours and a live texture.
    let read = unsafe {
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, state, copy)]);
        }
        let read = read_back(&host.device, &list.list, texture);
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
        }
        read
    };
    list.close();
    host.execute(&[&list]);
    let Some((buffer, footprint)) = read else { return };
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * f.Height as usize);
        if let Some(rgba) = to_rgba(f.Format.0 as u32, f.Width as usize, f.Height as usize, pitch, rows) {
            let name = format!(
                "frame-{frame}-{i:02}-{resource:?}-{}x{}-f{}",
                d.Width, d.Height, d.Format.0
            );
            save_as(&name, f.Width, f.Height, f.Width as usize * 4, false, &rgba);
        } else {
            tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        }
        buffer.Unmap(0, None);
    }
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// Rows of a host format as 8-bit RGBA: 8-bit color as is, everything else
/// as channel values scaled by [`Scale`] (single channels as grey). `None`
/// for formats not handled.
fn to_rgba(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(w * h * 4);
    let px = |x: usize, y: usize, n: usize| &rows[y * pitch + x * n..][..n];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
    // Channel values as floats, then scaled.
    type Decode = Box<dyn Fn(&[u8]) -> [f32; 3]>;
    let (n, decode): (usize, Decode) = match format {
        27..=32 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[0], p[1], p[2], 255]);
                }
            }
            return Some(out);
        }
        87..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        23..=25 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [(v & 0x3ff) as f32, (v >> 10 & 0x3ff) as f32, (v >> 20 & 0x3ff) as f32]
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=11 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        33..=34 => (4, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), 0.0])),
        1..=2 => (16, Box::new(move |b| {
            [0, 1, 2].map(|i| f32::from_bits(u32_at(b, i)))
        })),
        15..=16 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)), f32::from_bits(u32_at(b, 1)), 0.0])),
        35..=38 => (4, Box::new(move |b| [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0])),
        48..=52 => (2, Box::new(move |b| [f32::from(b[0]), f32::from(b[1]), 0.0])),
        // Single channels (and depth planes): shown as grey.
        39..=43 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        19..=22 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        44..=47 => (4, Box::new(move |b| [(u32_at(b, 0) & 0xff_ffff) as f32; 3])),
        53..=54 => (2, Box::new(move |b| [half(u16_at(b, 0)); 3])),
        55..=59 => (2, Box::new(move |b| [f32::from(u16_at(b, 0)); 3])),
        60..=65 => (1, Box::new(move |b| [f32::from(b[0]); 3])),
        _ => return None,
    };
    let values: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| decode(px(x, y, n)))
        .collect();
    let grey = matches!(format, 19..=22 | 39..=47 | 53..=65);
    // Channels the decoder fills; the rest of each [f32; 3] is padding.
    let channels = if grey {
        1
    } else if matches!(format, 15..=16 | 33..=38 | 48..=52) {
        2
    } else {
        3
    };
    let s = Scale::of(&values, channels, grey);
    tracing::info!(
        format,
        low = s.low,
        high = s.high,
        max = s.max,
        nonzero = s.nonzero,
        finite = s.finite,
        mean = s.mean,
        "target values"
    );
    for v in values {
        let c = if grey {
            [s.byte(v[0]); 3]
        } else {
            [0, 1, 2].map(|k| if k < channels { s.byte(v[k]) } else { 0 })
        };
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
}

/// How a float target's values map to 0..255 in its dump. Scaling by the
/// largest value let one hot pixel (a light source at thousands) turn a lit
/// HDR target black, so colour runs from 0 to about the 99.9th percentile of
/// the channel values, and single channels (`grey`, depth among them) from
/// about the 0.1st to the 99.9th; values outside saturate. Where that range is
/// empty (most of the target 0, or all of it negative), the minimum to the
/// maximum is used instead; a flat target is black at 0 or below and white
/// above. Percentiles come from whole pixels, one per stride at a hashed
/// place in it (which makes following a column pattern unlikely), at most
/// [`SAMPLES`] values (see [`sample_positions`]). Counts, maximum and mean
/// cover every pixel's decoded channels: not a decoder's padding, nor the
/// alpha of multichannel colour formats (an alpha-only format is decoded as
/// grey). Non-finite values are left out of all of it and dumped black.
#[derive(Debug, PartialEq)]
struct Scale {
    low: f32,
    high: f32,
    /// The largest finite value.
    max: f32,
    /// Finite decoded values that are not 0, and all finite decoded values.
    nonzero: usize,
    finite: usize,
    /// Over the finite decoded values (approximate where they cancel).
    mean: f64,
}

/// Values the percentiles of a [`Scale`] are taken from at most.
const SAMPLES: usize = 1 << 20;

/// The pixels (of `len`, with `channels` values each) a [`Scale`] samples:
/// at most `SAMPLES / channels` whole pixels, one per stride at a hashed
/// place in it, which makes lining up with a row's column pattern unlikely
/// (not impossible); the last stride may be partial and is sampled too.
fn sample_positions(len: usize, channels: usize) -> impl Iterator<Item = usize> {
    let pixels = (SAMPLES / channels.max(1)).max(1);
    let step = len.div_ceil(pixels).max(1);
    (0..len.div_ceil(step)).map(move |i| {
        // splitmix64: no short period, unlike a multiplicative step.
        let mut z = (i as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let start = i * step;
        start + (z % (len - start).min(step) as u64) as usize
    })
}

impl Scale {
    /// The scale of `pixels`, whose first `channels` entries are decoded.
    fn of(pixels: &[[f32; 3]], channels: usize, grey: bool) -> Scale {
        let channels = channels.clamp(1, 3);
        let (mut min, mut max, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
        let (mut nonzero, mut finite) = (0, 0);
        for &v in pixels.iter().flat_map(|p| &p[..channels]) {
            if v.is_finite() {
                (min, max) = (min.min(v), max.max(v));
                sum += f64::from(v);
                finite += 1;
                nonzero += usize::from(v != 0.0);
            }
        }
        if finite == 0 {
            (min, max) = (0.0, 0.0);
        }
        let mut sample: Vec<f32> = sample_positions(pixels.len(), channels)
            .flat_map(|i| pixels[i][..channels].iter().copied())
            .filter(|v| v.is_finite())
            .collect();
        let mut at = |q: f64| {
            if sample.is_empty() {
                return 0.0;
            }
            let i = ((sample.len() - 1) as f64 * q).round() as usize;
            *sample.select_nth_unstable_by(i, f32::total_cmp).1
        };
        let (mut low, mut high) = (if grey { at(0.001) } else { 0.0 }, at(0.999));
        if high <= low {
            (low, high) = (min, max);
            if !grey && min >= 0.0 {
                low = 0.0;
            }
        }
        Scale {
            low,
            high,
            max,
            nonzero,
            finite,
            mean: if finite == 0 {
                0.0
            } else {
                sum / finite as f64
            },
        }
    }

    /// A channel value as 0..255 (computed in f64, so even the widest
    /// finite range keeps its ends apart; non-finite is 0; with a flat
    /// range, 0 and below are 0 and the rest 255).
    fn byte(&self, v: f32) -> u8 {
        if !v.is_finite() {
            return 0;
        }
        let range = f64::from(self.high) - f64::from(self.low);
        if range <= 0.0 {
            return if v > 0.0 { 255 } else { 0 };
        }
        ((f64::from(v) - f64::from(self.low)) / range * 255.0).clamp(0.0, 255.0) as u8
    }
}

/// Pixels per side of the grid [`probe`] reads.
const GRID: u32 = 4;
/// Bytes between the grid's pixels in the readback buffer (placement
/// alignment).
const SLOT: u64 = 512;

/// Records copies of a grid of pixels of `texture` (8-bit color, in the
/// copy-source state) into a readback buffer, to tell black frames.
pub(crate) fn probe(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<ID3D12Resource> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if !matches!(d.Format.0, 27..=29 | 87 | 90 | 91) {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOT * u64::from(GRID * GRID),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        for i in 0..GRID * GRID {
            let (x, y) = (
                (2 * (i % GRID) + 1) * d.Width as u32 / (2 * GRID),
                (2 * (i / GRID) + 1) * d.Height / (2 * GRID),
            );
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: SLOT * u64::from(i),
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: d.Format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let pixel = D3D12_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + 1,
                bottom: y + 1,
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&pixel));
        }
        Some(buffer)
    }
}

/// Whether every pixel [`probe`] read is (near) black.
pub(crate) fn black(buffer: &ID3D12Resource) -> bool {
    let mut data = std::ptr::null_mut();
    // SAFETY: the copies finished (every submission is waited for).
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let bytes =
            std::slice::from_raw_parts(data.cast::<u8>(), (SLOT * u64::from(GRID * GRID)) as usize);
        let level = bytes
            .chunks(SLOT as usize)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .max()
            .unwrap_or(0);
        buffer.Unmap(0, None);
        let mut levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
        if levels.len() == 12 {
            levels.pop_front();
        }
        levels.push_back(level);
        level < 8
    }
}

/// The brightest probed channel of the last frames, oldest first: a fade
/// darkens over several, a glitch drops at once.
static LEVELS: std::sync::Mutex<std::collections::VecDeque<u8>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

pub(crate) fn recent_levels() -> Vec<u8> {
    let levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
    levels.iter().copied().collect()
}

#[cfg(test)]
mod scale_tests {
    use super::Scale;

    fn colour(values: impl Iterator<Item = f32>) -> Vec<[f32; 3]> {
        values.map(|v| [v, v, v]).collect()
    }

    #[test]
    fn one_hot_pixel_does_not_set_the_scale() {
        // A lit target: 10,000 pixels around 1.0 and one light at 11,088.
        let values = (0..10_000).map(|i| 0.5 + (i % 100) as f32 / 100.0);
        let pixels = colour(values.chain([11_088.0]));
        let s = Scale::of(&pixels, 3, false);
        assert_eq!((s.low, s.max), (0.0, 11_088.0));
        assert!(
            s.high < 1.5,
            "high {} follows the bulk, not the light",
            s.high
        );
        assert_eq!((s.nonzero, s.finite), (30_003, 30_003));
        assert_eq!(s.byte(11_088.0), 255);
        assert!(s.byte(1.0) > 150);
    }

    #[test]
    fn a_strided_sample_keeps_every_channel() {
        // More values than SAMPLES, so the sample strides: a uniform
        // (1, 10, 100) must still scale to its blue channel.
        let pixels = vec![[1.0, 10.0, 100.0]; super::SAMPLES];
        let s = Scale::of(&pixels, 3, false);
        assert_eq!((s.low, s.high), (0.0, 100.0));
        assert_eq!([1.0, 10.0, 100.0].map(|c| s.byte(c)), [2, 25, 255]);
    }

    #[test]
    fn a_stride_dividing_the_row_still_sees_every_column() {
        // Alternating columns of 1 and 100, twice SAMPLES pixels: a stride
        // of 6 pixels, which without jitter would see the 1s only.
        let pixels: Vec<[f32; 3]> = (0..2 * super::SAMPLES)
            .map(|i| [if i % 2 == 0 { 1.0 } else { 100.0 }; 3])
            .collect();
        let s = Scale::of(&pixels, 3, false);
        assert_eq!(s.high, 100.0);
        assert!(s.byte(1.0) < 10);
    }

    #[test]
    fn sample_positions_stay_in_range_and_within_the_cap() {
        use super::{SAMPLES, sample_positions};
        for (len, channels) in [(0, 3), (1, 3), (1 << 20, 3), (2048 * 2560, 3), (12_345, 1)] {
            let at: Vec<usize> = sample_positions(len, channels).collect();
            assert!(at.iter().all(|&i| i < len), "{len}");
            assert!(at.len() * channels <= SAMPLES, "{len}: {}", at.len());
            assert!(
                at.windows(2).all(|w| w[0] < w[1]),
                "{len}: in order, once each"
            );
            if len > 0 {
                assert!(!at.is_empty());
            }
        }
        // A partial last stride (349,525 pixels a stride of 3 apart, one
        // left over) is sampled too: only one place in it.
        let at: Vec<usize> = sample_positions(699_052, 3).collect();
        assert_eq!(at.last(), Some(&699_051));
        assert!(at.len() * 3 <= SAMPLES);
    }

    #[test]
    fn the_jitter_does_not_repeat_along_a_row() {
        // 512 x 1024 RGB: a stride of 2 pixels, a row every 256 strides. A
        // jitter that repeats each row samples the same 256 columns only.
        let (w, h) = (512, 1024);
        let mut columns: Vec<usize> = super::sample_positions(w * h, 3).map(|i| i % w).collect();
        columns.sort_unstable();
        columns.dedup();
        assert!(
            columns.len() > 500,
            "{} of {w} columns sampled",
            columns.len()
        );
    }

    #[test]
    fn padding_channels_are_not_counted() {
        // RG: the decoder fills blue with 0.
        let pixels = vec![[2.0, 4.0, 0.0]; 1000];
        let s = Scale::of(&pixels, 2, false);
        assert_eq!((s.finite, s.nonzero, s.mean), (2000, 2000, 3.0));
        // Grey: one value, repeated in all three.
        let s = Scale::of(&[[5.0; 3]; 10], 1, true);
        assert_eq!((s.finite, s.mean), (10, 5.0));
    }

    #[test]
    fn a_few_lit_pixels_on_black_still_show() {
        // 99.95% zero: the 99.9th percentile is 0, so the maximum is used.
        let values = std::iter::repeat_n(0.0, 19_990).chain(std::iter::repeat_n(7.0, 10));
        let s = Scale::of(&colour(values), 3, false);
        assert_eq!((s.low, s.high, s.nonzero), (0.0, 7.0, 30));
        assert_eq!((s.byte(0.0), s.byte(7.0)), (0, 255));
    }

    #[test]
    fn single_channels_drop_the_outer_tenth_of_a_percent() {
        let values = (0..10_000)
            .map(|i| 0.4 + (i % 1000) as f32 / 5000.0)
            .chain([-1e30, 1e30, f32::NAN, f32::INFINITY]);
        let s = Scale::of(&colour(values), 1, true);
        assert!(s.low >= 0.4 && s.high <= 0.6, "{} .. {}", s.low, s.high);
        assert_eq!((s.max, s.finite), (1e30, 10_002));
        assert_eq!((s.byte(f32::NAN), s.byte(f32::INFINITY)), (0, 0));
    }

    #[test]
    fn negative_and_extreme_ranges_still_map() {
        // All-negative colour: min..max, not 0..(a negative high).
        let s = Scale::of(&colour([-4.0, -2.0].into_iter()), 3, false);
        assert_eq!((s.low, s.high), (-4.0, -2.0));
        assert_eq!((s.byte(-4.0), s.byte(-2.0)), (0, 255));
        // The widest finite range keeps its ends apart (f32 would overflow).
        let s = Scale::of(&colour([-f32::MAX, f32::MAX].into_iter()), 1, true);
        assert_eq!((s.byte(-f32::MAX), s.byte(f32::MAX)), (0, 255));
    }

    #[test]
    fn an_empty_or_flat_target_has_a_range() {
        let s = Scale::of(&[], 3, false);
        assert_eq!((s.low, s.high, s.max, s.finite), (0.0, 0.0, 0.0, 0));
        // A target cleared to 0 is black, not white.
        let s = Scale::of(&[[0.0; 3]; 100], 3, false);
        assert_eq!((s.nonzero, s.byte(0.0)), (0, 0));
        let s = Scale::of(&[[3.0; 3]; 100], 1, true);
        assert_eq!((s.low, s.high, s.mean), (3.0, 3.0, 3.0));
    }
}

#[cfg(test)]
mod tests {
    use super::Stats;

    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }

    fn bytes(dwords: &[u32]) -> Vec<u8> {
        dwords.iter().flat_map(|d| d.to_le_bytes()).collect()
    }

    #[test]
    fn an_empty_buffer_reads_as_all_zero() {
        let s = Stats::of(&[0; 64]);
        assert!(s.all_zero);
        assert_eq!((s.bytes, s.dwords, s.zero, s.nonfinite), (64, 16, 16, 0));
        assert_eq!((s.min, s.max, s.mean), (0.0, 0.0, 0.0));
    }

    #[test]
    fn floats_are_summarized_and_nonfinite_ones_counted_apart() {
        let s = Stats::of(&bytes(&[
            1.5f32.to_bits(),
            (-2.0f32).to_bits(),
            0,
            f32::NAN.to_bits(),
            f32::INFINITY.to_bits(),
            4.5f32.to_bits(),
        ]));
        let expected = Stats {
            bytes: 24,
            all_zero: false,
            dwords: 6,
            zero: 1,
            nonfinite: 2,
            min: -2.0,
            max: 4.5,
            mean: 1.0,
        };
        assert_eq!(s, expected);
    }

    #[test]
    fn trailing_bytes_short_of_a_dword_are_left_out() {
        let mut b = bytes(&[0x3f80_0000]); // 1.0
        b.extend([0xff, 0xff]);
        let s = Stats::of(&b);
        assert_eq!((s.bytes, s.dwords, s.zero, s.max), (6, 1, 0, 1.0));
    }

    #[test]
    fn a_nonzero_byte_past_the_last_dword_is_not_all_zero() {
        // An 8-bit buffer of one byte, 0xff: no whole dword, but not empty.
        let s = Stats::of(&[0xff]);
        assert_eq!((s.bytes, s.dwords, s.zero), (1, 0, 0));
        assert!(!s.all_zero);
        let s = Stats::of(&[0, 0, 0, 0, 0, 0x01]);
        assert_eq!((s.dwords, s.zero), (1, 1));
        assert!(!s.all_zero);
    }

    #[test]
    fn notes_stay_with_their_frame() {
        let mut n = super::Notes::new();
        let never = |_: &u32| false;
        assert_eq!(n.add(10, 1, never), Some(0));
        assert_eq!(n.add(10, 2, never), Some(1));
        // The next frame's recorders may already note while 10 is dumped.
        assert_eq!(n.add(11, 3, never), Some(0));
        assert_eq!(n.take(10), [1, 2]);
        assert_eq!(n.take(11), [3]);
    }

    #[test]
    fn a_note_for_a_frame_already_taken_is_refused() {
        let mut n = super::Notes::new();
        let never = |_: &u32| false;
        n.add(10, 1, never);
        assert_eq!(n.take(10), [1]);
        // Its recorder read frame 10 before the present took the notes.
        assert_eq!(n.add(10, 2, never), None);
        assert!(n.take(11).is_empty());
        assert!(n.notes.is_empty());
    }

    #[test]
    fn a_range_is_noted_once_per_frame_and_again_in_the_next() {
        let mut n = super::Notes::new();
        assert_eq!(n.add(5, 7, |&x| x == 7), Some(0));
        assert_eq!(n.add(5, 7, |&x| x == 7), None);
        // The same range bound again in the next frame is that frame's.
        assert_eq!(n.add(6, 7, |&x| x == 7), Some(0));
        assert_eq!(n.take(5), [7]);
        assert_eq!(n.take(6), [7]);
    }

    #[test]
    fn overlapping_presents_keep_each_frames_notes() {
        let mut n = super::Notes::new();
        let never = |_: &u32| false;
        n.add(10, 1, never);
        n.add(11, 2, never);
        // Present 11 dumps before present 10 does.
        assert_eq!(n.take(11), [2]);
        // Frame 10's recorders may still note, and its present finds all.
        assert_eq!(n.add(10, 3, never), Some(1));
        assert_eq!(n.take(10), [1, 3]);
    }

    #[test]
    fn a_frame_takes_at_most_noted_max_ranges() {
        let mut n = super::Notes::new();
        let never = |_: &usize| false;
        for i in 0..super::NOTED_MAX {
            assert_eq!(n.add(1, i, never), Some(i));
        }
        assert_eq!(n.add(1, super::NOTED_MAX, never), None);
        assert_eq!(n.add(2, 0, never), Some(0));
        assert_eq!(n.take(1).len(), super::NOTED_MAX);
    }

    #[test]
    fn untaken_notes_go_once_stale() {
        let mut n = super::Notes::new();
        let never = |_: &u32| false;
        n.add(36, 1, never);
        n.add(37, 2, never);
        n.add(100, 3, never);
        assert_eq!(n.take(100), [3]);
        // 64 frames older than the newest dumped: gone; 63: kept.
        assert_eq!(n.take(36), [] as [u32; 0]);
        assert_eq!(n.take(37), [2]);
    }

    #[test]
    fn a_late_note_is_refused_once_its_frame_is_stale() {
        let mut n = super::Notes::new();
        let never = |_: &u32| false;
        n.add(10, 1, never);
        assert_eq!(n.take(10), [1]);
        // More dumps than any fixed list would hold, all within the window.
        for f in 11..=73 {
            n.take(f);
        }
        assert_eq!(n.add(10, 2, never), None);
        n.take(74);
        // Frame 10 is out of `taken` now, but stale: still refused.
        assert_eq!(n.add(10, 3, never), None);
        assert!(n.notes.is_empty());
    }

    #[test]
    fn frame_numbers_near_the_top_do_not_overflow() {
        let mut n = super::Notes::new();
        let never = |_: &u32| false;
        n.add(u64::MAX, 1, never);
        assert!(n.take(u64::MAX - 1).is_empty());
        assert_eq!(n.take(u64::MAX), [1]);
        // The other order: a note one frame older than the newest dumped.
        let mut n = super::Notes::new();
        n.add(u64::MAX - 1, 2, never);
        assert!(n.take(u64::MAX).is_empty());
        assert_eq!(n.take(u64::MAX - 1), [2]);
    }

    #[test]
    fn pieces_sum_up_to_the_whole() {
        let words: Vec<u32> = (0..1000u32)
            .map(|i| (i as f32 * 0.5 - 100.0).to_bits())
            .collect();
        let mut all = bytes(&words);
        all.push(7);
        let mut sum = super::Summing::new();
        for piece in all.chunks(64) {
            sum.add(piece);
        }
        assert_eq!(sum.finish(), Stats::of(&all));
    }
}
