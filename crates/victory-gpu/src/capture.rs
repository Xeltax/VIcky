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
/// too, and their draws traced at debug level. `VICTORY_CAPTURE_WORLD`
/// lists frames counted from the first one that ran 500 draws (the world's
/// first, whose number depends on how fast the title loaded).
pub(crate) fn listed(n: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let named = LIST.get_or_init(|| numbers("VICTORY_CAPTURE_FRAMES")).contains(&n);
    static OFFSETS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        let mut offsets = numbers("VICTORY_CAPTURE_WORLD");
        offsets.extend(numbers("VICTORY_CAPTURE_WORLD_ALSO"));
        offsets
    });
    let first = WORLD.load(std::sync::atomic::Ordering::Relaxed);
    named || first != u64::MAX && n.checked_sub(first).is_some_and(|after|
        [2, 6, 30, 80].contains(&after) || offsets.contains(&after))
}

fn numbers(var: &str) -> Vec<u64> {
    let list = std::env::var(var).unwrap_or_default();
    list.split(',')
        .filter_map(|f| f.trim().parse().ok())
        .collect()
}

/// With a capture folder, for the frames [`listed`]: `what` is added to
/// the frame's `frame-<n>-ops.txt`, one line per draw, dispatch, copy or
/// drop in the order recorded (the title records on several threads, so
/// lines of different lists interleave; each names its list). The file
/// says which pass writes which target, to find the pass whose output
/// stays empty.
#[cfg(feature = "diagnostics")]
pub(crate) fn op(list: usize, what: impl FnOnce() -> String) {
    use std::io::Write;
    let frame = crate::queue::presented() + 1;
    if !listed(frame) {
        // Before the world: the last lines are kept, and written out when
        // it starts (`before-world-ops.txt`): what the title drew just
        // before (Grand Theft Auto VI shows a title card there, whose text
        // did not show).
        if folder().is_some() && world_on() && WORLD.load(std::sync::atomic::Ordering::Relaxed) == u64::MAX {
            // At most 300 lines of a frame: the world's first frame alone
            // (thousands of operations, recorded before it is known to be
            // the world's) pushed everything before it out.
            static IN_FRAME: std::sync::Mutex<(u64, u32)> = std::sync::Mutex::new((0, 0));
            {
                let mut in_frame = IN_FRAME.lock().unwrap_or_else(|e| e.into_inner());
                if in_frame.0 != frame {
                    *in_frame = (frame, 0);
                }
                in_frame.1 += 1;
                if in_frame.1 > 300 {
                    return;
                }
            }
            let what = what();
            let mut ring = RING.lock().unwrap_or_else(|e| e.into_inner());
            if ring.len() >= RING_LINES {
                ring.pop_front();
            }
            ring.push_back(format!("frame {frame} list {list:#x} {what}"));
        }
        return;
    }
    let Some(folder) = folder() else {
        return;
    };
    static FILE: std::sync::Mutex<Option<(u64, u64, std::fs::File)>> = std::sync::Mutex::new(None);
    let what = what();
    let mut file = FILE.lock().unwrap_or_else(|e| e.into_inner());
    if file.as_ref().is_none_or(|f| f.0 != frame) {
        let Ok(made) = std::fs::File::create(folder.join(format!("frame-{frame}-ops.txt"))) else {
            return;
        };
        *file = Some((frame, 0, made));
    }
    if let Some((_, n, f)) = file.as_mut() {
        *n += 1;
        let _ = writeln!(f, "{n} list {list:#x} {what}");
    }
}

/// The last lines of [`op`] before the world.
#[cfg(feature = "diagnostics")]
static RING: std::sync::Mutex<std::collections::VecDeque<String>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
#[cfg(feature = "diagnostics")]
const RING_LINES: usize = 6000;

/// Whether `VICTORY_CAPTURE_WORLD` is set.
fn world_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CAPTURE_WORLD").is_some())
}

/// The first frame that ran 500 draws or more (`VICTORY_CAPTURE_WORLD`).
static WORLD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// At the present of `frame`: the draws run on the host so far.
pub(crate) fn frame_drew(frame: u64, drawn: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let before = LAST.swap(drawn, Relaxed);
    if world_on()
        && drawn.saturating_sub(before) >= 500
        && WORLD.compare_exchange(u64::MAX, frame, Relaxed, Relaxed).is_ok()
    {
        tracing::info!(frame, "the first frame of 500 draws or more: frames {} and {} are captured", frame + 2, frame + 6);
        #[cfg(feature = "diagnostics")]
        if let Some(folder) = folder() {
            let lines: Vec<String> = std::mem::take(&mut *RING.lock().unwrap_or_else(|e| e.into_inner())).into();
            let _ = std::fs::write(folder.join("before-world-ops.txt"), lines.join("
"));
        }
    }
}

/// Whether frame `n` is saved.
pub(crate) fn wanted(n: u64) -> bool {
    // Before the world, with `VICTORY_CAPTURE_WORLD`: every 16th frame too
    // (small pictures of menus, loading screens and title cards).
    let before_world = world_on() && n.is_multiple_of(16) && WORLD.load(std::sync::atomic::Ordering::Relaxed) == u64::MAX;
    // In the world: every 8th frame's picture, a timeline of the opening
    // cutscene (its shots do not fall on the same frame numbers from run to
    // run, so single frames cannot be compared between runs).
    let first = WORLD.load(std::sync::atomic::Ordering::Relaxed);
    let in_world = world_on() && first != u64::MAX && n > first && (n - first).is_multiple_of(8);
    folder().is_some() && (n.is_power_of_two() || listed(n) || before_world || in_world)
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
    match std::fs::create_dir_all(folder).and_then(|()| std::fs::write(&path, png)) {
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
    read_back_of(device, list, texture, 0)
}

/// As [`read_back`], of subresource `sub` (the stencil plane of a
/// one-level, one-slice depth-stencil texture is subresource 1).
fn read_back_of(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    sub: u32,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            sub,
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
                SubresourceIndex: sub,
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
        let len = buffer.GetDesc().Width as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
        save(frame, f.Width, f.Height, f.RowPitch as usize, bgra, rows);
        buffer.Unmap(0, None);
    }
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`
/// (what each holds at the end of the frame; formats normalized to show
/// structure: floats scaled by their largest value, depth by its range).
/// The host copies are held: the title may release a target before the
/// frame is presented.
/// Each with the first pipeline that wrote it as an unordered-access view
/// (`-by-<pipeline>` in the name; `-black` when its colour is all zeros).
/// `VICTORY_DUMP_WRITERS` (shader hashes) adds, in any frame, the textures
/// a pipeline with one of them writes, the first frame it runs.
#[allow(clippy::type_complexity)]
static TARGETS: std::sync::Mutex<
    Vec<(usize, std::sync::Arc<crate::resource::HostTexture>, String)>,
> = std::sync::Mutex::new(Vec::new());

thread_local! {
    /// The pipeline whose draw or dispatch this thread is binding.
    static WRITER: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Names the pipeline whose views this thread binds next (empty: none).
pub(crate) fn binding(pipeline: &str) {
    if dump_on() {
        WRITER.with(|w| pipeline.clone_into(&mut w.borrow_mut()));
    }
}

pub(crate) fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && std::env::var_os("VICTORY_DUMP_TARGETS").is_some())
}

/// `VICTORY_DUMP_WRITERS`' hashes not dumped yet.
static WRITERS: std::sync::Mutex<Option<Vec<String>>> = std::sync::Mutex::new(None);

/// Buffers such a pipeline writes: title memory, its bytes when first
/// bound in the frame, the pipeline. Logged at the present: dwords changed
/// and dwords not zero.
#[allow(clippy::type_complexity)]
static BUFFERS: std::sync::Mutex<Vec<(usize, Vec<u8>, String)>> = std::sync::Mutex::new(Vec::new());

/// Notes a title texture the frame being built renders to or writes (by a
/// command of `list`, whose submission dumps it again when the frame is
/// listed).
pub(crate) fn note_target(list: &crate::host::HostList, resource: crate::object::This) {
    if resource.is_null() || !dump_on() {
        return;
    }
    let by = WRITER.with(|w| w.borrow().clone());
    // Lists are recorded ahead of the frame that submits them.
    list.note_target(resource as usize, &by);
    if !listed(crate::queue::presented() + 1) {
        let mut writers = WRITERS.lock().unwrap_or_else(|e| e.into_inner());
        let writers = writers.get_or_insert_with(|| {
            let list = std::env::var("VICTORY_DUMP_WRITERS").unwrap_or_default();
            list.split(',').map(|h| h.trim().to_owned()).filter(|h| !h.is_empty()).collect()
        });
        if by.is_empty() || !writers.iter().any(|h| by.contains(h.as_str())) {
            return;
        }
        if let Some((at, len)) = crate::resource::buffer_bytes(resource) {
            let mut buffers = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
            if !buffers.iter().any(|b| b.0 == at as usize) {
                let mut before = vec![0u8; (len as usize).min(64 << 20) & !3];
                if crate::host::read_into(at, &mut before) {
                    buffers.push((at as usize, before, by));
                }
            }
            return;
        }
    }
    keep_target(resource, by);
}

/// Capture just the two upscale resources at present, without reading back
/// every render target. The operation trace identifies each resource's role.
pub(crate) fn note_upscale(colour: crate::object::This, output: crate::object::This) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| folder().is_some() && std::env::var_os("VICTORY_DUMP_UPSCALE").is_some_and(|v| v == "1"))
        && listed(crate::queue::presented() + 1)
    {
        keep_target(colour, "upscale input".to_owned());
        keep_target(output, "upscale output".to_owned());
    }
}

fn keep_target(resource: crate::object::This, by: String) {
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    if !targets.iter().any(|(r, _, _)| *r == resource as usize)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        targets.push((resource as usize, copy, by));
    }
}

/// Whether textures wait to be dumped at this present.
pub(crate) fn has_targets() -> bool {
    !TARGETS.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
        || !BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
}

/// At the present of frame `frame`: saves the noted targets.
pub(crate) fn dump_targets(frame: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let targets = std::mem::take(&mut *TARGETS.lock().unwrap_or_else(|e| e.into_inner()));
    let buffers = std::mem::take(&mut *BUFFERS.lock().unwrap_or_else(|e| e.into_inner()));
    // Writers dumped once.
    if let Some(writers) = WRITERS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        writers.retain(|h| {
            !targets.iter().any(|t| t.2.contains(h.as_str()))
                && !buffers.iter().any(|b| b.2.contains(h.as_str()))
        });
    }
    for (at, before, by) in &buffers {
        let mut after = vec![0u8; before.len()];
        if !crate::host::read_into(*at as *const u8, &mut after) {
            continue;
        }
        let dwords = |b: &[u8]| -> Vec<u32> {
            b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap_or_default())).collect()
        };
        let (old, new) = (dwords(before), dwords(&after));
        let changed = old.iter().zip(&new).filter(|(a, b)| a != b).count();
        let set = new.iter().filter(|&&x| x != 0).count();
        tracing::info!(
            frame,
            buffer = format_args!("{at:#x}"),
            dwords = new.len(),
            changed,
            set,
            by = by.as_str(),
            "a buffer the named pipeline writes, at the present"
        );
    }
    let Some(host) = crate::host::host() else { return };
    for (i, (resource, copy, by)) in targets.iter().enumerate() {
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
            continue;
        }
        // Every submission has run; the dump's list moves it there and back.
        let state = copy.state.load(Relaxed);
        let target = (i, *resource as crate::object::This, by.as_str());
        dump_one(host, texture, state, &format!("frame-{frame}"), target, &d);
    }
}

/// After submission `sub` of a listed frame has run: the textures its lists
/// rendered to or wrote, as `frame-<n>-s<sub>-<i>-...` (what each holds
/// then, to follow a frame pass by pass).
pub(crate) fn dump_submission(sub: u64, targets: &[(usize, String)]) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(host) = crate::host::host() else { return };
    let frame = crate::queue::presented() + 1;
    peek(&format!("frame-{frame}-s{sub:04}"));
    for (i, (resource, by)) in targets.iter().enumerate() {
        let Some(copy) = crate::resource::host_copy(*resource as crate::object::This) else {
            continue;
        };
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
            continue;
        }
        let state = copy.state.load(Relaxed);
        let target = (i, *resource as crate::object::This, by.as_str());
        dump_one(host, texture, state, &format!("frame-{frame}-s{sub:04}"), target, &d);
        // A depth texture's writable twin (`crate::twin`): what unordered
        // views wrote, which reaches the depth texture at its next use.
        for (p, plane) in copy.twin.get().and_then(Option::as_ref).iter().flat_map(|t| t.planes.iter().enumerate()) {
            // SAFETY: a live resource.
            let pd = unsafe { plane.texture.GetDesc() };
            let by = format!("{by}-twin{p}");
            let target = (i, *resource as crate::object::This, by.as_str());
            dump_one(host, &plane.texture, plane.state.load(Relaxed), &format!("frame-{frame}-s{sub:04}"), target, &pd);
        }
    }
}

/// `VICTORY_PEEK=<address>[:<bytes>],...` (hex addresses, 16 bytes by
/// default): those bytes of title memory, as dwords and floats, after each
/// submission of a listed frame.
fn peek(when: &str) {
    static LIST: std::sync::OnceLock<Vec<(u64, usize)>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| {
        std::env::var("VICTORY_PEEK")
            .unwrap_or_default()
            .split(',')
            .filter_map(|e| {
                let (a, n) = e.split_once(':').unwrap_or((e, "16"));
                let a = u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?;
                Some((a, n.trim().parse().ok()?))
            })
            .collect()
    });
    for &(at, len) in list {
        let mut bytes = vec![0u8; len.min(256)];
        if !crate::host::read_into(at as *const u8, &mut bytes) {
            continue;
        }
        let words: Vec<String> = bytes
            .chunks(4)
            .map(|c| {
                let w = u32::from_le_bytes(c.try_into().unwrap_or_default());
                format!("{w:#x}={}", f32::from_bits(w))
            })
            .collect();
        tracing::info!("peek {when} {at:#x}: {}", words.join(" "));
    }
}

/// In a listed frame: copies an indirect execution's records (`args`:
/// buffer, offset; in the indirect-argument state) and its count (in the
/// unordered-access state unless in the same buffer) to readback memory,
/// logged once the list has run: how many records ask for work, and the
/// first ones.
pub(crate) fn read_indirect(
    list: &crate::host::HostList,
    args: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, max): (u32, u32, u32),
    name: String,
) {
    struct Readback(ID3D12Resource);
    // SAFETY: D3D12 objects are free-threaded.
    unsafe impl Send for Readback {}
    // SAFETY: as above.
    unsafe impl Sync for Readback {}
    let Some(host) = crate::host::host() else { return };
    let len = u64::from(stride) * u64::from(max.max(1));
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len + 4,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let mut buffer: Option<ID3D12Resource> = None;
    let l = &list.list;
    use crate::host::transition;
    use windows::core::Interface;
    let (arg_state, copy) = (D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    // SAFETY: plain creation; an open list and live buffers in the states
    // named, put back after the copies.
    unsafe {
        if host
            .device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer)
            .is_err()
        {
            return;
        }
        let Some(buffer) = buffer else { return };
        l.ResourceBarrier(&[transition(args.0, arg_state, copy)]);
        l.CopyBufferRegion(&buffer, 0, args.0, args.1, len);
        if let Some((c, at)) = count {
            let same = c.as_raw() == args.0.as_raw();
            let state = if same { copy } else { D3D12_RESOURCE_STATE_UNORDERED_ACCESS };
            if !same {
                l.ResourceBarrier(&[transition(c, state, copy)]);
            }
            l.CopyBufferRegion(&buffer, len, c, at, 4);
            if !same {
                l.ResourceBarrier(&[transition(c, copy, state)]);
            }
        }
        l.ResourceBarrier(&[transition(args.0, copy, arg_state)]);
        list.keep(&buffer);
        let buffer = Readback(buffer);
        let counted = count.is_some();
        list.defer_after(move || {
            // Recorded ahead: logged if the frame running it is listed.
            if !listed(crate::queue::presented() + 1) {
                return;
            }
            let mut data = std::ptr::null_mut();
            if buffer.0.Map(0, None, Some(&mut data)).is_err() {
                return;
            }
            let words = std::slice::from_raw_parts(data.cast::<u32>(), (len as usize + 4) / 4);
            let n = counted.then(|| words[len as usize / 4]);
            let records = n.unwrap_or(max).min(max) as usize;
            let step = stride as usize / 4;
            let first = |r: usize| &words[r * step..r * step + step.min(4)];
            // Draws: vertices (or indices) × instances; dispatches: groups.
            let work = |r: usize| -> u64 {
                let w = first(r);
                match kind {
                    0 | 1 => u64::from(w[0]) * u64::from(w.get(1).copied().unwrap_or(1)),
                    _ => w.iter().take(3).map(|&x| u64::from(x)).product(),
                }
            };
            let busy = (0..records).filter(|&r| work(r) > 0).count();
            let total: u64 = (0..records).map(work).sum();
            let shown: Vec<&[u32]> = (0..records.min(4)).map(first).collect();
            tracing::info!(
                target: "victory_gpu::trace",
                "indirect kind {kind} max {max} count {n:?}: {busy} of {records} records with work, {total} in all; first {shown:?} {name}"
            );
            buffer.0.Unmap(0, None);
        });
    }
}

fn dump_one(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    prefix: &str,
    (i, resource, by): (usize, crate::object::This, &str),
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
        // The stencil plane of a depth-stencil target, which the title's
        // lighting reads (material class in the low bits; 0x08 "inside a
        // room", which the lamps of a room are multiplied by): the plane
        // after the depth plane of a one-level, one-slice texture.
        let stencil = (matches!(d.Format.0, 19..=22 | 44..=47)
            && d.MipLevels == 1
            && d.DepthOrArraySize == 1)
            .then(|| read_back_of(&host.device, &list.list, texture, 1))
            .flatten();
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
        }
        (read, stencil)
    };
    let (read, stencil) = read;
    list.close();
    host.execute(&[&list]);
    if let Some((buffer, footprint)) = stencil {
        stencil_values(&buffer, &footprint, prefix, resource);
    }
    let Some((buffer, footprint)) = read else { return };
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), buffer.GetDesc().Width as usize);
        // A 24-bit depth plane comes back as 32-bit words: which depth values
        // (low 24 bits) and which top bytes it holds, the most common first.
        if matches!(d.Format.0, 44..=47) {
            let mut depths = std::collections::HashMap::<u32, u32>::new();
            let mut tops = std::collections::HashMap::<u32, u32>::new();
            for y in 0..f.Height as usize {
                for x in 0..f.Width as usize {
                    let at = y * pitch + x * 4;
                    let w = u32::from_le_bytes(rows[at..at + 4].try_into().unwrap_or([0; 4]));
                    *depths.entry(w & 0xff_ffff).or_default() += 1;
                    *tops.entry(w >> 24).or_default() += 1;
                }
            }
            let most = |m: std::collections::HashMap<u32, u32>| {
                let mut v: Vec<(u32, u32)> = m.into_iter().collect();
                v.sort_by_key(|e| std::cmp::Reverse(e.1));
                (v.len(), v.into_iter().take(12).collect::<Vec<_>>())
            };
            tracing::info!(?resource, depths = ?most(depths), top_bytes = ?most(tops), "24-bit depth plane values (value, texels)");
        }
        // Tiny targets (exposure, averages): their values as they are.
        if f.Width * f.Height <= 16 {
            let texel = crate::resource::element(f.Format.0 as u32).map_or(4, |(b, _)| b as usize);
            let words: Vec<String> = (0..f.Height as usize)
                .flat_map(|y| rows[y * pitch..][..f.Width as usize * texel].chunks(4))
                .map(|c| {
                    let w = u32::from_le_bytes(c.try_into().unwrap_or_default());
                    format!("{w:#x}={}", f32::from_bits(w))
                })
                .collect();
            tracing::info!(?resource, format = f.Format.0, "tiny target {prefix}-{i:02}: {}", words.join(" "));
        }
        // A 24-bit depth plane copies out as 32-bit words (`R32_TYPELESS`):
        // decode it as depth, not as floats (all denormals, shown black).
        let format = if matches!(d.Format.0, 44..=47) { d.Format.0 } else { f.Format.0 };
        if let Some(rgba) = to_rgba(format as u32, f.Width as usize, f.Height as usize, pitch, rows) {
            let mut name = format!(
                "{prefix}-{i:02}-{resource:?}-{}x{}-f{}",
                d.Width, d.Height, d.Format.0
            );
            // The pipeline's hashes (its name is "[stage hash, ...]").
            let by: String = by.chars().filter(char::is_ascii_alphanumeric).collect();
            if !by.is_empty() {
                name.push_str("-by-");
                name.push_str(&by);
            }
            // Whether anything but zeros is there (a dump is normalized).
            if rgba.chunks(4).all(|p| p[..3] == [0, 0, 0]) {
                name.push_str("-black");
            }
            save_as(&name, f.Width, f.Height, f.Width as usize * 4, false, &rgba);
        } else {
            tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        }
        buffer.Unmap(0, None);
    }
}

/// Logs which values a stencil plane read back by [`read_back_of`] holds
/// (the most common first), how many texels have each of its bits, and the
/// box of the texels with bit 3 (0x08).
fn stencil_values(
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    prefix: &str,
    resource: crate::object::This,
) {
    let f = footprint.Footprint;
    let (w, h, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows of
    // one byte a texel, the last row without its padding.
    unsafe {
        if h == 0 || w == 0 || buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * (h - 1) + w);
        let mut values = [0u32; 256];
        let mut bits = [0u32; 8];
        // Left, top, right, bottom of the texels with bit 3.
        let mut inside = [u32::MAX, u32::MAX, 0, 0];
        for y in 0..h {
            for (x, &v) in rows[y * pitch..y * pitch + w].iter().enumerate() {
                values[usize::from(v)] += 1;
                for (bit, count) in bits.iter_mut().enumerate() {
                    *count += u32::from(v >> bit & 1);
                }
                if v & 8 != 0 {
                    inside = [
                        inside[0].min(x as u32),
                        inside[1].min(y as u32),
                        inside[2].max(x as u32),
                        inside[3].max(y as u32),
                    ];
                }
            }
        }
        let mut most: Vec<(usize, u32)> = values
            .iter()
            .copied()
            .enumerate()
            .filter(|v| v.1 != 0)
            .collect();
        most.sort_by_key(|v| std::cmp::Reverse(v.1));
        most.truncate(16);
        tracing::info!(
            prefix,
            ?resource,
            width = w,
            height = h,
            values = ?most,
            texels_with_bit = ?bits,
            bit3_box = ?inside,
            "stencil plane values (value, texels)"
        );
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
/// as channel values scaled so the largest finite one is white (single
/// channels: lowest black, highest white, as grey). `None` for formats not
/// handled.
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
        // B8G8R8A8 and B8G8R8X8 (89 is R10G10B10_XR_BIAS_A2, below).
        87 | 88 | 90..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        23..=25 | 89 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [(v & 0x3ff) as f32, (v >> 10 & 0x3ff) as f32, (v >> 20 & 0x3ff) as f32]
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=10 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        11..=14 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(u16_at(b, i))))),
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
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, max) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    // Colour scaled by the 99th percentile of the non-zero values, so a few
    // hot pixels do not hide the rest; how many are not zero, and the median.
    let mut set: Vec<f32> = values
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite() && *v != 0.0)
        .map(f32::abs)
        .collect();
    let nonzero = set.len() as f64 / (values.len() * 3).max(1) as f64;
    let mut at = |q: f64| -> f32 {
        if set.is_empty() {
            return 0.0;
        }
        let k = ((set.len() - 1) as f64 * q).round() as usize;
        *set.select_nth_unstable_by(k, f32::total_cmp).1
    };
    let (p50, p99) = (at(0.5), at(0.99));
    let not_finite = values.iter().flatten().filter(|v| !v.is_finite()).count();
    let high = if grey || p99 == 0.0 { max } else { p99 };
    let range = (high - low).max(1e-20);
    // How much of the target holds anything: a lit scene and one bright
    // lamp on black have the same highest value.
    let above = values.iter().filter(|v| v.iter().any(|c| c.is_finite() && *c > low)).count();
    let nonzero_percent = above as f32 * 100.0 / values.len().max(1) as f32;
    tracing::info!(format, low, high = max, p50, p99, nonzero, not_finite, nonzero_percent, "target values");
    // Small float colour targets (the 1x1 luminance and exposure targets):
    // every texel's four channels as numbers. The picture has no alpha and
    // eight bits a channel.
    if matches!(format, 1..=2 | 9..=10) && w * h <= 16 {
        for y in 0..h {
            for x in 0..w {
                let b = px(x, y, n);
                let texel: [f32; 4] = if n == 16 {
                    [0, 1, 2, 3].map(|i| f32::from_bits(u32_at(b, i)))
                } else {
                    [0, 1, 2, 3].map(|i| half(u16_at(b, i)))
                };
                tracing::info!(format, x, y, ?texel, "target texel");
            }
        }
    }
    // R32_UINT targets hold packed light (the title reads them back as
    // R9G9B9E5): as floats their low, high and share above the lowest say
    // nothing. How many texels hold any bit, and the highest channel value
    // of the packed reading.
    if format == 42 {
        let (mut set, mut packed_high) = (0usize, 0f32);
        for y in 0..h {
            for x in 0..w {
                let v = u32_at(px(x, y, 4), 0);
                set += usize::from(v != 0);
                let scale = 2f32.powi((v >> 27) as i32 - 24);
                for k in 0..3 {
                    packed_high = packed_high.max((v >> (9 * k) & 0x1ff) as f32 * scale);
                }
            }
        }
        let nonzero_bits_percent = set as f32 * 100.0 / (w * h).max(1) as f32;
        tracing::info!(format, nonzero_bits_percent, packed_high, "target values as packed light");
        // The same words as plain numbers, by class. Tables of indices and
        // masks say nothing as floats or as packed light: in the title's
        // shadow-mask table 0 is a block in full shadow, a word up to 255
        // one value for the block, above that a block of the atlas, and all
        // ones a texel never written.
        let mut classes = [0usize; 6];
        let mut highest_below_ones = 0u32;
        for y in 0..h {
            for x in 0..w {
                let v = u32_at(px(x, y, 4), 0);
                classes[match v {
                    0 => 0,
                    1..=15 => 1,
                    16..=255 => 2,
                    256..=0x00ff_ffff => 3,
                    u32::MAX => 5,
                    _ => 4,
                }] += 1;
                if v != u32::MAX {
                    highest_below_ones = highest_below_ones.max(v);
                }
            }
        }
        tracing::info!(
            format,
            width = w,
            height = h,
            zero = classes[0],
            one_to_15 = classes[1],
            to_255 = classes[2],
            to_24_bits = classes[3],
            larger = classes[4],
            all_ones = classes[5],
            highest_below_ones,
            "target values as words"
        );
    }
    // Float colours (scene light) go through a cube root: scaled linearly,
    // a night scene under one lamp of 50 units came out all black.
    let curved = matches!(format, 1..=2 | 9..=10 | 26);
    for v in values {
        let c = v.map(|c| {
            let t = ((c - low) / range).clamp(0.0, 1.0);
            ((if curved { t.cbrt() } else { t }) * 255.0) as u8
        });
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
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
mod tests {
    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }

    /// R16G16B16A16_UNORM is integers, not half floats; 89 is 10:10:10:2,
    /// not B8G8R8A8.
    #[test]
    fn unorm16_and_xr_bias_targets_decode_as_their_formats() {
        // Red 0x8000 (a half-float -0), green 0x4000, blue 0.
        let unorm16 = [0x00, 0x80, 0x00, 0x40, 0, 0, 0xff, 0xff];
        assert_eq!(super::to_rgba(11, 1, 1, 8, &unorm16), Some(vec![255, 127, 0, 255]));
        // Red 0x3ff, green 0x1ff, blue 0, alpha 3.
        let xr = (0x3ffu32 | 0x1ff << 10 | 3 << 30).to_le_bytes();
        assert_eq!(super::to_rgba(89, 1, 1, 4, &xr), Some(vec![255, 127, 0, 255]));
        // B8G8R8A8 still swaps to RGBA.
        assert_eq!(super::to_rgba(87, 1, 1, 4, &[1, 2, 3, 4]), Some(vec![3, 2, 1, 255]));
    }
}
