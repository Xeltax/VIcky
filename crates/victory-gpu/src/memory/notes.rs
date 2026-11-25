//! Shaders' notes in the card's memory (`VICTORY_NOTES_ON_CARD=1`, see
//! [`on_card`]).
//!
//! Every access a shader makes to title memory by address stores a note in
//! the `SLOT_FAULTS` buffer (a fault, a touch, or the sink), and every
//! store one more (its page's mark): `victory_shader::emit::MEMORY_NOTES`.
//! With that buffer in the PC's memory each of those stores crosses the
//! bus. A buffer in the card's memory that nobody read took the structure
//! builder's `CopyForBVHUpdate` from 28-70 ms of the card's time a frame to
//! 10-12, and the frame from 157-209 ms to 136-146 (RTX 4080, structure
//! builder on). Unread, the title stalled: the host has to see the notes.
//!
//! With the switch the buffer shaders write is in the card's memory, and
//! the host reads a copy of it:
//!
//! - once a frame a list of ours ([`list`]) copies the buffer into
//!   read-back memory and then clears it. Both are on the one direct queue,
//!   with nothing between them, so a note is either in that copy or still
//!   in the buffer for the next one;
//! - the list goes first in a submission, where the buffer is in the
//!   common state (buffers fall back to it at the end of every
//!   `ExecuteCommandLists`), and leaves it common for the lists behind it.
//!   It goes with the present's submission ([`presented`]), or, without
//!   presents, with the first submission of a frame;
//! - the host reads the copy (`crate::memory`, through [`take`]) once that
//!   submission has finished, as `crate::host::submissions` says (the late
//!   waits and the relay land their flights there too), at the first
//!   dispatch by address of a list after that: one that brings every copy
//!   up to date, the copies the faults number then too. It clears nothing:
//!   the card did;
//! - no copy is sent while the one before it has not been read, so the
//!   read-back memory is never written while it is read. What shaders note
//!   meanwhile waits in the buffer for the next copy.
//!
//! So the host sees every fault, mark and touch. With a copy at each
//! present that has finished by the next frame's first draw, it sees them
//! where it did with the buffer in the PC's memory; otherwise a frame
//! later at most when the card keeps up, and as late as the card is
//! behind when it does not. A page that faulted gets its copy that much
//! later; marks and touches count for `MARK_FRAMES` frames, which covers
//! it.
//!
//! shortcut: the whole buffer (8 MiB, most of it the page marks) is copied
//! and cleared every frame, about a third of a millisecond over the bus;
//! the pages the host has copies of are all it reads.
//!
//! The directory stays in the PC's memory: the host writes its entries
//! when it numbers copies (`Memory::map`) and shaders only read it.

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_shader::emit::MEMORY_NOTES;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC};
use windows::core::Interface;

/// `VICTORY_NOTES_ON_CARD=1`: shaders write their notes on title memory by
/// address (faults, page marks, touches) into a buffer in the card's own
/// memory, which a list of ours copies back and clears once a frame (see
/// the module). Unset: into a buffer in the PC's memory that the host reads
/// and clears through its mapping, as before.
///
/// shortcut: the host sees the notes up to a frame later than before; a
/// page shaders found without a copy gets its copy that much later.
pub(crate) fn on_card() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NOTES_ON_CARD").is_some_and(|v| v == "1"))
}

/// Where the host reads shaders' notes.
pub(super) enum Notes<'a> {
    /// The buffer shaders write, in the PC's memory and mapped
    /// (`MEMORY_NOTES` dwords): a note is cleared once read. The card may
    /// be writing: a note set after it is read is found the next time.
    Shared(*mut u32),
    /// A copy of the buffer in the card's memory, which the card cleared.
    /// Each note is taken once per copy, as the shared buffer's clear does
    /// (`taken`: the notes found set so far).
    Copied {
        notes: &'a [u32],
        taken: RefCell<HashSet<usize>>,
    },
}

impl<'a> Notes<'a> {
    pub(super) fn copied(notes: &'a [u32]) -> Notes<'a> {
        Notes::Copied {
            notes,
            taken: RefCell::new(HashSet::new()),
        }
    }

    /// Dword `i` (below `MEMORY_NOTES`) if it is set and not taken yet,
    /// else 0. It is taken.
    pub(super) fn take(&self, i: usize) -> u32 {
        match self {
            Notes::Shared(at) => {
                // SAFETY: inside the mapping (`MEMORY_NOTES` dwords); the
                // card may be writing: a note set after this read is found
                // the next time.
                let at = unsafe { at.add(i) };
                // SAFETY: as above.
                let note = unsafe { at.read_volatile() };
                if note != 0 {
                    // SAFETY: as above.
                    unsafe { at.write_volatile(0) };
                }
                note
            }
            Notes::Copied { notes, taken } => match notes[i] {
                0 => 0,
                note if taken.borrow_mut().insert(i) => note,
                _ => 0,
            },
        }
    }
}

/// The buffer shaders write and what copies it back.
pub(super) struct Card {
    /// In the card's memory, bound as `SLOT_FAULTS`; common between
    /// submissions.
    pub(super) buffer: ID3D12Resource,
    /// Read-back memory of the same size: the last copy.
    back: ID3D12Resource,
    dwords: u32,
    /// Copies `buffer` into `back` and clears it: recorded once, closed,
    /// and sent again for every copy (never while it may still run:
    /// [`Turn::send`] waits for the copy before to be read, which waits for
    /// its submission to finish).
    list: ID3D12GraphicsCommandList,
    /// What the list uses: its allocator, and the clear's view of `buffer`
    /// in a heap the list sets and in one only the CPU sees.
    _held: (
        ID3D12CommandAllocator,
        ID3D12DescriptorHeap,
        ID3D12DescriptorHeap,
    ),
}

// SAFETY: D3D12 objects are free-threaded; the struct holds no mapping.
unsafe impl Send for Card {}
// SAFETY: as above.
unsafe impl Sync for Card {}

impl Card {
    /// The buffer (`dwords` of them, zeros), its read-back memory, and the
    /// list that copies and clears it.
    fn new(device: &ID3D12Device, dwords: u32) -> windows::core::Result<Card> {
        let bytes = u64::from(dwords) * 4;
        let buffer = committed(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let back = committed(
            device,
            D3D12_HEAP_TYPE_READBACK,
            bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        // SAFETY: plain creation of objects of our own; the views are of a
        // live buffer, written into fresh heaps.
        unsafe {
            let heap = |flags| {
                device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: flags,
                    NodeMask: 0,
                })
            };
            let (shown, hidden) = (
                heap(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)?,
                heap(D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?,
            );
            let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: dwords,
                        StructureByteStride: 0,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                    },
                },
            };
            for heap in [&shown, &hidden] {
                device.CreateUnorderedAccessView(
                    &buffer,
                    None::<&ID3D12Resource>,
                    Some(&view),
                    heap.GetCPUDescriptorHandleForHeapStart(),
                );
            }
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            record(&list, &buffer, &back, bytes, &shown, &hidden);
            list.Close()?;
            Ok(Card {
                buffer,
                back,
                dwords,
                list,
                _held: (allocator, shown, hidden),
            })
        }
    }

    /// Hands `read` the last copy. Only once its submission has finished,
    /// and before the next copy is sent ([`take`]).
    fn read(&self, read: impl FnOnce(&[u32])) {
        let bytes = self.dwords as usize * 4;
        let mut data = std::ptr::null_mut();
        // SAFETY: read-back memory of ours, mapped for reading all of it.
        if let Err(e) = unsafe {
            self.back.Map(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: bytes,
                }),
                Some(&mut data),
            )
        } {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(error = %e, "title memory by address: the copy of the card's notes cannot be read"));
            return;
        }
        // SAFETY: the mapping holds `dwords` dwords; the copy into them has
        // run and the next is not sent before this one has been read.
        read(unsafe { std::slice::from_raw_parts(data.cast::<u32>(), self.dwords as usize) });
        // SAFETY: mapped above; nothing was written.
        unsafe { self.back.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 })) };
    }
}

/// A committed buffer of `bytes` in a heap of `kind`.
fn committed(
    device: &ID3D12Device,
    kind: D3D12_HEAP_TYPE,
    bytes: u64,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a buffer.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut buffer,
        )?;
    }
    buffer.ok_or_else(windows::core::Error::empty)
}

/// Records the copy of `buffer` (common: the list goes first in a
/// submission) into `back` (read-back memory, always a copy destination),
/// then the clear of `buffer` through its view at the start of `shown` (set
/// on the list) and of `hidden`. `buffer` ends common, as the lists behind
/// it in the submission find it.
///
/// # Safety
///
/// `list` is open and the resources live, of `bytes` each.
unsafe fn record(
    list: &ID3D12GraphicsCommandList,
    buffer: &ID3D12Resource,
    back: &ID3D12Resource,
    bytes: u64,
    shown: &ID3D12DescriptorHeap,
    hidden: &ID3D12DescriptorHeap,
) {
    use crate::host::transition;
    // SAFETY: the caller's open list and live resources; each barrier names
    // the state the one before left.
    unsafe {
        list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyBufferRegion(back, 0, buffer, 0, bytes);
        list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        list.SetDescriptorHeaps(&[Some(shown.clone())]);
        list.ClearUnorderedAccessViewUint(
            shown.GetGPUDescriptorHandleForHeapStart(),
            hidden.GetCPUDescriptorHandleForHeapStart(),
            buffer,
            &[0; 4],
            &[],
        );
        list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
}

/// The copy sent and not read yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pending {
    /// The submission it went first in (`crate::host::submissions`).
    submission: u64,
    /// The frame it was sent in.
    frame: u64,
    /// The first copy: it only clears what the buffer held when it was
    /// made (no shader had run with it), and is not read.
    first: bool,
}

/// When a copy is sent and when it is read.
#[derive(Debug)]
struct Turn {
    /// What [`opens`] said when the last copy was sent; `None` before the
    /// first.
    copied: Option<u64>,
    pending: Option<Pending>,
}

impl Turn {
    const fn new() -> Turn {
        Turn {
            copied: None,
            pending: None,
        }
    }

    /// Whether a copy goes first in submission `submission`, sent in
    /// `frame`, with [`opens`] saying `opens` now. If so it is counted as
    /// sent. Not while the one before has not been read: it would be
    /// written over, and the list may still run.
    fn send(&mut self, opens: u64, frame: u64, submission: u64) -> bool {
        if self.pending.is_some() || self.copied.is_some_and(|c| c >= opens) {
            return false;
        }
        self.pending = Some(Pending {
            submission,
            frame,
            first: self.copied.is_none(),
        });
        self.copied = Some(opens);
        true
    }

    /// The copy not read yet, if every submission up to `finished` has run
    /// and its own is one of them.
    fn landed(&self, finished: u64) -> Option<Pending> {
        self.pending.filter(|p| p.submission <= finished)
    }

    /// The copy has been read: the next may be sent.
    fn read(&mut self) {
        self.pending = None;
    }
}

/// The frame a copy sent now opens (what shaders note after it is that
/// frame's): the current frame, or the next once the current one has been
/// presented (`presented`: the frame after the last one presented).
fn opens(current: u64, presented: u64) -> u64 {
    current.max(presented)
}

static CARD: OnceLock<Option<Card>> = OnceLock::new();
static TURN: Mutex<Turn> = Mutex::new(Turn::new());
/// The frame after the last one presented; 0 before the first present.
static PRESENTED: AtomicU64 = AtomicU64::new(0);

fn turn() -> std::sync::MutexGuard<'static, Turn> {
    TURN.lock().unwrap_or_else(|e| e.into_inner())
}

/// The buffer in the card's memory, made on the first call; `None` if the
/// host refused it (the notes then stay in the PC's memory).
pub(super) fn card() -> Option<&'static Card> {
    CARD.get_or_init(|| {
        let host = crate::host::host()?;
        match Card::new(&host.device, MEMORY_NOTES) {
            Ok(card) => {
                crate::host::name(&card.buffer, || "title memory by address: notes on the card".to_owned());
                crate::host::name(&card.back, || "title memory by address: notes copied back".to_owned());
                tracing::info!(
                    "title memory by address: shaders' notes go to the card's memory, copied back once a frame (VICTORY_NOTES_ON_CARD)"
                );
                Some(card)
            }
            Err(e) => {
                tracing::warn!(error = %e, "title memory by address: no buffer for the notes in the card's memory; they stay in the PC's memory");
                None
            }
        }
    })
    .as_ref()
}

/// For submission `submission`, being prepared behind the submission lock
/// (`Host::submit`): the list that copies the notes back and clears them,
/// to go first, if a copy is due and the one before has been read. `None`
/// unless the notes are on the card.
pub(crate) fn list(submission: u64) -> Option<ID3D12CommandList> {
    let card = CARD.get()?.as_ref()?;
    let list: ID3D12CommandList = card.list.cast().ok()?;
    let frame = crate::frame::current();
    let opens = opens(frame, PRESENTED.load(Ordering::Relaxed));
    turn().send(opens, frame, submission).then_some(list)
}

/// A present is about to be submitted: the frame ends with it, so the copy
/// goes with it (or the first submission after it) and not with the next
/// frame's first submission.
pub(crate) fn presented() {
    PRESENTED.fetch_max(crate::frame::current() + 1, Ordering::Relaxed);
}

/// Hands `read` the last copy of the notes if its submission has finished
/// and it was not read yet (the first copy is not handed: see [`Pending`]).
/// Returns the frame the copy was sent in, if one was taken. One caller at
/// a time (`crate::memory` holds its lock).
pub(super) fn take(read: impl FnOnce(&[u32])) -> Option<u64> {
    let card = CARD.get()?.as_ref()?;
    let landed = turn().landed(crate::host::submissions().1)?;
    // Not under the lock: no copy is sent before `read` below.
    if !landed.first {
        card.read(read);
    }
    turn().read();
    Some(landed.frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::s;

    /// The first copy goes with the first submission and only clears; the
    /// next waits until it is read, and until a frame later.
    #[test]
    fn a_copy_waits_for_the_one_before_to_be_read() {
        let mut t = Turn::new();
        assert!(t.send(5, 5, 10), "the first copy goes at once");
        assert_eq!(t.landed(9), None, "its submission has not finished");
        let first = t.landed(10).expect("landed");
        assert!(first.first, "the first copy is not read");
        // Not read yet: no copy, even in a later frame.
        assert!(!t.send(6, 6, 11));
        t.read();
        // Read: a copy when the frame it opens is a later one.
        assert!(t.send(6, 6, 12));
        assert_eq!(
            t.landed(12),
            Some(Pending {
                submission: 12,
                frame: 6,
                first: false
            })
        );
        t.read();
        assert!(!t.send(6, 6, 13), "one copy a frame");
        assert!(t.send(7, 7, 14));
        // Submissions finish in order: a later number has passed it too.
        assert!(t.landed(20).is_some());
    }

    /// With presents, the copy goes with the present's submission, and the
    /// next frame's first submission sends none.
    #[test]
    fn a_present_carries_the_frame_s_copy() {
        let mut t = Turn::new();
        assert!(t.send(opens(3, 0), 3, 1));
        t.read();
        // Frame 3's draws: nothing; its present opens frame 4.
        assert!(!t.send(opens(3, 0), 3, 2));
        assert!(t.send(opens(3, 4), 3, 3), "the present's submission");
        t.read();
        assert!(!t.send(opens(4, 4), 4, 4), "frame 4's first submission");
        // A present held back in frame 4 still finds its copy read: it
        // opens frame 5.
        assert!(t.send(opens(4, 5), 4, 5));
        // Not read when the present comes: the next submission takes it.
        assert!(!t.send(opens(5, 6), 5, 6));
        t.read();
        assert!(t.send(opens(6, 6), 6, 7));
    }

    /// A note is taken once a read, from the shared buffer (cleared) and
    /// from a copy alike.
    #[test]
    fn a_note_is_taken_once() {
        let mut shared = vec![0u32, 7, 0, 9];
        let notes = Notes::Shared(shared.as_mut_ptr());
        assert_eq!(
            (notes.take(1), notes.take(1), notes.take(2), notes.take(3)),
            (7, 0, 0, 9)
        );
        assert_eq!(shared, [0, 0, 0, 0]);
        let copy = [0u32, 7, 0, 9];
        let notes = Notes::copied(&copy);
        assert_eq!(
            (notes.take(1), notes.take(1), notes.take(2), notes.take(3)),
            (7, 0, 0, 9)
        );
        // A new copy starts over.
        assert_eq!(Notes::copied(&copy).take(1), 7);
    }

    /// Stores `value` at byte `at` of the root UAV, as shaders note.
    const NOTE: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0)"
cbuffer What : register(b0) { uint at; uint value; };
RWByteAddressBuffer notes : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { notes.Store(at, value); }
"#;

    /// A device, its direct queue and the noting shader.
    struct Rig {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
        /// With `VICTORY_D3D12_DEBUG` set: the debug layer's messages (its
        /// errors fail the test: resource states the barriers get wrong).
        messages: Option<ID3D12InfoQueue>,
    }

    impl Rig {
        fn new(warp: bool) -> Option<Rig> {
            // SAFETY: plain DXGI and D3D12 set-up.
            unsafe {
                let debug = std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
                if debug {
                    let mut layer: Option<ID3D12Debug> = None;
                    if D3D12GetDebugInterface(&mut layer).is_ok() {
                        layer.iter().for_each(|l| l.EnableDebugLayer());
                    }
                }
                let mut device: Option<ID3D12Device> = None;
                if warp {
                    let factory: IDXGIFactory4 =
                        CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                    let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                    D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                } else {
                    D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                }
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let mut code: Option<ID3DBlob> = None;
                D3DCompile(
                    NOTE.as_ptr().cast(),
                    NOTE.len(),
                    s!("note"),
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )
                .expect("the test's shader compiles");
                let code = code?;
                let bytes = std::slice::from_raw_parts(
                    code.GetBufferPointer().cast::<u8>(),
                    code.GetBufferSize(),
                );
                let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).ok()?;
                let pso: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: bytes.as_ptr().cast(),
                            BytecodeLength: bytes.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;
                let messages = debug
                    .then(|| device.cast::<ID3D12InfoQueue>().ok())
                    .flatten();
                Some(Rig {
                    device,
                    queue,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pso,
                    messages,
                })
            }
        }

        /// The debug layer's errors since the last call.
        fn errors(&self) -> Vec<String> {
            let Some(queue) = &self.messages else {
                return Vec::new();
            };
            let mut errors = Vec::new();
            // SAFETY: each message is read into a buffer of the size the
            // queue reports, then the queue is emptied.
            unsafe {
                for i in 0..queue.GetNumStoredMessages() {
                    let mut size = 0;
                    if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                        continue;
                    }
                    let mut buffer = vec![0u64; size.div_ceil(8)];
                    let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if queue.GetMessage(i, Some(message), &mut size).is_ok()
                        && (*message).Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0
                    {
                        let m = &*message;
                        let text =
                            std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        errors.push(
                            String::from_utf8_lossy(text)
                                .trim_end_matches('\0')
                                .to_owned(),
                        );
                    }
                }
                queue.ClearStoredMessages();
            }
            errors
        }

        /// A closed list that notes `value` at dword `dword` of `notes`;
        /// with its allocator, kept until it has run.
        fn note(
            &self,
            notes: &ID3D12Resource,
            dword: u32,
            value: u32,
        ) -> (ID3D12CommandAllocator, ID3D12CommandList) {
            // SAFETY: plain creation; an open list of ours; a live buffer
            // with unordered access, common (promoted by the root view).
            unsafe {
                let allocator: ID3D12CommandAllocator = self
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .expect("an allocator");
                let list: ID3D12GraphicsCommandList = self
                    .device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .expect("a list");
                list.SetComputeRootSignature(&self.root);
                list.SetPipelineState(&self.pso);
                let constants = [dword * 4, value];
                list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                list.SetComputeRootUnorderedAccessView(1, notes.GetGPUVirtualAddress());
                list.Dispatch(1, 1, 1);
                list.Close().expect("the list closes");
                (allocator, list.cast().expect("a command list"))
            }
        }

        /// One submission of `lists`, waited for.
        fn run(&mut self, lists: &[ID3D12CommandList]) {
            let lists: Vec<Option<ID3D12CommandList>> = lists.iter().cloned().map(Some).collect();
            self.submitted += 1;
            // SAFETY: closed lists; our queue, fence and event.
            unsafe {
                self.queue.ExecuteCommandLists(&lists);
                self.queue
                    .Signal(&self.fence, self.submitted)
                    .expect("a signal");
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .expect("a wait");
                WaitForSingleObject(self.event, INFINITE);
                assert!(
                    self.device.GetDeviceRemovedReason().is_ok(),
                    "the device was lost"
                );
            }
            let errors = self.errors();
            assert!(errors.is_empty(), "the debug layer: {errors:?}");
        }
    }

    /// What shaders note in the card's memory comes back in the copy the
    /// list makes, and the list's clear leaves only what is noted after it
    /// for the next copy: on the card and on the software adapter.
    #[test]
    fn notes_on_the_card_come_back_once_and_are_cleared() {
        const DWORDS: u32 = 4096 + 17;
        let mut ran = 0;
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Some(mut rig) = Rig::new(warp) else {
                eprintln!("no {label} device; skipped");
                continue;
            };
            ran += 1;
            let card = Card::new(&rig.device, DWORDS).expect("the card's notes");
            let copy: ID3D12CommandList = card.list.cast().expect("a command list");
            let read = |card: &Card| {
                let mut out = Vec::new();
                card.read(|notes| out = notes.to_vec());
                out
            };
            // The first copy, alone: the buffer is clear behind it, whatever
            // it held when made (what it copies is not read, as `take`).
            rig.run(std::slice::from_ref(&copy));
            // Shaders note in a submission of their own.
            let (a, b) = (
                rig.note(&card.buffer, 3, 7),
                rig.note(&card.buffer, DWORDS - 1, 9),
            );
            rig.run(&[a.1.clone(), b.1.clone()]);
            // The next submission: the copy first, then more notes.
            let c = rig.note(&card.buffer, 5, 11);
            rig.run(&[copy.clone(), c.1.clone()]);
            let back = read(&card);
            assert_eq!(
                (back[3], back[DWORDS as usize - 1], back[5]),
                (7, 9, 0),
                "{label}: the first notes came back"
            );
            assert_eq!(
                back.iter().filter(|&&d| d != 0).count(),
                2,
                "{label}: nothing else"
            );
            // The copy after: only what came after the clear.
            rig.run(std::slice::from_ref(&copy));
            let back = read(&card);
            assert_eq!(back[5], 11, "{label}: the later note came back");
            assert_eq!(
                back.iter().filter(|&&d| d != 0).count(),
                1,
                "{label}: the clear took the rest"
            );
            drop((a, b, c));
        }
        if ran == 0 {
            eprintln!("no device ran the notes test");
        }
    }
}
