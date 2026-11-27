//! VIcky: keeps the card's clocks up between submissions
//! (`VICTORY_CARD_AWAKE=1|alt`; off by default). Port of foxhound 9f8838b1
//! (`host/awake.rs`, there `VICTORY_CARD_AWAKE=1|turns`).
//!
//! The card stands idle part of every frame, many times for a millisecond
//! or two, while the CPU prepares the next submission (with ray tracing on
//! here: some 46 % of the frame). The driver takes that for a light load
//! and lowers the clocks, and what comes next starts slow: in foxhound's
//! measure a captured frame whose lists take 30 ms sent in one go took
//! 38-40 ms with waits of 1.2-2.4 ms between its submissions.
//!
//! One group of 64 threads counts on a compute queue of its own, over and
//! over (about 2 ms a dispatch at full clocks, never more than two on the
//! way): it takes one or two of the card's thousands of wave slots, and the
//! card is never idle. Nothing is sent once no submission of the title's
//! has gone out for a quarter of a second (a title that stands still), nor
//! with `alt` in the phases without it.
//!
//! What it costs: the queue's dispatches share the card with the title's
//! work (a wave or two of 64 threads), and the card draws more power while
//! the CPU prepares. What it does not change: nothing of the title's is
//! read or written; the queue, its lists and its buffer are its own.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

/// About 2 ms at full clocks: some 500 sends a second.
const COUNT: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer sum : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    [loop] for (uint i = 0; i < 400000u; i++) x = x * 1664525u + 1013904223u;
    sum.Store(id.x * 4u, x);
}
"#;

/// Quiet this long after the title's last submission.
const QUIET_MS: u64 = 250;

/// The switch.
static CARD_AWAKE: crate::alt::Alt = crate::alt::Alt::new(
    "VICTORY_CARD_AWAKE",
    "the card is kept awake between submissions by a compute queue of its own (port of foxhound 9f8838b1)",
);

/// Since the last frame line: dispatches sent to keep the card awake.
static SENT: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// When a submission of the title's was last sent ([`now_ms`] plus one; 0:
/// never).
static LAST: AtomicU64 = AtomicU64::new(0);

fn wanted() -> bool {
    let (last, now) = (LAST.load(Ordering::Relaxed), now_ms());
    last != 0 && now + 1 < last + QUIET_MS && CARD_AWAKE.on()
}

/// A submission of the title's went to `device`'s direct queue.
pub(super) fn sent(device: &ID3D12Device) {
    if !CARD_AWAKE.set() {
        return;
    }
    LAST.store(now_ms() + 1, Ordering::Relaxed);
    // (With `alt`, the phase changes are said on the title's time too.)
    CARD_AWAKE.on();
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        // Runs as long as the process does.
        std::mem::forget(start(device, wanted));
    });
}

/// The frame line's clause (empty without the switch).
pub(super) fn summary(frames: f64) -> String {
    if !CARD_AWAKE.set() {
        return String::new();
    }
    format!(
        " card awake ({}): {:.1} dispatches of its own;",
        CARD_AWAKE.said(),
        SENT.swap(0, Ordering::Relaxed) as f64 / frames.max(1.0)
    )
}

/// The thread that keeps the card busy; it ends when this is dropped.
pub(crate) struct Awake {
    stop: std::sync::Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Awake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Keeps the card of `device` busy while `wanted`.
pub(crate) fn start(device: &ID3D12Device, wanted: impl Fn() -> bool + Send + 'static) -> Option<Awake> {
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let (device, ended) = (device.clone(), stop.clone());
    let thread = std::thread::Builder::new()
        .name("victory-card-awake".into())
        .spawn(move || {
            if let Err(e) = run(&device, &wanted, &ended) {
                tracing::warn!(error = %e, "the card is not kept awake (VICTORY_CARD_AWAKE)");
            }
        })
        .ok()?;
    Some(Awake { stop, thread: Some(thread) })
}

fn run(device: &ID3D12Device, wanted: &dyn Fn() -> bool, stop: &AtomicBool) -> windows::core::Result<()> {
    let code = super::compile(COUNT, windows::core::s!("main"), windows::core::s!("cs_5_1"))?;
    // SAFETY: plain object creation; the code lives through the calls, the
    // root signature is inside it. Two closed lists, sent in turn, whose
    // objects live until what was sent last has run.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let pso: ID3D12PipelineState = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
            ..Default::default()
        })?;
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            ..Default::default()
        })?;
        let mut sum: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut sum,
        )?;
        let sum = sum.ok_or_else(windows::core::Error::empty)?;
        let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)?;
        let mut lists = Vec::new();
        for _ in 0..2 {
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, &pso)?;
            list.SetComputeRootSignature(&root);
            list.SetComputeRootUnorderedAccessView(0, sum.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            list.Close()?;
            lists.push(Some(list.cast::<ID3D12CommandList>()?));
        }
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let mut value = 0u64;
        let mut result = Ok(());
        while !stop.load(Ordering::Relaxed) && result.is_ok() {
            if !wanted() {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            // Never more than two on the way.
            if value >= 2 && fence.GetCompletedValue() + 2 <= value {
                result = fence.SetEventOnCompletion(value - 1, event);
                if result.is_ok() {
                    WaitForSingleObject(event, 1000);
                    result = device.GetDeviceRemovedReason();
                }
                continue;
            }
            queue.ExecuteCommandLists(&lists[(value % 2) as usize..][..1]);
            value += 1;
            SENT.fetch_add(1, Ordering::Relaxed);
            result = queue.Signal(&fence, value);
        }
        if result.is_ok() && fence.GetCompletedValue() < value && fence.SetEventOnCompletion(value, event).is_ok() {
            WaitForSingleObject(event, INFINITE);
        }
        let _ = CloseHandle(event);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The thread sends its dispatches while wanted and ends, with all of
    /// them run, when it is let go; the device is still there.
    #[test]
    fn the_card_is_kept_awake_and_let_go() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let before = SENT.load(Ordering::Relaxed);
        let awake = start(&host.device, || true);
        assert!(awake.is_some());
        std::thread::sleep(Duration::from_millis(50));
        drop(awake);
        assert!(SENT.load(Ordering::Relaxed) > before, "no dispatch was sent");
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
