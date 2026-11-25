# VIcky `rt`: ray-tracing work branch

This branch is for making the title's compute ray tracing (the BVH structure builder plus traversal) usable on PC.

- **Base:** `fx-heapmap-8d`, which is fxceless/victory-fork `8dd30ab` + VIcky's heap map + fork PR #188 (Series S).
- **Status:** research. It builds, and the gpu, shader and hle suites and `cargo xtask policy` pass. Game measurements are below.

## What is merged on top of `fx-heapmap-8d`

| fork PR | author | what |
|---|---|---|
| #36 | adriaespii | the builder translates on 32-lane GPUs (`lane_uniform`) |
| #646 | mynzjerko | DispatchX flag `0x20` counts threads (fixes rehydration writes) |
| #579 | Revens2 | `VICTORY_BVH_SKIP=<indices>` drops builder stages (diagnostics) |
| #213 | Pablo-Barros | an absent BVH child is a miss |
| #346 | Gargalindis | stores through title memory by address mark their copy; only marked copies are snapshotted and compared |

**VIcky's own commits.** All are opt-in.
- `VICTORY_BACK_JOIN=<bytes>` writes nearby runs of shader writes back as one kernel call. The builder's output gave about 850,000 runs a frame, one `NtWriteVirtualMemory` each.
- `VICTORY_MEMORY_REFRESH=touched`: shaders note which copies they read (no extra store), and only those are refreshed, plus every copy once in 8 frames.
- `VICTORY_MEMORY_NAMED_WRITES=0`: a copy that is only named by a constant and never marked stops being snapshotted.
- `VICTORY_SNAPSHOT_POOL_MB=<n>` keeps snapshot buffers by size for reuse.
- `VICTORY_REFRESH_STAMP=1`: measured 0 skips; useless, will go.

**Recommended with the builder on:** `VICTORY_BACK_JOIN=256 VICTORY_MEMORY_REFRESH=touched VICTORY_MEMORY_NAMED_WRITES=0 VICTORY_SNAPSHOT_POOL_MB=2048`.

## Run

Build and run as for `fx-heapmap-8d` (see `VICKY.md`). The builder is off by default.

- `VICTORY_BVH_BUILDER=1` turns it on. It is all or nothing: every builder pipeline must build at its first dispatch.
- Our 4080 settings: no loop limit, `VICTORY_SHADER_OPT=0`, and the 14 `CS_ShadowRenderTiled*` hashes in `VICTORY_WAVE_DEEP_SKIPS_NOT` (see `VICKY.md`).
- Restore your save before each run. The game autosaves, so later runs start at a later point.
- Do two runs or more per setting: the spread between runs is large.

## Measurements so far (RTX 4080, Series X, one run each unless said)

| run | builder | ms/frame | GPU | preparing | copying back | notes |
|---|---|---|---|---|---|---|
| RT0 | off | 361 | 185 | 87 | 81 | 0 MB compared in windows |
| RT3 | on (before #346) | 4,523 | 286 | 1,678 | 2,518 | 16.2 GB snapshotted and compared a frame; 4.17 M page faults |
| RT4 | on + `VICTORY_INPLACE=1` (before #346) | 3,155–3,330 | 265–278 | 1,295–1,378 | 1,544–1,659 | 311 windows in place; 3,671 refused (page-pool views) |

| RT5b | on + #346 | 849–916 | 264–301 | 237–256 | 327–354 | 515 MB compared; all 1,023 copies (21.5 GB) refreshed |
| RT7 | on + #346 + touched, named, pool | 652–718 (same shot as RT5b's 790–871) | 214–238 | 151–179 | 264–281 | 6.4–7.8 GB refreshed |
| RT8 | RT7 + `VICTORY_BACK_JOIN=256` | **502–583** (1.7–2.0 fps) | 282–306 | 149–193 | **58–65** | 1,300 kernel writes instead of ~850,000 |

| **RT10** | RT8's switches, **without `VICTORY_GPU_TIMING`** | **253–315 (3.2–3.9 fps)** | 70–122 | 126–173 | 115–184 | the relay overlaps CPU and GPU: card 68% busy; the opening cutscene plays through to the first gameplay objective |

**Read this before comparing numbers.** `VICTORY_GPU_TIMING=1` turns off late wait and the relay (`host.rs` `late_stage`). Every submission is then waited for, so the CPU and GPU strictly alternate. RT0–RT9 above ran with it, to get per-pipeline GPU times, so their frame times are pessimistic. RT10 is the same build and switches as RT8 with timing off: about 1.8× faster.

### RTX 5090 (driver 32.0.16.1742, Ryzen 7 9800X3D, 32 lanes), Series X, `493cc0d`

Task 1 below. Two runs of 600 s per setting after one cache fill per mode, alternating off, on, off, on, each from the same save, no input once the world is up. No `VICTORY_GPU_TIMING`, so compare with RT10, not RT0–RT8. Other settings as "Run" above; builder on uses the recommended four switches. The ranges are the last 12 `fps` lines of each run.

| run | builder | fps | ms/frame | GPU | preparing | copying back | dropped draws | shader writes looked for on the CPU, per frame |
|---|---|---|---|---|---|---|---|---|
| 5090-off-1 | off | 4.9–5.0 | 199–204 | 105–112 | 65–72 | 156–172 | 336–350 | 0 MB in windows; 24.4–24.6 MB in 259–263 ranges of buffers' copies |
| 5090-off-2 | off | 5.0–5.1 | 198–202 | 104–109 | 65–70 | 157–165 | 339–350 | 0 MB in windows; 23.3–23.6 MB in 257–260 ranges |
| 5090-on-1 | on | 4.6–4.9 | 206–219 | 73–89 | 99–122 | 165–179 | 4 | 236–258 MB in 63–69 ranges of windows (175–184 MB snapshotted); 25.3–25.6 MB in buffers' copies |
| 5090-on-2 | on | 4.2–4.7 | 213–238 | 65–92 | 130–145 | 150–175 | 4 | 313–346 MB in 81–90 ranges of windows (215–247 MB snapshotted); 27.4–31.0 MB in buffers' copies |

- The builder adds about 6% (run 1) and 13% (run 2) to the frame here (198–204 ms off, 206–238 ms on). The extra time is in preparing (65–72 ms off, 99–145 ms on); the GPU part is lower with the builder on.
- With the builder on, draws dropped without a host pipeline fall from about 345 a frame to 4.
- The second builder-on run compared about 40% more window memory than the first, from the same save and cache, and spent about 25% longer preparing.
- All six runs (two fills of 900 s included) closed cleanly with exit code 0: no device removal, no panic. The same 10 warnings each run: 6 compute pipelines the host refuses (`0x80070057`) and 4 shaders that do not translate.
- Not looked at: the picture (no captures were taken), so nothing here on the dark corner.

- Without #346, the CPU found 85 MB of shader writes a frame by comparing about 16 GB, roughly 0.5%.
- With RT8's settings, copying back costs less than with the builder off. The GPU is now the biggest part.
- These are single runs, and the spread between runs is large.
- The picture with the builder on: the dark corner of the cell is no brighter yet.

## Roadmap: from about 3.5 fps toward 15–20

**Why it is possible.** The console runs this title at 30 fps on a 12-TFLOPS RDNA2 GPU. An RTX 4080 has about 49 TFLOPS, and the title's CPU code is x86-64 that runs natively. RT10's frame is about 280 ms against the console's 33 ms: the gap is our translation, not the hardware. shadPS4 (AMD GPU, unified memory, raw command buffers) shows that this class of runtime can reach playable speeds.

**Where RT10's frame goes, and the levers:**

| part | now | levers | possible gain |
|---|---|---|---|
| CPU waits (copy-back, flights blocking the next submission) | ~80–150 ms | precise write sets (a fill of upload memory waits 77 ms a frame on a declared range), lazy copy-back | ×1.3–1.5 soon |
| CPU preparing (refresh, descriptors, ~53,000 list calls a frame) | ~150 ms | GPU-written dirty bitmaps, fewer write-watch checks, recording spread over threads | ×2–3 |
| GPU work (card 68% busy, about 190 ms) | ~190 ms | the shader clean-up passes made safe (`VICTORY_SHADER_OPT=0` today), wave64 without barriers (tasks 8–9), ray queries on RT cores (task 7), less indirection for raw pointers | ×3–5 over time, ×1.5–2 per project |
| Series S profile | | works already | ×1.3–1.5 |

**Expectations.**
- Near term: 5–7 fps.
- 15–20 fps needs both the CPU and the GPU 3–5× faster. The GPU is the biggest part: about 190 ms where the console, four times weaker, takes 33. So the next step is a per-pass GPU profile (`VICTORY_GPU_TIMING=1 VICTORY_D3D12_DRED=1`), to find whether one or two programs dominate.

## Where to help (say on Discord which one you take, to avoid doubling up)

1. **Measure this branch on your card,** builder on and off, two runs each. Post the `fps` line and the `shader writes looked for on the CPU` line from the log. Numbers only: no captures or game data on GitHub. AMD and other 64-lane cards are wanted most, since the builder runs there in its native form.
2. **Correctness of the structure.** With the builder on, does traversal walk far more memory than one room needs (see fork #284's follow-up: 9 GB and more of copies asked for)? A host-side checker that walks the built nodes and counts invalid children, out-of-range boxes and depth would tell. It should log counts only.
3. **The crash without quarantine (fork #16, f16 box bounds).** Does this branch still crash on your card? If it does, which builder stage (`VICTORY_BVH_SKIP` bisect)?
4. **Is the dark area lit?** Compare builder on and off at the same scene. Which pass reads the traversal result: the RT shadow/GI denoiser, or reflections? `VICTORY_CAPTURE_WORLD_ALSO` dumps the per-frame ops list.
5. **#814 against #346.** #814 (staronthehorizon) has a GPU dirty-page bitmask plus Series S live RT. Who can run both on the same card and scene?
6. **Page pool in place: tried, gains little alone.** Fork #772 (`VICTORY_INPLACE=3 VICTORY_INPLACE_POOL=1`) put 18,870 windows (5.8 GB) in place and held in a run, but the frame barely changed (690–865 ms with GPU timing on). About 2,700 windows stay refused: page-pool views whose pages the title maps in scattered order, which would need tile mappings (those remove the device on NVIDIA). `VICTORY_INPLACE_POOL=1` also commits 6.1 GB up front, so it needs a big pagefile. Being tested now: the page directory pointing straight at the opened sections (`VICTORY_MEMORY_ARENAS=1`), so scattered pages need no tile mapping.

### Bigger projects (research leads, 2026-11-25)

7. **Hardware ray tracing for the traversal: the biggest GPU win, never done publicly.**
   - The title's traversal programs (about 64–70) are loops around `image_bvh_intersect_ray` with the same shape as a DXR `RayQuery`.
   - Recognise them in the translator and emit `RayQuery`/`TraceRayInline`, so the RT cores do the work. Build native BLAS/TLAS from the title's own build inputs instead of running its 52-program builder.
   - References:
     - AMD GPURT (MIT, https://github.com/GPUOpen-Drivers/gpurt) for exact semantics;
     - Mesa RADV `radv_nir_rt_common.c` for the node formats.
   - **A public HLSL implementation of the console's traversal exists, but it is "all rights reserved": read it if you must, never copy any of it.**
8. **Wave64 as two lanes per thread.** NVIDIA thread i runs console lanes i and i+32, as RDNA's own dual issue does.
   - No barriers; groups of several waves work.
   - Compute and mesh only; occupancy drops.
   - It could make the 19 compute programs that need 64-lane hardware run.
9. **Wave64: which operations really cross lane 32.** On RDNA2, `ds_swizzle`, DPP16/DPP8 and `permlane16`/`x16` stay inside a 32-lane half. Only these cross:
   - `readlane`/`writelane` with lane ≥ 32;
   - `ds_permute`/`ds_bpermute`;
   - 64-bit exec/VCC;
   - `mbcnt_hi`;
   - lockstep LDS hand-offs.

   Some "approximate" forms may be exact for free. dxil-spirv's lane-index heuristics (`dxil_waveops.cpp`) could also reclassify wave64 pixel programs as size-agnostic.
10. **Lazy copy-back.** Protect what the GPU wrote and copy it back only when the title's CPU touches it. Prior art:
    - Ryujinx `BufferModifiedRangeList`;
    - shadPS4 "precise" readback;
    - GMAC (ASPLOS 2010);
    - adaptive protection from gHyvi (USENIX ATC 2015).
11. **A dirty bitmap written by the GPU** (shadPS4 DMA): set with `InterlockedOr`, because shadPS4 found that a plain store loses bits on NVIDIA. Add a ring of the region IDs that changed, so the host reads only what changed.

The switches above came partly from a review by Claude Fable.
