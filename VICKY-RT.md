# VIcky `rt`: ray-tracing work branch

This branch is for making the title's compute ray tracing (the BVH structure builder plus traversal) usable on PC.

- **Base:** `fx-heapmap-8d` (fxceless/victory-fork `8dd30ab` + VIcky's heap map + fork PR #188, Series S), **merged with fork main up to `465b94f`** (2026-11-25 evening: its own submit thread, copy queue, draw windows, `VICTORY_JOIN_PARTS`, `VICTORY_BACK_OWN_VIEW`, `VICTORY_INDIRECT_GATHER`, and its descriptor fix: 1 untranslated program left in our scene instead of 5).
- **Status:** research. It builds, and the gpu, shader, hle and loader suites and `cargo xtask policy` pass, except one test that fails on fork main too (`indirect::tests::warp_budget_saturates_without_wraparound_and_preserves_draw_fields`, always when run alone). Game measurements are below.

## What is merged on top of `fx-heapmap-8d`

| fork PR | author | what |
|---|---|---|
| #36 | adriaespii | the builder translates on 32-lane GPUs (`lane_uniform`) |
| #646 | mynzjerko | DispatchX flag `0x20` counts threads (fixes rehydration writes) |
| #579 | Revens2 | `VICTORY_BVH_SKIP=<indices>` drops builder stages (diagnostics) |
| #213 | Pablo-Barros | an absent BVH child is a miss |
| #346 | Gargalindis | stores through title memory by address mark their copy; only marked copies are snapshotted and compared |
| #1032 | UwUCammie | with `VICTORY_COPY_PLAIN=1`, shader writes go back into title memory in place, guarded, instead of by kernel call (`VICTORY_WRITE_PLAIN=0` turns it off) |

**VIcky's own commits.** All are opt-in.
- `VICTORY_BACK_JOIN=<bytes>` writes nearby runs of shader writes back as one kernel call. The builder's output gave about 850,000 runs a frame, one `NtWriteVirtualMemory` each.
- `VICTORY_MEMORY_REFRESH=touched`: shaders note which copies they read (no extra store), and only those are refreshed, plus every copy once in 8 frames.
- `VICTORY_MEMORY_NAMED_WRITES=0`: a copy that is only named by a constant and never marked stops being snapshotted.
- `VICTORY_SNAPSHOT_POOL_MB=<n>` keeps snapshot buffers by size for reuse.
- `VICTORY_REFRESH_STAMP=1`: measured 0 skips; useless, will go.
- `VICTORY_NOTES_ON_CARD=1`: the notes of programs reading title memory by address stay in the card's memory and are copied back once a frame (not measured in a run yet).
- `VICTORY_BVH_COPY=skip`: the builder's copies of a structure onto itself are not recorded (RT32, below). `native` also runs a pass of ours for any other copy, and `check` compares that pass with the program (diagnostics).

**VIcky PRs merged:** #23 (g0shtj0sh): main's opt-in `VICTORY_NO_WATCHDOG=1` hook (upstream #454, fork #364), for slower PCs whose runs the title's watchdog kills at the world's entry. Nothing changes when it is unset.

**Recommended with the builder on:** `VICTORY_BVH_COPY=skip VICTORY_BACK_JOIN=256 VICTORY_MEMORY_REFRESH=touched VICTORY_MEMORY_NAMED_WRITES=0 VICTORY_SNAPSHOT_POOL_MB=2048`, plus the fork's: `VICTORY_ASYNC_SUBMIT=1 VICTORY_QUEUE2=1 VICTORY_INDIRECT_WINDOW=1 VICTORY_TEXTURE_NO_WAIT=1 VICTORY_COPY_PLAIN=1 VICTORY_JOIN_PARTS=1 VICTORY_BACK_OWN_VIEW=1 VICTORY_INDIRECT_GATHER=1 VICTORY_CARD_REHASH_RUNS=1`. Keep the capture harness rare (`VICTORY_CAPTURE_WORLD_EVERY=64`): a saved frame costs about 235 ms.

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
- The picture is in the next section.

### Task 4 on the RTX 5090: is the dark area lit, and which pass reads the traversal result (`493cc0d`)

**Is it lit?** In 4 of 5 runs with the builder on, in 0 of 5 with it off. Ten runs of 480–600 s from the same save, alternating, settings as above. Measured in the last picture of each run (the shot at the cell window) over the left wall, the left fifth of the frame: mean of the 8-bit R / G / B, and the share of pixels whose blue is not zero.

| | runs | left wall, mean R / G / B | pixels with blue |
|---|---|---|---|
| builder off | 5 of 5 | 19–29 / 3.6–7.9 / 1.0–2.3 | 46–53% |
| builder on, lit | 4 of 5 | 25–30 / 13.6–16.3 / 10.0–10.6 | 97–98% |
| builder on, not lit | 1 of 5 | 21 / 3.9 / 1.3 | 50% |

- Lit means the wall's block pattern and the poster on the right wall show; not lit is the red-only near-black wall of the builder-off runs.
- A run is one or the other from the first frame of the shot to the last (two builder-on runs sampled every 112 frames), so it is decided before the shot, not frame by frame.
- In the run that stayed dark the builder ran too (`the ray-tracing structure builder runs`, never turned off), and what the trace programs wrote looks much like the lit run's: the same 12 programs, and the same share of non-zero pixels in their targets except one target of the second reflection trace (48–78% non-zero in the dark run, 99.6% in the lit one; two frames each). What decides it is not found yet.

**Which passes read it.** From the ops lists of whole frames (`VICTORY_CAPTURE_WORLD_ALSO`), following what the programs with a node test write. 64 programs of the title hold `image_bvh*_intersect_ray`; 12 or 13 of them run in a frame of this scene, with the builder on or off. Five chains:

| what | traced by | read by |
|---|---|---|
| sun shadows | `CS_RTDistantShadowTrace32_Tiled` (+ LeakPlug, TemporalUpsample4X) | the 17 `CS_ShadowRenderTiled*` programs |
| irradiance cache | `CS_IrradianceCache_GatherVBuffer_WithFar`, `CS_IrradianceCache_ShadowTrace_WithFar` | `CS_IrradianceCache_Shade`, `_ShadeVerify`, `_IntegrateIrradianceSH`, `_SpatialFilter_Separable`, `CS_LightMixer_SampleIrradianceCache` |
| diffuse GI | `CS_Primary_GatherVBuffer_WithFar`, `CS_ScreenProbes_TraceRays_*`, `CS_GatherAlphaVBuffer_WithFar` | `CS_Primary_Shade`, `CS_Primary_SpatialFilter`, `CS_ScreenProbes_Inject`, `CS_ShadeAlpha` |
| reflections | `CS_RaytraceVBuffer_RayBinning_ShaderBinning_Mixed_WithFar` (2 dispatches), `CS_RaytraceShadows_Mixed` | `CS_GenerateShaderBins_Tiled`, `CS_RaytracingResolve` (146 material variants), `CS_RayMiss`, then `sr_denoising` (`CS_TemporalResolve*`, `CS_SpatialFilter*`) |
| local lights' shadow mask | `CS_TraceShadowRays_Checkerboard` | `CS_ShadowMaskPrunePrescalarizedLightMask`, then the `TL_PS_*` tiled lighting pixel programs |

So the answer to "denoiser or reflections" is both and more: the tiled sun shadows, the irradiance cache and the diffuse GI read it as well.

**Seen on the way.**
- The first of the two `CS_RaytraceVBuffer_*` dispatches of a frame writes four 1280×720 targets that are all zero, in all 7 frames looked at whose list holds the trace programs, builder on and off. The second one's targets hold values. The irradiance cache's gather target is 0.1–0.2% non-zero in all of them.
- 5 of the 6 `CS_TraceShadowRays_*` variants do not translate (`Full`, `FullAdaptive`, `Half`, `Quarter`, `Eighth`: "a descriptor read only as data"). Only `Checkerboard` ran in this scene; a scene that picks another variant gets no shadow mask.
- `frame-<n>-ops.txt` is cut short on some frames: 4–9 MB with a handful of draws and no trace program in it, where a whole frame is 14–22 MB. In the two builder-off runs every frame asked for from the 300th world frame on was (10 of 10). Check the size before comparing two frames.
- The 64 programs were listed with a local change to `translate --list` that prints each container's count of node tests; it can go into a pull request if wanted.

**Why one builder-on run stayed dark: not found; three causes ruled out** (`1a7eb50`, which differs from `493cc0d` in diagnostics only, plus local test switches that are not in a pull request).

- It is rarer than 1 in 5. Six more unchanged builder-on runs of 340 s, with pictures only (no target dumps), were all lit: 10 of 11 unchanged builder-on runs so far. The five changed runs below were lit too.
- Not the dynamic descriptor slots running out. All six dark runs log `no free dynamic descriptor slots` and none of the first four lit ones did, but refusing tables on purpose leaves the wall lit: every table of 48 slots or more refused in world frames 35–100 (2,800–5,300 of about 10,000 a frame), and every table refused in world frames 40–48. Two later lit runs ran out by themselves as well.
- Not the title's water path. The dark run bound the `ShallowWaterSim` programs and the lit runs with dumps did not, but 4 of the 6 pictures-only runs bound them and were lit. Which path the title takes moves with how long the dumps hold a frame.
- Not the 1,023 copies of title memory by address. Every builder-on run uses all 1,023 numbers by the second world frame (`no number left for another copy; its pages read zeros`). With the limit lowered to 512 (two runs) and to 256 (one run) the wall is still lit, though a page then went without a number more than 500,000 times in 340 s.
- The builder is only needed at the start for this wall: in the run that refused every table in world frames 40–48 the builder turned itself off there for the rest of the run (`one of its dispatches could not be bound`), and the wall was lit 580 frames later. With the builder off from the start it is dark in 5 of 5.
- What is left: the dark run differs from the lit ones before world frame 128 (the first picture of the shot), and its log has nothing the 15 lit runs lack except the points above. Catching it again takes about 15 runs.

- Without #346, the CPU found 85 MB of shader writes a frame by comparing about 16 GB, roughly 0.5%.
- With RT8's settings, copying back costs less than with the builder off. The GPU is now the biggest part.
- These are single runs, and the spread between runs is large.
- The picture with the builder on: the dark corner of the cell is lit on the 4080 too. Measured as in task 4 (mean blue of the left fifth of the last frame of the window shot; on the RTX 5090 about 10 when lit, 1–2 when dark): RT8 43–45, RT10 30, RT28 (no input) 29, the fork main merge (RT29b) 27–30.

### What we tried after RT10 (RTX 4080; most worth knowing so nobody repeats them)

GPU profiles come from `VICTORY_GPU_TIMING=1 VICTORY_D3D12_DRED=1`.
- **Per-command times include the next command's bind work:** there is now a `bind (work recorded while binding the next command)` line, worth 14–24 ms a frame.
- **The biggest builder stage is index 8, `CopyForBVHUpdate`:** a grid-stride copy of each BVH before its refit, about 57 a frame. Skipping it (`VICTORY_BVH_SKIP=8`, diagnostics only) took the frame's GPU time from 157–209 ms to 112 ms. It is bandwidth-bound.
  - The builder's bitonic sort runs only in full builds, about 8 a frame, and costs a few ms.
  - All 52 builder programs were named from the PDB path each container embeds.

| tried | result |
|---|---|
| Spreading the notes, marks and touches over 32 dwords instead of one | no change |
| Windows in VRAM (`VICTORY_CARD_WINDOWS_MB=4096`) | the BVH copy halved (23–27 ms); fps unchanged, because every window, CPU-written or not, went up |
| Widening `CopyForBVHUpdate` to 2,048 groups (exact) | 3× slower: it is bandwidth, not thread count |
| `VICTORY_ROOT_UAV_MB=8` + `VICTORY_NAMED_WRITE_MB=8` + page marks (RT18) | 4.5–5.0 fps in the gameplay scene, **but wrong colours** on a character: caps lose writes |
| Page marks for raw pointers and root views, no caps (RT23) | slower (2.0–2.9 fps): more settles |
| Never copying back what shaders write to windows in VRAM (an incorrect upper bound, RT22) | no gain: the card is now about 75% busy, so the GPU is the limit |

**Where that leaves us:** RT10's setup stays the best: 3–3.5 fps with the builder on, correct picture. The GPU is the limit. Next on our side:
- RT windows (builder and memory by address) in VRAM, with a budget of their own;
- the "descriptor read as data" translation gap. It keeps 5 of the 6 `CS_TraceShadowRays_*` programs from running (task 4);
- stage 0 of hardware ray tracing (an inventory of the traversal programs).

The page-mark switches (`VICTORY_MEMORY_PAGE_MARKS`, `VICTORY_NAMED_WRITE_MB`) are on this branch for experiments; don't turn them on for normal runs.

### RT31–RT32: the builder's copy writes each structure onto itself, so it is skipped

- **RT31** (`VICTORY_BVH_COPY=check`): in every `CopyForBVHUpdate` dispatch the source and the destination, two root parameters, held one title address, and constant 1 was unset. The title refits its structures in place, so the program reads each 16 bytes and writes them back where they were.
- **`VICTORY_BVH_COPY=skip`** records nothing for such a dispatch, about 63 a frame. That also removes the notes and page marks that made the host copy those pages back.
- Same window shot as RT28 (no input, RT10's switches, this branch's base):

| run | ms/frame | fps | GPU | copying back | draws |
|---|---|---|---|---|---|
| RT28, baseline | 353–418 | 2.4–2.8 | 166–204 | 228–287 | ~5,060 |
| RT32, skip (two agents building during the run) | 279–389 | 2.6–3.6 | 53–82 | 132–193 | ~5,000 |
| RT32b, skip | **210–273** | **3.7–4.8** | 41–79 | 127–158 | 4,750–4,930 |

- The picture is as in RT28: lit, same colours, the left fifth of the window shot at 27–28 mean blue.
- **The GPU is no longer the limit.** What is left is on the CPU: "preparing" (134–164 ms, mostly checking which title memory the CPU wrote) and the copy-back.

### RT33–RT35: on fork main `465b94f`, the frame is CPU-bound

Same window shot, builder on, `VICTORY_BVH_COPY=skip` and the recommended switches above:

| run | ms/frame (usual) | submit thread at work | copy-back | card idle | notes |
|---|---|---|---|---|---|
| RT33 | 260–265 | 232–240 | 57–59 | 148–152 | without #1032 |
| RT34, `VICTORY_HEAP_SYNC=0` | 177–185 | 174–182 | 65–68 | 65–73 | **wrong picture**: stale, smeared textures |
| RT35 / RT35b, with #1032 | **228–234 / 231–239** | 202–217 | 49–53 | 122–131 | picture as RT33 |

- The card is idle more than half the frame: the submit thread is the limit.
- Its biggest item is `sync_heaps`: 96 ms a frame mirroring about 100,000 descriptor records the title copies into its heaps as bytes, and the heap-buffer passes after it (about 200 ms on their thread). Turning the sync off is 30% faster but loses textures, so it must get cheaper, not go.
- Next: cheaper heap sync (a cache of made descriptors, pages in parallel), skipping refreshes of memory nothing wrote, and fewer settles (`VICTORY_MERGE_RUNS`).

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
