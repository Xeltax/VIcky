# VIcky roadmap

*Updated 2026-11-27. Main is at `3703fca9`.*

**Goal:** a Series X profile with the title's own ray tracing ON (`VICTORY_BVH_BUILDER=1`).
- First step: **10 fps or more in moving scenes** on an RTX 4080-class PC.
- Then: 15–20 fps.

RT-off numbers are not a target here.

**How to take a task:**
1. Pick an item below and say on Discord that you take it, to avoid doubling up.
2. Open one PR per lever, against `main`.

Details on every switch and measurement are in [VICKY-RT.md](VICKY-RT.md).

## Where we are

Measured on an RTX 4080, Ryzen 9 9900X and 32 GB, Series X profile, RT on, with the recommended set in VICKY-RT.md. The figure is the true fps over the last 12 frame-time lines of a 420–480 s run.

| Scene | fps | Notes |
|---|---|---|
| Chapter-1 boat mission (moving) | **9.52** | lines 8.6–10.0; it was 7.34 on the morning of 2026-11-27 |
| Prologue cell | **7.27** | CCD_PLACE ON phases 7.5–7.6 |

**Where the frame goes** (prologue profile, before `SYNC_FINE` and `CCD_PLACE`):

The submitting thread is at work about **137 ms** a frame:

| Part | ms |
|---|---|
| Preparing | 88 |
| — copies | 30 |
| — snapshots | 14 |
| — other | 44 |
| Heap/descriptor sync join | 23 |
| Wait for the card (texture copy-back settle) | 21 |

- **The card is only ~54 % busy.** The wall is the CPU-side sync of the console's unified memory to the PC's split memory.
- **The title's recording threads wait on our locks.** The lock probe (P7) measured, per frame and summed over threads:

| Lock | Takes | Waited | Time waiting |
|---|---|---|---|
| `made` | 7,758 | 982 | 165 ms |
| `BUFFERS` | 289k | 72.8k | 72 ms |
| `kept` | 136k | 10.8k | 8.9 ms |

Only 5–16 % of these takes cross CCDs, so the cost is contention itself, not CCD placement.

## Ruled out (measured or researched; don't redo without a new idea)

**Measured on our tree:**
- **Physical-page mirror** of title memory, with D3D12 tiles mapped at several offsets.
  - The 4080 supports it (probe P0).
  - But only 48.7 % of the descriptor slots per frame would go away (probe P2, threshold 60 %).
- **Copy-queue pull from heaps opened over title memory** (probe P8). It is not cheaper for the typical run, and tiles from such heaps remove the device.
- **Recreate-At-GPUVA**: tier 0 on the 4080.
- **`PARALLEL_MIRROR`**: −0.35 fps.
- **The water set** (`VICTORY_WAVE_PAIRS_ONLY` / `VICTORY_LDS_FLAGS` for the water programs): turquoise water, grey tiles, and green specks in vegetation.

**Researched, not available here:**
- **Hardware dirty tracking or coherence for a GeForce under Windows**:
  - IOMMU dirty bits;
  - GPU page faults;
  - SVM;
  - PML before Zen 6;
  - PKU.

  No user-mode API exposes them.

## Status (2026-11-27 night)

| Item | Switch | Result | Verdict |
|---|---|---|---|
| **A1** | `VICTORY_BUFFERS_STRIPES` | `BUFFERS` waits 72 ms → 0; fps +0.11 | Done. The recording threads' lock waits are off the critical path. |
| **A2** | `VICTORY_MADE_TURNS` | -0.11 fps | Left off |
| **B1** | `VICTORY_HEAP_SYNC_AT_SUBMIT` | Join 25 → 22-24 ms; -0.03 fps | Left off |
| **B2** | `VICTORY_PAGE_PLACE` + `VICTORY_INPLACE_CONSTANTS` | Constants heap in place, but -0.6 fps | Left off: the card's PCIe reads cost more than the copies |

| **Lot 6, sync need** | `VICTORY_HEAP_SYNC_NEED` | +0.25 fps | Optional |
| **Lot 6, pass first** | `VICTORY_MADE_PASS_FIRST` | -0.32 fps; suspect for a dark boat hull and green vegetation when combined with the other two | Left off |
| **Lot 6, dense first** | `VICTORY_DENSE_PAGE_FIRST` | +0.11 fps | Optional |

Lot 6's always-on SSE2 `changed_runs` and per-thread counts: picture clean, 9.41 fps on the boat mission.

| **Picture: faces** | `VICTORY_FACE_D16` | 16-bit buffer path for the face network; engaged in 34 shaders; no fps cost | Recommended |
| **Picture: transmission** | guard fix (on by default) | `2da4beb3` builds | Done |
| **Picture: water** | water set + `VICTORY_WATER_TILE_SLOTS` | Grey tiles gone; water lit turquoise | Look to judge |
| **Picture: posterized sky** | `VICTORY_FILTER_3D` | Diamonds on the water | Left off |
| **Picture: interior mips** | `VICTORY_INTERIOR_MIP_RESIDENCY` | No effect | Left off |

| **Picture: clothing / streamed textures** | `NO_WAIT_FALLBACK` + `DS_HOLD` + `PAGE_WRITES=all` + `BCPACK_FIXES` + `RECHECK_PARTLY` + `HEAP_RECHECK_MIPS` (every 960) | Never-re-uploaded stale slots 82 → 3-15 per sample; shirt clean; no fps cost | Recommended |

| **Structural stage 1** | `VICTORY_TABLE_BOUND` | **+1.66 fps boat (10.48 fps run), +1.18 prologue**; stage-2 criterion passes (64-72 %) | Optional. With `TABLE_BOUND_FRESH` (stage 1b) the picture is clean but the gain shrinks to +0.17 boat / +0.39 prologue: the speed came from skipping the staleness work. Next: stage 2 |

| **Structural stage 2** | `VICTORY_TABLES_LEFT` | **+0.72 fps prologue (-14 % frame time)**, neutral on the boat; 0 check violations; heap sync 33 -> 19-28 ms, pass 49-58 -> 26-28 ms, copies flat; picture clean | Optional. Profiled (wall2): the boat frame is the submitting thread's own work. Since `fx-main-lanes`: holds swap key and copy together (a fix); `TABLES_LEFT_HEAL_SPREAD=1` spreads the heal (opt-in) |

| **Structural stage 3** | `VICTORY_SUBMIT_LANES` | Not measured yet. Texture refreshes on the wide threads, joined before the card's copies; the sweep's last drops on the reaper after the send. Expected: mission stretch S -15..-25 ms, settled -4..-8 | Opt-in. Run `=alt` on the boat and the prologue |

| **Stage 4 counter** | `VICTORY_MADE_READ` line (always on) | Not measured yet. Share of the draws' heap-mirror calls that only look | Stage 4 goes ahead at 70 % or more |

**Since 2026-11-28 the priority is picture fidelity**: textures, faces and mouths, resolution, water and vegetation. After that comes a structural rework of the descriptor path for fps (the submitting thread and the buffer-view pass are two equal walls), instead of more small levers.

Next focus: the submitting thread itself, ~137 ms. In order:

1. B3, the snapshots.
2. C1-C2, the texture copy-back's 21 ms.
3. The 44 ms of "other" preparing work, which still needs a breakdown.

## Tracks

Each item lists its goal, the measure to beat, and where to start.

### A. Lock contention: the title's recording threads

- **A1. `BUFFERS` lock.** Shard it, or make the lookup lock-free. Today: 289k takes a frame, a quarter of them wait, 72 ms waited. Start in `crates/victory-gpu/src/descriptor/` and `draw.rs`; measure with `VICTORY_LOT4_PROBE=1` (the P7 line) and an `=alt` switch.
- **A2. `made` lock.** It is held per heap-sync round: 165 ms waited a frame. Aim for finer pieces, or no hold while a round's chunk is built.
- **A3. Descriptor-slot design from the avatarmanz / cool_fox lineage.**
  - Their draws mark the table slots and the sync leaves them alone: ~10–12k records mirrored a frame, against our ~95k.
  - Our attempt (`TABLES_LEFT`, 2026-11-26) measured slower and was dropped.
  - Someone who knows that design well should port it properly.

### B. The submitting thread

- **B1. Heap sync while the submission waits in line.** Port Belobug `673be0e` + `a609a2f` (`HEAP_SYNC_AT_SUBMIT`). It needs Belobug's round-based sync thread, about 1.2k lines. Target: the 23 ms join.
- **B2. Constants heap in place.**
  - Placing each physical page at its section offset at first mapping would make 100 % of the constants-heap-sized buffers (238 MB) one run (probe P4).
  - Then `VICTORY_INPLACE_CONSTANTS` can place them.
  - Start in `crates/victory-hle/src/modules/xmem.rs` and `inplace.rs`.
- **B3. Title memory as the snapshot.** Only a few pages a frame should need a true before-image; ~200 MB a frame are snapshotted today. First fix probe P3, which sampled nothing in the prologue. Target: 14 ms.
- **B4. `REFRESH_SET`** (Kemerd, trustno1 PR #18). Time the refresh steps `REFRESH_FRONT` already skips before porting it; the expected gain is a few ms.
- **B5. `VICTORY_THIN`** (list calls recorded compactly and translated by several threads; cool_fox lineage). About +1 fps in one A/B on another tree.

### C. Card wait and copy-back

- **C1. `VICTORY_TEXTURE_BACK_ASYNC`** (port of trustno1 `2d1dc382`): +0.39 fps in one noisy run. Confirm it over two runs on the boat mission, then move it into the recommended set or drop it.
- **C2. Lazy copy-back.** Leave what the GPU wrote on the card, and copy it back only when the title's CPU touches it. First fix probe P6, which sampled nothing.

### D. Ray tracing correctness

- **D1. Stale builder input.** With `VICTORY_BVH_INPUT_CHECK=1`, 15 of 89 rehydration dispatches a frame had a range that differed from title memory at submission. Find which copy is late.
- **D2. Parent-link stores.** With `VICTORY_BVH_LINK_STORES=keep`, the boat save no longer crashed on this build. Confirm it over more runs before keeping the stores by default.
- **D3. Hardware RT for traversal.** Emit `RayQuery` for the title's traversal loops and build native BLAS/TLAS from its inputs (VICKY-RT.md, "Bigger projects" 7). This is the biggest GPU lever, for when the CPU stops being the wall. Kemerd announced the same direction.

### E. Picture

| Item | Problem | Lead |
|---|---|---|
| E1 | Water light programs `5dc3b25986622949` and `6f225b641c5b0708` on 32-lane cards | Run turquoise with grey tiles. |
| E2 | Green specks in shore vegetation (`926c46bab3b86bd6` paired, LDS flags `2cc4c197ee29e291`); a few remain on main | |
| E3 | Purple trees after a character passes in front, and cutscene problems (seen on another tree) | Close to Marvisak/VIctory #3279: translucency, skin and blur-transmission classes in the tiled lighting pass |
| E4 | Posterized sky and water, seen in one boat run | Reproduce first. |
| E5 | Faces | d16 table strides fix from the cool_fox lineage; check against our tree |

### F. Presentation (later)

- **F1. DLSS Super Resolution and Frame Generation through Streamline.** Port from foxhound / trustno1 (~3.6–4k lines). NVIDIA advises frame generation from a base of about 60 fps, and an RTX 40 gets 2x only. Do this after the CPU work.

### G. Infrastructure

- **G1. Flaky tests in victory-gpu.** `warp_budget_saturates_without_wraparound_and_preserves_draw_fields` fails about 1 run in 3–5, and the test binary rarely crashes with 0xc0000005. Both also happen on the earlier main.
- **G2. A shared benchmark protocol**, so numbers from different PCs compare:
  - the same two scenes: the prologue cell, and the chapter-1 boat mission save;
  - RT on;
  - true fps over the last 12 lines;
  - `=alt` paired medians for single switches.
- **G3. Memory pressure at world entry** on a new exe's first run. Low-memory kills of the run happen on 32 GB machines.

## Proposed split (2026-11-27, to confirm on Discord)

| Who | Items |
|---|---|
| Azure (Cuebert) | A3 (descriptor-slot design), B5 (`THIN`), E3 and E5 (purple trees, cutscenes, faces) |
| Xeltax | A1, A2, B1–B3, C1–C2, D1–D2 |
| Open to anyone | B4, E1, E2, E4, F1, G1 |
| Together | G2 |

## Rules for PRs

- **One lever per PR**, opt-in behind its own `VICTORY_...` switch. Add an `=alt` mode (20 s on / 20 s off) when it is a speed lever.
- **Measure RT on**: the fps line, plus the paired median from an `=alt` run, on the prologue and on the boat mission save if you have one.
- **Report numbers only** on GitHub: no captures, no game data, no shader dumps.
- **Credit ports by commit SHA.**
- **Before review**: all suites (`victory-gpu`, `victory-shader`, `victory-hle`, `victory-loader`, `victory-formats`) and `cargo xtask policy` must pass.
