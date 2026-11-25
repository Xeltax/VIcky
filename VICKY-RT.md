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

**VIcky's own commit:** `VICTORY_SNAPSHOT_POOL_MB=<n>` (opt-in) keeps snapshot buffers by size for reuse, so they don't page-fault again on every list.

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

- Without #346, the CPU found 85 MB of shader writes a frame by comparing about 16 GB, roughly 0.5%.
- On a 5080, #346 measured 529–577 ms with the builder on. Our 4080 runs with #346 are in progress.
- The picture with the builder on: the dark corner of the cell is no brighter yet.

## Where to help (say on Discord which one you take, to avoid doubling up)

1. **Measure this branch on your card,** builder on and off, two runs each. Post the `fps` line and the `shader writes looked for on the CPU` line from the log. Numbers only: no captures or game data on GitHub. AMD and other 64-lane cards are wanted most, since the builder runs there in its native form.
2. **Correctness of the structure.** With the builder on, does traversal walk far more memory than one room needs (see fork #284's follow-up: 9 GB and more of copies asked for)? A host-side checker that walks the built nodes and counts invalid children, out-of-range boxes and depth would tell. It should log counts only.
3. **The crash without quarantine (fork #16, f16 box bounds).** Does this branch still crash on your card? If it does, which builder stage (`VICTORY_BVH_SKIP` bisect)?
4. **Is the dark area lit?** Compare builder on and off at the same scene. Which pass reads the traversal result: the RT shadow/GI denoiser, or reflections? `VICTORY_CAPTURE_WORLD_ALSO` dumps the per-frame ops list.
5. **#814 against #346.** #814 (staronthehorizon) has a GPU dirty-page bitmask plus Series S live RT. Who can run both on the same card and scene?
6. **Page pool in place** (fork #772, `VICTORY_INPLACE_POOL=1`). With the builder on, about 3,700 windows are page-pool views that `INPLACE=1` refuses. Porting #772's pool part onto this branch could remove most of the remaining copies.

## Coming next (being tested locally)

All are opt-in switches:
- `VICTORY_MEMORY_REFRESH=touched`: shaders note which copies they read, and only those are refreshed.
- `VICTORY_MEMORY_NAMED_WRITES=0`: copies named only by a constant and never marked stop being snapshotted.
- `VICTORY_REFRESH_STAMP=1`: no second check of a copy under the same write-watch asking.

Ideas from a review by Claude Fable.
