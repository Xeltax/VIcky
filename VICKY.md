# VIcky: branch `fx-heapmap-8d`

This branch is the base that renders the world **lit and textured** on NVIDIA: [fxceless/victory-fork](https://github.com/fxceless/victory-fork) at `8dd30ab`, plus VIcky's commits on top: a **heap map** for NVIDIA's view-heap cap and a **per-program switch** for one wave64 form. (Branches `fx-heapmap-4e` and `fx-heapmap` are the same work on the fork's older `4e17478` and `7b0f736`; the measurements below were made on those.)

**Measured on the RTX 4080 (driver 621.01), 2 runs of 9 min from the same save:** the cell shell (walls, ceiling, floor, the wall around the window) is drawn in **2 of 2** runs, where the previous base (`4e17478`) drew it in about 1 run in 4 (1 of 4 here, and about 1 in 4 over 20 runs on an RTX 4090 in [fxceless/victory-fork#62](https://github.com/fxceless/victory-fork/issues/62)). That is the fork's own `VICTORY_VIEW_GUARD` (on by default), the fix for stale host copies of pages the title writes through another view. Also measured: **3.5–4.5 fps** in the cell (2.2–2.5 before), up to 10,631 draws per frame, **4** programs that don't translate (9–12 before), no device removal. Settings: no loop limit, `VICTORY_SHADER_OPT=0`, the 14 shadow programs in `VICTORY_WAVE_DEEP_SKIPS_NOT` (below); the base's new defaults otherwise. Still there: a mosaic strip at the bottom of some frames, and areas lit only by ray-traced light are dark.

**Series S mode (`--console series-s`) works on this branch** with fxceless/victory-fork PR #188 by sagouindupm merged in (the title's second present plane, and a plain stretch for its upscale dispatch; without it Series S shows a black window after loading). RTX 4080, same save and settings: **5.7–6.0 fps** in the world (Series X: 3.5–4.5), about 4,400 draws per frame, no device removal. Run it as `victory.exe run <game> --execute --console series-s`.

Since `4e17478` the base also clamps view indices on every host (a 2^21-view heap where the host takes one, else the capped 1,000,000), probing the full 2^21 range itself, so VIcky's own full-heap probe was dropped; the heap map is set where that probe finds the cap. Its faster defaults (`c96b4f2`) are on: the stage-2 copy-back thread (`VICTORY_RELAY=0` turns it off), wide copy-back (`VICTORY_WIDE_BACK=0`), card copies from 1 MiB up to 1,024 MB (`VICTORY_CARD_BUFFERS_MIN_MB=16 VICTORY_CARD_BUFFERS_MB=512` restore the old values) and 256 lists in flight (`VICTORY_LATE_WAIT_LISTS=128`). The shader cache is now keyed on the translator's source and switches, not on the executable (`VICTORY_SHADER_CACHE_KEY=exe`: the old key); the heap map is in that key, the per-program lists are not (each program's decision is in its own key). For `victory translate --warm` on a capped host, set `VICTORY_VIEW_LIMIT=999999`: the warm-up then translates with the heap map too.

**All of the rendering work below the `VIcky:` commits is fxceless's, under GPL-3.0.** Their README is kept unchanged and describes it. Please report rendering issues of the base to them (our findings so far: [fxceless/victory-fork#62](https://github.com/fxceless/victory-fork/issues/62)).

## What this branch adds: a heap map for NVIDIA's view-heap cap

The title requests **1,245,184** shader-visible views. On an 8 GB RTX 4060 Laptop with driver 617.14, D3D12 accepted 1,000,001 views and refused the full title heap. The base probes the translator's full 2,097,152-slot range before compiling shaders and uses its **1,000,000**-slot layout if that allocation fails; VIcky's heap map is set there. On a capped host the base keeps the first ~737,855 title slots at their own index and lets every view past them read the null slot.

Measured on an RTX 4080 (diagnostics scan of the title's heap, every 30 s in the world):
- 650k–700k title slots hold a view, up to slot 1,245,182.
- With the usual 2^18 dynamic slots, **about 335,000 live views sit past the mirrored range**, so bindless reads of them come back null.

The heap map:
- keeps title slots `0..131,071` at their own index (tables need that);
- gives every title slot past that which holds a view a slot of a **pool** (606,782 slots), while it holds the view;
- keeps the base's dynamic slots and guard exactly where they were;
- has translated shaders look `ResourceDescriptorHeap` indices past the window up in a map (one word per title slot, `!host`; 0 is the null slot). Tables past the window go through the base's fresh path.

Switches:
- `VICTORY_HEAP_MAP=0`: exactly the base's behaviour.
- `VICTORY_HEAP_WINDOW=<n>`: the identity window (default 131,072).

## A per-program switch for one wave64 form

The base's exact wave64 form "without skips, scalars followed" (`6a2d9f9`) is on for every program or, with `VICTORY_WAVE_DEEP_SKIPS=0`, off for every program. On a 32-lane host, 22 programs end in it (14 cascade-shadow tiled renderers, including the hair ones; 4 visibility-buffer classification/resolve; 2 skinned-attach particle updates; a ray pre-pass; a lightning point selection). On an RTX 4080 keeping those 22 out of the form brings back the cell's ceiling, the walls' and the prison uniform's colours (see below). So it is now chosen per program, by the hash the log names a shader by (`#hash` in `victory translate --list`):
- `VICTORY_WAVE_DEEP_SKIPS_NOT=<hash>,<hash>,...`: off for the listed programs;
- `VICTORY_WAVE_DEEP_SKIPS_ONLY=<hash>,...`: off for every program not listed;
- `VICTORY_WAVE_DEEP_SKIPS=0`: off for all, as before.

A program's decision is in its own shader cache key, so a new list translates again only the programs whose decision it changes. To find the programs offline: `VICTORY_HOST_WAVE=32 VICTORY_WAVE_WHY=1 victory translate --list`; a program in that form prints `WAVE64-RETRY #<hash> without skips, scalars followed: wave Some(32)` last.

Tests: see the end of this file.

## Measured on an RTX 4080 (driver 621.01, Windows 11, Series X profile)

| tree | untranslated programs | draws per frame (max) | device removal |
|---|---|---|---|
| fork `578d459` | 17 | 5,272 | none in 23 min |
| fork `7b0f736` | 9 | 6,728 | none in 6 min |

(The rows below were measured on `7b0f736`; this branch's `4e17478` keeps the same 22 programs in the deep-skips form, checked offline.)

| tree | untranslated programs | draws per frame (max) | device removal |
|---|---|---|---|
| `7b0f736` + heap map | 9 | 6,454 | none in 5 min |
| `7b0f736` + heap map, no loop limit, all 22 programs out of the deep-skips form | 12 | 7,182 | none in 7 min |
| `7b0f736` + heap map, no loop limit, **the 14 cascade-shadow programs** out of the deep-skips form | 12 | ~6,900 | none |

**On `4e17478` (and likely here: the clean-up passes did not change up to `8dd30ab`), also set `VICTORY_SHADER_OPT=0`.** The base's new shader clean-up passes (`1684989`, on by default) leave the room's walls, roof and floor out of the picture on an RTX 4080 (A/B on the same save: with them, sky shows where the walls are; with `VICTORY_SHADER_OPT=0` the walls are there), and on an AMD RX 7800 XT the same happens with them on (UNYTIX's run).

**The clean-up passes audited (branch `fx-rt-shaderopt`):** no pass was found that changes what a program does. Each pass was read against the IR's rules (dead code, folding, reuse, branch facts, reads within a block, the wave64 first-lane exchange), and a new diagnostic, `VICTORY_SHADER_OPT_CHECK=<n>`, runs every translated program before and after the passes side by side on `n` made-up threads and compares what each does to the world (stores, atomics, barriers, outputs, discards): no difference over the whole pack (25,699 programs, 4 threads each), nor over the cell scene's programs with the new default. Three changes all the same, for what one thread cannot show: resource handles stay in their block (fork PR #317's AMD compiler crash), the writer no longer folds two empty latches of one branch into one block (a join with two values from one block; no program of the pack has such a pair yet), and wave operations are no longer dropped or merged by default (new bit 64, `waves`: declared with effects, they may keep a driver from moving LDS accesses across them, which the translation of a console wave's lanes sharing LDS in step may rely on). The new default (63, all but 64) leaves 41.0 % fewer host instructions on the scene's programs (42.3 % before). New switches: names in `VICTORY_SHADER_OPT` (`dead,fold,reuse,facts,loads,first,waves`, `all`, `none`), and `VICTORY_SHADER_OPT_NOT` / `_ONLY=<hash>,...` per program, keyed per program in the shader cache. Caution: the walls-missing A/B was made on `4e17478`, when the shell was missing in about 3 runs of 4 whatever the setting; on this base the passes still need an A/B in the game.

**UNYTIX's fix (VIcky PR #9) is in:** D3D12 refused the mesh pipeline `[mesh f5e50eb2108c425b, pixel ce3f0432a08f1d35]` because the title enables blending on an `R8_UINT` target; blending is now turned off on integer targets. Measured on the RTX 4080 (same save, `VICTORY_SHADER_OPT=0`, the 14 programs out, no loop limit): the pipeline is now made. **Correction:** a first run with the fix showed the walls at the window present and lit, but two exact repeats did not (sky/horizon where the walls are), so that picture was chance, not the fix. Runs of one build and one save differ a lot here; the G-buffer and depth do hold the walls (target dumps), so what varies is a lighting or composition pass (fog/volumetrics or the clustered local lights, both translated in approximate wave64 forms on 32-lane hosts; being tested).

**Best settings so far (RTX 4080): no `VICTORY_SHADER_LOOP_LIMIT`, `VICTORY_SHADER_OPT=0` on this branch, and only the 14 cascade-shadow programs kept out of the deep-skips form.** The prison cell then renders almost normally: painted brick walls, ceiling and lights, floor, posters, the uniform's colour and print, clean hair, the yard through the window. About 2–2.5 fps. Some artifacts remain (dark bands and rectangles on some frames), and areas lit only by ray-traced light stay dark (the base's ray tracing is off).

```
VICTORY_WAVE_DEEP_SKIPS_NOT=922e1d0ee1e19a85,d7db57731ada919b,ab5455fc6abb804f,be6ce5d849d51127,191ecaed66eea3b1,31d703207b51549a,56f3bf2d9388e5a7,804b342cb4b9d1e9,abd7a22106898c66,1d44a723bdebbf27,b4b0427232707598,240cb516c1af0bc3,466ec41d3f9d24c0,65de113e23fa4fdd
```

What each piece does (one run each, same scene):
- `VICTORY_SHADER_LOOP_LIMIT=32` (which some testers use against hangs) **smears the hair**; without it the hair is right (also reported for an RTX 2080 on the base's tracker).
- **The 14 cascade-shadow tiled renderers are wrong in the deep-skips form:** with them in it (the base's default) the cell's ceiling is missing (sky shows through) and the room is dark.
- **The other 8 are better in it:** keeping all 22 out brings the ceiling back but leaves the room much darker than keeping only the 14 out.
- Which of the 14 is responsible is being bisected (5 `CS_ShadowRenderTiledCascades*` vs 9 `CS_ShadowRenderTiledMisc*`).

The heap map: in this base few views past the window are actually read (its pool stays under 65,536 slots), so its effect here is small and not yet shown by a matched A/B.

## Known problems (seen here, one machine)

- With the base's defaults, missing ceilings/walls and dark surfaces: see the best settings above.
- One mesh pipeline was refused by the driver (`f5e50eb2108c425b` + pixel `ce3f0432a08f1d35`, E_INVALIDARG); UNYTIX's integer-blend fix above makes it.
- 2–3 fps; ~360 draws per frame are dropped because the base's ray-tracing structure builder is off by default.

## Build and run

```powershell
cargo build --profile diag -p victory --features diagnostics
$env:VICTORY_WAVE_DEEP_SKIPS_NOT = '<the 22 hashes above>'   # no loop limit
target\diag\victory.exe run <game dir>\<the title's exe> --execute
```

You need your own copy of the game; no game files are in this repository. The first run translates every shader and is slow and glitchy until the pipelines are built.

The older VIcky research (the integration-tree base, the #1269 measurements, the diagnostics) stays on `main`.

## Tests (this branch at `8dd30ab`, RTX 4080 and WARP)

`victory-gpu` (with `diagnostics`) 157 passed, 3 ignored; `victory-shader` 222 passed, 3 ignored; `victory-hle` 99 passed; `cargo xtask policy`: 175 files ok.
