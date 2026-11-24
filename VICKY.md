# VIcky: branch `fx-heapmap`

This branch is the base that renders the world **lit and textured** on NVIDIA: [fxceless/victory-gta6](https://github.com/fxceless/victory-gta6) at `7b0f736`, plus VIcky's **heap map** on top.

**All of the rendering work below the three `VIcky:` commits is fxceless's, under GPL-3.0.** Their README is kept unchanged and describes it. Please report rendering issues of the base to them.

## What this branch adds: a heap map for NVIDIA's view-heap cap

NVIDIA takes at most **1,000,000** shader-visible views; the title's view heap has **1,245,184** slots. On a capped host the base keeps the first ~737,855 title slots at their own index and lets every view past them read the null slot.

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

Tests: `victory-gpu` 129 passed (117 on the base), `victory-shader` 206 passed (198 on the base).

## Measured on an RTX 4080 (driver 621.01, Windows 11, Series X profile)

| tree | untranslated programs | draws per frame (max) | device removal |
|---|---|---|---|
| fork `578d459` | 17 | 5,272 | none in 23 min |
| fork `7b0f736` | 9 | 6,728 | none in 6 min |
| `7b0f736` + heap map | 9 | 6,454 | none in 5 min |
| `7b0f736` + heap map, no loop limit, the 22 programs out of the deep-skips form | 12 | 7,182 | none in 7 min |

**Best settings so far (RTX 4080):** no `VICTORY_SHADER_LOOP_LIMIT`, and `VICTORY_WAVE_DEEP_SKIPS_NOT=` the 22 programs below. The cell scene then has its ceiling, walls, the uniform's colour and clean hair, at about 2.5 fps.
- `VICTORY_SHADER_LOOP_LIMIT=32` (which some testers use against hangs) **breaks the hair** (smeared strands) here; without it the hair is right (also reported for an RTX 2080 on the base's tracker).
- With the 22 programs in the deep-skips form (the base's default) the cell's ceiling is missing (sky shows through) and much of the scene is dark.
- Which of the 22 is responsible is being bisected.

```
VICTORY_WAVE_DEEP_SKIPS_NOT=922e1d0ee1e19a85,d7db57731ada919b,ab5455fc6abb804f,be6ce5d849d51127,191ecaed66eea3b1,31d703207b51549a,56f3bf2d9388e5a7,804b342cb4b9d1e9,abd7a22106898c66,1d44a723bdebbf27,b4b0427232707598,240cb516c1af0bc3,466ec41d3f9d24c0,65de113e23fa4fdd,90fe2f65e9e589b5,c1eee35dbc50a9ef,41aa4d3d51094531,ff8fe948b132510f,da903250b26c2b55,d2b169b82f67ec5b,578751e8c1661b17,1f1fbd86759b3da2
```

The heap map: in this base few views past the window are actually read (its pool stays under 65,536 slots), so its effect here is small and not yet shown by a matched A/B.

## Known problems (seen here, one machine)

- With the base's defaults, missing ceilings/walls and dark surfaces: see the best settings above.
- One mesh pipeline is refused by the driver (`f5e50eb2108c425b` + pixel `ce3f0432a08f1d35`, E_INVALIDARG).
- 2–3 fps; ~360 draws per frame are dropped because the base's ray-tracing structure builder is off by default.

## Build and run

```powershell
cargo build --profile diag -p victory --features diagnostics
$env:VICTORY_WAVE_DEEP_SKIPS_NOT = '<the 22 hashes above>'   # no loop limit
target\diag\victory.exe run <game dir>\<the title's exe> --execute
```

You need your own copy of the game; no game files are in this repository. The first run translates every shader and is slow and glitchy until the pipelines are built.

The older VIcky research (the integration-tree base, the #1269 measurements, the diagnostics) stays on `main`.
