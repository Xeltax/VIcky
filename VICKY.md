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

Tests: `victory-gpu` 128 passed (117 on the base), `victory-shader` 200 passed (198 on the base).

## Measured on an RTX 4080 (driver 621.01, Windows 11, Series X profile)

| tree | untranslated programs | draws per frame (max) | device removal |
|---|---|---|---|
| fork `578d459` | 17 | 5,272 | none in 23 min |
| fork `7b0f736` | 9 | 6,728 | none in 6 min |
| `7b0f736` + heap map | 9 | 6,454 | none in 5 min |

With the heap map, more textures appear in the world (seen on screen, one run so far). A matched A/B (`VICTORY_HEAP_MAP=0`) is in progress.

## Known problems (seen here, one machine)

- **Many walls are invisible.** The heap map does not fix it, so the cause is elsewhere: GPU culling, vertex data, or one mesh pipeline the driver refuses (`f5e50eb2108c425b`). Next on our list.
- **Hair regression:** hair is clean with `578d459` and smeared with `7b0f736`, with or without the heap map. It probably comes from the base's wave64 changes after `578d459` (`6a2d9f9`, `e9765a7`). Not bisected yet.
- 2–3 fps; ~360 draws per frame are dropped because the base's ray-tracing structure builder is off by default.

## Build and run

```powershell
cargo build --profile diag -p victory --features diagnostics
$env:VICTORY_SHADER_LOOP_LIMIT = '32'   # what the base's testers use on NVIDIA
target\diag\victory.exe run <game dir>\<the title's exe> --execute
```

You need your own copy of the game; no game files are in this repository. The first run translates every shader and is slow and glitchy until the pipelines are built.

The older VIcky research (the integration-tree base, the #1269 measurements, the diagnostics) stays on `main`.
