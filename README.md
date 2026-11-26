# VIcky

**VIcky** is an open research fork that works on running the Xbox Series build of the title on Windows PCs with NVIDIA graphics cards, from game files you supply yourself. It builds on **[VIctory](https://github.com/Marvisak/VIctory)** (Marvisak and contributors) and, for its rendering base, on **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**. All three are GPL-3.0.

**This repository contains no game files, captures or assets.** You need your own copy of the game.

## Use `main`

`main` is the only work branch. It is fxceless/victory-fork (`8dd30ab`, merged with fork main up to `465b94f`) plus VIcky's heap map and the ray-tracing work that used to live on the `rt` branch. Measured on an RTX 4080 (16 GB), Ryzen 9 9900X, 32 GB, Series X profile, the title's ray tracing on:

| | |
|---|---|
| Frame rate | **about 6.2 fps** with the settings in [VICKY-RT.md](VICKY-RT.md) ("Fastest measured"): three runs at 6.00, 6.37 and 6.18 fps (host `fps` lines, long frames included) |
| Picture | prologue lit and textured, ray-traced light on, the bottom mosaic strip fixed |
| Known problems | a dark lower-right quarter in some rooms; stair-step edges on some shadows (a fix is being tested); Series S mode (`--console series-s`) renders 960×540 upscaled and looks grainy |
| Memory | entering the world takes up to about 30 GB of commit; on a 32 GB PC, give Windows a large page file |

[VICKY-RT.md](VICKY-RT.md) lists every switch, the measurements behind them and what was tried and dropped. [VICKY.md](VICKY.md) documents the heap-map base.

## What VIcky adds and found

- **A heap map for NVIDIA's 1,000,000-view cap.** The title's view heap has 1,245,184 slots. The map gives the live views past the cap host slots instead of a null view.
- **Per-program wave64 forms** (`VICTORY_WAVE_DEEP_SKIPS_NOT/_ONLY`), offered upstream as [fxceless/victory-fork#779](https://github.com/fxceless/victory-fork/pull/779).
- **Ray tracing:** the structure builder runs on 32-lane cards (fork PRs merged here, see VICKY-RT.md), and its cost was cut: the builder's self-copies are skipped (`VICTORY_BVH_COPY=skip`), and shader writes are written back as joined runs (`VICTORY_BACK_JOIN`).
- **The submit thread's walls**, all opt-in: the heap sync and the heap's buffer pass on threads of their own, an early frame fence, dense pages patched on the card, cheaper guards and sparse reads. Together they took the frame from about 2.5 fps to about 6.2 fps with ray tracing on.
- **Dispatches in parts** (myokex's fix, ported): full-screen compute passes larger than 65,535 groups run whole, which removed the bottom mosaic strip.
- **Findings reported upstream:** the tiled cascade-shadow programs in one wave64 form ([#62](https://github.com/fxceless/victory-fork/issues/62)); a shader loop limit of 32 smears hair; the shader clean-up passes drop geometry (`VICTORY_SHADER_OPT=0`).

## Build and run

```powershell
cargo build --profile diag -p victory --features diagnostics
target\diag\victory.exe run "<your game folder>\<the game's exe>" --execute
```

- Turn ray tracing on with `VICTORY_BVH_BUILDER=1`, and add the settings from VICKY-RT.md.
- The first run translates every shader and is slow and glitchy until the pipelines are built. A new build's first run is also slower while the driver compiles in the background: measure from the second run.

## Branches

- `main`: everything current.
- `fx-heapmap`: kept while pull requests against it are open.
- Older branches are kept as tags (`archive/...`), nothing was lost.

## Credits and history

- VIctory's own README is kept as [README.victory.md](README.victory.md), fxceless/victory-fork's as [README.fork.md](README.fork.md).
- Fixes from contributors and other forks are credited in VICKY-RT.md and VICKY.md, and in the commits.
- Work here is AI-assisted (Claude Code). Contributions are welcome through pull requests to `main`.
