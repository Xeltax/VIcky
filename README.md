# VIcky

**VIcky** is an open research fork that works on running the Xbox Series build of the title on Windows PCs with NVIDIA graphics cards, from game files you supply yourself. It builds on **[VIctory](https://github.com/Marvisak/VIctory)** (Marvisak and contributors) and, for its current rendering base, on **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**. All three are GPL-3.0.

**This repository contains no game files, captures or assets.** You need your own copy of the game.

## Use branch [`fx-heapmap-8d`](https://github.com/Xeltax/VIcky/tree/fx-heapmap-8d)

It is fxceless/victory-fork at `8dd30ab` plus VIcky's commits. Measured on an RTX 4080 (16 GB, driver 621.01), 9-minute runs from the same save:

| | |
|---|---|
| Prologue | lit and textured: cutscene, cell, yard; the cell's walls and ceiling drawn in 2 of 2 runs |
| Frame rate | 3.5–4.5 fps (Series X profile); **5.7–6.0 fps in Series S mode** (`--console series-s`) |
| Stability | no device removal in any of the runs |
| Known problems | areas lit only by the title's ray-traced light are dark; a mosaic strip at the bottom of some frames |

Settings, switches and the measurements behind them are in that branch's [VICKY.md](https://github.com/Xeltax/VIcky/blob/fx-heapmap-8d/VICKY.md).

## What VIcky adds and found

- **A heap map for NVIDIA's 1,000,000-view cap.** The title's view heap has 1,245,184 slots, and about 335,000 of its live views lie past what a capped heap mirrors. The map gives them host slots instead of a null view.
- **Per-program wave64 forms.** `VICTORY_WAVE_DEEP_SKIPS_NOT/_ONLY` chooses one translation form per program. It is offered upstream as [fxceless/victory-fork#779](https://github.com/fxceless/victory-fork/pull/779).
- **Findings reported upstream:**
  - The 14 tiled cascade-shadow programs are wrong in one wave64 form ([#62](https://github.com/fxceless/victory-fork/issues/62)).
  - A shader loop limit of 32 smears hair.
  - The shader clean-up passes drop geometry; `VICTORY_SHADER_OPT=0` brings it back. Confirmed on NVIDIA and on AMD by UNYTIX.
  - Test reports on Series S presentation ([#188](https://github.com/fxceless/victory-fork/pull/188)).
  - An earlier measurement for XCL ([xcl #1269](https://github.com/xcl-dev/xcl/issues/1269)).
- **Contributors' fixes merged here:**
  - UNYTIX: blending on integer targets, so a refused mesh pipeline is now built; a clean window close.
  - coldek: full-heap probe, descriptor rollback, null CBV.

## Build and run

```powershell
git checkout fx-heapmap-8d
cargo build --profile diag -p victory --features diagnostics
target\diag\victory.exe run "<your game folder>\<the game's exe>" --execute            # Series X profile
target\diag\victory.exe run "<your game folder>\<the game's exe>" --execute --console series-s
```

The first run translates every shader and is slow and glitchy until the pipelines are built.

## Credits and history

- VIctory's own README is kept as [README.victory.md](README.victory.md).
- This `main` branch is VIcky's older research tree, based on VIctory's integration branch; [VICKY.md](VICKY.md) documents it.
- Work here is AI-assisted (Claude Code). Contributions are welcome through pull requests to `fx-heapmap-8d`.
