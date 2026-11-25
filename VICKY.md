# VIcky: a research fork of VIctory

> **Latest (2026-11-25): use branch [`fx-heapmap-8d`](https://github.com/Xeltax/VIcky/tree/fx-heapmap-8d).** It is [fxceless/victory-fork](https://github.com/fxceless/victory-fork) at `8dd30ab` (all of its rendering work is fxceless's, GPL-3.0) plus VIcky's commits (heap map for NVIDIA's 1,000,000-view cap, per-program wave64 switch, contributors' fixes). On an RTX 4080: a lit, textured prologue with the cell's walls in 2 of 2 runs, 3.5–4.5 fps, no device removal; **Series S mode (`--console series-s`) 5.7–6.0 fps**. Settings and measurements are in that branch's [VICKY.md](https://github.com/Xeltax/VIcky/blob/fx-heapmap-8d/VICKY.md). (`fx-heapmap` and `fx-heapmap-4e` are older bases.)
> **FR —** dernière branche : `fx-heapmap-8d` (fork de fxceless à `8dd30ab`, crédité, + nos commits) : prologue éclairé et texturé, murs de la cellule présents 2 runs sur 2, 3,5–4,5 i/s sur RTX 4080, et 5,7–6 i/s en mode Series S (`--console series-s`).
>
> What follows is the older research tree on `main`, based on VIctory's integration branch.

VIcky is a **research fork of [VIctory](https://github.com/Marvisak/VIctory)**. Its goal is to find out why the retail title's world renders black or flat grey on PC, and to fix it, in the open. It is not an official release. It is not affiliated with VIctory's maintainers, with XCL, or with the title's publisher.

> **FR —** VIcky est un fork de recherche de VIctory. Le jeu se lance, charge le monde et répond aux commandes, mais la scène 3D s'affiche noire ou grise. Ce dépôt rassemble les meilleures branches de la communauté et des outils de diagnostic, pour trouver ensemble quelle passe de rendu casse et pourquoi. Toute aide est la bienvenue (issues, mesures, correctifs).

**Rule #1, as in VIctory: nothing from the game is ever committed.** That covers captures, shader dumps, extracted tables and raw logs. You supply your own copy of the title at run time. See [docs/asset-policy.md](docs/asset-policy.md); `cargo xtask policy` checks it.

## What is in this tree

| Part | Source |
|---|---|
| Base: VIctory plus about 176 open upstream PRs (wave64 stack #439, #493, #483, #206, #566, #453…) | r4tedx [`integration/testers`](https://github.com/r4tedx/VIctory/tree/integration/testers) at `92bc27b` (see [INTEGRATION.md](INTEGRATION.md)) |
| Mirror 256-byte-aligned CBV tables in place, so the dynamic descriptor slots are not used up (VIctory #840) | alessiodeia `1d21e06` |
| Logs which texture each present plane comes from, for listed frames | fuzeofficlal `9a10d7a` |
| Capture frames by run time (`VICTORY_CAPTURE_AT_SECS`), and trace every draw and dispatch of a captured frame with its programs and the reason it was dropped | this fork |
| Guess the stride of bindless structured buffers, from accesses at constant offsets. A record read whole up to a 16-byte multiple keeps that size; otherwise the stride is the smallest one the shader declares that holds every access. `VICTORY_COMBO_GUESS_STRIDE=0` turns it off; `=any` takes the extent as is. It unblocks 41 of 81 failing compute programs, among them the instance culling | this fork (#1) |
| Typeless views on typeless textures take the family's first concrete format | port of Xvania7's VIctory #876 |
| Mesh output bounds, every primitive initialised | integration update 6 (VIctory #587) |
| An empty buffer view, or one that runs past its host copy, becomes a null view instead of removing the device | this fork |
| `VICTORY_DUMP_FAILED_SHADERS=<folder>` keeps the containers of shaders that do not translate, for `victory shader-info --disasm`. They are title data: keep them local | this fork |
| Root arguments the console spills to memory: compute programs whose root layout starts at user SGPR 16 read it through `s[0:1]`. This was the first cause of missing compute programs on NVIDIA here (34 programs → 1). | VIctory #302 + #489 (dadou000, louketa, Brazlucas), with `initialized_sgprs` from `main` |
| Mask dwords at data joins, and `s_bcnt1`'s SCC of a mask | VIctory #878 (permadiego `7ab94ab`), ported by hand |
| Wave64 mesh code numbers its lanes the host's way on 32-lane hosts | Peppacodes `ed9afc0` (VIctory #940), adapted to this tree's wave64 stack (`VICTORY_MESH_CONSOLE_LANES=1` turns it off) |
| `VICTORY_SHOW_GBUFFER=<n>` presents G-buffer target `n` (default 1, the packed normals) instead of the title's black picture | this fork |
| **Bindless slots made and kept up to date** (`ResourceDescriptorHeap[]`: buffers, constant buffers, placed and streamed textures). This is what brought the first real albedo colours. | VIctory #413 (wbkepler), merged here with all of this tree's changes |
| A descriptor only read by `s_buffer_load` is a raw buffer (+291 programs offline, `a scalar memory read as data` 291 → 0) | VIctory #953 (MartinIndra02), brought in by UNYTIX (#8) |
| Dispatch bindings decoded from the console layout, for refused programs too (`dispatch-bindings` lines), and `VICTORY_TRACE_UAV_SIZE=<W>x<H>` (`uav-create` lines) | this fork |
| `VIcky: view heap …` at each captured frame: views over the whole title heap, how many lie past the host heap, their distribution; `VICTORY_BINDLESS_BUDGET_US` | this fork |

## Build and run

Same as VIctory: Rust as pinned in `rust-toolchain.toml`, and a `victory.toml` pointing at your copy of the title (see `victory.example.toml`). Use a diagnostics build:

```
cargo build --profile diag -p victory --features diagnostics
target\diag\victory.exe run "<path to the title's exe>" --execute
```

With `VICTORY_SHADER_LOOP_LIMIT=32`, testers report no device removal on several NVIDIA cards.

## Diagnostics this fork adds

- **`VICTORY_CAPTURE=<folder>`, `VICTORY_DUMP_TARGETS=1`, `VICTORY_CAPTURE_AT_SECS=170`.** These save the first frame built after 170 s of the run and every render target it writes, in pass order. Menus take a different number of frames each run, so a time is easier to aim with than a frame number. The world starts about 70 s in.
- **`VICTORY_LOG=victory_gpu::draw=debug,info`.** For that frame this logs every draw and dispatch, in order:
  - `draw … [vertex <hash> + pixel <hash>] targets …`
  - `dispatch XxYxZ compute <hash>`
  
  A program that has no host pipeline is marked `MISSING <reason>`; one that was left out is marked `SKIPPED`.

More switches (diagnostics builds):

- `VICTORY_SHOW_GBUFFER=1`: see the world's geometry live, in false colours.
- `VICTORY_ADAPTER=<part of a name>`: run on another GPU. For example `radeon` selects an integrated AMD GPU, which has native 64-lane waves.
- `VICTORY_HOST_LANES_MAX=32`: translate as for a 32-lane host on a GPU that has 64-lane waves. Combined with the previous switch, this gives an A/B of the wave64 emulation on one GPU and one driver.
- `VICTORY_VIEW_HEAP_CAP=1000000`: cap the view heap as NVIDIA does, on a GPU that takes the whole heap.
- `VICTORY_DEBUG_PSO=<hash>,…`: log a mesh pipeline's whole blend state and its target formats.
- `VICTORY_DUMP_SHADERS=<folder>` (and `VICTORY_DUMP_SHADERS_ONLY=<hash>,…`): keep translated containers. They are title data: keep them local.
- `VICTORY_TIME_CS=<file of record names>` with `victory translate --pipelines --jobs 1`: time the direct-DXIL path and the first driver compile, with the driver's cache off (needs Windows developer mode).

- `VICTORY_RT_MISS=1`: ray-traced passes run with every ray missing (an unresolvable address reads zeros; `image_bvh*_intersect_ray` hits no node) instead of being dropped. Offline, on the compute programs kept from refused translations, 41 → 64 emit and validate.
- `VICTORY_DEBUG_PS_PROBE=<hash>:<pc hex>:<vgpr>[:class]`: that pixel shader writes to render target 0 what the instruction at `pc` wrote to VGPRs `vgpr`.. (with `class`: red NaN, green zero, blue positive). Look at it with `VICTORY_SHOW_GBUFFER=0`. A probed translation is not cached.
- The shader cache key includes `VICTORY_COMBO_GUESS_STRIDE`, `VICTORY_SHADER_LOOP_LIMIT` and `VICTORY_MESH_CONSOLE_LANES` (issue #6).
- **Pitfall:** after runs on another GPU (`VICTORY_ADAPTER`), move `%LOCALAPPDATA%\victory\cache\pipelines` aside. Here the shared pipeline library (123 MB, plus a 548 MB temporary file left by a killed run) left about 1,960 pipelines "being made" for minutes on the next RTX run.

## What we know so far

Measured on an RTX 4080 (Windows 11), in diagnostics builds of this tree, unless a line says otherwise.

- **The world loads and runs at about 1 frame per second.** About 930–980 draws and dispatches per frame are dropped, of about 2,150; about 250 of those are only waiting for their pipelines to compile. The UI, subtitles and the loading art all render.
- **The heap cap is the blocker on NVIDIA (2026-11-24, night). It explains the black parts of the albedo and the black lighting.**
  - **The first real albedo colours:** with #413, G-buffer RT0 holds textured characters (skin, hair, clothes) under `VICTORY_SHOW_GBUFFER=0`. At frame 2066 it was 6.8 % covered, and it fills as slots are made.
  - **Many triangles stay black:** the view heap report at frame 2066 counted 502,197 title slots holding views, the highest at 1,245,182, and **217,439 past the host heap** (RTX 4080: 737,855 title slots mirrored under its 1,000,000 cap). A bindless read of such a slot reads the null guard slot.
  - **The used slots:** views per 64Ki slots are `[56k, 39k, 0, 0, 0, 24k, 40k, …, 25k]`; the empty runs are 110,940–327,680 and 341,414–360,448. Removing the gaps still needs ~1,009,410 slots, so only a slot map can fit the ~500k live views.
  - **Lighting:** the 160x90 `f24` texture the lighting reads **is written** by translated computes (`914e86141fa3b37f`, `08055d95e9ee79c0`, 40x23x17), but **through a UAV at heap slot ~904,335**, past the cap. The writes are lost, and every HDR target stays exactly 0.
  - **Next:** map title slots past the cap onto free host slots (a map buffer read in `heap_guard`; mirroring and the bindless sync write the mapped slot).
  - `VICTORY_RT_MISS=1` in the title: 104 → 88 untranslated programs, no device removal, HDR still 0 (expected while the lighting inputs are lost).
- **Lighting, newest leads (2026-11-24, late):**
  - **The HDR target is black on an AMD RX 7800 XT too** (0.11 % non-zero, UNYTIX in #7). So wave64 emulation is not why lighting is black.
  - In the traced frame, **the lighting draws read a 160x90 `f24` UAV texture that nothing writes** (2560x1440 in 16x16 tiles, likely a tile classification). Its likely writers, judged by their 160x90 grids, are computes refused on 32-lane hosts ("console wave64 lanes in a group of more than 64 threads": `5c6224a6bc3a4066`, `0c028454b58543f0`) or others (`23d507cbb6a22e86`, `2da4beb386509b77`). This is inferred: dispatch bindings are not traced yet.
  - A 16-byte buffer read by every HDR writer and by post-processing looks like the exposure.
  - **Bindless textures are never uploaded in this tree.** Only textures reached through descriptor tables are brought up to date, and the G-buffer's pixel shaders sample through `ResourceDescriptorHeap`. VIctory #413 (wbkepler) fixes this. It is merged locally (branch `combo3-413`, all victory-gpu tests pass) and waits for a title run. It is the first suspect for the zero albedo.
  - With `VICTORY_SHADER_LOOP_LIMIT=32`, a pixel lane that reaches the limit is killed (discard), which can also leave a lighting target at its clear value.
- **Latest (2026-11-24 evening):**
  - **Spilled root arguments (VIctory #302/#489).** The first cause of missing compute programs on NVIDIA here was "an address in an SGPR that is no root parameter": 12 of the 15 most-dispatched missing computes in a traced world frame, at code offsets 0x2c–0x40. They read their root arguments through the spill table `s[0:1]` points to. With #302/#489, that class drops from 34 programs to 1, untranslated programs drop from 129 to 115, and dropped draws fall to about 200–650 per frame.
  - **The G-buffer view shows real objects:** tables, lamps, a basketball hoop, plates, a phone, whole characters. Much of the scene is still missing, and a web of long thin lines remains on some views.
  - **Albedo (G-buffer RT0, `R8G8B8A8_TYPELESS` viewed as sRGB) is exactly zero**, in every frame. These causes are ruled out, with evidence:
    - a typeless RTV: the title gives typed formats;
    - the sRGB export: exports always take the console's 32_ABGR patch;
    - a dropped `SV_Target0`: it is stored;
    - missing texture memory: only 81 all-zero reads;
    - the view heap cap (below).

    The opaque G-buffer shader writes RT0 with blending off. Each channel is a legacy multiply, `saturate(1 - texA.x) × texB.x × …`, so a zero sample in either factor gives exactly 0. The blended layers use `SRC_ALPHA / INV_SRC_ALPHA`, with alpha = `s26 × pow(tex, s6)`.
  - **Native wave64 vs NVIDIA.** On the CPU's integrated Radeon (RDNA2, 32–64-lane waves, `VICTORY_ADAPTER=radeon`), the same build runs about 4,350–4,540 draws per frame, about what a private Linux/vkd3d build with a lit scene shows (4,519). The final picture there showed coloured light for the first time. On the RTX 4080 the same build runs 1,000–2,800 draws per frame. The iGPU runs out of memory for some pipelines and is very slow, so its picture stays partial.
  - **NVIDIA's view heap cap is exactly 1,000,000.** A probe of `CreateDescriptorHeap` on the RTX 4080 accepted 1,000,000 and refused 1,048,576 and every size above. With 2^18 dynamic slots, 737,855 title slots remain, of the 1,245,184 the title asks for. "title slot past the host heap" never fired, so the title wrote no descriptor past that.
  - **A tree on VIctory #189** (wave64 `Emulate`, multi-wave groups) **is worse than this one** on the RTX 4080. It had 673 untranslated programs against 115, 900–2,600 draws per frame, and a device removal at 2 min 18 s. It ran at 1.6–2.1 fps, but only because it ran less work.
  - **`VICTORY_SHADER_LOOP_LIMIT=1024`** (as another tester uses) left about 1,900 pipelines compiling for four minutes on a cold cache. With 32 they compile in one to two minutes.
- **Earlier:** compute `f0afa89ecfb6944e` caused the long thin lines. Found by elimination over the 38 programs the guess unblocked: it read a 128-byte record with a guessed stride of 136, a size the shader declares for another resource. With the 16-byte rule, the lines are gone and there is no device removal in 4 min 40 s. About 2,550 draws per frame run and about 560 are dropped. The HDR / lighting targets are still black (issue #2).
- **Earlier (stride guess + view guard): instance culling runs.** Dropped draws went from ~950 to ~565 per frame, and about 2,550 now run. The G-buffer shows recognisable objects (a picnic table, a chair, a lamp, a basketball hoop). It is also crossed by long thin lines: triangles whose vertices fly off, so some meshes get wrong positions. That's next. No device removal in 5 minutes.
- **Before that: the G-buffer was partly drawn.** In a world frame (a cutscene), the 2560x1440 G-buffer targets hold some geometry: curtains, a roof, plates, a light shaft. It is striped and noisy, and much of the scene is missing. The depth target matches.
- **Every HDR / lighting target is black.** The post-process output is a flat colour, which is why the screen is grey. So the first link that breaks is lighting, right after the G-buffer.
- **The traced world frame (frame 2514, about 4,500 draws and dispatches) shows where the breaks are:**
  - **Instance culling.** Compute `e30970f1727063fd` is `MISSING` (a structured buffer of unknown stride) **321 times per frame**. It sits between the G-buffer's mesh draws, alternating with computes `1ea7e2dbf731c8dc` and `4115dd63f99a09d7`: the GPU-driven culling that picks what the next mesh draws draw. Without it most objects are never drawn. VIctory's INTEGRATION.md names the same unknown-stride stop for the four instance-culling programs of VIctory #71.
  - **Lighting.** The lighting does run, and it writes the HDR target (`2560x1440 f10`):
    - a full-screen draw: vertex `8157a974717876a9` + pixel `3ded230430ee2590`, also writing a `2560x1440 f61`;
    - then mesh draws: `a26422c01dcf5612` + `6a3a8d3fe52b4eab`, and `f84fa6626772ebf0` + `f9a4d2f8d13f46ef`.
    
    The pixel shader reads a structured buffer of 2,044 elements, a 160x90 `f24` texture and two 3D textures. Its output is black, so its inputs are probably empty. Candidates are the computes still `MISSING` in the frame: `3b90165a77d95cdf` 186× (an address in an SGPR that is no root parameter), `f6e47efa50226beb` 129×, `5552dac6a2b7a1fd` 55×, and about 10 more at around 50× each.
- **The step from the 1440p scene to the 4K output** is the same open question as VIctory #841, where the scene is fully drawn at 1440p and the 3840x2160 output stays zero.
- **Shader translation is still the main source of drops.** The classes that drop the most draws in our runs:
  - `v_readfirstlane` / cross-lane operations in divergent control flow;
  - an address in an SGPR that is no root parameter;
  - a structured buffer of unknown stride;
  - a lane mask read as data.
- **Descriptor heap.** The title's view heap has 1,245,184 slots; NVIDIA caps a shader-visible heap at 1,000,000. The title never reaches its upper slots in our runs.
- **RAM, not VRAM, runs out.** On a 31 GB machine the process committed 25.6 GB, and 1.5 GB were left free.
  - The title's memory lives in system RAM, including its streaming pool of about 6 GB.
  - Each buffer the GPU reads has a host copy, also in system memory (`shadow.rs`). Ranges that shaders write are also snapshotted per submission.
  - Pipelines compile in the background: 1,000+ of them on a first run with a new build.
  - Ideas: use the title's memory in place (below); put GPU-only buffers in video memory.
- **Memory-copy cost (about 2/3 of a frame).** Experiments, not in this tree yet:
  - A pagefile section created with `SEC_COMMIT` and smaller than 4 GiB opens as a host D3D12 heap (`OpenExistingHeapFromFileMapping`). The GPU then reads and writes the title's pages directly, with no copy.
  - At 4 GiB or more, or with `SEC_RESERVE`, opening it fails.
  - Reserved (tiled) resources mapped onto such a heap removed the device.
  - Buffers that are one run of a section can be used in place. The streamed pages are scattered, so the gain so far is small.

## How to help

- Run the tree with the diagnostics above and post what you find in an issue: which pass first writes black, which program is `MISSING` there, your GPU and driver. **Post text only, no game images.**
- Pick a translation class above and look into VIctory's translator (`crates/victory-shader`).
- Related threads: VIctory #841, #850, #840, #71; XCL ivall/xcl#505, #1031.

## Credits and licence

VIctory and every PR in `integration/testers` belong to their authors (see `INTEGRATION.md` and the git history). This fork is GPL-3.0, like VIctory ([LICENSE](LICENSE)).
