# VIcky ray-tracing work (`main`, formerly the `rt` branch)

This work is for making the title's compute ray tracing (the BVH structure builder plus traversal) usable on PC.

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

**VIcky's own commits.** All are opt-in, except the dispatch parts (last item).
- `VICTORY_BACK_JOIN=<bytes>` writes nearby runs of shader writes back as one kernel call. The builder's output gave about 850,000 runs a frame, one `NtWriteVirtualMemory` each.
- `VICTORY_MEMORY_REFRESH=touched`: shaders note which copies they read (no extra store), and only those are refreshed, plus every copy once in 8 frames.
- `VICTORY_MEMORY_NAMED_WRITES=0`: a copy that is only named by a constant and never marked stops being snapshotted.
- `VICTORY_SNAPSHOT_POOL_MB=<n>` keeps snapshot buffers by size for reuse.
- `VICTORY_REFRESH_STAMP=1`: measured 0 skips; useless, will go.
- `VICTORY_NOTES_ON_CARD=1`: the notes of programs reading title memory by address stay in the card's memory and are copied back once a frame (not measured in a run yet).
- `VICTORY_HEAP_SYNC_CACHE=1 VICTORY_HEAP_BUFFERS_SKIP_SAME=1 VICTORY_HEAP_SYNC_THREADS=4`: a cheaper sync of the descriptor records the title copies into its heaps (buffer views kept by record and copied with `CopyDescriptorsSimple`, notes that cannot change skipped, pages read on 4 threads). `VICTORY_HEAP_SYNC_CHECK=1` also does one in 64 the slow way and counts differences (0 in RT40). RT40/RT40b below.
- `VICTORY_BVH_COPY=skip`: the builder's copies of a structure onto itself are not recorded (RT32, below). `native` also runs a pass of ours for any other copy, and `check` compares that pass with the program (diagnostics).
- `VICTORY_HEAP_SYNC_BESIDE=1`: the heap sync runs on a thread of its own beside the refreshes; the submitting thread only waits at the join (about 25 ms a frame less on it). RT48b below.
- `VICTORY_FENCE_SENT=1`: the frame-begin fence (the value the title signals and waits on before recording the next frame) is published to the title when everything before it is sent, not when the card has run it; it still lands in order and never goes backwards. While a value is ahead, a byte that both the card and the title changed stays the title's in the relay's merges (the console's order: the card's writes before a signal are done when its value is reached). `VICTORY_FENCE_SENT_CHECK=1` counts the bytes each rule kept (diagnostics, about 40 ms a frame); `VICTORY_FENCE_SENT_KEEP_TITLE=1` also stores only the 4-byte words the card wrote (slower). RT55–RT60c below. **It pays only together with `VICTORY_HEAP_SYNC_BESIDE=1`**: each alone gives about 200 ms a frame, the two together about 180–189.
- Shader clean-up passes (`fx-rt-shaderopt`, see `VICKY.md`): audited and checked side by side, two fixes (resource handles stay in their block, as in fork PR #317; no folding of two empty latches), wave operations kept by default (new default 63), pass names and per-program `VICTORY_SHADER_OPT_NOT` / `_ONLY`. The translator changed, so the shader cache key changed: **warm the cache again after updating** (`victory translate --warm`).
- **Dispatches in parts, on by default** (branch `fx-rt-dispatchsplit`): myokex's fix, from ZSlayerHQ/victory-world-runtime `4436f8f` and `892d259`, ported.
  - The problem: the host runs at most 65,535 thread groups in a dimension of one dispatch; the console runs any number. A full-screen compute pass asks for about 72,000 to 76,000 groups in x. It was cut at 65,535, which left a blocky strip along the bottom of the frame.
  - The fix: translated compute programs add a first-group number the host sets to their group number in x, and a larger dispatch runs in parts (up to 16).
  - `VICTORY_DISPATCH_PARTS` unset (or `seen`): a single-record indirect dispatch runs in parts only once its program has been seen asking for more. RT67: strip gone, 186 ms a frame, no cost.
  - `all`: every single-record indirect dispatch runs in parts. RT64a: strip gone, but about 20 ms of card time a frame more on an RTX 4080.
  - `0`: cut, as before.
  - The log line `dispatches in parts` counts both paths.
  - The translator changed, so the shader cache key changed: **warm the cache again after updating** (`victory translate --warm`).
- **The submit thread's walls (2026-11-26 night, all opt-in; RT65–RT81 below).** Together they take the frame from about 5.0 to about 5.5–5.7 fps on the RTX 4080 (host `fps` lines, long frames included):
  - `VICTORY_HEAP_PASS_BESIDE=1`: the pass over the heap's buffer views (once a frame, 110–200 ms inside one list call of the title) runs on a thread of ours; the title's recording thread leaves a placeholder and returns; the join is at the start of the submission. `VICTORY_HEAP_PASS_BESIDE_CHECK=1` compares one pass in 16 slot by slot (RT65: 0 of 5,178,864 slots differed).
  - `VICTORY_HEAP_PASS_SPREAD=1`: each pass takes the written slots plus one sixteenth of all buffer slots, instead of every slot every 16th pass (a ~300 ms pass every 16 frames).
  - `VICTORY_HEAP_PASS_ONE_LOOK=1` (the pass's own making no longer marks the slot again, so a slot the title wrote once is looked at once, not twice) and, with the spread, `VICTORY_HEAP_PASS_DUE=1` (the window holds only the buffer slots no pass looked at in the last 16, not a sixteenth of all ~460,000 buffer slots). RT83b/c (this branch, with the dispatch parts): the pass takes ~74,000–78,000 slots instead of ~119,000 (window ~11,000 instead of ~29,000; ~37,000 re-marks spared a pass), 116–144 ms a pass, and the submitting thread waits 0–5 ms for it instead of ~29; check: 58 passes, 4,204,312 slots, 0 differed. Frame: 5.48 / 5.30 fps (RT82c/d without them: 5.12 / 5.35): the submitting thread (~178 ms) is now the wall.
  - `VICTORY_HEAP_PASS_THREADS=<n>` (opt-in, not run yet; 4 suggested): the pass makes its slots on `n` threads (its own and `n - 1` more, made for the pass), chunks of 64 slots taken in turn. The heap's slot state is held by them together (its read lock); everyone else (title writes, the heap sync, draws, the texture pass) takes it alone as before; each slot's key and heap map entry are atomics only the thread making that slot touches, and the heap map's pool and the cache of made views have small locks of their own. A thread that finds the pool used up gives back no other slot's pool slot: those slots are made after the threads (counted on a line of their own). The results are put in slot order on the pass's thread, so the refreshes, the splice and the join are as before. What may differ: the order the slots are made in, so which pool slot a title slot gets (and, if the pool is used up and too few are given back, which slots go without one), and which records the cache keeps. `VICTORY_HEAP_PASS_THREADS_CHECK=1` (diagnostics) runs one such pass in 16 again on one thread and compares slot by slot (view, copy, range, record made, buffer held).
  - Always on: a line `the pass over buffer views, where its time goes` (per pass: threads' time and start to end, waits for the slot state's lock and how long it was held, the rest split by part with the time stamp counter in one chunk in 8, and the slots by what they came to).
    - PT0/PB1 (one thread, 139-147 ms a pass): lock waits only 3-5 ms; the diagnostics' look at the buffer each slot shows 33-59 ms, kept ranges 17-22, host copies looked up 13-18, the walk 14-18, decisions 11-14, notes 11-14. So the work, not waiting.
    - Since: that look (diagnostics builds only) is off unless `VICTORY_HEAP_PASS_SHOWN=1` or a check of the pass is on (the memory line's count of views whose slot showed another copy stays 0 without it); the kept ranges are a vector by slot instead of a hash map (same ranges); the cache of made views picks a record's set by the top bits of its product, so views at one address with other elements no longer share one set of two. The line also counts, of the slots made from their record already, how many found the copy and range they had kept (what a per-slot cache of them could spare at most).
  - With `VICTORY_FENCE_SENT=1`, the bytes the relay's merges keep as the title's now go up at the landing (on by default under the fence; `VICTORY_FENCE_SENT_PUT_UP=0` turns it off; `VICTORY_FENCE_SENT_PUT_UP_RUNS=4096` recommended, 64 by default is too low for this scene).
  - `VICTORY_DENSE_PAGE_WHOLE=1` and `VICTORY_DENSE_PAGE_PATCH=1`: a card copy's page with more changed runs than `VICTORY_MERGE_RUNS` (the title's 182 MB heap: thousands of 1–3-byte runs a page) no longer makes the submitting thread wait for the card: a page no shader wrote lately goes up whole, others are patched on the card under a byte mask (`card = (card & ~mask) | (title & mask)`), queued behind the flights. `VICTORY_DENSE_PAGE_PATCH_CHECK=1` reads one page in 16 back (RT77: 0 wrong bytes). Settles went from 2.0 to 0 a frame.
  - `VICTORY_GUARD_SUMMARY=1` (bits over the clean blocks) and `VICTORY_GUARD_ABSENT_KNOWN=1` (holes of view memory asked about once a frame, not at every asking): the guards' 32 ms a frame were 23 ms of `VirtualQuery` on uncommitted blocks (36,640 a frame); now 15–17 ms. `VICTORY_GUARD_ABSENT_CHECK=1` re-asks one answer in 64 (RT76b: 0 of 372,487 wrong). Also `VICTORY_GUARD_HOT`, `VICTORY_GUARD_VIEWS_BATCH` (no measurable gain).
  - `VICTORY_FILL_LATCH_PIECE=1`: a fill that raced a flight latches only its 64 KiB piece, not every fill for the rest of the run (that run-wide latch cost 27–39 ms a frame when it tripped).
  - `VICTORY_MERGE_RUNS=4096` with the above.
  - `VICTORY_SPARSE_PROBE=1`: a copy of 64 KiB or more that met a hole is read around it by the kernel's own count and page by page, not by asking `VirtualQuery` region by region (about 2 ms a question in the title's big pools: RT83c's 15–23 fallbacks a frame took 34–42 ms). Same bytes (unit tests against the old way on reserved regions with every kind of hole); a hole longer than 1 MiB is still asked about once. The frame line's "sparse reads" now also says the ranges read around holes, the kernel reads (and failed ones) and the questions. RT84a/b (this branch, all the above): sparse reads 6.7–9.3 ms a frame (0.3 `VirtualQuery` a frame); **RT84b / RT84c 6.00 / 6.37 fps** (true means 5.80 / 6.27), the bottom strip fixed; the submitting thread 159 ms.
- **The submitting thread's copies (2026-11-27 night).** Profile PF1 (main, `VICTORY_PROFILE=1`, ~150 s in the world): `Shadow::update` is ~30 % of `Host::submit`, 93 % of it from `Batch::flush` (at the steps that write title memory and at the submission's end), 7 % from `flush_one`; inside it the guards' walk 36 % (`VirtualQuery` about holes 21 %), whole copies 46 % (in place 24 %, `ReadProcessMemory` 21 %), the journal's collect 6 %. `Shadow::writing`, `Flights::declare` and the scan of pending ranges do not show.
  - Diagnostics, always on: the copies line gives the distinct (copy, part) pairs a frame (a part ends at each title copy or work step), of them view memory and those refreshed again after their shaders' first write, and ends with "copies split" (fills, title copies, askings ahead, updates and those before a shader write, askings of the journal or guards, pages copied, the journal's collect).
  - `VICTORY_REFRESH_FRONT=1` (opt-in, not run yet; after codec-d/cool_fox dc2a3ad4 and 37332f4d, on the submitting thread): at the start of each part of a submission every copy the part refreshes is brought up to date once, over the span of its refreshes before the part first gives it to shaders (what the batch would merge for it there); the part's refresh steps inside the span are skipped, later ones stay in the batch. It turns on `VICTORY_REFRESH_ONCE`'s rule (a range up to date in a submission is not brought again until a step of ours writes title memory there; title writes of other threads go to the next submission, as with `VICTORY_ASK_ONCE`); each part is brought only after the step before it ran. `VICTORY_REFRESH_FRONT_CHECK=1` (diagnostics) does one part in 16 both ways and counts the bytes that differ.
  - `VICTORY_LIGHT_REFRESH=1|alt` (opt-in, not run yet; fork 94ed363c by fxceless, after avatarmanz 2aa8134b and 74045e46, from its form in `fx-main-forkmerge` 78162a4): a bit per 64 KiB block read-only by our hand and unwritten since; a range whose blocks all have theirs is answered from the journal with no guards' lock and no kernel call, and an update that would copy nothing returns at once. Holes get a bit only where `VICTORY_GUARD_ABSENT_KNOWN` (`=1` or `forever`) answers for them and only in a placeholder reservation (pages come there only through the runtime's told mappings); 16 holes with bits are asked of Windows again per submission, one found filled is said once. The fork's absent-once notes and commit hook are not ported. `alt`: twenty seconds with, twenty without.

- **Stair-step shadow edges and the characters' heads (2026-11-26).** The 8x8-pixel stair-steps on shadow edges came from `CS_ShadowRenderTiledCascadesAlphaCBPPPC` (#f133e35e), the only program of the scene the translator refused ("a lane mask exchanged where not every lane runs"), so its indirect dispatch was dropped.
  - DexNom's recovery of a wave64 mask restored from scalar copies (`structure.rs`, DexNom 76749ce) makes it translate. `VICTORY_WAVE_MASK_COPIES` (default `retry`) applies it only as a second attempt after that exact refusal; offline, only #f133e35e changes (`0`: never, `all`: every program first, or hash prefixes).
  - With the program running, heads and skin vanished (RT88a). A store-drop bisect (`VICTORY_STORE_DROP=<hash>:<offsets>`, W1a) named its read-modify-write at `0x2957c`, which sets bit 14 of a per-tile word in a typed buffer. The translator declares typed accesses as float, so through a 32-bit integer view the bits were converted and the tile's other flags wiped.
  - **Fix, on by default:** typed buffer views of 32-bit integer formats are made in the float format of the same width (after fork #684, GamingRipperDoc, `bit_exact_format`; `VICTORY_TYPED_BIT_EXACT=0` turns it off). F1: stair-steps gone, heads and skin drawn, no light rings.
  - Diagnostics kept: `VICTORY_STORE_DROP`, `VICTORY_SKIP_PIPELINES=<hash>@<n>`, `VICTORY_CAPTURE_ON=<hash>`, `VICTORY_TYPED_OFFSET=refuse` (no program of this title needs it, after VIctory #3009), `VICTORY_WAVE_SITE_PAIRS=<hash>|all` (fork #1002's exact forms, Bruno Magalhaes; with `VICTORY_WAVE_MASK_COPIES=0`).
  - The translator changed, so the shader cache key changed: **warm the cache again after updating** (`victory translate --warm`).
- **Loading: physical pages from sections of at most 2 MiB** (`VICTORY_SECTION_MB=2`, opt-in, recommended; after fork PR #1274, RyanCase94 and the #1089 reporters, and myokex 48451e9). The title's page mapping calls waited on the lock Windows' large-page worker holds on one ~7 GB section. Four 360-s runs on the RTX 4080 (S1a-S1d, off / 2 / off / 2): remapping 73.7 / 2.9 / 69.4 / 3.3 s (framelog), time in Windows' calls 103.6 / 3.2 / 90.4 / 3.4 s, calls of 1 ms or more 15,027 / 10 / 15,089 / 18, frames over 1 s in the loading window 15 / 4 / 16 / 6; steady fps unchanged. A DirectStorage destination that crosses sections is decoded into a buffer and written run by run. Counters: the `physical pages, since the start:` line.
- **Small title buffers' copies in the card's memory** (`VICTORY_CARD_BUFFERS_MIN_KB=64` with `VICTORY_CARD_BUFFERS_MB=2048`, opt-in, recommended; after fxceless c36971b2 and avatarmanz 17c4717b; `VICTORY_CARD_HASH_LEN` hashes only a buffer's own bytes, following `MIN_KB` unless set). Four 360-s runs on the RTX 4080, Series X, RT on (A = `CARD_BUFFERS_MB=2048` alone, B = A + `MIN_KB=64`): A 5.80 / 5.97, B 6.27 / 6.24 fps (true means): copying back 59-62 ms a frame instead of 74-75, copies 89-108 ms instead of 63-76. No black shards seen in the captures, no "no card copy for a title buffer".
- **Write-combining copy-back left on the card** (`VICTORY_SKIP_WC_BACK=1|2`, opt-in, NOT recommended together with the card buffers above; after fork PR #868, Chipsraine and iGoldenNinja; `VICTORY_SKIP_WC_COMPARE` not ported). `VICTORY_CENSUS=1` logs what the card copies send, compare and fetch back: in our scene ~99 MB a frame fetched back (95 MB write-combining), of which `=1` can leave 24 MB on the card and `=2` 29 MB; 65 MB lie in pages shown at more than one address (never left); 35 MB are taken again by the copy that wrote them. Alone (K1a-d, off / on / off / on): 5.63 / 5.98 / 5.68 / 5.67 fps true (6.14 / 6.11 against 5.69 / 5.75 on the host fps lines), copying back 84-85 to 71 ms. With `CARD_BUFFERS_MIN_KB=64` (X/Y runs): no gain over the card buffers alone (5.14 / 6.38 against 6.21 / 5.55) and 3-8 "another copy took bytes left on the card" a run (small card copies reading pages left there), so use one or the other.
- Tried and dropped the same day: `VICTORY_NAN_SCRUB` (trustno1-coder; zeroing NaN/inf in colour exports darkened the whole interior on this title), `VICTORY_GUARD_SHARED` (reader-writer guard locks; 5.29 / 6.02 against 5.78, guards ~18 ms either way), and `VICTORY_TABLES_LEFT` (table slots left to the draws, after cool_fox/avatarmanz/myokex; the heap pass fell from ~60 to ~12 ms but copies rose to 90-115 ms: 5.32 / 4.92 against 5.68, and the light rings came with that build).
- The window caption names the console and whether the builder runs (`Series S, RT on`, `RT off`, `RT stopped`).

**VIcky PRs merged:**
- #23 (g0shtj0sh): main's opt-in `VICTORY_NO_WATCHDOG=1` hook (upstream #454, fork #364), for slower PCs whose runs the title's watchdog kills at the world's entry. Nothing changes when it is unset.
- #24 (wololo-wololo): `VICTORY_BVH_CHECK=<frames>` (diagnostics) walks the ray-tracing structures that raw-pointer dispatches name and logs counts of what is wrong (task 2, section below).

**Recommended with the builder on:** `VICTORY_BVH_COPY=skip VICTORY_HEAP_SYNC_CACHE=1 VICTORY_HEAP_BUFFERS_SKIP_SAME=1 VICTORY_HEAP_SYNC_THREADS=4 VICTORY_HEAP_SYNC_BESIDE=1 VICTORY_FENCE_SENT=1 VICTORY_BACK_JOIN=256 VICTORY_MEMORY_REFRESH=touched VICTORY_MEMORY_NAMED_WRITES=0 VICTORY_SNAPSHOT_POOL_MB=2048`, plus the fork's: `VICTORY_ASYNC_SUBMIT=1 VICTORY_QUEUE2=1 VICTORY_INDIRECT_WINDOW=1 VICTORY_TEXTURE_NO_WAIT=1 VICTORY_COPY_PLAIN=1 VICTORY_JOIN_PARTS=1 VICTORY_BACK_OWN_VIEW=1 VICTORY_INDIRECT_GATHER=1 VICTORY_CARD_REHASH_RUNS=1`. Keep the capture harness rare (`VICTORY_CAPTURE_WORLD_EVERY=64`): a saved frame costs about 235 ms. If the title's watchdog ends the run at the world's entry (frames of 20–25 s while it loads), add `VICTORY_NO_WATCHDOG=1` (PR #23); it changes nothing once the world is up.

**Also recommended:** `VICTORY_SECTION_MB=2` (loading) and `VICTORY_CARD_BUFFERS_MB=2048 VICTORY_CARD_BUFFERS_MIN_KB=64` (both above).

**Fastest measured (RT81b/c, RTX 4080, before the dispatch parts were merged):** the above plus `VICTORY_HEAP_PASS_BESIDE=1 VICTORY_HEAP_PASS_SPREAD=1 VICTORY_FENCE_SENT_PUT_UP_RUNS=4096 VICTORY_MERGE_RUNS=4096 VICTORY_DENSE_PAGE_WHOLE=1 VICTORY_DENSE_PAGE_PATCH=1 VICTORY_GUARD_SUMMARY=1 VICTORY_GUARD_ABSENT_KNOWN=1 VICTORY_FILL_LATCH_PIECE=1`, plus `VICTORY_HEAP_PASS_ONE_LOOK=1 VICTORY_HEAP_PASS_DUE=1` (RT83b/c, above) and `VICTORY_SPARSE_PROBE=1` (RT84b/c: 6.00 / 6.37 fps, bottom strip fixed). On this branch, with the dispatch parts on (the full-screen pass now runs all its tiles: about 5–10 ms more card time), the same settings give 5.12 / 5.35 fps (RT82c/d); 5.69 / 5.48 was without them. A new build's first run carries the driver's background shader work for several minutes (process user time about 1,500 ms a 5-s line instead of about 900): measure from the second run.

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

### Task 2 on the RTX 5090: a checker for the structures, and what it counts (`3fbc9cb`)

**The checker.** `VICTORY_BVH_CHECK=<frames>` (diagnostics build): every that many frames, the addresses the dispatches of shaders that read title memory by address named are tried as structures and walked from their root in title memory. It logs counts only (`victory_gpu::bvhcheck`): one line for the structures of instances, one for the structures of triangles, and up to six structures with something wrong. A check of about 50 structures takes 15–25 ms; of 2,000 while the world loads, about 1 s.

**The layout it needed, measured (not from any document).** A structure begins with a header at the address a dispatch names; node pointers count from that address; the dword at byte 48 of the header is the root pointer. Found by looking for box nodes that fit together in 1.3 GB of the copies shaders had touched, without assuming a base (how is in `bvhcheck.rs`).

**What it counts as wrong:** a pointer of kind 7, a pointer to a node of zeros, a triangle with a vertex that is not finite, a child that reaches outside the box its parent holds for it (0.2% of room for 16-bit bounds), a node reached twice. A child whose box no ray can enter (not finite, or minimum above maximum) is not followed and not counted as wrong. "Far" is a structure whose nodes lie more than 256 MB apart: a pointer that leaves it.

**Builder on, the cell scene, one run of 400 s, a check every 64 frames.** The thirteen checks after loading (the fourth check on), range over them:

| | structures | box nodes | triangles | instances | outside | kind 7 | zero node | reached twice | bad triangle |
|---|---|---|---|---|---|---|---|---|---|
| of instances | 1–4 | 263–979 | 2–18 | 478–1,610 | 4–84 | 3–27 | 0–1 | 0–3 | 0 |
| of triangles | 38–58 | 4,060–6,238 | 8,143–12,629 | 84–133 | 1,389–2,502 | 95–155 | 50–69 | 92–167 | 35–113 |

- 36–53 of the 41–62 structures named in a check have something wrong, and 8–20 are far.
- In the structures of triangles about one child in eight is outside its parent's box (1,389–2,502 of the 12,000–19,000 box nodes and triangles walked), and they hold 84–133 instance pointers, which a structure of triangles should not.
- While the world loads (the first three checks) 1,021–2,094 structures are named, most of them shipped ones: 438–751 have something wrong and 103–182 are far.

**The shipped structures, for comparison** (one snapshot of that scene, 181 structures by the same rule): 126 have nothing wrong. The larger shipped ones (1,000–10,800 box nodes each, 12–14 deep, 0.3–2.9 MB from first node to last) have no kind 7, no zero node and no node reached twice; the only thing counted is a child outside its box, at most 0.4% of what is walked (92 of 37,500 in the largest), which may be the check's room for rounding being a little tight.

**So, to the question:** yes. With the builder on, the structures the builder makes hold pointers that lead out of the structure, hundreds of megabytes or more away (8–20 structures a check), which is memory a traversal that follows them asks copies for. The larger shipped structures stay within 3 MB each.

**What this does not show.**
- The check reads title memory at one moment. A structure the builder is rebuilding may be half old and half new there, since its stores come back a submission later. Some of what is counted may be that and not the builder's output.
- Instance nodes are not followed, so a bottom-level structure is only checked when a dispatch names it.
- With the builder off no dispatch names an address in a form the check reads (0 named in 15 checks), so there is no builder-off row.

### Task 3 on the RTX 5090: no crash without quarantine (`493cc0d` to `3fbc9cb`)

**It does not crash here.** 27 runs with `VICTORY_BVH_BUILDER=1` and `VICTORY_BVH_QUARANTINE` unset, 3.6 hours in all, on `493cc0d`, `1a7eb50` and `3fbc9cb`: no panic, no device removal, no access violation, and the title never died. Every run was closed by the script and gave exit code 0.

- The longest is 30 minutes on `3fbc9cb` (5,317 frames, the builder whole to the end); the others are 5 to 15 minutes.
- In 25 of them the builder ran whole to the end (`the ray-tracing structure builder runs`, never turned off).
- In the other two it turned itself off mid-run (`one of its dispatches could not be bound`, once forced by a local test, once after input in the window), both within the first minute of the world. The title went on to the end of the run in both: a builder that stops between frames did not take the title down.
- So there was nothing to bisect with `VICTORY_BVH_SKIP`.

**What this does not cover.** All 27 runs are the same scene, the cell cutscene, with no input after the menus. Nothing here says the open world is safe, and task 2 above counts pointers in the builder's structures that lead out of them: the stores that crashed run 98 may still happen where they hit nothing the title reads in this scene.

### Task 5 on the RTX 5090: fork #814 against this branch (`b18fce0`), same card and scene

**This branch is about twice as fast here: 4.9–5.5 fps against 2.5–2.7.** Two runs each of 600 s from the same save, alternating, the cell scene, Series X mode, builder on, no capture, after a warm-up run each. #814 is `staronthehorizon:rt-preliminary` at `ece4a19` (4 commits on fork main `9376fd6`; this branch has 89 commits it lacks). The last twelve `fps` lines of each run, and the last three `shader writes looked for on the CPU` lines:

| | run | fps | ms/frame | preparing | GPU | copying back | dropped draws | compared in windows |
|---|---|---|---|---|---|---|---|---|
| this branch, `b18fce0` | 1 | 4.9–5.5 | 183–206 | 92–111 | 25–46 | 43–55 | 4 | 324–351 MB |
| | 2 | 5.1–5.4 | 186–197 | 93–101 | 37–48 | 48–62 | 4 | 358–393 MB |
| #814, `ece4a19` | 1 | 2.5–2.7 | 364–393 | 247–276 | 61–73 | 84–131 | 0 | 216–220 MB |
| | 2 | 2.5–2.7 | 367–397 | 250–283 | 57–74 | 97–123 | 0 | 240–241 MB |

- **Switches.** Both ran with everything "Recommended with the builder on" above. #814 also had its own `VICTORY_BVH_QUARANTINE=0 VICTORY_CARD_FETCH_INFLIGHT=1`, and not its two for 8 GB cards (`VICTORY_TEXTURE_DROP_MIPS`, `VICTORY_TEXTURE_ROOM_PERCENT`). #814 predates most of the recommended switches and ignores the ones it does not know, so this compares the two branches as they are, not the dirty-page bitmask against #346 alone.
- **Where the time goes.** #814 spends about 170 ms a frame more preparing (247–283 against 92–111), about 55 ms more copying back and about 25 ms more on the GPU.
- **What #814 does better here.** It compares about a third less in windows (216–241 MB against 324–393 MB). It drops no draw, where this branch drops 4 a frame. It does not run out of copy numbers of title memory by address: `no number left for another copy` is in neither #814 run and in both of this branch's.
- **Both held.** The builder ran whole to the end in all four runs, with no panic and no device removal. Both #814 runs were still running 180 s after the window was asked to close and were killed by the script; this branch's runs closed by themselves with exit code 0.
- **Not looked at:** the picture (no capture was taken), and Series S mode, which is what #814 was written and measured for.

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

**RT36–RT41, what came next** (same scene and switches as RT35):

| run | change | ms/frame | verdict |
|---|---|---|---|
| RT36 / RT36b | `VICTORY_MERGE_RUNS=65536`; `VICTORY_MERGE_GAP=64 VICTORY_MERGE_RUNS=4096` | 256–269 / 230–242 | settles go to 0 ms, but copies and copy-back grow as much: no gain |
| RT37 | per-4 MiB-cell skip of refreshes that find nothing (branch `fx-rt-cells`) | (check on) | correct (0 stale), but at most ~2.5 ms: a refresh that finds nothing costs only 6 µs |
| RT38 | `VICTORY_NOTES_ON_CARD=1` | 260–278 | no gain while the card is idle half the frame |
| RT39 | small fills copied in place (not kept) | 269–284 | 9× more page faults; copies doubled |
| **RT40 / RT40b** | **cheaper heap sync** (check on / off) | **211–215 / 199–203** | **kept**: about 5 fps; sync 96 → 45 ms a frame; 0 differences; picture as before |
| RT41 | RT40b + `VICTORY_MERGE_GAP=64 VICTORY_MERGE_RUNS=4096` | 240–254 | worse: copies 136–172 ms |

**RT48–RT60, the next two walls** (RT40b's settings; ms/frame is the log's "usual frame", mean of the last 30 lines, 2.5 minutes, unless a range is given):

| run | change | ms/frame | verdict |
|---|---|---|---|
| RT48b | `VICTORY_HEAP_SYNC_BESIDE=1` | 197–202 | the submitting thread 25 ms lighter, but the frame does not move: the title's recording waits for the frame-begin fence |
| RT58 | `VICTORY_FENCE_SENT=1` alone | 202 | no gain: the submitting thread is the wall |
| RT55b / RT55c | both (first version, check on / off) | 197 / **181** | about 10% faster; the check counted ~9 KB a frame both the card and the title changed, kept as the card's |
| RT60 | both + the title-wins rule + `_CHECK` + `_KEEP_TITLE` | 218–221 | correct (title-only bytes overwritten: 0), but the check and the word stores cost ~40 ms |
| **RT60c** | **both + the title-wins rule** (in this branch) | **189** (176–204) | **kept**; ~7 ms slower than RT55c: 1.8 settles a frame for title memory changed under shader writes instead of 1.0 |
| RT53c | shader clean-up passes on (default 63) | 193–197 | walls drawn; no GPU gain on this card (NVIDIA's compiler likely removes the same) |
| RT54 | `VICTORY_JOIN_LISTS=24 VICTORY_JOIN_MOST=2` | 203–216 | worse: 16 submissions a frame |
| RT56 / RT57 | heap pass skips its own unchanged slots | 188–194 / 188–189 | marginal alone, worse with the fence; dropped |
| RT50c | snapshots kept across frames | 210–218 | correct, no gain (shaders that pick views by index make pieces stale); dropped |

**RT61–RT81, the submit thread's walls** (RTX 4080, same scene; **fps from the host's `fps` lines, mean of the last 24 (about 2 minutes), long frames included** — the "usual frame" leaves out long frames and read 10–20 ms low):

| run | change | fps (ms) | verdict |
|---|---|---|---|
| RT60c–RT63 | rt 9510272 (heap sync beside + safe early fence) | 4.96–5.15 (195–203) | the base of the night |
| RT61 | `VICTORY_SPREAD=alt` + spare processors | worse (≈216 ms usual) | the widened groups spin; dropped on this 12-core part |
| RT62 / RT63 | `VICTORY_REFRESH_ONCE`, `VICTORY_WIDE_REFRESH`, `VICTORY_VIEW_KNOWN` | ~4 ms / 0 / 0 | parked / dropped |
| RT65 | heap pass beside, check on | slower | correct (0 of 5.18 M slots differed); recording is no longer the wall, the submit thread is (2 settles a frame) |
| RT66d / RT68 | + settle fix, `VICTORY_MERGE_RUNS` 4096 / 65536 | 5.17 / worse | raising the caps only raises the runs wanted |
| RT74 | + `VICTORY_DENSE_PAGE_WHOLE=1` | 4.48 | settles 2.0 → 0.8; names the page (the title's heap) |
| RT76b | guard summary + `VICTORY_GUARD_ABSENT_KNOWN=1` | 4.82 (first run of its build) | guards 32 → 17 ms, submit thread 131 ms |
| RT77 | + `VICTORY_DENSE_PAGE_PATCH=1`, check on | 5.34 (188) | settles 0.2, check 0 wrong bytes |
| RT78 / RT78b | all of the above | 5.12 / 5.21 | the full heap pass every 16th frame and a run-wide fill latch remain |
| RT79 / RT79b | + `VICTORY_D3D12_BACKGROUND=0` | 5.45 / 5.26 | **not adopted**: the character's hair washed out in both (picture A/B pending) |
| **RT81b / RT81c** | + `VICTORY_HEAP_PASS_SPREAD=1 VICTORY_FILL_LATCH_PIECE=1` | **5.69 (176) / 5.48 (186)** | fastest, two clean runs; settles 0–0.3, submit thread 165 ms; picture as RT78 |
| RT82c / RT82d | the same on this branch (7e8e6a8: + dispatch parts) | 5.12 / 5.35 (the first two runs of the build were the driver's) | the bottom strip fixed; the price of the full pass |

- Next: copy numbers. Every builder-on run uses all 1,023 numbers of title memory by address by the second world frame, and later pages read zeros; that fits #24's count of structures with nodes of zeros and pointers far out. Recycling idle numbers is being written.

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

**Open: the dark lower-right quarter of the picture.** It predates the dispatch parts.
- It is lit only when plain single-record indirect dispatches go through the parts' command signature, which sets a root constant (`VICTORY_DISPATCH_PLAIN=rootsig`, RT72, at the cost of `all`).
- These did not light it: a full wait before each of them (RT71), a full wait after (RT73), and the parts' records and count without the root constant (RT70).
- A bisect by program (`VICTORY_DISPATCH_PLAIN_ONLY=<hash beginnings>`) is pending.

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
