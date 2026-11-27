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

**VIcky's own commits.** All are opt-in, except the dispatch parts (last item) and the parent-link stores of the builder's box rehydration programs, left out by default since 2026-11-27 (`VICTORY_BVH_LINK_STORES=keep` keeps them; see "The chapter-1 save crash" below).
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
  - `VICTORY_HEAP_PASS_THREADS=<n>` (opt-in, NOT recommended: with 4 the pass halves, 130 to 67 ms, but fps fall, 6.41 / 6.33 to 5.67 / 5.61, as the extra threads slow the title's recording): the pass makes its slots on `n` threads (its own and `n - 1` more, made for the pass), chunks of 64 slots taken in turn. The heap's slot state is held by them together (its read lock); everyone else (title writes, the heap sync, draws, the texture pass) takes it alone as before; each slot's key and heap map entry are atomics only the thread making that slot touches, and the heap map's pool and the cache of made views have small locks of their own. A thread that finds the pool used up gives back no other slot's pool slot: those slots are made after the threads (counted on a line of their own). The results are put in slot order on the pass's thread, so the refreshes, the splice and the join are as before. What may differ: the order the slots are made in, so which pool slot a title slot gets (and, if the pool is used up and too few are given back, which slots go without one), and which records the cache keeps. `VICTORY_HEAP_PASS_THREADS_CHECK=1` (diagnostics) runs one such pass in 16 again on one thread and compares slot by slot (view, copy, range, record made, buffer held).
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
  - `VICTORY_REFRESH_FRONT=1` (opt-in, measured together: "For fps" below; after codec-d/cool_fox dc2a3ad4 and 37332f4d, on the submitting thread): at the start of each part of a submission every copy the part refreshes is brought up to date once, over the span of its refreshes before the part first gives it to shaders (what the batch would merge for it there); the part's refresh steps inside the span are skipped, later ones stay in the batch. It turns on `VICTORY_REFRESH_ONCE`'s rule (a range up to date in a submission is not brought again until a step of ours writes title memory there; title writes of other threads go to the next submission, as with `VICTORY_ASK_ONCE`); each part is brought only after the step before it ran. `VICTORY_REFRESH_FRONT_CHECK=1` (diagnostics) does one part in 16 both ways and counts the bytes that differ.
  - `VICTORY_LIGHT_REFRESH=1|alt` (opt-in, measured together: "For fps" below; fork 94ed363c by fxceless, after avatarmanz 2aa8134b and 74045e46, from its form in `fx-main-forkmerge` 78162a4): a bit per 64 KiB block read-only by our hand and unwritten since; a range whose blocks all have theirs is answered from the journal with no guards' lock and no kernel call, and an update that would copy nothing returns at once. Holes get a bit only where `VICTORY_GUARD_ABSENT_KNOWN` (`=1` or `forever`) answers for them and only in a placeholder reservation (pages come there only through the runtime's told mappings); 16 holes with bits are asked of Windows again per submission, one found filled is said once. The fork's absent-once notes and commit hook are not ported. `alt`: twenty seconds with, twenty without.
- **Cuts to the write-watch asking, ported from other GPL-3.0 trees (2026-11-27; not measured yet).** The submitting thread spent 25-35 ms a frame on the copies' asking of the kernel and the guards and on lookups in the journal (`crates/victory-gpu/src/watch.rs`, `shadow.rs`). Each change is its own commit, credited there.
  - Always on, answers unchanged (a test for each): the journal's asked-ahead ranges are found by binary search over their starts and running furthest end (ScarlettPort 50a9f7c5; no scan, and `ask_once` no longer quadratic in its ranges); `ask_ahead` takes an iterator and builds no range list when the submission's one asking already stands (ScarlettPort 23870346); the flights' written ranges are joined in place (ScarlettPort fbf382d1); the journal's pages are kept in pieces of 64 pages (4 MiB) instead of one B-tree (Biskwit/VIctory-reloaded 27b7a9002: 111 -> 20 ns a look in their test of the two alone).
  - Always on, a crash fix: `title_memory_remapped` forgot the blocks a guard had made read-only without giving them `PAGE_READWRITE` back, so a refresh that guarded a view between its mapping and the runtime's word about it left the title a read-only page with no guard to open it, and its next store was a fatal access violation (fxceless/victory-fork#1447, CwWwWwW; test `a_guard_forgotten_by_the_remap_gives_its_page_back`).
  - Diagnostics, always on: the copies line's "copies split" clause ends with `write watch: N GetWriteWatch calls X ms over N pages, N allocation queries, N journal lookups` (calls of the kernel's asking, failed ones too, per frame).
  - `VICTORY_WATCH_RESERVATION=1|alt` (opt-in, not measured yet; fxceless/victory-fork #1365 by spk-c, after Kelvaroq 1270c42; needs `VICTORY_ASK_ONCE`, on by default): the submission's one asking groups the sorted, deduplicated ranges by Windows allocation (one `NtQueryVirtualMemory` where two ranges could join), one `GetWriteWatch` call for a group inside one allocation, the pages between its ranges noted in the journal; a gap up to the smaller of the two spans is allowed; a group that fails is asked by piece; view memory stays with the guards. If the runtime holds the mapping lock the ordinary asking is made. The coverage is forgotten at a remap, and a write the runtime completed under the asking counts as written after it (`title_memory_written`). The PR's own single sample (10,000 fragmented ranges, 100 reservations): calls 1,307 -> 100, 4 KiB pages asked 3.9 M -> 0.48 M, asking 15.6 -> 8.8 ms; but with ONE reservation and fragmented ranges asking went 5.98 -> 9.43 ms, so use it only where the title's memory is in many allocations. Not ported: the PR's commit hook (we have none) and its stats line (the copies line has the counts). `alt`: twenty seconds with, twenty without, logged `a submission's write-watch asking goes by reservation from now (VICTORY_WATCH_RESERVATION=alt): ON|off, as usual` with the askings, milliseconds, calls, pages and allocation queries an asking of the phase before.
  - Not ported: Biskwit/VIctory-reloaded d4e8d3a45 `VICTORY_FAULT_FAST` (write faults open a 64 KiB block under a per-block gate, not the guards' lock; their run: guard-lock waits -55 %, frame unchanged). It is 560 lines that rest on parts our guards do not have: lock-free bit sets kept beside the guards' `protected` and `aliased` sets (`PROTECTED`, `ALIASED`), the `GuardsHeld` wrapper, a list of faults' openings drained at every holder of the lock, and a gate taken at every site that changes a protection (protect, alias, unguard, cover, publish, hole and remap paths). Our guards keep `protected` in a B-tree behind the lock, so it would be a rewrite of that part, not a port. If the title's job threads' waits at the guards' lock are the next cost, start there (their fault books: wait median 0.1 us, p99 2-4 ms).
- **The heap's slot lock and the fills (2026-11-27, profile PF2; all opt-in; measured together: "For fps" below).** PF2 (`xcl-notes\fable-2026-11-27-pf2.md`): the title's recording threads wait ~194 ms a frame (about 39 ms each) for the host heap's slot state (`HostHeap::made`), 116 ms of it in the mirror of descriptors they copy, while the heap sync's mirroring and the pass over buffer views hold it; the sync spins ~13.5 ms on the one lock of the slots' resources (`kept`); the submitting thread makes 5,500–6,800 kernel copies a frame for fills under 256 bytes (5.4 ms). Judge them by the frame log's `title's turn` (burst, `longest recording thread`) and the host fps lines.
  - `VICTORY_MIRROR_TRY=1` (lever A): a copied descriptor (`CopyDescriptorsSimple`, `CopyDescriptors`) whose heap's lock is held is not waited for: it goes on the heap's list and is made by the next heap sync ahead of its walk, or earlier by the passes over buffer views and textures picked by index (ahead of taking their slots), from its record as it is then. A draw that reads the slot before makes it itself, as any changed slot; views made by calls are mirrored at the call as before. Off with `VICTORY_HEAP_SYNC=0`; a list holds at most 65,536 slots (then the call waits, as before); a slot whose page is gone at the drain is not made. `=check`: made at the call as well, and the drain counts the slots whose record changed in between. Grep the start line `mirrored without waiting for the host heap's lock`, and the end of the heap sync line (`descriptor records the title copied`): `VICTORY_MIRROR_TRY: per frame N copied slots mirrored at the call, N left as the heap's lock was held, N drained by the heap sync, N by the heap passes, N not made as their page was gone, N made waiting as the list was full` (with `=check` also `check: N compared at the drain, N whose record differed from the one made at the call`).
  - `VICTORY_HEAP_KEPT_SPLIT=1` (lever B): the resources of a host heap's mirrored slots and of its dynamic slots (the draws' tables) each behind a lock of their own, and the heap sync's mirroring takes the mirrored part's lock, and that of the slots marked for the pass over buffer views, once per chunk of 64 slots instead of once per slot. Same resources, same order. Grep the start line `VICTORY_HEAP_KEPT_SPLIT=1`; look at the heap sync's `ms of their own` (with `VICTORY_HEAP_SYNC_BESIDE=1`) and its join wait.
  - `VICTORY_FILLS_PLAIN=1|alt` (lever C, port of fork de4146a0 by fxceless; needs `VICTORY_COPY_PLAIN=1`): a list's fills of under 256 bytes are copied in place, the kernel's copy behind them for a page that is not there. Grep the start line `fills of under 256 bytes are copied in place`, and on the in-place copies line (`copies of title memory made in place per frame`) the new `fills of under 256 bytes made in place (VICTORY_FILLS_PLAIN): N` against the `N of under 256 bytes` left to the kernel; `alt` says each change (`from now (VICTORY_FILLS_PLAIN=alt)`).
- **The texture pass and created views off the slot lock, and the sync by place in the burst (2026-11-27, profile PF3; the switches opt-in; measured together: "For fps" below).** PF3 (`xcl-notes\fable-2026-11-27-pf3.md`, the "all five" configuration): the texture pass waits ~77 ms a frame for `HostHeap::made` (one write lock per due slot it makes), views the title makes by calls ~32 ms more (`=1` covered copies only); the submitting thread waits ~30 ms a frame at the heap sync's join. Judge by `title's turn` (`longest recording thread`), the frame line's `heap buffers ... passes`, the pass line (`ms in its pass over textures picked by index`) and the host fps lines.
  - Diagnostics, always on: the `a window miss` line is said one draw in 65,536 (`a window miss (one in 65536)`; it was one in 256, ~28 lines a second in the world, ~27 ms a frame of the recording threads in diagnostics runs); `VICTORY_WINDOW_MISS_LOG=1` says one in 256 as before. The counts stay on `indirect commands per frame by their clamp`.
  - `VICTORY_TEXTURE_PASS_CHUNK=1|alt` (lever 1a): the pass over textures picked by index takes its due slots 64 at a time in the queue's order: the looks before a making slot by slot, then the slots to be made made together under one lock (`HostHeap::mirror_chunk`, as the buffer-view pass), then their uploads. Same slots, same records, same order; the byte and time budget is tested per chunk (up to 63 slots past it). Grep the start line `(VICTORY_TEXTURE_PASS_CHUNK=1)`; with `alt` (20 s off, 20 s on, from the first pass) each change: `(VICTORY_TEXTURE_PASS_CHUNK=alt): ON` / `(VICTORY_TEXTURE_PASS_CHUNK=alt): off`, with `since the change before: N passes, X ms a pass, N slots made, N chunks`.
  - `VICTORY_MIRROR_TRY=2` (lever 1b): as `=1`, and views the title makes by calls (`Create*View`, `descriptor::record`) go the same way (`HostHeap::mirror_created`): left on the heap's list when the lock is held, drained by the next heap sync or heap pass like the copied ones. `=check2`: as `=check` for both (`=check` alone stays copies only). Grep the start line `and the views it makes by calls, are mirrored without waiting` (`VICTORY_MIRROR_TRY=2` / `=check2: ...`), and on the heap sync line after the copies' part: `views made by calls (=2): N mirrored at the call, N left as the heap's lock was held`.
  - Diagnostics, always on (lever 2, step 0; `VICTORY_HEAP_SYNC_AHEAD` itself is not done): a line after the heap sync's, `heap sync by place in the burst: 1st X ms / Y slots / Z ms waited; 2nd ...; 17th and later ...; the present's own ...`, per frame: by the submission's place since the present's own submission, the sync's own ms, the slots it mirrored (with those `VICTORY_MIRROR_TRY` left to it) and the submitting thread's ms waited at the join (the whole sync without `VICTORY_HEAP_SYNC_BESIDE=1`); it ends with the submissions a frame by place. Go on to `VICTORY_HEAP_SYNC_AHEAD` only if the 1st (or 1st and 2nd) carry most of the wait.
- **The heap sync's cost a slot (2026-11-27, profile PF4; the switches opt-in, not measured yet).** PF4 (`xcl-notes\fable-2026-11-27-pf4.md`, `main` 7fbe8ca with the fps set): the frame is the submitting thread's work, and its largest wait is the heap sync's join (~30 ms a frame). The sync costs ~0.66 us a mirrored slot (~96,900 slots a frame, ~64 ms), nearly all bookkeeping: notes of what a slot holds 20.4 ms, locks 13.3, heap map 8.6, kept resources 6.3, the record read again 5.4, loads 5.0; the device calls are ~3.5 ms. Judge in this order: us a slot on the `heap sync by place in the burst` line (ms over slots), `ms mirroring` on the heap sync line (`VICTORY_HEAP_SYNC_THREADS` part), the sync's `ms of their own` and `ms at the join` (`VICTORY_HEAP_SYNC_BESIDE`), then `at work`, the present and the host fps lines; with `=alt`, `xcl-notes\alt-fps.sh`. If the first three move and the join does not, the PF4 model is wrong, not the lever.
  - Diagnostics, always on: the heap sync line (`descriptor records the title copied`) ends with `the sync's slots by kind per frame: N made from the record they hold, N no view of ours; notes by the slot's last kind -> this one: plain->plain N, plain->bare N, plain->other N, bare->plain N, bare->bare N, bare->other N, other->plain N, other->bare N, other->other N`. Only the sync's own slots are counted (not draws, passes or calls). Lever 1's ground is plain->bare, bare->plain and bare->bare (plain->plain is `VICTORY_HEAP_BUFFERS_SKIP_SAME`'s already); if most notes come from or go to `other` (texture views), lever 1 gains nearly nothing.
  - `VICTORY_HEAP_NOTE_KIND=1|alt|check` (lever 1): each title slot keeps what its last note left it as (a plain buffer view's; bare: no view, a constant buffer view or another view neither of a buffer nor of a texture, in none of the sets; other: a texture's). A note between plain and bare does only what differs (the stride, the slot into or out of the buffer slots, the mark for the buffer-view pass), without the locks of the texture queue and of the placed spans; bare to bare does nothing; notes from or to a texture view are full, as before. Restriction: a slot that the texture passes queue again from the spans just while it is noted (they read the spans and fill the queue under two locks) stays queued until its next full note, as with `VICTORY_HEAP_BUFFERS_SKIP_SAME`; `=check` counts such cases. Grep the start line `(VICTORY_HEAP_NOTE_KIND=1)`; on the heap sync line `VICTORY_HEAP_NOTE_KIND: N notes per frame the short way`. `=check` makes every note the full way and looks at the sets first: `VICTORY_HEAP_NOTE_KIND=check (every note, since the last line): N notes from and to plain or bare looked at, N where the short way would have left other sets` (and up to 32 warnings `VICTORY_HEAP_NOTE_KIND=check: the short way would differ`); expect 0. `=alt` (20 s off, then 20 s on, from the first heap sync; the phase changes at a sync's start): `(VICTORY_HEAP_NOTE_KIND=alt): ON` / `(VICTORY_HEAP_NOTE_KIND=alt): off`, with `since the change before: N heap syncs, X ms a sync, N slots mirrored, N notes the short way`.
  - `VICTORY_HEAP_SYNC_LEAN=1|alt|check` (lever 2; only the walk with `VICTORY_HEAP_SYNC_THREADS`, which the fps set has): the sync's chunks of 64 slots make each slot from the record the page's copy held (no second read in place under the lock); the chunk's heap map entries are written at its end after one fence, and the pool slots it gave back go to the pool after a second fence and one count of the submissions started (before a pool slot is taken, what the chunk gave back goes to the pool first, as one by one; the count of finished submissions that frees pool slots is read once a chunk); the keeps (with `VICTORY_HEAP_KEPT_SPLIT=1`) and the marks are gathered in the chunk instead of through a thread-local per slot (made before a `reclaim`, which changes slots at once); the switches and the pool's diagnostics line once a chunk. Same slots, same order, same host heap and map (unit test). Restrictions: a record the title rewrote between the page read and the mirroring is mirrored as the page held it, the newer one by the next sync (its page is marked again); the pool slots given back are quarantined with a count read up to a chunk later (never earlier); calls, `VICTORY_MIRROR_TRY` drains, draws and passes are as before. Grep the start line `VICTORY_HEAP_SYNC_LEAN=1`; on the heap sync line `VICTORY_HEAP_SYNC_LEAN: N slots per frame in lean chunks`. `=check` makes each slot from its record read again in place, as before, and counts the others: `VICTORY_HEAP_SYNC_LEAN=check: N slots per frame in lean chunks, N whose record in place differed from the page copy's`. `=alt`: `(VICTORY_HEAP_SYNC_LEAN=alt): ON` / `(VICTORY_HEAP_SYNC_LEAN=alt): off`, with `since the change before: N heap syncs, X ms a sync, N slots mirrored, N of them in lean chunks`.
  - Predicted (PF4 note, a model): lever 1 notes 20.4 to 10-13 ms, lever 2 -6 to -10 ms of the sync; together with each other the join 31-33 to 21-26 ms, the frame -5 to -10 ms. Run the two `=alt` alone first (one at a time, the other at `1` or off), then `=check` once each.
- **Work off the submitting thread: card copies with changes, holes remembered (2026-11-27, profile PF3 lever 3; both opt-in; measured in one scene with `=alt` on top of the fps set: `CARD_DIRTY_LIST` 0.00 fps, `SPARSE_KNOWN` +0.05 and +0.08 to +0.19 fps (paired medians), so neither is in the fps set).** With the "For fps" set the frame is the submitting thread's work, so each ms of `Host::card_before` (10.75 ms a frame in PF3, 5.5 of it `card_dirty`) and of `read_probing` (7.3 ms) counts. Judge by `preparing` and `at work` on the frame line, the lines below, and the host fps lines; with `alt`, by the on and off phases inside one run.
  - `VICTORY_CARD_DIRTY_LIST=1|alt|check`: a copy in the card's memory goes on a list when a change is noted for it (under its notes' lock, once until taken), and `card_dirty` takes the list instead of walking every card copy at each submission. Same ranges and patches taken, in another order. `check`: the list, then the walk; copies the walk finds with changes and off the list are counted (and taken). Grep the start line `(VICTORY_CARD_DIRTY_LIST=`; on the card line (`host copies in the card's memory:`, every 5 s) the end `copies with changes taken from their list (VICTORY_CARD_DIRTY_LIST=...): N takes since the line before, X us a take, N copies looked at a take, N with something to send` (`check` adds `N copies with changes the list did not hold (check)`: should be 0). `alt` (20 s the walk, 20 s the list): `(VICTORY_CARD_DIRTY_LIST=alt): ON` / `(VICTORY_CARD_DIRTY_LIST=alt): off`, with `since the change before: N takes, X us a take, ...`.
  - `VICTORY_SPARSE_KNOWN=1|alt|check` (lever E of the PF2 note): a host copy of 64 KiB or more that was read around holes keeps them, per (address, length); its next copy copies the runs between them whole (in place with `VICTORY_COPY_PLAIN=1`, else one kernel read a run), tries each hole's first page with one kernel read and zeros the hole. The range goes the usual way (and its holes are found again) when a run does not read, a hole's first page reads, it was not probed for 16 frames (2 such a frame at most), or the runtime mapped, unmapped or reserved memory over it (`title_memory_remapped`, new reservations; a probe that overlapped such a change is not kept). What can be missed: a page mapped in the middle of a hole with nobody telling stays zeros until the range's next probe (16 frames at most). `check`: always the usual way, remembered ranges compared with what it finds. Grep the start line `(VICTORY_SPARSE_KNOWN=`; on the frame line's `sparse reads X ms (...)` the clause `(holes remembered, VICTORY_SPARSE_KNOWN=...: N ranges read around them X ms with N kernel reads, N of them failed; N ranges with holes read the usual way X ms, of them N for their age, N as a run did not read, N as a hole's first page read; N ranges known, N forgotten on mappings, N probes not kept` (`check`: `; check: N compared, N differed`). `alt` (20 s the usual way, 20 s around remembered holes; the holes are learned in both): `(VICTORY_SPARSE_KNOWN=alt): ON` / `(VICTORY_SPARSE_KNOWN=alt): off`, with `since the change before: N ranges read around remembered holes, X us a range; N ranges with holes read the usual way, X us a range`.
- **The snapshots beside the join (2026-11-27, lever L6 of `xcl-notes\fable-2026-11-27-road10.md`; opt-in; measured: two `=alt` runs on top of the fps set, paired medians +0.13 and +0.23 fps, 6 of 10 phases above their neighbours; the submitting thread's snapshots 1.80 -> 1.26 ms a submission, about 5 ms a frame; optional, not in the fps set yet).** PF4: the snapshots of a submission's shader-written ranges are ~12.4 ms a frame of the submitting thread (its share of 177-213 MB of copying), taken just before the heap sync's join: they count in every submission whose join does not wait, and `Host::card_before` (6.7-8 ms) comes after both. Judge by `preparing` and `at work` on the frame line and the host fps lines; with `alt`, by the on and off phases inside one run. Expect the heap sync's `ms at the join after the snapshots` (`VICTORY_HEAP_SYNC_BESIDE` part of the heap sync line) to rise by what the snapshots covered of it before: the gain is the snapshots' ms where the join did not wait, and what now runs beside `card_before`. Predicted (road10, a model): -6 to -10 ms of the submitting thread, less as the join shrinks.
  - `VICTORY_SNAPSHOT_BESIDE=1|alt`: the snapshots are given to the threads kept for wide work where the submitting thread took them (refreshes done, every range declared), and are done just before the submission's lists go to the card (`ExecuteCommandLists`, sent ahead or waited for). They copy beside the join, `make_waiting`, the settle when the card's results run low and the copies in the card's memory. They wait in a queue of their own, which the kept threads take only when the shared one is empty (the heap sync's page reads and a settle's compares go first) and which no caller of the wide threads takes; the submitting thread takes those not started when it comes for them, then waits for the others. Same bytes in the same snapshots: nothing in between writes a range this submission snapshots (a settle's copy-backs write only their own flights' ranges, which `Shadow::writing` left out of this submission's snapshots; `card_before` writes copies in the card's memory, which have none), a snapshot's memory stays until its copy's last declared range is done, and nothing reads the snapshots before the card has run. A panic in between waits for them all the same. Not used with `VICTORY_WIDE_BACK=0` or `VICTORY_KEEP_SNAPSHOTS=1` (a warning says so). Grep the start line `(VICTORY_SNAPSHOT_BESIDE=`; on the recording line (`recording per frame`) `snapshots X ms (VICTORY_SNAPSHOT_BESIDE=1: X ms beside the join, in N submissions a frame, N of them done when the submitting thread came for them; X ms of the snapshots figure is that thread taking those not started and waiting)`: with the switch the `snapshots` figure is the submitting thread's part only (giving them, taking the ones not started, waiting). `alt` (20 s on the submitting thread, 20 s beside, from the first submission): `(VICTORY_SNAPSHOT_BESIDE=alt): ON` / `(VICTORY_SNAPSHOT_BESIDE=alt): off`, with `since the change before: N submissions, X ms a submission of snapshots on the submitting thread, X ms a submission of their copying beside the join`.
- **The heap sync's slots made on several threads (2026-11-27; `VICTORY_PARALLEL_MIRROR=1|alt|check`, opt-in, not measured yet; port of noaclr 589848256d, merged there as 73a8bdf21c).** With the fps set the frame is the submitting thread's work and its largest wait is the heap sync's join (~30 ms a frame): the sync mirrors 93-97k slots a frame at ~0.66 us each (57-64 ms of its own) on one thread (`VICTORY_HEAP_SYNC_THREADS` spreads only the page reading), and the heap's slot state (`made`) is held ~100 ms a frame between the sync and the pass over buffer views (PF4, road10).
  - What it does (`descriptor/host/par_mirror.rs`): the sync gathers the due slots of all of a heap's pages (as `VICTORY_HEAP_SYNC_THREADS` reads them; without that switch the pages are read on the sync's thread first) and, from 128 slots on, makes them in rounds of 2,048: the sync holds `made` alone for the round, as it holds it for a chunk of 64 without the switch, and the round's slots are cut in up to 8 chunks in slot order (as many as the wide threads, the kept ones and the sync's own: 7 on a 24-32-thread CPU) made on threads kept for them alone (`victory-mirror-N`: a caller of the wide threads runs any queued job while it waits, the submitting thread with its locks held) and the sync's, each chunk as the sync's chunk makes it (the usual way, or the lean chunk with `VICTORY_HEAP_SYNC_LEAN`; the notes by kind as before). `made` is sharded like noaclr's: slot i is made with piece i % 64 locked, and no thread holds two pieces; a stale slot (a placed texture view made stale by another's) of the same piece is unmade at once, one of another piece once the thread let go of its own, under that piece's lock and only if it still holds a placed texture view outside the spans (noaclr's check); a slot that comes to hold nothing lets go of its resource at once (another thread may make it again before the chunk's keeps are made); a thread gives back no other slot's pool slot when the heap map's pool is used up: those slots are made once the round's threads are done (`reclaim`, as for the pass's threads). With the switch on, the pass over buffer views gets 8 threads from 256 slots when `VICTORY_HEAP_PASS_THREADS` asks for none, and the pass's threads (either way) make each slot with its piece locked.
  - Benchmark (`par_mirror::tests::rounds_83k`, ignored; diag profile, this machine, 7 threads, 83,000 slots a round, heaps of the tests, not the title's): null views with `NOTE_KIND` 4.8 -> 2.4 ms (1.9x); a mix of buffer, constant, null and texture views changing every round 22.7 -> 18.6-19.6 ms with `NOTE_KIND` (1.15-1.25x) but 26.4 -> 30.7-35.3 ms without (0.75-0.86x); noaclr's constant views without `NOTE_KIND` 15.7 -> 16.3-19.3 ms. The threads' summed time is 5-9 times the one thread's: they take the notes' shared sets (`buffers`, the texture queue, the spans) in turn for every slot, as PF4's profile found the sync lock-bound (notes 33 %, spinning 13 ms). So: only with `VICTORY_HEAP_NOTE_KIND=1` (in the fps set), and the in-game counters decide; next levers if it helps a little: those sets' updates gathered per chunk as the marks are. Already done here: a chunk's views looked up under one read of the table of views (the threads took its lock in turn for every slot), the pieces' locks a cache line each, the notes' counters gathered per thread, the keep-views check skips the table for a slot holding no record (exactly as before: generations start at 1).
  - Differences from one thread: the order in which a round's slots are made (each slot by itself), so which pool slot a title slot past the window takes, what the cache keeps and, of two views of one round that make each other stale, which one stays (the one noted last, in the threads' order). A slot made null may hold its last resource until it is made again. Unit tests: rounds on several threads leave the same state as one thread, title slot by title slot, usual and lean, with and without a heap map and split keeps; mutual stale pairs across pieces and chunks end without deadlock, one view of each pair made; `rounds_83k` (ignored) times 83,000 constant views one thread against the rounds.
  - Grep: the start line `(VICTORY_PARALLEL_MIRROR=1; port of noaclr 589848256d)` (or `=alt`, `=check`); on the heap sync line (`descriptor records the title copied`) `VICTORY_PARALLEL_MIRROR: N slots per frame made on several threads in N holds of the slot state, N chunks run in parallel, X ms of the sync's mirroring wall time against Y ms summed over its threads, N stale slots of other pieces unmade after (N left: made again meanwhile), N made once the threads were done (pool used up); N passes over buffer views on 8 threads`. Wall time well under the summed thread time is the spread working; summed time far above the one-thread mirroring (`ms mirroring` of `VICTORY_HEAP_SYNC_THREADS` in an off phase) means the threads wait for each other's locks (placed spans, texture queue, buffer set).
  - What must move: the join (`ms at the join after the snapshots` of `VICTORY_HEAP_SYNC_BESIDE`, and `ms waited` on the `heap sync by place in the burst` line) and the frame line's `submit thread: at work`; then fps. The sync's `ms of their own` should fall by about the ratio of wall to summed time.
  - `=alt`: `(VICTORY_PARALLEL_MIRROR=alt): ON` / `(VICTORY_PARALLEL_MIRROR=alt): off, on one thread as usual`, with `since the change before: N heap syncs, X ms a sync, N slots mirrored, N of them on several threads` (for `alt-fps.sh`; run it alone, not with another `=alt`). `=check`: one round in 16 is made from records read once before it, then made again one by one in slot order under the same hold; the heap sync line ends with `VICTORY_PARALLEL_MIRROR=check (since the last line): N rounds made again one by one (N left out: a view made another stale), N slots, N whose key that changed`, and the first 32 changes are said (`VICTORY_PARALLEL_MIRROR=check: title slot`). Expect 0 changed keys (a texture view whose texture got its host copy between the two makings would count one).
  - Not done from noaclr's commit: the draws, copies and texture passes still take `made` whole (here `made` also holds the heap map's pool, whose `reclaim` gives back other slots' pool slots, and the read lock of the pass's threads), so the pieces only part the threads that make slots together; noaclr's chunks run on its wide threads, ours on threads of their own (the sync's) and on the pass's own threads (`VICTORY_HEAP_PASS_THREADS`' machinery).
- **The constants heap in place (2026-11-27; `VICTORY_INPLACE_CONSTANTS=1`, opt-in, not measured yet; port of fxceless/victory-fork bcc542151d).** The title's ~174 MB constants heap (one title buffer, at 0x3de00000 in fxceless' runs: the per-draw constants, which the skinning passes also write through one unordered view of all of it) is by default a copy in the card's memory: every submission refreshes it from title memory (write-watch askings, pages copied in), sends the changed ranges up, compares the declared shader-write ranges on the card by hash, and fetches and merges what differs. With the switch, a title buffer of `VICTORY_INPLACE_CONSTANTS_MIN_MB` (64) or more that lies below 4 GiB in view memory, as one run of a section, and every title buffer inside it, become buffers placed in a host heap opened (`OpenExistingHeapFromAddress1`) on a view of our own of the same section pages. The card then reads and writes the title's bytes where the title writes them, and nothing is refreshed, sent, hashed, compared, fetched or merged for it. The risk is the card's reads crossing the bus (fxceless, run 75: a skinning list took 4.5 ms in the card's memory and 50.5 ms over the bus), so the card's time can grow while the submitting thread's shrinks.
  - Preconditions (otherwise the heap stays a copy, and the log says why):
    - The heap one run of one section. Without `VICTORY_PAGE_PLACE=1` that takes `VICTORY_SECTION_MB` unset (one section for each allocation of physical pages, as in fxceless' runs), and even then our runs found it scattered (the boat mission below). With `=2` (the "also recommended" loading switch) the heap's pages lie in some 90 sections of 2 MiB and it is turned down as view memory "from several places or sections"; a larger value holds it whole only if no section boundary falls inside it. **With `VICTORY_PAGE_PLACE=1`** ("Physical pages placed at their first mapping", below) the heap is one run of the one section below 4 GiB, with or without `VICTORY_SECTION_MB`.
    - Not with `VICTORY_FENCE_SENT=1` while submissions are sent ahead (the relay or `VICTORY_LATE_WAIT`). The title is then told a frame begins before the frame before's lists have run, and may rewrite constants those lists still read in place (a copy in the card's memory is refreshed behind them on the queue). The switch is turned down at start with a warning, so measure without `VICTORY_FENCE_SENT` on both sides of the A/B.
    - The heap is opened only while no title buffer in its pages has a copy yet: one memory, one form. If a buffer inside it was copied first, the log says `a title buffer in its pages has a copy already`.
    - `VICTORY_INPLACE=2` must still not be combined with late wait (unchanged; that is the private-memory stage, not this switch).
  - Restrictions:
    - While the heap stands, its pages stay locked in the PC's memory and our view keeps the section alive; it is never unmapped. If the title maps other pages there, the region is forgotten and its buffers let go (not seen in fxceless' runs).
    - There is no `alt` mode. A descriptor slot keeps the form it was made with, so no buffer changes form while the title runs, and `=alt` is turned down with a warning. Compare a launch with `=1` against one without, at equal draws.
    - The card's writes in place are not seen by write watching: another host copy of the same bytes that is not a title buffer (a window, a texture placed there) is not told of them.
  - Indirect executions whose arguments or count are read in place are clamped alone, right before they run (`VICTORY_INPLACE_CONSTANTS_ORDER`, on by default, so it comes with the heap in place; `=0` turns it off; port of avatarmanz/VIctory-7900XTX 631400f598, 3e04ca0afa and 15c3d0ea7e, `VICTORY_PORT_CONSTANTS_ORDER`, found through the review of Grimkaaah's tree). Our `VICTORY_INDIRECT_WINDOW` and `VICTORY_INDIRECT_GATHER` clamp records ahead, in the window's first pass or the list's prelude, so with the heap in place those records can be read before the title's culling passes write them. In avatarmanz' runs, trees and palms vanished for runs of frames and eyes were drawn wrong (pop-5, pop-11 to pop-13), and clamping alone fixed it (pop-14). A batch opened after the last shader that may write buffers was not enough (pop-15), and that is what our windows and gathering already do, so only the alone form is ported. Expected cost: one clamp pass of ours per such execution (fps 0 or slightly lower). Grep the line said at the first such execution, `are clamped alone, right before they run, not in windows or gathered (VICTORY_INPLACE_CONSTANTS_ORDER`, and with the copies lines `  indirect executions with arguments or count in place per frame: N, of them N clamped alone right before they ran (VICTORY_INPLACE_CONSTANTS_ORDER); from N ranges of 1 MiB, the busiest: ...`. These executions no longer count on the windows line. With `=0` a warning says the records may be read before they are written. Look at the vegetation and the eyes when judging the switch.
  - Grep: `the constants heap is read and written by the card in place (VICTORY_INPLACE_CONSTANTS=1)` (start). When the region opens: `a title buffer of the constants heap in place` (with `start`, `bytes`, `mb`: the heap's address and size). Turned down: `a big title buffer not in place` (with `why`), `VICTORY_INPLACE_CONSTANTS=1 turned down` or `VICTORY_INPLACE_CONSTANTS=alt turned down`. Every 10 s: `the constants heap in place (VICTORY_INPLACE_CONSTANTS=1): N regions N MB ...`, with the buffers placed and the refusals by cause.
  - Diagnostics, always on (with or without the switch): with the copies lines, `  the big title buffers per frame (64 MB or more, VICTORY_INPLACE_CONSTANTS' ground; the switch on|off): 0x...+N MB <form>: N refreshes X ms, N MB copied in from title memory; N MB sent up to the card in N ranges, N pieces of 64 KiB compared on the card, N MB fetched back, N MB merged into title memory; in the PC's memory N MB into snapshots, N MB compared, N MB merged`. The form is `a copy in the card's memory`, `a copy in the PC's memory` or `in place`. Read the switch-off runs first: the heap's refreshes ms, MB copied in, sent and fetched are the ground the switch can take away. With the switch on they should fall to ~0 with the form `in place`. Whether that is worth the bus is then decided by the card's time (`on the GPU` on the fps line, `the card's time by what ran`) and the host fps lines.

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

**Chapter-1 boat mission with ray tracing (2026-11-27, verified on an RTX 4080, Series X, RT on):** with this main, a save in the boat mission loads, the boat floats and the mission plays, with no extra switch: the texture copy-back (`VICTORY_TEXTURE_BACK`, on by default) gives the title's physics the water it computes on the card, and the builder's box rehydration programs' parent-link stores are left out by default (`VICTORY_BVH_LINK_STORES=keep` keeps them; with them the title crashed on that save's loading screen). Add `VICTORY_LATE_WAIT=2` in water scenes (at the relay's default stage 3 testers elsewhere saw the water surface vanish for single frames); `VICTORY_LATE_WATER=1` at stage 3 is the narrower wait to try instead, not measured yet (water section below). Measured: 480 s on that save, RT on, the fps set below plus `VICTORY_LATE_WAIT=2 VICTORY_REAPER=1`: no crash, 7.34 fps true mean of the last 12 lines (lines 7.8-8.5, about 122-127 ms a frame: preparing ~68, copy-back ~23-27, a wait for the card of ~17 ms that is the copy-back's read-back). The prologue cell with the same build: 5.32 fps true (a slow stretch mid-run; last lines 6.3-7.5), about level with 6.91 before the batch. `VICTORY_INPLACE_CONSTANTS` does not place the constants heap here ("view memory from several places") even without `VICTORY_FENCE_SENT`/`VICTORY_SECTION_MB`: leave it off, or try it with `VICTORY_PAGE_PLACE=1` and without `VICTORY_FENCE_SENT` (roadmap B2, section "Physical pages placed at their first mapping").

**Lot 5 measured (2026-11-27 night, prologue cell, RT on, RTX 4080):** recommended set plus one `=alt` switch per run. These runs loaded slowly, so only 2-3 ON phases count; treat the medians as indications.

| Switch | Paired median | What the probes show | Verdict |
|---|---|---|---|
| `VICTORY_BUFFERS_STRIPES` | +0.11 fps | P7: `BUFFERS` waits went from 72 ms a frame to 0 (297k striped reads, 2 waits) | optional, harmless |
| `VICTORY_MADE_TURNS` | -0.11 fps | `made` waits went from 165 to 117 ms a frame | leave off |
| `VICTORY_HEAP_SYNC_AT_SUBMIT` | -0.03 fps | 7 of 9 syncs a frame start early and are taken, but the join only goes from 25 to 22-24 ms | leave off |

`VICTORY_PAGE_PLACE=1 VICTORY_INPLACE_CONSTANTS=1` puts the 174 MB constants heap in place. Both runs are without `VICTORY_FENCE_SENT`, which these two need, and are true fps over 12 lines:

| Settings | fps |
|---|---|
| without `FENCE_SENT` | 7.16 |
| without `FENCE_SENT`, heap in place | 6.56 |

The card reading the heap over PCIe costs more than the copies it saves: leave both off.

Best prologue run so far with the recommended set: **8.49 fps true** (L5a). Runs of the same set spread from 7.0 to 8.5 there.

Removing the `BUFFERS` waits did not move the frame. The recording threads' lock waits run in parallel with the submitting thread, so they are not on the critical path today: **the submitting thread's ~137 ms is the wall**.

The BC7 rotation fix (port of fxceless 06702fe8 via trustno1 PR #23 6726920a) is always on.

**Recommended since 2026-11-27 evening (lots 2-3 and the foxhound/trustno1/Belobug port batch; RTX 4080, Series X, RT on):** the fps set below plus `VICTORY_SYNC_FINE=1 VICTORY_CCD_PLACE=1`, and `VICTORY_COMPOSITE_GRADE_EDGE=1` for the picture. In water scenes, use no `VICTORY_LATE_WAIT` and no water set (`VICTORY_WAVE_PAIRS_ONLY`/`VICTORY_LDS_FLAGS` for the water programs).

Measured as true fps over the last 12 frame-time lines:

| Scene | Settings | fps |
|---|---|---|
| chapter-1 boat mission | fps set + `SYNC_FINE=1` + `CCD_PLACE=1` | **9.52** (lines 8.6-10.0) |
| chapter-1 boat mission | fps set only | 8.51 |
| chapter-1 boat mission | fps set + `VICTORY_LATE_WAIT=2` (main of the morning) | 7.34 |
| prologue cell | fps set + `SYNC_FINE=1` + `CCD_PLACE=alt` | 7.27 (CCD ON phases 7.5-7.6, off 6.9) |

Paired `=alt` medians, one switch per run:

| Switch | Scene | Paired median | Verdict |
|---|---|---|---|
| `CCD_PLACE` | prologue | +0.42 fps (+0.59 on top of `SYNC_FINE`) | use |
| `SYNC_FINE` | prologue | +0.35 fps | use |
| `TEXTURE_BACK_ASYNC` | boat | +0.39 fps (noisy, 5 phases) | optional |
| `WIDE_THREADS` | prologue | +0.13 fps | neutral |
| `CARD_AWAKE` | prologue | +0.12 fps | neutral |
| `SPANS_QUICK` | prologue | +0.07 fps | neutral |
| `LATE_WATER` | boat | -0.11 fps | neutral |
| `PARALLEL_MIRROR` | prologue | -0.35 fps | leave off |

Water picture with this build:

- Without any switch, the boat renders whole and the water is right.
- The water set's paired form for `5dc3b25986622949`/`6f225b641c5b0708` turns the water turquoise with grey tiles.
- `926c46bab3b86bd6` with LDS flags `2cc4c197ee29e291` puts bright green specks in shore vegetation.

So leave the set off. The black specks in the sun's glint go away with `VICTORY_COMPOSITE_GRADE_EDGE=1` (see its section; checked in the boat mission, 8.81 fps true on a cache translated the run before).

With `VICTORY_BVH_LINK_STORES=keep`, the boat save no longer crashed on its load.

**For fps, recommended since 2026-11-27:** `VICTORY_REFRESH_FRONT=1 VICTORY_LIGHT_REFRESH=1 VICTORY_HEAP_KEPT_SPLIT=1 VICTORY_FILLS_PLAIN=1 VICTORY_TEXTURE_PASS_CHUNK=1 VICTORY_HEAP_NOTE_KIND=1` on top of the above; `VICTORY_MIRROR_TRY=2` optional (see below). RTX 4080, Series X, RT on, window scene, 420 s, no input, true fps over the last 12 frame-time lines:
- Base (the recommended set above): 6.19 / 6.53 / 6.46 fps (medians of the last 24 fps lines 6.3 / 6.6 / 6.6).
- The five above plus `VICTORY_MIRROR_TRY=2`: 7.04 / 6.83 / 6.60 (medians 6.9 / 6.8 / 6.8), about +0.4 fps (+7 %). The five without it: 6.72 (median 6.8) and, on `main` 6c841fb as pushed (first run of the exe), 7.01 (median 7.0).
- Same scene, `VICTORY_LIGHT_REFRESH=alt` (20 s on, 20 s off, inside one run, the others on): 6.57 fps on against 5.98 off, every on phase above the off phases next to it. Without the title-side relief (`VICTORY_MIRROR_TRY`, the texture pass in chunks, the quieter window-miss line) the same switch measured 5.99 against 5.97: a saving on the submitting thread only reaches the frame once the title's recording threads stop waiting for the heap's slot lock.
- What moves: copies 57-81 ms a frame to 23-29; the title's turn (`title's turn` after each `usual frame` line) 27-58 ms to 3-10; the frame is then the submitting thread's work (`at work` about equal to `in the present call`).
- `VICTORY_MIRROR_TRY` leaves descriptors the title copies (and, with `=2`, views it makes) for the next heap sync when the slot lock is held; `=check` counted 15-31 slots a frame, of 656-897 compared, whose record changed between the call and the sync. No difference was seen in the captures, but it is the only one of the six with that kind of risk.
- `VICTORY_HEAP_NOTE_KIND=1` (morning of 2026-11-27, on top of the five): same-scene `=alt` runs, paired medians +0.26 and +0.16 fps (each ON phase against its off neighbours; 6 of 7 phases above); `=check` found 0 notes of ~480,000 a line where the short way would leave other sets. `VICTORY_HEAP_SYNC_LEAN=1`: +0.09 and +0.11 (one phase of the second run below), and 14-17 slots a frame of ~93,000 made from the page copy's record when the title rewrote it meanwhile (the newer one at the next sync): optional, not in the set. The heap sync's slots by kind (always on): about 18.8k plain->bare, 19.1k bare->plain and 15.5k bare->bare notes a frame, the ground of `NOTE_KIND`.
- Tried the same night and not kept (same-scene `=alt` runs or A/B, fps within the noise or lower): `VICTORY_ASYNC_PRESENT=1` (5.95 / 6.11 / 5.56 against 6.38 / 6.28), `VICTORY_HEAP_WORKER=1` with the pass changes (6.07 / 5.93 against 6.42 / 6.50), one heap sync per burst (same mirroring work, the join wait rose 28 to 44 ms), a heap sync run ahead between submissions (paired -0.07 fps), a list of changed card copies instead of the walk (0.00), and remembered holes for sparse reads (+0.05). The last three are on local branches, not merged.
- Run-to-run spread is large in this scene (same settings: 6.38-7.43 fps), mostly from how much title memory the frame touches; same-run comparisons with the `=alt` switches (fps of the on and off phases added up apart, the first phase after the world's entry left out) resolve about 0.2 fps.
- `VICTORY_REAPER=1` (opt-in, not measured yet; port of Grein a2cf68bb1e via Grimkaaah f2586ef4d7, GPL-3.0): the host lists of landed flights are reset and kept for reuse, and retired objects dropped, on a thread `victory-reaper` instead of on the thread that lands them (`victory-submit` at the end of each submission sent ahead, `victory-late-wait`, the relay). Same work, same lag; only the thread changes. Grimkaaah measured 250 resets a frame, 20 ms of threads' time, -2.3 to -3.8 ms a frame on an RX 7900 XTX; our profile put most resets on `victory-late-wait`, so the gain on the submitting thread is expected small. **What to grep:** the `fps` line now ends its parenthesis with `reaper on|off: submit N lists X ms resetting + Y ms dropping objects per frame; late-wait ...; reaper ...` (per thread, only threads that did any), and the start line `VICTORY_REAPER=1: landed host lists are reset`. A run is useful if `submit` falls and `reaper` takes it.
- `VICTORY_COPY_AHEAD=1` (opt-in, not measured yet; port of Grimkaaah f2586ef4d7, GPL-3.0): a title buffer's host copy is made when the title declares the buffer (`shadow::register`), on a thread `victory-ahead`, instead of at the first view a draw names in it. Same code as the lazy copy (in place, or in the card's memory for big ones), made under the lock of the table of buffers, so the title's threads can wait for it when they declare, release or ask for a buffer. **Costs extra shared memory** (the copies of buffers never drawn from too; 0.7 GB more in Grimkaaah's runs, and our commit limit is tight). Grimkaaah: the world's first frame 5.2 -> 3.8 s (3,399 copies, 1 GB, made while loading), 12.5 -> 13.9 fps in one A/B, 90th percentile 50 -> 59 ms (worse, to watch). Expected here: about 0 in the steady cell benchmark; it should show in the world-entry freeze and streaming hitches. **What to grep:** `VICTORY_COPY_AHEAD=1: a buffer's host copy is made` at the start, and `copy ahead: N host copies of declared buffers made ahead` (logged each time the queue runs empty; copies, MB, ms making, buffers skipped); compare the first frame's length and the memory line with and without.

**Fastest measured (RT81b/c, RTX 4080, before the dispatch parts were merged):** the above plus `VICTORY_HEAP_PASS_BESIDE=1 VICTORY_HEAP_PASS_SPREAD=1 VICTORY_FENCE_SENT_PUT_UP_RUNS=4096 VICTORY_MERGE_RUNS=4096 VICTORY_DENSE_PAGE_WHOLE=1 VICTORY_DENSE_PAGE_PATCH=1 VICTORY_GUARD_SUMMARY=1 VICTORY_GUARD_ABSENT_KNOWN=1 VICTORY_FILL_LATCH_PIECE=1`, plus `VICTORY_HEAP_PASS_ONE_LOOK=1 VICTORY_HEAP_PASS_DUE=1` (RT83b/c, above) and `VICTORY_SPARSE_PROBE=1` (RT84b/c: 6.00 / 6.37 fps, bottom strip fixed). On this branch, with the dispatch parts on (the full-screen pass now runs all its tiles: about 5–10 ms more card time), the same settings give 5.12 / 5.35 fps (RT82c/d); 5.69 / 5.48 was without them. A new build's first run carries the driver's background shader work for several minutes (process user time about 1,500 ms a 5-s line instead of about 900): measure from the second run.

## Lot 4 ports: CPU-sync levers from foxhound, trustno1 and Belobug (2026-11-27, `fx-main-ports4`; opt-in unless said, none measured yet)

Aimed at the RT-on frame (~137 ms of submitting thread: preparing 88, the heap sync's join 23, the copy-back's settle 21). Every new switch is off by default, takes `1` or `alt` (twenty seconds off, twenty on, in turn from the first asking, each change said as `(<NAME>=alt): ON` / `: off, as usual` for `xcl-notes\alt-fps.sh`; run one `alt` at a time). The shared helper is `crates/victory-gpu/src/alt.rs`.

**Correctness fixes, always on:**
- **Dynamic descriptor slots go back only to the host heap that took them** (port of trustno1 `47747d62`, fxceless #1324). A list records its dynamic slot ranges by title heap address and freed them in whatever host heap stood at that address at its reset: a title heap let go before its lists were reset and a new one made at the same address had its slots freed by the old lists (two live lists sharing slots). Each host heap now has an identity (`HostHeap::id`), recorded with the range and checked by `release`. No switch, no log line; test `dynamic_slots_are_given_back_only_to_the_heap_that_took_them`.
- **A raw buffer view's range stays inside its buffer** (port of foxhound `9f8838b1` `raw_view_clamp`, fxceless PR #1266 by lichenlang89, `4be7db46`..`dd5e8c9e`). `raw_at`/`raw_uav_at` (root views with a size) cut the length to the buffer, never move the start, and make a null view for a start past the last dword or off 16 bytes. `VICTORY_RAW_VIEW_CLAMP=0`: the range as asked, as before. Grep `a raw buffer view the title's range did not cover` (first and every 4096th, with `kind`, `cut`, `offset`, `len`, `size`).
- **Console lanes in a group of several host waves take their threads from a group counter** (port of trustno1 `854f57ff` and its test `2dc3df1c`, fxceless #1243, fork commit `c125c0c`). NVIDIA fills 2-D groups from 16 x 2 tiles, so rounding a host wave's first flat id down gave two waves of a 32 x 2 group the same console threads (trustno1's test on an RTX 5070 Ti: 32 of 64 dwords written). Such programs (console lanes, group larger than the host wave, not the halves form) now count their threads in an LDS dword; lanes are placed by rank among active lanes, a partial last wave after the full ones. The halves form uses the same arithmetic (same result while waves are full); our paired form (`twin`) is unchanged. `VICTORY_THREAD_TICKET=0`: the old rounding. **Translator change: the cache key changes, warm the cache again** (`victory translate --warm`; the first run translates everything).

**Opt-in levers:**
- **`VICTORY_TEXTURE_BACK_ASYNC=1|alt`** (port of trustno1 `2d1dc382`, PR #7, there `VICTORY_TEXBACK_ASYNC` on by default; aimed at the 21 ms copy-back settle). The relay's landing thread only copies a copy-back's texels out of its read-back buffer; a writer thread (`victory-texture-back`) waits for the lock against mapping changes, tiles and writes into title memory, newest copy per allocation only. The relay goes on landing the rest meanwhile. Every settle (synchronous submission, present, fence or memory-value wait), the present's wait of `VICTORY_LATE_WATER`, and a submission whose preparing reads that memory (unless `VICTORY_TEXTURE_BACK_AHEAD`) also wait for the writer (`texback::drain`); copy-backs landed by other threads are written in place after the writer's. What changes for the title: a fence or bottom-of-pipe value behind a flight can be seen before its copy-backs are written. `LEAN` works as before. Grep on the fps line's copy-back clause: `(async <mode>: N handed to the writer, N replaced there by a newer one, the writer X ms waiting for its locks and Y ms tiling and writing; N waits for it, Z ms)`; with `=1` at start `texture copy-backs the relay lands are written into title memory by a thread of their own (VICTORY_TEXTURE_BACK_ASYNC=1)`. What must move: `settled for: a synchronous submission or present` ms on the late-wait lines (with `LATE_WAIT=2`) and the copy-back clause's ms.
- **`VICTORY_SPANS_QUICK=1|alt`** (port of Belobug `a284ca74`, on by default there; aimed at the heap sync's join). A write of a heap slot with no placed texture view leaves the spans of placed textures and their lock alone when the slot is not in them (a bit per title slot, kept only with the switch set). Ours already spares that lock for plain buffer views and bare slots (`VICTORY_HEAP_NOTE_KIND`); this spares it for the other views. Compatible with `VICTORY_PARALLEL_MIRROR` (bits change only under the slot's piece lock) and `HEAP_SYNC_BESIDE`. Grep on the heap sync line (`descriptor records the title copied`): `(VICTORY_SPANS_QUICK <mode>: per frame N writes of slots left the spans of placed textures alone, N took their lock)`.
- **`VICTORY_SYNC_FINE=1|alt`** (port of Belobug `a284ca74`; the `VICTORY_HEAP_SYNC_THREADS: N threads` count on the sync line is then the number of jobs). With `VICTORY_HEAP_SYNC_THREADS` of 2 or more, the sync's pages are read in jobs of 2 that the wide threads share, not one job per thread. With `VICTORY_PARALLEL_MIRROR`, a round's slots are made in chunks of 64 that the crew shares, not one chunk per thread. Same reads and makes, other cut. Grep on the heap sync line: `(VICTORY_SYNC_FINE <mode>: jobs of 2 pages, chunks of 64 slots)`, with the `VICTORY_HEAP_SYNC_BESIDE` part's `ms at the join` to compare.
- **`VICTORY_CARD_AWAKE=1|alt`** (port of foxhound `9f8838b1` `host/awake.rs`, there `=1|turns`; the card is ~54 % busy with RT on). A thread of its own sends one group of 64 threads that counts for ~2 ms, over and over, on a compute queue of its own (at most two on the way), while a title submission went out in the last 250 ms, so the driver keeps the clocks up between submissions. Nothing of the title's is touched; it costs a wave or two of the card and power. Grep on the fps line: `card awake (<mode>): N dispatches of its own;`; failure: `the card is not kept awake (VICTORY_CARD_AWAKE)`.
- **`VICTORY_HELPER_PRIORITY=1|alt`** (port of trustno1 `80c70b71`, PR #22; there on by default and measured on an i5-12400F: 8.6 -> 9.6 fps). The submitting thread (`victory-submit`, with `VICTORY_ASYNC_SUBMIT=1`) and the threads it waits for run above normal priority: `victory-wide-*`, `victory-heap-sync`, `victory-mirror-*`, `victory-heap-pass-*`, `victory-relay`, `victory-texture-back`. There the submitting thread was already above normal; here it is raised with them. Each looks at the switch before each piece of work. Grep: `(VICTORY_HELPER_PRIORITY=1)` at start, or the `=alt` lines.
- **`VICTORY_DSTORAGE_PRIORITY=<-2..2>|alt`** (victory-hle; foxhound `9f8838b1` default `-1`, the companion of `HELPER_PRIORITY`): the threads that unpack streamed assets at that priority (`-1` below normal; mind that `1` here is a priority, above normal, not "on"); `alt`: `-1` and normal in turn. Unset or `0`: normal, as before. Grep: `dstorage: workers (VICTORY_DSTORAGE_PRIORITY)` with `priority=`, and `(VICTORY_DSTORAGE_PRIORITY=alt): ON/off`.
- **`VICTORY_WIDE_THREADS=1|alt|<n>`** (port of foxhound `9f8838b1`). Ours keep a quarter of the logical processors for wide work, 2 to 6 (6 on a 9900X); foxhound three eighths, 2 to 8 (8). `1`: foxhound's rule; `<n>` (2-32): that many; `alt`: foxhound's in the phases with it, ours in the others (the extra threads take nothing then); the parallel mirroring's crew keeps our size under `alt`. Grep: `(VICTORY_WIDE_THREADS=1)` or `threads kept for wide work (VICTORY_WIDE_THREADS=<n>)`, or the `=alt` lines.
- **`VICTORY_JOIN_COPIES=alt`** (we had `=1`, the same code as foxhound `9f8838b1`): now also `alt`. Grep `title buffer copies recorded per frame: N, of them N continue the one before (... now on|off|alt, ON|alt, off)`.

**Not ported, and why:**
- **foxhound `VICTORY_RELAY_COPY`** (relay fetches on the copy queue): ours already has it as **`VICTORY_QUEUE2=1`** (bit 1: fetches of card copies' pieces on a copy queue, "sends ahead" from `Second::order_done`, later sends left out by the merge), in every rt-run with `-Fork`. Foxhound's extras (`torn` fetch checks, `had` refetch skipping) belong to its own landing code.
- **Belobug `673be0ea` + `a609a2fc`** (the heaps' round asked at the title's submit, under `VICTORY_PORT_JOURNAL` with the journal in 64 parts, 1,200 lines): built on Belobug's round-based heap-sync thread (`SYNC_ROUNDS`), which ours is not (`HEAP_SYNC_BESIDE` starts one sync per submission after the sweep, counted as started, joined before the next sweep). Since ported as **`VICTORY_HEAP_SYNC_AT_SUBMIT=1|alt`** on the sync beside, without the journal: next section.
- **trustno1 `af242dab` (PR #34) `place_constant`**: it snapshots on the card the constants written earlier in the same list by `place_copy` (title buffer copies done at their place on the card, `COPIES_IN_PLACE` / `CARD_STEPS`), to end `BEHIND_PLACED` refusals. Our tree has neither `place_copy` nor card steps: our title buffer copies are CPU steps at submission, before the constant fills, so the refusal it fixes cannot happen. Our `inplace::place_constants` (`VICTORY_INPLACE_CONSTANTS`) is another thing (the constants heap opened in place) and is unchanged.
- **trustno1 PR #18 `REFRESH_SET`** (`e0703c13`; the author's newer Kemerd/VIctory-NVIDIA-RT main `e7aadc40` has the same code): reviewed in `xcl-notes\pr18-refresh-set.md`; 700-800 lines against our `Batch`, our ~22k refresh steps a frame are already skipped by `REFRESH_FRONT`: measure the skipped steps' cost first.

## The next submission's heap sync while it waits in line: `VICTORY_HEAP_SYNC_AT_SUBMIT=1|alt` (2026-11-27, `fx-main-syncatsubmit`; port of Belobug `673be0e` + `a609a2f`; opt-in, not measured yet)

Aimed at the heap sync's join (roadmap item B1). On the prologue (RTX 4080, RT on) the submitting thread's frame is ~137 ms:
- 88 ms preparing;
- **23 ms waiting at the join** of the heap sync that runs beside it (`VICTORY_HEAP_SYNC_BESIDE=1`);
- 21 ms waiting for the card.

A submission's sync starts when its preparing starts, and the join is the sync's excess over the preparing. The `heap sync by place in the burst` line shows who carries it: the 6th and 7th syncs of a frame, and the present's own, whose preparing is short. Meanwhile the sync's thread idles between one submission's sync and the next.

**What it does.** Used only with `VICTORY_HEAP_SYNC_BESIDE=1` (a warning says so otherwise).
- When a submission's asking of title memory ends (after its steps, `watch::ask_once_end`) and a submission of the title's waits in a queue's line (or the title is in its present), the next submission's sync is given to the sync's thread at once, behind this submission's own.
- It runs while the submitting thread takes the snapshots, sends the lists, lands flights and walks the lines.
- The next submission takes it as its own sync, where the sync beside would start, if it asked for the heaps' pages after everything that submission must see. It is then joined where the sync beside is joined.
- If not, the submission waits for its end and starts a sync of its own, as without the switch.

Code: `crates/victory-gpu/src/descriptor/at_submit.rs` (its doc comment holds the whole argument). Hooks: `host.rs` (`Host::submit`), `queue.rs` (hand-overs, waits passed, `runs_waiting`), `descriptor/beside.rs` (the lane shared, a give without the wait for the asking).

**What a submission must see.** Stamps come from one counter, taken with a full fence.
- The title's records written before it handed the lists over: each host list is stamped by the title's own execute call (`HostList::handed`).
- For a part behind a wait the title's own hand may have met, the records written before our thread passed the wait (Belobug `a609a2f`'s floor per queue). A fence wait, or a memory value found in memory, raises the queue's floor, and the parts taken behind it carry it. A value wait passed for a write of a flight in the air, or of a part joined before it, is the card's order and raises nothing.
- Lists of ours (the present's copy, our checks) carry no stamp: they need a sync asked after the title's latest call, its present included.

A sync ahead is also not taken by a submission that settled flights before its sync's place: a waited-for submission, one whose preparing reads what flights write, or the sweep's forced settle. Those landings come before the sync without the switch. A sweep that is due waits for a sync ahead to end first, and the submission may still take it (sync, then sweep: the order without the sync beside).

**Why not Belobug's whole design.**
- Belobug's sync thread runs numbered rounds asked at any moment. Its journal of written pages was cut into 64 parts with locks of their own, so a round may ask while a submission's asking stands. Ours may not (fxceless/victory-fork#410: a page found written then gets a number above that asking's, and a host copy holding it is copied again at every refresh). Our journal has moved far from Belobug's (light refresh, reservations, guards), so the rework does not port. Here the sync ahead asks only once the asking before it has ended, and the next submission waits for that asking before its own.
- Belobug's rounds may be shared by several submissions. That is `VICTORY_HEAP_SYNC_BURST`, measured on 2026-11-27 (SB1/SB2): the same slots mirrored by fewer syncs that overlap less, and the join rose from 28 to 44 ms. Here each submission keeps a sync of its own, only begun earlier.
- Belobug's `prep_wrote` (our steps writing a heap's memory) is not needed: the steps of the submission before are done when its asking ends, and a submission's own steps follow its sync with or without the switch.
- Unlike `VICTORY_HEAP_SYNC_AHEAD` (an extra sync between submissions, stopped at every submission's start: measured neutral, the stop cost 5.85 ms 4.6 times a frame), nothing stops a sync ahead: it is the next submission's own.

**Same descriptor results.** The slots a submission's lists read are mirrored from records written before its hand-over (and its waits), as with the sync at its start.
- Test `syncs_started_ahead_leave_the_slots_as_the_usual_syncs` (needs a device) runs three submissions over a placed, write-watched sampler heap. It runs them once with a sync at each submission's start, and once with syncs started ahead (taken, not taken, taken). The host slots are the same after each join.
- The other tests check that a sync ahead is taken only if it asked after the hand-over and nothing was settled first, that one not taken ends before the own sync asks, that a sweep waits for it and may still take it, the floors, and the line.

**What changes, and the risk.**
- What the title writes after handing a submission over, and before that submission's own sync would have asked, goes to the next sync instead. That window is at most the few ms between the asking ahead and the submission's start. `VICTORY_ASYNC_SUBMIT=1` already assumes it: title memory a submission reads is not rewritten between its submit call and its preparing.
- The lists of the submission whose asking just ended may meet, between their join and their `ExecuteCommandLists`, a slot mirrored by the next submission's sync. Its record was written after those lists were handed over: a race the title does not run on the console.
- Landings of flights stay unordered with the sync, as with the relay without the switch.

**Compatible with:**
- `VICTORY_SYNC_FINE`, `VICTORY_SPANS_QUICK`, `VICTORY_HEAP_NOTE_KIND`, `VICTORY_PARALLEL_MIRROR`, `VICTORY_HEAP_SYNC_LEAN`: inside each sync, unchanged;
- `VICTORY_MIRROR_TRY`: each sync, ahead or not, drains its lists;
- `VICTORY_FENCE_SENT`: the frame-begin signal is no wait; waits for fences pass on their landing, and raise the floor;
- `VICTORY_CCD_PLACE`, `VICTORY_HELPER_PRIORITY`: the syncs ahead run on the same `victory-heap-sync` thread, placed and raised as before;
- `VICTORY_SNAPSHOT_BESIDE`.

Not with levels on demand (the sync beside is off there).

**Grep.**
- At start: `the next submission's heap sync starts while it waits in line (port of Belobug 673be0e + a609a2f) (VICTORY_HEAP_SYNC_AT_SUBMIT=1)`.
- On the heap sync line (`descriptor records the title copied`), after the `VICTORY_HEAP_SYNC_BESIDE` part: `(VICTORY_HEAP_SYNC_AT_SUBMIT <mode>: per frame N syncs started while the next submission waited in line, N taken by it, N not (N a hand-over or a wait passed after their asking, N flights settled first), N asked with nothing in line; X ms waited for those not taken and before N sweeps; the submitting thread at the join: ON X ms a frame (+Y before) over N frames, off X ms a frame (+Y before) over N frames)`. "Before" is what the submission waited for a sync ahead before its sync's place.
- The sync beside's own part counts the syncs ahead that were not taken among its syncs, with their own time.
- `=alt`: `... (VICTORY_HEAP_SYNC_AT_SUBMIT=alt): ON` / `: off, as usual`, for `xcl-notes\alt-fps.sh`. The stamps are taken in both phases, and a sync started ahead in an ON phase may be taken by the first submission of the off phase after it.

**What must move**, in this order:
1. "taken by it" close to "syncs started", and few "not";
2. the ON join well under the off one, and with it the beside part's `ms at the join after the snapshots` and `ms waited` on the line by place in the burst;
3. `at work` on the frame line, then the present, then the fps (`alt-fps.sh`'s paired median).

The sync's own ms should stay the same: each changed slot is still mirrored once. If many syncs are "not" taken for "a hand-over or a wait passed", the next refinement is the floor of fence waits: it is raised at the pass, as Belobug's, and could be raised to the fence's landing instead. Memory traffic beside the snapshots may eat some of the gain (the `SYNC_BURST` lesson).

## Quiet pages up whole, and three every-build savings (2026-11-27, `fx-main-densefirst`; profile PF5b's lever 3 and D1, D2, D8)

Profile PF5b (prologue, RTX 4080, RT on) put the submitting thread at about 137 ms of a 147 ms frame. This branch takes on the part of it that is the title's small writes under flights' shader writes, and three costs every build pays. `VICTORY_DENSE_PAGE_FIRST` is opt-in and not measured in the game yet; the rest is always on and changes no result.

### `VICTORY_DENSE_PAGE_FIRST=1|alt` (opt-in; needs `VICTORY_DENSE_PAGE_WHOLE=1`)

**The finding.** While flights that may write a card copy are in the air, the title's changed bytes in that copy go up as runs (`Shadow::merge_under_writes`). PF5b's line: about 400 pages looked at a frame, 13,000-15,600 changed runs of about 14 bytes. Each run is a note, a range and a `CopyBufferRegion` of its own ("sent up in 13,768 ranges"). That is most of `card_before` (14.5 ms) and of the look for them (`stale_under_flights`, 12.7 ms; the compare 4.9 of it). `VICTORY_DENSE_PAGE_WHOLE` already sends a page up whole, but only one with more runs than are left (88.6 pages a frame in PF5b, 5.5 MB).

**What it does.** A page whose 64 KiB piece the card's compare has not flagged for `VICTORY_DENSE_PAGE_WHOLE_FRAMES` (64) goes onto the whole-page list at once, before its title bytes are read and its runs counted (`shadow.rs`, `merge_under_writes_as`). It takes one run of what is left (`VICTORY_MERGE_RUNS`), so the pages that are not quiet keep the rest.
- The same rules as `DENSE_PAGE_WHOLE`: only a quiet piece; never a page holding shaders' bytes left on the card (`VICTORY_SKIP_WC_BACK`); put in only if nothing settles the flights (`whole_pages_done`, else dropped and refreshed as any page); queued behind the flights in the air and ahead of the next lists; noted as sent whole, so a landing that finds its piece written by those flights counts it.
- Where the page is put in (`card_put_whole`), its runs are counted against the CPU's side on the title bytes read there:
  - none: nothing goes up, and the page is only taken, as before;
  - fewer than `VICTORY_DENSE_PAGE_FIRST_RUNS` (2 by default): it goes up as its runs, as under the cap (one run is one range either way, with fewer bytes and no shader write gone over);
  - else it goes up whole.
- The CPU's side takes the runs only; the bytes between them are equal already.
- `alt`: twenty seconds off, twenty on, each change said as `... (VICTORY_DENSE_PAGE_FIRST=alt): ON` / `: off, as usual`, for `xcl-notes\alt-fps.sh`. Off phases are exactly the code before.
- Without `VICTORY_DENSE_PAGE_WHOLE=1` (and so the relay) it warns and stays off.

**Grep.**
- At start, with `=1`: `a quiet page of a card copy that the title wrote under flights' shader writes goes up whole before its runs are counted (VICTORY_DENSE_PAGE_FIRST=1)`.
- On the line `title bytes put in under flights' shader writes per frame`, after the `VICTORY_DENSE_PAGE_WHOLE` clause: `quiet pages sent up whole before their runs were counted (VICTORY_DENSE_PAGE_FIRST, on|alt, ON|alt, off): N pages (X MB) holding N changed runs, so N copy regions fewer; N pages with under 2 runs sent as their N runs, N with nothing changed, N found but settled all the same`.
- "Pages looked at" on the same line counts these pages too, so it reads the same in both phases.

**What must move** (ON against off phases):
1. "changed runs" on that line, from 13,000-15,600 toward 1-3 thousand, and the frame line's "sent up in N ranges" by about the "copy regions fewer";
2. `card_before` and `stale_under_flights` in a profile; "at work" on the frame line;
3. the card's send-and-hash lists' time.

Expected bytes up: about 300 more pages of up to 64 KiB, some 15-20 MB a frame more over PCIe (83 MB today).

**The risk: low-medium.** It is `DENSE_PAGE_WHOLE`'s assumption on three to four times as many pages. A whole page takes the title's bytes everywhere in it, also where a flight still in the air has its shaders write a byte the title did not change, if that piece was quiet for 64 frames. The counter `whole pages a flight in the air then wrote (must be about 0)` on the same line covers these pages too; it read 0.00 a frame and 2 since the start in PF5b. If it climbs, raise `VICTORY_DENSE_PAGE_WHOLE_FRAMES` or `VICTORY_DENSE_PAGE_FIRST_RUNS`. A shader write the lists do not declare is not compared, and a whole page goes over it unseen, as with `DENSE_PAGE_WHOLE`.

### Always on, no behaviour change

- **`changed_runs` 64 bytes at a time (D2).** The compare of title memory against a copy (`shadow.rs`: the merges under flights' shader writes, and the refresh of pages holding bytes left on the card) walked each run, and the bytes around it, byte by byte. Now each block of 64 bytes is compared as four 16-byte words (SSE2; plain 8-byte words where there is none), giving a mask with a bit per byte that differs, and the runs are the mask's edges (`trailing_zeros`). A run open at a block's end goes on into the next; the rest after the last whole block is compared in a block padded with equal bytes.
  - A plain 8-byte word scan was tried first, starting each load where the last `trailing_zeros` pointed. It was 0.5-0.9x the byte loop: on short runs every load waited for the one before. Here the loads of a page do not depend on each other.
  - Same runs, byte for byte (tests below). Micro-benchmark (Ryzen 9 9900X; the code of both, extracted into a standalone `rustc -O` program, best of 5-7 rounds), ns per 64 KiB page:

| Page | before | now | |
|---|---|---|---|
| runs of 10-18 bytes, 60-140 apart (PF5b's shape, 567 runs) | 3,745-3,875 | 1,974-2,242 | 1.7-2.0x |
| runs of 1-3 bytes, 4-8 apart (RT66d's dense table, 8,165 runs) | 18,138-24,521 | 10,628-12,267 | 1.7-2.0x |
| runs of 10-18 bytes, 1,000-3,000 apart (32-33 runs) | 2,409-2,598 | 1,095-1,439 | 1.8-2.2x |
| no change | 1,498-1,506 | 895-996 | 1.5-1.7x |
| PF5b's shape, both sides out of the caches (128 MiB walked page by page) | 14,004-14,016 (4.7 GB/s) | 5,427-5,469 (12.0 GB/s) | 2.6x |

  The cold row is the one that matches PF5b: the pages looked at are nearly whole 64 KiB pages (PF5b's 88.6 whole pages a frame were 5.54 MB), so the compare reads some 23 MB a frame, not the profile note's 1.5 MB, and 4.9 ms for that is 4.7 GB/s, the "before" figure. So the compare should fall from about 4.9 to about 1.9 ms a frame. In the repo: `cargo test -p victory-gpu --profile diag --lib changed_runs_bench -- --ignored --nocapture` (hot rows only).
- **`plaincopy`'s counters per thread (D1).** The copies-in-place counters (`VICTORY_COPY_PLAIN`: copies, bytes, pages not there, left to the kernel, fills; and the writes back) were nine adjacent shared atomics. They were written twice per copy by the submitting thread, the relay, the pass over buffer views and the wide copy threads at once: `atomic_add` in `plaincopy::copy` was 2.0 ms a frame of the submitting thread. They now use lot 4's per-thread blocks (`crate::perthread::Counters`) and are summed at the line. The totals are the same, add for add, and the line's text is unchanged. Micro-benchmark (same program; `perthread.rs` itself; two adds per copy, wall time per copy):

| threads copying at once | shared atomics | per thread | |
|---|---|---|---|
| 1 | 7.25-7.32 ns | 1.08-1.48 ns | 4.9-6.8x |
| 2 | 19.2-20.1 ns | 1.18-1.24 ns | 16x |
| 4 | 40.8-41.4 ns | 1.21-1.33 ns | 31-34x |
| 6 | 64.3-65.7 ns | 1.39-1.42 ns | 45-47x |
| 8 | 98.5-103.7 ns | 1.56-1.62 ns | 61-67x |

- **The CCD placement's thread walk off the present (D8).** With `VICTORY_CCD_PLACE=1` (in the recommended set; the switch works in diagnostics builds only, the ones our fps runs use), every twenty seconds `lot4::ccd::apply` took a snapshot of every thread of every process, then a handle and a description per thread of ours. It did this on the title's render thread inside the present: 0.35 ms a frame on average in PF5b, so some 50 ms in one frame every twenty seconds.
  - Now it runs on a short-lived unnamed thread of its own; the `=1` first line and the `=alt` phase lines are said there, once the threads are placed, with the same text.
  - The walk also comes before the lock of the registered threads, so a recording thread registering meanwhile does not wait for it.
  - The same placement, a few ms later. If a placement is still running when the next is due (twenty seconds on, never expected), the new one runs where it is asked, as before.

### Left as they were

- **D6, the heap sync spinning on its mutexes** (`descriptor/host.rs`, `note_buffer_into`; `Mutex::lock` 8.2 ms own). Not local: it is the descriptor code another branch is changing now (`fx-main-syncneed`), and lever 1 takes the sync's time off the critical path.
- **D7, two clock reads per refresh in `Batch::update`** (`host.rs`, 0.25 ms). They feed the frame line's "copies" time in every build; removing them changes the line.
- **`plaincopy::faulted_lately` reads the clock per copy** (0.15 ms). Reading it once a submission would change when the faulted blocks are forgotten.
- **D3** (the 13,500 small copies) is what `VICTORY_DENSE_PAGE_FIRST` is for. **D4** (`made`) and **D5** (whole windows snapshotted) are levers 2 and 5, not waste to cut in place.
- **The diagnostics-only costs** (the frame lines printed inside the present, the world captures' compression, `cardmeter`, `copystat::range`, the clock reads around steps, the counting allocator): about 6-8 ms on the serial path of our measuring builds, the same in both phases of an `=alt` run. None was trivial to move.

### Tests

`cargo test -p victory-gpu --lib` (debug):
- `shadow::merge_tests::changed_runs_match_the_byte_scan_at_every_alignment`: one run of every length at every place, in slices of 0-40, 63-65, 127-129, 192 and 200 bytes, both sides at all eight alignments in memory (two of them for the longer slices); then two runs with every gap of 1-70 bytes, the first starting at every place of a block, joined over gaps as `VICTORY_MERGE_GAP` joins them. All against a byte-by-byte scan.
- `shadow::merge_tests::changed_runs_match_the_byte_scan_at_random`: changes from every second byte to one in 64, slices of every length 0-200 at every pair of the eight alignments, and whole 64 KiB pages.
- `shadow::merge_tests::block_masks_agree`: the SSE2 mask and the plain words' agree for every byte alone with each kind of change (low bit, high bit, all bits, ...) and on 20,000 random blocks.
- The existing `changed_runs_are_exactly_the_bytes_that_differ` and every test that goes through `changed_runs` (relay merges, patches, fence-sent rules).
- `shadow::card_merge_tests::a_quiet_page_goes_up_whole_before_its_runs_are_counted` (on a real card copy and write-watched title memory; three pages: a dense table, one byte, the same bytes again). Without the switch, and with it but no piece quiet, the dense page settles and nothing is put in. With every piece quiet: no settle, no runs counted, three pages left to go up. Put in: the dense page goes up whole and is noted as sent whole, the one-byte page as its run, the unchanged page not at all; all three are taken, and the CPU's side holds title memory.
- `plaincopy::tests::counts_of_every_thread_are_taken_once`: six threads' adds reach the taken totals once each, each in its own place.

The suites, each by its own exit code: victory-gpu 463 passed, 0 failed, 20 ignored (`warp_budget_saturates_without_wraparound_and_preserves_draw_fields` passed in this run); victory-shader 256, victory-hle 115, victory-loader 30, victory-formats 27, none failed; `cargo xtask policy` ok; the diag exe builds (`cargo build --locked --profile diag -p victory --features diagnostics`).

## Water and boats: textures shaders write are copied back (on by default, `VICTORY_TEXTURE_BACK`)

- **What.** A list whose shaders write (through unordered access) a texture of one of four kinds copies it into a read-back buffer when it closes; once the list has run, the texels are tiled into the texture's title memory in the console's layout. The kinds (width x height, DXGI format): 256x256 R32_FLOAT (41, the ocean's height map), 256x256 R32G32_FLOAT (16, its displacement map), and the min/max planes 64x64 R32G32_FLOAT (16) and 128x128 R16G16_UINT (36).
- **Why.** The title's CPU physics samples those textures; boats float on them. On the console the CPU and the GPU share memory, so the compute passes' stores are there for the CPU. Here they stayed in the host copy, nothing copied them back: in our run S2B the boat of the first boat mission sat in the shallows and the mission failed. It is not NVIDIA-specific: upstream sinks on AMD too (Marvisak/VIctory#2248). With the copy-back, testers of fxceless/victory-fork (thread #863) report the boat afloat on RTX 3070 to 5090 and an RX 9070 XT, Series X and S.
- **Where from** (`texback.rs`, `host/texture_back.rs`, victory-gpu and victory-formats only: no translator change, no cache rewarm):
  - Geckomacabre's vx-layer `eee5667d`, ported as fork PR #999 by cxy0072019-sketch with tickets on each copy's title allocation (skipped if the memory was replaced, remapped or released; merged by fxceless as `f5c03f3b0f`);
  - fork PR #1376 (spk-c, `86d446b74f`): a stale target is registered again at the next write in the same list instead of the read-back being dropped;
  - on by default as in TaynanGT's fork (`23925a4aa2`);
  - fork PR #1359 (fxceless): read-back buffers kept by power-of-two size and used again (one committed buffer per copy-back was ~770 a second; it ran Linux/NVIDIA out of DMA mappings, fork #1340);
  - fork PR #1331 (cxy0072019-sketch): the one 256x256 R32_FLOAT plane in the render-XOR layout `SW_64KB_R_X` is tiled too (it was refused once a run, `a texture copied back could not be tiled e=Unsupported`); layouts still not addressable are skipped before the copy;
  - the state-restore part of fork PR #1170 (BradRoland): the texture is left in the state the list's recording left it in, not in COPY_SOURCE.
- **Switches.** `VICTORY_TEXTURE_BACK` unset or `1`: the four kinds; `0`: off; or a list `<width>x<height>f<format>,...` (the fewest that float: `64x64f16,128x128f36`, but then most of the water around the boat is not drawn; keep all four). Mind the value when following other trees: in vx-layer `=1` was an empty list (off), in fork #999 unset was off. `VICTORY_READBACK_POOL=0`: a buffer made per copy, as before; `VICTORY_READBACK_POOL_MB` (256): the most bytes kept spare.
- **What to grep** (strip ANSI colour codes first, field values are coloured):
  - once, at the first texture bound for unordered access: `textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK) kinds=[(256, 256, 41), (256, 256, 16), (64, 64, 16), (128, 128, 36)]`, or with `=0` `no texture shaders write is copied back into title memory (VICTORY_TEXTURE_BACK=0)`;
  - `texture readback landed copied=N bytes=N buffers_reused=N buffers_made=N`, at the first landing and every 128th (`buffers_made` should stop growing);
  - `skipping texture readbacks whose title allocation changed skipped=N` (powers of two);
  - warnings that should stay absent: `a texture copied back could not be tiled`, `water readback skipped before GPU copy: unsupported console layout`.
- **Cost, measured elsewhere, not on this tree yet.** CQ187901, RTX 4070: about 39-43 ms a frame, with 52-61 MB a frame copied back against 30-43 without (part of it is more geometry once the hull sits in the water). CuteBooty: 350-374 textures (102-108 MB) per 5 s, 40-230 ms of CPU per 5 s. Here the landing runs where the other read-backs land: on the relay's thread (`VICTORY_RELAY=2`, the default), in the late-wait settle, or on the waited path; the recording side adds a barrier pair and the `CopyTextureRegion`s at each such list's close. Only lists that write these four kinds pay.
- **Constraint: `VICTORY_LATE_WAIT=2`.** At stage 3 the water surface vanishes for single frames (CuteBooty, fork #863): with the present no longer a settle point, a copy can land during the title's next frame; `VICTORY_LATE_WAIT=2` cured it. **Our recommended sets above do not hold it:** they leave `VICTORY_LATE_WAIT` unset, and with the relay on (the default unless `VICTORY_RELAY=0`) `late_stage` in `host.rs` then picks stage 3. For water scenes add `VICTORY_LATE_WAIT=2` (the relay stays on; the present settles the flights again, an fps cost not measured here). The `(VICTORY_LATE_WAIT=2)` label on the `title bytes put in under flights' shader writes` line is printed at stages 2 and 3 alike, so it does not tell them apart; the per-frame `settled for: a synchronous submission or present` count does (about 0 at stage 3, as in S2A; at least 1 a frame expected at stage 2).
- **`VICTORY_LATE_WATER=1|alt`: stage 3 kept, only the copy-backs waited for (opt-in, not measured yet; `host/late_water.rs`).** In the boat scene stage 2's settle at the present cost 14-29 ms a frame, median 23 (V4, 10th to 90th percentile of its late-wait lines, about one a second: `settled for: a synchronous submission or present 1.0`, all of the frame's waiting for the card). With the switch, at stage 3 (`VICTORY_LATE_WAIT` unset, relay on), the title's present, once the frame is all sent (what the queues still held forced out, the present's copy sent as a flight), waits until no flight that copies textures back is in the air or being landed, and returns. Flights land in order: that is the frame's last such flight and everything before it; the frame's tail behind it and the present's copy land beside the next frame, as at stage 3.
  - Why that should be enough: stages 2 and 3 differ at the present only. What lands there that the title reads without a wait we can see is the copy-back: its CPU reads the maps, while our own reads of that memory (a texture, a title buffer copy, a fill) already wait for its flight, as its ranges are among the flights' written ranges. On the console the card's stores are in the CPU's memory as soon as it has run them, and a frame's early passes (the ocean's compute) have run long before the next frame reads them, fence or not; the relay keeps the order of fences and labels at every stage. With `VICTORY_FENCE_SENT=1` the next frame's beginning is published after this wait.
  - CuteBooty tried a wait of this kind in fork #863 (6-14 ms a frame, the dropouts rare but not gone, more with `VICTORY_ASYNC_SUBMIT=1`; where it stood is not said). Ours comes after the present's forced run of what the queues still held, which a wait at stage 2's first settle point would miss.
  - Not covered: `VICTORY_ASYNC_PRESENT=1` (the present returns before the frame is sent), `VICTORY_WAIT_FORCE=0`. Without the relay (`VICTORY_LATE_WAIT=3 VICTORY_RELAY=0`) a present with such a flight in the air settles every flight. At stages 1 and 2 it has nothing to do (said once at start).
  - With `VICTORY_TEXTURE_BACK_AHEAD=1` (copy-back section below) our own reads of that memory no longer wait for the flight: they take the last landing's texels. `VICTORY_LATE_WATER` still makes the present wait for the copy-backs, for the title's reads; the two switches are independent (both opt-in), so set them one at a time when measuring.
  - **What to grep:** at start `presents wait until the flights that copy textures back into title memory have landed, and only for them (VICTORY_LATE_WATER=1; alt: in turn)`; on the fps line, after the reaper and texture copy-back clauses, `; late water (VICTORY_LATE_WATER=1): N settles a frame it caused, X ms a frame waiting; N presents a frame, N with flights copying textures back in the air`. `alt` (20 s off, 20 s on): `(VICTORY_LATE_WATER=alt): ON` / `(VICTORY_LATE_WATER=alt): off`, each with `since the change before: N presents, N of them with such flights in the air, N waited for them, X ms waiting in all`; the clause then says `alt, ON now` or `alt, off now`. Its waits also count on the late-wait line as `settled for: a synchronous submission or present` (between 0 and 1 a frame with it; 1.0 at stage 2) and in that line's ms.
  - **To verify (boat scene, `fx-run\master-scene2`, the fps set plus `VICTORY_REAPER=1`, no `VICTORY_LATE_WAIT`):** (1) with `=1`, the surface is drawn in every capture (`VICTORY_CAPTURE_WORLD_EVERY=64`: a captured frame is itself presented the old way, which settles, but what it shows was drawn after a present that did not). One frame in 64 is seen, so look through the whole run's set (and a video if one can be had). (2) fps against `VICTORY_LATE_WAIT=2` on the same save (true mean of the last 12 lines), and the clause's ms against stage 2's 14-29 (median 23). `=alt` compares on and off within one run, but the off phases may blink. (3) If the surface still blinks with `=1`, the title reads something of the frame's tail as well: note it here and keep `VICTORY_LATE_WAIT=2`.
- **A cheaper copy-back (2026-11-27; two opt-in switches with `=alt`, not measured yet).**
  - **Where the time goes, from the logs we have** (V2 prologue and V4 chapter-1 boat scene of integ-1127, RTX 4080, RT on, against MP2 = main 05060cf without the copy-back):
    - Prologue: the copy-back adds no wait for the card. V2 landed 384 to 511 copies in all: the 384th 47 s after the first, fewer than 128 in the 2 min 45 s after. The settles for `title memory read ... a texture` come in bursts with the copy-back and without it: 0.14 a frame (38 of 291 lines) in V2, 0.12 (32 of 284) in MP2. `on the GPU` over the gameplay lines is 7.3 ms in V2 against 12.1 ms in MP2. The ~20 ms in V2's last lines is one of those bursts; MP2 has them too (15.9 and 19.5 ms).
    - Boat scene (`VICTORY_LATE_WAIT=2`): `on the GPU` is 15-25 ms a frame in its last minute. Nearly all of it is the present's settle (`settled for: a synchronous submission or present 1.0`): stage 2 waits there for the card and for the flights' landing. The landing includes the copy-back's tiling and writes, about 16 a frame (`texture readback landed` every 128 copies about once a second at 8 fps). The read settles there are 0 to 1.1 a frame.
    - So the copy-back costs: the stage-2 present settle that water scenes need, the landing's CPU time inside it, and now and then a submission whose preparing reads memory a copy-back in the air writes.
  - **Always on (no switch): port of TaynanGT `f0019f2e20`** (Taynan Piza, GPL-3.0), its texture half. Each thread that lands copy-backs keeps its buffer for the title's bytes (a new one of up to 512 KiB was made at every landing) and a `TilingCache` (victory-formats) of the column runs and row offsets of the mips it tiled. Its other half (`land_card` with joined protected ranges) has no counterpart here: our relay merges each fetched piece on its own, with binary searches.
  - **`VICTORY_TEXTURE_BACK_LEAN=1` (or `alt`)**, exact for the title:
    - a copy that a later copy of the same memory in the same flight supersedes is not landed (a texture written by several lists of one submission was copied and landed once per list);
    - texels title memory holds already are not written again (the write opens the guards over them and makes textures placed there due again).
    - Nothing tells the title anything between the read-backs of one flight's lists (its bottom-of-pipe writes and the signals behind it come after all of them), so it reads the same bytes at the same moments. It shortens the landing, and at stage 2 the present's wait with it.
  - **`VICTORY_TEXTURE_BACK_AHEAD=1` (or `alt`)**, a trade-off:
    - the memory copy-backs write no longer holds back a submission sent ahead whose preparing reads it (a texture of the title's in the same memory, a buffer copy, a fill): it goes without waiting for the flights in the air;
    - such a read takes the last landing's texels, older by the flights in the air (about a frame; on the console a pass after the compute pass reads what it wrote). If it reads while a landing writes, it gets a mix until the next landing writes again.
    - Only what the card draws from title memory can lag. Where these textures are sampled through their own host textures, nothing changes.
    - It does not remove the present's settle at stage 2.
  - **What the boat relies on, and why both keep it afloat.** The title's CPU physics reads the textures' title memory. A copy-back still lands before the fence signals and bottom-of-pipe writes behind its flight are set, so a title that waits for its frame's fence before it reads still finds the texels there. LEAN only leaves out copies no one could observe. AHEAD only changes what our own preparing reads, never when a copy lands. All four kinds stay copied:
    - the min/max planes alone float the boat (three testers in fork #863);
    - the 256x256 maps are read by about 30 places in the title's code (vx-layer's read watch), and without them most of the surface around the boat is not drawn;
    - `VICTORY_TEXTURE_BACK=64x64f16,128x128f36` is therefore not a saving to take.
  - **What to grep** (strip ANSI codes):
    - the `fps` line's parenthesis now ends with `texture copy-back: N copied X MB, N landed X MB in X ms (lean off|on|alt, ON|alt, off: N superseded, N unchanged), N settles for a read of its memory, X ms waiting for the card (ahead ...: N such reads went ahead);`, per frame (absent until the first copy is recorded);
    - with `=alt`: `texture copy-backs land lean: ... from now (VICTORY_TEXTURE_BACK_LEAN=alt): ON` / `: off, as usual`, and `submissions that read memory texture copy-backs in the air write go ahead without waiting for them: ... from now (VICTORY_TEXTURE_BACK_AHEAD=alt): ON` / `: off, as usual`. The first 20 s are off.
  - **How to measure:** one `alt` switch at a time (their phases would coincide). In the boat scene with `VICTORY_LATE_WAIT=2`:
    - LEAN: `landed ... ms`, `superseded`, `unchanged`, `on the GPU`, and fps, ON phases against off phases;
    - AHEAD: `settles for a read of its memory` and its ms in the off phases against `such reads went ahead` in the ON phases, and fps.
    - Look at the water's surface in the captures of both phases. The boat must stay afloat in both.
- **Not covered.** The water's picture (colour, dark tiles: the paired wave64 form of `5dc3b25986622949` and `6f225b641c5b0708`, a translator change, next section) is separate from buoyancy. Later crashes in the same mission (title `+0x3de6491` and others, fork #1209) have no known fix.

## Water picture: the paired wave64 form for the water's programs (opt-in per program, not measured in the game yet)

**What is wrong.** With ray tracing on (RTX 4080, 32-lane waves) the chapter-1 water floats the boat but its sun glint has black specks. Run V4-integ-scene2 (`d23f864a`, ANSI stripped, `refused` / `dropped` / `does not translate`):
- The driver refuses seven compute programs translated as `WaveSize(64)` (`host pipeline refused error=... (0x80070057)`), and their dispatches are dropped every frame they come (`dropped 1.0/frame without a host pipeline: refused [compute ...]`):
  - the water's transmission volumetrics, `5dc3b25986622949` (`water_shading:CS_TransmissionVolumetrics_Tiled`) and `6f225b641c5b0708` (`..._Tiled_LocalLights`), 45 drop lines each, from the boat scene on;
  - the reflection tiling `2cc4c197ee29e291` (`sr_tracing:CS_Tiling_Generate_Opaque`), 66 drop lines;
  - four that are not water: `02384a60fda6f2d0` (`vfx_triangleClip:CS_DefragHeap_ComputeRelocations`), `6b82bc7d22016cb5` (`vfx_triangleClip:CS_CollectTriangles_Meshlets`), `2da4beb386509b77` (`alphaResolve:CS_AlphTransmissionPrefilterMip`), `369cbe906ea72764` (`GpuParticleUpdate:CS_Visibility`).
- `926c46bab3b86bd6` (`water_shading:CS_TransmissionVolumetricsSubmerged`) does not translate (`a lane mask holding lanes elsewhere read as data`, at 0x13c8) and is dropped too (0.2/frame, 2 lines).
- Why they keep 64 lanes (`VICTORY_WAVE_WHY=1`, offline): LDS in a group over 32 lanes, read and written in one lane-branch region; for the two water programs every other form stops on a wave reduction or a first-lane read where not every lane runs, for `2cc4c197` on a lane mask exchanged where not every lane runs.

**What is added (opt-in, for the listed programs only; with no switch set no translation changes, checked offline below):**
- **The paired form** (`emit/paired.rs`, `emit/mod.rs`'s `paired`): one host wave of 32 threads runs one console wave of 64 lanes, each thread lanes n and n + 32; every per-lane branch is taken by the whole wave, so every thread runs every block and the group's barriers meet; ballots, wave reductions and prefixes, `v_readlane`, `v_readfirstlane`, `ds_bpermute` (within its 32-lane half) and the LDS accesses (bounds checked, a group barrier before each one but a read right after a read) see all 64 lanes. From fork PR #1397 (ByFxbian, after fork PR #854 by PiloteCode and darkworldhm's `840243e2`; merged by hand in fork `b1d777a807`).
- **Ordered FP32 reductions** (`emit/paired_fp32.rs`): a whole-wave float sum, minimum or maximum is replayed step by step in the console's order on the bits (flush to zero, signed zeros, NaN priority), so the result is the console's bit for bit; an FP64 sum of one constant is counted exactly. Fork PR #1419 (ByFxbian, fork `8a32d0ba17`) with fxceless's `3ab5c12ba3` (only MODE writes that touch FP32 rounding or denormals stop it; the water programs set `hwreg(MODE, 2, 2)` only) and `e3307c7727` (exact FP64 constant sums). It needs the guest float mode confirmed: without `VICTORY_WAVE_PAIRS_FP32=non-ieee-rne-ftz` the three water programs are refused in the paired form (`paired FP32 reduction requires a confirmed non-IEEE RNE/FTZ guest mode`) and keep their old outcome. (The fork also writes such programs without fast-math flags; our modules never carry any.)
- **LDS accumulators set at the start** (`emit/lds_init.rs`, fxceless `a918163849`): an LDS dword every write sets to one constant and every atomic treats as its identity (a tile's starting minimum and maximum, which one thread stores after the per-pixel early returns) is also stored by every thread first thing, behind a group barrier, so a tile whose corner thread returned early does not combine with stale LDS (the fork's "flickering dark tiles"). In the fork for every compute program; here only in the paired form. `VICTORY_LDS_INIT_HOIST=0` turns it off. It changes the three water programs' translations, not `2cc4c197`'s.
- **LDS flag regions** (`emit/lds_flags.rs`, fxceless `c005a245f0`, after DrTardis1963 `1b5c3a1e`): a lane-branch region that only raises an LDS flag may run as two host waves of 32. In the fork on for every program (over its whole pack it changed only `2cc4c197`); here per program.
- **Switches.** Each listed program's shader cache key takes its decision (and, in the paired form, the two settings); the lists are in no key, so changing a list translates again only the programs it adds or removes:
  - `VICTORY_WAVE_PAIRS_ONLY=<hash>,...`: these programs try the paired form first, else the forms they had (unset or `0`: none);
  - `VICTORY_WAVE_PAIRS_FP32=non-ieee-rne-ftz`: confirms the guest float mode for the ordered FP32 reductions;
  - `VICTORY_LDS_INIT_HOIST=0`: no accumulator stores at the start in the paired form;
  - `VICTORY_LDS_FLAGS=<hash>,...` (or `all`, the fork's default): LDS flag regions as two host waves.
- **Suggested for a first water run:** `VICTORY_WAVE_PAIRS_ONLY=5dc3b25986622949,6f225b641c5b0708,926c46bab3b86bd6 VICTORY_WAVE_PAIRS_FP32=non-ieee-rne-ftz VICTORY_LDS_FLAGS=2cc4c197ee29e291`. (`2cc4c197` also translates in the paired form; the LDS flag form is the one on fork main, for every program, since 2026-11-26.)

**Offline checks** (`victory translate --validate --pipelines` on the title's `shaders.rpf`, `VICTORY_HOST_WAVE=32 VICTORY_VIEW_LIMIT=1000000`, the RTX 4080's driver for the pipelines; no game run):

| program | `d23f864a` | with the switches above |
|---|---|---|
| `5dc3b25986622949` | `WaveSize(64)`, validates, driver refuses | paired form, validates, pipeline created |
| `6f225b641c5b0708` | `WaveSize(64)`, validates, driver refuses | paired form, validates, pipeline created |
| `926c46bab3b86bd6` | does not translate (mask read as data) | paired form, validates, pipeline created (0.9 MB of DXIL: expect a long first compile) |
| `2cc4c197ee29e291` | `WaveSize(64)`, validates, driver refuses | `VICTORY_LDS_FLAGS`: 32-lane waves (local waves, the exact idiom form, without early-outs), validates, pipeline created; in the paired form too |
| `02384a60`, `369cbe90` | `WaveSize(64)`, refused | unchanged: the paired form refuses their LDS operations |
| `2da4beb3` | `WaveSize(64)`, refused | unchanged: several waves per group would have to meet after a divergent exit |
| `6b82bc7d` | `WaveSize(64)`, refused | unchanged: over 32 KiB of LDS with the paired form's counter |

**Which translations change.** With none of the four switches set: none. Over the title's whole shader pack (`shaders.rpf`: 26,503 containers, 26,084 programs, same host settings), `d23f864a`'s translator and this one emit the same DXIL, byte for byte, for all 25,636 programs that emit, and refuse the same programs for the same reasons. With the switches: only the programs the lists name, as in the table.

**What to grep in a run:** `WAVE64-PAIRS-ONLY #<hash> translated in the paired form (VICTORY_WAVE_PAIRS_ONLY)` once per listed program when it is translated (or `paired form refused: <why>; the other forms`); no `host pipeline refused` and no `dropped ... refused [compute <hash>]` for the four hashes; `first bind ... hash=5dc3b25986622949` still there in the boat scene. With `VICTORY_WAVE_WHY=1`, `WAVE64-LDS-FLAGS ... leaks=[]` for `2cc4c197`.

**Not measured in the game yet.** Fork reports: with the two water programs paired and the FP32 switch, "the water has its green colour" (fxceless); the LDS-init fix for the dark tiles was not yet confirmed in a game run there. Nobody has run `926c46bab3b86bd6` paired that we know of. Costs to watch: a paired group has half the threads, each doing two lanes' work, with group barriers around LDS writes; what these programs cost per frame is not known here. Compare a boat-scene capture with `d23f864a`'s (the specks in the glint, the water's colour) and the fps lines.

**The translator's source changed, so the shader cache key changed for every program: warm the cache again** (`victory translate --warm` with the run's switches, lists included, since the listed programs are filed under their own keys; or one rebuild run) before measuring.

**Not ported:** the fork's global `VICTORY_WAVE_PAIRS=1` (the paired form as a retry for every program), `17533c5c6d` (`VICTORY_WAVE_PAIRS_SPILLED`, for the structure builder's spilled programs, which the paired form refuses here), #1397's loop-exit regions for the wave-wide form (`VICTORY_WAVE_WIDE=1` only), and the fork's hardware fixtures (`wave64_after_loop.rs`).

## Lot 4 probes (`VICTORY_LOT4_PROBE=1`, opt-in)

**Results (2026-11-27, prologue, RT on, one probe run):**

| Probe | Result |
|---|---|
| P0 | Reserved buffers up to 16 GiB work. One tile mapped at several offsets reads and writes the same, including shader stores, for heaps in the card's memory and in the PC's (with or without a CPU mapping). Tiles from a heap opened over host memory remove the device. `UpdateTileMappings` costs 1.7-2.7 us a range. Recreate-At-GPUVA is tier 0. |
| P2 | 48.7 % of 167.7k slots a frame are buffer or constant views inside live copies, under the 60 % bar for a physical-page mirror. |
| P4 | Placing pages at first mapping would make 14.4 % of the refused window bytes one run, but 100 % of the constants-heap-sized buffers (238 MB). |
| P7 | Per frame, the lock waits summed over threads are `made` 165 ms and `BUFFERS` 72 ms (289k takes, a quarter waited); only 5-16 % of takes cross CCDs. |
| P8 | Dead: the copy-queue pull is not cheaper for the typical run. |
| P3, P6 | Sampled nothing on this scene. |


Two measurements that decide whether lot 4 builds a physical-page mirror: the title's 64 KiB frames as tiles of one reserved buffer, each mapped at every offset the title maps it (`crates/victory-gpu/src/lot4/device.rs`; the host side's P1-P7 under "Lot 4 probes (host side)" below, same switch). Off (unset, or anything but `1`), they cost one load of a flag per refresh and per run copied. On, P0 adds a second or two to the device's creation (six short child processes) and P8 works one frame in 64: **leave it off for fps runs.** Grep `lot4 probe` (strip the colour codes first).

- **P0, once, when the device is made** (`lot4 probe P0: ...`). In the run's own process, questions only: the tiled-resources tier and GPU address bits; `Recreate At GPUVA: tier N` (asked only, or `not known to this runtime`); the runtime (`D3D12Core.dll` path, SDK version) and driver; `QueryVideoMemoryInfo` budgets, local and non-local; `reserved buffers: 2 GiB ok, 4 GiB ..., 8 GiB ..., 16 GiB ...` (made and let go).
  - Every tile mapping runs in a child process on the same adapter (`victory.exe` again, `VICTORY_LOT4_CHILD=<mode>`): a mapping into a heap opened over host memory removed the device before, the software adapter lost its device to the large aliasing test, and a child loses only its own. Its lines start with the mode: `[card-heap]` one tile of a heap in the card's memory at two offsets of a 4 MiB buffer; `[card-heap-big]` in the largest reserved buffer, `UpdateTileMappings on a direct queue, N ranges ...: 1: X us CPU (...), then Y us until the queue had done it; 64: ...; 1024: ...; 16384: ...`, then one tile at several offsets (4 GiB and beyond when the buffer reaches there) read by copies and written by a compute shader (`yes`/`NO`); `[pc-heap-cpu]` a heap in the PC's memory the CPU maps, `[pc-heap]` one it does not; `[opened-cross]` and `[opened]` a heap opened with `OpenExistingHeapFromAddress1`, with a cross-adapter reserved buffer and a plain one. Each ends `VERDICT <mode> WORKS` or `FAILS: <why>`, then how the child ended.
  - Last: `VERDICT: ... -> the mirror ALIVE` or `DEAD: <why>`, and `; limited: <why>`. Dead: no reserved buffer, one tile not read the same at two offsets of the small buffer, or 20 ms or more for 4,096 ranges. Limited: buffers only under 4 GiB, the large buffer failing where the small works (smaller arenas), no tiles in the PC's memory (VRAM only), tiles only from heaps of ours (pages still copied in), a shared budget under 12 GB.
- **P8, one frame in 64** (`lot4 probe P8: sample K (frame F): ...`): the frame's CPU copies into host copies are timed. Of those refreshed from write-watched memory, the one that copied most and is private memory committed whole is opened in place (the in-place path's call) and its runs are pulled again by a copy queue of ours, one `CopyBufferRegion` a run, into a buffer of ours of the kind the CPU copies into. The line gives (a) today's CPU copy in ms (and its refreshes with the asking), (b) the pull's CPU ms to record and submit and its GPU ms by timestamps, the ratio, then `running medians ... (a)/(b) Rx -> PASS` or `-> DEAD (not 2x cheaper in CPU ms)`, and the frame's runs and MB from watched and unwatched memory.
  - Read the running medians of the last sample. Dead also if the frame's runs from write-watched memory exceed about 2,000 (the line says so). The heap's opening is given apart: a kept design opens a region once.
  - `verify (once): N of N bytes pulled equal the title's bytes` checks the pull. `no region to pull: ...` says why a sampled frame had none.
  - (b) writes a buffer of ours, not the host copy (a copy queue must not write what the direct queue may use, `queue2.rs`); for a copy in the card's memory it would also skip today's second send, so (b) is conservative there. The heap over the title's pages stands until the pull lands (milliseconds); meanwhile the title cannot free or protect them.

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

### The chapter-1 save crash at `+0x108557`: the builder's parent-link stores, left out by default (2026-11-27)

**The crash.** With `VICTORY_BVH_BUILDER=1` (RTX 4080, 32-lane waves), loading the user's chapter-1 save crashed the title at `+0x108557`, reading `0xffffffffffffffff`, about 130 s after `the ray-tracing structure builder runs` (run S2A). fxceless/victory-fork#16 (follow-ups 13 and 14, RTX 4090) saw the same kind of crash: the object read was overwritten with builder data a few KiB past a rehydrated structure, and only the parent-link stores of two rehydration programs can write at an offset that far, since each takes it from a child pointer. Analysis: `xcl-notes/fable-2026-11-27-rehydrate.md`.

**The runs on that save:**

| switch | stages run | crash |
|---|---|---|
| none (S2A) | all | yes |
| `VICTORY_BVH_SKIP=47-50` | rehydration skipped | no |
| `VICTORY_BVH_SKIP=49-50` | the two box rehydration programs skipped; every copy number used up in the builder's first frame, as in S2A | no |
| `VICTORY_STORE_DROP=caa78b8c:b44+b90+bcc+c18+c54+ca0+cdc+d28,28918b71:904+950+98c+9d8+a14+a60+a9c+ae8` | all, without the 16 parent-link stores of #caa78b8c (16-bit boxes) and #28918b71 (32-bit boxes) | no |

**The translation is not where the pointers go wrong** (as far as two checks can tell):
- Read through: the two programs were translated offline as a run translates them (`set_host_max_wave(32)`, same sizes as the run's `first bind` lines) and disassembled with `dxc -dumpbin`. The 64-lane bit count of `EXEC`, the wave sum and both exclusive prefix sums (`wavePrefixOp` per host wave plus the lower half's total, through the LDS exchange), the partial-group mask (`USER_DISPATCH_THREADS`) on every sum, ballot and store, the child select loop and the link address `c1 + ((((child << 3) & !63) - c2) >> 4 & 0x0ffffffc)` all give what the console's code gives.
- Run on the card: both translations, run in place on the RTX 4080 (roots 1 and 2 on one buffer, as the title binds them) over synthetic compressed data, 512 groups, last groups of 64, 40, 33, 32, 31, 17 and 1 threads, 0–4 children per thread of kinds 0, 4 and 5, four data sets each: every node's child words and every parent link exactly as the console's code writes them, nothing else in the link table. 56 runs, 0 differences. (A scratch test that reads the program from the title's files, so not in the tree.)
- The meetings of the two host waves are group barriers with a shared-memory fence only (mode 9, no UAV fence), and these programs rehydrate in place, so the D3D12 memory model would let a host wave's node stores pass the other half's loads of the same bytes. On this card that did not happen in the 56 runs.

**So the most likely cause is the data the programs are given:** a compressed region whose host copy is not the title's at that moment (streaming reuses addresses: a structure rehydrated earlier at the same address reads as compressed data of f16 bounds and node words, which is what follow-up 14 found in the stray dwords), or an earlier stage's output not yet there. Not proven.

**The workaround, on by default.** `per_program::BVH_LINK_STORES` leaves those 16 stores out of the two programs' translations, through the `VICTORY_STORE_DROP` machinery (each program's shader cache key names its dropped stores). `VICTORY_BVH_LINK_STORES=keep` keeps them; `VICTORY_STORE_DROP` still adds its own. The log says it once: `VICTORY_BVH_LINK_STORES: the parent-link stores of #caa78b8c and #28918b71 are left out`.
- What is lost: the parent links of rehydrated structures. A refit walks up them; traversal does not read them. A rehydrated structure the title later refits would refit from whatever those dwords hold.
- What it does not fix: nodes built from wrong child pointers are still written, inside the structure (rays may miss or hit wrong there).
- **The translator's source changed, so the shader cache key changed: warm the cache again** (`victory translate --warm`, or one rebuild run) before measuring.
- To verify in a run: S2A's switches (builder on, no `VICTORY_BVH_SKIP` or `VICTORY_STORE_DROP`), the save restored: the line above in the log, no exception at `+0x108557` for 300 s or more after `the ray-tracing structure builder runs` (two runs, as for any "no crash"). Then once with `VICTORY_BVH_LINK_STORES=keep`: it should crash as S2A did.
- Next, to find where the pointers go wrong: log, for one stray link (its address outside the structure), the dispatch's root address and the group's region the program read (4 KiB, 8 KiB for 32-bit boxes), from the host copy just before the dispatch and from title memory at the same moment.

### Is the builder's rehydration input the title's bytes? `VICTORY_BVH_INPUT_CHECK=1` (diagnostics builds, off by default)

The two box programs match the console on synthetic data, so the bad parent-link addresses most likely come from bad input. This switch compares, for each dispatch of the four rehydration stages, the bytes their root views read with title memory.

**Run it.** Build `cargo build --profile diag -p victory --features diagnostics`. Run as in "Run" above (builder on) with the save restored, and add:

- `VICTORY_BVH_INPUT_CHECK=1`
- `VICTORY_BVH_LINK_STORES=keep` to see the crash path (the 16 parent-link stores kept; the title crashes on the chapter-1 save's loading screen). Without it (the default) the stores are left out and the title survives: the check works the same either way, since it reads the input and not the stores.
- Optionally `VICTORY_BVH_COPY=skip`, `VICTORY_LATE_WAIT=2 VICTORY_REAPER=1` as in the runs above.

It reads and compares up to 4 MiB per range, twice per dispatch, so frames get slower while it is on. Off, nothing changes.

**What is compared.** The programs: 4ddd6f9702fab2a7 (DecompressHeader), 5e68767498c57eb6 (LeafRehydration), caa78b8ca62bc3d6 (Box16Rehydration), 28918b7150a7aad5 (Box32Rehydration). Per dispatch, each distinct address of their root views (roots 1 and 2 are normally one address, shown as `u1+u2`; `s` is an SRV, `u` a UAV), from the address to the end of its host copy, which is what the view reads. Two reads of the host copy's bytes against title memory:

- at recording: before the list's refresh of the copy has run, so a difference is normal when `refresh_queued=true`;
- at submission: after that refresh, which is what the card will read.

**The lines.**

- `VICTORY_BVH_INPUT_CHECK: a dispatch of a builder rehydration program reads a range whose host copy differs from title memory (the first 32 such dispatches)`: fields `frame`, `program`, `name`, `groups`, `ranges` (compared), `bad_ranges`, and `detail`, one entry per bad range, joined by ` | `:
  `u1+u2 at 0xADDR (copy N[ in place| in the card's memory], 0xLEN bytes): at recording X of Y bytes differ, first at +0xOFF; at submission X of Y bytes differ, first at +0xOFF; refresh_queued=true|false pending=0xB pages_never_taken=P/Q taken_asking=LO..HI asking_now=N`
  - A range is bad when bytes differ at submission, or at recording with no refresh queued in the list.
  - `first at +0xOFF` is the offset from the range's address. `X bytes equal` when none differ.
  - `copy N` is the host copy's own number (`Shadow::id`, not the 1,023 title-memory numbers).
  - `refresh_queued`: a refresh covering the range is queued in this list. `false` means nothing in the list brings the copy up to date before the dispatch.
  - `pending`: bytes of the range an earlier submission not yet copied back may be writing. There the copy is newer than title memory, so a difference is not a stale copy (the programs rehydrate in place, so one dispatch's output is the next one's input).
  - `pages_never_taken=P/Q`: pages of the range the copy never took from title memory. `taken_asking=LO..HI`: the lowest and highest write-watch asking that took the others (the age: compare with `asking_now`, `not asked ahead` when the range is not in a range asked ahead).
- `VICTORY_BVH_INPUT_CHECK, since the last line: dispatches of the builder's rehydration programs (...)`: at the first checked dispatch of a frame, for the frames since the last line (the first 200 lines, then every 100th): `dispatches`, `dispatches_with_bad_range`, `ranges`, `differ_at_recording`, `differ_at_submission`, `differ_at_recording_no_refresh_queued`, `submission_mib_compared`, `submission_bytes_differing`, `ranges_with_pending_writes`, `views_without_copy`. Submission counts lag a frame or two.

**Reading it.**

- `differ_at_submission` stays 0 and no detail lines: the card is given the title's bytes at submission; look at the title's memory changing between recording and submission, or at the program's own output.
- Details with `refresh_queued=false` and `pending=0x0`: a range the dispatch reads that nothing refreshes: the stale-copy cause.
- Details with `pages_never_taken` above 0: the copy never took those pages (they read as the title's zero or stale bytes).
- Details with `pending` above 0 only: the earlier stage's own output not yet copied back, not stale input.
- A `first at +0x...` offset tells where in the 4 MiB the first bad byte lies; with `VICTORY_BVH_LINK_STORES=keep` compare it with the group's region (4 KiB of input per group for 16-bit boxes, 8 KiB for 32-bit boxes).

## Lot 4 probes (host side, 2026-11-27)

Branch `fx-lot4-host-dev`. One counting build for lot 4's decision (`xcl-notes\lot4-plan-2026-11-27.md`; the probes as section D of `fable-2026-11-27-sources.md` lays them out). The device probes P0 and P8 ("Lot 4 probes" above, `fx-lot4-probe-dev` merged here) read the same switch: one run with `VICTORY_LOT4_PROBE=1` gives all of P0-P8.

### The diagnostics counters are kept per thread (always on, a fix)

The fps runs use the diagnostics exe, whose counters were arrays of adjacent shared atomics: eight to a cache line, each add a locked add from the title's recording threads, the heap sync, the pass and the submitting thread at once (`copystat`'s 128 counters, `framelog`'s totals, the heap's per-slot `FAST` counters, the pass's `HEAP_PASS_FAST`). Worst of all, the diagnostics allocator added two locked adds on one shared line for every allocation and every release, on every thread. The frame log's "longest recording thread" was one shared word compared and swapped at each list call of the leading thread and read at everyone else's.

`crate::perthread` gives each thread a block of its own (aligned to 128 bytes, written with a plain load and store, no locked instruction); the log lines sum the blocks. Totals and log text are the same as before, add for add (the sums wrap, so a release on one thread of a block made on another still nets out). A thread gives its block back when it ends, to the next thread that starts (the counts stay in the block and the sums; the pass over buffer views starts threads of its own at each pass, so blocks cannot be handed out once and for all); with more than 512 threads alive at once the rest share one more block with locked adds as before (P7's line says how many hold one now, at most, and how many found none free). Tests: `perthread::tests::sums_match_under_several_threads` (12 threads, adds and subtractions across threads), `takes_while_adding_lose_nothing`, `blocks_come_back_when_threads_end` (2,000 threads one after another, none shares a block, sums kept), `each_thread_sets_its_own`.

Micro-benchmark (`cargo test -p victory-gpu --profile diag --lib perthread::tests::bench -- --ignored --nocapture`; each thread adds to four adjacent counters in turn, ns an add, wall over adds per thread):

| threads | shared adjacent atomics (before) | per-thread blocks (now) | |
|---|---|---|---|
| 1 | 4.24 ns | 1.16 ns | 3.7x |
| 2 | 13.25 ns | 1.09 ns | 12.2x |
| 4 | 31.77 ns | 1.44 ns | 22.1x |
| 8 | 74.09 ns | 2.01 ns | 36.9x |
| 12 | 110.28 ns | 2.66 ns | 41.5x |

(Ryzen 9 9900X, `--profile diag`, 4,000,000 adds a thread. The 12-thread case spans both CCDs.) A counted lock site with the probes off, against the plain lock (`lot4::locks::tests::bench_off_against_plain`, one thread, 20,000,000 locks): 7.55 / 7.61 ns and 7.60 / 7.57 ns, the same within the noise.

Expected in a run: a few ms a frame less on the threads that count most (the recording threads' 41,000 list calls, the sync's and pass's 87,000 + 67,000 slots, every allocation); not measured in a game run yet.

### The probes: `VICTORY_LOT4_PROBE=1` (off by default)

Each says one line with the frame log's reading (every five seconds), per frame over the reading's frames (P6: per sampled frame). Off, every hook is one load of a flag (and in builds without `diagnostics` the hooks are compiled out). With the probes on, the counting itself costs some time (P2's lookup per slot, P3's compares on the relay, P6's protections one frame in 64): use the probe run for counts, not for fps.

| probe | grep | what it counts | kill |
|---|---|---|---|
| P1 mirror size | `lot4 probe P1:` | 64 KiB blocks under live host copies of buffers and windows (census thread every reading): in card / PC memory, over XMem private memory / views of physical pages / host addresses; blocks found written (copied in from title memory, `Shadow::copy`) per frame and distinct over the reading | with P0: over ~3 GB and no PC-memory tiles, or "host addresses" over 20 % |
| P2 slot share | `lot4 probe P2:` | of the slots the heap sync mirrors (the title's own descriptor calls left out) and the pass over buffer views makes: buffer views whose range lies wholly in P1's blocks (the last census's), constant buffer views likewise, texture views, others, none; count and % | **under 60 % of slots: the mirror is not worth weeks** |
| P3 replaceable snapshots | `lot4 probe P3:` | snapshot bytes and blocks taken (`Shadow::writing`); at each landing (`shadow::find_all`), one 64 KiB block in 16 of the declared ranges compared with title memory before the shaders' bytes go in: equal (title memory still is the before-image) or not; the differing ones per frame (scaled by 16), with or without a difference in the 64 frames before. A sampled block that an earlier landing wrote after its snapshot is not judged (title memory then holds that landing's bytes, the snapshot older ones; C3 could still use title memory there), only counted | **under 80 % of the bytes equal: idea 3 is dead**; differences without history over ~20 a frame: too many settles |
| P4 lazy page placement | `lot4 probe P4:` | simulated beside the real table: each physical page gets its place at its first mapping (its address; a page whose address another took first goes elsewhere, and keeps its place when mapped again). For the live windows over view memory and the title buffers of the constants heap's size (64 MB, `VICTORY_INPLACE_CONSTANTS_MIN_MB`) and more: bytes one run today / scattered / with holes, and of the bytes not one run today those the simulated places make one run; the three largest buffers listed. Behaviour unchanged | **under half of the bytes not one run today: idea 4 is dead** (the constants heap alone becoming one run keeps a narrow one alive) |
| P5 alias census | `lot4 probe P5:` (two lines) | physical pages mapped now by how many addresses show them (1, 2, 3, 4+), over all and over the blocks under live copies, with the bytes shown at more than one address; sections of physical pages; the title's reservations (`XMemVirtualAlloc` at an address of its choice) by page size, write-combining and uncached, since the start | context |
| P6 lazy write-back | `lot4 probe P6:` | one frame in 64 (half way between P8's samples, on the same frame count, never while P8 records), each block a landing writes into title memory (buffers' write-back, texture copy-back) is made no-access right after the write and kept so through the next frame; faults open it again and are counted (read or write, by the title's code or the runtime's image); the runtime's own kernel reads and writes (`copy_mapped`, `read_into`, `write_mapped`, `watch::writing`) open their range first and are counted apart; handler µs a fault, and an estimate with a 64 KiB fetch each. Only committed read-write private memory at console addresses that no view shows is sampled. `VICTORY_LOT4_PROBE_FAULTS=0` leaves P6 out | context (over ~300 blocks read a sampled frame: lazy read-back is dead) |
| P7 locks | `lot4 probe P7:` | per lock, per frame: taken, taken after a wait, ms waited, % taken on the other CCD than the lock's last taker, tries that found it held: `made` alone / shared (the pass's threads) / its pieces (`VICTORY_PARALLEL_MIRROR`), the table of views (read, written), `BUFFERS`, `placed`, `kept`, the heap's buffer slots, `changed`, `textures`, `next` (dynamic slots), the guards, the made views' cache; the CCDs found and the placement's state. Since `fx-main-locks` also `BUFFERS` read on stripes and written (changes no longer count in `BUFFERS`), the windows (`LOOSE`: alone, on stripes, written), `made` handed over between a chunk's slots, and who reads the tables of copies (section "Lock contention" below) | every lock under 10^4 a frame or under 20 % cross-CCD: lock bouncing is not a cost |

How P7 counts: a lock is tried first and only one held elsewhere is timed, so an acquisition that does not wait reads no clock. The last taker's CCD is one word per kind of lock (all heaps' `made` share one), written only when it changes.

### `VICTORY_CCD_PLACE=1|alt` (P7's switch, off by default; works without the probes)

The CCDs are read at start from the L3 caches (`GetLogicalProcessorInformationEx`, `RelationCache`), not assumed; the start line names them (`CCD 0 (L3) processors 0-11 ..., CCD 1 ... 12-23`). With the switch, as CPU sets (`SetThreadSelectedCpuSets`):
- on the CCD holding most of processors 0-15 (where the title pins its own threads): the submitting thread (`victory-submit`, and any thread that submits), the heap sync (`victory-heap-sync`), the parallel mirror's threads (`victory-mirror-*`), the pass's threads (`victory-heap-pass*`) and the title's recording threads (every thread that makes a list call);
- on the other: the bulk copy threads (`victory-wide-*`, `victory-relay`, and `victory-texture-back` with `VICTORY_TEXTURE_BACK_ASYNC`).

A thread the title pinned to processors of the other CCD only keeps its pinning (a hard affinity wins; the lines count those as "pinned away"). Off, each placed thread gets back the CPU sets it had. `=alt`: twenty seconds without, twenty with, from the first frame; each change is logged as the other switches' (`threads placed by CCD from now (VICTORY_CCD_PLACE=alt): ON, ...` / `: off, ...`), so `xcl-notes\alt-fps.sh` reads it, and the line also gives what the phase before took: frames, ms a frame, the submit thread at work, recording on all threads, the longest recording thread, and the ms waited for `made` (with `VICTORY_LOT4_PROBE=1`). Kill: no phase differs by more than 3 % on "at work" or the recording sum.

### What to grep in the probe run

`lot4 probe P` (seven kinds of line), `lot 4 probes on` at the start, `threads placed by CCD` with the switch. Read them in this order: P1 and P2 together (does the mirror pay?), P3 (idea 3), P4 (idea 4), P7 and the `=alt` phases (CCD placement), P5 and P6 for context.

## Lock contention: `VICTORY_BUFFERS_STRIPES` and `VICTORY_MADE_TURNS` (roadmap A1 and A2, 2026-11-27, `fx-main-locks`; opt-in, not measured in the game yet)

P7 (`VICTORY_LOT4_PROBE=1`, prologue, RTX 4080 + Ryzen 9 9900X, per frame summed over threads): `made` 7,758 taken, 982 after a wait, 165 ms waited; `BUFFERS` 289,420 taken, 72,824 after a wait, 72.3 ms waited; `kept` 136,402 / 10,841 / 8.9 ms. Only 5-16 % of the takes cross CCDs: the cost is the contention. Both switches are off by default, take `1` or `alt` (twenty seconds as before, twenty with the switch, from the first asking; each change logged as `... (<NAME>=alt): ON` / `: off, as usual`, which `xcl-notes\alt-fps.sh` reads), and change no result, only who waits for whom.

### Who takes `BUFFERS`, and why

`BUFFERS` (`crate::shadow`) is the table of title buffers by (start, resource), each with its host copy once made. Every lookup of a host copy takes it (`shadow::find`, `find_any`): the walk to the innermost buffer holding the asked range (a `BTreeMap` range, back over at most 64 entries), the copy made on first use, a reference count taken. A lookup the table does not answer (memory outside buffer objects) then takes `LOOSE`, the table of windows, which P7 did not count before. The table changes only when the title makes or releases a buffer (`register` / `unregister`, from `resource.rs`), with `VICTORY_COPY_AHEAD` and, for copies in place, `in_place_let_go`.

The takes of the probe run, by caller (from the same run's lines):
- **The pass over buffer views** (`victory-heap-pass`, beside the submission): ~58,000 a frame (`57683 host copies looked up` on its parts line), through `descriptor::host::Lookups` (`make_slot`, `shown_copy`, `buffer_copy_with`), each under `made` in the pass's chunks of 64: the lookups are the largest part of the pass's held time (`host copies looked up 14.8` ms of the chunk's parts).
- **The title's recording threads**, the rest, ~230,000 a frame over ~41,000 list calls: the draws' and dispatches' root views (`draw.rs` `root_view`, `root_uav`, `root_srv_table`, and `host_word`'s view bias, which looks the same address up again), vertex and index buffers (`prepare_for`, `find_any`), indirect arguments and counts (`execute_indirect`), shader pointers' pages (`memory.rs` `map`) and the builder's inputs (`bvh_input`, `bvh_copy`).
- **The heap sync** takes none: it makes no buffer view (a view not for a draw is left to the draw that reads it).

What a lookup holds the lock for: the walk and an `Arc` clone, about 0.1-0.2 us on 6,000 buffers (the bench below, one thread). It waited about 1 us each time it waited (72.3 ms / 72,824): many short holds on one line, from a dozen threads on both CCDs. What it writes: nothing of the table, except the copy made on first use; the reference count it takes is on the copy, not the table.

### `VICTORY_BUFFERS_STRIPES=1|alt` (A1): the tables of host copies read on striped locks

`crate::stripes::Striped`: a table behind a whole lock and 32 read locks (stripes), each on two cache lines of its own.
- A lookup with the switch on reads the table on its thread's stripe (`crate::perthread::slot() % 32`, the block the per-thread counters already give each thread): readers on different stripes touch no line in common, and none waits for another.
- With the switch off (unset, or the off phases of `alt`) a lookup takes the whole lock alone, as the one mutex was taken.
- A change of the table takes the whole lock, then (whenever the switch is set at all) the 32 stripes' write locks in order: it waits for the readers of each stripe, and none comes in until it is done. With the switch unset no reader takes a stripe and changes take the whole lock only, as before.
- The copy made on first use is no longer made with the table held alone: the entry's copy is a `OnceLock`, set under a lock of its own (`MAKING`, one making at a time, as before) with the table read. On a stripe, a first use holds up only the other first uses and the writers, not the other lookups (alone, it holds up every lookup alone, as before). `VICTORY_INPLACE_CONSTANTS`' look at the other buffers' copies (`constants_fresh`) is made under `MAKING` too, so it sees no copy made meanwhile, and no buffer comes or goes while the table is read.
- `LOOSE` (the windows) goes the same way: found with the table read; a window to make is made with it written, the window chosen again (on a stripe another thread may have made it meanwhile; alone, the whole lock is kept, as before). `let_go_idle` writes it, so no window is handed out between its look at a window's reference count and the removal.
- Same tables, same walks, same answers; only the locks differ. Lock order: whole, then stripes in rising order; a reader holds one of them and never takes another of the same table.
- The `alt` helper (`crate::alt`, every `=alt` switch) now looks at the phase it last said before it swaps it: asked on every lookup, the swap wrote one shared line from every thread each time, the very sharing this takes away. Same lines said.

Grep: `(VICTORY_BUFFERS_STRIPES=1)` at the start (`the tables of host copies are read on striped locks`), or the `=alt` lines `the tables of host copies are read on striped locks from now (VICTORY_BUFFERS_STRIPES=alt): ON` / `: off, as usual`.

P7 (`lot4 probe P7:`) now counts:
- `BUFFERS` (held alone: the lookups with the switch off and the rare walks over the whole table), `BUFFERS (read, striped)`, `BUFFERS (written)` (changes; before this branch they counted in `BUFFERS`);
- `windows (LOOSE)`, `windows (LOOSE, read, striped)`, `windows (LOOSE, written)` (not counted before);
- `BUFFERS read by the title's threads N, the heap sync N, the pass over buffer views N, the submitting thread N, other threads of ours N` and the same for `windows`, whatever the lock taken (by thread name: `victory-heap-sync*`/`victory-mirror-*`, `victory-heap-pass*`, `victory-submit*`, other `victory*`, the rest the title's).

What must move in a probe run with the switch on: `BUFFERS (read, striped)` with ~0 after a wait in place of `BUFFERS`' 72 ms, `BUFFERS (written)` small (its count is the title's buffer makings and releases), and the pass's `host copies looked up` and held time on its parts line; then the recording threads' share of the frame.

### `VICTORY_MADE_TURNS=1|alt` (A2): `made` handed to a waiting thread between a chunk's slots

The heap sync makes the slots it finds written in chunks of 64 (`mirror_due`, some 25-45 us held) and the pass over buffer views in chunks of 64 (some 35 us held: 43 ms over 1,245 chunks a pass in the probe run); between two chunks each lets `made` go and takes it again at once. The title's recording threads take `made` for one slot at a time (each view the title makes or descriptor it copies, `HostHeap::mirror`). The standard lock lets nobody in line first: a thread that waits sleeps, and the chunk maker, still running, takes the lock back before the sleeper wakes. So a waiter can wait through several chunks: P7's 982 waits a frame averaged 168 us, and the pass's own waits 188 us (33.2 ms over 177 waits a pass), four to five chunks' worth.

With the switch on:
- a thread that takes `made` for one slot or a few (`HostHeap::made_alone`: `mirror`, the texture slot looks and remakes, the diagnostics' looks) and finds it held says so (a per-heap count, on lines of its own) and spins for up to 30 us before it sleeps;
- a chunk maker (the sync's `mirror_due` and lean chunks, the pass's chunks made alone) looks at that count before each slot. When someone waits, its chunk ends there (its keeps and marks made, as at a chunk's end), it lets `made` go, waits up to 100 us for a waiting thread to have taken it, spins for it back, then sleeps for it.
- The chunk makers do not say when they wait for each other: the sync and the pass would otherwise hand `made` to each other slot by slot.

What each slot's making does is the same; the chunks are cut where a waiting thread comes in, as if they were smaller there. Not applied to the pass's threads holding `made` together (`VICTORY_HEAP_PASS_THREADS`), to `VICTORY_PARALLEL_MIRROR`'s rounds, or to `make_room`.

Grep: `(VICTORY_MADE_TURNS=1)` at the start (`the heap's slot state goes to a thread waiting for it between a chunk's slots`), or the `=alt` lines. P7: `made (alone)`'s waits (now also the one-slot takers' spins) and `made handed to a waiting thread between a chunk's slots N taken (N after a wait, X ms waited ...)`: N turns a frame and the ms the chunk makers spent letting the waiters in. What must move: `made (alone)`'s ms waited (165 a frame) and its waits' average, then the recording threads' time; watch the heap sync's own ms (`heap sync by place in the burst`) and the join (`ms waited` there) for the cost.

### Tests and micro-benchmarks

Unit tests (`cargo test -p victory-gpu --lib stripes turns`, on the host device):
- `stripes::tests`: `writers_exclude_striped_readers` (a write waits for a reader on a stripe; readers alone and on stripes read together; an upgrade from alone keeps the whole lock), `same_lookups_alone_and_striped` (the lookup's walk over random tables, 40,000 asks: the same key both ways), `twelve_threads_mixed_modes` (10 readers on stripes and alone, 2 writers by `write` and by `upgrade`: no write ever seen half done, the final table the writes').
- `shadow::stripes_tests` (equivalence and stress on the real tables): `lookups_on_stripes_give_what_they_give_alone` (nested, side by side and overlapping buffers and windows: the same copy alone and on stripes, whichever way made it first; copies made ahead made once; a released buffer's range then takes a window), `a_copy_asked_for_by_twelve_threads_at_once_is_made_once` (12 threads, 16 buffers without a copy yet, a barrier: one copy each, every thread gets it), `twelve_threads_look_up_change_and_walk_together` (8 lookup threads changing ways every 512 asks must always get the copies found at the start, 2 threads register, look up and release their own buffers, 1 walks both tables whole, 1 writes both; 1.5 s, a deadlock fails the test after 30 s).
- `descriptor::host::fast_tests` (A2): `turns_leave_every_slot_as_without_them` (three frames of records copied in as bytes, the sync's chunks and the pass's, with four threads taking `made` all along: every slot's record, held resource, notes and the pass's copies and ranges are the same as without the turns, and `made` was handed over), `twelve_threads_take_turns_on_one_heap` (the sync's chunks on one thread, the pass's on one, ten one-slot makers, 12 rounds each over slots of its own, turns on: no deadlock, and every slot ends as the same rounds made one after the other on one thread leave it).
- `lot4::locks::tests::readers_by_thread_name`.

Micro-benchmarks (Ryzen 9 9900X, `--profile diag`, two other agents building beside, so the absolute numbers are a little noisy; 12 threads span both CCDs):

`cargo test -p victory-gpu --profile diag --lib stripes::tests::bench_lookups -- --ignored --nocapture`: a table of 6,000 buffers of the title buffers' shape, each thread 400,000 lookups (the walk to the innermost buffer and a reference count), one write in 1,000 (two runs each, ns a lookup per thread, wall; in brackets the part spent taking the lock):

| threads | one mutex (before) | on stripes | |
|---|---|---|---|
| 1 | 124 (27) | 124 (27) | 1.0x |
| 4 | 857-866 (664-672) | 136-140 (29-31) | 6.2x |
| 8 | 3,228-3,232 (2,838-2,844) | 261-265 (94-96) | 12.3x |
| 12 | 5,286-5,289 (4,845-4,853) | 426-429 (225-226) | 12.4x |

The 12-thread stripes figure is mostly the writes: one in 1,000 lookups here is some 28,000 writes a second, each taking the 33 locks; the title's buffer makings and releases are far rarer (P7's `BUFFERS (written)` will say).

`cargo test -p victory-gpu --profile diag --lib descriptor::host::fast_tests::bench_turns -- --ignored --nocapture`: one thread makes 8,192 slots in the sync's chunks over and over (records rewritten each round), eleven threads each make one slot of their own every 50 us through `HostHeap::mirror` (the title's views and copies); 2 s each, two runs each way:

| | the one-slot makers' time in `mirror` | the sync |
|---|---|---|
| turns off | 26,497-27,641 calls; mean 747-781 us, median 2.5-2.6, p99 3,439-3,500, max 6,016-7,854 | 2.55-2.56 M slots/s |
| turns on | 422,228-423,955 calls; mean 1.7-1.9 us, median 1.5-1.6, p99 5.3-6.5, max 226-242 | 1.69-1.74 M slots/s |

Off, the one-slot makers wait through whole runs of the sync's chunks (the starvation the probe run shows, worse here because this sync never pauses); on, they wait about one slot. The sync pays about 1.5 us a turn (0.19 us more a slot, 0.33 s a second over some 211,000 one-slot calls a second, each at most one turn; the eleven makers at one call every 4.7 us are some four times the title's rate of takes of `made`, and most of the title's takes find it free and need no turn): at P7's ~1,000 waits a frame, about 1.5 ms a frame of the sync's and the pass's time.

### Risks

- **A1, the writes cost more.** With the switch set, a change of a table takes 33 locks (whole and 32 stripes) and waits for the readers of each stripe; `let_go_idle` writes `LOOSE` before every submission. Fine at the title's rate of buffer makings and releases if it is what it seems (P7's `BUFFERS (written)` and `windows (LOOSE, written)` will show it); a title that made and released thousands of buffers a frame would pay a few us each.
- **A1, a first use still holds a stripe.** A copy made on first use is made with the reader's stripe held (and `MAKING`): a writer that waits for that stripe holds the whole lock and the lower stripes meanwhile, so the readers on those stripes and every reader alone wait for the making, as every lookup waited for it before. Making the copy with no lock of the table held would mean throwing away a copy made twice, and its in-place and card bookkeeping; not done.
- **A1, no lookup inside a lookup.** A thread that held a stripe and asked the same table again would deadlock behind a waiting writer (the standard read-write lock lets a waiting writer go first). Nothing does today (the one mutex would have deadlocked at once); new code must not look a copy up while it holds a table.
- **A1, what stays shared.** The reference count of a popular copy (the constants heap's) is still one line every lookup of it writes, and a window's `used`/`asked` notes too; those are not lock waits but they remain.
- **A2, the chunk makers pay for the turns.** About 1.5 us a turn in the bench, up to 100 us when a waiting thread is slow to take `made` (it slept); the heap sync's join can grow by what the sync loses. Watch the sync's own ms (`heap sync by place in the burst`) and the join with the `=alt` run.
- **A2, the waiters spin.** A one-slot taker spins up to 30 us before it sleeps (a recording thread's core is busy meanwhile); when the holder makes no turns (`VICTORY_PARALLEL_MIRROR`'s rounds, the pass's threads holding `made` together, `make_room`) that is 30 us of spinning for nothing before the usual sleep.
- **A2, only one-slot takers are let in.** The sync and the pass do not hand `made` to each other (they would take turns slot by slot); their waits for each other are as before (the pass waited 33 ms a frame in the probe run).
- **A2 with `VICTORY_MIRROR_TRY=2`.** With it the title's copies and views do not wait for `made` at all (left to the next sync when it is held), so the turns help only the other one-slot takers; the turns are the way without `MIRROR_TRY`'s risk (records the title rewrote before the sync).
- **Neither is measured in the game yet.** Both change only who waits for whom; the micro-benchmarks show the mechanism, not the frame. The probe run with each `=alt` decides.

## Physical pages placed at their first mapping (`VICTORY_PAGE_PLACE=1`, opt-in, not measured yet; roadmap B2)

Branch `fx-main-pageplace`. P4 above, made real, so that `VICTORY_INPLACE_CONSTANTS` can open the constants heap.

**Why.** Today a physical page's place in a section is fixed when the title allocates it (`make_pages` in `crates/victory-hle/src/modules/xmem.rs`), and the title maps its pages in an order of its own. A buffer it maps at consecutive addresses therefore shows pages from scattered places. `VICTORY_INPLACE_CONSTANTS` needs the heap to be one run of one section: a host heap over two views side by side is refused (`OpenExistingHeapFromAddress1`, `E_INVALIDARG`, tried on the software adapter). So it turned the heap down as "view memory from several places".
- In the boat save's probe run (P2, `VICTORY_SECTION_MB=2`), P4 counted 113,446 pages placed at their address and none elsewhere.
- The 174 MB constants heap at 0x25380000 was scattered in 88 runs today, and one run with the simulated places. So was the 64 MB buffer at 0x285c20000 (33 runs), but it lies above 4 GiB, where the in-place switch does not look.

**What the switch does.** When the title maps a physical page for the first time (`XMemMapPhysicalPages`, its scatter form included), the page takes the place its address names in a section of placement, and keeps that place for good.
- **Windows.** The address space is cut into windows, each a section of its own, made when the first page is placed in it (reserved, committed page by page when mapped, as today).
  - Below 4 GiB there is one window of 4 GiB. Everything the title maps there (its 32-bit graphics memory, where the constants heap lies) is one section, and a buffer mapped there at consecutive addresses is one run of it, whatever the order of the calls.
  - Above 4 GiB the windows are `VICTORY_SECTION_MB` MiB if that is set, so its loading gain is kept (Windows' large-page worker holds a section's lock while it walks that section's views). Without it they are 4 GiB.
  - A reserved section costs no memory until it is mapped: eight sections of 4 GiB moved the kernel's paged pool by 0.0 MB on this PC.
  - A page first mapped somewhere else (a pool view, say) takes its place there, in that address's window. A buffer whose pages were first mapped in order above 4 GiB with `VICTORY_SECTION_MB=2` is therefore one run per 2 MiB window only, and would still be turned down; the line's largest run and the constants heap's `the last ...: <why>` tell.
- **Aliases.** A page mapped at a second address (or again after an unmap, or by the GPU's page-mapping copies) shows the same place: an alias, as before of its allocation's place. Pages mapped again in the same order at another address are one view again.
- **Collisions.** If another page already has the place (it was mapped there first and is still allocated, or freed but still shown somewhere), the page keeps the place its allocation gave it, as without the switch. Counted.
- **Written first.** A page written before its first mapping (a DirectStorage destination not mapped yet) keeps its allocation's place and its bytes. Counted.
- **Freed and reused.** A freed page's place is given back when no view shows it any more. A page placed there later is zeroed at its first mapping, so a new page still reads zero, as a page of a new section does. Counted as "given again".
- An allocation's own sections are still made (they hold the pages that are not placed). An allocation's section closes once all its pages are placed elsewhere or freed, and a window's section once no page or view is left in it; the GPU side is told of both, as with `VICTORY_SECTION_MB`.

**With `VICTORY_INPLACE_CONSTANTS=1`.** The heap's pages are taken to be first mapped at the heap's own addresses: in P2, when its copy was made, P5 found all its 2,784 pages shown at one address only, and P4 found them one run. So the heap is one run of the section below 4 GiB, and the existing opening (a view of ours of that run, `OpenExistingHeapFromAddress1`) takes it with no other change. The order and fence rules of that switch's section stay as they are:
- `VICTORY_SECTION_MB=2` no longer turns the heap down: with the placement it only cuts the windows above 4 GiB.
- `VICTORY_FENCE_SENT=1` with submissions sent ahead (the relay or `VICTORY_LATE_WAIT`) still turns `VICTORY_INPLACE_CONSTANTS` down at start, on purpose. The title is then told a frame begins while the frame before's lists still run, and the title rewrites the heap's per-draw constants (9,100-9,800 ranges sent up a frame in C2-s2, which looks like constants kept at fixed places rather than a ring moving ahead). In place, those lists could read the new frame's constants; a copy in the card's memory is refreshed behind them on the queue. **So measure B2 without `VICTORY_FENCE_SENT` on both sides**: A = the run set without `VICTORY_FENCE_SENT`; B = A + `VICTORY_PAGE_PLACE=1 VICTORY_INPLACE_CONSTANTS=1`. Comparing B with the full run set (FENCE_SENT on) answers which of the two levers is worth more.
- `VICTORY_INPLACE_CONSTANTS_ORDER` (indirect executions read in place clamped alone, on by default with the switch) comes with it as before.
- There is no `=alt`: a buffer cannot change form while the title runs (the switch's section).

**What to grep.**
- At start: `physical pages are placed at their first mapping (VICTORY_PAGE_PLACE=1)`, with `high_mb` (the windows above 4 GiB).
- Every ten seconds, one line: `physical pages placed at their first mapping (VICTORY_PAGE_PLACE=1): N MB placed now in N runs of consecutive addresses (the largest N MB at 0x...), N places of freed pages still shown; since the start N pages placed at their address, N kept at their allocation's place (N their place taken first, N written before their first mapping, N no section of placement), N places given back, N given again (zeroed); N sections of placement made, N MB (below 4 GiB one of 4096 MB, above windows of N MB); the constants heap: ...`.
- The constants heap part reads one of:
  - `a copy (VICTORY_INPLACE_CONSTANTS off)`;
  - `a copy (VICTORY_INPLACE_CONSTANTS=1 turned down at start: VICTORY_FENCE_SENT=1 with submissions sent ahead)`;
  - `IN PLACE: 0x...+174 MB (VICTORY_INPLACE_CONSTANTS=1), N title buffers placed in regions so far; turned down N (...)`;
  - `NOT in place (...)`, with the refusals by cause and `the last 0x...+N MB: <why>`.
- Read "kept at their allocation's place" first: P4 predicts about 0. The largest run should cover the heap (174 MB at its address).
- If no section of placement can be made, a warning says `no section of placement: the page keeps its allocation's place (VICTORY_PAGE_PLACE)` (eight at most).
- The mapping costs stay on `physical pages, since the start:` (maps, unmaps, ms in Windows' calls, sections made, now counting the sections of placement too).

**What the CPU should save per frame (to measure).** The heap's work today is on the copy lines. C2-s2 (boat save, 9.4 fps, 105.8 ms a frame, switch off) said for `0x33f60000+174 MB a copy in the card's memory`, per frame:

| Work | The heap's share | Where it is counted on the copy lines |
|---|---|---|
| Refreshes from title memory | 12-13 refreshes, 1.1-1.2 ms, 9.1-9.6 MB copied in | fps line `copies X ms Y MB` (of 10.7 ms, 94.2 MB); `copies per frame:` `refreshes`, `runs`, and `updates of host copies` on the split |
| Sends to the card | 11.0-11.2 MB in 9,100-9,800 ranges | `host copies in the card's memory:` `MB sent up in N ranges` (of 48.8 MB in 13,691 ranges) |
| Compares on the card | 3,400-3,600 pieces of 64 KiB | the same line's `pieces compared on the card` (of 19,138); `our passes over copies in the card's memory` (hash and compare passes) |
| Fetches and merges | 3.0-3.1 MB fetched back, 2.2 MB merged into title memory | the same line's `fetched (MB)` (of 54.5 MB) and `more waits for the card`; fps line `copying back X ms` (27.3 ms) |
| Guards | the title's writes to the heap's guarded blocks | recording line `guards X ms` and `copies per frame:` `guard faults` |

The big buffers' line (`the big title buffers per frame`) gives the heap's own row. With the heap in place, that row's form should read `in place`, with 0 refreshes, sends, compares, fetches and merges.
- What leaves the submitting thread: the refreshes (about 1.2 ms), the recording of some 9,500 range copies a frame, the heap's share of the card copies' hash and compare passes, and about 3 MB of the copy-back.
- Expected: a few ms of the ~106 ms frame, perhaps 5 %. It is not measured.
- Against it: the card reads the heap over the bus. In fxceless' run 75, one skinning list took 4.5 ms in the card's memory and 50.5 ms over the bus. Watch `on the GPU` and the card's time by what ran. The heap also stays locked in the PC's memory while it stands.

**Risks.**
- **Loading and mapping cost.**
  - All the title's mappings below 4 GiB share one section's lock again, which is what `VICTORY_SECTION_MB=2` removed for the 7 GB pool. Above 4 GiB nothing changes. Compare the `physical pages, since the start:` line (maps, unmaps, in Windows' calls, calls of 1 ms or more) with a run without the switch over the same loading.
  - A first mapping costs a few table lookups per page, plus one `CreateFileMapping` per window first used (some thousands of 2 MiB windows with `VICTORY_SECTION_MB=2`, about as many as the allocations' sections, which are still made too).
  - A place given again costs a 64 KiB zeroing.
  - Pages mapped in order are fewer views, not more: a call's consecutive places are one view.
- **Memory.** A place given back keeps its committed pages until a page takes it again, or until its window's section closes when nothing is left in it. Below 4 GiB the one window rarely closes, so memory the title frees there is not returned to Windows until reused. Today an allocation's section returns it once all its pages are freed and unmapped. Watch "places given back" against "given again", and the commit.
- **Correctness.**
  - The places change where the bytes of view memory live, never which bytes a title address shows.
  - Aliases, DirectStorage writes through views of ours, the GPU side's guards over several views of a page, and the page-mapping copies all go through the same page table as before.
  - The first run of the switch should still be checked for the picture and for crashes in loading.
- **Off,** the switch costs a look at whether any page waits for its place, in the mapping and freeing calls. The GPU side's line is not registered.

**Tests.**
- `xmem::tests`:
  - `placement_switch_and_windows`;
  - `pages_first_mapped_at_consecutive_addresses_are_one_run` (four views without the switch, one with it; aliases, a place kept, places held while shown and given back);
  - `a_place_taken_first_keeps_the_allocations_place`;
  - `freed_places_are_given_back_and_read_zero_when_given_again`;
  - `pages_written_before_their_first_mapping_keep_their_place`;
  - `a_run_over_a_window_bound_is_one_run_per_window`;
  - `the_32_bit_memory_is_one_section_of_placement`;
  - `the_placement_line_says_what_is_placed`.
- `inplace::tests`:
  - `warp_the_constants_heap_placed_at_its_addresses_is_one_run_in_place` (on the software adapter: the same eight one-page calls in one order, as allocated, are turned down as scattered; placed at their addresses, they are one run, the region opens, and the card's reads, copies and shader stores meet the title's bytes);
  - `the_placement_line_names_the_constants_heap`.

## `VICTORY_COMPOSITE_GRADE_EDGE=1` (off by default; port of Cuebert 818c5a9)

Black squares and specks in the water, such as the sun's glint with ray tracing on, come from the Composite pixel shader's first colour-grading lookup (a 33-cubed LUT, sampler constants `0x0092 0x00c00000 0x14500000 0`): coordinates near the cube's faces sample past the edge texels. With the switch at exactly `1`, the translator clamps that lookup's three coordinates to `[0.5/33, 32.5/33]` (the edge texel centres), so linear sampling stays inside the LUT. The ACES lookup after it is untouched. It applies to one pixel shader only (DXBC digest `348a8c9e...fc6a`, 4808-byte DXIL part, sample at machine offset `0xefc`); if that sample no longer has the reviewed clamp sampler the shader fails to translate rather than being changed silently. Code: `lower_with_grade_edge` and the image sample in `crates/victory-shader/src/emit/lower.rs`, the gate `grade_edge_enabled` in `emit/mod.rs`. The switch is part of the shader cache fingerprint (`build.rs` lists every `"VICTORY_..."` string), so **rewarm the shader cache** after changing it. Credit: Cuebert/VIctory `818c5a9` ("Publish F895 player source snapshot"); read only, nothing from that repo was run.

## Roadmap: from about 3.5 fps toward 15–20 (2026-11-25; superseded by [ROADMAP.md](ROADMAP.md), kept for history)

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
