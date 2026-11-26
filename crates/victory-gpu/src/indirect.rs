//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! The scratch buffer is left readable as indirect arguments after a clamp
//! and stays so until the next clamp into it, which moves it straight back
//! to unordered access; it goes back to the common state once, when the
//! list closes. The executions that read it record no barrier of their
//! own, so draws that read records clamped together (`crate::draw`'s
//! windows) follow each other with nothing between them. Before, each
//! execution was followed by a move of the scratch buffer back to the
//! common state, which drivers may treat as a full wait.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! A single dispatch record runs in parts when it asks for more than 65,535
//! groups in x ([`bound_split`], [`PART`]).
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups in y and z and in x past [`MOST_PARTS`] parts or in a dispatch of
//! several records, 2^20 per dispatch, one-dimensional mesh dispatches,
//! the budget) cut legitimate work beyond them; raise them if the title
//! needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 2: the records are gathered
    // (`gather`): record i lies in the source where word i of the table
    // behind `count_buf` says (from `count_offset` on), and goes to slot i.
    // Bit 3: one dispatch record that runs in parts (`bound_split`): its
    // groups in x as asked, up to what PARTS parts of 65,535 run, written
    // as PARTS records of four words (the number of the part's first group
    // in x, then its groups) and after them the number of parts to run.
    // Bit 4, with bit 3: one part at most (x cut as without bit 3), the
    // program noted as below (`bound_split_one`).
    // From bit 8 on: the number of the dispatch's program (`bound_noted`),
    // noted when a record asks for more groups in x than one dispatch runs.
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// The vertices or indices an instance from which a draw record runs empty.
static const uint MOST_DRAW = 1u << 23;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// The most parts a dispatch runs in (`MOST_PARTS`), and the most groups in
// a dimension of one part (what the host runs).
static const uint PARTS = 16;
static const uint PART = 65535;
// Programs noted (`NOTED`), after the count of notes in the budget's word 7.
static const uint NOTED = 8;
// Each group charges the budget once: the costs of its records go here,
// one thread charges their sum, saturating as each record would in lane
// order, and every thread then reads whether its own record fitted. A
// compare and swap per record, 64 threads racing for one word, cost a
// round for each record of a window, and in memory the CPU maps (where the
// budget was, and is with VICTORY_INDIRECT_CARD_BUDGET=0) every atomic
// crosses the bus on a card of its own.
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint largest_asked;
groupshared uint all_asked;
// Notes the dispatch's program (`has_count` from bit 8 on), if given, when
// its record asked for `asked` groups in x, more than one dispatch runs and
// no more than the parts run: its dispatches run in parts from then on
// (VICTORY_DISPATCH_PARTS unset or seen).
void note_program(uint asked) {
    uint program = has_count >> 8;
    if (program != 0 && asked > PART && asked <= PARTS * PART) {
        uint slot;
        budget.InterlockedAdd(28, 1u, slot);
        if (slot < NOTED) {
            budget.Store(32u + slot * 4u, program);
        }
    }
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    bool split = (has_count & 8u) != 0;
    bool one_part = (has_count & 16u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    // Threads past the records take part in the group's charge (cost 0)
    // and store nothing.
    bool active = id.x < eff_count;
    uint src_at = args_offset + id.x * stride;
    // (A branch, not a select: without the bit the view behind `count_buf`
    // is a count of four bytes, or the source itself, and is not read here.)
    [branch]
    if ((has_count & 4u) != 0) {
        src_at = count_buf.Load(count_offset + id.x * 4u);
    }
    uint dst_at = id.x * stride;
    uint cost = 0;
    uint asked = 0;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    bool beyond = false;
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
    }
    if (active && kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 want = src_args.Load2(src_at);
        if (ahead && (want.x > (1u << 24) || want.y > (1u << 24))) {
            want = uint2(0, 0);
        }
        uint count = min(want.x, 1u << 24);
        uint instances = min(want.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        // A record that asks for MOST_DRAW vertices or indices an instance
        // or more is no draw of the title's (no mesh of a console title has
        // 2.8 million triangles): it runs empty and charges nothing. Run
        // 206's window check found one list with 90 draws of 2^24 vertices
        // and one instance each. Clamped one draw at a time, the list's
        // budget lets eight of them run (2^27 in all) and empties the other
        // 82; served from a window, whose records are charged once however
        // often they are drawn, all 90 ran: 1.5 thousand million vertices
        // in one list, with windows on in the runs that lost the device
        // (201, 204).
        if (count >= MOST_DRAW) {
            count = 0;
            instances = 0;
            cost = 0;
        }
        draw_counts = uint2(count, instances);
    } else if (active && kind == 2) {
        uint3 want = src_args.Load3(src_at);
        asked = want.x;
        beyond = max(want.x, max(want.y, want.z)) > PART;
        uint3 g = min(want, uint3(PART, PART, PART));
        // In parts: x as asked, up to what the parts run (past that, cut
        // to one part as before: a slot nothing filled, or more than the
        // host runs).
        if (split && !one_part && want.x <= PARTS * PART) {
            g.x = want.x;
        }
        // y * z fits 32 bits; the whole count may not (x * y neither, in
        // parts). Past GROUPS the dispatch runs empty.
        uint yz = g.y * g.z;
        if (yz != 0 && g.x > GROUPS / yz) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else if (active) {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is a
        // slot the title has not filled (the title leaves all
        // ones there): empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        asked = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind >= 2) {
        // Diagnostics, in the budget buffer's next words (below): the
        // largest group count a mesh record asked for, and the groups
        // asked in all; the largest a dispatch record asked for in x.
        uint was;
        InterlockedMax(largest_asked, asked, was);
        if (kind > 2) {
            InterlockedAdd(all_asked, min(asked, 1u << 24), was);
        }
    }
    // The group's own records: none for a group past a count buffer's
    // count (an execution may ask for thousands and count a few), which
    // touches the budget not at all, as its threads did not before.
    uint first = id.x - gi;
    uint n = eff_count > first ? min(64u, eff_count - first) : 0u;
    if (gi == 0 && n > 0) {
        // What the records charge, from `before` on, in lane order: a
        // record that does not fit whole runs empty and takes the rest.
        // Only the group's own records are walked (one, clamped alone).
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                if (c > 0 && next < BUDGET) {
                    next = (BUDGET - next <= c) ? BUDGET : (next + c);
                }
            }
            if (next == before) break;
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        uint at = before;
        for (uint k = 0; k < n; k++) {
            uint c = costs[k];
            uint fit = 1;
            if (c > 0) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at <= c) ? BUDGET : (at + c);
            }
            fits[k] = fit;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && kind > 2 && n > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
    if (gi == 0 && kind == 2 && n > 0) {
        uint was;
        budget.InterlockedMax(12, largest_asked, was);
    }
    if (!active) {
        // In parts, a record past the count buffer's count: no part runs.
        if (split && id.x == 0) {
            dst_args.Store(PARTS * 16u, 0u);
        }
        return;
    }
    bool over = fits[gi] == 0;
    // A dispatch record that asks for more groups in a dimension than one
    // dispatch runs (`beyond`): counted in budget words 4 to 6 as run in
    // parts, or as cut.
    if (kind == 2 && split) {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        uint parts = (groups.x == 0 || groups.y == 0 || groups.z == 0) ? 0u : (groups.x + PART - 1u) / PART;
        for (uint part = 0; part < PARTS; part++) {
            uint part_first = part * PART;
            uint part_x = groups.x > part_first ? min(groups.x - part_first, PART) : 0u;
            dst_args.Store4(part * 16u, uint4(part_first, part_x, groups.y, groups.z));
        }
        dst_args.Store(PARTS * 16u, parts);
        uint was;
        if (parts > 1) {
            budget.InterlockedAdd(16, 1u, was);
            budget.InterlockedAdd(20, parts, was);
        } else if (beyond) {
            budget.InterlockedAdd(24, 1u, was);
            note_program(asked);
        }
        return;
    }
    if (beyond) {
        uint was;
        budget.InterlockedAdd(24, 1u, was);
        note_program(asked);
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    make_from(device, SOURCE)
}

/// The clamp compiled from `source` (ours, or in the benchmark the one it
/// replaced).
fn make_from(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, at offset `0`. The scratch buffer is left in `INDIRECT_ARGUMENT`
/// until the next clamp into it: executions read it with no barrier of
/// their own. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp or alignment is not
/// available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    clamp_into(list, c, &dst, budget, src, count, spec, 0).then_some(dst)
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). The records past
/// the ones the title filled are whatever the buffer holds, and each
/// record clamped charges the list's budget when it is clamped: a slot of
/// all ones, clamped to the most a mesh dispatch runs (65,535 groups),
/// charged a sixteenth of the budget, so sixteen of them emptied every
/// draw of the list after them (run 93: "the largest group count asked for
/// 4294967295", 43 to 153 such records a frame). Here a record that asks
/// for more than can run is left empty and charges nothing; a record
/// clamped alone is clamped and charged as before.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, None)?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    clamp_into(list, c, &dst, budget, src, None, spec, AHEAD).then_some(dst)
}

/// [`bound`] for one dispatch record (`stride` bytes, bounded by the count
/// in `count` when given) whose groups may be more in x than the host
/// runs in one dispatch: the clamp writes [`MOST_PARTS`] records of
/// [`PART_STRIDE`] bytes at offset 0 (the number of the part's first group
/// in x, then its groups: at most [`PART`] in x) and the number of parts
/// that run at [`PARTS_COUNT_AT`], for one execution with that count
/// (`crate::draw`'s indirect dispatches; [`split_signature`]). A record
/// past what the parts run is cut to one part, as before.
pub(crate) fn bound_split(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    stride: u32,
) -> Option<ID3D12Resource> {
    let spec = (2, stride, 1);
    validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(SPLIT_BYTES)?);
    clamp_into(list, c, &dst, budget, src, count, spec, IN_PARTS).then_some(dst)
}

/// [`bound`] for dispatch records of the program numbered `program`
/// ([`program_parts`]): a record asking for more groups in x than one
/// dispatch runs, and no more than the parts run, is clamped as before and
/// its program noted in the list's budget words 7 to 15, which
/// [`note_budget_words`] takes a list late ([`Parts::Seen`], the default).
pub(crate) fn bound_noted(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    program: u32,
) -> Option<ID3D12Resource> {
    if program == 0 || program > MOST_PROGRAMS {
        return bound(list, src, count, spec);
    }
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    clamp_into(list, c, &dst, budget, src, count, spec, program << PROGRAM_SHIFT).then_some(dst)
}

/// [`bound_split`] for a dispatch that takes the parts' records and count
/// but runs one part at most: x is cut to [`PART`] as without the parts,
/// and the program numbered `program` is noted as [`bound_noted`] notes it
/// (`VICTORY_DISPATCH_PLAIN=split`, [`plain_mode`]).
pub(crate) fn bound_split_one(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    stride: u32,
    program: u32,
) -> Option<ID3D12Resource> {
    let spec = (2, stride, 1);
    validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(SPLIT_BYTES)?);
    let noted = if program <= MOST_PROGRAMS { program << PROGRAM_SHIFT } else { 0 };
    clamp_into(list, c, &dst, budget, src, count, spec, IN_PARTS | ONE_PART | noted).then_some(dst)
}

/// Runs a split dispatch's records ([`bound_split`]) in `scratch` from
/// `at`, up to [`MOST_PARTS`] of them as many as the count after them says:
/// through `signature`, which sets the number of each part's first group at
/// `base` (root parameter, dword) when given; otherwise through a plain
/// dispatch signature of [`PART_STRIDE`] bytes, from the records' second
/// words. A root argument a command signature sets is undefined after the
/// execution (D3D12), so the number is set to 0 again.
pub(crate) fn execute_parts(
    l: &ID3D12GraphicsCommandList,
    signature: &ID3D12CommandSignature,
    base: Option<(u32, u32)>,
    scratch: &ID3D12Resource,
    at: u64,
) {
    let first = at + if base.is_some() { 0 } else { 4 };
    // SAFETY: an open list with the program bound; the scratch buffer is
    // readable as arguments and holds the records and their count; `base`
    // is a dword of the program's user-data constants.
    unsafe {
        l.ExecuteIndirect(signature, MOST_PARTS, scratch, first, Some(scratch), at + PARTS_COUNT_AT);
        if let Some((param, dword)) = base {
            l.SetComputeRoot32BitConstant(param, 0, dword);
        }
    }
}

/// Bits of the clamp's `has_count` ([`record_clamp_how`]): a window
/// clamped ahead of its draws ([`bound_ahead`]); one dispatch in parts
/// ([`bound_split`]), at most one part ([`bound_split_one`]); from
/// [`PROGRAM_SHIFT`] on, the program to note ([`bound_noted`]).
const AHEAD: u32 = 2;
const IN_PARTS: u32 = 8;
const ONE_PART: u32 = 16;
const PROGRAM_SHIFT: u32 = 8;
/// The most programs numbered for the clamp's notes.
const MOST_PROGRAMS: u32 = (1 << (32 - PROGRAM_SHIFT)) - 1;
/// The programs one list's clamps note (the budget's words 8 to 15); more
/// are counted (word 7) and noted by later lists.
const NOTED: usize = 8;

/// The most thread groups the host runs in a dimension of one dispatch.
/// The console runs any number: a larger dispatch runs here in parts of
/// this many groups in x (myokex's fix, ZSlayerHQ/victory-world-runtime
/// 4436f8f: there two parts, for indirect dispatches by programs that read
/// their x number alone). A translated program that reads its group's
/// number in x adds the number of its part's first group, a root constant
/// (`victory_shader::emit::USER_GROUP_BASE`) the host sets between the
/// parts; one that does not read it needs nothing (its groups differ in
/// nothing but their count). Clamped to 65,535, the groups past it never
/// ran, and what they would have written kept what it held before: run
/// 200 counted indirect compute records asking for 77,000 to 98,000 groups
/// in x, 0.2 to 0.8 a frame.
pub(crate) const PART: u32 = 65_535;
/// The most parts a dispatch runs in: 16 parts, 1,048,560 groups in x
/// (a 4K screen in 8x8 tiles is 129,600 groups, eight per tile 1,036,800),
/// about the most groups the clamp lets one dispatch run anyway
/// (`GROUPS`, 2^20). A dispatch asking for more in x runs the first
/// [`MOST_PARTS`] parts (direct) or one part (indirect), as before.
pub(crate) const MOST_PARTS: u32 = 16;
/// Bytes of a part's record in a split dispatch's scratch buffer
/// ([`bound_split`]): the number of the part's first group in x, then its
/// groups in x, y and z.
pub(crate) const PART_STRIDE: u32 = 16;
/// Where the number of parts to run follows the parts' records.
pub(crate) const PARTS_COUNT_AT: u64 = MOST_PARTS as u64 * PART_STRIDE as u64;
/// Scratch bytes of a split dispatch.
const SPLIT_BYTES: u64 = PARTS_COUNT_AT + 4;

/// Which dispatches run in parts (`VICTORY_DISPATCH_PARTS`; the translated
/// programs read their group base whatever it says).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Parts {
    /// `0`: none; cut as before.
    Off,
    /// `all`: direct dispatches past [`PART`] groups in x, and every
    /// indirect dispatch of one record through the parted path
    /// ([`bound_split`]). Its command signature sets a root constant, and
    /// that costs each such execution about 20 microseconds of card time
    /// (RT64b: about 20 ms a frame more on an RTX 4080, for some 970 of
    /// them).
    All,
    /// Unset (or `seen`, or anything else), the default: direct dispatches
    /// as with [`Parts::All`]; an indirect dispatch of one record through
    /// the parted path only if its program (by its hash, whatever pipeline
    /// or record it comes from) has been seen asking for more groups in x
    /// than one dispatch runs (noted by the clamp, [`bound_noted`], and
    /// known a list late: its first such dispatch runs cut, as before); the
    /// others through the plain path, as before the parts. It fixes the
    /// strip along the bottom of the frame at no cost (RT67: 186 ms a frame,
    /// one program seen, one parted execution a frame).
    Seen,
}

/// `VICTORY_DISPATCH_PARTS`'s mode. Pure, for testing.
fn parse_parts(value: Option<&str>) -> Parts {
    match value.map(str::trim) {
        Some("0") => Parts::Off,
        Some("all") => Parts::All,
        _ => Parts::Seen,
    }
}

/// The mode `VICTORY_DISPATCH_PARTS` sets ([`Parts`]).
pub(crate) fn parts_mode() -> Parts {
    static MODE: std::sync::OnceLock<Parts> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_parts(std::env::var("VICTORY_DISPATCH_PARTS").ok().as_deref()))
}

/// Whether dispatches of more groups in x than the host runs run in parts
/// at all ([`Parts::Off`]: no).
pub(crate) fn parts_on() -> bool {
    parts_mode() != Parts::Off
}

/// With [`Parts::Seen`] (the default), what a single-record indirect
/// dispatch that does not take the parted path does instead
/// (`VICTORY_DISPATCH_PLAIN`, any of these, comma-separated; none by
/// default: the plain path of before the parts). For telling why RT67
/// (seen) left the lower-right quarter of the picture dark where RT64
/// (every such dispatch parted) lit it, which no difference in the records
/// explains. RT72 (`rootsig`) lit it, RT70 (`split`), RT71 (`wait`) and
/// RT73 (`after`) did not: the root-constant signature itself does it.
/// - `split`: everything the parted path does but the root constant: the
///   clamp's part records and their count ([`bound_split_one`]: one part at
///   most, its program noted), run through a plain dispatch signature from
///   the records' second words with the clamp's count, where the plain path
///   runs the clamped record with the title's count buffer;
/// - `wait`: a full wait (every shader write, and an aliasing barrier on
///   every resource) right before the execution, after the program's bind,
///   as a driver may wait before an execution whose command signature sets
///   root constants (RT64's about 20 microseconds of card time for each);
/// - `after`: the same full wait right after the execution, for what reads
///   its output next (RT70 and RT71, `split` and `wait`, left the quarter
///   dark);
/// - `rootsig`: `split`, and a program that takes its first group's number
///   runs through the parted path's command signature, which sets it (to 0,
///   one part): the parted path for a record within one dispatch, x cut
///   past it. It should light the quarter at the parted path's cost if the
///   signature's root constant is what does it.
///
/// `VICTORY_DISPATCH_PLAIN_ONLY` (hash beginnings, comma-separated) limits
/// them to those programs, to find the pass by halving the list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Plain {
    pub(crate) split: bool,
    pub(crate) wait: bool,
    pub(crate) after: bool,
    pub(crate) rootsig: bool,
}

impl Plain {
    /// The variants' names, comma-separated (for the line).
    fn names(self) -> String {
        let names = [(self.split, "split"), (self.wait, "wait"), (self.after, "after"), (self.rootsig, "rootsig")];
        names.iter().filter(|n| n.0).map(|n| n.1).collect::<Vec<_>>().join(",")
    }
}

/// `VICTORY_DISPATCH_PLAIN`'s variants. Pure, for testing.
fn parse_plain(value: Option<&str>) -> Plain {
    let mut plain = Plain::default();
    for word in value.unwrap_or("").split(',').map(str::trim) {
        match word {
            "split" => plain.split = true,
            "wait" => plain.wait = true,
            "after" => plain.after = true,
            "rootsig" => plain.rootsig = true,
            _ => {}
        }
    }
    plain
}

/// The variants `VICTORY_DISPATCH_PLAIN` asks for ([`Plain`]); none with
/// `VICTORY_DISPATCH_PARTS=all` or `0`.
pub(crate) fn plain_mode() -> Plain {
    static MODE: std::sync::OnceLock<Plain> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        if parts_mode() != Parts::Seen {
            return Plain::default();
        }
        parse_plain(std::env::var("VICTORY_DISPATCH_PLAIN").ok().as_deref())
    })
}

/// The hash beginnings of `VICTORY_DISPATCH_PLAIN_ONLY` (none: every
/// program). Pure, for testing.
fn parse_only(value: Option<&str>) -> Vec<String> {
    value.unwrap_or("").split(',').map(str::trim).filter(|w| !w.is_empty()).map(str::to_owned).collect()
}

fn plain_only() -> &'static [String] {
    static ONLY: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    ONLY.get_or_init(|| parse_only(std::env::var("VICTORY_DISPATCH_PLAIN_ONLY").ok().as_deref()))
}

/// [`plain_mode`] for the program with hash `hash`: none if
/// `VICTORY_DISPATCH_PLAIN_ONLY` leaves it out.
pub(crate) fn plain_for(hash: &str) -> Plain {
    let only = plain_only();
    if only.is_empty() || only.iter().any(|w| hash.starts_with(w.as_str())) {
        plain_mode()
    } else {
        Plain::default()
    }
}

/// A full wait: every earlier shader write done, and an aliasing barrier
/// on every resource, which drivers take for waiting until the card is
/// idle ([`Plain::wait`]).
pub(crate) fn full_wait() -> [D3D12_RESOURCE_BARRIER; 2] {
    let aliasing = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: std::mem::ManuallyDrop::new(None),
                pResourceAfter: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    [uav_barrier(), aliasing]
}

/// The programs of indirect dispatches numbered for the clamp's notes
/// ([`Parts::Seen`]): number by hash (from 1), hash by number, and whether
/// the program was seen asking for more groups in x than one dispatch runs.
#[derive(Default)]
struct Programs {
    numbers: std::collections::HashMap<String, u32>,
    hashes: Vec<String>,
    seen: Vec<bool>,
}

static PROGRAMS: std::sync::Mutex<Option<Programs>> = std::sync::Mutex::new(None);

/// The number of the program with hash `hash` for the clamp's notes
/// ([`bound_noted`]; 0 past [`MOST_PROGRAMS`]), and whether it was seen
/// asking for more groups in x than one dispatch runs: then its indirect
/// dispatches run in parts ([`Parts::Seen`]).
pub(crate) fn program_parts(hash: &str) -> (u32, bool) {
    let mut programs = PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
    let programs = programs.get_or_insert_with(Programs::default);
    if let Some(&number) = programs.numbers.get(hash) {
        return (number, programs.seen[number as usize - 1]);
    }
    if programs.hashes.len() >= MOST_PROGRAMS as usize {
        return (0, false);
    }
    programs.hashes.push(hash.to_owned());
    programs.seen.push(false);
    let number = programs.hashes.len() as u32;
    programs.numbers.insert(hash.to_owned(), number);
    (number, false)
}

/// Marks the programs the clamp noted ([`bound_noted`]) as seen asking for
/// more groups in x than one dispatch runs; the first time, said.
fn programs_seen(noted: &[u32]) {
    let mut programs = PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(programs) = programs.as_mut() else {
        return;
    };
    for &number in noted {
        let Some(seen) = number.checked_sub(1).and_then(|i| programs.seen.get_mut(i as usize)) else {
            continue;
        };
        if !*seen {
            *seen = true;
            tracing::info!(
                program = %programs.hashes[number as usize - 1],
                "an indirect dispatch asked for more than {PART} groups in x: its program's indirect dispatches \
                 run in parts from now on (VICTORY_DISPATCH_PARTS unset or seen)"
            );
        }
    }
}

/// The programs seen asking for more groups in x than one dispatch runs.
fn programs_seen_count() -> usize {
    let programs = PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
    programs.as_ref().map_or(0, |p| p.seen.iter().filter(|&&s| s).count())
}

/// Counts a single-record indirect dispatch run through the parted path
/// ([`bound_split`]) or the plain one, for the line ([`parts_line`]).
pub(crate) fn note_path(parted: bool) {
    PATHS[usize::from(!parted)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Single-record indirect dispatches since the last line: through the
/// parted path, through the plain one.
static PATHS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The parts [`record_parts`] runs `x` groups in: 1 up to [`PART`], at most
/// [`MOST_PARTS`].
pub(crate) fn parts_of(x: u32) -> u32 {
    x.div_ceil(PART).clamp(1, MOST_PARTS)
}

/// Records `Dispatch(x, y, z)` in `l` with the program bound: past [`PART`]
/// groups in x in [`parts_of`] parts, each after setting the number of its
/// first group at `base` (the root parameter and dword of the program's
/// `USER_GROUP_BASE`; none if the program does not read its x number),
/// which is 0 again afterwards. A dispatch of [`PART`] groups or fewer is
/// recorded as it is. The parts recorded.
pub(crate) fn record_parts(l: &ID3D12GraphicsCommandList, base: Option<(u32, u32)>, [x, y, z]: [u32; 3]) -> u32 {
    if x <= PART {
        // SAFETY: an open list with the program bound.
        unsafe { l.Dispatch(x, y, z) };
        return 1;
    }
    let parts = parts_of(x);
    // SAFETY: an open list with the program bound; `base` is a dword of
    // its root signature's user-data constants.
    unsafe {
        for part in 0..parts {
            let first = part * PART;
            if let Some((param, dword)) = base {
                l.SetComputeRoot32BitConstant(param, first, dword);
            }
            l.Dispatch((x - first).min(PART), y, z);
        }
        if let Some((param, dword)) = base {
            l.SetComputeRoot32BitConstant(param, 0, dword);
        }
    }
    parts
}

/// [`record_parts`] for a dispatch of the title's, counted for the line
/// ([`parts_line`]); with [`parts_on`] off, recorded as it is.
pub(crate) fn dispatch_parts(l: &ID3D12GraphicsCommandList, base: Option<(u32, u32)>, groups: [u32; 3]) -> u32 {
    use std::sync::atomic::Ordering::Relaxed;
    let [x, ..] = groups;
    PARTS_DIRECT[2].fetch_max(u64::from(x), Relaxed);
    if !parts_on() {
        // SAFETY: an open list with the program bound.
        unsafe { l.Dispatch(groups[0], groups[1], groups[2]) };
        return 1;
    }
    let parts = record_parts(l, base, groups);
    if parts > 1 {
        PARTS_DIRECT[0].fetch_add(1, Relaxed);
        PARTS_DIRECT[1].fetch_add(u64::from(parts), Relaxed);
    }
    if x > PART * MOST_PARTS || groups[1] > PART || groups[2] > PART {
        PARTS_DIRECT[3].fetch_add(1, Relaxed);
    }
    parts
}

/// Direct dispatches since the last line: run in parts, the parts they
/// ran, the largest group count asked for in x, and those asking for more
/// in a dimension than the parts run (cut).
static PARTS_DIRECT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// Indirect dispatch records since the last line, from the lists' budget
/// words 3 to 6 (a run late): the largest group count a record asked for
/// in x, the records run in parts, their parts, the records asking for
/// more in a dimension than one dispatch runs that ran cut.
static PARTS_INDIRECT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Words of a list's budget buffer from word 3 on that the clamp writes
/// for dispatches ([`note_budget_words`]).
pub(crate) const DISPATCH_WORDS: usize = 5 + NOTED;

/// Takes what the clamp noted in a list's budget words 3 to 15 (`words`):
/// the largest group count a record asked for in x, the records run in
/// parts, their parts, the records cut, the count of programs noted
/// ([`bound_noted`]) and the first [`NOTED`] of them.
pub(crate) fn note_budget_words(words: [u32; DISPATCH_WORDS]) {
    use std::sync::atomic::Ordering::Relaxed;
    let [largest, split, parts, cut, noted] = [0, 1, 2, 3, 4].map(|i| words[i]);
    PARTS_INDIRECT[0].fetch_max(u64::from(largest), Relaxed);
    PARTS_INDIRECT[1].fetch_add(u64::from(split), Relaxed);
    PARTS_INDIRECT[2].fetch_add(u64::from(parts), Relaxed);
    PARTS_INDIRECT[3].fetch_add(u64::from(cut), Relaxed);
    if noted > 0 {
        programs_seen(&words[5..5 + (noted as usize).min(NOTED)]);
    }
}

/// The frame line of dispatches in parts, per frame over `frames`.
pub(crate) fn parts_line(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [split, parts, largest, cut] = [0, 1, 2, 3].map(|i| PARTS_DIRECT[i].swap(0, Relaxed));
    let [i_largest, i_split, i_parts, i_cut] = [0, 1, 2, 3].map(|i| PARTS_INDIRECT[i].swap(0, Relaxed));
    let [parted, plain] = [0, 1].map(|i| PATHS[i].swap(0, Relaxed));
    tracing::info!(
        "dispatches in parts (VICTORY_DISPATCH_PARTS {}) per frame: direct {:.2} split into {:.2} parts, {:.2} cut, \
         the largest group count asked for in x {largest}; indirect {:.2} split into {:.2} parts, {:.2} cut, the \
         largest group count asked for in x {i_largest} (a list late; one dispatch runs at most {PART}); \
         single-record indirect dispatches {:.2} through the parted path, {:.2} through the plain one; {} programs \
         seen asking for more in x",
        match (parts_mode(), plain_mode()) {
            (Parts::Off, _) => "off".to_owned(),
            (Parts::All, _) => "all".to_owned(),
            (Parts::Seen, plain) if plain == Plain::default() => "seen".to_owned(),
            (Parts::Seen, plain) => format!(
                "seen, VICTORY_DISPATCH_PLAIN {}{}",
                plain.names(),
                if plain_only().is_empty() { String::new() } else { format!(" for {} hash beginnings", plain_only().len()) }
            ),
        },
        split as f64 / frames,
        parts as f64 / frames,
        cut as f64 / frames,
        i_split as f64 / frames,
        i_parts as f64 / frames,
        i_cut as f64 / frames,
        parted as f64 / frames,
        plain as f64 / frames,
        programs_seen_count(),
    );
}

/// The command signature of a split dispatch ([`bound_split`]) of a
/// program with root signature `root` that takes the number of its first
/// group at `base` (root parameter, dword): each part sets that constant,
/// then dispatches. None if the device refuses it.
fn make_split_signature(device: &ID3D12Device, root: &ID3D12RootSignature, (param, dword): (u32, u32)) -> Option<ID3D12CommandSignature> {
    let arguments = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: param,
                    DestOffsetIn32BitValues: dword,
                    Num32BitValuesToSet: 1,
                },
            },
        },
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        },
    ];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: PART_STRIDE,
        NumArgumentDescs: arguments.len() as u32,
        pArgumentDescs: arguments.as_ptr(),
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of two arguments that outlives the call; the
    // root signature is live and has that constant.
    match unsafe { device.CreateCommandSignature(&desc, root, &mut signature) } {
        Ok(()) => signature,
        Err(e) => {
            tracing::warn!(error = %e, param, dword, "split dispatch command signature refused");
            None
        }
    }
}

/// [`make_split_signature`] on the host device, made once per root
/// signature and place (the root signature is kept alive with it, so its
/// address names it).
pub(crate) fn split_signature(root: &ID3D12RootSignature, base: (u32, u32)) -> Option<ID3D12CommandSignature> {
    type Made = std::collections::HashMap<(usize, u32, u32), (ID3D12RootSignature, Option<ID3D12CommandSignature>)>;
    static MADE: std::sync::Mutex<Option<Made>> = std::sync::Mutex::new(None);
    use windows::core::Interface;
    let key = (root.as_raw() as usize, base.0, base.1);
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    let made = made.get_or_insert_with(Default::default);
    if let Some((_, signature)) = made.get(&key) {
        return signature.clone();
    }
    let signature = make_split_signature(&host()?.device, root, base);
    made.insert(key, (root.clone(), signature.clone()));
    signature
}

/// [`record_clamp_as`] into `list`'s scratch buffer `dst`, from the state
/// the list left it in; afterwards the list knows it is readable as
/// indirect arguments.
#[allow(clippy::too_many_arguments)]
fn clamp_into(
    list: &HostList,
    c: &Clamp,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    how: u32,
) -> bool {
    let before = list.indirect_state(dst);
    let recorded = record_clamp_how(c, &list.list, (dst, before), budget, src, count, spec, how);
    if recorded {
        list.indirect_readable(dst);
    }
    recorded
}

/// Records the clamp into `dst`, which is in state `before` (common, or
/// indirect arguments after an earlier clamp into it), and leaves `dst` in
/// `INDIRECT_ARGUMENT`. `ahead`: the records are a window clamped ahead of
/// their draws ([`bound_ahead`]). False, with nothing recorded, if the
/// layout or the addresses do not fit the clamp.
#[allow(clippy::too_many_arguments)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
) -> bool {
    record_clamp_how(c, l, dst, budget, src, count, spec, if ahead { AHEAD } else { 0 })
}

/// [`record_clamp_as`] with the bits of `has_count` past the count's:
/// [`AHEAD`] or [`IN_PARTS`] (`records` 1, `kind` 2; `dst` takes
/// [`SPLIT_BYTES`]).
#[allow(clippy::too_many_arguments)]
fn record_clamp_how(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    how: u32,
) -> bool {
    if how & IN_PARTS != 0 && (kind != 2 || records != 1) {
        return false;
    }
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return false;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let constants = [kind, stride, records, args_offset, has_count | how, count_offset];
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, scratch destination, and budget buffers. `src` and `count` stay
    // in `COMMON` / `GENERIC_READ` via implicit buffer promotion while `dst`
    // transitions `before -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`. Moving
    // it out of `INDIRECT_ARGUMENT` orders this clamp's writes after every
    // earlier execution's reads of the scratch buffer; nothing else writes it.
    // Diagnostics: counted; `VICTORY_BARRIER_TEST=indirect` leaves the two
    // waits for every earlier shader write out (the state changes stay).
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let first = [
        uav_barrier(),
        transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    ];
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    let skip = usize::from(!waits);
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        l.ResourceBarrier(&first[skip..]);
        if records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            l.Dispatch(records.div_ceil(64), 1, 1);
        }
        l.ResourceBarrier(&second[skip..]);
    }
    true
}

/// The most records of one argument buffer gathered for a list
/// ([`gather`]): a multiple of the clamp's 64 threads a group, so that the
/// threads past the last record read inside the table.
pub(crate) const GATHER_MOST: u32 = 4096;
/// The widest record gathered: its slot in the scratch buffer.
pub(crate) const GATHER_STRIDE: u32 = 32;

/// The records of one argument buffer that a list's draws read, to be
/// clamped in one pass before the list runs ([`gather`]): the buffer (a
/// host copy, and its number), the records' kind and stride, the scratch
/// buffer whose slot `i` holds record `i` clamped, and the table of the
/// records' offsets in the buffer (upload memory of the list).
pub(crate) struct GatherSet {
    pub(crate) src: ID3D12Resource,
    pub(crate) src_id: u64,
    pub(crate) kind: u32,
    pub(crate) stride: u32,
    pub(crate) scratch: ID3D12Resource,
    table_gpu: u64,
    table: usize,
    /// A record's offset in the buffer, and its slot.
    slots: std::collections::HashMap<u32, u32>,
}

// SAFETY: D3D12 objects are free-threaded; the table is the list's own
// upload memory, written under the list's lock before the list is submitted.
unsafe impl Send for GatherSet {}

impl GatherSet {
    pub(crate) fn new(src: ID3D12Resource, src_id: u64, (kind, stride): (u32, u32), scratch: ID3D12Resource, (table_gpu, table): (u64, *mut u8)) -> Self {
        Self { src, src_id, kind, stride, scratch, table_gpu, table: table as usize, slots: Default::default() }
    }

    /// The records gathered so far.
    pub(crate) fn records(&self) -> u32 {
        self.slots.len() as u32
    }

    /// The slot of the record at `offset`: the one it has, or the next
    /// free one. None when the set is full.
    pub(crate) fn slot(&mut self, offset: u32) -> Option<u32> {
        if let Some(&slot) = self.slots.get(&offset) {
            return Some(slot);
        }
        let slot = self.slots.len() as u32;
        if slot >= GATHER_MOST {
            return None;
        }
        // SAFETY: the table holds `GATHER_MOST` words.
        unsafe { (self.table as *mut u32).add(slot as usize).write_volatile(offset) };
        self.slots.insert(offset, slot);
        Some(slot)
    }

    /// The scratch buffer `src`'s set uses, to keep for the list's next recording.
    pub(crate) fn into_scratch(self) -> ID3D12Resource {
        self.scratch
    }
}

/// Whether [`gather`] can take a record of `kind` and `stride` at `offset`
/// of a buffer at `src_gpu`: what [`record_gather_as`] asks of its
/// arguments, checked when the record is taken, because a record once
/// given a slot is drawn from the slot.
pub(crate) fn gatherable(src_gpu: u64, offset: u64, (kind, stride): (u32, u32)) -> Option<u32> {
    let min = min_stride(kind)?;
    if stride < min || stride > GATHER_STRIDE || !stride.is_multiple_of(4) || !offset.is_multiple_of(4) || !src_gpu.is_multiple_of(4) {
        return None;
    }
    u32::try_from(offset).ok().filter(|o| o.checked_add(stride).is_some())
}

/// For a single record of `kind` and `stride` at `offset` of the host copy
/// `src` (its number `src_id`) that a draw of `list` is about to read: a
/// slot in the list's scratch buffer for that argument buffer, which the
/// list's prelude fills with the record clamped, in one pass with all the
/// other records the list's draws read (`HostList::prelude`,
/// [`record_gathers`]). Nothing is recorded in the list. The scratch
/// buffer and the record's place in it; None if the list no longer
/// gathers (something in it may write arguments), the set is full, or the
/// record does not fit the clamp: the caller clamps it as before.
///
/// Why: the title keeps its mesh records scattered through one argument
/// buffer of about 100 MB (from one draw to the next 16 to 33,056 bytes
/// on, or back), so a window of records at one step cannot serve them, and
/// about 1,450 draws a frame each had a compute pass of ours in front of
/// them: the card left drawing for compute and came back, about nine
/// microseconds each (run 243, and the windows' own measurement).
pub(crate) fn gather(list: &HostList, (src, src_id): (&ID3D12Resource, u64), offset: u64, spec: (u32, u32)) -> Option<(ID3D12Resource, u64)> {
    clamp()?;
    // SAFETY: a live D3D12 buffer resource.
    let Some(offset) = gatherable(unsafe { src.GetGPUVirtualAddress() }, offset, spec) else {
        GATHER_NONE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    };
    list.gather_slot(src, src_id, offset, spec)
}

/// Since the last line: records that got no slot because they do not fit
/// the clamp, because their list had stopped gathering, because the list
/// gathers from too many buffers, because their set was full, because a
/// buffer could not be made.
pub(crate) static GATHER_NONE: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Records the clamp of `records` gathered records into `dst`, which is in
/// state `before`, and leaves `dst` in `INDIRECT_ARGUMENT`: record `i` is
/// read at the offset word `i` of the table at `table_gpu` names, in the
/// buffer `src`, and stored clamped at `i * stride`. Charged to the budget
/// word at `budget` as records clamped one by one are. False, with nothing
/// recorded, if the layout or the addresses do not fit the clamp.
fn record_gather_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    src: &ID3D12Resource,
    table_gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    // SAFETY: live D3D12 buffer resources.
    let (src_gpu, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if gatherable(src_gpu, 0, (kind, stride)).is_none()
        || records == 0
        || records > GATHER_MOST
        || !dst_gpu.is_multiple_of(4)
        || !budget.is_multiple_of(4)
        || !table_gpu.is_multiple_of(4)
    {
        return false;
    }
    let constants = [kind, stride, records, 0u32, 4u32, 0u32];
    let first = [uav_barrier(), transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)];
    let second = [
        uav_barrier(),
        transition(dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
    ];
    // SAFETY: an open list; our root signature and pipeline; live source,
    // table, scratch destination and budget buffers. The first wait puts
    // the pass behind every earlier shader write (the title's argument
    // passes of the lists before); the second puts the list's draws behind
    // the pass.
    unsafe {
        l.ResourceBarrier(&first);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, src_gpu);
        l.SetComputeRootShaderResourceView(2, table_gpu);
        l.SetComputeRootUnorderedAccessView(3, dst_gpu);
        l.SetComputeRootUnorderedAccessView(4, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&second);
    }
    true
}

/// Records, into a list's prelude `l`, the passes that fill the scratch
/// buffers of `sets` ([`gather`]), charged to the words of `budget`, which
/// is zeroed first (the gathered records have a budget of their own: the
/// list's own clamps zero and charge theirs inside the list, after this).
/// False if anything could not be recorded: the slots then hold no clamped
/// records and the list must not run.
pub(crate) fn record_gathers(l: &ID3D12GraphicsCommandList, sets: &[GatherSet], budget: &ID3D12Resource) -> bool {
    use windows::core::Interface;
    let (Some(c), Ok(l2)) = (clamp(), l.cast::<ID3D12GraphicsCommandList2>()) else {
        return false;
    };
    // SAFETY: a live buffer.
    let at = unsafe { budget.GetGPUVirtualAddress() };
    let zeros = [0u64, 4, 8].map(|offset| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: at + offset, Value: 0 });
    // SAFETY: an open list; the budget buffer is the list's own and common
    // between its runs.
    unsafe {
        l.ResourceBarrier(&[transition(budget, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
        l2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
        l.ResourceBarrier(&[transition(budget, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    }
    let mut all = true;
    for set in sets.iter().filter(|s| s.records() > 0) {
        all &= record_gather_as(c, l, (&set.scratch, D3D12_RESOURCE_STATE_COMMON), at, &set.src, set.table_gpu, (set.kind, set.stride, set.records()));
    }
    // SAFETY: as above.
    unsafe {
        l.ResourceBarrier(&[transition(budget, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)]);
    }
    all
}

/// Whether every draw served from a window is checked against a fresh
/// clamp of its record (`VICTORY_INDIRECT_WINDOW_CHECK=1`, with
/// `VICTORY_INDIRECT_WINDOW=1`; off by default). A window clamps its
/// records once, ahead of their draws; if anything wrote a record between
/// the window's clamp and its draw, the draw reads the old one. The check
/// clamps the record again just before the draw (into a buffer of its own,
/// charging a budget of its own) and compares the two on the card: the
/// counts are logged ("window check"), and the first record that differs in
/// each run of a list is named. For telling whether windows can be trusted
/// in a scene before they are on by default; it costs a clamp and a compare
/// per window draw.
pub(crate) fn window_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_INDIRECT_WINDOW_CHECK").is_some_and(|v| v == "1")
    })
}

/// The compare of [`window_check`]: `words` dwords of the record a window
/// clamped (`t0`) against the same record clamped just now (`t1`). Counts
/// the checks and the records that differ in `u0`, keeps the first record
/// that differs there (eight dwords of each), and zeroes the fresh clamp's
/// own budget (`u1`) for the next check.
const CHECK_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Check : register(b0) {
    uint words;
};
ByteAddressBuffer window_record : register(t0);
ByteAddressBuffer fresh_record : register(t1);
RWByteAddressBuffer results : register(u0);
RWByteAddressBuffer fresh_budget : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    bool same = true;
    for (uint i = 0; i < words; i++) {
        if (window_record.Load(4 * i) != fresh_record.Load(4 * i)) {
            same = false;
        }
    }
    uint was;
    results.InterlockedAdd(4, 1, was);
    if (!same) {
        uint before;
        results.InterlockedAdd(0, 1, before);
        if (before == 0) {
            for (uint k = 0; k < min(words, 8u); k++) {
                results.Store(8 + 4 * k, window_record.Load(4 * k));
                results.Store(40 + 4 * k, fresh_record.Load(4 * k));
            }
        }
    }
    fresh_budget.Store3(0, uint3(0, 0, 0));
}
"#;

fn compare() -> Option<&'static Clamp> {
    static COMPARE: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    COMPARE
        .get_or_init(|| {
            let device = &host()?.device;
            match make_from(device, CHECK_SOURCE) {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no window check");
                    None
                }
            }
        })
        .as_ref()
}

/// A list's buffers for [`window_check`]: the fresh clamps' records and
/// budget (the card's memory), and the results (memory the CPU maps:
/// records that differed, checks, then the first differing record as the
/// window had it and as the fresh clamp made it, eight dwords each).
pub(crate) struct WindowCheck {
    args: ID3D12Resource,
    budget: ID3D12Resource,
    results: ID3D12Resource,
    data: usize,
}

// SAFETY: D3D12 objects are free-threaded; the mapping is read only at
// submission, when the list's last run has finished.
unsafe impl Send for WindowCheck {}
// SAFETY: as above.
unsafe impl Sync for WindowCheck {}

impl WindowCheck {
    pub(crate) fn new() -> Option<WindowCheck> {
        let host = host()?;
        let (results, data) = host.shared_buffer(256)?;
        // SAFETY: the mapping holds 256 bytes; nothing runs yet.
        unsafe { std::ptr::write_bytes(data, 0, 256) };
        Some(WindowCheck {
            args: host.uav_buffer(64 << 10)?,
            budget: host.uav_buffer(256)?,
            results,
            data: data as usize,
        })
    }

    /// What the list's last run found, zeroed for the next: checks, records
    /// that differed, and the first that did (as the window had it, as the
    /// fresh clamp made it). Only between runs of the list.
    pub(crate) fn take(&self) -> (u32, u32, [u32; 8], [u32; 8]) {
        let words = self.data as *mut u32;
        // SAFETY: the list's own mapped words; its last run has finished.
        unsafe {
            let at = |k: usize| words.add(k).read_volatile();
            let window = std::array::from_fn(|k| at(2 + k));
            let fresh = std::array::from_fn(|k| at(10 + k));
            let found = (at(1), at(0), window, fresh);
            std::ptr::write_bytes(words, 0, 18);
            found
        }
    }
}

/// Checks and differing records seen by [`window_check`] so far.
static CHECKED: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`window_check`] for a draw served from a window: the record of `kind`
/// at `src`, `step` bytes long, which the window clamped into `scratch` at
/// `at`, is clamped again into the list's check buffer and compared. Runs
/// compute work of ours, so the caller binds its pipeline again after it.
/// False if nothing was recorded.
pub(crate) fn check_hit(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    (kind, step): (u32, u32),
    scratch: &ID3D12Resource,
    at: u64,
) -> bool {
    const CHECK_USE: usize = usize::MAX - 0x78;
    let (Some(c), Some(cmp), Some(check)) = (clamp(), compare(), list.window_check()) else {
        return false;
    };
    if list.first_use(CHECK_USE) {
        // At this recording's submission: what the list's last run found.
        let data = check.data;
        let words = WindowCheck {
            args: check.args.clone(),
            budget: check.budget.clone(),
            results: check.results.clone(),
            data,
        };
        list.defer_host(move || {
            let (checked, differed, window, fresh) = words.take();
            if checked == 0 {
                return;
            }
            use std::sync::atomic::Ordering::Relaxed;
            let total = CHECKED[0].fetch_add(u64::from(checked), Relaxed) + u64::from(checked);
            let bad = CHECKED[1].fetch_add(u64::from(differed), Relaxed) + u64::from(differed);
            if differed > 0 {
                tracing::warn!(
                    differed,
                    checked,
                    ?window,
                    ?fresh,
                    "window check: a draw served from a window read a record that \
                     differs from a fresh clamp of it"
                );
            }
            if total / 10_000 != (total - u64::from(checked)) / 10_000 {
                tracing::info!(checked = total, differed = bad, "window check");
            }
        });
    }
    // SAFETY: live buffers.
    let (budget, results, fresh, window) = unsafe {
        (
            check.budget.GetGPUVirtualAddress(),
            check.results.GetGPUVirtualAddress(),
            check.args.GetGPUVirtualAddress(),
            scratch.GetGPUVirtualAddress() + at,
        )
    };
    // The record again, by the window's rules, into the check's buffer
    // (common before and after every check).
    let common = D3D12_RESOURCE_STATE_COMMON;
    let spec = (kind, step, 1);
    if !record_clamp_as(c, &list.list, (&check.args, common), budget, src, None, spec, true) {
        return false;
    }
    let args = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    let words = step / 4;
    // SAFETY: an open list; the compare's own root signature and pipeline;
    // the scratch buffer is readable as arguments (and stays so), the
    // check's buffer was left so by the clamp above.
    unsafe {
        list.list.ResourceBarrier(&[
            transition(scratch, args, D3D12_RESOURCE_STATES(args.0 | read.0)),
            transition(&check.args, args, read),
        ]);
        list.list.SetComputeRootSignature(&cmp.root);
        list.list.SetPipelineState(&cmp.pso);
        list.list.SetComputeRoot32BitConstants(0, 1, (&raw const words).cast(), 0);
        list.list.SetComputeRootShaderResourceView(1, window);
        list.list.SetComputeRootShaderResourceView(2, fresh);
        list.list.SetComputeRootUnorderedAccessView(3, results);
        list.list.SetComputeRootUnorderedAccessView(4, budget);
        list.list.Dispatch(1, 1, 1);
        list.list.ResourceBarrier(&[
            transition(scratch, D3D12_RESOURCE_STATES(args.0 | read.0), args),
            transition(&check.args, read, common),
        ]);
    }
    true
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The compare of copies in the card's memory (`crate::host`), on the
    /// software adapter. Told 1 it flags the pieces whose bytes are not
    /// what was hashed last and leaves the hashes: a piece a shader changed
    /// and a later one changed back then looks untouched, which is right
    /// only if the piece was fetched after both. Told 2 (submissions sent
    /// ahead with the relay, whose pieces are fetched flight by flight) it
    /// stores the hash it made: each compare flags what changed since the
    /// compare before it, also a change back.
    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::CARD_COMPARE).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // One pass over the three pieces; what it left in `differs` (7s
        // where it stored nothing).
        let run = |warp: &mut WarpHarness, mode: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(PIECES.div_ceil(64), 1, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        let at = PIECE + 1234;
        // SAFETY (each): inside the mapping.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [7, 7, 7]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
    }

    /// The fetch inside flights (`crate::host::CARD_FETCH`), on the software
    /// adapter, in one list behind the compare as `Host::card_before`
    /// records it: each flagged piece takes a ticket and, while it is
    /// within the slots of the lowest ticket in the air, its bytes go to
    /// its slot; an unflagged piece's entry says so; a ticket past the
    /// slots is written but copies nothing.
    #[test]
    fn warp_card_fetch_copies_flagged_pieces_to_their_slots() {
        use crate::host::{CARD_COMPARE, CARD_FETCH, INFLIGHT_NONE, INFLIGHT_SLOT};
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, CARD_COMPARE).expect("the card compare must compile on WARP");
        let fetch = compile_cs(&warp.device, CARD_FETCH).expect("the card fetch must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        // The copy starts 40 bytes into its piece 0: piece 0 is short, the
        // others straddle title pieces and start off a multiple of 16.
        const SKEW: u32 = 40;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES + 1));
        let (_differs, _, differs_gpu) = warp.shared_buffer(64);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(4 + 64);
        let slots = 1u32;
        let (_fetched, fetched_ptr, fetched_gpu) = warp.shared_buffer(INFLIGHT_SLOT * u64::from(slots));
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mappings hold what is written.
        unsafe {
            std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size);
            table_ptr.cast::<u32>().write_volatile(0);
        }
        let pieces = PIECES + 1;
        let run = |warp: &mut WarpHarness, mode: u32, landed: u32| -> Vec<u32> {
            // SAFETY: our open list, live buffers, the shaders' own root
            // signatures; the mappings hold what is read and written.
            unsafe {
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, SKEW, size as u32, pieces, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(1, 1, 1);
                if mode == 2 {
                    warp.list.ResourceBarrier(&[uav_barrier()]);
                    warp.list.SetComputeRootSignature(&fetch.root);
                    warp.list.SetPipelineState(&fetch.pso);
                    // Two groups for four pieces: the loop over pieces.
                    let constants = [0u32, SKEW, size as u32, pieces, 0, landed, slots, 2];
                    warp.list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(2, differs_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(3, table_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(4, fetched_gpu);
                    warp.list.Dispatch(2, 1, 1);
                }
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            (0..=pieces as usize).map(|i| unsafe { table_ptr.cast::<u32>().add(i).read_volatile() }).collect()
        };
        // SAFETY (each): inside the mapping of the copy.
        let set = |at: usize, value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // SAFETY (each): the slot's mapping holds `INFLIGHT_SLOT` bytes.
        let slot = |len: usize| unsafe { std::slice::from_raw_parts(fetched_ptr, len).to_vec() };
        run(&mut warp, 0, 0);
        // Piece 2 of the copy (title piece 2: bytes 2 * 64 KiB - 40 on), its
        // slot from the multiple of 16 below to the one above its end.
        let from = 2 * PIECE - SKEW as usize;
        let widened = from & !15;
        assert_ne!(widened, from);
        set(from + 100, content[from + 100] ^ 0x40);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table, [1, INFLIGHT_NONE, INFLIGHT_NONE, 0, INFLIGHT_NONE]);
        let mut wanted = content[widened..widened + INFLIGHT_SLOT as usize].to_vec();
        wanted[from + 100 - widened] ^= 0x40;
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted, "piece 2 in slot 0");
        // Two pieces flagged with one slot, ticket 0 not landed: tickets 1
        // and 2 get none and the slot keeps what it held.
        set(5, content[5] ^ 1);
        set(PIECE + 7, content[PIECE + 7] ^ 1);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table[0], 3);
        let mut tickets = [table[1], table[2]];
        tickets.sort_unstable();
        assert_eq!(tickets, [1, 2]);
        assert_eq!(&table[3..], [INFLIGHT_NONE, INFLIGHT_NONE]);
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted);
        // Every ticket up to 2 landed: the next one, 3, gets slot 0. Piece
        // 0 starts at the copy's start.
        set(9, content[9] ^ 2);
        let table = run(&mut warp, 2, 3);
        assert_eq!(table, [4, 3, INFLIGHT_NONE, INFLIGHT_NONE, INFLIGHT_NONE]);
        let mut piece0 = content[..PIECE - SKEW as usize].to_vec();
        piece0[5] ^= 1;
        piece0[9] ^= 2;
        assert_eq!(slot(PIECE - SKEW as usize), piece0);
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }

        // Snapshot `dst_buf` after Call #1 (copy 5 * STRIDE bytes, then leave
        // `dst_buf` readable as arguments again, as the clamp left it).
        // SAFETY: open command list; the clamp left `dst_buf` in `INDIRECT_ARGUMENT`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        // The scratch buffer goes straight from arguments back to unordered
        // access.
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged, the slots of all ones come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    #[test]
    fn gathered_records_must_fit_the_clamp() {
        // a mesh record 12 bytes wide at a 4-byte offset of a buffer on a 4-byte address
        assert_eq!(gatherable(0x1000, 0x6d71550, (10, 12)), Some(0x6d71550));
        assert_eq!(gatherable(0x1000, 0, (1, 20)), Some(0));
        // not on four bytes, too narrow, too wide, an unknown kind, past 4 GiB
        assert_eq!(gatherable(0x1000, 0x6d71552, (10, 12)), None);
        assert_eq!(gatherable(0x1002, 0x40, (10, 12)), None);
        assert_eq!(gatherable(0x1000, 0x40, (10, 8)), None);
        assert_eq!(gatherable(0x1000, 0x40, (10, GATHER_STRIDE + 4)), None);
        assert_eq!(gatherable(0x1000, 0x40, (7, 12)), None);
        assert_eq!(gatherable(0x1000, 1 << 32, (10, 12)), None);
        assert_eq!(gatherable(0x1000, u64::from(u32::MAX) - 3, (10, 12)), None);
    }

    #[test]
    fn warp_gathered_records_come_out_as_if_clamped_one_by_one() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 12;
        // Records scattered through a buffer as the title keeps them: 16 to
        // thousands of bytes apart, not in order, one of them twice over
        // the budget's most (65,535 groups), one of all ones (garbage).
        let records: [(u32, [u32; 3]); 7] = [
            (0x3350, [5, 1, 1]),
            (0x0040, [519, 1, 1]),
            (0x0050, [0, 1, 1]),
            (0x7ff0, [70000, 1, 1]),
            (0x1230, [u32::MAX, u32::MAX, u32::MAX]),
            (0x0010, [9, 7, 3]),
            (0x2000, [1, 0, 0]),
        ];
        let total = 0x8000usize;
        let mut src = vec![0x5Au8; total];
        for (at, words) in records {
            for (w, word) in words.into_iter().enumerate() {
                let i = at as usize + w * 4;
                src[i..i + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let slots = (GATHER_MOST * STRIDE) as u64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
        let (_table_buf, table_ptr, table_gpu) = warp.shared_buffer(u64::from(GATHER_MOST) * 4);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(slots);
        let dst_buf = warp.default_uav(slots);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
            std::ptr::write_bytes(table_ptr, 0, GATHER_MOST as usize * 4);
            for (i, (at, _)) in records.into_iter().enumerate() {
                (table_ptr as *mut u32).add(i).write_volatile(at);
            }
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(readback_ptr, 0xEE, slots as usize);
        }
        assert!(record_gather_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            &src_buf,
            table_gpu,
            (KIND_MESH, STRIDE, records.len() as u32),
        ));
        // SAFETY: an open list; the scratch buffer, which the pass left
        // readable as arguments, is copied for the check.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, slots);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (spent, out) = unsafe {
            (
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), records.len() * 3).to_vec(),
            )
        };
        // Slot i holds record i as the clamp leaves a mesh record clamped
        // alone: the groups bounded to 65,535 and the other two to one.
        let want: Vec<[u32; 3]> = records.iter().map(|(_, w)| [w[0].min(65535), w[1].min(1), w[2].min(1)]).collect();
        for (i, want) in want.iter().enumerate() {
            assert_eq!(&out[i * 3..i * 3 + 3], want, "slot {i} (the record at {:#x})", records[i].0);
        }
        let charged: u32 = want.iter().map(|w| w[0] * w[1] * w[2] * 128).sum();
        assert_eq!(spent, charged, "each record charged once, as clamped alone");
        // The source is read only.
        // SAFETY: a mapped buffer.
        assert_eq!(unsafe { std::slice::from_raw_parts(src_ptr.cast_const(), total) }, &src[..]);
    }

    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                ahead,
            ));
            // SAFETY: an open list; the scratch buffer, which the clamp left
            // readable as arguments, is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
            } else {
                // Thirteen slots at 65,535 groups: sixteen would be all of it.
                assert_eq!(spent, charged_filled + 13 * 65535 * 128, "clamped one by one, each slot is charged the most");
                assert_eq!(largest, u32::MAX);
            }
        }
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            // The first record is no draw (2^24 indices): it runs empty
            // and charges nothing.
            let words: [u32; 6] = [
                if i == 0 { 1 << 24 } else { 1 << 22 }, // IndexCount
                if i == 0 { 1 } else { u32::MAX }, // InstanceCount -> clamps to 16 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
            false,
        ));

        // SAFETY: open command list; copy clamped scratch buffer (left
        // readable as arguments) for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 22) && rec[1] == 16 {
                assert!(i == 1 || i == 2, "the first two real draws fit the budget");
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, the scratch buffer's
    /// tracked state, `Host::execute`, and `HostList::reset` end-to-end on
    /// the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        let args = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`, which stays readable
        // as arguments between them.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64)).unwrap();
        assert_eq!(
            list.indirect_state(&b1),
            args,
            "a clamp leaves the scratch readable as arguments"
        );
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128)).unwrap();
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );
        assert_eq!(list.indirect_state(&b2), args);

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset. The new buffer starts out common.
        let grown = list.indirect_scratch(5000 * 20).unwrap();
        assert_eq!(
            list.indirect_state(&grown),
            D3D12_RESOURCE_STATE_COMMON,
            "a new scratch buffer is common"
        );
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000)).unwrap();
        assert_eq!(b3.as_raw(), grown.as_raw());
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );
        assert_eq!(list.indirect_state(&b3), args);
        assert_eq!(
            list.indirect_state(&b1),
            D3D12_RESOURCE_STATE_COMMON,
            "the replaced buffer went back to common"
        );

        list.close();
        // With the debug layer on (`VICTORY_D3D12_DEBUG=1`, diagnostics):
        // every barrier recorded named the state the buffer was in. Read
        // before the submission, which moves the messages to the log.
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations; the buffer decayed to common when the
        // submission ran.
        assert_eq!(
            list.indirect_state(&b3),
            D3D12_RESOURCE_STATE_COMMON,
            "a reset list starts with a common scratch buffer"
        );
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096)).unwrap();
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        assert_eq!(list.indirect_state(&b4), args);
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
    }

    /// A group charges its records in lane order: with 300 left of the
    /// budget, the first draw (100) fits, the second (250) does not and
    /// takes the rest, and the ones after it run empty. (Racing for the
    /// word record by record, which of them fitted was up to the GPU.)
    #[test]
    fn warp_a_group_charges_its_records_in_lane_order() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DRAW: u32 = 0;
        const STRIDE: u32 = 16;
        const BUDGET_CAP: u32 = 1 << 27;
        let counts = [100u32, 250, 50, 100];
        let total = counts.len() * STRIDE as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
        let dst_buf = warp.default_uav(total as u64);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            for (i, &count) in counts.iter().enumerate() {
                let words = [count, 1, 7, 9];
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    src_ptr.add(i * STRIDE as usize),
                    16,
                );
            }
            std::ptr::write_bytes(budget_ptr, 0, 256);
            (budget_ptr as *mut u32).write_volatile(BUDGET_CAP - 300);
        }
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_DRAW, STRIDE, counts.len() as u32),
            false,
        ));
        // SAFETY: an open list; the clamped records are copied for the check.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (spent, out) = unsafe {
            (
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4)
                    .to_vec(),
            )
        };
        assert_eq!(out[0..4], [100, 1, 7, 9], "the first record fits");
        for i in 1..counts.len() {
            assert_eq!(out[i * 4..i * 4 + 4], [0, 0, 7, 9], "record {i} runs empty");
        }
        assert_eq!(spent, BUDGET_CAP, "the record that did not fit took the rest");
    }

    /// The list's budget starts empty at each submission, wherever it
    /// lives (the card's copy is zeroed in the list): a recording whose
    /// clamps spend all of it is followed, after the reset, by one whose
    /// small record still runs.
    #[test]
    fn host_list_budget_starts_empty_at_each_submission() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let _ = host.log_messages();
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(KIND_DISPATCH as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: STRIDE,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: a valid single-argument command signature description.
            unsafe {
                host.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        };
        // Three records of 65,535 x 16 groups: about half the budget each.
        let Some((big, big_ptr)) = host.shared_buffer(3 * u64::from(STRIDE)) else {
            return;
        };
        let Some((small, small_ptr)) = host.shared_buffer(u64::from(STRIDE)) else {
            return;
        };
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet.
        unsafe {
            for i in 0..3 {
                let words = [65535u32, 16, 1];
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    big_ptr.add(i * STRIDE as usize),
                    12,
                );
            }
            std::ptr::copy_nonoverlapping([1u32, 1, 1].as_ptr().cast::<u8>(), small_ptr, 12);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for round in 0..2 {
            // Spends the budget (clamped, not executed), then one group.
            bound(&list, (&big, 0), None, (KIND_DISPATCH, STRIDE, 3)).unwrap();
            let scratch = bound(&list, (&small, 0), None, (KIND_DISPATCH, STRIDE, 1)).unwrap();
            // SAFETY: an open list; the worker's root signature and
            // pipeline; buffers the test holds until the list has run.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                list.list.ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
            }
            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            // The small record comes after the big ones spent the budget:
            // it runs empty, in every round alike (none carries over).
            assert_eq!(hits, 0, "round {round}: the spent budget empties the record after it");
        }
        // A recording with only the small record: the budget is empty
        // again, so it runs.
        let scratch = bound(&list, (&small, 0), None, (KIND_DISPATCH, STRIDE, 1)).unwrap();
        // SAFETY: as above.
        unsafe {
            list.list.SetComputeRootSignature(&worker.root);
            list.list.SetPipelineState(&worker.pso);
            list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            list.list.ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
        }
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
        // SAFETY: the list has run; a mapped buffer.
        let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
        assert_eq!(hits, 1, "the budget starts empty at the next submission");
    }

    /// The window check: a record nothing wrote since its window clamped
    /// it compares equal; one written on the card after the window's clamp
    /// (an untracked writer) is found, and named as the window had it and
    /// as it is now.
    #[test]
    fn host_list_window_check_finds_a_record_written_under_its_window() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let _ = host.log_messages();
        const KIND_DISPATCH: u32 = 2;
        const STEP: u32 = 32;
        let records = [[3u32, 1, 1], [5, 1, 1], [7, 1, 1], [9, 1, 1]];
        let total = records.len() * STEP as usize;
        let (Some((src, src_ptr)), Some((patch, patch_ptr))) =
            (host.shared_buffer(total as u64), host.shared_buffer(64))
        else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0, total);
            for (i, words) in records.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    src_ptr.add(i * STEP as usize),
                    12,
                );
            }
            std::ptr::copy_nonoverlapping([11u32, 2, 1].as_ptr().cast::<u8>(), patch_ptr, 12);
        }
        let spec = (KIND_DISPATCH, STEP, records.len() as u32);
        let scratch = bound_ahead(&list, (&src, 0), spec).unwrap();
        // Record 0: nothing wrote it.
        assert!(check_hit(&list, (&src, 0), (KIND_DISPATCH, STEP), &scratch, 0));
        // Record 1 is written on the card after the window's clamp.
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        // SAFETY: an open list; `src` was read by the clamp (promoted to a
        // shader read), `patch` is common; both live until the list has run.
        unsafe {
            list.list.ResourceBarrier(&[transition(&src, read, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.list.CopyBufferRegion(&src, u64::from(STEP), &patch, 0, 12);
            list.list.ResourceBarrier(&[transition(&src, D3D12_RESOURCE_STATE_COPY_DEST, read)]);
        }
        let at = u64::from(STEP);
        assert!(check_hit(&list, (&src, at), (KIND_DISPATCH, STEP), &scratch, at));
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        let (checked, differed, window, fresh) = list.window_check().unwrap().take();
        list.reset();
        assert_eq!((checked, differed), (2, 1), "two checks, the written record differs");
        assert_eq!(window[..3], [5, 1, 1], "as the window clamped it");
        assert_eq!(fresh[..3], [11, 2, 1], "as it is now");
    }

    /// Records the dispatch records `groups` (three words each, `step`
    /// bytes apart, the rest of each record ones) into a fresh mapped
    /// buffer of `warp`'s device.
    fn dispatch_records(warp: &WarpHarness, groups: &[u32], step: u32) -> ID3D12Resource {
        let total = groups.len() * step as usize;
        let (buffer, ptr, _) = warp.shared_buffer(total as u64);
        let mut bytes = vec![0xffu8; total];
        for (i, &g) in groups.iter().enumerate() {
            for (w, word) in [g, 1, 1].into_iter().enumerate() {
                let at = i * step as usize + w * 4;
                bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        // SAFETY: the mapping holds `total` bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, total) };
        buffer
    }

    /// Two windows in one list, as `crate::draw` records them: each clamps
    /// its records ahead, then one execution per record reads the scratch
    /// buffer with no barrier between them; the second clamp moves the
    /// scratch buffer straight from arguments back to unordered access.
    /// Every record runs once, with the group count it asked for.
    #[test]
    fn warp_window_executions_read_their_records_with_no_barrier_between() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        // The title names 12 bytes and keeps its records 32 apart.
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let sources: Vec<ID3D12Resource> =
            windows.iter().map(|g| dispatch_records(&warp, g, STEP)).collect();
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let dst_buf = warp.default_uav(u64::from(STEP) * 4);
        let sig = warp.signature(KIND_DISPATCH, STRIDE);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        let mut state = D3D12_RESOURCE_STATE_COMMON;
        for (groups, src) in windows.iter().zip(&sources) {
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, state),
                budget_gpu,
                (src, 0),
                None,
                (KIND_DISPATCH, STEP, groups.len() as u32),
                true,
            ));
            state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            // SAFETY: an open list; the worker's root signature and
            // pipeline; the scratch buffer is readable as arguments.
            unsafe {
                warp.list.SetComputeRootSignature(&warp.worker.root);
                warp.list.SetPipelineState(&warp.worker.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                for i in 0..groups.len() as u64 {
                    warp.list.ExecuteIndirect(&sig, 1, &dst_buf, i * u64::from(STEP), None, 0);
                }
            }
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (hits, spent) = unsafe {
            (
                (hits_ptr as *const u32).read_volatile(),
                (budget_ptr as *const u32).read_volatile(),
            )
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        assert_eq!(hits, expected, "every record of both windows runs once");
        assert_eq!(spent, expected * 64, "and is charged what it asked for");
    }

    /// The windows of `warp_window_executions_read_their_records_with_no_barrier_between`
    /// through a host list (`bound_ahead`, the list's tracked scratch state,
    /// `Host::execute`), twice: on the first recording the scratch buffer
    /// starts common, on the second (after the reset) it starts common
    /// again. With the debug layer on (`VICTORY_D3D12_DEBUG=1` in a
    /// diagnostics build) no barrier names a state the buffer is not in.
    #[test]
    fn host_list_window_executions_need_no_barrier_of_their_own() {
        let Some(host) = super::host() else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(KIND_DISPATCH as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: STRIDE,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: a valid single-argument command signature description.
            unsafe {
                host.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        };
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let mut sources = Vec::new();
        for groups in windows {
            let total = groups.len() * STEP as usize;
            let Some((buffer, ptr)) = host.shared_buffer(total as u64) else {
                return;
            };
            let mut bytes = vec![0xffu8; total];
            for (i, &g) in groups.iter().enumerate() {
                for (w, word) in [g, 1, 1].into_iter().enumerate() {
                    let at = i * STEP as usize + w * 4;
                    bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
            // SAFETY: the mapping holds `total` bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, total) };
            sources.push(buffer);
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for _ in 0..2 {
            // SAFETY: the mapping holds 256 bytes; nothing runs.
            unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
            for (groups, src) in windows.iter().zip(&sources) {
                let spec = (KIND_DISPATCH, STEP, groups.len() as u32);
                let scratch = bound_ahead(&list, (src, 0), spec).unwrap();
                assert_eq!(list.indirect_state(&scratch), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                // SAFETY: an open list; the worker's root signature and
                // pipeline; buffers the test holds until the list has run.
                unsafe {
                    list.list.SetComputeRootSignature(&worker.root);
                    list.list.SetPipelineState(&worker.pso);
                    list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                    for i in 0..groups.len() as u64 {
                        let at = i * u64::from(STEP);
                        list.list.ExecuteIndirect(&signature, 1, &scratch, at, None, 0);
                    }
                }
            }
            list.close();
            // With the debug layer on: nothing it objects to was recorded.
            // Read before the submission, which moves the messages to the log.
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            assert_eq!(hits, expected, "every record of both windows runs once");
        }
    }

    /// A program that counts each of its groups at its own slot: its number
    /// in x plus the number of its part's first group (dword 2, where a
    /// translated program reads `USER_GROUP_BASE`), in rows of dword 0
    /// groups, planes of dword 1 rows.
    const PARTS_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), UAV(u0)"
cbuffer Part : register(b0) {
    uint row;
    uint rows;
    uint first;
};
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    uint was;
    hits.InterlockedAdd(((g.z * rows + g.y) * row + g.x + first) * 4u, 1u, was);
}
"#;

    /// The slots of `PARTS_CS` that did not count one group, and what they
    /// counted (the first eight), of `total`; and whether the slot past
    /// them is untouched.
    fn not_once(hits: *const u8, total: usize) -> (Vec<(usize, u32)>, bool) {
        // SAFETY: the caller's mapping holds `total + 1` words.
        let words = unsafe { std::slice::from_raw_parts(hits.cast::<u32>(), total + 1) };
        let wrong = words[..total].iter().enumerate().filter(|&(_, &h)| h != 1).map(|(i, &h)| (i, h)).take(8).collect();
        (wrong, words[total] == 0)
    }

    /// Up to 65,535 groups in x a dispatch is one part; past that as many
    /// as cover them, at most `MOST_PARTS`.
    #[test]
    fn small_dispatches_are_one_part_and_large_ones_as_many_as_cover_them() {
        assert_eq!(PART, 65_535);
        for (x, parts) in [
            (0, 1),
            (1, 1),
            (65_535, 1),
            (65_536, 2),
            (98_000, 2),
            (131_070, 2),
            (131_071, 3),
            (MOST_PARTS * PART, MOST_PARTS),
            (MOST_PARTS * PART + 1, MOST_PARTS),
            (u32::MAX, MOST_PARTS),
        ] {
            assert_eq!(parts_of(x), parts, "{x} groups");
        }
        assert_eq!(PARTS_COUNT_AT, u64::from(MOST_PARTS) * 16);
    }

    /// A direct dispatch of 98,000 groups (and others) through
    /// `record_parts`: every group runs exactly once (its slot counts 1,
    /// the slot past them 0), and the number of the first group is 0 again
    /// afterwards (one more group lands on slot 0). A dispatch of 65,535
    /// groups or fewer is one dispatch. A program without a group base (one
    /// that does not read its x number) runs the parts all the same.
    #[test]
    fn warp_dispatch_in_parts_runs_every_group_once() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let counter = compile_cs(&warp.device, PARTS_CS).expect("the counting shader must compile");
        let cases: [([u32; 3], u32); 6] = [
            ([98_000, 1, 1], 2),
            ([65_535, 1, 1], 1),
            ([65_536, 1, 1], 2),
            ([1_000, 3, 2], 1),
            ([70_000, 2, 1], 2),
            ([200_000, 1, 1], 4),
        ];
        for (groups, parts) in cases {
            let total = groups.iter().product::<u32>() as usize;
            let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(4 * (total as u64 + 1));
            // SAFETY: the mapping holds `total + 1` words; our open list,
            // the counter's own root signature and pipeline.
            unsafe {
                std::ptr::write_bytes(hits_ptr, 0, 4 * (total + 1));
                warp.list.SetComputeRootSignature(&counter.root);
                warp.list.SetPipelineState(&counter.pso);
                let constants = [groups[0], groups[1], 0];
                warp.list.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootUnorderedAccessView(1, hits_gpu);
            }
            let ran = record_parts(&warp.list, Some((0, 2)), groups);
            // SAFETY: as above.
            unsafe { warp.list.Dispatch(1, 1, 1) };
            warp.submit_and_wait();
            assert_eq!(ran, parts, "{groups:?}");
            // SAFETY: the mapping holds `total + 1` words.
            let first = unsafe { hits_ptr.cast::<u32>().read_volatile() };
            assert_eq!(first, 2, "{groups:?}: group 0, and the one after the parts with the base back at 0");
            // SAFETY: the mapping holds `total + 1` words.
            unsafe { hits_ptr.cast::<u32>().write_volatile(1) };
            let (wrong, past) = not_once(hits_ptr, total);
            assert!(wrong.is_empty(), "{groups:?}: slots that did not count one group: {wrong:?}");
            assert!(past, "{groups:?}: a group past the dispatch ran");
        }
        // No group base: the groups' count is what counts.
        let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        // SAFETY: the mapping holds 256 bytes; our open list, the worker's
        // own root signature and pipeline.
        unsafe {
            std::ptr::write_bytes(hits_ptr, 0, 256);
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
        }
        assert_eq!(record_parts(&warp.list, None, [98_000, 1, 1]), 2);
        warp.submit_and_wait();
        // SAFETY: as above.
        assert_eq!(unsafe { hits_ptr.cast::<u32>().read_volatile() }, 98_000);
    }

    /// An indirect dispatch record clamped in parts (`IN_PARTS`) and run as
    /// one execution through the split command signature (a root constant,
    /// then the dispatch) with the count the clamp wrote: every group of
    /// 98,000 (and of the other records) runs exactly once; the parts'
    /// records, their count and the budget's notes (words 3 to 6) are as
    /// asked; a record past what the parts run is cut to one part, as
    /// before, and counted as cut; small records are one part as they were.
    /// Then the same record run by a program without a group base, from the
    /// records' second words (a plain dispatch signature, 16 bytes apart).
    #[test]
    fn warp_indirect_dispatch_in_parts_runs_every_group_once() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let counter = compile_cs(&warp.device, PARTS_CS).expect("the counting shader must compile");
        let signature = make_split_signature(&warp.device, &counter.root, (0, 2)).expect("the split command signature");
        let plain = warp.signature(2, PART_STRIDE);
        const LAST: u32 = MOST_PARTS * PART;
        // (asked, parts, groups in x run, budget words 3 to 6: the largest
        // asked in x, split, parts, cut)
        let cases: [([u32; 3], u32, u32, [u32; 4]); 8] = [
            ([98_000, 1, 1], 2, 98_000, [98_000, 1, 2, 0]),
            ([65_535, 1, 1], 1, 65_535, [65_535, 0, 0, 0]),
            ([65_536, 1, 1], 2, 65_536, [65_536, 1, 2, 0]),
            ([5, 3, 2], 1, 5, [5, 0, 0, 0]),
            ([0, 1, 1], 0, 0, [0, 0, 0, 0]),
            ([70_000, 2, 1], 2, 70_000, [70_000, 1, 2, 0]),
            ([LAST, 1, 1], MOST_PARTS, LAST, [LAST, 1, MOST_PARTS, 0]),
            ([LAST + 1, 1, 1], 1, PART, [LAST + 1, 0, 0, 1]),
        ];
        for (asked, parts, run, notes) in cases {
            let total = (run * asked[1] * asked[2]) as usize;
            let (src, src_ptr, _) = warp.shared_buffer(16);
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(4 * (total as u64 + 1));
            let (readback, readback_ptr, _) = warp.shared_buffer(SPLIT_BYTES);
            let dst = warp.default_uav(SPLIT_BYTES);
            // SAFETY: mappings of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr, 12);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(hits_ptr, 0, 4 * (total + 1));
                std::ptr::write_bytes(readback_ptr, 0xEE, SPLIT_BYTES as usize);
            }
            let common = D3D12_RESOURCE_STATE_COMMON;
            assert!(record_clamp_how(&warp.clamp, &warp.list, (&dst, common), budget_gpu, (&src, 0), None, (2, 12, 1), IN_PARTS));
            // SAFETY: our open list; the counter's own root signature and
            // pipeline; the clamp left `dst` readable as arguments.
            unsafe {
                warp.list.SetComputeRootSignature(&counter.root);
                warp.list.SetPipelineState(&counter.pso);
                let constants = [run, asked[1], 0];
                warp.list.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootUnorderedAccessView(1, hits_gpu);
                warp.list.ExecuteIndirect(&signature, MOST_PARTS, &dst, 0, Some(&dst), PARTS_COUNT_AT);
                warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                warp.list.CopyBufferRegion(&readback, 0, &dst, 0, SPLIT_BYTES);
            }
            warp.submit_and_wait();
            // SAFETY: the mapping holds what the copy wrote.
            let made: Vec<u32> = unsafe { std::slice::from_raw_parts(readback_ptr.cast::<u32>(), SPLIT_BYTES as usize / 4).to_vec() };
            assert_eq!(made[(PARTS_COUNT_AT / 4) as usize], parts, "{asked:?}: the parts' count");
            for part in 0..MOST_PARTS {
                let first = part * PART;
                let x = if run > first { (run - first).min(PART) } else { 0 };
                let at = (part * 4) as usize;
                assert_eq!(made[at..at + 4], [first, x, asked[1], asked[2]], "{asked:?}: part {part}");
            }
            let (wrong, past) = not_once(hits_ptr, total);
            assert!(wrong.is_empty(), "{asked:?}: slots that did not count one group: {wrong:?}");
            assert!(past, "{asked:?}: a group past the dispatch ran");
            // SAFETY: the mapping holds 256 bytes.
            let words = unsafe { budget_ptr.cast::<[u32; 7]>().read_volatile() };
            assert_eq!(words[3..7], notes, "{asked:?}: the budget's notes");
            assert_eq!(words[0], run * asked[1] * asked[2] * 64, "{asked:?}: charged what ran");
        }
        // Without a group base: the record from its parts' second words.
        let asked = [98_000u32, 1, 1];
        let (src, src_ptr, _) = warp.shared_buffer(16);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let dst = warp.default_uav(SPLIT_BYTES);
        // SAFETY: mappings of these sizes.
        unsafe {
            std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr, 12);
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        let common = D3D12_RESOURCE_STATE_COMMON;
        assert!(record_clamp_how(&warp.clamp, &warp.list, (&dst, common), budget_gpu, (&src, 0), None, (2, 12, 1), IN_PARTS));
        // SAFETY: our open list; the worker's own root signature and
        // pipeline; the clamp left `dst` readable as arguments.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(&plain, MOST_PARTS, &dst, 4, Some(&dst), PARTS_COUNT_AT);
        }
        warp.submit_and_wait();
        // SAFETY: as above.
        assert_eq!(unsafe { hits_ptr.cast::<u32>().read_volatile() }, 98_000);
    }

    /// A count buffer that says no record: no part runs (the count the
    /// clamp writes is 0, not what the scratch held); one that says one
    /// record: all of them run.
    #[test]
    fn warp_indirect_dispatch_in_parts_past_the_count_runs_nothing() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let counter = compile_cs(&warp.device, PARTS_CS).expect("the counting shader must compile");
        let signature = make_split_signature(&warp.device, &counter.root, (0, 2)).expect("the split command signature");
        let (src, src_ptr, _) = warp.shared_buffer(16);
        let (count, count_ptr, _) = warp.shared_buffer(16);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(4 * 98_001);
        let dst = warp.default_uav(SPLIT_BYTES);
        // The count of a run before: what a stale scratch would run.
        let mut before = D3D12_RESOURCE_STATE_COMMON;
        for (records, ran) in [(1u32, 1u32), (0, 0), (1, 1)] {
            // SAFETY: mappings of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping([98_000u32, 1, 1].as_ptr().cast(), src_ptr, 12);
                count_ptr.cast::<u32>().write_volatile(records);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(hits_ptr, 0, 4 * 98_001);
            }
            assert!(record_clamp_how(&warp.clamp, &warp.list, (&dst, before), budget_gpu, (&src, 0), Some((&count, 0)), (2, 12, 1), IN_PARTS));
            before = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            // SAFETY: as in the test above.
            unsafe {
                warp.list.SetComputeRootSignature(&counter.root);
                warp.list.SetPipelineState(&counter.pso);
                let constants = [98_000u32, 1, 0];
                warp.list.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootUnorderedAccessView(1, hits_gpu);
                warp.list.ExecuteIndirect(&signature, MOST_PARTS, &dst, 0, Some(&dst), PARTS_COUNT_AT);
            }
            warp.submit_and_wait();
            // SAFETY: the mapping holds 98,001 words.
            let counted: u32 = unsafe { std::slice::from_raw_parts(hits_ptr.cast::<u32>(), 98_001) }.iter().sum();
            assert_eq!(counted, ran * 98_000, "count {records}");
        }
    }

    /// `VICTORY_DISPATCH_PARTS`: `0` none, `all` every single-record
    /// dispatch, anything else (or unset) only programs seen asking for
    /// more.
    #[test]
    fn the_parts_switch_reads_off_all_and_seen() {
        assert_eq!(parse_parts(None), Parts::Seen);
        assert_eq!(parse_parts(Some("seen")), Parts::Seen);
        assert_eq!(parse_parts(Some(" seen ")), Parts::Seen);
        assert_eq!(parse_parts(Some("1")), Parts::Seen);
        assert_eq!(parse_parts(Some("all")), Parts::All);
        assert_eq!(parse_parts(Some(" all ")), Parts::All);
        assert_eq!(parse_parts(Some("0")), Parts::Off);
    }

    /// A program keeps its number; once the clamp notes it (budget words 7
    /// and on, a list late), its dispatches run in parts; a program not
    /// noted does not, nor does a number past the notes the list kept.
    #[test]
    fn a_program_the_clamp_noted_runs_in_parts_from_then_on() {
        let (a, seen) = program_parts("test program a, noted");
        assert!(!seen && a != 0);
        assert_eq!(program_parts("test program a, noted"), (a, false));
        let (b, _) = program_parts("test program b, not noted");
        let (c, _) = program_parts("test program c, past the kept notes");
        assert!(b != a && c != a && c != b);
        let mut words = [0u32; DISPATCH_WORDS];
        // Nine noted, the first eight kept: `a` and seven of nobody's.
        words[4] = 9;
        words[5] = a;
        words[6..].fill(u32::MAX);
        note_budget_words(words);
        assert_eq!(program_parts("test program a, noted"), (a, true));
        assert_eq!(program_parts("test program b, not noted"), (b, false));
        assert_eq!(program_parts("test program c, past the kept notes"), (c, false));
    }

    /// The clamp, told a program (`bound_noted`), clamps a record as
    /// before and notes the program (budget word 7 counts, words 8 to 15
    /// keep the first eight) when the record asks for more groups in x than
    /// one dispatch runs and no more than the parts run; not for a record
    /// within one dispatch, past the parts, or past it in y only, nor when
    /// told no program.
    #[test]
    fn warp_clamp_notes_the_program_of_a_record_past_one_dispatch() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const PROGRAM: u32 = 5;
        const LAST: u32 = MOST_PARTS * PART;
        // (records, each asking, the program told, records cut, notes
        // counted, the record clamped)
        let cases: [(u32, [u32; 3], u32, u32, u32, [u32; 3]); 6] = [
            (1, [98_000, 1, 1], PROGRAM, 1, 1, [PART, 1, 1]),
            (1, [65_535, 1, 1], PROGRAM, 0, 0, [PART, 1, 1]),
            (1, [LAST + 1, 1, 1], PROGRAM, 1, 0, [PART, 1, 1]),
            (1, [5, 70_000, 1], PROGRAM, 1, 0, [5, PART, 1]),
            (1, [98_000, 1, 1], 0, 1, 0, [PART, 1, 1]),
            (10, [98_000, 1, 1], PROGRAM, 10, 10, [PART, 1, 1]),
        ];
        for (records, asked, program, cut, noted, clamped) in cases {
            let (src, src_ptr, _) = warp.shared_buffer(12 * u64::from(records));
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback, readback_ptr, _) = warp.shared_buffer(12);
            let dst = warp.default_uav(12 * u64::from(records));
            // SAFETY: mappings of these sizes.
            unsafe {
                for r in 0..records as usize {
                    std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr.add(12 * r), 12);
                }
                std::ptr::write_bytes(budget_ptr, 0, 256);
            }
            let common = D3D12_RESOURCE_STATE_COMMON;
            let how = program << PROGRAM_SHIFT;
            assert!(record_clamp_how(&warp.clamp, &warp.list, (&dst, common), budget_gpu, (&src, 0), None, (2, 12, records), how));
            // SAFETY: our open list; the clamp left `dst` readable as arguments.
            unsafe {
                warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                warp.list.CopyBufferRegion(&readback, 0, &dst, 0, 12);
            }
            warp.submit_and_wait();
            // SAFETY: the mappings hold what the clamp and the copy wrote.
            let (words, made) = unsafe { (budget_ptr.cast::<[u32; 17]>().read_volatile(), readback_ptr.cast::<[u32; 3]>().read_volatile()) };
            let case = format!("{records} x {asked:?}, program {program}");
            assert_eq!(made, clamped, "{case}: clamped as before");
            assert_eq!(words[6], cut, "{case}: cut");
            assert_eq!(words[7], noted, "{case}: notes counted");
            let kept = (noted as usize).min(NOTED);
            assert!(words[8..8 + kept].iter().all(|&w| w == PROGRAM), "{case}: notes kept {:?}", &words[8..16]);
            assert!(words[8 + kept..17].iter().all(|&w| w == 0), "{case}: nothing past the notes {:?}", &words[8..17]);
        }
    }

    /// `VICTORY_DISPATCH_PLAIN`: any of `split`, `wait`, `after`, `rootsig`,
    /// or nothing; `VICTORY_DISPATCH_PLAIN_ONLY`: hash beginnings.
    #[test]
    fn the_plain_switch_reads_its_variants() {
        assert_eq!(parse_plain(None), Plain::default());
        assert_eq!(parse_plain(Some("split")), Plain { split: true, ..Plain::default() });
        assert_eq!(parse_plain(Some("wait")), Plain { wait: true, ..Plain::default() });
        assert_eq!(parse_plain(Some("after")), Plain { after: true, ..Plain::default() });
        assert_eq!(parse_plain(Some("rootsig")), Plain { rootsig: true, ..Plain::default() });
        let all = parse_plain(Some(" wait , split,after,rootsig "));
        assert_eq!(all, Plain { split: true, wait: true, after: true, rootsig: true });
        assert_eq!(all.names(), "split,wait,after,rootsig");
        assert_eq!(parse_plain(Some("nothing")), Plain::default());
        assert!(parse_only(None).is_empty());
        assert_eq!(parse_only(Some(" 2dc364df, ab ,")), ["2dc364df", "ab"]);
    }

    /// `VICTORY_DISPATCH_PLAIN=rootsig`: the one-part records
    /// (`bound_split_one`) run through the parted path's command signature,
    /// which sets the first group's number to 0: a small record runs every
    /// group once, a record past one dispatch runs its first 65,535 groups
    /// once (cut, as without the parts).
    #[test]
    fn warp_one_part_through_the_root_constant_signature_runs_its_groups_once() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let counter = compile_cs(&warp.device, PARTS_CS).expect("the counting shader must compile");
        let signature = make_split_signature(&warp.device, &counter.root, (0, 2)).expect("the split command signature");
        // (asked, groups that run in x)
        for (asked, run) in [([1_000u32, 3, 1], 1_000u32), ([98_000, 1, 1], PART)] {
            let all = (asked[0] * asked[1] * asked[2]) as usize;
            let total = (run * asked[1] * asked[2]) as usize;
            let (src, src_ptr, _) = warp.shared_buffer(16);
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(4 * (all as u64 + 1));
            let dst = warp.default_uav(SPLIT_BYTES);
            // SAFETY: mappings of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr, 12);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(hits_ptr, 0, 4 * (all + 1));
            }
            let common = D3D12_RESOURCE_STATE_COMMON;
            assert!(record_clamp_how(&warp.clamp, &warp.list, (&dst, common), budget_gpu, (&src, 0), None, (2, 12, 1), IN_PARTS | ONE_PART));
            // SAFETY: our open list; the counter's own root signature and
            // pipeline; the clamp left `dst` readable as arguments.
            unsafe {
                warp.list.SetComputeRootSignature(&counter.root);
                warp.list.SetPipelineState(&counter.pso);
                let constants = [run, asked[1], 0];
                warp.list.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootUnorderedAccessView(1, hits_gpu);
            }
            execute_parts(&warp.list, &signature, Some((0, 2)), &dst, 0);
            warp.submit_and_wait();
            let (wrong, past) = not_once(hits_ptr, total);
            assert!(wrong.is_empty(), "{asked:?}: slots that did not count one group: {wrong:?}");
            assert!(past, "{asked:?}: a group past what runs ran");
        }
    }

    /// After an execution of parts whose command signature sets the number
    /// of each part's first group (`execute_parts`), a plain dispatch of the
    /// same program, direct or indirect, with no constant set again in
    /// between, counts its groups from 0: the number is 0 again, not the
    /// last part's (D3D12 leaves a root argument a command signature sets
    /// undefined). Every group of all three runs exactly once. (WARP gives
    /// back the value set before the execution: the test passed with the
    /// reset taken out too, so it guards the plain dispatches here but
    /// cannot show a driver that leaves the last part's number.)
    #[test]
    fn warp_plain_dispatches_after_parts_count_from_group_0() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let counter = compile_cs(&warp.device, PARTS_CS).expect("the counting shader must compile");
        let signature = make_split_signature(&warp.device, &counter.root, (0, 2)).expect("the split command signature");
        let plain = warp.signature(2, 12);
        const GROUPS: u32 = 98_000;
        const DIRECT: u32 = 1_000;
        const INDIRECT: u32 = 500;
        let (src, src_ptr, _) = warp.shared_buffer(16);
        let (args, args_ptr, _) = warp.shared_buffer(16);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(4 * (u64::from(GROUPS) + 1));
        let dst = warp.default_uav(SPLIT_BYTES);
        // SAFETY: mappings of these sizes.
        unsafe {
            std::ptr::copy_nonoverlapping([GROUPS, 1, 1].as_ptr().cast(), src_ptr, 12);
            std::ptr::copy_nonoverlapping([INDIRECT, 1, 1].as_ptr().cast(), args_ptr, 12);
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(hits_ptr, 0, 4 * (GROUPS as usize + 1));
        }
        let common = D3D12_RESOURCE_STATE_COMMON;
        assert!(record_clamp_how(&warp.clamp, &warp.list, (&dst, common), budget_gpu, (&src, 0), None, (2, 12, 1), IN_PARTS));
        // SAFETY: our open list; the counter's own root signature and
        // pipeline; the clamp left `dst` readable as arguments, `args` is
        // read as arguments from the common state.
        unsafe {
            warp.list.SetComputeRootSignature(&counter.root);
            warp.list.SetPipelineState(&counter.pso);
            let constants = [GROUPS, 1, 0];
            warp.list.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
            warp.list.SetComputeRootUnorderedAccessView(1, hits_gpu);
        }
        execute_parts(&warp.list, &signature, Some((0, 2)), &dst, 0);
        // SAFETY: as above; nothing set again.
        unsafe {
            warp.list.Dispatch(DIRECT, 1, 1);
            warp.list.ExecuteIndirect(&plain, 1, &args, 0, None, 0);
        }
        warp.submit_and_wait();
        // SAFETY: the mapping holds `GROUPS + 1` words.
        let hits = unsafe { std::slice::from_raw_parts(hits_ptr.cast::<u32>(), GROUPS as usize + 1) };
        let want = |slot: u32| match slot {
            s if s < INDIRECT => 3,
            s if s < DIRECT => 2,
            s if s < GROUPS => 1,
            _ => 0,
        };
        let wrong: Vec<(u32, u32, u32)> =
            (0..=GROUPS).filter(|&s| hits[s as usize] != want(s)).map(|s| (s, hits[s as usize], want(s))).take(8).collect();
        assert!(wrong.is_empty(), "slots (slot, counted, wanted): {wrong:?}");
    }

    /// `VICTORY_DISPATCH_PLAIN=split` (`bound_split_one`): the parts'
    /// records and count, one part at most: a record past one dispatch is
    /// cut to 65,535 groups as without the parts and its program noted; a
    /// small one is one part as it asked. Run from the records' second
    /// words by a plain signature of 16 bytes, the cut record runs 65,535
    /// groups.
    #[test]
    fn warp_one_part_records_cut_and_note_the_program() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const PROGRAM: u32 = 7;
        let plain = warp.signature(2, PART_STRIDE);
        // (asked, records, cut, notes, groups run)
        let cases: [([u32; 3], [u32; 3], u32, u32, u32); 3] = [
            ([98_000, 1, 1], [PART, 1, 1], 1, 1, PART),
            ([5, 3, 2], [5, 3, 2], 0, 0, 30),
            ([0, 1, 1], [0, 1, 1], 0, 0, 0),
        ];
        for (asked, clamped, cut, noted, ran) in cases {
            let (src, src_ptr, _) = warp.shared_buffer(16);
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(256);
            let (readback, readback_ptr, _) = warp.shared_buffer(SPLIT_BYTES);
            let dst = warp.default_uav(SPLIT_BYTES);
            // SAFETY: mappings of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr, 12);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(hits_ptr, 0, 256);
            }
            let common = D3D12_RESOURCE_STATE_COMMON;
            let how = IN_PARTS | ONE_PART | (PROGRAM << PROGRAM_SHIFT);
            assert!(record_clamp_how(&warp.clamp, &warp.list, (&dst, common), budget_gpu, (&src, 0), None, (2, 12, 1), how));
            // SAFETY: our open list; the worker's own root signature and
            // pipeline; the clamp left `dst` readable as arguments.
            unsafe {
                warp.list.SetComputeRootSignature(&warp.worker.root);
                warp.list.SetPipelineState(&warp.worker.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            }
            execute_parts(&warp.list, &plain, None, &dst, 0);
            // SAFETY: as above.
            unsafe {
                warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                warp.list.CopyBufferRegion(&readback, 0, &dst, 0, SPLIT_BYTES);
            }
            warp.submit_and_wait();
            // SAFETY: the mappings hold what the clamp, the copy and the
            // worker wrote.
            let (made, words, hits) = unsafe {
                (
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>(), SPLIT_BYTES as usize / 4).to_vec(),
                    budget_ptr.cast::<[u32; 16]>().read_volatile(),
                    hits_ptr.cast::<u32>().read_volatile(),
                )
            };
            let parts = u32::from(clamped.iter().all(|&g| g > 0));
            assert_eq!(made[(PARTS_COUNT_AT / 4) as usize], parts, "{asked:?}: one part at most");
            assert_eq!(made[..4], [0, clamped[0], clamped[1], clamped[2]], "{asked:?}: the part");
            assert_eq!(made[4..8], [PART, 0, clamped[1], clamped[2]], "{asked:?}: nothing in the second part");
            assert_eq!([words[6], words[7]], [cut, noted], "{asked:?}: cut, notes");
            if noted > 0 {
                assert_eq!(words[8], PROGRAM, "{asked:?}: the program noted");
            }
            assert_eq!(hits, ran, "{asked:?}: groups run");
        }
    }

    /// The parts through a host list (`bound_split`, the cached
    /// `split_signature`, `Host::execute`), as `crate::draw` runs an
    /// indirect dispatch: every one of 98,000 groups runs once. With the
    /// debug layer on (`VICTORY_D3D12_DEBUG=1` in a diagnostics build) it
    /// finds nothing to object to (the count is read from the arguments'
    /// own buffer).
    #[test]
    fn host_list_dispatch_in_parts_runs_every_group_once() {
        let Some(host) = super::host() else {
            return;
        };
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let counter = compile_cs(&host.device, PARTS_CS).expect("the counting shader must compile");
        let Some(signature) = split_signature(&counter.root, (0, 2)) else {
            panic!("no split command signature on the host");
        };
        const GROUPS: u32 = 98_000;
        let (Some((src, src_ptr)), Some((hits, hits_ptr))) = (host.shared_buffer(16), host.shared_buffer(4 * (u64::from(GROUPS) + 1))) else {
            return;
        };
        // SAFETY: mappings of these sizes; nothing runs.
        unsafe {
            std::ptr::copy_nonoverlapping([GROUPS, 1, 1].as_ptr().cast(), src_ptr, 12);
            std::ptr::write_bytes(hits_ptr, 0, 4 * (GROUPS as usize + 1));
        }
        let scratch = bound_split(&list, (&src, 0), None, 12).expect("bounded in parts");
        // SAFETY: an open list; the counter's own root signature and
        // pipeline; buffers the test holds until the list has run.
        unsafe {
            list.list.SetComputeRootSignature(&counter.root);
            list.list.SetPipelineState(&counter.pso);
            let constants = [GROUPS, 1, 0];
            list.list.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
            list.list.SetComputeRootUnorderedAccessView(1, hits.GetGPUVirtualAddress());
            list.list.ExecuteIndirect(&signature, MOST_PARTS, &scratch, 0, Some(&scratch), PARTS_COUNT_AT);
            list.list.SetComputeRoot32BitConstant(0, 0, 2);
        }
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
        let (wrong, past) = not_once(hits_ptr, GROUPS as usize);
        assert!(wrong.is_empty(), "slots that did not count one group: {wrong:?}");
        assert!(past, "a group past the dispatch ran");
    }
}

/// The benchmark of the clamp and its barriers (ignored by default).
#[cfg(test)]
mod bench;

/// What one small draw costs this card, bare and with one thing changed
/// between every two draws: a measurement, not a check (it asserts only
/// that the card answered). The timing run of the lit scene (run 187) has
/// some 11,000 small commands a frame at 5 to 16 microseconds each, and no
/// log says whether that is what we put around a command or what the card
/// takes for one. Each variant here is 10,000 one-triangle draws into a
/// 2560x1440 target, timed on the card by one pair of timestamps around
/// the whole batch (not one per command, which would time the stamps).
///
/// Run by name, no game needed:
/// `cargo test -p victory-gpu --release -- --ignored bare_command_cost --nocapture`
#[cfg(test)]
mod bare_cost {
    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use crate::host::{compile, transition};

    /// Draws in a batch.
    const DRAWS: u32 = 10_000;
    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 1440;

    /// One small triangle in the middle of the target (about 180 pixels),
    /// and two pixel programs that read one texel of a texture through a
    /// table, so that the view heap set is in use.
    const DRAW: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> image : register(t0);
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(RS)]
float4 first(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}
[RootSignature(RS)]
float4 second(float4 at : SV_Position) : SV_Target {
    return 1.0 - image.Load(int3(0, 0, 0));
}
"#;

    /// A compute pass of next to nothing, as our clamp of an indirect
    /// record is: one group of one thread that writes one word.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void work() {
    words.Store(0, words.Load(0) + 1);
}
"#;

    /// What changes between every two draws of a batch.
    #[derive(Clone, Copy, PartialEq)]
    enum Between {
        /// Nothing.
        Nothing,
        /// The pipeline: two alternate.
        Pipeline,
        /// A second texture goes from target to shader read and back.
        BarrierPair,
        /// The depth texture bound for testing goes from write to read or
        /// back.
        DepthMove,
        /// The pair of descriptor heaps: two pairs alternate.
        Heaps,
        /// The colour target: two alternate.
        Targets,
        /// A compute pass of ours of one group, and the pipeline back.
        Compute,
        /// The same between two waits for every earlier shader write, as
        /// the clamp records them.
        ComputeWithWaits,
    }

    struct Rig {
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pipelines: [ID3D12PipelineState; 2],
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        words: ID3D12Resource,
        other: ID3D12Resource,
        depth: ID3D12Resource,
        /// Kept alive: the views below name them.
        _kept: Vec<ID3D12Resource>,
        _target_heaps: [ID3D12DescriptorHeap; 2],
        heap_pairs: [[Option<ID3D12DescriptorHeap>; 2]; 2],
        tables: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
        targets: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        depth_written: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_tested: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
    }

    fn texture(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn heap(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32, visible: bool) -> ID3D12DescriptorHeap {
        let flags = if visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        // SAFETY: plain heap creation.
        unsafe {
            device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        }
    }

    fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: code(vs),
                PS: code(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Depth is tested and never written, so the texture may be
                // in the write state or in the read state under one view.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            device.CreateGraphicsPipelineState(&desc).unwrap()
        }
    }

    impl Rig {
        /// The default adapter, as the layer takes it; `None` without one
        /// or on a software adapter.
        fn new() -> Option<Self> {
            let vs = compile(DRAW, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = [
                compile(DRAW, s!("first"), s!("ps_5_1")).ok()?,
                compile(DRAW, s!("second"), s!("ps_5_1")).ok()?,
            ];
            let cs = compile(WORK, s!("work"), s!("cs_5_1")).ok()?;
            // SAFETY: plain object creation on a device of our own; the
            // views are made in heaps that live as long as the rig.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
                    .ok()?
                    .cast::<IDXGIAdapter1>()
                    .ok()?
                    .GetDesc1()
                    .ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    return None;
                }
                let card = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_owned();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..]).ok()?;
                let pipelines = [
                    pipeline(&device, &root, &vs, &ps[0]),
                    pipeline(&device, &root, &vs, &ps[1]),
                ];
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..]).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.as_ptr().cast(),
                            BytecodeLength: cs.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                let size = (WIDTH, HEIGHT);
                let colour = |state| texture(&device, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
                let target = [
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                ];
                let other = colour(D3D12_RESOURCE_STATE_RENDER_TARGET);
                let depth = texture(
                    &device,
                    size,
                    DXGI_FORMAT_D32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let image = texture(
                    &device,
                    (4, 4),
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                let words = buffer(
                    &device,
                    64 << 10,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let back = buffer(
                    &device,
                    16,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;

                // Target and depth views.
                let rtvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, false);
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut targets = [rtvs.GetCPUDescriptorHandleForHeapStart(); 2];
                targets[1].ptr += rtv_step;
                device.CreateRenderTargetView(&target[0], None, targets[0]);
                device.CreateRenderTargetView(&target[1], None, targets[1]);
                let dsvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, false);
                let dsv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                let depth_written = dsvs.GetCPUDescriptorHandleForHeapStart();
                let mut depth_tested = depth_written;
                depth_tested.ptr += dsv_step;
                device.CreateDepthStencilView(&depth, None, depth_written);
                let tested = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                    ..Default::default()
                };
                device.CreateDepthStencilView(&depth, Some(&tested), depth_tested);

                // Two pairs of shader-visible heaps: views (one view of the
                // small texture each) and samplers (none used).
                let view_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                ];
                let sampler_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                ];
                for views in &view_heaps {
                    device.CreateShaderResourceView(&image, None, views.GetCPUDescriptorHandleForHeapStart());
                }
                let tables = [
                    view_heaps[0].GetGPUDescriptorHandleForHeapStart(),
                    view_heaps[1].GetGPUDescriptorHandleForHeapStart(),
                ];
                let heap_pairs = [
                    [Some(view_heaps[0].clone()), Some(sampler_heaps[0].clone())],
                    [Some(view_heaps[1].clone()), Some(sampler_heaps[1].clone())],
                ];

                // The first list: both targets and the depth texture cleared.
                list.ClearRenderTargetView(targets[0], &[0.0; 4], None);
                list.ClearRenderTargetView(targets[1], &[0.0; 4], None);
                list.ClearDepthStencilView(depth_written, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                let [first_target, second_target] = target;
                let mut rig = Rig {
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pipelines,
                    work_root,
                    work,
                    words,
                    other,
                    depth,
                    _kept: vec![first_target, second_target, image],
                    _target_heaps: [rtvs, dsvs],
                    heap_pairs,
                    tables,
                    targets,
                    depth_written,
                    depth_tested,
                    stamps,
                    back,
                    card,
                };
                rig.finish();
                Some(rig)
            }
        }

        /// Closes the open list, runs it, waits for the card and opens the
        /// list again.
        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        /// What every list starts with: a new list has nothing set.
        fn begin(&self) {
            let l = &self.list;
            // SAFETY: an open list and live objects and views.
            unsafe {
                l.SetDescriptorHeaps(&self.heap_pairs[0]);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pipelines[0]);
                l.SetGraphicsRootDescriptorTable(0, self.tables[0]);
                l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_tested));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: WIDTH as f32,
                    Height: HEIGHT as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                }]);
                // A triangle list.
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            }
        }

        /// One batch of [`DRAWS`] draws in `lists` lists run one by one,
        /// with `between` recorded before every draw: the microseconds a
        /// draw took on the card, and the milliseconds recording took.
        fn batch(&mut self, between: Between, lists: u32) -> (f64, f64) {
            let (target, read, written, tested) = (
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_DEPTH_READ,
            );
            let started = std::time::Instant::now();
            let mut drawn = 0u32;
            for part in 0..lists {
                let l = &self.list;
                // SAFETY: an open list; our query heap; live objects.
                unsafe {
                    if part == 0 {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    }
                }
                self.begin();
                // SAFETY: an open list with everything set; live objects.
                // Every state moved is moved back within the batch (the
                // draws of a list are an even number).
                unsafe {
                    for _ in 0..DRAWS / lists {
                        let odd = (drawn % 2) as usize;
                        match between {
                            Between::Nothing => {}
                            Between::Pipeline => l.SetPipelineState(&self.pipelines[odd]),
                            Between::BarrierPair => {
                                l.ResourceBarrier(&[transition(&self.other, target, read)]);
                                l.ResourceBarrier(&[transition(&self.other, read, target)]);
                            }
                            Between::DepthMove => {
                                let (from, to) = if odd == 0 { (written, tested) } else { (tested, written) };
                                l.ResourceBarrier(&[transition(&self.depth, from, to)]);
                            }
                            Between::Heaps => {
                                l.SetDescriptorHeaps(&self.heap_pairs[odd]);
                                l.SetGraphicsRootDescriptorTable(0, self.tables[odd]);
                            }
                            Between::Targets => {
                                l.OMSetRenderTargets(1, Some(&self.targets[odd]), false, Some(&self.depth_tested));
                            }
                            Between::Compute | Between::ComputeWithWaits => {
                                let waits = between == Between::ComputeWithWaits;
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetComputeRootSignature(&self.work_root);
                                l.SetPipelineState(&self.work);
                                l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                                l.Dispatch(1, 1, 1);
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                        }
                        l.DrawInstanced(3, 1, 0, 0);
                        drawn += 1;
                    }
                    if part + 1 < lists {
                        // The next list of the batch: this one is sent on
                        // its own and the list is opened again at once
                        // (the allocator is not reset before the card is
                        // done with all of them).
                        l.Close().unwrap();
                        let run: ID3D12CommandList = l.cast().unwrap();
                        self.queue.ExecuteCommandLists(&[Some(run)]);
                        l.Reset(&self.allocator, None).unwrap();
                    } else {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
                    }
                }
            }
            let recording = started.elapsed().as_secs_f64() * 1e3;
            self.finish();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: a read-back buffer the finished list resolved two
            // timestamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                stamps[1].saturating_sub(stamps[0])
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            (ticks as f64 * 1e6 / frequency as f64 / f64::from(drawn.max(1)), recording)
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn bare_command_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        eprintln!("bare command cost on {}: {DRAWS} one-triangle draws into {WIDTH}x{HEIGHT}, three batches each", rig.card);
        // The first batch pays for whatever the driver does at first use.
        rig.batch(Between::Nothing, 1);
        let variants = [
            ("(a) nothing changes between draws", Between::Nothing, 1),
            ("(b) another pipeline every draw", Between::Pipeline, 1),
            ("(c) a barrier pair on a second texture every draw", Between::BarrierPair, 1),
            ("(d) the tested depth texture moved between write and read every draw", Between::DepthMove, 1),
            ("(e) another pair of descriptor heaps every draw", Between::Heaps, 1),
            ("(f) another colour target every draw", Between::Targets, 1),
            ("(g) nothing changes, in 100 lists run one by one", Between::Nothing, 100),
            ("(h) a compute pass of ours of one group before every draw", Between::Compute, 1),
            ("(i) the same pass between two waits for earlier shader writes", Between::ComputeWithWaits, 1),
        ];
        let mut least = Vec::new();
        for (name, between, lists) in variants {
            let mut runs: Vec<(f64, f64)> = (0..3).map(|_| rig.batch(between, lists)).collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            eprintln!(
                "{name}: {:.2} microseconds a draw on the card (the three: {:.2}, {:.2}, {:.2}); recording took {:.1} ms",
                runs[0].0, runs[0].0, runs[1].0, runs[2].0, runs[0].1
            );
            least.push(runs[0].0);
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}
