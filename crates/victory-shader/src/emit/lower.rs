//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, Bin, Cast, Rmw, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, flags};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Elsewhere, Export, Input, Interp, Io, Target};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, MEMORY, Resources, Role, Site, Source};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// The ray-tracing node test (`image_bvh_intersect_ray`), a child module so
/// it shares the emitter's state.
#[path = "intersect.rs"]
pub(super) mod intersect;
/// The image instructions, likewise.
mod image;
/// The vector ALU instructions, likewise.
mod valu;

type Why = &'static str;

/// How an interpolation evaluates an input channel.
#[derive(Debug, Clone, Copy)]
enum Eval {
    /// Loaded with this DXIL interpolation mode (1: constant, P0).
    Mode(u8),
    /// From the vertex values of a constant element: P0 when `p0`, plus
    /// i (P1 - P0), plus j (P2 - P0), weights as floats.
    Weights {
        p0: bool,
        i: Option<V>,
        j: Option<V>,
    },
}

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub root_uavs: Vec<u8>,
    /// Reads or writes title memory by address (the directory, the faults
    /// and the copies: `shader.rs` `SLOT_MEMORY`).
    pub memory: bool,
    /// Stores to title memory by address, or runs atomics on it.
    pub memory_writes: bool,
    /// Reads buffer views' title addresses (`shader.rs`
    /// `USER_VIEW_ADDRESS`).
    pub view_addresses: bool,
    /// Takes descriptors from the resource heap itself (root type 5, not
    /// a table), and whether as unordered access views.
    pub heap: bool,
    pub heap_writes: bool,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// The wave size the shader requires of the host, if any.
    pub wave_size: Option<u32>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Heap handles declared as constant buffers.
    #[cfg(test)]
    pub heap_constants: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Whether it declares rasterizer-ordered views.
    #[cfg(test)]
    pub rovs: bool,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
}

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    Fail(ShaderError),
}

pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    loop {
        match Lower::new(parts, &bad).run() {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

/// The f16 bits of inline float constant `c` (240..=248: 0.5, -0.5, 1.0,
/// -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2 pi)).
fn inline_f16(c: u8) -> u32 {
    [
        0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
    ][usize::from(c - 240)]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// Values (missing in `vals`) whose low 16 bits are known: a 16-bit
    /// result written over a value with no DXIL form keeps its high half.
    low16: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    /// Descriptor dwords with no host value (addresses, formats, all of a constant
    /// buffer's): no program reads them as numbers ([`Self::dead`]).
    stale: HashSet<Value>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    /// Phis read only as the value a guarded VGPR write keeps in the lanes
    /// outside its guard: such a read stops no translation (what those
    /// lanes keep is zero where it has no DXIL form), but what an edge can
    /// bring is still made (`fill_phis`).
    kept_phis: HashSet<Value>,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    /// The interpolations by site, as [`Self::settle_interps`] leaves them.
    interps: HashMap<Site, Interp>,
    handles: HashMap<(usize, Site, u8, bool, (u32, u32)), V>,
    /// Element strides read from the host's table by heap slot
    /// ([`super::shader::SLOT_STRIDES`]), by block and descriptor load.
    strides: HashMap<(usize, Site, u8), V>,
    /// The heap map's handle ([`Lower::heap_remap`]), made once in the
    /// entry block.
    heap_map: Option<V>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// [`Lowered::heap`], [`Lowered::heap_writes`].
    heap: [bool; 2],
    /// [`Lowered::memory_writes`].
    memory_written: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// Pixel: an ordered section (POPS), whose UAVs the host makes
    /// rasterizer-ordered views.
    pops: bool,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// IR blocks whose `s_barrier` not every lane is sure to reach (a lane
    /// branch lies between the entry and the block, or the program has a
    /// loop a lane can be cut out of): the barrier there keeps its fences
    /// and does not wait for the group ([`super::group_sync`]).
    fence_only: HashSet<usize>,
    /// A group barrier before every LDS access (a console wave run as two host waves).
    split_lds: bool,
    /// With `split_lds`, LDS accesses in lane branches: per IR block whether every lane runs
    /// it, and whether the host waves meet at its start and before its exit ([`lds_meetings`]).
    lds_meet: Option<(Vec<bool>, Vec<(bool, bool)>)>,
    /// A console wave64 program run as two host waves of 32 that exchange through LDS: the first
    /// of two exchange dwords, whether the group has no upper half (32 threads or fewer), and per
    /// IR block whether every lane runs it (exchanges need both halves there).
    exchange: Option<(u32, bool, Vec<bool>)>,
    /// With `exchange`: nothing is exchanged, each host wave acts as a wave
    /// of its own ([`with_local_waves`]).
    local: bool,
    /// The instruction being lowered may read a lane mask as a number under
    /// `local` (a bit count, a first set bit, `v_mbcnt`).
    local_ok: bool,
    /// With `local`: every mask read and lane read is taken per host wave
    /// ([`with_local_waves_any`]).
    local_any: bool,
    /// With `exchange`: VGPR values every lane holds alike ([`lane_uniform`]); a first-lane
    /// read of one needs nothing from the other host wave.
    uniform: HashSet<Value>,
    /// With `exchange`: the wave ORs that only steer a loop over each lane's own bits, which
    /// a host wave takes over its own lanes wherever they stand ([`super::own_bits`]).
    own_bits: super::own_bits::OwnBits,
    /// The console lane (0..63) of this thread, made in the entry block.
    clane: Option<V>,
    /// The first exchange dword of this thread's console wave ([`Self::exchange_pair`]).
    pair: Option<V>,
    /// A console wave64 program in host waves of 32: the LDS dword counting the console threads
    /// handed to host waves (`wave_base`), and the base made in the entry block.
    halves: Option<u32>,
    wave_base: Option<V>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// The 16/64-bit round mode (MODE bits 3:2) each block starts with, and
    /// the one in force at the instruction being lowered (see [`round16`]).
    round16: Vec<Option<u8>>,
    mode16: Option<u8>,
}

/// The 16/64-bit round mode (MODE bits 3:2: 0 nearest even, 3 toward zero)
/// after `inst`, from `mode` before it (`None`: not known).
fn round16_after(inst: &Inst, mode: Option<u8>) -> Option<u8> {
    match (inst.name(), &inst.fields) {
        ("s_round_mode", Fields::Sopp { imm }) => Some((imm >> 2 & 3) as u8),
        (
            "s_setreg_imm32_b32" | "s_setreg_b32",
            Fields::Sop {
                src, imm: Some(h), ..
            },
        ) => {
            let (id, off, size) = (h & 0x3F, h >> 6 & 0x1F, (h >> 11 & 0x1F) + 1);
            if id != 1 || off >= 4 || off + size <= 2 {
                return mode;
            }
            match src.first() {
                Some(&Operand::Literal(x)) if off <= 2 && off + size >= 4 => {
                    Some((x >> (2 - off) & 3) as u8)
                }
                _ => None,
            }
        }
        _ => mode,
    }
}

/// The 16/64-bit round mode each block starts with: nearest even at the
/// program's start, then as the MODE writes on the way set it; `None` where
/// paths that set it differently join (or a write is not a constant).
fn round16(p: &crate::ir::Program) -> Vec<Option<u8>> {
    // None: not reached yet; Some(m): the mode (m None: not known).
    let mut entry: Vec<Option<Option<u8>>> = vec![None; p.blocks.len()];
    entry[0] = Some(Some(0));
    let order = crate::ir::reverse_postorder(&p.blocks);
    loop {
        let mut changed = false;
        for &b in &order {
            let Some(mut m) = entry[b] else { continue };
            for inst in &p.blocks[b].insts {
                m = round16_after(inst, m);
            }
            for s in p.blocks[b].exit.successors() {
                let new = match entry[s] {
                    None => Some(m),
                    Some(old) if old == m => continue,
                    Some(_) => Some(None),
                };
                if entry[s] != new {
                    entry[s] = new;
                    changed = true;
                }
            }
        }
        if !changed {
            return entry.into_iter().map(Option::flatten).collect();
        }
    }
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 20;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared. Only a mask's bit
    /// count is taken, over the host's wave (the console compiler's test of
    /// a value for uniformity, `WaveActiveAllEqual`): lane numbers stay the
    /// console's, so the first lane and the count below a lane are refused.
    None,
}

/// What a value holding a lane mask read as a number stops with.
pub(super) const LANE_MASK: Why = "a lane mask read as data";
/// A mask with lanes outside EXEC where it is written, read as a number where the lanes that
/// run that block are not all of the wave.
pub(super) const MASK_ELSEWHERE: Why = "a lane mask holding lanes elsewhere read as data";

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
/// Per IR block: every lane of the wave that started the program runs it – no per-lane branch
/// separates it from the entry (it lies in no lane branch's region: the blocks reachable from
/// the branch that do not post-dominate it). None with an indirect exit (not followed).
fn all_lane_blocks(st: &Structured) -> Option<Vec<bool>> {
    let n = st.exits.len();
    let mut all = vec![true; n];
    for (_, region) in lane_regions(st)? {
        for b in (0..n).filter(|&b| region[b]) {
            all[b] = false;
        }
    }
    Some(all)
}

/// Every per-lane branch (by its block) with its region: the blocks reachable from it that do
/// not post-dominate it, which only some lanes run. Not the branches the wave takes together
/// ([`Structured::wide`]), nor those on scalar conditions. None with an indirect exit.
fn lane_regions(st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch { taken, not_taken, .. } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End => {}
            LaneExit::Indirect => return None,
        }
    }
    // post-dominators (bitsets; a virtual exit after every End)
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = (0..n).map(|b| if succ[b].is_empty() { let mut v = vec![0; words]; v[b / 64] |= 1 << (b % 64); v } else { full.clone() }).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for w in 0..words {
                    v[w] &= pdom[t][w];
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let mut regions = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        let LaneExit::Branch { when, .. } = *e else { continue };
        if st.wide.contains(&d) || st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_))) {
            continue; // a wave-wide or uniform branch: all lanes go the same way
        }
        // blocks reachable from the lane branch that do not post-dominate it
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            if seen[b] || pdom[d][b / 64] >> (b % 64) & 1 == 1 {
                continue;
            }
            seen[b] = true;
            stack.extend(succ[b].iter().copied());
        }
        regions.push((d, seen));
    }
    Some(regions)
}

/// The per-lane branches (by block) whose regions hold a block of `needs` (blocks where two
/// host waves must meet, [`take_needs`]): the ones to take wave-wide
/// ([`crate::structure::structure_wide`]) so that every lane runs those blocks. None if one
/// of them is a branch whose wave-wide meaning is not known ([`Structured::wave_any`]).
pub(super) fn branches_around(st: &Structured, needs: &[usize]) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for (d, region) in lane_regions(st)? {
        if needs.iter().any(|&b| region.get(b).copied().unwrap_or(false)) {
            st.wave_any.get(d).copied().flatten()?;
            out.push(d);
        }
    }
    Some(out)
}

/// Whether a refusal is of code that needs every lane of the wave where it stands (two host
/// waves meeting there), which the other forms of the program may give it.
pub(super) fn needs_every_lane(what: &str) -> bool {
    what.ends_with("where not every lane runs") || what == MASK_ELSEWHERE
}

/// The blocks where the last lowering on this thread wanted every lane and did not have it
/// (its "... where not every lane runs" refusals), which it forgets.
pub(super) fn take_needs() -> Vec<usize> {
    NEEDS.take()
}

fn need(block: usize) {
    NEEDS.with_borrow_mut(|n| {
        if !n.contains(&block) {
            n.push(block);
        }
    });
}

/// VGPR values that are the same in every lane holding them (`all` is [`all_lane_blocks`]):
/// written by an instruction every arriving lane runs (no guard), lane by lane, from scalars,
/// constants and such values; or loaded from an address made of them. The compiler keeps
/// wave-uniform data in VGPRs and fetches it with `v_readfirstlane`.
/// shortcut: a value joined by a phi counts only where every predecessor is a block all lanes
/// run (no lane branch can have split the arriving lanes); loop-carried values never do.
fn lane_uniform(p: &Program, st: &Structured, all: &[bool]) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut same: HashSet<Value> = HashSet::new();
    for b in crate::ir::reverse_postorder(&p.blocks) {
        let block = &p.blocks[b];
        let whole = block.preds.iter().all(|&q| all.get(q).copied().unwrap_or(false));
        for phi in &block.phis {
            if matches!(phi.reg, Reg::V(_)) && whole && phi.args.iter().all(|a| same.contains(a)) {
                same.insert(phi.value);
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            let ops: Vec<Operand> = match &inst.fields {
                Fields::Valu(v)
                    if v.dpp.is_none()
                        && v.sdwa.is_none()
                        && !name.starts_with("v_mbcnt")
                        && !name.starts_with("v_cmpx")
                        && !name.contains("lane") =>
                {
                    v.src.clone()
                }
                _ => continue,
            };
            if st.guards[b][i].is_some() {
                continue; // lanes outside EXEC keep another value
            }
            let alike = ops.iter().all(|op| match *op {
                Operand::Vector { reg, .. } => {
                    // every dword of the operand the instruction reads from that register on
                    let read: Vec<Value> =
                        inst.reads.iter().filter(|r| matches!(r.0, Reg::V(x) if x >= reg)).map(|r| r.1).collect();
                    !read.is_empty() && read.iter().all(|x| same.contains(x))
                }
                // a scalar is one value; a lane mask is not, unless a compare of such values made it
                Operand::Scalar { code, .. } if code != NULL => {
                    inst.read(Reg::S(code)).is_none_or(|x| !is_mask(x) || same.contains(&x))
                }
                _ => true,
            });
            if alike {
                same.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
            }
        }
    }
    same
}

/// What a register value holds, for [`wave_local`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wave {
    /// Nothing that depends on how the threads are divided into waves.
    Plain,
    /// The number of the wave's first lane in a mask (`s_ff1`).
    First,
    /// A reduction over the wave's lanes: a count, a sum, a minimum...
    Total(super::wave::Op),
    /// A count or sum over the lanes below this one.
    Prefix,
    /// Derived from what a first-lane atomic returned: the slots the wave reserved.
    Base,
    /// A mix the idiom does not make.
    Mixed,
}

/// Whether a console wave64 program may run with each host wave as a wave of its own
/// ([`with_local_waves`]): its wave-wide values are used only as the allocation idiom uses
/// them, which does not care how the threads are divided into waves. A first-lane number only
/// picks the first lane; a count or sum only goes into that lane's atomic on a shared counter;
/// a prefix only goes onto the base that atomic returned; the first lane does nothing else.
/// Anything else done with them takes one wave to be the whole thread group (a tile's result,
/// say) and is refused, with the offset and what was found.
/// Lane masks read as numbers by other instructions and lane reads outside whole-wave sections
/// are stopped by the lowering itself.
// shortcut: `v_readfirstlane` of a value that differs between lanes is taken as it is (this
// host wave's first lane), and a guarded write of a plain value over a wave-dependent one is
// the only mix of old and new lanes that is looked at.
pub(super) fn wave_local(p: &Program, st: &Structured) -> Result<(), (usize, &'static str)> {
    use super::wave::{Op, Shape};
    const FIRST: Why = "a wave's first-lane number used as data";
    const TOTAL: Why = "a count or sum over the wave used as data";
    const PREFIX: Why = "a count or sum over the lanes below used without a reserved base";
    const MIXED: Why = "a value that depends on the wave on some paths only";
    const EFFECT: Why = "a write by the wave's first lane other than its reservation";
    fn of(kind: &HashMap<Value, Wave>, v: Option<Value>) -> Wave {
        v.and_then(|v| kind.get(&v).copied()).unwrap_or(Wave::Plain)
    }
    fn put(kind: &mut HashMap<Value, Wave>, v: Value, k: Wave) {
        if k == Wave::Plain {
            kind.remove(&v);
        } else {
            kind.insert(v, k);
        }
    }
    let wild = |k: Wave| !matches!(k, Wave::Plain | Wave::Base);
    let why = |k: Wave| match k {
        Wave::First => FIRST,
        Wave::Total(_) => TOTAL,
        Wave::Prefix => PREFIX,
        _ => MIXED,
    };
    // a mask (by its low dword) under which only a wave's first lane runs
    let first_mask = |firsts: &HashSet<Value>, m: Option<Value>| {
        m.and_then(|m| st.masks.get(&m)).is_some_and(|&b| {
            st.bools
                .support(b)
                .iter()
                .any(|a| matches!(a, Atom::Lane(v) if firsts.contains(v)))
        })
    };
    // One instruction outside a section: the kind of what it writes, and whether it is a
    // first lane's reservation.
    let step = |inst: &Inst, kind: &HashMap<Value, Wave>, firsts: &mut HashSet<Value>| -> Result<(Wave, bool), Why> {
        let name = inst.name();
        let val = |op: &Operand| match *op {
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
            Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
            _ => None,
        };
        let ops: Vec<Operand> = match &inst.fields {
            Fields::Valu(v) => v.src.clone(),
            Fields::Sop { dst, src, .. } => {
                let mut o = src.clone();
                if matches!(name, "s_addk_i32" | "s_mulk_i32") {
                    o.extend(dst.iter().copied());
                }
                o
            }
            Fields::Buffer { data, .. } if name.contains("_atomic_") || name.contains("_store_") => {
                data.iter().copied().collect()
            }
            _ => Vec::new(),
        };
        let kinds: Vec<Wave> = ops.iter().map(|o| of(kind, val(o))).collect();
        // Everything else it reads (addresses, descriptors, SCC, the high dwords of wide
        // operands) must not depend on the wave. Not EXEC, and not the old value of a VGPR it
        // writes, which only the lanes outside EXEC keep.
        let named: Vec<Value> = ops.iter().filter_map(&val).collect();
        for &(r, v) in &inst.reads {
            let kept = matches!(r, Reg::V(_)) && inst.writes.iter().any(|w| w.0 == r);
            if matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1) || kept || named.contains(&v) {
                continue;
            }
            let k = of(kind, Some(v));
            if wild(k) {
                return Err(why(k));
            }
        }
        let w = kinds.iter().copied().find(|&k| wild(k)).unwrap_or(Wave::Mixed);
        let sum = |k: Wave| k == Wave::Total(Op::AddU);
        let is_mask = |o: &Operand| {
            val(o).is_some_and(|v| st.masks.contains_key(&v) || st.highs.contains_key(&v))
        };
        let mut reserved = false;
        let out = match (name, &kinds[..]) {
            ("s_ff1_i32_b64" | "s_ff1_i32_b32", _) if ops.first().is_some_and(is_mask) => Wave::First,
            ("s_bcnt1_i32_b64" | "s_bcnt1_i32_b32", _) if ops.first().is_some_and(is_mask) => {
                Wave::Total(Op::AddU)
            }
            // bits of a mask below the lane; with -1 for the mask, the lane's number
            ("v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32", [_, b]) => {
                match (matches!(ops[0], Operand::Scalar { .. }), *b) {
                    (_, Wave::Prefix) | (true, Wave::Plain | Wave::Base) => Wave::Prefix,
                    (false, Wave::Plain | Wave::Base) => Wave::Plain,
                    (_, k) => return Err(why(k)),
                }
            }
            // nothing wave-dependent goes in; what derives from a reserved base stays marked
            _ if !kinds.iter().any(|&k| wild(k)) => {
                let derives = matches!(inst.fields, Fields::Valu(_) | Fields::Sop { .. });
                if derives && kinds.contains(&Wave::Base) { Wave::Base } else { Wave::Plain }
            }
            ("v_mov_b32" | "s_mov_b32", [k]) => *k,
            ("v_cmp_eq_i32" | "v_cmp_eq_u32" | "v_cmpx_eq_i32" | "v_cmpx_eq_u32", [a, b]) => {
                // the lane whose own number is the first lane's
                let lane = match (*a, *b) {
                    (Wave::First, k) if !wild(k) => &ops[1],
                    (k, Wave::First) if !wild(k) => &ops[0],
                    _ => return Err(why(w)),
                };
                if !matches!(lane, Operand::Vector { .. }) {
                    return Err(FIRST);
                }
                let mask = match &inst.fields {
                    _ if name.starts_with("v_cmpx") => inst.write(Reg::S(EXEC)),
                    Fields::Valu(v) => match v.dst {
                        Some(Operand::Scalar { code, .. }) => inst.write(Reg::S(code)),
                        _ => None,
                    },
                    _ => None,
                };
                firsts.extend(mask);
                Wave::Plain
            }
            ("v_add_nc_u32" | "s_add_i32", [a, b]) => match (*a, *b) {
                (Wave::Prefix, Wave::Base) | (Wave::Base, Wave::Prefix) => Wave::Base,
                (Wave::Prefix, Wave::Plain | Wave::Prefix) | (Wave::Plain, Wave::Prefix) => Wave::Prefix,
                (x, y) if sum(x) && sum(y) => x,
                _ => return Err(why(w)),
            },
            ("v_sub_nc_u32" | "s_sub_i32", [a, b]) => match (*a, *b) {
                (Wave::Base, Wave::Prefix) => Wave::Base,
                (Wave::Prefix, Wave::Plain) => Wave::Prefix,
                (Wave::Plain, y) if sum(y) && matches!(ops[0], Operand::Const(128)) => y,
                _ => return Err(why(w)),
            },
            ("s_mul_i32" | "v_mul_lo_u32" | "v_mul_u32_u24", [a, b]) => match (*a, *b) {
                (x, Wave::Plain) | (Wave::Plain, x) if sum(x) || x == Wave::Prefix => x,
                _ => return Err(why(w)),
            },
            ("s_lshl_b32", [x, Wave::Plain]) | ("v_lshlrev_b32", [Wave::Plain, x])
                if sum(*x) || *x == Wave::Prefix =>
            {
                *x
            }
            ("v_lshl_add_u32", [Wave::Prefix, Wave::Plain, c]) if !wild(*c) => {
                if *c == Wave::Base { Wave::Base } else { Wave::Prefix }
            }
            // the reservation: the first lane adds its wave's total to a shared counter
            (_, [Wave::Total(op)]) if name.contains("_atomic_") => {
                let want = match op {
                    Op::AddU => "buffer_atomic_add",
                    Op::MinU => "buffer_atomic_umin",
                    Op::MaxU => "buffer_atomic_umax",
                    Op::MinI => "buffer_atomic_smin",
                    Op::MaxI => "buffer_atomic_smax",
                    Op::Or => "buffer_atomic_or",
                    Op::And => "buffer_atomic_and",
                    Op::Xor => "buffer_atomic_xor",
                    _ => "",
                };
                if name != want || !first_mask(firsts, inst.read(Reg::S(EXEC))) {
                    return Err(TOTAL);
                }
                reserved = true;
                Wave::Base
            }
            _ => return Err(why(w)),
        };
        Ok((out, reserved))
    };
    let sections: Vec<_> = (0..p.blocks.len()).map(|b| super::wave::sections(p, b)).collect();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let pass = |kind: &mut HashMap<Value, Wave>, firsts: &mut HashSet<Value>, check: bool| -> Result<(), (usize, Why)> {
        for &b in &order {
            let block = &p.blocks[b];
            for phi in &block.phis {
                let mut k = Wave::Plain;
                for (n, &a) in phi.args.iter().enumerate() {
                    let x = of(kind, Some(a));
                    k = match (k, x) {
                        _ if n == 0 => x,
                        (k, x) if k == x => k,
                        (Wave::Plain, m @ (Wave::Base | Wave::Prefix))
                        | (m @ (Wave::Base | Wave::Prefix), Wave::Plain) => m,
                        _ => Wave::Mixed,
                    };
                }
                put(kind, phi.value, k);
            }
            for (i, inst) in block.insts.iter().enumerate() {
                // inside a whole-wave section: what the simulation found each value to be
                if let Some((_, sec)) = sections[b].iter().find(|(s, sec)| i > **s && i < sec.end) {
                    for &(_, w) in &inst.writes {
                        let k = match sec.shapes.get(&w) {
                            None => Wave::Mixed,
                            Some(&Shape::K(_)) => Wave::Plain,
                            Some(&Shape::Value(v)) => of(kind, Some(v)),
                            Some(&Shape::Masked(src, _, _)) => of(kind, Some(src)),
                            Some(&Shape::Active(op, src, mask, _)) => match of(kind, Some(src)) {
                                Wave::Plain | Wave::Base => Wave::Total(op),
                                // the first lane's own sum of a total is that total
                                Wave::Total(o) if o == op && first_mask(firsts, Some(mask)) => Wave::Total(op),
                                _ => Wave::Mixed,
                            },
                            Some(&Shape::Prefix(Op::AddU, src, mask, _, _)) => match of(kind, Some(src)) {
                                Wave::Plain | Wave::Base => Wave::Prefix,
                                Wave::Total(Op::AddU) if first_mask(firsts, Some(mask)) => Wave::Prefix,
                                _ => Wave::Mixed,
                            },
                            Some(_) => Wave::Mixed,
                        };
                        put(kind, w, k);
                    }
                    continue;
                }
                let name = inst.name();
                if name.starts_with("s_cbranch") {
                    continue; // with the block's exit, below
                }
                let (k, reserved) = match step(inst, kind, firsts) {
                    Ok(r) => r,
                    Err(why) if check => return Err((inst.machine.offset, why)),
                    Err(_) => (Wave::Mixed, false),
                };
                // the same test as `Lower::inst`'s `effect`
                let effect = matches!(inst.fields, Fields::Exp { .. })
                    || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
                    || name.contains("store")
                    || name.contains("atomic")
                    || matches!(inst.fields, Fields::Ds { .. })
                        && !(name.starts_with("ds_read")
                            || name.starts_with("ds_swizzle")
                            || name.contains("permute"));
                if check && effect && !reserved && first_mask(firsts, inst.read(Reg::S(EXEC))) {
                    return Err((inst.machine.offset, EFFECT));
                }
                let guarded = st.guards[b][i].is_some();
                for &(r, w) in &inst.writes {
                    let mut kw = match r {
                        // "some lane is counted" keeps the count's kind; any other SCC is one bit of data
                        Reg::Scc if !name.starts_with("s_bcnt1") => {
                            if wild(k) { Wave::Mixed } else { Wave::Plain }
                        }
                        _ => k,
                    };
                    // a guarded write leaves the old value in the lanes outside EXEC
                    if guarded
                        && matches!(r, Reg::V(_))
                        && kw == Wave::Plain
                        && matches!(of(kind, inst.read(r)), Wave::First | Wave::Total(_) | Wave::Prefix)
                    {
                        kw = Wave::Mixed;
                    }
                    put(kind, w, kw);
                }
            }
            // a branch on "some lane is counted": fine where the structure made it the lane's
            // own bit (the mask is EXEC under a scalar condition)
            if check
                && let Some(last) = block.insts.last()
                && last.name().starts_with("s_cbranch_scc")
                && let Some(scc) = last.read(Reg::Scc)
                && wild(of(kind, Some(scc)))
            {
                let own = matches!(st.exits[b], LaneExit::Branch { when, .. }
                    if !st.bools.support(when).iter().any(|a| matches!(a, Atom::Uniform(v) if *v == scc)));
                if !own {
                    return Err((last.machine.offset, TOTAL));
                }
            }
        }
        Ok(())
    };
    let mut kind: HashMap<Value, Wave> = HashMap::new();
    let mut firsts: HashSet<Value> = HashSet::new();
    // to a fixpoint (values carried around loops), then once more to say what is wrong
    let mut settled = false;
    for _ in 0..16 {
        let before = (kind.clone(), firsts.len());
        pass(&mut kind, &mut firsts, false)?;
        if (&before.0, before.1) == (&kind, firsts.len()) {
            settled = true;
            break;
        }
    }
    if !settled {
        return Err((0, MIXED));
    }
    pass(&mut kind, &mut firsts, true)
}

/// The mode of a group barrier the translation adds (host waves meeting):
/// a wait for the group with a shared-memory fence; the fence alone with
/// `VICTORY_GROUP_SYNC=0` ([`super::group_sync`]).
fn meeting_mode() -> u32 {
    if super::group_sync() == 0 { 8 } else { 1 | 8 }
}

/// LDS accesses of a console wave run as two host waves, where some sit in blocks that only
/// some lanes run (`all` is [`all_lane_blocks`]): per IR block, whether the host waves meet (a
/// group barrier) at its start and before its exit. `None` if barriers cannot keep the
/// console's order.
///
/// The console runs the wave's lanes in step, so an LDS write is seen by every later read. Two
/// host waves keep that order if they meet between any two accesses of which one writes. A
/// barrier must be run by every thread, so it cannot stand in a lane branch's region; but
/// every way into and out of such a region passes a block all lanes run, and the waves meet
/// there: before the exit of an all-lane block into a region with LDS accesses, and at the
/// start of an all-lane block entered from one. Accesses in all-lane blocks get their own
/// barrier as before. What that cannot order is a write and a read inside one region (blocks
/// joined without an all-lane block between them): refused.
/// shortcut: two writes in one region are taken not to hit the same dword from both halves.
fn lds_meetings(p: &crate::ir::Program, st: &Structured, all: &[bool]) -> Option<Vec<(bool, bool)>> {
    let n = st.exits.len();
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    // regions: blocks not every lane runs, joined by the edges between them
    let mut region: Vec<usize> = (0..n).collect();
    fn find(region: &mut [usize], mut b: usize) -> usize {
        while region[b] != b {
            region[b] = region[region[b]];
            b = region[b];
        }
        b
    }
    for b in 0..n {
        for t in succ(b) {
            if !all[b] && !all[t] {
                let (x, y) = (find(&mut region, b), find(&mut region, t));
                region[x] = y;
            }
        }
    }
    // per region: LDS reads, LDS writes
    let mut reads = vec![false; n];
    let mut writes = vec![false; n];
    for (b, block) in p.blocks.iter().enumerate().take(n) {
        if all[b] {
            continue;
        }
        let r = find(&mut region, b);
        for i in &block.insts {
            let name = i.name();
            if !matches!(i.fields, Fields::Ds { .. }) || name.starts_with("ds_swizzle") || name.contains("permute") {
                continue;
            }
            if name.starts_with("ds_read") {
                reads[r] = true;
            } else {
                writes[r] = true;
            }
        }
    }
    if (0..n).any(|r| reads[r] && writes[r]) {
        return None;
    }
    let mut meet = vec![(false, false); n];
    for b in 0..n {
        for t in succ(b) {
            if all[b] && !all[t] {
                let r = find(&mut region, t);
                meet[b].1 |= reads[r] || writes[r];
            } else if !all[b] && all[t] {
                let r = find(&mut region, b);
                meet[t].0 |= reads[r] || writes[r];
            }
        }
    }
    Some(meet)
}

/// Masks held as numbers or constant lane sets in guards or lane branches: the lane tests its
/// own bit by its console lane.
fn lane_constant_masks(st: &Structured) -> bool {
    let whens = st.exits.iter().filter_map(|e| match *e {
        LaneExit::Branch { when, .. } => Some(when),
        _ => None,
    });
    let mut work: Vec<Bool> = st.guards.iter().flatten().flatten().copied().chain(whens).collect();
    // Through lane-mask joins too: an edge that brings a number or a constant lane set needs
    // the lane's own bit of it.
    // shortcut: masks only selects read (`v_cndmask_b32`, carries in) are not followed; such
    // a program still stops with "a constant lane mask" or "an opaque lane mask".
    let (mut seen, mut joins) = (HashSet::new(), HashSet::new());
    while let Some(g) = work.pop() {
        if !seen.insert(g) {
            continue;
        }
        for a in st.bools.support(g) {
            match a {
                Atom::Opaque(_) | Atom::Constant(_) => return true,
                Atom::Phi(id) if joins.insert(id) => {
                    if let Some(edges) = st.phi_masks.get(&id) {
                        work.extend(edges.iter().copied());
                    }
                }
                _ => {}
            }
        }
    }
    false
}

/// The widest wave the host GPU runs (0: not known – no host-specific forms); see
/// [`set_host_max_wave`].
static HOST_MAX_WAVE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The host GPU's `WaveLaneCountMax`: 32 lets console wave64 compute programs whose cross-lane
/// code stays inside 32-lane halves run in waves of 32 instead of being refused.
pub fn set_host_max_wave(lanes: u32) {
    HOST_MAX_WAVE.store(lanes, std::sync::atomic::Ordering::Relaxed);
}

thread_local! {
    /// See [`take_needs`].
    static NEEDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    /// See [`with_local_waves`].
    static LOCAL_WAVES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// See [`with_local_waves_any`].
    static LOCAL_ANY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// As [`with_local_waves`], with every other use of a lane mask as a number
/// and every lane read and write taken per host wave too: the program runs
/// as it would on a console of 32-lane waves, lanes numbered 0..31 or
/// 32..63 by half. Right for code that only relies on what HLSL's wave
/// operations promise; a program that takes one wave to be its whole
/// thread group gets per-half results. The last resort for programs that
/// do not translate otherwise.
pub(super) fn with_local_waves_any<R>(f: impl FnOnce() -> R) -> R {
    let was = LOCAL_ANY.replace(true);
    let r = with_local_waves(f);
    LOCAL_ANY.set(was);
    r
}

/// Runs `f` with console wave64 programs on 32-lane hosts lowered with each
/// host wave as a wave of its own ("local"): no exchange between the two
/// halves of a console wave, so no barriers and no need for every lane to
/// reach the code. What HLSL's wave operations promise holds for any
/// division of the threads into waves, and the idiom the title's compute
/// programs use them for does not care which wave a thread is in: count or
/// sum over the wave, the first lane reserves that many slots with an atomic
/// add, every lane takes the base (`WaveReadLaneFirst`) plus its prefix.
/// Each lane still gets a slot of its own; only the order differs.
/// Only that idiom's operations are lowered this way: a lane mask read as
/// a number by anything but a bit count, a first-set-bit or `v_mbcnt`, and
/// lane reads and writes outside whole-wave sections, stop the translation.
/// shortcut: where a count or sum goes is not checked. A program that uses
/// a wave-wide result as data for the whole console wave (a tile's maximum,
/// say, with one wave per thread group) gets its half's result. It is used
/// only for programs that do not translate otherwise.
pub(super) fn with_local_waves<R>(f: impl FnOnce() -> R) -> R {
    let was = LOCAL_WAVES.replace(true);
    let r = f();
    LOCAL_WAVES.set(was);
    r
}

/// Whether the results of 16-bit float operations (`v_add_f16`, `v_mul_f16`,
/// `v_cvt_f16_f32`, 16-bit interpolation ...) are rounded on their bits as the console
/// rounds them ([`Lower::narrow16`]: to nearest even, or toward zero where the program's
/// MODE says so) instead of as the host's `f32tof16` does (toward zero on NVIDIA, the
/// RX 6700 XT and WARP: the last place of a result that is not exact is then one lower
/// than the console's). On unless `VICTORY_F16_NEAREST=0`, which keeps the host's
/// conversion (it saves the instructions the rounding adds to every such operation).
/// The wave sum of 16-bit floats ([`Lower::half_sum`]) is always exact.
fn f16_nearest_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_F16_NEAREST.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_F16_NEAREST").is_none_or(|v| v != "0"))
}

/// `VICTORY_STRICT_JOINS`: a join that only another join reads counts as
/// read (`Lower::fill_phis`).
fn strict_joins() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_STRICT_JOINS.get() {
        return on;
    }
    static STRICT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STRICT.get_or_init(|| std::env::var_os("VICTORY_STRICT_JOINS").is_some())
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`strict_joins`], for its thread only.
    pub(super) static TEST_STRICT_JOINS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`f16_nearest_on`], for its thread only.
    pub(super) static TEST_F16_NEAREST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// `VICTORY_WAVE_OWN_BITS=0`: a wave OR where not every lane runs is refused as before
/// ([`super::own_bits`]).
fn own_bits_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAVE_OWN_BITS").is_none_or(|v| v != "0"))
}

/// The host wave size translations are made for now: what
/// [`set_host_max_wave`] was last given (0: not known). Part of what a
/// translation depends on, so a cache of translations keys on it.
pub fn host_max_wave() -> u32 {
    #[cfg(test)]
    if let Some(lanes) = TEST_HOST_WAVE.get() {
        return lanes;
    }
    HOST_MAX_WAVE.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    /// A test's host wave size, for its thread only ([`set_host_max_wave`] is the process's).
    pub(super) static TEST_HOST_WAVE: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

pub(super) fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// The parameters packing 16-bit rows two per dword and where each lays
/// its elements out: (first row, [(dword, register, column, columns)]).
///
/// The console gives every 16-bit element whole dwords. A parameter leads
/// with the first element of its first row, then the first element of its
/// second row. An element that shares a register behind another one (the
/// signature packs a short element into the free columns of an earlier
/// register) is NOT in that row's halves: it follows in the first
/// parameter, in row order, that has whole dwords left. Grand Theft Auto
/// VI's packed pedestrian shaders show both cases: TEXCOORD7 (register 1,
/// columns 2-3, behind the two columns of TEXCOORD0) is exported in dword 3
/// of parameter 0, after TEXCOORD1's two dwords; TEXCOORD6 (register 4,
/// column 3, behind the three columns of TEXCOORD4) in dword 3 of
/// parameter 3, after TEXCOORD5 and TEXCOORD7, while the high half of
/// TEXCOORD4's second dword stays zero.
type PackedLayout = Vec<(u8, Vec<(u8, u8, u8, u8)>)>;

fn packed_layout(elements: &[SigElement], packs: &[(u8, Option<u8>)]) -> PackedLayout {
    let of_row = |r: u8| {
        let mut of: Vec<&SigElement> = elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .collect();
        of.sort_by_key(|e| e.col);
        of
    };
    let mut packs = packs.to_vec();
    packs.sort_unstable();
    packs.dedup_by_key(|p| p.0);
    // Per parameter: its first row, the dwords used, the elements.
    let mut layout: Vec<(u8, u8, Vec<(u8, u8, u8, u8)>)> = Vec::new();
    let mut behind = Vec::new();
    for &(a, b) in &packs {
        let mut placed = Vec::new();
        let mut at = 0u8;
        if !of_row(a).is_empty() {
            for r in [Some(a), b].into_iter().flatten() {
                let of = of_row(r);
                let Some((lead, more)) = of.split_first() else {
                    continue;
                };
                placed.push((at, r, lead.col, lead.cols));
                at += lead.cols.div_ceil(2);
                behind.extend(more.iter().map(|e| (r, e.col, e.cols)));
            }
        }
        layout.push((a, at, placed));
    }
    behind.sort_unstable();
    for (r, col, cols) in behind {
        let need = cols.div_ceil(2);
        if let Some(p) = layout.iter_mut().find(|p| p.1 + need <= 4) {
            p.2.push((p.1, r, col, cols));
            p.1 += need;
        }
    }
    layout.into_iter().map(|(a, _, of)| (a, of)).collect()
}

/// Where half `h` (dword h / 2, the high half when odd) of the parameter
/// whose first 16-bit row is `a` comes from: a register and column of
/// `elements`, none for padding ([`packed_layout`]; `packs` lists every
/// parameter packing two 16-bit rows per dword).
fn packed_half(
    elements: &[SigElement],
    packs: &[(u8, Option<u8>)],
    a: u8,
    h: u8,
) -> Option<(u8, u8)> {
    let layout = packed_layout(elements, packs);
    let (_, of) = layout.iter().find(|p| p.0 == a)?;
    of.iter().find_map(|&(dword, row, col, cols)| {
        let k = h.checked_sub(2 * dword)?;
        (k < cols).then_some((row, col + k))
    })
}

/// The pixel inputs packing two 16-bit rows per dword, as (first row,
/// second row) ([`packed_layout`]).
fn input_packs(io: &StageIo, inputs: &[SigElement]) -> Vec<(u8, Option<u8>)> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return Vec::new();
    };
    let partners = half_input_partners(io, inputs);
    entries
        .iter()
        .filter(|i| i.fp16 && i.pack2)
        .map(|i| {
            let b = if i.row2 != i.row {
                Some(i.row2)
            } else {
                partners.get(&i.row).copied()
            };
            (i.row, b)
        })
        .collect()
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]):
/// a packed row's partner is the next 16-bit row of the signature that no
/// output entry names, taken in row order.
fn half_outputs(io: &StageIo, outputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex = || entries.iter().filter(|o| o.param & 0x80 == 0);
    let mut free: Vec<u8> = outputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !per_vertex().any(|o| o.row == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut halves: Vec<_> = per_vertex().filter(|o| o.fp16).collect();
    halves.sort_by_key(|o| o.row);
    halves
        .into_iter()
        .map(|o| {
            let b = o
                .pack2
                .then(|| free.iter().position(|&r| r > o.row))
                .flatten();
            (o.row, (o.pack2, b.map(|i| free.remove(i))))
        })
        .collect()
}

/// Pixel inputs packing two 16-bit rows per dword whose second row the
/// metadata leaves out (`row2` = `row`): as for outputs
/// ([`half_outputs`]), the next 16-bit row of the signature that no input
/// entry names, taken in row order (rows the metadata does give are taken
/// first).
fn half_input_partners(io: &StageIo, inputs: &[SigElement]) -> HashMap<u8, u8> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return HashMap::new();
    };
    let packed = || entries.iter().filter(|i| i.fp16 && i.pack2);
    let mut free: Vec<u8> = inputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !entries.iter().any(|i| i.row == r))
        .filter(|&r| !packed().any(|i| i.row2 != i.row && i.row2 == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut missing: Vec<_> = packed().filter(|i| i.row2 == i.row).collect();
    missing.sort_by_key(|i| i.row);
    let mut partners = HashMap::new();
    for i in missing {
        if let Some(k) = free.iter().position(|&r| r > i.row) {
            partners.insert(i.row, free.remove(k));
        }
    }
    partners
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

impl<'a> Lower<'a> {
    fn new(x: &'a Parts<'a>, bad: &'a HashMap<Value, Why>) -> Self {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let mut console_lanes = false;
        // LDS accesses get a group barrier each (a console wave split in two host waves)
        let mut split_lds = false;
        // with `split_lds`, LDS accesses in lane branches: the blocks every lane runs, and
        // where the host waves meet around those branches (`lds_meetings`)
        let mut lds_meet: Option<(Vec<bool>, Vec<(bool, bool)>)> = None;
        // a console wave64 program in two host waves of 32 (`exchange`)
        let mut exchange: Option<(u32, bool, Vec<bool>)> = None;
        // a console wave64 program in host waves of 32: the LDS dword of `wave_base`'s counter
        let mut halves: Option<u32> = None;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            console_lanes = insts().any(|i| {
                let n = i.name();
                n.starts_with("ds_swizzle")
                    || n.contains("permute")
                    || n.starts_with("v_mbcnt")
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
                    || reads_mask_as_data(x.st, i)
            }) || lane_constant_masks(x.st)
                // (a branch the wave takes together asks the wave)
                || !x.st.wide.is_empty();
            let mut wave = if x.native.wave64 { 64 } else { 32 };
            let group = cs.threads.iter().product::<u32>();
            let one_wave = group <= wave;
            // Only a group with peers can share LDS through implicit wave lockstep.
            let wave_shared_lds = group > 1 && one_wave && s.lds_dwords > 0;
            // A host whose waves are at most 32 lanes (NVIDIA) refuses `WaveSize(64)`. A console
            // wave64 program whose cross-lane code stays inside 32-lane halves – DPP rows,
            // `ds_swizzle`, `v_permlane16` / `v_permlanex16`, which the ISA confines there – runs
            // exactly in waves of 32: the thread relabelling makes each host wave one half of a
            // console wave (lanes 32k..32k+31). Not when it crosses halves (`v_mbcnt`,
            // `v_readlane`, `v_writelane`, `ds_(b)permute`, masks as numbers or lane-constant
            // masks), nor when LDS may be shared within the console wave without barriers (a
            // group over 32 threads with LDS): those still need 64-lane waves.
            if wave == 64 && host_max_wave() == 32 && (console_lanes || wave_shared_lds) {
                let crosses = insts().any(|i| {
                    let n = i.name();
                    n.starts_with("v_mbcnt")
                        || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                        || n.starts_with("ds_permute")
                        || n.starts_with("ds_bpermute")
                        || reads_mask_as_data(x.st, i)
                }) || lane_constant_masks(x.st)
                    || !x.st.wide.is_empty();
                if !crosses && (s.lds_dwords == 0 || group <= 32) {
                    wave = 32;
                } else {
                    // (Groups of several console waves too: each console wave is two host
                    // waves, which exchange through their own pair of dwords.)
                    // LDS the console shares within its one wave without barriers: split into
                    // two waves of 32, a group barrier before every LDS access keeps their order
                    // – where every thread of the group gets there (no lane branch around it,
                    // no guard), else a barrier would hang the group
                    let lds = |i: &Inst| {
                        matches!(i.fields, Fields::Ds { .. })
                            && !(i.name().starts_with("ds_swizzle") || i.name().contains("permute"))
                    };
                    if let Some(all) = all_lane_blocks(x.st) {
                        let ok = p.blocks.iter().enumerate().all(|(b, blk)| {
                            blk.insts.iter().all(|inst| {
                                !lds(inst) || all.get(b).copied().unwrap_or(false)
                            })
                        });
                        let mut lds_ok = !(s.lds_dwords > 0 && group > 32) || ok;
                        if !lds_ok && let Some(meet) = lds_meetings(p, x.st, &all) {
                            lds_meet = Some((all.clone(), meet));
                            lds_ok = true;
                        }
                        if !lds_ok {
                            // (the blocks whose LDS accesses not every lane runs: with the
                            // branches around them taken wave-wide they get their barriers)
                            for (b, blk) in p.blocks.iter().enumerate() {
                                if !all.get(b).copied().unwrap_or(false) && blk.insts.iter().any(lds) {
                                    need(b);
                                }
                            }
                        }
                        if lds_ok {
                            wave = 32;
                            split_lds = s.lds_dwords > 0 && group > 32;
                            if crosses {
                                // lanes 32..63 are the other host wave: masks, lane reads and lane
                                // numbers go through two exchange dwords past the program's LDS
                                exchange = Some((s.lds_dwords, group <= 32, all));
                                s.lds_dwords += 2 * group.div_ceil(64);
                            }
                        }
                    }
                }
                // `VICTORY_WAVE_WHY=1`: why a program keeps 64-lane waves (one line per reason)
                if wave == 64 && std::env::var_os("VICTORY_WAVE_WHY").is_some() {
                    let mut why: Vec<&str> = Vec::new();
                    for i in insts() {
                        let n = i.name();
                        let r = if n.starts_with("v_mbcnt") {
                            "v_mbcnt"
                        } else if n == "v_readlane_b32" {
                            "v_readlane"
                        } else if n == "v_writelane_b32" {
                            "v_writelane"
                        } else if n.starts_with("ds_permute") || n.starts_with("ds_bpermute") {
                            "ds_(b)permute"
                        } else if reads_mask_as_data(x.st, i) {
                            "mask as data"
                        } else {
                            continue;
                        };
                        if !why.contains(&r) {
                            why.push(r);
                        }
                    }
                    if lane_constant_masks(x.st) {
                        why.push("lane-constant mask");
                    }
                    if s.lds_dwords > 0 && group > 32 {
                        why.push("LDS in a group over 32");
                        match all_lane_blocks(x.st) {
                            None => why.push("an indirect exit"),
                            Some(all) if lds_meetings(p, x.st, &all).is_none() => {
                                why.push("an LDS read and write in one lane-branch region")
                            }
                            Some(_) => {}
                        }
                    }
                    why.sort_unstable();
                    eprintln!("WAVE64-WHY group={group} lds={} [{}]", s.lds_dwords > 0, why.join(", "));
                }
            }
            if wave == 32 && x.native.wave64 {
                // console threads go to host waves in the order the waves take them (`wave_base`)
                halves = Some(s.lds_dwords);
                s.lds_dwords += 1;
            }
            if console_lanes || wave_shared_lds {
                s.wave_size = Some(wave);
            }
        }
        // The title's barriers a thread of the group may never reach, which
        // leaves the others waiting on the host: those in blocks some lanes
        // do not run, and all of them in a program with a loop closed by a
        // plain jump (a lane still in it at the iteration limit leaves the
        // program there, `abort_block`).
        let fence_only: HashSet<usize> = if insts().any(|i| i.name() == "s_barrier") {
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut place = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                place[b] = n;
            }
            let back = |from: usize, to: usize| {
                let at = |b: usize| place.get(b).copied().unwrap_or(usize::MAX);
                at(from) != usize::MAX && at(to) <= at(from)
            };
            let cut_short = x
                .st
                .exits
                .iter()
                .enumerate()
                .any(|(b, e)| matches!(*e, LaneExit::Jump(t) if back(b, t)));
            let all = all_lane_blocks(x.st);
            (0..p.blocks.len())
                .filter(|&b| p.blocks[b].insts.iter().any(|i| i.name() == "s_barrier"))
                .filter(|&b| {
                    cut_short
                        || all
                            .as_ref()
                            .is_none_or(|all| !all.get(b).copied().unwrap_or(false))
                })
                .collect()
        } else {
            HashSet::new()
        };
        let mut no_barrier = HashSet::new();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        Lower {
            s,
            x,
            bad,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            low16: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: HashMap::new(),
            lane_bits: HashMap::new(),
            phi_at,
            mask_phis: HashMap::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            kept_phis: HashSet::new(),
            stale: HashSet::new(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().cloned().collect(),
            handles: HashMap::new(),
            strides: HashMap::new(),
            heap_map: None,
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            heap: [false; 2],
            memory_written: false,
            src0: None,
            console_lanes,
            split_lds,
            lds_meet,
            local: exchange.is_some() && LOCAL_WAVES.get(),
            local_ok: false,
            local_any: LOCAL_ANY.get(),
            uniform: match &exchange {
                Some((_, false, all)) if !LOCAL_WAVES.get() => lane_uniform(p, x.st, all),
                _ => HashSet::new(),
            },
            own_bits: match &exchange {
                Some((_, false, _)) if !LOCAL_WAVES.get() && own_bits_on() => {
                    super::own_bits::find(p, x.st, std::env::var_os("VICTORY_WAVE_WHY").is_some())
                }
                _ => Default::default(),
            },
            exchange,
            clane: None,
            pair: None,
            halves,
            wave_base: None,
            pops: matches!(x.native.io, StageIo::Ps { db_shader_control, .. }
                if db_shader_control & 1 << 16 != 0),
            flat: None,
            no_barrier,
            fence_only,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: HashMap::new(),
            balloting: HashSet::new(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: HashSet::new(),
            code_table: None,
            counters: HashMap::new(),
            abort: None,
            round16: round16(p),
            mode16: Some(0),
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| a.offset & !3)
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.scalar.get(&(b, i)).is_some_and(|a| a.param == MEMORY) {
            return self.memory_flat(inst, b, i);
        }
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global memory write")?;
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            if let Some(bias) = self.root_bias(a.param) {
                off = self.s.m.bin(c, Bin::Add, off, bias);
            }
            Some(self.handle((b, i), &src, false, (11, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s
                        .m
                        .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                let own = self.own_bits.totals.contains(&w);
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh, own)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    ///
    /// `own`: a wave OR that only steers a loop over each lane's own bits
    /// ([`super::own_bits`]); where two host waves cannot exchange their halves' ORs it is
    /// this host wave's.
    fn shape(&mut self, shape: super::wave::Shape, own: bool) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
        };
        // Mesh: no host wave size can be declared, so a reduction is over this host wave
        // only. A minimum or maximum is taken that way: the compiler's "next distinct
        // value" loop (the wave's minimum, then the lanes that hold it) serves every
        // lane however the lanes are divided into waves. A sum, a bit reduction or a
        // prefix would be of part of a console wave: still refused.
        if self.mask_data == MaskData::None
            && (prefix.is_some()
                || !matches!(
                    op,
                    Op::MinU | Op::MaxU | Op::MinI | Op::MaxI | Op::MinF | Op::MaxF
                ))
        {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        if op == Op::AddH {
            return match prefix {
                Some(_) => Err("a wave prefix of an operation other than a sum"),
                None => self.half_sum(x),
            };
        }
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        // two host waves of 32 hold the console's 64 lanes: their results combine
        let r = if self.local || self.exchange.as_ref().is_some_and(|e| e.1) {
            // this host wave's own result (and prefix): each wave on its own, or a group of
            // 32 threads or fewer, which one host wave holds whole (combining with the 0 a
            // missing half "gives" broke AND, MIN and signed MAX)
            r
        } else if own && prefix.is_none() && op == Op::Or && !self.all_lanes(self.cur_ir) {
            // each thread walks the bits of this value and works only at its own, which the
            // host wave's OR holds: the same run as with the console wave's OR
            r
        } else if self.exchange.is_some() {
            if !self.all_lanes(self.cur_ir) {
                need(self.cur_ir);
                return Err("a wave reduction where not every lane runs");
            }
            let bits = |l: &mut Self, v: V| if float { l.as_int(v) } else { v };
            let back = |l: &mut Self, v: V| if float { l.as_float(v) } else { v };
            match prefix {
                Some(_) => {
                    // the upper half adds the lower half's total to its prefix
                    let (k0, ku) = (self.s.m.i8(0), self.s.m.i8(u8::from(!float)));
                    let total = self.s.op(c, &format!("waveActiveOp.{sfx}"), Attr::NoUnwind, ty, 119, &[arg, k0, ku]);
                    let tb = bits(self, total);
                    let c = self.cur;
                    let (lo, _) = self.exchange_halves(c, tb);
                    let lo = back(self, lo);
                    let cl = self.console_lane();
                    let k32 = self.s.m.i32(32);
                    let upper = self.s.m.cmp(c, pred::IUGE, cl, k32);
                    let zero = if float { self.s.m.f32(0) } else { self.s.m.i32(0) };
                    let add = self.s.m.select(c, upper, lo, zero);
                    self.s.m.bin(c, Bin::Add, r, add)
                }
                None => {
                    let rb = bits(self, r);
                    let c = self.cur;
                    let (lo, hi) = self.exchange_halves(c, rb);
                    let (lo, hi) = (back(self, lo), back(self, hi));
                    match op {
                        Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
                        Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
                        Op::And => self.s.m.bin(c, Bin::And, lo, hi),
                        Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
                        _ => {
                            // FMax 35, FMin 36, IMax 37, IMin 38, UMax 39, UMin 40
                            let code = match op {
                                Op::MaxF => 35,
                                Op::MinF => 36,
                                Op::MinU => 40,
                                Op::MaxU => 39,
                                Op::MinI => 38,
                                _ => 37,
                            };
                            self.s.op(c, &format!("binary.{sfx}"), Attr::ReadNone, ty, code, &[lo, hi])
                        }
                    }
                }
            }
        } else {
            r
        };
        Ok(if float { self.as_int(r) } else { r })
    }

    /// The sum of two 16-bit floats (the low halves of `a` and `b`) as the console's
    /// `v_add_f16` gives it: added in f32 and rounded to 16 bits, to nearest
    /// ([`Self::f16_nearest`]).
    fn add_halves(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let f32t = self.s.t.f32;
        let low = self.s.m.i32(0xFFFF);
        let wide = |l: &mut Self, x: V| {
            let h = l.s.m.bin(c, Bin::And, x, low);
            l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h])
        };
        let (fa, fb) = (wide(self, a), wide(self, b));
        let f = self.s.m.bin(c, Bin::Add, fa, fb);
        self.f16_nearest(f)
    }

    /// `WaveActiveSum` of 16-bit floats (`x`: each lane's term in its low half, +0 in the
    /// lanes outside the mask), bit for bit as the console adds it: every addition rounds to
    /// 16 bits, so the order counts, and the compiler's is the balanced tree over the lanes
    /// in their order (lanes 2k and 2k+1, then pairs, ... then the wave's two halves;
    /// [`super::wave::buddies`]). Each lane adds its partner's partial sum, read from the
    /// lane whose number differs in one bit, step by step; two host waves of 32 then exchange
    /// their halves' sums for the last addition. The high half of the result is lane 0's,
    /// which no `v_add_f16` writes.
    /// That needs every lane of a full wave there to be read: refused where not every lane
    /// runs, in a thread group that does not fill its waves, and for each half-wave on its
    /// own (half a sum is no sum).
    fn half_sum(&mut self, x: V) -> Result<V, Why> {
        const WHERE: Why = "a wave sum of 16-bit floats where not every lane runs";
        let threads = self.s.threads.ok_or("a wave sum of 16-bit floats outside compute")?;
        if threads.iter().product::<u32>() % 64 != 0 {
            return Err("a wave sum of 16-bit floats in a wave that is not full");
        }
        if self.local {
            return Err("a wave sum of 16-bit floats in a half-wave");
        }
        // two host waves of 32 that exchange, or one host wave of 64 lanes
        let halves = match (&self.exchange, self.s.wave_size) {
            (Some((_, false, _)), _) => true,
            (None, Some(64)) if self.x.p.wave64 => false,
            _ => return Err("a wave sum of 16-bit floats on these waves"),
        };
        let every = if halves {
            self.all_lanes(self.cur_ir)
        } else {
            all_lane_blocks(self.x.st).is_some_and(|all| all.get(self.cur_ir).copied().unwrap_or(false))
        };
        if !every {
            if halves {
                need(self.cur_ir);
            }
            return Err(WHERE);
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let mut sum = x;
        for step in 0..if halves { 5 } else { 6 } {
            let bit = self.s.m.i32(1 << step);
            let partner = self.s.m.bin(c, Bin::Xor, lane, bit);
            let theirs = self.read_lane(sum, partner);
            sum = self.add_halves(sum, theirs);
        }
        let (high, low, zero) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF), self.s.m.i32(0));
        let first = self.read_lane(x, zero);
        let first = self.s.m.bin(c, Bin::And, first, high);
        let sum = self.s.m.bin(c, Bin::And, sum, low);
        let mine = self.s.m.bin(c, Bin::Or, first, sum);
        if !halves {
            return Ok(mine);
        }
        let (lo, hi) = self.exchange_halves(c, mine);
        let c = self.cur;
        let total = self.add_halves(lo, hi);
        let total = self.s.m.bin(c, Bin::And, total, low);
        let first = self.s.m.bin(c, Bin::And, lo, high);
        Ok(self.s.m.bin(c, Bin::Or, first, total))
    }

    fn run(mut self) -> Result<Lowered, Stop> {
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        self.settle_interps();
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.cur = self.s.entry;
        self.find_sections();
        self.inputs()?;
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            if self.lds_meet.as_ref().is_some_and(|(_, meet)| meet[b].0) {
                self.lds_meeting();
            }
            let mut i = 0;
            self.mode16 = self.round16[b];
            while i < p.blocks[b].insts.len() {
                if let Some(&lanes) = self.x.st.discards.get(&(b, i)) {
                    // A pixel's discard: the lanes stay on as helpers.
                    let at = self.cur;
                    let offset = p.blocks[b].insts[i].machine.offset;
                    let kill = self
                        .boolean(lanes, at)
                        .map_err(|w| unsupported(offset, w))?;
                    self.s
                        .op(at, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
                }
                self.inst(b, i)?;
                self.mode16 = round16_after(&p.blocks[b].insts[i], self.mode16);
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    for j in i + 1..end {
                        self.section_inst(b, j, i);
                    }
                    i = end;
                    continue;
                }
                i += 1;
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            if self.lds_meet.as_ref().is_some_and(|(_, meet)| meet[b].1) {
                self.lds_meeting();
            }
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        self.fill_phis()?;
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let cmp_samplers = self.s.cmp_samplers;
        let wave_size = self.s.wave_size;
        let memory = self
            .s
            .root_srvs
            .iter()
            .any(|r| r.0 == super::shader::SLOT_MEMORY);
        let view_addresses = self.user_reads.values().any(|u| {
            use super::shader::USER_VIEW_ADDRESS;
            matches!(u.root, RootType::Other(code)
                if code == USER_VIEW_ADDRESS || code == USER_VIEW_ADDRESS + 1)
        });
        Ok(Lowered {
            wave_size,
            cmp_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            memory,
            memory_writes: self.memory_written,
            view_addresses,
            heap: self.heap[0],
            heap_writes: self.heap[1],
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            heap_constants: self.s.heap_constants,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            #[cfg(test)]
            rovs: self.s.flags & flags::ROVS != 0,
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
        })
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                // Most programs only interpolate with them: loaded on first
                // use as a value (`get`).
                Reg::V(_) if matches!(io_inputs.get(&v), Some(Input::Barycentric { .. })) => {
                    continue;
                }
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    // A scalar parked in a VGPR lane (`ir::unspill`), read where the
                    // paths only seem to allow it before any save: the lane of a VGPR
                    // nobody wrote, and VGPRs nobody set read as zero.
                    None if n >= crate::ir::PARKED => Val::I32(self.s.m.i32(0)),
                    // The primitive mask (pixel), which console code only
                    // moves to M0 for its interpolation (spilled and reloaded
                    // on the way); the host interpolates by itself, so the
                    // value goes unread.
                    None if self.s.pixel && self.x.io.initialized_sgprs == Some(n + 2) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    // The ordered section's collision id, and (`operand`) the
                    // id of the last wave that left it: 0, so no wave waits;
                    // the host's rasterizer-ordered views order the accesses.
                    None if self.pops && self.x.io.initialized_sgprs == Some(n + 1) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { kind, index } => return self.barycentric(kind, index),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let is_mask = self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v);
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(w)) => return Err(w),
            Some(&val) => return Ok(val),
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        if let Some(loaded) = self.lazy_barycentric(v) {
            loaded?;
            return Ok(self.vals[&v]);
        }
        Err("a value defined after its use")
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (see
    /// `inputs`): `None` if it is not one.
    fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        let (kind, index) = self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(w) => Err(w),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    /// shortcut: triangles only; on lines and points the console's i and j
    /// are not checked against the host's weights.
    fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let Some(mode) = interp_mode(kind) else {
            return Val::Missing("pull-model barycentrics");
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one): DXIL allows two.
        let bary = |e: &SigElement| e.kind == 28;
        let id = match self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0))
        {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword - self.user_first();
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        Val::I32(self.user_at(index))
    }

    /// The user-data index that is dword 0 of the host's user data: 0, or
    /// 16 for a program with a spilled user-data table, whose parameters
    /// all sit at 16 and above. (The host's root signature has 64 dwords:
    /// a builder stage with constants up to index 45, eight root views,
    /// their biases and the memory directory would not fit with 16 dwords
    /// nothing reads in front.)
    fn user_first(&self) -> u32 {
        if crate::spill::is_spilled(self.x.native) {
            16
        } else {
            0
        }
    }

    /// A user-data dword no register of the console's holds: one the host
    /// computes when it binds (`root` says what, an `Other` code of
    /// `shader.rs`), placed after the layout's own dwords.
    fn host_word(&mut self, root: RootType, slot: u8, dword: u32) -> V {
        let known = self
            .user_reads
            .values()
            .find(|u| u.root == root && u.slot == slot && u.dword == dword)
            .map(|u| u.index);
        let index = match known {
            Some(index) => index,
            None => {
                let first = self.user_first();
                let layout = &self.x.native.layout;
                let mut index = layout
                    .iter()
                    .map(|e| u32::from(e.user_data) + e.sgprs)
                    .max()
                    .unwrap_or(0)
                    .saturating_sub(first);
                while self.user_reads.contains_key(&index) {
                    index += 1;
                }
                self.user_reads.insert(
                    index,
                    UserData {
                        index,
                        slot,
                        root,
                        dword,
                    },
                );
                index
            }
        };
        self.user_at(index)
    }

    /// User-data dword `index`, loaded in the entry block.
    fn user_at(&mut self, index: u32) -> V {
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        self.s.m.extract(entry, r, index % 4)
    }

    /// Whether some instruction computes with `v` (scalar or vector
    /// arithmetic), rather than only naming it in a descriptor operand,
    /// moving it (at most `depth` moves are followed) or comparing it: a
    /// waterfall loop keeps a descriptor's dwords per lane, reads the first
    /// lane's back and compares the lanes' with them.
    fn computed_with(&self, v: Value, depth: u32) -> bool {
        self.x.p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
            if !i.reads.iter().any(|r| r.1 == v) {
                return false;
            }
            let name = i.name();
            let moves = matches!(
                name,
                "s_mov_b32"
                    | "s_mov_b64"
                    | "v_mov_b32"
                    | "v_readfirstlane_b32"
                    | "v_readlane_b32"
                    | "v_writelane_b32"
                    | "v_cndmask_b32"
            );
            match &i.fields {
                Fields::Sop { .. } | Fields::Valu(_) if moves => {
                    depth > 0 && i.writes.iter().any(|w| self.computed_with(w.1, depth - 1))
                }
                Fields::Sop { .. } | Fields::Valu(_) => {
                    !(name.starts_with("v_cmp") || name.starts_with("s_cmp"))
                }
                _ => false,
            }
        })
    }

    /// The title address (low dword, bits from 32 up) of the buffer view
    /// whose descriptor the scalar load at `site` reads from the start of
    /// a table entry, as the host tells it when it binds (`shader.rs`
    /// `USER_VIEW_ADDRESS`); none for anything else. (The console's
    /// descriptor holds more in the second dword, the stride from bit 16
    /// up; code that takes the address masks those away.)
    fn view_address(&mut self, site: Site) -> Option<(V, V)> {
        use super::shader::USER_VIEW_ADDRESS;
        let a = self.scalar.get(&site).copied()?;
        let rp = *self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table || a.dynamic.is_some() || a.offset % 32 != 0 {
            return None;
        }
        let dword = a.offset / 4;
        let used = self
            .x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)?;
        if used.dwords != 4 || !matches!(used.kind, ResourceType::Srv | ResourceType::Uav) {
            return None;
        }
        let lo = self.host_word(RootType::Other(USER_VIEW_ADDRESS), rp.slot, dword);
        let hi = self.host_word(RootType::Other(USER_VIEW_ADDRESS + 1), rp.slot, dword);
        Some((lo, hi))
    }

    /// `lo`, `hi` plus the instruction's byte offset `off`, with the carry.
    fn address_plus(&mut self, lo: V, hi: V, off: i32) -> (V, V) {
        if off == 0 {
            return (lo, hi);
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let k = self.s.m.i32(off as u32);
        let sum = self.s.m.bin(c, Bin::Add, lo, k);
        // Up: a carry when the sum wrapped below `lo`. Down: a borrow when
        // it wrapped above.
        let wrapped = self
            .s
            .m
            .cmp(c, if off > 0 { pred::IULT } else { pred::IUGT }, sum, lo);
        let one = self.s.m.cast(c, Cast::ZExt, wrapped, i32t);
        let hi = self.s.m.bin(c, if off > 0 { Bin::Add } else { Bin::Sub }, hi, one);
        (sum, hi)
    }

    /// Title memory at the 40-bit address `lo`, `hi` (`shader.rs`
    /// `SLOT_MEMORY`): the handle of the raw view of the host copy holding
    /// it and the byte in that view. A page without a copy reads zeros
    /// (and drops writes) through the null view at the first slot, and its
    /// number is left for the host, for lanes that run (`guard`).
    fn memory(&mut self, lo: V, hi: V, guard: Option<V>) -> (V, V) {
        use super::shader::{MEMORY_FAULTS, MEMORY_PAGES, SLOT_FAULTS, SLOT_MEMORY, USER_MEMORY_BASE};
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k16, k8) = (k(self, 16), k(self, 0xFF));
        let low = self.s.m.bin(c, Bin::LShr, lo, k16);
        let high = self.s.m.bin(c, Bin::And, hi, k8);
        let high = self.s.m.bin(c, Bin::Shl, high, k16);
        let page = self.s.m.bin(c, Bin::Or, low, high);
        // Inside the directory whatever the address (lanes that do not run
        // compute one too); its last entry is never filled.
        let last = k(self, MEMORY_PAGES - 1);
        let page = self
            .s
            .op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[page, last]);
        let k3 = k(self, 3);
        let at = self.s.m.bin(c, Bin::Shl, page, k3);
        let directory = self.s.root_srv_handle(SLOT_MEMORY);
        let (iu, both, align) = (self.s.m.undef(i32t), self.s.m.i8(3), k(self, 4));
        let entry = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            res,
            139,
            &[directory, at, iu, both, align],
        );
        let (copy, start) = (self.s.m.extract(c, entry, 0), self.s.m.extract(c, entry, 1));
        let (copy, start) = (self.as_int(copy), self.as_int(start));
        // A page without a copy: its number, at a place by the number, or
        // at the sink for lanes that do not run and pages that have one.
        let zero = k(self, 0);
        let mut missing = self.s.m.cmp(c, pred::EQ, copy, zero);
        if let Some(g) = guard {
            missing = self.s.m.bin(c, Bin::And, missing, g);
        }
        let places = k(self, MEMORY_FAULTS - 1);
        let place = self.s.m.bin(c, Bin::URem, page, places);
        let one = k(self, 1);
        let place = self.s.m.bin(c, Bin::Add, place, one);
        let k2 = k(self, 2);
        let place = self.s.m.bin(c, Bin::Shl, place, k2);
        let place = self.s.m.select(c, missing, place, zero);
        let faults = self.s.root_uav_handle(SLOT_FAULTS);
        let void = self.s.t.void;
        let mask = self.s.m.i8(1);
        self.s.op(
            c,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[faults, place, iu, page, iu, iu, iu, mask, align],
        );
        let base = self.host_word(RootType::Other(USER_MEMORY_BASE), 0, 0);
        let slot = self.s.m.bin(c, Bin::Add, base, copy);
        let h = self.s.heap_handle(c, slot, false, true, (11 | 0x1000, 0));
        let k64k = k(self, 0xFFFF);
        let within = self.s.m.bin(c, Bin::And, lo, k64k);
        let byte = self.s.m.bin(c, Bin::Add, start, within);
        (h, byte)
    }

    /// `global_load_*`, `global_store_*` and `global_atomic_*` on title
    /// memory by address (`resource::MEMORY`): the address is the VGPR
    /// pair, each lane its own.
    fn memory_flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            offset,
            lds,
            cache,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *lds || !matches!(saddr, None | Some(Operand::Scalar { code: NULL, .. })) {
            return Err("a global address with a scalar base");
        }
        let Some(Operand::Vector { reg: areg, .. }) = *addr else {
            return Err("a global address not in VGPRs");
        };
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a global address")?;
            l.i32_of(v)
        };
        let (lo, hi) = (word(self, 0)?, word(self, 1)?);
        let (lo, hi) = self.address_plus(lo, hi, i32::from(*offset));
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (i32t, res, void) = (self.s.t.i32, self.s.t.res_ret, self.s.t.void);
        let iu = self.s.m.undef(i32t);
        let align = self.s.m.i32(4);
        let name = inst.name();
        let count = |rest: &str| match rest {
            "dword" => Ok(1u8),
            "dwordx2" => Ok(2),
            "dwordx3" => Ok(3),
            "dwordx4" => Ok(4),
            _ => Err(name),
        };
        if let Some(rest) = name.strip_prefix("global_load_") {
            let n = count(rest)?;
            let Some(Operand::Vector { reg, .. }) = *dst else {
                return Err("a global load result");
            };
            let (h, byte) = self.memory(lo, hi, guard);
            let mask = self.s.m.i8((1u8 << n) - 1);
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                res,
                139,
                &[h, byte, iu, mask, align],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, u32::from(k));
                let x = self.as_int(x);
                self.write(inst, Reg::V(reg + k), x, guard);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("global_store_") {
            let n = count(rest)?;
            let d = data.as_ref().ok_or("global store data")?;
            let vals = self.vgprs(inst, d, n)?;
            let (h, byte) = self.memory(lo, hi, guard);
            self.memory_written = true;
            let mut args = vec![h, byte, iu];
            args.extend(&vals);
            args.resize(3 + 4, iu);
            args.push(self.s.m.i8((1u8 << n) - 1));
            args.push(align);
            let at = self.begin_guarded(guard);
            self.s
                .op(self.cur, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let Some(op) = name.strip_prefix("global_atomic_") else {
            return Err(name);
        };
        let d = data.as_ref().ok_or("global atomic data")?;
        let v = self.vgprs(inst, d, 1)?[0];
        // With `glc` the value from before comes back, in the result
        // register the instruction names.
        let back = match (*dst, cache.glc) {
            (Some(Operand::Vector { reg, .. }), true) => Some(reg),
            _ => None,
        };
        let old = match back.and_then(|reg| inst.read(Reg::V(reg))) {
            Some(x) => self.i32_of(x)?,
            None => self.s.m.i32(0),
        };
        let (h, byte) = self.memory(lo, hi, guard);
        self.memory_written = true;
        // DXIL's atomic operations: add 0, and 1, or 2, xor 3, signed
        // minimum 4 and maximum 5, unsigned 6 and 7, exchange 8.
        let simple = match op {
            "add" => Some((0, v)),
            "sub" => {
                let zero = self.s.m.i32(0);
                Some((0, self.s.m.bin(c, Bin::Sub, zero, v)))
            }
            "and" => Some((1, v)),
            "or" => Some((2, v)),
            "xor" => Some((3, v)),
            "smin" => Some((4, v)),
            "smax" => Some((5, v)),
            "umin" => Some((6, v)),
            "umax" => Some((7, v)),
            "swap" => Some((8, v)),
            "fmin" | "fmax" => None,
            _ => return Err(name),
        };
        let at = self.begin_guarded(guard);
        let was = match simple {
            Some((code, x)) => {
                let code = self.s.m.i32(code);
                self.s.op(
                    self.cur,
                    "atomicBinOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    78,
                    &[h, code, byte, iu, iu, x],
                )
            }
            None => {
                // A float minimum or maximum has no atomic operation on
                // the host: the dword is exchanged for the smaller (larger)
                // of itself and the value until no other thread got in
                // between. Each failed pass means another one's succeeded;
                // the limit only keeps a broken card from hanging.
                let head = self.cur;
                let mask = self.s.m.i8(1);
                let first = self.s.op(
                    head,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    res,
                    139,
                    &[h, byte, iu, mask, align],
                );
                let first = self.s.m.extract(head, first, 0);
                let first = self.as_int(first);
                let (pass, done) = (self.s.m.new_block(), self.s.m.new_block());
                self.s.m.terminate(head, Term::Br(pass));
                self.cur = pass;
                let seen = self.s.m.phi(pass, i32t);
                let n = self.s.m.phi(pass, i32t);
                let (f32t, fv) = (self.s.t.f32, self.as_float(v));
                let fseen = self.as_float(seen);
                // `dx.op.binary`: FMax 35, FMin 36.
                let pick = if op == "fmin" { 36 } else { 35 };
                let wanted = self
                    .s
                    .op(pass, "binary.f32", Attr::ReadNone, f32t, pick, &[fseen, fv]);
                let wanted = self.as_int(wanted);
                let found = self.s.op(
                    pass,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &[h, byte, iu, iu, seen, wanted],
                );
                let one = self.s.m.i32(1);
                let n1 = self.s.m.bin(pass, Bin::Add, n, one);
                let lost = self.s.m.cmp(pass, pred::NE, found, seen);
                let limit = self.s.m.i32(1 << 16);
                let more = self.s.m.cmp(pass, pred::IULT, n1, limit);
                let again = self.s.m.bin(pass, Bin::And, lost, more);
                self.s.m.terminate(pass, Term::CondBr(again, pass, done));
                let zero = self.s.m.i32(0);
                self.s.m.set_phi(seen, vec![(first, head), (found, pass)]);
                self.s.m.set_phi(n, vec![(zero, head), (n1, pass)]);
                self.cur = done;
                found
            }
        };
        let was = self.end_guarded(at, &[(was, old)])[0];
        if let Some(reg) = back {
            self.write(inst, Reg::V(reg), was, None);
        }
        Ok(())
    }

    /// `s_load_dword*` from title memory by address (`resource::MEMORY`):
    /// the SGPR pair plus the instruction's offsets.
    fn memory_smem(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            base: Operand::Scalar { code: base, .. },
            offset,
            soffset,
            ..
        } = inst.fields
        else {
            return Err("a scalar load from memory");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u8,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory access by address"),
        };
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::S(base + k)).ok_or("an address not in SGPRs")?;
            l.i32_of(v)
        };
        let (mut lo, mut hi) = (word(self, 0)?, word(self, 1)?);
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            // An offset register counts up from the address.
            let so = self.operand(inst, &soffset)?;
            let sum = self.s.m.bin(c, Bin::Add, lo, so);
            let carry = self.s.m.cmp(c, pred::IULT, sum, lo);
            let carry = self.s.m.cast(c, Cast::ZExt, carry, i32t);
            hi = self.s.m.bin(c, Bin::Add, hi, carry);
            lo = sum;
        }
        let (iu, align) = (self.s.m.undef(i32t), self.s.m.i32(4));
        for first in (0..count).step_by(4) {
            let n = (count - first).min(4);
            let (lo, hi) = self.address_plus(lo, hi, offset + 4 * i32::from(first));
            let (h, byte) = self.memory(lo, hi, None);
            let mask = self.s.m.i8((1u8 << n) - 1);
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                res,
                139,
                &[h, byte, iu, mask, align],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, u32::from(k));
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + first + k), x, None);
            }
        }
        Ok(())
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        // A mask that holds lanes outside EXEC where it is written (the complement of one,
        // say) is the ballot of the lanes that are there: all of them, where two exchanging
        // host waves know every lane runs that block.
        let made = self.def_block.get(&lo).copied();
        let whole = self.exchange.is_some() && !self.local && made.is_none_or(|b| self.all_lanes(b));
        if !self.x.st.active.contains(&lo) && !whole && !self.ballots.contains_key(&lo) {
            // Written by a scalar operation over other masks: that
            // operation over their numbers, as the console does.
            match self.mask_number(lo) {
                Ok(xy) => {
                    self.ballots.insert(lo, xy);
                }
                Err(_) => {
                    if self.exchange.is_some() && !self.local {
                        need(made.unwrap_or(0));
                    }
                    return Err(MASK_ELSEWHERE);
                }
            }
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry).
                    None => self.s.entry,
                    Some(&b) if b == self.cur_ir => self.cur,
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let bit = bit?;
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let r = self
                    .s
                    .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                self.s.flags |= flags::WAVE_OPS;
                let mut xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
                if self.exchange.is_some() {
                    // a host wave's ballot is its half's 32 bits: the halves exchange them
                    let b = self.def_block.get(&lo).copied();
                    if self.local {
                        if !self.local_ok && !self.local_any {
                            return Err("a lane mask read as data in a half-wave");
                        }
                    } else if !self.exchange.as_ref().is_some_and(|e| e.1) && !b.is_none_or(|b| self.all_lanes(b)) {
                        // (a group of 32 or fewer has no other half to exchange with)
                        need(b.unwrap_or(0));
                        return Err("a lane mask exchanged where not every lane runs");
                    }
                    xy = self.exchange_halves(at, xy.0);
                }
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high { y } else { x })
    }

    /// Both dwords of a mask holding lanes that do not reach the block
    /// writing it (a complement, a copy made where EXEC is not known), so
    /// no ballot there gives it: with console lanes, the scalar operation
    /// that writes it applied to its operands' numbers, as the console does.
    fn mask_number(&mut self, lo: Value) -> Result<(V, V), Why> {
        const ELSEWHERE: Why = "a lane mask holding lanes elsewhere read as data";
        if self.mask_data != MaskData::Console {
            return Err(ELSEWHERE);
        }
        let p = self.x.p;
        // A register the compiler reuses: a mask on one path, data on another.
        if let Some(&(b, i)) = self.phi_at.get(&lo)
            && p.blocks[b].phis[i]
                .args
                .iter()
                .any(|a| !self.x.st.masks.contains_key(a))
        {
            return Err("a register joining a lane mask and data, read as data");
        }
        let &b = self.def_block.get(&lo).ok_or(ELSEWHERE)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))
            .ok_or(ELSEWHERE)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return Err(ELSEWHERE);
        };
        let wave64 = self.x.native.wave64;
        let width = if wave64 { "_b64" } else { "_b32" };
        let op = inst
            .name()
            .strip_prefix("s_")
            .and_then(|n| n.strip_suffix(width))
            .ok_or(ELSEWHERE)?;
        if inst.write(Reg::S(*code)) != Some(lo) || inst.write(Reg::S(EXEC)).is_some() {
            return Err(ELSEWHERE);
        }
        let at = if b == self.cur_ir {
            self.cur
        } else {
            self.tail[b].ok_or("a lane mask written after its use")?
        };
        // Dword `h` of source `k` (inline constants sign-extend).
        let word = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            let x = match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or(ELSEWHERE)?;
                    return l.i32_of(v);
                }
                Some(&Operand::Const(c)) => {
                    let x = crate::resource::inline(c).ok_or(ELSEWHERE)?;
                    match h {
                        0 => x,
                        _ if (x as i32) < 0 => u32::MAX,
                        _ => 0,
                    }
                }
                Some(&Operand::Literal(x)) if h == 0 => x,
                Some(Operand::Literal(_)) => 0,
                _ => return Err(ELSEWHERE),
            };
            Ok(l.s.m.i32(x))
        };
        let mut out = [self.s.m.i32(0); 2];
        for h in 0..1 + u8::from(wave64) {
            let a = word(self, 0, h)?;
            let ones = self.s.m.i32(u32::MAX);
            out[usize::from(h)] = match op {
                "mov" => a,
                "not" => self.s.m.bin(at, Bin::Xor, a, ones),
                "cselect" => {
                    let c = word(self, 1, h)?;
                    let scc = inst.read(Reg::Scc).ok_or(ELSEWHERE)?;
                    let cond = match self.get(scc)? {
                        Val::Bit(x) => x,
                        Val::I32(x) => {
                            let z = self.s.m.i32(0);
                            self.s.m.cmp(at, pred::NE, x, z)
                        }
                        Val::Missing(w) => return Err(w),
                    };
                    self.s.m.select(at, cond, a, c)
                }
                _ => {
                    let c = word(self, 1, h)?;
                    // (operation, complement of src1, complement of the result)
                    let (f, not_c, not_r) = match op {
                        "and" => (Bin::And, false, false),
                        "or" => (Bin::Or, false, false),
                        "xor" => (Bin::Xor, false, false),
                        "andn2" => (Bin::And, true, false),
                        "orn2" => (Bin::Or, true, false),
                        "nand" => (Bin::And, false, true),
                        "nor" => (Bin::Or, false, true),
                        "xnor" => (Bin::Xor, false, true),
                        _ => return Err(ELSEWHERE),
                    };
                    let c = if not_c {
                        self.s.m.bin(at, Bin::Xor, c, ones)
                    } else {
                        c
                    };
                    let r = self.s.m.bin(at, f, a, c);
                    if not_r {
                        self.s.m.bin(at, Bin::Xor, r, ones)
                    } else {
                        r
                    }
                }
            };
        }
        Ok((out[0], out[1]))
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            // src_scc: SCC as 0 or 1.
            Operand::Const(253) => {
                let bit = self.scc_in(inst)?;
                Ok(self.s.m.cast(self.cur, Cast::ZExt, bit, self.s.t.i32))
            }
            // `src_pops_exiting_wave_id` (see `inputs`).
            Operand::Const(239) if self.pops => Ok(self.s.m.i32(0)),
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// An operand read as f16 data: GFX10 puts an inline f16 constant in
    /// the low half of the operand (upstream PR #637, checked there against
    /// the instruction encodings; the half-and-half reading taken from PR
    /// #458 gave the high half the constant too).
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => Ok(self.s.m.i32(inline_f16(c))),
            _ => self.operand(inst, op),
        }
    }

    /// VGPR source `k` read for its low 16 bits alone, when only those are
    /// known (`low16`); the caller applies the modifiers and selects.
    fn low16_src(&self, inst: &Inst, k: usize) -> Option<V> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        match v.src.get(k) {
            Some(&Operand::Vector { reg, .. }) if v.dpp.is_none() => inst
                .read(Reg::V(reg))
                .and_then(|o| self.low16.get(&o).copied()),
            _ => None,
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let sel = Self::sdwa_src(inst, k);
                // A byte or word select within the low half.
                let low = matches!(sel, Some((0 | 1 | 4, _)));
                let x = match self.low16_src(inst, k).filter(|_| low) {
                    Some(x) => x,
                    None => self.operand(inst, &op)?,
                };
                Ok(match sel {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// A 32-bit denormal as the console's float mode reads it: a zero of its
    /// sign, decided on the bits (the host's own denormal mode varies).
    fn ftz(&mut self, f: V) -> V {
        let c = self.cur;
        let x = self.as_int(f);
        let (exp, sign, zero) = (
            self.s.m.i32(0x7F80_0000),
            self.s.m.i32(0x8000_0000),
            self.s.m.i32(0),
        );
        let e = self.s.m.bin(c, Bin::And, x, exp);
        let den = self.s.m.cmp(c, pred::EQ, e, zero);
        let s = self.s.m.bin(c, Bin::And, x, sign);
        let r = self.s.m.select(c, den, s, x);
        self.as_float(r)
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but of the operands with denormals read as zeros and with
    /// -0 ordered below +0 (RDNA2 ISA, `v_max_f32`), where the compare and
    /// select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let (a, b) = (self.ftz(a), self.ftz(b));
        self.max_min_flushed(max, a, b)
    }

    /// [`Self::max_min`] of operands without denormals.
    fn max_min_flushed(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, as a 16-bit float result takes it:
    /// rounded as the program's MODE says ([`Self::narrow16`]), or, with
    /// `VICTORY_F16_NEAREST=0`, as the host's own conversion rounds
    /// ([`f16_nearest_on`]).
    fn f16_result(&mut self, f: V) -> V {
        if f16_nearest_on() {
            self.narrow16(f)
        } else {
            let i32t = self.s.t.i32;
            self.s
                .op(self.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
        }
    }

    /// `f` as an f16 in the low bits, rounded as the program's MODE says at
    /// this point: toward zero where it set that for 16-bit results (the
    /// title does around its 16-bit interpolation), else to nearest even;
    /// f32 denormals read as zero; on the bits, where the host's own
    /// conversion rounds as it likes (toward zero on the RX 6700 XT and WARP).
    // shortcut: rounding toward +-infinity, or a mode not known at this
    // point (paths that set it differently join), rounds to nearest even.
    fn narrow16(&mut self, f: V) -> V {
        let rtz = self.mode16 == Some(3);
        let c = self.cur;
        let x = self.as_int(f);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k13, k16, k31) = (k(self, 13), k(self, 16), k(self, 31));
        let t = self.s.m.bin(c, Bin::LShr, x, k16);
        let m = k(self, 0x8000);
        let sign = self.s.m.bin(c, Bin::And, t, m);
        let m = k(self, 0x7FFF_FFFF);
        let a = self.s.m.bin(c, Bin::And, x, m);
        // Normal halves: the exponent rebased, the mantissa rounded.
        let bias = k(self, 0x3800_0000);
        let rebased = self.s.m.bin(c, Bin::Sub, a, bias);
        let normal = if rtz {
            self.s.m.bin(c, Bin::LShr, rebased, k13)
        } else {
            let one = k(self, 1);
            let lsb = self.s.m.bin(c, Bin::LShr, rebased, k13);
            let lsb = self.s.m.bin(c, Bin::And, lsb, one);
            let half = k(self, 0xFFF);
            let up = self.s.m.bin(c, Bin::Add, rebased, half);
            let up = self.s.m.bin(c, Bin::Add, up, lsb);
            self.s.m.bin(c, Bin::LShr, up, k13)
        };
        let (inf, big) = (k(self, 0x7C00), k(self, if rtz { 0x7BFF } else { 0x7C00 }));
        let over = self.s.m.cmp(c, pred::IUGE, normal, inf);
        let normal = self.s.m.select(c, over, big, normal);
        let infinity = k(self, 0x7F80_0000);
        let is_inf = self.s.m.cmp(c, pred::EQ, a, infinity);
        let normal = self.s.m.select(c, is_inf, inf, normal);
        // Denormal halves: the mantissa shifted into units of 2^-24.
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::LShr, a, k23);
        let k126 = k(self, 126);
        let s = self.s.m.bin(c, Bin::Sub, k126, e);
        let far = self.s.m.cmp(c, pred::IUGT, s, k31);
        let s = self.s.m.select(c, far, k31, s);
        let (frac, hidden, zero) = (k(self, 0x7F_FFFF), k(self, 0x80_0000), k(self, 0));
        let mant = self.s.m.bin(c, Bin::And, a, frac);
        let mant = self.s.m.bin(c, Bin::Or, mant, hidden);
        let f32_denormal = self.s.m.cmp(c, pred::EQ, e, zero);
        let mant = self.s.m.select(c, f32_denormal, zero, mant);
        let q = self.s.m.bin(c, Bin::LShr, mant, s);
        let den = if rtz {
            q
        } else {
            let one = k(self, 1);
            let s1 = self.s.m.bin(c, Bin::Sub, s, one);
            let half = self.s.m.bin(c, Bin::LShr, mant, s1);
            let half = self.s.m.bin(c, Bin::And, half, one);
            let below = self.s.m.bin(c, Bin::Shl, one, s1);
            let below = self.s.m.bin(c, Bin::Sub, below, one);
            let rest = self.s.m.bin(c, Bin::And, mant, below);
            let rest = self.s.m.cmp(c, pred::NE, rest, zero);
            let rest = self.s.m.cast(c, Cast::ZExt, rest, self.s.t.i32);
            let lsb = self.s.m.bin(c, Bin::And, q, one);
            let tie = self.s.m.bin(c, Bin::Or, rest, lsb);
            let up = self.s.m.bin(c, Bin::And, half, tie);
            self.s.m.bin(c, Bin::Add, q, up)
        };
        let min_normal = k(self, 0x3880_0000);
        let small = self.s.m.cmp(c, pred::IULT, a, min_normal);
        let r = self.s.m.select(c, small, den, normal);
        let is_nan = self.s.m.cmp(c, pred::IUGT, a, infinity);
        // A NaN keeps the top of its payload, quieted (as the RX 6700 XT,
        // RDNA2 like the console, converts).
        let payload = self.s.m.bin(c, Bin::LShr, a, k13);
        let (low10, quiet) = (k(self, 0x3FF), k(self, 0x7E00));
        let payload = self.s.m.bin(c, Bin::And, payload, low10);
        let nan = self.s.m.bin(c, Bin::Or, payload, quiet);
        let r = self.s.m.select(c, is_nan, nan, r);
        self.s.m.bin(c, Bin::Or, r, sign)
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `f` as an f16 in the low bits, rounded to nearest, ties to even, as the console's
    /// 16-bit float operations round, whatever the host's own conversion does: D3D lets
    /// `f32tof16` round toward zero, and the NVIDIA driver's does (measured on an RTX 4070:
    /// `wave64_tests`). The conversion toward zero, or the next half away from zero where
    /// that one is nearer, or as near and the first one odd. Past the largest half the next
    /// one counts as 2^16, the bound IEEE rounds by, so 65520 and up become infinity.
    /// (`f` is an f32 result of two halves here; rounding it again to 16 bits gives the
    /// rounding of the exact result, 24 bits being more than twice 11 and two.)
    fn f16_nearest(&mut self, f: V) -> V {
        let c = self.cur;
        let f32t = self.s.t.f32;
        let lo = self.f16_toward_zero(f);
        let (one, zero) = (self.s.m.i32(1), self.s.m.i32(0));
        let hi = self.s.m.bin(c, Bin::Add, lo, one);
        let size = |l: &mut Self, h: V| {
            let w = l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
            l.unary(6, w)
        };
        let (below, above) = (size(self, lo), size(self, hi));
        let (magnitude, infinity) = (self.s.m.i32(0x7FFF), self.s.m.i32(0x7C00));
        let m = self.s.m.bin(c, Bin::And, hi, magnitude);
        let top = self.s.m.cmp(c, pred::EQ, m, infinity);
        let bound = self.s.m.f32(65536.0f32.to_bits());
        let above = self.s.m.select(c, top, bound, above);
        let fa = self.unary(6, f);
        // (both differences are exact: the three values lie within one half's step)
        let under = self.s.m.bin(c, Bin::Sub, fa, below);
        let over = self.s.m.bin(c, Bin::Sub, above, fa);
        // Ordered: never for NaN, which stays what the conversion made of it.
        let nearer = self.s.m.cmp(c, pred::OLT, over, under);
        let tie = self.s.m.cmp(c, pred::OEQ, over, under);
        let odd = self.s.m.bin(c, Bin::And, lo, one);
        let odd = self.s.m.cmp(c, pred::NE, odd, zero);
        let tie = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, nearer, tie);
        self.s.m.select(c, up, hi, lo)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// Whether an f32 value is NaN, tested in integer arithmetic so host
    /// drivers (such as NVIDIA) do not fold `fcmp uno` away under fast-math.
    fn is_nan_f32(&mut self, f: V) -> V {
        let c = self.cur;
        let bits = self.as_int(f);
        let mask = self.s.m.i32(0x7FFF_FFFF);
        let abs = self.s.m.bin(c, Bin::And, bits, mask);
        let inf = self.s.m.i32(0x7F80_0000);
        self.s.m.cmp(c, pred::IUGT, abs, inf)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let (a, b, third) = (self.ftz(a), self.ftz(b), self.ftz(third));
        let lo = self.max_min_flushed(false, a, b);
        let hi = self.max_min_flushed(true, a, b);
        let t = self.max_min_flushed(false, hi, third);
        let median = self.max_min_flushed(true, lo, t);
        let least = self.max_min_flushed(false, lo, third);
        let nan_a = self.is_nan_f32(a);
        let nan_b = self.is_nan_f32(b);
        let nan_3 = self.is_nan_f32(third);
        let nan_ab = self.s.m.bin(c, Bin::Or, nan_a, nan_b);
        let nan = self.s.m.bin(c, Bin::Or, nan_ab, nan_3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// A console wave64 program in host waves of 32: the first console thread of this host wave
    /// (0, 32, 64, ...). A GPU need not give a host wave consecutive thread indices (NVIDIA's
    /// 8 x 8 groups), so each wave takes the next 32 from an LDS counter instead: every thread
    /// zeroes it, the group meets, each wave's first lane adds 32 and the wave takes what it
    /// found. Branch-free, in the entry block, which every thread runs.
    fn wave_base(&mut self) -> Option<V> {
        let slot = self.halves?;
        if let Some(v) = self.wave_base {
            return Some(v);
        }
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let lds = self.s.lds();
        let k = self.s.m.i32(slot);
        let ptr = self.s.m.element(e, lds, k);
        let zero = self.s.m.i32(0);
        self.s.m.store(e, ptr, zero);
        self.group_sync(e);
        let lane = self.s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let first = self.s.m.cmp(e, pred::EQ, lane, zero);
        let k32 = self.s.m.i32(32);
        let add = self.s.m.select(e, first, k32, zero);
        let ptr = self.s.m.element(e, lds, k);
        let old = self.s.m.atomic(e, Rmw::Add, ptr, add);
        let base = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[old]);
        self.s.flags |= flags::WAVE_OPS;
        self.wave_base = Some(base);
        Some(base)
    }

    /// The console thread index of this thread in the group: `wave_base` plus the host lane
    /// (host waves of 32), else the wave's first thread index rounded down plus the lane.
    fn console_thread(&mut self) -> V {
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let lane = self.s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        if let Some(base) = self.wave_base() {
            return self.s.m.bin(e, Bin::Add, base, lane);
        }
        let flat = self.s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[]);
        let first = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[flat]);
        let wave = self.s.wave_size.unwrap_or(32);
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        self.s.m.bin(e, Bin::Add, base, lane)
    }

    /// This thread's console lane (0..63): its relabelled thread index (`thread_id`), whose host
    /// wave holds a 32-aligned run of console lanes. Without `exchange`: the host lane.
    fn console_lane(&mut self) -> V {
        if self.exchange.is_none() {
            return self.lane_index();
        }
        if let Some(v) = self.clane {
            return v;
        }
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let f = self.console_thread();
        let _ = i32t;
        let k63 = self.s.m.i32(63);
        let cl = self.s.m.bin(e, Bin::And, f, k63);
        self.s.flags |= flags::WAVE_OPS;
        self.clane = Some(cl);
        cl
    }

    /// A group barrier with an LDS fence in `block`.
    fn group_sync(&mut self, block: usize) {
        let mode = self.s.m.i32(meeting_mode());
        let void = self.s.t.void;
        self.s.op(block, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// `exchange`: each host wave gives `mine` (the same in all its lanes) for its half; both get
    /// (lower half's, upper half's). A group with no upper half gets 0 for it.
    fn exchange_halves(&mut self, at: usize, mine: V) -> (V, V) {
        let Some((slot, small, _)) = self.exchange.as_ref().map(|(a, b, _)| (*a, *b, ())) else {
            let z = self.s.m.i32(0);
            return (mine, z);
        };
        if small {
            let z = self.s.m.i32(0);
            return (mine, z);
        }
        if self.local {
            // each host wave on its own: its half holds `mine`, the other nothing
            let cl = self.console_lane();
            let (k32, z) = (self.s.m.i32(32), self.s.m.i32(0));
            let upper = self.s.m.cmp(at, pred::IUGE, cl, k32);
            let lo = self.s.m.select(at, upper, z, mine);
            let hi = self.s.m.select(at, upper, mine, z);
            return (lo, hi);
        }
        // the pair of dwords of this thread's console wave: its half's, then both
        let pair = self.exchange_pair(slot);
        let cl = self.console_lane();
        let five = self.s.m.i32(5);
        let half = self.s.m.bin(at, Bin::LShr, cl, five);
        let index = self.s.m.bin(at, Bin::Add, pair, half);
        let lds = self.s.lds();
        let ptr = self.s.m.element(at, lds, index);
        self.s.m.store(at, ptr, mine);
        self.group_sync(at);
        let one = self.s.m.i32(1);
        let k1 = self.s.m.bin(at, Bin::Add, pair, one);
        let p0 = self.s.m.element(at, lds, pair);
        let p1 = self.s.m.element(at, lds, k1);
        let lo = self.s.m.load(at, p0);
        let hi = self.s.m.load(at, p1);
        // the slots are written again only after every thread read them
        self.group_sync(at);
        (lo, hi)
    }

    /// The first of the two exchange dwords of this thread's console wave (made in the entry
    /// block): a group of several console waves has a pair for each.
    fn exchange_pair(&mut self, slot: u32) -> V {
        if let Some(v) = self.pair {
            return v;
        }
        let e = self.s.entry;
        let thread = self.console_thread();
        let six = self.s.m.i32(6);
        let wave = self.s.m.bin(e, Bin::LShr, thread, six);
        let one = self.s.m.i32(1);
        let twice = self.s.m.bin(e, Bin::Shl, wave, one);
        let base = self.s.m.i32(slot);
        let v = self.s.m.bin(e, Bin::Add, base, twice);
        self.pair = Some(v);
        v
    }

    /// The console's decision of the branch closing IR block `b`, which the wave takes
    /// together ([`Structured::wide`]): `when` is the lane's own bit, and the branch is taken
    /// when some lane's `when` holds (`execnz`, `vccnz`, SCC of a mask operation set) or
    /// when every lane's does (`execz`, `vccz`, that SCC clear), [`Structured::wave_any`].
    /// Each host wave asks its lanes and the halves exchange the answers, so both go the
    /// same way and can meet again inside the branch.
    /// SCC of a mask operation as data ([`Structured::scc_some`]): whether some lane of the
    /// console wave has its bit of the result, asked where the operation stands. The lanes
    /// that are elsewhere have no bit (the result is ANDed with EXEC or is EXEC), so with
    /// the console's waves the host wave's answer is the console's; two host waves of 32
    /// exchange theirs, which needs every lane there.
    fn scc_some(&mut self, scc: Value) -> Result<V, Why> {
        let &m = self.x.st.scc_some.get(&scc).ok_or(LANE_MASK)?;
        // (not for a console wave cut into host waves that do not exchange)
        if !self.console_lanes || self.local || self.exchange.is_none() && self.halves.is_some() {
            return Err(LANE_MASK);
        }
        let made = self.def_block.get(&scc).copied();
        let at = match made {
            None => self.s.entry,
            Some(b) if b == self.cur_ir => self.cur,
            Some(b) => self.tail[b].ok_or("a lane mask written after its use")?,
        };
        let bit = self.boolean(m, at)?;
        let (i1t, i32t) = (self.s.t.i1, self.s.t.i32);
        let some = self.s.op(at, "waveAnyTrue", Attr::NoUnwind, i1t, 113, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        if !self.exchange.as_ref().is_some_and(|e| !e.1) {
            return Ok(some);
        }
        if !made.is_none_or(|b| self.all_lanes(b)) {
            need(made.unwrap_or(0));
            return Err("a lane mask exchanged where not every lane runs");
        }
        let some = self.s.m.cast(at, Cast::ZExt, some, i32t);
        let (lo, hi) = self.exchange_halves(at, some);
        let both = self.s.m.bin(at, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(at, pred::NE, both, z))
    }

    fn wave_wide(&mut self, b: usize, when: Bool) -> Result<V, Why> {
        const WHY: Why = "a wave-wide branch where not every lane runs";
        let any = self.x.st.wave_any.get(b).copied().flatten().ok_or("a wave-wide branch on a scalar condition")?;
        if self.exchange.is_none() && self.halves.is_none() && self.console_lanes && self.mask_data == MaskData::Console {
            // The host's waves are the console's: the wave's own answer. The lanes that are
            // elsewhere have no bit set, as on the console, so "some lane" needs no one but
            // the lanes here, and "every lane's bit is clear" is true of the others anyway.
            let c = self.cur;
            let mine = self.boolean(when, c)?;
            let i1t = self.s.t.i1;
            self.s.flags |= flags::WAVE_OPS;
            let (name, code) = if any { ("waveAnyTrue", 113) } else { ("waveAllTrue", 114) };
            return Ok(self.s.op(c, name, Attr::NoUnwind, i1t, code, &[mine]));
        }
        if self.local || !self.exchange.as_ref().is_some_and(|e| !e.1) {
            return Err(WHY);
        }
        if !self.all_lanes(b) {
            need(b);
            return Err(WHY);
        }
        let c = self.cur;
        let mine = self.boolean(when, c)?;
        // what the halves exchange is "some lane speaks against" either way (for `any`:
        // some lane's `when` holds; else: some lane's does not), so a half with no lane says
        // nothing
        let set = if any {
            mine
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, mine, t)
        };
        let (i1t, i32t) = (self.s.t.i1, self.s.t.i32);
        let some = self.s.op(c, "waveAnyTrue", Attr::NoUnwind, i1t, 113, &[set]);
        self.s.flags |= flags::WAVE_OPS;
        let some = self.s.m.cast(c, Cast::ZExt, some, i32t);
        let (lo, hi) = self.exchange_halves(c, some);
        let c = self.cur;
        let both = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, if any { pred::NE } else { pred::EQ }, both, z))
    }

    /// Whether every lane runs IR block `b` (always true without `exchange`).
    fn all_lanes(&self, b: usize) -> bool {
        self.exchange.as_ref().is_none_or(|(_, _, all)| all.get(b).copied().unwrap_or(false))
    }

    /// The bit of lane `lane` in the mask `lo`, `hi`.
    fn lane_bit_of(&mut self, at: usize, lane: V, lo: V, hi: V) -> V {
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let low = self.s.m.cmp(at, pred::IULT, lane, k32);
        let word = self.s.m.select(at, low, lo, hi);
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let lane = if self.exchange.is_some() {
            self.console_lane()
        } else {
            self.s.op(at, "waveGetLaneIndex", Attr::ReadOnly, self.s.t.i32, 111, &[])
        };
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    /// Whether every lane outside the guard of the export at (b, i) runs
    /// the `done` export later in block b without its own guard holding,
    /// i.e. is discarded there.
    fn discarded_at_done(&self, b: usize, i: usize) -> bool {
        let guards = &self.x.st.guards[b];
        let g = guards[i].unwrap_or(Bool::TRUE);
        self.exports.iter().any(|(&(eb, ei), e)| {
            e.done && eb == b && ei > i && {
                let done = guards[ei].unwrap_or(Bool::TRUE);
                !self.x.st.bools.meet(&[(done, true), (g, false)])
            }
        })
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let o = inst.read(reg);
                let read = o.is_some_and(|o| self.read_phis.contains(&o));
                let old = match o.map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                // What lanes keep is zero where it has no DXIL form (as
                // above), so keeping a phi does not make it live.
                if let Some(o) = o.filter(|_| !read)
                    && self.read_phis.remove(&o)
                {
                    self.kept_phis.insert(o);
                }
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                // On the flushed result; a zero or denormal product comes out
                // +0 (measured on RDNA2 with IEEE mode off).
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.cur;
                let k = self.s.m.f32(factor.to_bits());
                let r = self.ftz(f);
                let r = self.s.m.bin(c, Bin::Mul, r, k);
                let x = self.as_int(r);
                let (exp, zero) = (self.s.m.i32(0x7F80_0000), self.s.m.i32(0));
                let e = self.s.m.bin(c, Bin::And, x, exp);
                let tiny = self.s.m.cmp(c, pred::EQ, e, zero);
                let x = self.s.m.select(c, tiny, zero, x);
                f = self.as_float(x);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        if self.halves.is_some() {
            let f = self.console_thread();
            self.s.flags |= flags::WAVE_OPS;
            let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
            return Val::I32(match c {
                0 => self.s.m.bin(e, Bin::URem, f, xk),
                1 => {
                    let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                    self.s.m.bin(e, Bin::URem, t, yk)
                }
                _ => self.s.m.bin(e, Bin::UDiv, f, xy),
            });
        }
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.lane_index();
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mut mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        // D3D12 caps the outputs of the declared maxima at 32 KiB. Lane n
        // writes only vertex n and primitive n, so maxima past the lanes are
        // never reached: they come down to the lane count when the outputs
        // do not fit.
        use super::container::{MESH_OUTPUT_LIMIT, mesh_output_bytes};
        let (outputs, prims) = (&self.s.outputs, &self.s.prims);
        let fits = |m: super::shader::Mesh| {
            mesh_output_bytes(outputs, prims, m.max_vertices, m.max_primitives) <= MESH_OUTPUT_LIMIT
        };
        if !fits(mesh) {
            mesh.max_vertices = mesh.max_vertices.min(threads);
            mesh.max_primitives = mesh.max_primitives.min(threads);
            // shortcut: outputs past the cap even at the lane count are
            // still emitted with the lower maxima (drivers take them);
            // refuse them here if one hangs.
            // Experiment (`VICTORY_MESH_OUTPUT_STRICT=1`): they are refused.
            if !fits(mesh) && std::env::var_os("VICTORY_MESH_OUTPUT_STRICT").is_some() {
                return Err("a mesh shader whose outputs pass the host's 32 KiB");
            }
            self.s.mesh = Some(mesh);
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let z = self.s.m.i32(0);
        // Every primitive of the declared count starts culled, also those
        // past the lanes (a count of 256 with 128 lanes): lane n culls
        // primitives n, n + lanes, ... The host draws all it was promised,
        // and one never written has whatever indices its memory held: long
        // triangles over the whole target, each pixel of them shaded.
        let passes = mesh.max_primitives.div_ceil(threads.max(1));
        for pass in 0..passes {
            let index = if pass == 0 {
                flat
            } else {
                let past = self.s.m.i32(pass * threads);
                self.s.m.bin(self.cur, Bin::Add, flat, past)
            };
            let guard = ((pass + 1) * threads > mesh.max_primitives)
                .then(|| self.s.m.cmp(self.cur, pred::IULT, index, p));
            let at = self.begin_guarded(guard);
            self.s.op(
                self.cur,
                "emitIndices",
                Attr::NoUnwind,
                void,
                169,
                &[index, z, z, z],
            );
            self.end_guarded(at, &[]);
        }
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// `x` in the first lane of this host wave where `guard` holds, and whether it has such a
    /// lane: the console reads the first lane of EXEC, and a lane that got here with EXEC clear
    /// (a guard, not a branch) is not one. With none, `x` of the wave's first lane here.
    fn first_lane(&mut self, x: V, guard: Option<V>) -> (V, V) {
        let c = self.cur;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let first = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
        let Some(g) = guard else {
            return (first, self.s.m.bool(true));
        };
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self.s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[g]);
        let (lo, hi) = (self.s.m.extract(c, r, 0), self.s.m.extract(c, r, 1));
        let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
        // FirstbitLo (dx.op 32): the lowest set bit
        let flo = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
        let fhi = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
        let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
        let in_lo = self.s.m.cmp(c, pred::NE, lo, z);
        let in_hi = self.s.m.cmp(c, pred::NE, hi, z);
        let any = self.s.m.bin(c, Bin::Or, in_lo, in_hi);
        let lane = self.s.m.select(c, in_lo, flo, fhi);
        // no lane in EXEC: read this lane (one that is here); the read is not used
        let me = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let lane = self.s.m.select(c, any, lane, me);
        let at = self.read_lane(x, lane);
        (self.s.m.select(c, any, at, first), any)
    }

    /// Whether a quad-local read must reach discarded lanes: after a pixel's
    /// discard they are the host's helpers, which only quad reads see (wave
    /// reads give 0 from them on AMD).
    fn helpers_read(&self, quad: bool) -> bool {
        quad && self.s.pixel && !self.x.st.discards.is_empty()
    }

    /// `v_readfirstlane_b32`'s result `first` of `own` in a pixel shader:
    /// a helper lane takes its own value instead.
    ///
    /// On the console, whole-quad mode puts helper lanes in EXEC, so a
    /// waterfall loop (read the first lane's value, run the lanes with that
    /// value, take them out of EXEC) serves helper lanes too. On the host,
    /// wave operations leave helper lanes out (DXIL before
    /// `WaveOpsIncludeHelperLanes`): once only helper lanes are left in such
    /// a loop, `WaveReadLaneFirst` reads no lane, no lane matches, and the
    /// loop runs to [`LOOP_LIMIT`] in every such lane: draws of 0.3 to 1.3 s
    /// each, then DEVICE_HUNG. With its own value a helper lane matches in
    /// the first pass and leaves; what a helper lane computes only feeds
    /// derivatives, and its stores are dropped.
    fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let r = if mesh_quad || self.helpers_read(quad) {
                self.quad_pick(x, src)
            } else {
                self.read_lane(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if name == "ds_bpermute_b32" {
            // Backward permute: no LDS is touched. Each lane takes the data
            // register of the lane its address names, (address + offset) / 4
            // within the wave; a source lane outside EXEC gives 0. (Grand
            // Theft Auto VI's checkerboard shadow-mask generators read their
            // neighbours' samples with it; without them every shadowed light
            // was multiplied by an empty mask.)
            self.cross_lane(false)?;
            if self.exchange.as_ref().is_some_and(|e| !e.1) {
                // a console wave of 64 run as two host waves: the lane named
                // may be in the other one
                return Err("a lane permute across two host waves");
            }
            let at = self.vgprs(inst, addr.as_ref().ok_or("a permute address")?, 1)?[0];
            let x = self.vgprs(inst, data.first().ok_or("a permute source")?, 1)?[0];
            let o = self.s.m.i32(offset);
            let at = self.s.m.bin(c, Bin::Add, at, o);
            let two = self.s.m.i32(2);
            let src = self.s.m.bin(c, Bin::LShr, at, two);
            let last = self.s.m.i32(self.wave() - 1);
            let src = self.s.m.bin(c, Bin::And, src, last);
            let r = self.read_lane(x, src);
            // 0 from a lane that is not running here (outside EXEC, or
            // gone through a branch): the ballot of the lanes that are.
            let on = guard.unwrap_or_else(|| self.s.m.bool(true));
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let ballot = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
            let (lo, hi) = (
                self.s.m.extract(c, ballot, 0),
                self.s.m.extract(c, ballot, 1),
            );
            let running = self.lane_bit_of(c, src, lo, hi);
            let z = self.s.m.i32(0);
            let r = self.s.m.select(c, running, r, z);
            let reg = dst_reg.ok_or("a permute result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        // shortcut: LDS addresses are taken as dword-aligned and in bounds
        // (out-of-bounds console accesses read 0 and drop writes).
        let dword = |l: &mut Self, bytes: u32| -> V {
            let c = l.cur;
            let o = l.s.m.i32(bytes);
            let at = l.s.m.bin(c, Bin::Add, a, o);
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let lds = l.s.lds();
            l.s.m.element(c, lds, index)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        // Bytes and halves: the dword holding them, and their bit position
        // in it (naturally aligned, so they never straddle dwords).
        // `_d16` reads fill a register's low half (`_d16_hi` its high half)
        // and keep the other; `_d16_hi` writes take the data's high half.
        let base = name.trim_end_matches("_hi").trim_end_matches("_d16");
        let sub = match base {
            "ds_read_u8" | "ds_read_i8" | "ds_write_b8" => Some(8u32),
            "ds_read_u16" | "ds_read_i16" | "ds_write_b16" => Some(16),
            _ => None,
        };
        let sub = sub.filter(|_| {
            base == name
                || name.ends_with("_d16") && base.starts_with("ds_read")
                || name.ends_with("_d16_hi")
        });
        if let Some(bits) = sub {
            let p = dword(self, offset);
            let o = self.s.m.i32(offset);
            let at = self.s.m.bin(c, Bin::Add, a, o);
            let three = self.s.m.i32(3);
            let byte = self.s.m.bin(c, Bin::And, at, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let mask = self.s.m.i32((1u32 << bits) - 1);
            if name.starts_with("ds_read") {
                let reg = dst_reg.ok_or("an LDS read result")?;
                let w = self.s.m.load(c, p);
                let x = self.s.m.bin(c, Bin::LShr, w, sh);
                let x = if base.contains("_i") {
                    let k = self.s.m.i32(32 - bits);
                    let x = self.s.m.bin(c, Bin::Shl, x, k);
                    self.s.m.bin(c, Bin::AShr, x, k)
                } else {
                    self.s.m.bin(c, Bin::And, x, mask)
                };
                let x = if base == name {
                    x
                } else {
                    let old = match inst.read(Reg::V(reg)) {
                        Some(v) => self.i32_of(v)?,
                        None => self.s.m.i32(0),
                    };
                    let half = self.s.m.i32(0xFFFF);
                    let x = self.s.m.bin(c, Bin::And, x, half);
                    let (x, keep) = if name.ends_with("_hi") {
                        let k16 = self.s.m.i32(16);
                        (self.s.m.bin(c, Bin::Shl, x, k16), half)
                    } else {
                        (x, self.s.m.i32(0xFFFF_0000))
                    };
                    let old = self.s.m.bin(c, Bin::And, old, keep);
                    self.s.m.bin(c, Bin::Or, old, x)
                };
                self.write(inst, Reg::V(reg), x, guard);
            } else {
                // Clear, then set: two atomics, so lanes writing other
                // bytes of the dword keep theirs.
                let mut x = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                if name != base {
                    // The data's high half.
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let x = self.s.m.bin(c, Bin::And, x, mask);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let m = self.s.m.bin(c, Bin::Shl, mask, sh);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, m, ones);
                use super::bitcode::Rmw;
                let at = self.begin_guarded(guard);
                self.s.m.atomic(self.cur, Rmw::And, p, keep);
                self.s.m.atomic(self.cur, Rmw::Or, p, x);
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_read") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS read of bytes or halves");
            };
            let reg = dst_reg.ok_or("an LDS read result")?;
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let p = dword(self, off + 4 * k);
                    let x = self.s.m.load(self.cur, p);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS write of bytes or halves");
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let p = dword(self, off);
                self.s.m.store(self.cur, p, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        let op = name
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        use super::bitcode::Rmw;
        let rmw = match op.as_str() {
            "add_u32" => Some(Rmw::Add),
            "sub_u32" => Some(Rmw::Sub),
            "and_b32" => Some(Rmw::And),
            "or_b32" => Some(Rmw::Or),
            "xor_b32" => Some(Rmw::Xor),
            "min_i32" => Some(Rmw::Min),
            "max_i32" => Some(Rmw::Max),
            "min_u32" => Some(Rmw::UMin),
            "max_u32" => Some(Rmw::UMax),
            "wrxchg_b32" => Some(Rmw::Xchg),
            "cmpst_b32" | "max_f32" | "min_f32" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
        let outside = match old_reg {
            Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        let p = dword(self, offset);
        if let Some(max) = match op.as_str() {
            "max_f32" => Some(true),
            "min_f32" => Some(false),
            _ => None,
        } {
            let at = self.begin_guarded(guard);
            let r = self.lds_float_min_max(p, v0, max);
            let r = self.end_guarded(at, &[(r, outside)])[0];
            if let Some(reg) = dst_reg {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        let at = self.begin_guarded(guard);
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self
                .s
                .m
                .cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    /// `ds_max_f32` (`max`) or `ds_min_f32` on the LDS dword at `p` with
    /// the float bits `data`; the old bits return.
    ///
    /// DXIL has no float atomics on group-shared memory, so this is a
    /// compare-exchange loop around [`Lower::lds_float_pick`], which gives
    /// what RDNA2 writes (measured on gfx1036, reported on the pull
    /// request): an sNaN, data first, then memory, is written quieted;
    /// otherwise a NaN loses to a number (two quiet NaNs give the data);
    /// numbers compare as floats with -0 below +0.
    // shortcut: f32 denormals compare as kept; with the console's MODE
    // flushing them they would order as ±0 first (MODE is not modelled).
    fn lds_float_min_max(&mut self, p: V, data: V, max: bool) -> V {
        let pre = self.cur;
        let init = self.s.m.load(pre, p);
        let (lp, exit) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(pre, Term::Br(lp));
        let i32t = self.s.t.i32;
        let old = self.s.m.phi(lp, i32t);
        self.cur = lp;
        let new = self.lds_float_pick(old, data, max);
        let prev = self.s.m.cmpxchg(lp, p, old, new);
        let done = self.s.m.cmp(lp, pred::EQ, prev, old);
        self.s.m.set_phi(old, vec![(init, pre), (prev, lp)]);
        self.s.m.terminate(lp, Term::CondBr(done, exit, lp));
        self.cur = exit;
        old
    }

    /// What `ds_max_f32` (`max`) or `ds_min_f32` writes over the dword
    /// `mem` with `data` (float bits; see [`Lower::lds_float_min_max`]).
    fn lds_float_pick(&mut self, mem: V, data: V, max: bool) -> V {
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (abs_mask, inf, quiet, zero) = (
            k(self, 0x7FFF_FFFF),
            k(self, 0x7F80_0000),
            k(self, 0x40_0000),
            k(self, 0),
        );
        let nan = |l: &mut Self, x: V| {
            let m = l.s.m.bin(c, Bin::And, x, abs_mask);
            l.s.m.cmp(c, pred::IUGT, m, inf)
        };
        let snan = |l: &mut Self, x: V| {
            let n = nan(l, x);
            let q = l.s.m.bin(c, Bin::And, x, quiet);
            let sig = l.s.m.cmp(c, pred::EQ, q, zero);
            l.s.m.bin(c, Bin::And, n, sig)
        };
        // A total order of the numbers as signed integers, -0 below +0:
        // negative floats' magnitude bits reversed.
        let key = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            let sign = l.s.m.bin(c, Bin::AShr, x, k31);
            let flip = l.s.m.bin(c, Bin::And, sign, abs_mask);
            l.s.m.bin(c, Bin::Xor, x, flip)
        };
        let (dk, mk) = (key(self, data), key(self, mem));
        let take = self
            .s
            .m
            .cmp(c, if max { pred::SGT } else { pred::SLT }, dk, mk);
        let r = self.s.m.select(c, take, data, mem);
        let (dn, mn) = (nan(self, data), nan(self, mem));
        // One NaN: the number; two quiet NaNs: the data.
        let r = self.s.m.select(c, mn, data, r);
        let both = self.s.m.bin(c, Bin::And, dn, mn);
        let only_data = {
            let t = self.s.m.bool(true);
            let not_both = self.s.m.bin(c, Bin::Xor, both, t);
            self.s.m.bin(c, Bin::And, dn, not_both)
        };
        let r = self.s.m.select(c, only_data, mem, r);
        // Signalling NaNs win, quieted: the data's first.
        let qm = self.s.m.bin(c, Bin::Or, mem, quiet);
        let sm = snan(self, mem);
        let r = self.s.m.select(c, sm, qm, r);
        let qd = self.s.m.bin(c, Bin::Or, data, quiet);
        let sd = snan(self, data);
        self.s.m.select(c, sd, qd, r)
    }

    // ---- instructions ----
    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.local_ok = name.starts_with("s_bcnt1_i32_")
            || name.starts_with("s_ff1_i32_")
            || name.starts_with("v_mbcnt_");
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            // Mesh: "do all lanes hold the first lane's value?" (`uniform_test`), which
            // each host wave answers for itself, and a count of a mask's lanes (the
            // fire-light loops, which then run over the host wave).
            MaskData::None => self.uniform_test(inst) || name == "s_bcnt1_i32_b64",
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        if self.split_lds
            && matches!(inst.fields, Fields::Ds { .. })
            && !(name.starts_with("ds_swizzle") || name.contains("permute"))
            && self.lds_meet.as_ref().is_none_or(|(all, _)| all[b])
        {
            // the two host waves of a console wave meet before each LDS access: what the other
            // half wrote is visible, and what it still reads is not overwritten
            let mode = self.s.m.i32(meeting_mode());
            let void = self.s.t.void;
            self.s.op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        }
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                let mut mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
                // Without the wait for the group (bit 0) where not every
                // lane arrives, or where the switch says so.
                let waits = match super::group_sync() {
                    0 => false,
                    1 => !self.fence_only.contains(&b),
                    _ => true,
                };
                if !waits {
                    mode &= !1;
                }
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            // The ray and node test. A failure is a refusal at once: an
            // instruction without a side effect otherwise only stops the
            // program where its results are needed, and two trace programs
            // came out without their test that way.
            Fields::Image { .. } if name.starts_with("image_bvh") => {
                return self
                    .bvh(inst, b, i)
                    .map_err(|why| unsupported(offset, why));
            }
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } => Err("flat memory"),
        };
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    /// The two host waves of a console wave meet (a group barrier with a shared-memory
    /// fence): LDS writes before are seen by reads after.
    fn lds_meeting(&mut self) {
        let mode = self.s.m.i32(meeting_mode());
        let void = self.s.t.void;
        self.s.op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// Whether `v` is a register no program reads as a number: a descriptor dword without
    /// a host value ([`Self::stale`]), a lane mask's dword (outside compute, where a ballot
    /// gives the mask as data), or a join of only such values. Compiled code joins numbers
    /// with numbers: where one of these meets numbers at a join, the register is dead on
    /// that path (a variable set in one arm of a branch, its use behind a flag kept in a
    /// VGPR).
    fn dead(&self, v: Value, seen: &mut Vec<Value>) -> bool {
        if self.stale.contains(&v) {
            return true;
        }
        let st = self.x.st;
        if self.mask_data != MaskData::Console
            && (st.masks.contains_key(&v) || st.highs.contains_key(&v))
        {
            return true;
        }
        let Some(&(b, i)) = self.phi_at.get(&v) else {
            return false;
        };
        if seen.contains(&v) {
            return true;
        }
        seen.push(v);
        self.x.p.blocks[b].phis[i]
            .args
            .iter()
            .all(|&a| self.dead(a, seen))
    }

    /// `s_bcnt1_i32_b*` of the lanes that differ from the first lane's value:
    /// `v_readfirstlane_b32 s, v`, `v_cmp_eq_u32 m, s, v`, `s_xor_b* d, m, exec`. The
    /// compiler tests a value for being the same in the whole wave this way, to take a
    /// scalar path; both paths give each lane the same result, so the lanes of a host
    /// wave may ask it of their own first lane, whatever the wave's size.
    fn uniform_test(&self, inst: &Inst) -> bool {
        let p = self.x.p;
        let writer = |v: Value| -> Option<&'a Inst> {
            let b = *self.def_block.get(&v)?;
            p.blocks[b]
                .insts
                .iter()
                .find(|i| i.writes.iter().any(|w| w.1 == v))
        };
        let scalar = |i: &Inst, op: &Operand| match *op {
            Operand::Scalar { code, .. } if code != NULL && code != EXEC => i.read(Reg::S(code)),
            _ => None,
        };
        if !inst.name().starts_with("s_bcnt1_i32_") {
            return false;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return false;
        };
        let Some(xor) = src.first().and_then(|o| scalar(inst, o)).and_then(writer) else {
            return false;
        };
        let Fields::Sop { src, .. } = &xor.fields else {
            return false;
        };
        if !matches!(xor.name(), "s_xor_b64" | "s_xor_b32")
            || !src
                .iter()
                .any(|o| matches!(o, Operand::Scalar { code: EXEC, .. }))
        {
            return false;
        }
        let Some(cmp) = src.iter().find_map(|o| scalar(xor, o)).and_then(writer) else {
            return false;
        };
        let Fields::Valu(f) = &cmp.fields else {
            return false;
        };
        if !matches!(cmp.name(), "v_cmp_eq_u32" | "v_cmp_eq_i32") || f.sdwa.is_some() || f.dpp.is_some() {
            return false;
        }
        let lanes = f.src.iter().find_map(|o| match *o {
            Operand::Vector { reg, .. } => cmp.read(Reg::V(reg)),
            _ => None,
        });
        let first = f.src.iter().find_map(|o| scalar(cmp, o)).and_then(writer);
        lanes.is_some()
            && first.is_some_and(|fl| {
                fl.name() == "v_readfirstlane_b32"
                    && matches!(&fl.fields, Fields::Valu(g)
                        if matches!(g.src.first(), Some(&Operand::Vector { reg, .. })
                            if fl.read(Reg::V(reg)) == lanes))
            })
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) {
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        if name == "s_getpc_b64" {
            // The address of the next instruction is no number the host has: tables in
            // the code are read by their tracked address. As a number the counter is
            // that instruction's offset in the code, so an address paths join before
            // one load is the byte in the code, whichever table each path took
            // (resource tracking gives such a join as the computed term, with no base).
            let z = self.s.m.i32(0);
            let at = self.s.m.i32(inst.machine.offset as u32 + 4);
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            self.set(lo, Val::I32(at));
            self.set(hi, Val::I32(z));
            return Ok(());
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                // Of a signed operand, a literal sign-extends too.
                Some(&Operand::Literal(x)) if name.ends_with("_i64") => {
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            // SCC is whether bit src1 (mod 32, or mod 64 for the 64-bit form)
            // of src0 is clear (bitcmp0) or set (bitcmp1).
            let (bit, k31) = (src32(self, 1)?, self.s.m.i32(31));
            let idx = self.s.m.bin(c, Bin::And, bit, k31);
            let word = if op.ends_with("_b64") {
                let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                let k32 = self.s.m.i32(32);
                let upper = self.s.m.bin(c, Bin::And, bit, k32);
                let z = self.s.m.i32(0);
                let upper = self.s.m.cmp(c, pred::NE, upper, z);
                self.s.m.select(c, upper, b, a)
            } else {
                src32(self, 0)?
            };
            let shifted = self.s.m.bin(c, Bin::LShr, word, idx);
            let one = self.s.m.i32(1);
            let tested = self.s.m.bin(c, Bin::And, shifted, one);
            let want = self.s.m.i32(u32::from(op.starts_with('1')));
            let r = self.s.m.cmp(c, pred::EQ, tested, want);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_quadmask_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let low = self.quad_mask(a);
                    let high = self.quad_mask(b);
                    let k8 = self.s.m.i32(8);
                    let high = self.s.m.bin(c, Bin::Shl, high, k8);
                    let mask = self.s.m.bin(c, Bin::Or, low, high);
                    let zero = self.s.m.i32(0);
                    self.set(lo, Val::I32(mask));
                    self.set(hi, Val::I32(zero));
                    let nonzero = self.s.m.cmp(c, pred::NE, mask, zero);
                    self.set(scc, Val::Bit(nonzero));
                    Ok(())
                }
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let r = self.s.m.bin(c, Bin::Add, na, nb);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff1_i32_b64" => {
                    // The lowest set bit; -1 for none (as DXIL's first bit low).
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let fb = self.s.m.select(c, none, fb, fb32);
                    let low = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, low, fa, fb);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_mov_b64" => {
                    // Each half by itself: two numbers that travel as a pair (two scalars
                    // parked in neighbouring VGPR lanes, say) need not both have a value.
                    for (h, v) in [(0, lo), (1, hi)] {
                        let val = match half(self, 0, h) {
                            Ok(a) => Val::I32(a),
                            Err(why) => Val::Missing(why),
                        };
                        self.set(v, val);
                    }
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_quadmask_b32" => {
                let pixels = src32(self, 0)?;
                (self.quad_mask(pixels), None)
            }
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = src32(self, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::Add, t, b), None)
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                // SCC: whether the first operand was chosen (RDNA2 ISA).
                (self.s.m.select(c, t, a, b), Some(t))
            }
            "s_abs_i32" => {
                let a = src32(self, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cselect_b32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// Packs whether each consecutive group of four pixels has a set bit.
    fn quad_mask(&mut self, pixels: V) -> V {
        let c = self.cur;
        let one = self.s.m.i32(1);
        let shifted = self.s.m.bin(c, Bin::LShr, pixels, one);
        let mut mask = self.s.m.bin(c, Bin::Or, pixels, shifted);
        // Fold each nibble into its bottom bit, then gather those bits into
        // the low byte in three stages without needing a 64-bit operation.
        for (distance, keep) in [
            (2, 0x1111_1111),
            (3, 0x0303_0303),
            (6, 0x000f_000f),
            (12, 0x0000_00ff),
        ] {
            let distance = self.s.m.i32(distance);
            let shifted = self.s.m.bin(c, Bin::LShr, mask, distance);
            let combined = self.s.m.bin(c, Bin::Or, mask, shifted);
            let keep = self.s.m.i32(keep);
            mask = self.s.m.bin(c, Bin::And, combined, keep);
        }
        mask
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        if self.scalar.get(&site).is_some_and(|a| a.param == MEMORY) {
            return self.memory_smem(inst);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // A view the program picks from the view heap by an index it
        // computes (`ResourceHeap`) has no used-resource entry: a buffer
        // view or a constant buffer view, and the host's heap holds a raw
        // buffer view for either (a constant buffer view of the host's
        // wants 256-byte alignment, the title's are on 16; module notes,
        // "Constant buffers"). Read as raw bytes, as a table's SRV is
        // below. Read as a constant buffer (the host's validation:
        // "Descriptor Type in Heap: SRV, Register Type: CBV", heap index
        // 17) the loads gave nothing: Grand Theft Auto VI's meshlet culling
        // (BatchCullMeshlets, BatchCullPreSkinnedMeshlets,
        // BatchCullBatchCSMeshlets) reads its batch records, mesh headers
        // and instance transforms this way, so it passed no meshlet and
        // every indirect mesh draw ran empty.
        // The host's table of strides by heap slot tells a slot with such
        // a view (not 0) from anything else (a texture, nothing), where
        // the load does not run and reads zeros: the index comes from
        // data, and a texture's descriptor read as a buffer's is a wild
        // address.
        let picked = match kind {
            None => self.slot_stride(src)?,
            Some(_) => None,
        };
        if let Some(stride) = picked {
            let zero = self.s.m.i32(0);
            let is_buffer = self.s.m.cmp(self.cur, pred::NE, stride, zero);
            let at = self.begin_guarded(Some(is_buffer));
            let h = self.handle(site, src, false, (11, 0))?;
            let mut raw = Vec::new();
            for k in 0..count {
                let b = self.cur;
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(b, Bin::Add, addr, off);
                let mask = self.s.m.i8(1);
                let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
                let r = self.s.op(
                    b,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, a, u, mask, al],
                );
                let x = self.s.m.extract(b, r, 0);
                let x = self.as_int(x);
                raw.push((x, zero));
            }
            let raw = self.end_guarded(at, &raw);
            for (k, x) in raw.into_iter().enumerate() {
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // So is constant data in a table: the slot holds a raw view of it
        // (the title's own slot, or the one the host makes for the table),
        // so tables need no copies of their constant buffers per draw.
        let table_constants =
            matches!(kind, Some(u) if u.kind == ResourceType::Cbv && u.root == RootType::Table);
        if table_constants || matches!(kind, Some(u) if u.kind == ResourceType::Srv) {
            let h = self.handle(site, src, false, (11, 0))?;
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let mask = self.s.m.i8(1);
                let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, a, u, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base + 4 * k;
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // How the program uses the descriptor tells the view's kind: an image or buffer
        // instruction before a scalar buffer load, which reads any buffer.
        let x = self.x;
        let uses = |of: Site| {
            x.res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
                .filter_map(move |d| {
                    d.source.leaves().into_iter().find_map(|l| match *l {
                        Source::Loaded { load, first } if load == of => Some((d.site, first)),
                        _ => None,
                    })
                })
        };
        let by_scalar_load = |u: &(Site, u8)| {
            matches!(x.p.blocks[u.0.0].insts[u.0.1].fields, Fields::Smem { .. })
        };
        let use_of = |of: Site| uses(of).find(|u| !by_scalar_load(u)).or_else(|| uses(of).next());
        // The console compiler loads a descriptor again wherever it needs it, and for a
        // size it reads the dwords of a load that nothing uses as a descriptor (a texture
        // picked from the heap by an index held per lane: one load for its width, one for
        // its base level, one for each read of the image). Another load of as many dwords
        // from the same address reads the same descriptor, so what uses that one tells
        // this one's kind. The handle is still made from this load's own operands, which
        // hold here.
        // A waterfall loop reads the index of each load from the first
        // lane of a VGPR: two loads at the first lane of one VGPR's value
        // read one descriptor too (a loop that only asks for the size, and
        // the loop that samples).
        let lane_source = |v: Value| {
            let i = self.def_inst(v)?;
            match i.fields {
                Fields::Valu(ref f) if i.name() == "v_readfirstlane_b32" => match f.src[..] {
                    [Operand::Vector { reg, .. }] => i.read(Reg::V(reg)),
                    _ => None,
                },
                _ => None,
            }
        };
        let same_address = |a: &Address, b: &Address| {
            a.param == b.param
                && a.offset == b.offset
                && (a.dynamic == b.dynamic
                    || a.dynamic.and_then(lane_source).is_some()
                        && a.dynamic.and_then(lane_source) == b.dynamic.and_then(lane_source))
        };
        let address = self.scalar.get(&site).copied();
        let twin = || {
            let a = address?;
            let same = |s: Site| x.p.blocks[s.0].insts[s.1].name() == inst.name();
            x.res
                .scalar
                .iter()
                .filter(|(s, b)| *s != site && same_address(b, &a) && same(*s))
                .find_map(|(s, _)| use_of(*s))
        };
        let use_at = use_of(site).or_else(twin);
        let Some((at, first)) = use_at else {
            // A descriptor kept per lane in VGPRs (`Resources::lane_copied`): the code only
            // copies its dwords and compares them, lane against first lane, and every load
            // that reaches those lanes reads one address. Each dword stands in as its place
            // in the table, so the lanes compare equal as they do on the console.
            if self.x.res.lane_copied.contains(&site)
                && let Some(a) = self.scalar.get(&site).copied()
            {
                for k in 0..count {
                    let x = self.s.m.i32(a.offset / 4 + k);
                    self.write(inst, Reg::S(code + k as u8), x, None);
                }
                return Ok(());
            }
            // A sampler's descriptor: no program reads its dwords as numbers, so where one
            // meets numbers at a join the register is dead on that path (`dead`), as a
            // descriptor's address or format is.
            let sampler = x.res.descriptors.iter().any(|d| {
                d.role == Role::Sampler
                    && d.source.leaves().into_iter().any(
                        |l| matches!(*l, Source::Loaded { load, .. } if load == site),
                    )
            });
            if sampler {
                self.stale.extend(inst.writes.iter().map(|w| w.1));
                return Err("a sampler's descriptor read as data");
            }
            // A buffer view the code only takes the address of, to walk
            // the memory behind it (a ray-tracing structure): the address
            // from the host, the other dwords have no value.
            if count <= 4
                && let Some((lo, hi)) = self.view_address(site)
            {
                for k in 0..count {
                    let reg = Reg::S(code + k as u8);
                    match k {
                        0 => self.write(inst, reg, lo, None),
                        1 => self.write(inst, reg, hi, None),
                        _ => {
                            if let Some(v) = inst.write(reg) {
                                self.vals.insert(v, Val::Missing(OTHER));
                                self.stale.insert(v);
                            }
                        }
                    }
                }
                return Ok(());
            }
            return Err("a descriptor read only as data");
        };
        let src = Source::Loaded { load: site, first };
        let used = self.used_kind(&src);
        let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
        let un = ui.name();
        let uav = un.contains("store")
            || un.contains("atomic")
            || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let (props, image) = match &ui.fields {
            // A ray-tracing structure in a buffer of the host: read raw.
            Fields::Image { .. } if un.starts_with("image_bvh") => ((11, 0), None),
            Fields::Image { dim, .. } => {
                let kind = match dim {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    4 => 6,
                    5 => 7,
                    _ => return Err("an image dimension (cube or MSAA)"),
                };
                ((kind | u, 0x409), Some(kind))
            }
            Fields::Buffer { idxen, format, .. } => {
                let stride = used.map_or(0, |u| u.stride);
                let props = if un.starts_with("buffer_load_format")
                    || un.starts_with("buffer_store_format")
                {
                    (10 | u, 0x409)
                } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4) {
                    (12 | u, stride)
                } else {
                    (11 | u, 0)
                };
                if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                    self.stale.extend(inst.writes.iter().map(|w| w.1));
                    return Err("a constant buffer's descriptor read as data");
                }
                (props, None)
            }
            // Read by scalar buffer loads alone: the raw view `smem` reads it through.
            Fields::Smem { .. } if used.is_some_and(|u| u.kind == ResourceType::Srv) => {
                ((11, 0), None)
            }
            // A constant buffer: nothing of its descriptor is a number.
            Fields::Smem { .. } => {
                self.stale.extend(inst.writes.iter().map(|w| w.1));
                return Err("a constant buffer's descriptor read as data");
            }
            _ => return Err("a scalar memory read as data"),
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some() && !uav {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                let levels = if uav { one } else { field(self, 3) };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => {
                // A raw view counts bytes; the descriptor's record count is in elements.
                let n = field(self, 0);
                let stride = used.map_or(0, |u| u.stride);
                let n = if props.0 & 0xFF == 11 && stride > 1 {
                    let k = self.s.m.i32(stride);
                    self.s.m.bin(c, Bin::UDiv, n, k)
                } else {
                    n
                };
                let mut words = vec![(2, n)];
                // Its address too, where the code computes with it and the
                // host can tell it.
                let computed = (0..2u8).any(|k| {
                    inst.write(Reg::S(code + first + k))
                        .is_some_and(|v| self.computed_with(v, 6))
                });
                if first == 0
                    && computed
                    && let Some((lo, hi)) = self.view_address(site)
                {
                    words.extend([(0, lo), (1, hi)]);
                }
                words
            }
        };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                        self.stale.insert(v);
                    }
                }
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from the program's constant table.
    fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(value) = a.dynamic {
            let mut d = self.i32_of(value)?;
            // A term held per lane reached the address through
            // `v_readfirstlane_b32`: the first lane's.
            let per_lane = self.def_block.get(&value).is_some_and(|&b| {
                let writes = |i: &Inst| i.writes.iter().any(|w| w.1 == value && matches!(w.0, Reg::V(_)));
                self.x.p.blocks[b].insts.iter().any(writes)
            });
            if per_lane {
                self.s.flags |= flags::WAVE_OPS;
                let i32t = self.s.t.i32;
                d = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[d]);
            }
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            // A normalized format's ends are exactly 1.0 and -1.0. The host
            // may divide through a reciprocal and land just below 1.0: RDNA
            // 4 loaded 16_16_UNORM's 65535 as the half 0x3BFF, not 0x3C00.
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let top = (1u64 << w) - 1;
                    let max = self.s.m.f32((top as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let (high, one) = (self.s.m.i32(top as u32), self.s.m.f32(1.0f32.to_bits()));
                    let at_top = self.s.m.cmp(c, pred::EQ, field, high);
                    let q = self.s.m.select(c, at_top, one, q);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let top = (1u32 << (w - 1)) - 1;
                    let max = self.s.m.f32((top as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    // The lowest value is -1.0 too, as is the one above it.
                    let (high, one) = (self.s.m.i32(top), self.s.m.f32(1.0f32.to_bits()));
                    let at_top = self.s.m.cmp(c, pred::SGE, s, high);
                    let q = self.s.m.select(c, at_top, one, q);
                    let low = self.s.m.i32(top.wrapping_neg());
                    let minus_one = self.s.m.f32((-1.0f32).to_bits());
                    let at_bottom = self.s.m.cmp(c, pred::SLE, s, low);
                    let q = self.s.m.select(c, at_bottom, minus_one, q);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// Channels of buffer format `format`, as [`Self::format_load`] gives
    /// them, as d16 data: 16 bits each, two per dword, low half first (an
    /// odd last high half 0). Integer formats keep their low 16 bits; the
    /// others become f16.
    fn d16_channels(&mut self, format: u8, channels: Vec<V>) -> Result<Vec<V>, Why> {
        let (_, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let mut halves = Vec::new();
        for x in channels {
            halves.push(if matches!(numeric, Numeric::Uint | Numeric::Sint) {
                let m = self.s.m.i32(0xFFFF);
                self.s.m.bin(c, Bin::And, x, m)
            } else {
                let (f, i32t) = (self.as_float(x), self.s.t.i32);
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            });
        }
        let mut out = Vec::new();
        for pair in halves.chunks(2) {
            out.push(match *pair {
                [lo, hi] => {
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                }
                [lo] => lo,
                _ => unreachable!("chunks of one or two"),
            });
        }
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let mut x = match address.dynamic {
                    Some(v) => Some(self.i32_of(v)?),
                    None => None,
                };
                if let Some(bias) = self.root_bias(address.param) {
                    x = Some(match x {
                        Some(d) => self.s.m.bin(self.cur, Bin::Add, d, bias),
                        None => bias,
                    });
                }
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// For a root view the host binds as a raw view that may begin before
    /// the title's address (`shader.rs` `USER_VIEW_BIAS`): the bytes to add
    /// to offsets through it. Every root UAV, and the root SRVs of programs
    /// with a spilled user-data table (the others keep the views they
    /// have).
    fn root_bias(&mut self, param: usize) -> Option<V> {
        use super::shader::USER_VIEW_BIAS;
        let rp = *self.x.native.layout.get(param)?;
        if rp.root != RootType::Descriptor {
            return None;
        }
        let kind = self
            .x
            .native
            .used
            .iter()
            .find(|u| u.user_data == rp.user_data)
            .map(|u| u.kind);
        let biased = match kind {
            Some(ResourceType::Uav) => true,
            Some(ResourceType::Srv) => crate::spill::is_spilled(self.x.native),
            _ => false,
        };
        biased.then(|| self.host_word(RootType::Other(USER_VIEW_BIAS), rp.slot, 0))
    }

    /// Whether a descriptor comes from the resource heap itself (root type
    /// 5, not a table) on some path.
    fn is_heap_direct(&self, src: &Source) -> bool {
        src.leaves().iter().any(|l| match l {
            Source::Loaded { load, .. } | Source::Fetched { load, .. } => self
                .scalar
                .get(load)
                .and_then(|a| self.x.native.layout.get(a.param))
                .is_some_and(|p| p.root == RootType::ResourceHeap),
            _ => false,
        })
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } =
            src
        {
            return self.used_kind(args.first()?);
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// A handle for the descriptor `src` read at `site`.
    fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        let props = if self.pops && props.0 & 0x1000 != 0 {
            self.s.flags |= flags::ROVS;
            (props.0 | 0x2000, props.1)
        } else {
            props
        };
        // A descriptor from the resource heap itself: the host cannot tell
        // which slots the program reads, or writes through unordered access.
        if !sampler && self.is_heap_direct(src) {
            self.heap[0] = true;
            self.heap[1] |= props.0 & 0x1000 != 0;
        }
        match src {
            Source::Loaded { load, first } => {
                let key = (self.cur, *load, *first, sampler, props);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                // A waterfall loop's load at the first lane's offset reads
                // each lane's own entry (see `resource::Waterfall`).
                let per_lane = self.x.res.waterfall.loads.contains(load);
                let h = self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                let comparison = props.0 & 0x8000 != 0;
                let slot = self.samplers.iter().enumerate().position(|(n, w)| {
                    *w == words && (self.s.cmp_samplers >> n & 1 != 0) == comparison
                });
                let n = match slot {
                    Some(n) => n,
                    None => {
                        let n = self.samplers.len();
                        if n >= u64::BITS as usize {
                            return Err("more than 64 immediate sampler slots");
                        }
                        self.samplers.push(words);
                        self.s.cmp_samplers |= u64::from(comparison) << n;
                        n
                    }
                };
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_srv_handle(rp.slot))
                    }
                    // A root UAV: a raw buffer whether the access reads
                    // (11) or writes (11 with the UAV bit). Indexed
                    // accesses reach here as raw ones at index times the
                    // entry's stride (`buffer`).
                    (RootType::Descriptor, Some(ResourceType::Uav), 11 | 0x100b) => {
                        Ok(self.s.root_uav_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 0x100b) => {
                        Err("a write through a root SRV")
                    }
                    _ => Err("a typed root view"),
                }
            }
            Source::Phi { block, args }
                if sampler && matches!(args[..], [Source::Constant(_), Source::Constant(_)]) =>
            {
                // One of two immediate samplers, chosen by uniform control
                // flow: what reads it runs with both, the edge taken picks
                // the result (static samplers cannot be indexed).
                let [a, b] = &args[..] else {
                    unreachable!()
                };
                let ha = self.handle(site, a, sampler, props)?;
                let hb = self.handle(site, b, sampler, props)?;
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (j, &q) in self.x.p.blocks[*block].preds.iter().enumerate() {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail = self.tail[q].ok_or("a descriptor chosen by control flow")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    incoming.push((self.s.m.bool(j == 0), at));
                }
                let phi = self.s.m.phi(join, self.s.t.i1);
                self.s.m.set_phi(phi, incoming);
                self.s.sampler_choices.push((ha, hb, phi));
                Ok(ha)
            }
            Source::Phi { block, args } => {
                let _ = site;
                let phi = self.descriptor_phi(*block, args, sampler, 8)?;
                Ok(self.s.heap_handle(self.cur, phi, sampler, false, props))
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                let index = self.s.m.select(self.cur, cond, set, clear);
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let index = self.lane_heap_index(src, sampler)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
        }
    }

    /// The heap index a descriptor phi picks. Each predecessor computes its
    /// heap index at its end and the join picks one with a phi; a
    /// predecessor whose descriptor is itself chosen by control flow (a
    /// phi of an earlier join, which dominates it) gives that join's phi,
    /// at most `depth` joins deep.
    fn descriptor_phi(
        &mut self,
        block: usize,
        args: &[Source],
        sampler: bool,
        depth: u32,
    ) -> Result<V, Why> {
        const WHY: Why = "a descriptor chosen by control flow";
        let p = self.x.p;
        let mut incoming = Vec::new();
        for (&q, arg) in p.blocks[block].preds.iter().zip(args) {
            if self.block[q].is_none() {
                continue;
            }
            let tail = self.tail[q].ok_or(WHY)?;
            let at = self.edges.get(&(q, block)).copied().unwrap_or(tail);
            let index = match arg {
                Source::Loaded { load, first } => {
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.heap_index(*load, *first, sampler);
                    self.cur = here;
                    index?
                }
                Source::Phi { block: inner, args: inner_args } if depth > 0 => {
                    self.descriptor_phi(*inner, inner_args, sampler, depth - 1)?
                }
                _ => return Err(WHY),
            };
            incoming.push((index, at));
        }
        let join = self.block[block].ok_or(WHY)?;
        let phi = self.s.m.phi(join, self.s.t.i32);
        self.s.m.set_phi(phi, incoming);
        Ok(phi)
    }

    /// The bytes per element of the buffer view `src` names, if the program
    /// picks it from the view heap at an index it computes (root type
    /// `ResourceHeap`): read from the host's table by heap slot in the
    /// current block.
    fn slot_stride(&mut self, src: &Source) -> Result<Option<V>, Why> {
        let Source::Loaded { load, first } = src else {
            return Ok(None);
        };
        let heap = self
            .scalar
            .get(load)
            .and_then(|a| self.x.native.layout.get(a.param))
            .is_some_and(|p| p.root == RootType::ResourceHeap);
        if !heap {
            return Ok(None);
        }
        let key = (self.cur, *load, *first);
        if let Some(&v) = self.strides.get(&key) {
            return Ok(Some(v));
        }
        let c = self.cur;
        // The table is by title slot: the index before any heap map.
        let (slot, _) = self.title_heap_index(*load, *first, false)?;
        // Inside the table whatever the index (lanes that do not run
        // compute one too): a root view has no bounds of its own.
        let last = self.s.m.i32(super::shader::SLOT_STRIDES_COUNT - 1);
        let i32t = self.s.t.i32;
        let slot = self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[slot, last]);
        let two = self.s.m.i32(2);
        let byte = self.s.m.bin(c, Bin::Shl, slot, two);
        let h = self.s.root_srv_handle(super::shader::SLOT_STRIDES);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, byte, iu, mask, align],
        );
        let f = self.s.m.extract(c, r, 0);
        let v = self.as_int(f);
        self.strides.insert(key, v);
        Ok(Some(v))
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block (or, chosen by control flow, at the
    /// join).
    fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self
                        .x
                        .st
                        .masks
                        .get(mask)
                        .ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m, self.cur)?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            Source::LanePhi { block, args } => {
                // Each predecessor computes its heap index at its end (what
                // it reads is there: the dwords are, on that edge); the join
                // picks the one of the edge the lane came by. An edge no
                // lane reading it takes gives index 0.
                let join =
                    self.block[*block].ok_or("a per-lane descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail =
                        self.tail[q].ok_or("a per-lane descriptor joined from a later block")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(arg) => {
                            let here = std::mem::replace(&mut self.cur, at);
                            let index = self.lane_heap_index(arg, sampler);
                            self.cur = here;
                            index?
                        }
                        None => self.s.m.i32(0),
                    };
                    incoming.push((index, at));
                }
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        let root = match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(r @ (RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap)) => r,
            _ => return Err("a descriptor outside the heaps"),
        };
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        let index = self.s.m.bin(c, Bin::LShr, at, shift);
        Ok(self.host_index(index, root, sampler))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block, as the host
    /// heap has it ([`Lower::host_index`]).
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let (index, root) = self.title_heap_index(load, first, sampler)?;
        Ok(self.host_index(index, root, sampler))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block, as the title's
    /// address gives it; and the root type of what it is read through.
    fn title_heap_index(
        &mut self,
        load: Site,
        first: u8,
        sampler: bool,
    ) -> Result<(V, RootType), Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        let root = match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(r @ (RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap)) => r,
            _ => return Err("a descriptor outside the heaps"),
        };
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok((self.s.m.bin(c, Bin::LShr, addr, shift), root))
    }

    /// Heap index `index` of a descriptor read through a root parameter of
    /// type `root`, as the host heap has it: a view taken from the resource
    /// heap itself is at its title index, which the host may map
    /// ([`Lower::heap_remap`]); a table's index is the host's already (the
    /// host makes a table that reaches past the map's identity window
    /// afresh in slots of its own), and so is a sampler's.
    fn host_index(&mut self, index: V, root: RootType, sampler: bool) -> V {
        match super::shader::view_heap_map() {
            Some((window, slot)) if root == RootType::ResourceHeap && !sampler => {
                self.heap_remap(index, window, slot)
            }
            _ => index,
        }
    }

    /// Title heap index `index` of a view as the host's, under a heap map
    /// ([`super::shader::set_view_heap_map`]): below the identity window
    /// `window`, itself; past it, the slot the heap map gives. A host view
    /// heap capped below the title's slots (NVIDIA takes 1,000,000 views,
    /// Grand Theft Auto VI's heap has 1,245,184) mirrors title slots below
    /// the window at their own index and gives each one past it that holds
    /// a view a slot of its own; the heap map, a raw buffer at view heap
    /// slot `slot`, holds per title slot from the window on that slot's
    /// bitwise not. An entry never written (or a read past the map, which
    /// gives 0) is `u32::MAX`, which [`super::shader::Shader::heap_handle`]
    /// then clamps to the heap's null slot.
    ///
    /// Branchless (a load and selects): [`Lower::heap_index`] may run in a
    /// predecessor's tail for a phi. The map's handle is uniform, made once
    /// in the entry block; only the offset is per lane.
    fn heap_remap(&mut self, index: V, window: u32, slot: u32) -> V {
        let map = match self.heap_map {
            Some(map) => map,
            None => {
                let entry = self.s.entry;
                let at = self.s.m.i32(slot);
                let map = self.s.heap_handle(entry, at, false, false, (11, 0));
                self.heap_map = Some(map);
                map
            }
        };
        let c = self.cur;
        let w = self.s.m.i32(window);
        // The entry's byte offset; a title index so far past the map that
        // it would wrap onto it reads past the end instead.
        let d = self.s.m.bin(c, Bin::Sub, index, w);
        let limit = self.s.m.i32(1 << 30);
        let near = self.s.m.cmp(c, pred::IULT, d, limit);
        let two = self.s.m.i32(2);
        let byte = self.s.m.bin(c, Bin::Shl, d, two);
        let past = self.s.m.i32(!3u32);
        let byte = self.s.m.select(c, near, byte, past);
        let (iu, mask, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(1), self.s.m.i32(4));
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[map, byte, iu, mask, align],
        );
        let x = self.s.m.extract(c, r, 0);
        let entry = self.as_int(x);
        let ones = self.s.m.i32(u32::MAX);
        let mapped = self.s.m.bin(c, Bin::Xor, entry, ones);
        let inside = self.s.m.cmp(c, pred::IULT, index, w);
        self.s.m.select(c, inside, index, mapped)
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(interp) = self.interps.get(&site).cloned() else {
            return Err("the first half of an interpolation");
        };
        let name = inst.name();
        let (dst, src) = match &inst.fields {
            Fields::Vintrp { dst, src, .. } => (*dst, Some(*src)),
            Fields::Valu(v) => {
                if v.abs != 0 || v.neg != 0 {
                    return Err("interpolation operand modifiers");
                }
                (v.dst.ok_or("an interpolation")?, v.src.get(1).copied())
            }
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A float operand of the instruction.
        let float = |l: &mut Self, op: Option<Operand>| -> Result<V, Why> {
            let x = l.operand(inst, &op.ok_or("an interpolation operand")?)?;
            Ok(l.as_float(x))
        };
        // What the result adds to the evaluated channel (a second half's
        // first half, when the halves evaluate on their own).
        let mut add = None;
        let (attribute, eval) = match interp {
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => {
                let mode = interp_mode(barycentric).ok_or("pull-model interpolation")?;
                let eval = if self.per_vertex(attribute.row) {
                    // The element is read per vertex elsewhere, so it is
                    // constant: weigh its vertices with SV_Barycentrics.
                    let mut ij = [None; 2];
                    for (index, w) in ij.iter_mut().enumerate() {
                        let x = match self.barycentric(barycentric, index as u8) {
                            Val::I32(x) => x,
                            Val::Missing(w) => return Err(w),
                            Val::Bit(_) => unreachable!(),
                        };
                        *w = Some(self.as_float(x));
                    }
                    Eval::Weights {
                        p0: true,
                        i: ij[0],
                        j: ij[1],
                    }
                } else {
                    Eval::Mode(mode)
                };
                (attribute, eval)
            }
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, Eval::Mode(1)),
            // P10 and P20: what the first and second halves weigh, P1 - P0
            // and P2 - P0, which is what the hardware holds for an
            // attribute the program also interpolates.
            Interp::Vertex { attribute, vertex } if self.interpolated(attribute.row) => {
                let one = Some(self.s.m.f32(1f32.to_bits()));
                let (i, j) = if vertex == 0 {
                    (one, None)
                } else {
                    (None, one)
                };
                (attribute, Eval::Weights { p0: false, i, j })
            }
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
            // A half of an interpolation with computed i or j: P0 + i P10,
            // or the old dst (f16: the third operand) + j P20.
            Interp::Weighted { attribute } if name.starts_with("v_interp_p1") => {
                let i = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: true,
                        i,
                        j: None,
                    },
                )
            }
            Interp::Weighted { attribute } => {
                let old = match &inst.fields {
                    Fields::Valu(v) => v.src.get(2).copied(),
                    _ => Some(dst),
                };
                add = Some(float(self, old)?);
                let j = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: false,
                        i: None,
                        j,
                    },
                )
            }
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| {
                    let packs = if i.pack2 {
                        input_packs(&self.x.native.io, &self.s.inputs)
                    } else {
                        Vec::new()
                    };
                    (i.pack2, packs)
                }),
            _ => None,
        };
        let f16 = name.ends_with("_f16");
        // The evaluated channel plus `add`, as a float.
        let total = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let x = l.eval_channel(row, ch, eval)?;
            let x = l.as_float(x);
            Ok(match add {
                Some(a) => l.s.m.bin(l.cur, Bin::Add, a, x),
                None => x,
            })
        };
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => {
                let x = total(self, attribute.row, attribute.channel)?;
                self.as_int(x)
            }
            Some((pack2, packs)) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, &packs, attribute.row, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    match src {
                        Some((row, ch)) => total(l, row, ch).map(Some),
                        None => Ok(None),
                    }
                };
                let to_f16 = |l: &mut Self, f: V| l.f16_result(f);
                let h = 2 * attribute.channel + u8::from(attribute.high);
                let c = self.cur;
                if name == "v_interp_p1ll_f16" {
                    // An f32 result from 16-bit vertex values.
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    self.as_int(f)
                } else if f16 {
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    if matches!(inst.fields, Fields::Valu(_)) {
                        // omod and clamp are part of the f16 instruction;
                        // op_sel[3] picks the destination half.
                        let f = self.half_modifiers(inst, f);
                        let lo = to_f16(self, f);
                        return self.write_half(inst, lo, None);
                    }
                    let lo = to_f16(self, f);
                    self.low_half(inst, reg, lo)
                } else if matches!(eval, Eval::Mode(1)) {
                    // The whole dword of a flat input: both halves.
                    let h = 2 * attribute.channel;
                    let z = self.s.m.i32(0);
                    let lo = match half(self, h)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let hi = match half(self, h + 1)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// Whether the program reads attribute row `row` per vertex (an
    /// interpolation with computed barycentrics, or P10 or P20): then its
    /// element is declared constant and read with `attributeAtVertex`.
    fn per_vertex(&self, row: u8) -> bool {
        self.interps.values().any(|i| match *i {
            Interp::Weighted { attribute } => attribute.row == row,
            Interp::Vertex { attribute, vertex } => {
                attribute.row == row && vertex != 2 && self.interpolated(row)
            }
            Interp::Interpolated { .. } => false,
        })
    }

    /// Whether the program interpolates attribute row `row` (so the
    /// hardware holds P1 - P0 and P2 - P0 for it).
    fn interpolated(&self, row: u8) -> bool {
        self.interps.values().any(|i| match *i {
            Interp::Interpolated { attribute, .. } | Interp::Weighted { attribute } => {
                attribute.row == row
            }
            Interp::Vertex { .. } => false,
        })
    }

    /// Channel `ch` of input row `row` evaluated as `eval` says, as raw bits.
    fn eval_channel(&mut self, row: u8, ch: u8, eval: Eval) -> Result<V, Why> {
        let (id, col) = self
            .find_input(row, ch)
            .ok_or("an input the signature lacks")?;
        let mode = match eval {
            Eval::Mode(mode) => mode,
            Eval::Weights { .. } => 1,
        };
        let e = &mut self.s.inputs[id];
        if e.interp != mode && e.usage != 0 {
            return Err("one input interpolated two ways");
        }
        e.interp = mode;
        let Eval::Weights { p0, i, j } = eval else {
            return match self.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            };
        };
        let v0 = self.vertex_input(id, col, 0);
        let c = self.cur;
        let mut acc = p0.then_some(v0);
        for (k, w) in [(1, i), (2, j)] {
            let Some(w) = w else { continue };
            let vk = self.vertex_input(id, col, k);
            let d = self.s.m.bin(c, Bin::Sub, vk, v0);
            acc = Some(match acc {
                Some(a) => self.fmad(w, d, a),
                None => self.s.m.bin(c, Bin::Mul, w, d),
            });
        }
        let acc = acc.ok_or("an interpolation")?;
        Ok(self.as_int(acc))
    }

    /// Vertex `k` (0 the provoking vertex, as `barycentric` orders them)
    /// of a constant input element's channel, as a float.
    fn vertex_input(&mut self, id: usize, col: u8, k: u8) -> V {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let int = e.comp != 9;
        let (ty, suffix) = if int {
            (self.s.t.i32, ".i32")
        } else {
            (self.s.t.f32, ".f32")
        };
        self.s.flags |= flags::BARYCENTRICS;
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.i8(k),
        ];
        let v = self.s.op(
            entry,
            &format!("attributeAtVertex{suffix}"),
            Attr::ReadNone,
            ty,
            137,
            &args,
        );
        if int {
            self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.f32)
        } else {
            v
        }
    }

    /// An interpolation whose i or j register the compiler reused on
    /// another path (`Interp::Interpolated::unless`) interpolates with the
    /// barycentric found only if no interpolating lane can have been there
    /// ([`Self::avoids`]). Otherwise each of its halves evaluates on its
    /// own, from what the register holds and the attribute's vertex values,
    /// as an interpolation with computed barycentrics does
    /// (`Interp::Weighted`).
    fn settle_interps(&mut self) {
        let mut unproven = Vec::new();
        for (&site, interp) in &self.interps {
            if let Interp::Interpolated {
                attribute,
                first,
                unless,
                ..
            } = interp
                && !unless.is_empty()
                && !self.avoids(site, unless)
            {
                unproven.push((site, *attribute, first.clone()));
            }
        }
        for (site, attribute, first) in unproven {
            for half in first {
                self.interps
                    .entry(half)
                    .or_insert(Interp::Weighted { attribute });
            }
            self.interps.insert(site, Interp::Weighted { attribute });
        }
    }

    /// Whether no lane that runs the instruction at `site` has been where
    /// `unless` says: a lane that came along such an edge or ran (skipped)
    /// such an instruction meets its condition, which the instruction's must
    /// contradict. Those lane bits must not change in between, so none of
    /// those blocks is on a cycle. A lane mask joined where such an edge
    /// enters holds, for the lanes that came along it, the edge's mask.
    fn avoids(&self, (b, i): Site, unless: &[Elsewhere]) -> bool {
        let st = self.x.st;
        let runs = [
            (st.reach[b], true),
            (st.guards[b][i].unwrap_or(Bool::TRUE), true),
        ];
        let mut atoms = st.bools.support(runs[0].0);
        atoms.extend(st.bools.support(runs[1].0));
        unless.iter().all(|&u| {
            let (mut all, at, mut links) = match u {
                Elsewhere::Edge(q, to) => {
                    let all = self.lane_edge(q, to);
                    let block = &self.x.p.blocks[to];
                    let n = block.preds.iter().position(|&p| p == q).expect("an edge");
                    let links: Vec<(Bool, Bool)> = block
                        .phis
                        .iter()
                        .filter(|ph| atoms.contains(&Atom::Phi(ph.value)))
                        .filter_map(|ph| {
                            Some((*st.masks.get(&ph.value)?, *st.masks.get(&ph.args[n])?))
                        })
                        .collect();
                    (all, to, links)
                }
                Elsewhere::Write((w, k)) | Elsewhere::Skip((w, k)) => {
                    let guard = st.guards[w][k].unwrap_or(Bool::TRUE);
                    let ran = matches!(u, Elsewhere::Write(_));
                    (vec![(st.reach[w], true), (guard, ran)], w, vec![])
                }
            };
            all.extend(runs);
            // Each joined mask equals the incoming one: try both values.
            links.truncate(6);
            let some_lane = (0..1u32 << links.len()).any(|bits| {
                let mut all = all.clone();
                for (k, &(joined, incoming)) in links.iter().enumerate() {
                    let v = bits >> k & 1 != 0;
                    all.extend([(joined, v), (incoming, v)]);
                }
                st.bools.meet(&all)
            });
            !self.on_cycle(at) && !some_lane
        })
    }

    /// What holds in a lane that goes from IR block `q` to `b`.
    fn lane_edge(&self, q: usize, b: usize) -> Vec<(Bool, bool)> {
        let st = self.x.st;
        let mut all = vec![(st.reach[q], true)];
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = st.exits[q]
            && taken != not_taken
        {
            all.push((when, taken == b));
        }
        all
    }

    /// Whether no lane goes from IR block `q` to `b`.
    fn never_taken(&self, q: usize, b: usize) -> bool {
        !self.x.st.bools.meet(&self.lane_edge(q, b))
    }

    /// Whether IR block `b` can reach itself.
    fn on_cycle(&self, b: usize) -> bool {
        let blocks = &self.x.p.blocks;
        let mut seen = vec![false; blocks.len()];
        let mut stack = blocks[b].exit.successors();
        while let Some(x) = stack.pop() {
            if x == b {
                return true;
            }
            if !std::mem::replace(&mut seen[x], true) {
                stack.extend(blocks[x].exit.successors());
            }
        }
        false
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    /// VOP3 f16 output modifiers of the f32 result `f`: omod, then clamp,
    /// applied before the half conversion so it rounds once (the order
    /// `write_f` uses).
    fn half_modifiers(&mut self, inst: &Inst, f: V) -> V {
        let Fields::Valu(v) = &inst.fields else {
            return f;
        };
        let (omod, clamp) = (v.omod, v.clamp);
        let mut f = f;
        if omod != 0 {
            let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(omod)];
            let k = self.s.m.f32(factor.to_bits());
            f = self.s.m.bin(self.cur, Bin::Mul, f, k);
        }
        if clamp {
            f = self.unary(7, f);
        }
        f
    }

    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            // Console cache coherence: gl0 writes back and invalidates the
            // per-CU shader L0, gl1 invalidates GL1, so one wavefront's
            // writes become visible to others. D3D12 already guarantees UAV
            // coherence across wavefronts and dispatches, DXIL has no
            // equivalent instruction, and this project's host-visible
            // writeback comes from command-stream GpuWrite ranges rather
            // than shader cache hints. Ordering is kept by the surrounding
            // s_waitcnt/s_barrier, which the translator does lower. The
            // host has no console buffer cache to invalidate.
            return Ok(());
        }
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // Byte and 16-bit accesses: bits, and whether a load sign-extends.
        let narrow = match name {
            "buffer_load_ubyte" | "buffer_store_byte" => Some((8, false)),
            "buffer_load_sbyte" => Some((8, true)),
            "buffer_load_ushort" | "buffer_store_short" => Some((16, false)),
            "buffer_load_sshort" => Some((16, true)),
            _ => None,
        };
        // d16: two 16-bit channels per VGPR, low half first (buffer and
        // tbuffer formats alike).
        let d16 = name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() || narrow.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        // A buffer the program picks from the view heap by an index it
        // computes: no used-resource entry gives its stride, the host's
        // table by heap slot does; the access becomes a raw one at the
        // element's byte. (Also tells the host this shader reads such
        // buffers.)
        let mut index = index;
        if let Some(stride) = self.slot_stride(src)?
            && let Some(element) = index
            && !typed
            && used.is_none()
        {
            let at = self.s.m.bin(c, Bin::Mul, element, stride);
            off = self.s.m.bin(c, Bin::Add, at, off);
            index = None;
        }
        // A structured buffer by its used-resource entry's stride: raw
        // bytes too. The host makes its buffer views raw (a raw read
        // through a structured view gives zeros on NVIDIA), and one kind
        // of view can then serve every shader that reads the buffer.
        if let (Some(element), Some(u)) = (index, used)
            && !typed
            && u.kind != ResourceType::Cbv
            && u.stride >= 4
            && u.stride.is_multiple_of(4)
        {
            let k = self.s.m.i32(u.stride);
            let at = self.s.m.bin(c, Bin::Mul, element, k);
            off = self.s.m.bin(c, Bin::Add, at, off);
            index = None;
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let table = used.is_some_and(|u| u.root == RootType::Table);
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            if table {
                // A table's constant buffer is a raw view (as the scalar
                // loads read it, `smem`).
                let h = self.handle(site, src, false, (11, 0))?;
                let (iu, mask, align) = (
                    self.s.m.undef(self.s.t.i32),
                    self.s.m.i8(1),
                    self.s.m.i32(4),
                );
                for k in 0..count {
                    let four = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, byte, four);
                    let r = self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, a, iu, mask, align],
                    );
                    let x = self.s.m.extract(c, r, 0);
                    let x = self.as_int(x);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if let Some((bits, signed)) = narrow {
            // Below a dword: raw buffers are read and written in dwords on
            // 4 bytes, so the dword that holds the value is read and
            // shifted, or changed with two atomic operations that touch
            // only the value's own bits (other lanes may write the rest of
            // the dword at the same time). A 16-bit value at a dword's
            // last byte continues in the next dword.
            let byte = match index {
                None => off,
                Some(index) if used.is_some_and(|u| u.stride == 1) => {
                    self.s.m.bin(c, Bin::Add, index, off)
                }
                Some(_) => return Err("a byte or 16-bit buffer access by index"),
            };
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let k3 = self.s.m.i32(3);
            let within = self.s.m.bin(c, Bin::And, byte, k3);
            let aligned = self.s.m.bin(c, Bin::Sub, byte, within);
            let shift = self.s.m.bin(c, Bin::Shl, within, k3);
            let four = self.s.m.i32(4);
            let next = self.s.m.bin(c, Bin::Add, aligned, four);
            let ones = self.s.m.i32(if bits == 8 { 0xff } else { 0xffff });
            let zero = self.s.m.i32(0);
            let k8 = self.s.m.i32(8);
            let straddles = self.s.m.cmp(c, pred::EQ, within, k3);
            if !store {
                let mask = self.s.m.i8(1);
                let align = self.s.m.i32(4);
                let dword = |l: &mut Self, at: V| {
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, at, iu, mask, align],
                    );
                    let f = l.s.m.extract(c, r, 0);
                    l.as_int(f)
                };
                let low = dword(self, aligned);
                let mut x = self.s.m.bin(c, Bin::LShr, low, shift);
                if bits == 16 {
                    let high = dword(self, next);
                    let high = self.s.m.bin(c, Bin::Shl, high, k8);
                    let high = self.s.m.select(c, straddles, high, zero);
                    x = self.s.m.bin(c, Bin::Or, x, high);
                }
                x = self.s.m.bin(c, Bin::And, x, ones);
                if signed {
                    let up = self.s.m.i32(32 - bits);
                    x = self.s.m.bin(c, Bin::Shl, x, up);
                    x = self.s.m.bin(c, Bin::AShr, x, up);
                }
                let Some(Operand::Vector { reg, .. }) = *data else {
                    return Err("a buffer result");
                };
                self.write(inst, Reg::V(reg), x, guard);
                return Ok(());
            }
            let d = data.as_ref().ok_or("buffer store data")?;
            let v = self.vgprs(inst, d, 1)?[0];
            let v = self.s.m.bin(c, Bin::And, v, ones);
            let all = self.s.m.i32(u32::MAX);
            let clear = self.s.m.bin(c, Bin::Shl, ones, shift);
            let keep = self.s.m.bin(c, Bin::Xor, clear, all);
            let set = self.s.m.bin(c, Bin::Shl, v, shift);
            let keep_next = self.s.m.i32(0xffff_ff00);
            let set_next = self.s.m.bin(c, Bin::LShr, v, k8);
            let (and, or) = (self.s.m.i32(1), self.s.m.i32(2));
            let i32t = self.s.t.i32;
            let at = self.begin_guarded(guard);
            for (op, x) in [(and, keep), (or, set)] {
                let args = [h, op, aligned, iu, iu, x];
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args);
            }
            if bits == 16 {
                let second = self.begin_guarded(Some(straddles));
                for (op, x) in [(and, keep_next), (or, set_next)] {
                    let args = [h, op, next, iu, iu, x];
                    self.s
                        .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args);
                }
                self.end_guarded(second, &[]);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let (h, coords) = match index {
                // A raw buffer (stride 1) read by index: the index is the byte.
                Some(index) if used.is_some_and(|u| u.stride == 1) => {
                    let byte = self.s.m.bin(c, Bin::Add, index, off);
                    (self.handle(site, src, false, (11 | u, 0))?, [byte, iu, iu])
                }
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride < 4 || !stride.is_multiple_of(4) {
                        return Err("a structured buffer of unknown stride");
                    }
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    (h, [index, off, iu])
                }
                None => (self.handle(site, src, false, (11 | u, 0))?, [off, iu, iu]),
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let r = if op == u32::MAX {
                // cmpswap: data is (new, compare).
                let mut args = vec![h];
                args.extend(coords);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                let mut args = vec![h, self.s.m.i32(op)];
                args.extend(coords);
                args.push(vals[0]);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 || *offen || *offset != 0 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                (69, "bufferStore.f32")
            } else {
                match index {
                    // A raw buffer (stride 1) written by index: the index is the byte.
                    Some(index) if used.is_some_and(|u| u.stride == 1) => {
                        let byte = self.s.m.bin(c, Bin::Add, index, off);
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([byte, iu]);
                    }
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride < 4 || !stride.is_multiple_of(4) {
                            return Err("a structured buffer of unknown stride");
                        }
                        args.push(self.handle(site, src, false, (12 | u, stride))?);
                        args.extend([index, off]);
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let mut xs = self.format_load(h, off, f, count)?;
            if d16 {
                xs = self.d16_channels(f, xs)?;
            }
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                // A raw buffer (stride 1) read by index: the index is the byte.
                Some(index) if used.is_some_and(|u| u.stride == 1) => {
                    let byte = self.s.m.bin(c, Bin::Add, index, off);
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, byte, iu, mask, align],
                    )
                }
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride < 4 || !stride.is_multiple_of(4) {
                        return Err("a structured buffer of unknown stride");
                    }
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, index, off, mask, align],
                    )
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = self
                        .s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        if e.compressed {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let c = self.cur;
        // A pixel export whose lanes outside its guard the `done` export
        // after it discards: what those lanes write is lost, so it stores in
        // every lane.
        if let Some(g) = guard
            && !mesh
            && !(self.s.pixel && !e.done && self.discarded_at_done(b, i))
        {
            if !(e.done && self.s.pixel) {
                return Err("a conditional export");
            }
            let t = self.s.m.bool(true);
            let kill = self.s.m.bin(c, Bin::Xor, g, t);
            self.s
                .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
        }
        if mesh && e.target == Target::Primitive {
            return self.emit_indices(e, guard);
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, (pack2, _))) = halves {
            let packs: Vec<(u8, Option<u8>)> = self
                .half_outputs
                .iter()
                .filter(|(_, h)| h.0)
                .map(|(&r, h)| (r, h.1))
                .collect();
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, &packs, row, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                stores.push((id, col, value, None));
            }
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports.
        let at = if mesh {
            let flat = self.mesh_thread();
            let limit = if let Some(m) = self.s.mesh {
                if primitive {
                    m.max_primitives
                } else {
                    m.max_vertices
                }
            } else {
                0
            };
            let threads = self.s.threads.map(|t| t[0]).unwrap_or(u32::MAX);
            let mesh_guard = if limit > 0 && threads > limit {
                let lim = self.s.m.i32(limit);
                let in_bounds = self.s.m.cmp(self.cur, pred::IULT, flat, lim);
                match guard {
                    Some(g) => Some(self.s.m.bin(self.cur, Bin::And, g, in_bounds)),
                    None => Some(in_bounds),
                }
            } else {
                guard
            };
            self.begin_guarded(mesh_guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        for (id, col, value, half) in stores {
            let mut x = self.i32_of(value)?;
            if let Some(high) = half {
                // An f16 half, declared a 32-bit float on the host.
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            let el = &mut elements(self, primitive)[id];
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Each corner takes ten
    /// bits: a 9-bit vertex index and, above it, its edge flag (bits 9, 19
    /// and 29), which only the console's edge-flag rasterisation reads and
    /// which is no part of the index. Bit 31 culls it (all indices 0: no
    /// area).
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let flat = self.mesh_thread();
        let guard = if let Some(mesh) = self.s.mesh {
            let threads = self.s.threads.map(|t| t[0]).unwrap_or(u32::MAX);
            if threads > mesh.max_primitives {
                let p = self.s.m.i32(mesh.max_primitives);
                let in_bounds = self.s.m.cmp(self.cur, pred::IULT, flat, p);
                match guard {
                    Some(g) => Some(self.s.m.bin(self.cur, Bin::And, g, in_bounds)),
                    None => Some(in_bounds),
                }
            } else {
                guard
            }
        } else {
            guard
        };
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let mut culled = self.s.m.cmp(c, pred::SLT, x, z);
        // A primitive that names a vertex past the declared count, or one
        // no lane writes (lane n writes vertex n alone), is culled too: the
        // host would read a vertex nothing set, at any position.
        let written = self.s.mesh.map(|m| {
            m.max_vertices
                .min(self.s.threads.map_or(u32::MAX, |t| t[0]))
        });
        let mut indices = [z; 3];
        for (k, slot) in indices.iter_mut().enumerate() {
            let (shift, mask) = (self.s.m.i32(10 * k as u32), self.s.m.i32(0x1FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            if let Some(n) = written {
                let n = self.s.m.i32(n);
                let past = self.s.m.cmp(c, pred::IUGE, i, n);
                culled = self.s.m.bin(c, Bin::Or, culled, past);
            }
            *slot = i;
        }
        let mut args = vec![flat];
        for i in indices {
            args.push(self.s.m.select(c, culled, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    _ if self.x.st.wide.contains(&b) => {
                        let c = self.wave_wide(b, when).map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                // A back edge is taken only below the iteration limit.
                let back =
                    |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let cond = if back(self, taken) {
                    let ok = self.count_iteration(taken, pred::IULT);
                    self.s.m.bin(self.cur, Bin::And, cond, ok)
                } else if back(self, not_taken) {
                    let stop = self.count_iteration(not_taken, pred::IUGE);
                    self.s.m.bin(self.cur, Bin::Or, cond, stop)
                } else {
                    cond
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
                for (to, via) in [(taken, t), (not_taken, f)] {
                    if !back(self, to) {
                        // From an edge block, or from here.
                        let from = if Some(via) == self.block[to] {
                            self.cur
                        } else {
                            via
                        };
                        self.enter_loop(b, to, from);
                    }
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        // Per stage unless the switch names one: a loop that never meets its
        // exit (a count read from a buffer the host has not got yet, say)
        // runs to the limit in every lane. A million iterations a pixel
        // hung the GPU for seconds a draw until Windows reset it
        // (PS_Textured_Eight, 2026-11-24, run 12).
        // shortcut: a pixel loop past 128 iterations, a vertex or mesh loop
        // past 256 or a compute loop past 4,096 is cut short. (They were
        // 1,024, 4,096 and 65,536: the opening cutscene then lost the card
        // 14-25 frames in, in a minimap or shadow mesh draw; with every
        // loop cut at 64 it ran 90 frames. Raise them again once shaders
        // stop reading data that is not there yet.)
        let limit = match super::loop_limit() {
            LOOP_LIMIT if self.s.pixel => 1 << 7,
            LOOP_LIMIT if self.s.mesh.is_some() || self.s.threads.is_none() => 1 << 8,
            LOOP_LIMIT => 1 << 12,
            named => named,
        };
        let limit = self.s.m.i32(limit);
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, c));
        self.s.m.cmp(c, p, next, limit)
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    /// Whether no lane entering IR block `b` from its `n`th predecessor
    /// reads what `reg` holds there before something overwrites it. The
    /// lane is followed along its per-lane branches, with the phis it
    /// passes taking that path's values; a branch whose compares have
    /// constant operands on the path is followed one way. Scalar arithmetic
    /// on the value passes it on; any other read of it, or a branch or guard
    /// that depends on it, makes it live. Paths meeting at a block with the
    /// same registers holding it go on with what they agree on. This is the
    /// console compiler's flag after a join (`if (c) { x = load; f = .. }
    /// if (f) use(x)`, with `f` 0 on the other path), where the register
    /// that holds `x` holds a stale value, often a lane mask, on that path.
    fn dead_on_edge(&self, b: usize, n: usize, reg: Reg) -> bool {
        let p = self.x.p;
        let st = self.x.st;
        // Values computed from the stale one on some path followed.
        let mut tainted = HashSet::new();
        let mut work = vec![(b, n, HashMap::new(), BTreeSet::from([reg]))];
        // Per block reached and registers holding tainted values on entry:
        // the phi values known there, over the paths so far.
        let mut at: HashMap<(usize, BTreeSet<Reg>), HashMap<Value, u32>> = HashMap::new();
        let mut visits = 0;
        while let Some((b, n, mut env, mut regs)) = work.pop() {
            let block = &p.blocks[b];
            let known: Vec<_> = block
                .phis
                .iter()
                .map(|phi| (phi.value, self.konst(&env, phi.args[n])))
                .collect();
            for (v, k) in known {
                match k {
                    Some(k) => env.insert(v, k),
                    None => env.remove(&v),
                };
            }
            for phi in &block.phis {
                if regs.contains(&phi.reg) {
                    tainted.insert(phi.value);
                }
            }
            let key = (b, regs.clone());
            if let Some(old) = at.get(&key) {
                let agreed: HashMap<Value, u32> = old
                    .iter()
                    .filter(|&(v, k)| env.get(v) == Some(k))
                    .map(|(&v, &k)| (v, k))
                    .collect();
                if agreed.len() == old.len() {
                    continue;
                }
                env = agreed;
            }
            at.insert(key, env.clone());
            visits += 1;
            if visits > 4 * p.blocks.len() + 64 {
                return false;
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if regs.is_empty() {
                    break;
                }
                let guard = st.guards[b][i];
                if guard.is_some_and(|g| self.touches(g, &env, &tainted)) {
                    return false;
                }
                // The closing branch is the lane's exit, tested below.
                if inst.reads.iter().any(|r| regs.contains(&r.0))
                    && !inst.name().starts_with("s_cbranch")
                {
                    // Scalar arithmetic only passes the value on.
                    let pure = matches!(inst.fields, Fields::Sop { .. })
                        && inst.write(Reg::S(EXEC)).is_none()
                        && !inst.name().starts_with("s_setreg");
                    if !pure {
                        return false;
                    }
                    for &(r, w) in &inst.writes {
                        regs.insert(r);
                        tainted.insert(w);
                    }
                } else if guard.is_none() {
                    for w in &inst.writes {
                        regs.remove(&w.0);
                    }
                }
            }
            if regs.is_empty() {
                continue;
            }
            let into = |t: usize| p.blocks[t].preds.iter().position(|&q| q == b);
            let mut go = |t: usize, env, regs| match into(t) {
                Some(n) => {
                    work.push((t, n, env, regs));
                    true
                }
                None => false,
            };
            let ok = match st.exits[b] {
                LaneExit::End => true,
                LaneExit::Indirect => false,
                LaneExit::Jump(t) => go(t, env, regs),
                LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } => {
                    let (no, yes) = self.can_be(when, &env, &mut HashMap::new());
                    !(no && yes && self.touches(when, &env, &tainted))
                        && (!yes || go(taken, env.clone(), regs.clone()))
                        && (!no || go(not_taken, env, regs))
                }
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// Whether deciding `b`, given the values in `env`, may test an atom
    /// of a `tainted` value.
    fn touches(&self, b: Bool, env: &HashMap<Value, u32>, tainted: &HashSet<Value>) -> bool {
        let mut stack = vec![b];
        let mut seen = HashSet::new();
        while let Some(b) = stack.pop() {
            let Some((atom, lo, hi)) = self.x.st.bools.node(b) else {
                continue;
            };
            if !seen.insert(b) {
                continue;
            }
            let v = match atom {
                Atom::Lane(v)
                | Atom::Uniform(v)
                | Atom::Phi(v)
                | Atom::Quad(v)
                | Atom::Opaque(v) => Some(v),
                Atom::Constant(_) => None,
            };
            if v.is_some_and(|v| tainted.contains(&v)) {
                return true;
            }
            match v.and_then(|v| self.compared(env, v)) {
                Some(true) => stack.push(hi),
                Some(false) => stack.push(lo),
                None => stack.extend([lo, hi]),
            }
        }
        false
    }

    /// Whether `b` can be false, and true, given the values in `env`.
    fn can_be(
        &self,
        b: Bool,
        env: &HashMap<Value, u32>,
        memo: &mut HashMap<Bool, (bool, bool)>,
    ) -> (bool, bool) {
        if b == Bool::TRUE || b == Bool::FALSE {
            return (b == Bool::FALSE, b == Bool::TRUE);
        }
        if let Some(&r) = memo.get(&b) {
            return r;
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let v = match atom {
            Atom::Lane(v) | Atom::Uniform(v) => Some(v),
            _ => None,
        };
        let r = match v.and_then(|v| self.compared(env, v)) {
            Some(true) => self.can_be(hi, env, memo),
            Some(false) => self.can_be(lo, env, memo),
            None => {
                let (a, c) = self.can_be(lo, env, memo);
                let (d, e) = self.can_be(hi, env, memo);
                (a || d, c || e)
            }
        };
        memo.insert(b, r);
        r
    }

    /// The instruction writing `v`.
    fn def_inst(&self, v: Value) -> Option<&'a Inst> {
        let b = *self.def_block.get(&v)?;
        let p = self.x.p;
        p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))
    }

    /// The constant `v` holds, `env` giving phi values: moves of constants.
    fn konst(&self, env: &HashMap<Value, u32>, v: Value) -> Option<u32> {
        if let Some(&k) = env.get(&v) {
            return Some(k);
        }
        let inst = self.def_inst(v)?;
        let op = match &inst.fields {
            Fields::Valu(w)
                if inst.name() == "v_mov_b32" && w.sdwa.is_none() && w.dpp.is_none() =>
            {
                w.src.first()?
            }
            Fields::Sop { src, .. } if inst.name() == "s_mov_b32" => src.first()?,
            Fields::Sop { imm, .. } if inst.name() == "s_movk_i32" => {
                return Some(i32::from((*imm)? as i16) as u32);
            }
            _ => return None,
        };
        self.konst_operand(env, inst, op)
    }

    fn konst_operand(&self, env: &HashMap<Value, u32>, inst: &Inst, op: &Operand) -> Option<u32> {
        match *op {
            Operand::Const(c) => crate::resource::inline(c),
            Operand::Literal(x) => Some(x),
            Operand::Scalar { code, .. } if code != NULL => {
                self.konst(env, inst.read(Reg::S(code))?)
            }
            Operand::Vector { reg, .. } => self.konst(env, inst.read(Reg::V(reg))?),
            _ => None,
        }
    }

    /// The lane's bit of the 32-bit integer compare writing `v` (a VALU
    /// result before EXEC masking, or SCC), when its operands are constants.
    fn compared(&self, env: &HashMap<Value, u32>, v: Value) -> Option<bool> {
        let inst = self.def_inst(v)?;
        let name = inst.name();
        let rest = ["v_cmpx_", "v_cmp_", "s_cmp_"]
            .iter()
            .find_map(|p| name.strip_prefix(p))?;
        let (op, ty) = rest.rsplit_once('_')?;
        let (a, b) = match &inst.fields {
            Fields::Valu(w) if w.sdwa.is_none() && w.dpp.is_none() => {
                (w.src.first()?, w.src.get(1)?)
            }
            Fields::Sop { src, .. } => (src.first()?, src.get(1)?),
            _ => return None,
        };
        let (a, b) = (
            self.konst_operand(env, inst, a)?,
            self.konst_operand(env, inst, b)?,
        );
        let (sa, sb) = (a as i32, b as i32);
        Some(match (op, ty) {
            ("f", _) => false,
            ("t", _) => true,
            ("eq", "u32" | "i32") => a == b,
            ("ne" | "lg", "u32" | "i32") => a != b,
            ("lt", "u32") => a < b,
            ("le", "u32") => a <= b,
            ("gt", "u32") => a > b,
            ("ge", "u32") => a >= b,
            ("lt", "i32") => sa < sb,
            ("le", "i32") => sa <= sb,
            ("gt", "i32") => sa > sb,
            ("ge", "i32") => sa >= sb,
            _ => return None,
        })
    }

    /// The incoming edges (phi, predecessor index) over which a value can
    /// reach a read. A phi that only another read phi consumes is live too
    /// (else a value with no DXIL form reaching a read through two joins
    /// would be replaced by zero), but only over edges a lane can take on
    /// its way there: each step back adds the branch a lane took on that
    /// edge, and a contradiction (the lane that skipped an arm is not one
    /// that ran it) leaves the edge dead. A back edge forgets the branches
    /// (the loop redefines their conditions). Fewer conditions are weaker,
    /// so keeping only the last few stays sound.
    fn live_edges(&self) -> HashSet<(Value, usize)> {
        const MAX_KNOWN: usize = 4;
        let (p, st) = (self.x.p, &self.x.st);
        let mut work: Vec<(Value, Vec<(Bool, bool)>)> =
            self.read_phis.iter().map(|&v| (v, Vec::new())).collect();
        let mut live = HashSet::new();
        let mut seen = HashMap::<Value, Vec<Vec<(Bool, bool)>>>::new();
        while let Some((v, known)) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            let done = seen.entry(v).or_default();
            // Known of fewer lanes than before: nothing new is live.
            if done.iter().any(|d| d.iter().all(|c| known.contains(c))) {
                continue;
            }
            done.push(known.clone());
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                if self.tail[q].is_none() {
                    continue;
                }
                let mut k = Vec::new();
                // Forward edges out of a block that is no loop's latch (a
                // loop's exit is also taken at its iteration limit).
                if let LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } = st.exits[q]
                    && taken != not_taken
                    && self.rpo[taken] > self.rpo[q]
                    && self.rpo[not_taken] > self.rpo[q]
                {
                    k.clone_from(&known);
                    k.push((when, taken == b));
                    k.drain(..k.len().saturating_sub(MAX_KNOWN));
                    if !st.bools.meet(&k) {
                        continue;
                    }
                } else if self.rpo[q] < self.rpo[b] {
                    k.clone_from(&known);
                }
                live.insert((v, n));
                let arg = p.blocks[b].phis[i].args[n];
                if self.phi_at.contains_key(&arg) {
                    work.push((arg, k));
                }
            }
        }
        live
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        // Lane-mask joins first (what an edge brings may ask for more joins,
        // of masks and of numbers): each edge's mask as the structure
        // tracked it, a constant or a scalar-built one included.
        let st = self.x.st;
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let args = st
                .phi_masks
                .get(&phi.value)
                .ok_or_else(|| unsupported(offset, "a lane-mask join of an untracked mask"))?;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let x = self.boolean(args[n], qb).map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        // A join that only another join reads is read wherever that one is,
        // over the edges a lane can take there ([`Self::live_edges`]). Left
        // unmarked, a value without a DXIL form coming into it became zero
        // without a word (the looked-up tint of the material resolve
        // programs, whose size computation had no value, was zero in every
        // lane that way).
        // shortcut: only with `VICTORY_STRICT_JOINS` set (the offline pack
        // listing sets it to count them): 40 programs that translate today
        // stopped by name when every join of a read join counted as read
        // (vehicle paint pixel shaders at "the first half of an
        // interpolation", foliage mesh shaders); until those are lowered
        // they keep drawing with that zero, and only joins an instruction
        // reads count.
        let live = strict_joins().then(|| self.live_edges());
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            // A join of numbers with dead registers (`dead`): the paths that bring those
            // never read the result, so they bring zero. Not for a join that is itself a
            // lane mask (a ballot gives that one as data).
            let st = self.x.st;
            let numbers = !bit
                && !st.masks.contains_key(&phi.value)
                && !st.highs.contains_key(&phi.value)
                && phi
                    .args
                    .iter()
                    .any(|a| matches!(self.vals.get(a), Some(Val::I32(_))));
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                let read = match &live {
                    Some(live) => live.contains(&(phi.value, n)),
                    None => self.read_phis.contains(&phi.value),
                };
                // Read, or kept by a guarded write (`kept_phis`): what the
                // edge can bring is made either way; only a read stops the
                // translation where it has no form.
                let wanted = read || self.kept_phis.contains(&phi.value);
                if wanted && !self.vals.contains_key(&arg) {
                    let _ = self.lazy_barycentric(arg);
                    // A root parameter nothing read before the join (user
                    // data is loaded on first use, in the entry block).
                    if !self.vals.contains_key(&arg)
                        && let Some(&(param, dword)) = self.params.get(&arg)
                    {
                        let val = self.user(param, dword);
                        self.vals.insert(arg, val);
                    }
                }
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Missing(_)) | None, false)
                        if numbers && self.dead(arg, &mut Vec::new()) =>
                    {
                        self.s.m.i32(0)
                    }
                    // A lane mask joined with numbers (a traversal that keeps its lanes' masks on
                    // a stack with its node numbers): the mask comes as the number it is, the
                    // ballot of its lanes where it is written.
                    (Some(Val::Missing(_)) | None, false)
                        if wanted
                            && self.mask_data == MaskData::Console
                            && (st.masks.contains_key(&arg) || st.highs.contains_key(&arg)) =>
                    {
                        match self.mask_dword(arg) {
                            Ok(x) => x,
                            Err(why) => return Err(Stop::Retry(phi.value, why)),
                        }
                    }
                    // SCC of a mask operation ("some lane is left") joined with a scalar test:
                    // the compiler branches around code that sets SCC and tests SCC after it,
                    // the skip bringing "no lane" along. As data it is "some lane of the wave
                    // has the bit" - right only if the wave took that skip together, as the
                    // console does: a lane that skipped alone would bring "some lane" where the
                    // console has the skipped code's SCC. So the branch must be one the wave
                    // takes together; else its region is named for `emit`'s retry to widen it.
                    (Some(Val::Missing(_)) | None, true)
                        if wanted
                            && self.mask_data == MaskData::Console
                            && st.scc_some.contains_key(&arg) =>
                    {
                        let per_lane = !st.wide.contains(&q)
                            && matches!(st.exits[q], LaneExit::Branch { when, .. }
                                if !st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_))));
                        if per_lane {
                            for &other in p.blocks[b].preds.iter().filter(|&&o| o != q) {
                                need(other);
                            }
                            return Err(Stop::Retry(phi.value, LANE_MASK));
                        }
                        match self.scc_some(arg) {
                            Ok(x) => x,
                            Err(why) => return Err(Stop::Retry(phi.value, why)),
                        }
                    }
                    // No lane from this edge reads it, or no lane comes
                    // along it at all: any value will do.
                    _ if read
                        && (self.never_taken(q, b) || self.dead_on_edge(b, n, phi.reg)) =>
                    {
                        if bit {
                            self.s.m.bool(false)
                        } else {
                            self.s.m.i32(0)
                        }
                    }
                    (Some(Val::Bit(_)), false) if read => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    (Some(Val::Missing(why)), _) if read => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ if read => {
                        return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        Ok(())
    }
}

/// The DXIL interpolation mode of a kind of barycentrics (none for the
/// pull model).
fn interp_mode(kind: Barycentric) -> Option<u8> {
    Some(match kind {
        Barycentric::PerspectiveCenter => 2,
        Barycentric::PerspectiveCentroid => 3,
        Barycentric::PerspectiveSample => 6,
        Barycentric::LinearCenter => 4,
        Barycentric::LinearCentroid => 5,
        Barycentric::LinearSample => 7,
        Barycentric::PerspectivePullModel => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower, packed_half};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    #[test]
    fn packed_pixel_inputs_find_their_second_row() {
        use super::super::container::SigElement;
        use super::half_input_partners;
        use crate::native::{PsInput, Usage};
        let entry = |row, row2, pack2| PsInput {
            usage: Usage::Generic,
            row,
            mask: 0xF,
            attr: row,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        let element = |row| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: true,
        };
        // Rows 0 and 3 pack and leave their second row out; 5 names its
        // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
        let io = StageIo::Ps {
            inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
        let p = half_input_partners(&io, &sig);
        assert_eq!(p.get(&0), Some(&1));
        assert_eq!(p.get(&3), Some(&4));
        assert_eq!(p.get(&5), None);
    }

    /// The packed pedestrian layouts of Grand Theft Auto VI: an element
    /// behind another one in its register follows in the first parameter
    /// with a dword left, and is not a half of its register's row.
    #[test]
    fn packed_elements_behind_another_take_the_first_free_dword() {
        use super::super::container::SigElement;
        use super::packed_half;
        let element = |index, row, col, cols| SigElement {
            name: "TEXCOORD".into(),
            index,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col,
            cols,
            usage: 0,
            half: true,
        };
        // MS_VS_PedTransformPreSkinD_Packed 005fe88ec1bdc0c7 and its pixel
        // shader 93667197d3d05e1c: TEXCOORD6 is one value at register 4
        // column 3; parameter 3 (rows 6 and 7) carries it in dword 3.
        let sig = vec![
            element(1, 2, 0, 4),
            element(2, 3, 0, 4),
            element(4, 4, 0, 3),
            element(6, 4, 3, 1),
            element(3, 5, 0, 4),
            element(5, 6, 0, 4),
            element(7, 7, 0, 2),
            element(9, 9, 0, 3),
            element(11, 10, 0, 3),
            element(15, 12, 0, 4),
        ];
        let packs = [
            (2, Some(3)),
            (4, Some(5)),
            (6, Some(7)),
            (9, Some(10)),
            (12, None),
        ];
        let at = |a, h| packed_half(&sig, &packs, a, h);
        // Parameter 2: TEXCOORD4's three values, a zero half, TEXCOORD3.
        assert_eq!(at(4, 2), Some((4, 2)));
        assert_eq!(at(4, 3), None);
        assert_eq!(at(4, 4), Some((5, 0)));
        assert_eq!(at(4, 7), Some((5, 3)));
        // Parameter 3: TEXCOORD5, TEXCOORD7, then TEXCOORD6 alone.
        assert_eq!(at(6, 3), Some((6, 3)));
        assert_eq!(at(6, 5), Some((7, 1)));
        assert_eq!(at(6, 6), Some((4, 3)));
        assert_eq!(at(6, 7), None);
        // Parameters without such an element are as before.
        assert_eq!(at(9, 3), None);
        assert_eq!(at(9, 4), Some((10, 0)));
        assert_eq!(at(12, 3), Some((12, 3)));
        assert_eq!(at(12, 4), None);
        // The common layout (29805d23a495cdf8): TEXCOORD7 is two values at
        // register 1 columns 2-3; parameter 0 is TEXCOORD0, TEXCOORD1,
        // TEXCOORD7.
        let sig = vec![
            element(0, 1, 0, 2),
            element(7, 1, 2, 2),
            element(1, 2, 0, 4),
            element(2, 3, 0, 4),
            element(4, 4, 0, 3),
        ];
        let packs = [(3, Some(4)), (1, Some(2))];
        let at = |a, h| packed_half(&sig, &packs, a, h);
        assert_eq!(at(1, 1), Some((1, 1)));
        assert_eq!(at(1, 2), Some((2, 0)));
        assert_eq!(at(1, 5), Some((2, 3)));
        assert_eq!(at(1, 6), Some((1, 2)));
        assert_eq!(at(1, 7), Some((1, 3)));
        assert_eq!(at(3, 6), Some((4, 2)));
        assert_eq!(at(3, 7), None);
        // A row no element holds gives nothing, as before.
        assert_eq!(packed_half(&sig, &[(8, None)], 8, 0), None);
    }

    /// Whether a synthetic compute program translates with these input registers.
    fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .map(|_| ())
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                // The layout's user SGPRs (at least two), 512 bytes of LDS.
                rsrc2: 1 << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

    #[test]
    fn buffer_cache_invalidations_lower_as_no_ops() {
        // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
        // host to invalidate its cached view of a buffer.
        assert!(lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl0_inv");
        assert!(lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl1_inv");
    }

    /// Lowers a one-thread compute program that copies four root constants
    /// (s0..s3) into v1..v4, runs `body`, and stores v0 to a raw buffer from a
    /// descriptor table, so the vector instructions in `body` have an effect.
    fn lowers_valu(body: &[u32]) -> Result<(), crate::ShaderError> {
        let mut words = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xFA00_0000,
        ];
        // v_mov_b32 v1..v4, s0..s3
        words.extend((0..4u32).map(|i| 0x7E00_0200 | (i + 1) << 17 | i));
        words.extend_from_slice(body);
        words.extend_from_slice(&[
            0xE070_0000, // buffer_store_dword v0, off, s[8:11], 0
            0x8002_0000,
            S_ENDPGM,
        ]);
        let constants = RootParam {
            sgprs: 4,
            ..root(RootType::Constants, 0, 0)
        };
        let buffer = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        lower_with(
            &words,
            vec![constants, root(RootType::Table, 4, 1)],
            vec![buffer],
        )
        .map(|_| ())
    }

    #[test]
    fn the_24_bit_multiplies_lower_both_halves() {
        // VOP2 `op v0, v1, v2`: v_mul_i32_i24 and v_mul_u32_u24 (the low
        // dword) and v_mul_hi_i32_i24 and v_mul_hi_u32_u24 (the high one).
        for op in [0x09u32, 0x0A, 0x0B, 0x0C] {
            let word = op << 25 | 2 << 9 | 0x101;
            lowers_valu(&[word]).unwrap_or_else(|e| panic!("VOP2 {op:#x}: {e}"));
        }
    }

    #[test]
    fn sums_of_absolute_differences_lower() {
        // VOP3 `op v0, v1, v2, v3`: v_sad_u8, v_sad_hi_u8, v_sad_u16,
        // v_sad_u32 and v_msad_u8.
        for op in [0x15Au32, 0x15B, 0x15C, 0x15D, 0x171] {
            let words = [0xD400_0000 | op << 16, 0x101 | 0x102 << 9 | 0x103 << 18];
            lowers_valu(&words).unwrap_or_else(|e| panic!("VOP3 {op:#x}: {e}"));
        }
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];

        for words in [add, sub] {
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn v_cndmask_accepts_uniform_and_constant_masks() {
        // Uniform select (`s_cselect_b32`) and inline constant masks in `v_cndmask_b32`
        // (resolved in `Structured::other` or directly as constants).
        let words = [
            0xBF06_8000u32, // s_cmp_eq_u32 s0, 0
            0x8500_80C1,    // s_cselect_b32 s0, -1, 0
            0xD501_0001,
            0x0001_0080,    // v_cndmask_b32 v1, 0, 1, s0
            0xD501_0002,
            0x0201_02C1,    // v_cndmask_b32 v2, 1, v1, -1
            0xD834_0000,
            0x0000_0200,    // ds_write_b32 v0, v2
            S_ENDPGM,
        ];
        lowers(&words, 0x8001).unwrap();
    }

    fn perm_features(selector: u32) -> u64 {
        // Synthetic compute shader:
        // v0 is the thread ID and can supply a dynamic selector.
        // Store the result in LDS to keep the permutation live.
        let words = [
            0x7E02_02FFu32,
            0x817F_34AB,
            0x7E04_02FF,
            0xFE01_56CD,
            0xD744_0003,
            257 | (258 << 9) | (selector << 18),
            0x3408_0082,
            0xD834_0000,
            0x0000_0304,
            S_ENDPGM,
        ];

        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [64, 1, 1],
            }),
            code,
        };

        let p = ir::build(&native.code, false, 256).unwrap();

        assert!(
            p.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.name() == "v_perm_b32")
        );

        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

        let io = io::raise(&p, &native).unwrap();

        let emitted = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();

        let container = crate::Container::parse(&emitted.dxil).unwrap();

        let features = container.part(crate::FourCC::SFI0).unwrap();

        u64::from_le_bytes(features.data.try_into().unwrap())
    }

    #[test]
    fn perm_constant_selector_needs_no_int64() {
        assert_eq!(perm_features(128) & 0x8000, 0);
    }

    #[test]
    fn perm_dynamic_selector_needs_no_int64() {
        assert_eq!(
            perm_features(256) & 0x8000,
            0,
            "dynamic v_perm_b32 must not require INT64"
        );
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    /// A constant buffer in a table is read through the raw view the
    /// host keeps at its slot, by scalar loads and per lane alike: no heap
    /// handle is declared a constant buffer (the host's constant buffer
    /// views want an alignment the title's lack).
    #[test]
    fn a_tables_constant_buffer_is_read_as_a_raw_view() {
        let layout = || vec![root(RootType::Table, 0, 0)];
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        let load = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        ];
        let scalar = [
            0xF420_0404u32,
            0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
            0x7E02_0210, // v_mov_b32 v1, s16
        ];
        let per_lane = [
            0xE030_2000u32,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        ];
        let store = [
            0xD834_0000u32,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        for body in [&scalar[..], &per_lane[..]] {
            let words = [&load[..], body, &store[..]].concat();
            let l = lower_with(&words, layout(), used()).unwrap();
            assert_eq!(l.heap_constants, 0);
            assert!(l.root_cbvs.is_empty() && l.root_srvs.is_empty());
            assert!(!l.heap, "a table's view, not one picked from the heap");
        }
    }

    /// A view the program picks from the resource heap by an index from
    /// data may be a buffer view or a constant buffer view: a scalar load
    /// reads the raw view at the slot where the stride table has a number
    /// for it, and declares no constant buffer.
    #[test]
    fn a_constant_buffer_picked_from_the_heap_is_read_as_a_raw_view() {
        let words = [
            0x8F06_8500u32, // s_lshl_b32 s6, s0, 5
            0xBE84_0301,    // s_mov_b32 s4, s1
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF408_0202,
            0x0C00_0000, // s_load_dwordx4 s[8:11], s[4:5], s6
            0xF420_0404,
            0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
            0x7E02_0210, // v_mov_b32 v1, s16
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = vec![
            root(RootType::Constants, 0, 0),
            root(RootType::ResourceHeap, 1, 1),
        ];
        let l = lower_with(&words, layout, vec![]).unwrap();
        assert_eq!(l.heap_constants, 0);
        assert_eq!(l.root_srvs, [crate::emit::SLOT_STRIDES]);
        assert!(l.heap && !l.heap_writes);
    }

    /// Tables of 2D textures at s0 and samplers at s1.
    fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn a_descriptor_loaded_only_for_its_size_takes_the_kind_of_its_reload() {
        // The record count of the buffer at s0 (count - 1 as an index), then
        // the same descriptor loaded again for the indexed read.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0x810C_C10A, // s_add_i32 s12, s10, -1
            0xF408_0402,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[4:5], null
            0x7E04_020C, // v_mov_b32 v2, s12
            0xE030_2000,
            0x8004_0102, // buffer_load_dword v1, v2, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let used = vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        lower_with(&words, layout, used).unwrap();
    }

    #[test]
    fn a_waterfall_loop_that_only_sizes_takes_the_kind_of_the_sampled_one() {
        // Each lane names table entry (lane & 1) by its byte offset in v7.
        // A first waterfall loop only reads the width bits of the entry
        // (`v_alignbit_b32`), a second one samples it; the width goes to the
        // buffer at dword 16 of the table.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0203,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x360E_0081, // v_and_b32 v7, 1, v0
            0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xBE97_037E, // s_mov_b32 s23, exec_lo
            0xBE95_037E, // s_mov_b32 s21, exec_lo
            0x7E28_0507, // L1: v_readfirstlane_b32 s20, v7
            0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
            0xF40C_0302,
            0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xD54E_0008,
            0x0278_1A0E, // v_alignbit_b32 v8, s14, s13, 30
            0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
            0xBF89_FFF7, // s_cbranch_execnz L1
            0xBEFE_0317, // s_mov_b32 exec_lo, s23
            0xBE95_037E, // s_mov_b32 s21, exec_lo
            0x7E28_0507, // L2: v_readfirstlane_b32 s20, v7
            0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
            0xF40C_0302,
            0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
            0xBF89_FFF6, // s_cbranch_execnz L2
            0xBEFE_0317, // s_mov_b32 exec_lo, s23
            0xE070_1000,
            0x800A_0810, // buffer_store_dword v8, v16, s[40:43], 0 offen
            S_ENDPGM,
        ];
        let (layout, used) = texture_and_uav_tables();
        lower_wave(&words, layout, used, false).unwrap();
    }

    #[test]
    fn compute_spills_beside_a_descriptor_word_read_back() {
        // Dword 0 of a buffer descriptor (no host value) and a root constant
        // spilled to lanes 0 and 1 of v5 in a compute program; reading lane
        // 1 back is the constant (the descriptor's lane is never read).
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xD761_0005,
            0x0001_0008, // v_writelane_b32 v5, s8, 0
            0xD761_0005,
            0x0001_0201, // v_writelane_b32 v5, s1, 1
            0xD760_000C,
            0x0001_0305, // v_readlane_b32 s12, v5, 1
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0x7E06_020C, // v_mov_b32 v3, s12
            0xD834_0000,
            0x0000_0300, // ds_write_b32 v0, v3
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let used = vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        lower_with(&words, layout, used).unwrap();
    }

    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    /// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
    pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let (layout, mut used) = texture_tables();
        used.push(UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 16,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        });
        (layout, used)
    }

    #[test]
    fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
        use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
        for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
            let (layout, used) = texture_and_uav_tables();
            let l = lower_wave(words, layout, used, false).unwrap();
            // One image handle, from the join's per-lane heap index.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn scalar_bit_tests_lower() {
        // Each tests one bit of a constant into SCC and branches on it (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let bit_tests: [&[u32]; 4] = [
            &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
            &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
            ],
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
            ],
        ];
        for head in bit_tests {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words, 0).unwrap();
        }
    }

    /// Runs `f` with `VICTORY_STRICT_JOINS` on (joins read only through
    /// other joins count as read; off by default, see `fill_phis`).
    fn strictly<R>(f: impl FnOnce() -> R) -> R {
        let was = super::TEST_STRICT_JOINS.replace(Some(true));
        let r = f();
        super::TEST_STRICT_JOINS.set(was);
        r
    }

    #[test]
    fn a_missing_value_stays_missing_through_two_live_joins() {
        strictly(missing_through_two_live_joins);
    }

    fn missing_through_two_live_joins() {
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // skip the first overwrite
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
            0xBF85_0001, // skip the second overwrite
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // use the result to branch
            0xBE83_0380,
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        // Compare to a direct read so future timestamp support works in both cases.
        let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[8..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
        words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
        lowers(&words, group_ids_xy).expect("dead timestamp");
    }

    #[test]
    fn a_value_no_lane_from_its_edge_reads_needs_no_form() {
        // s6 is a timestamp (no DXIL form) where the load is skipped and 5
        // where it runs; a flag in v1, set only with the load, guards the
        // read after the join (the console compiler's form, where the
        // skipped path leaves a stale lane mask in the register).
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0x7E02_0280, // v_mov_b32 v1, 0
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0002, // s_cbranch_scc1 -> join
            0xBE86_0385, // s_mov_b32 s6, 5
            0x7E02_02C1, // v_mov_b32 v1, -1
            0x7D84_02C1, // join: v_cmp_eq_u32 vcc_lo, -1, v1
            0xBF86_0003, // s_cbranch_vccz -> end
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // s_cbranch_scc1 -> end
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        lowers(&words, group_ids_xy).expect("the timestamp is never read");
        // With the flag set on both paths the timestamp is read: as refused
        // (or accepted) as a direct read.
        words[2] = 0x7E02_02C1; // v_mov_b32 v1, -1
        let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[9..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
    }

    #[test]
    fn a_missing_value_on_a_path_no_lane_takes_is_dead() {
        strictly(missing_on_a_path_no_lane_takes);
    }

    fn missing_on_a_path_no_lane_takes() {
        let refusal = |code: &[u32]| match lowers(code, 1 << 7) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        // The timestamp stays in s6 only where s0 == 0, and those lanes
        // overwrite it before the second join: it never reaches the read.
        let mut joins = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // s_cbranch_scc1: s0 == 0 keeps the timestamp
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF84_0001, // s_cbranch_scc0: s0 != 0 keeps that 0
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001,
            0xBE83_0380,
            S_ENDPGM,
        ];
        assert_eq!(refusal(&joins), None);
        // Tested on the other side, the timestamp reaches the read.
        joins[5] = 0xBF85_0001; // s_cbranch_scc1
        assert!(refusal(&joins).is_some());
        // The same through a VGPR whose old value a guarded write keeps
        // (lanes outside it get 0 for a value with no DXIL form, as for any
        // such write): keeping does not make the join live.
        let kept = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0x7E02_0206, // v_mov_b32 v1, s6
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0001, // s_cbranch_scc1: s0 == 0 keeps the timestamp
            0x7E02_0280, // v_mov_b32 v1, 0
            0xBF84_0004, // s_cbranch_scc0: s0 != 0 keeps that 0
            0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
            0x7E02_0282, // v_mov_b32 v1, 2 (the other lanes keep v1)
            0xBEFE_03C1, // s_mov_b32 exec_lo, -1
            0x7E02_0283, // v_mov_b32 v1, 3
            0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
            0xBF87_0001, // s_cbranch_vccnz
            0xBE83_0380,
            S_ENDPGM,
        ];
        assert_eq!(refusal(&kept), None);
    }

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(
            buffer_format(44),
            Some((&[2, 10, 10, 10][..], Numeric::Unorm))
        );
        assert_eq!(
            buffer_format(45),
            Some((&[2, 10, 10, 10][..], Numeric::Snorm))
        );
        assert_eq!(
            buffer_format(46),
            Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
        );
        assert_eq!(
            buffer_format(47),
            Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
        );
        assert_eq!(
            buffer_format(48),
            Some((&[2, 10, 10, 10][..], Numeric::Uint))
        );
        assert_eq!(
            buffer_format(49),
            Some((&[2, 10, 10, 10][..], Numeric::Sint))
        );
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }

    #[test]
    fn packed_halves_fill_their_parameters() {
        // 16-bit rows as a title's signature has them: row 0 holds two
        // elements (x y and z w), row 3 is 32-bit, row 4 has one channel at w.
        let el = |row: u32, col: u8, cols: u8, half: bool| super::SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(row),
            col,
            cols,
            usage: 0,
            half,
        };
        let sig = [
            el(0, 0, 2, true),
            el(0, 2, 2, true),
            el(1, 0, 4, true),
            el(2, 0, 3, true),
            el(3, 0, 4, false),
            el(4, 0, 3, true),
            el(4, 3, 1, true),
            el(5, 0, 3, true),
        ];
        // The parameters as the program's metadata names them: rows 0 and
        // 1, rows 2 and 4, row 5 alone.
        let packs = [(0, Some(1)), (2, Some(4)), (5, None)];
        // Each half as row and channel ("-": padding or past the end).
        let param = |a| {
            (0..8)
                .map(|h| match packed_half(&sig, &packs, a, h) {
                    Some((r, c)) => format!("{r}{}", ["x", "y", "z", "w"][usize::from(c)]),
                    None => "-".into(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        // Each row's first element, then (it sits behind another in its
        // register) row 0's z w.
        assert_eq!(param(0), "0x 0y 1x 1y 1z 1w 0z 0w");
        // Three channels take two dwords; row 4's w fills the first
        // parameter with room, after row 5.
        assert_eq!(param(2), "2x 2y 2z - 4x 4y 4z -");
        assert_eq!(param(5), "5x 5y 5z - 4w - - -");
        // No parameter starts in rows 1, 3 or 4.
        for a in [1, 3, 4] {
            assert_eq!(param(a), "- - - - - - - -");
        }
    }
}
