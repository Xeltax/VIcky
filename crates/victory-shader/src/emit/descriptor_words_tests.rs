//! Descriptors whose dwords a program reads only as numbers
//! (`descriptor_words.rs`): synthetic GFX10 compute programs lowered to
//! DXIL, validated by `dxil.dll` when present, and run on the host GPU and
//! WARP. Each loads a descriptor that no instruction uses as one and stores
//! fields of it: an image's size (a program that measures a texture it
//! does not sample), a buffer's address taken from the resource heap at an
//! index it computes (the ray-tracing shadow programs), and a buffer's
//! record count (a program that skips its work for an empty buffer).

use super::host::{Dispatch, Host, Root, Stride, View};
use super::shader::{SLOT_VIEW_ADDRESSES, USER_VIEW_ADDRESS};
use super::{HostShader, lower, validator};
use crate::ShaderError;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM0: u32 = 0xBF8C_C07F;
/// An SGPR operand that is a literal dword after the instruction.
const LIT: u8 = 0xFF;
/// An inline constant `n` (0..=64) as an SGPR operand.
const fn k(n: u8) -> u8 {
    0x80 + n
}

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0300 | (sdst as u32) << 16 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | (op as u32 & 0x7F) << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}
const S_AND_B32: u8 = 0x0E;
const S_OR_B32: u8 = 0x10;
const S_LSHL_B32: u8 = 0x1E;
const S_LSHR_B32: u8 = 0x20;
const S_BFE_U32: u8 = 0x27;

/// SMEM `s_load_dword<n> s[dst:...], s[base:base+1], <soffset or null>, offset`
/// (`op`: 2 for x4, 3 for x8).
const fn s_load(op: u8, dst: u8, base: u8, soffset: Option<u8>, offset: u32) -> [u32; 2] {
    let so = match soffset {
        Some(s) => s as u32,
        None => 125,
    };
    [
        0xF400_0000 | (op as u32) << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        so << 25 | (offset & 0x1F_FFFF),
    ]
}

/// VOP1 `v_mov_b32 vdst, s<src>`.
const fn v_mov_from_sgpr(vdst: u8, src: u8) -> u32 {
    0x7E00_0200 | (vdst as u32) << 17 | src as u32
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// Stores `v0..v<n>` to dwords 0.. of the raw UAV whose descriptor is in
/// `s[16:19]`, and ends.
fn store_and_end(words: &mut Vec<u32>, n: u8) {
    for v in 0..n {
        words.extend(buffer_store_dword(v, 16, 4 * u16::from(v)));
    }
    words.push(S_ENDPGM);
}

/// The root constants `s0..s3` (slot 0), `root` at `s4` (slot 1) and the
/// table of the output UAV at `s5` (slot 2).
fn layout(root: RootType) -> Vec<RootParam> {
    vec![
        RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
            base_register: 0,
            user_data: 0,
            slot: 0,
        },
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 1,
        },
        RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 5,
            slot: 2,
        },
    ]
}

/// The output UAV: slot 2 of the table at `s5` (both tables start at heap
/// index 0).
fn output() -> UsedResource {
    UsedResource {
        user_data: 5,
        slot: 2,
        ..range(ResourceType::Uav, 16, 4, 0)
    }
}

/// Moves the output table's address to `s[6:7]` and clears the high half
/// of `s[4:5]`.
fn prologue() -> Vec<u32> {
    vec![s_mov_b32(6, 5), s_mov_b32(7, k(0)), s_mov_b32(5, k(0))]
}

/// A range of the table at `s4`.
fn range(kind: ResourceType, table_offset_dwords: u32, dwords: u8, stride: u32) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 4,
        slot: 1,
    }
}

/// Lowers a one-thread wave32 compute program; the DXIL is validated where
/// there is a `dxil.dll`.
fn lower_cs(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 6 << 1, // 6 user SGPRs: s0..s5
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok(shader)
}

/// The program's instructions as SP3 text, to check the encodings above.
fn text(words: &[u32]) -> Vec<String> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::gfx10::decode(&code)
        .expect("decode")
        .iter()
        .map(|i| i.text(false))
        .collect()
}

/// Runs `shader` once per set of root constants with `views` at heap
/// indices 0.. and `roots` after the user data, on the hardware GPU and
/// WARP (each skipped without a device): per device and set, the UAVs'
/// contents in heap order. User data the host computes comes from `host`.
fn run(
    shader: &HostShader,
    constants: &[[u32; 4]],
    host_word: impl Fn(RootType, u32) -> u32,
    views: &[View],
    roots: impl Fn() -> Vec<(u32, Root)>,
) -> Vec<(&'static str, Vec<Vec<Vec<u32>>>)> {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut sets = Vec::new();
        for set in constants {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match u.root {
                    RootType::Constants => set[u.dword as usize],
                    // The table starts at heap index 0.
                    RootType::Table => 0,
                    other => host_word(other, u.dword),
                };
            }
            let mut all = vec![(0, Root::Constants(ud))];
            all.extend(roots());
            let got = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: all,
                        views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            sets.push(got);
        }
        out.push((label, sets));
    }
    out
}

/// No host word but those a test names.
fn no_host_word(root: RootType, dword: u32) -> u32 {
    panic!("unexpected user data {root:?} dword {dword}")
}

/// An image picked from the resource heap at an index from a root
/// constant, never sampled: the program takes its width - 1 (dword 1 bits
/// 30-31 and dword 2 bits 0-13), height - 1 (dword 2 bits 14-29), base and
/// last level (dword 3 bits 12-15 and 16-19) out of its descriptor.
fn measures_an_image() -> Vec<u32> {
    let mut w = prologue();
    w.push(sop2(S_LSHL_B32, 3, 0, k(5))); // the image's slot from s0
    w.extend(s_load(3, 8, 4, Some(3), 0)); // s_load_dwordx8 s[8:15], s[4:5], s3
    w.extend(s_load(2, 16, 6, None, 0x40)); // the output UAV
    w.extend([
        S_WAITCNT_LGKM0,
        sop2(S_LSHR_B32, 20, 9, k(30)),
        sop2(S_LSHL_B32, 21, 10, k(2)),
        sop2(S_OR_B32, 20, 20, 21),
        sop2(S_AND_B32, 20, 20, LIT),
        0x3FFF,
        sop2(S_BFE_U32, 21, 10, LIT),
        14 | 16 << 16,
        sop2(S_BFE_U32, 22, 11, LIT),
        12 | 4 << 16,
        sop2(S_BFE_U32, 23, 11, LIT),
        16 | 4 << 16,
        v_mov_from_sgpr(0, 20),
        v_mov_from_sgpr(1, 21),
        v_mov_from_sgpr(2, 22),
        v_mov_from_sgpr(3, 23),
    ]);
    store_and_end(&mut w, 4);
    w
}

#[test]
fn the_test_programs_decode_as_written() {
    let image = text(&measures_an_image());
    assert_eq!(image[..3], ["s_mov_b32 s6, s5", "s_mov_b32 s7, 0", "s_mov_b32 s5, 0"]);
    assert_eq!(image[3], "s_lshl_b32 s3, s0, 5");
    assert_eq!(image[4], "s_load_dwordx8 s[8:15], s[4:5], s3");
    assert_eq!(image[5], "s_load_dwordx4 s[16:19], s[6:7], 0x000040");
    assert_eq!(image[11], "s_bfe_u32 s21, s10, lit(0x0010000e)");
    assert_eq!(image[14], "v_mov_b32 v0, s20");
    assert_eq!(image[18], "buffer_store_dword v0, v0, s[16:19], 0");
    assert_eq!(image[19], "buffer_store_dword v1, v0, s[16:19], 0 offset:4");
    let address = text(&takes_an_address(true));
    assert_eq!(address[4], "s_load_dwordx4 s[8:11], s[4:5], s3");
    assert_eq!(address[7], "s_and_b32 s9, s9, lit(0x0000ffff)");
    let count = text(&tests_a_record_count());
    assert_eq!(count[3], "s_load_dwordx4 s[8:11], s[4:5], null");
    assert_eq!(count[6], "v_mov_b32 v0, s10");
}

/// The program measures the image at the slot it picks, as the console's
/// descriptor gives it (the host view's own size, levels from 0).
#[test]
fn an_image_only_measured_gives_its_size() {
    let shader = lower_cs(&measures_an_image(), layout(RootType::ResourceHeap), vec![output()])
        .expect("lowered");
    assert!(shader.root_srvs.is_empty(), "{:?}", shader.root_srvs);
    let (a, b) = ([[0.0f32; 4]; 8 * 4], [[0.0f32; 4]; 4 * 2]);
    let views = [View::Texture(8, 4, &a), View::Texture(4, 2, &b), View::RawUav(4)];
    let runs = run(&shader, &[[0; 4], [1, 0, 0, 0]], no_host_word, &views, Vec::new);
    for (label, sets) in &runs {
        assert_eq!(sets[0][0], [7, 3, 0, 0], "{label}: 8x4, one level");
        assert_eq!(sets[1][0], [3, 1, 0, 0], "{label}: 4x2");
    }
}

/// A buffer descriptor picked from the resource heap at an index from a
/// root constant; the program keeps its address (dwords 0 and 1, the high
/// bits masked) and, with `address`, stores it, else stores a constant.
fn takes_an_address(address: bool) -> Vec<u32> {
    let mut w = prologue();
    w.push(sop2(S_LSHL_B32, 3, 0, k(5)));
    w.extend(s_load(2, 8, 4, Some(3), 0)); // s_load_dwordx4 s[8:11], s[4:5], s3
    w.extend(s_load(2, 16, 6, None, 0x40)); // the output UAV
    w.push(S_WAITCNT_LGKM0);
    if address {
        w.extend([sop2(S_AND_B32, 9, 9, LIT), 0xFFFF]);
        w.extend([v_mov_from_sgpr(0, 8), v_mov_from_sgpr(1, 9)]);
    } else {
        w.extend([v_mov_from_sgpr(0, k(1)), v_mov_from_sgpr(1, k(2))]);
    }
    store_and_end(&mut w, 2);
    w
}

/// The address comes from the host's table of view addresses by slot, at
/// the slot the program picks.
#[test]
fn a_buffer_address_picked_from_the_heap_reads_the_hosts_table() {
    let shader = lower_cs(&takes_an_address(true), layout(RootType::ResourceHeap), vec![output()])
        .expect("lowered");
    assert_eq!(shader.root_srvs, [SLOT_VIEW_ADDRESSES]);
    // Two dwords per slot: the low dword, then the bits from 32 up.
    let table = vec![0x0123_4500, 0x12, 0x89AB_CDE0, 0x00FF, 0, 0];
    let views = [View::RawUav(1), View::RawUav(1), View::RawUav(2)];
    let roots = || {
        vec![(1, Root::View { srv: true, data: table.clone() })]
    };
    let runs = run(&shader, &[[0; 4], [1, 0, 0, 0]], no_host_word, &views, roots);
    for (label, sets) in &runs {
        assert_eq!(sets[0][2], [0x0123_4500, 0x12], "{label}: slot 0");
        assert_eq!(sets[1][2], [0x89AB_CDE0, 0xFF], "{label}: slot 1");
    }
}

/// A load whose dwords nothing reads gets no value and asks the host for
/// nothing; a dword the host has no value for stops the program.
#[test]
fn only_dwords_read_ask_for_a_value() {
    let shader = lower_cs(&takes_an_address(false), layout(RootType::ResourceHeap), vec![output()])
        .expect("lowered");
    assert!(shader.root_srvs.is_empty(), "{:?}", shader.root_srvs);
    // The format and type (dword 3) of a buffer of the heap.
    let mut w = takes_an_address(false);
    let at = w.iter().position(|&x| x == v_mov_from_sgpr(0, k(1))).unwrap();
    w[at] = v_mov_from_sgpr(0, 11);
    let refused = lower_cs(&w, layout(RootType::ResourceHeap), vec![output()]);
    let Err(ShaderError::Unsupported { what, .. }) = refused else {
        panic!("lowered a buffer's format read as data");
    };
    assert_eq!(what, "a descriptor's address or format read as data");
    // The second dword alone of four of the heap: perhaps an image's
    // (width bits), so no address is given for it.
    w[at] = v_mov_from_sgpr(0, 9);
    let refused = lower_cs(&w, layout(RootType::ResourceHeap), vec![output()]);
    let Err(ShaderError::Unsupported { what, .. }) = refused else {
        panic!("lowered the second dword alone of a heap entry");
    };
    assert_eq!(what, "a descriptor's address or format read as data");
}

/// A structured buffer's descriptor at a fixed place in a table, never
/// used as one: the program stores its record count (dword 2) and the low
/// dword of its address.
fn tests_a_record_count() -> Vec<u32> {
    let mut w = prologue();
    w.extend(s_load(2, 8, 4, None, 0)); // s_load_dwordx4 s[8:11], s[4:5], null
    w.extend(s_load(2, 16, 6, None, 0x40)); // the output UAV
    w.extend([S_WAITCNT_LGKM0, v_mov_from_sgpr(0, 10), v_mov_from_sgpr(1, 8)]);
    store_and_end(&mut w, 2);
    w
}

/// The record count comes from the view's size in elements of the range's
/// stride; the address from the host when it binds.
#[test]
fn an_unused_buffers_record_count_comes_from_its_view() {
    let used = vec![
        range(ResourceType::Srv, 0, 4, 16),
        output(),
    ];
    let shader = lower_cs(&tests_a_record_count(), layout(RootType::Table), used).expect("lowered");
    let host_word = |root: RootType, dword: u32| match (root, dword) {
        (RootType::Other(USER_VIEW_ADDRESS), 0) => 0xCAFE_0000,
        (RootType::Other(code), 0) if code == USER_VIEW_ADDRESS + 1 => 0x12,
        other => panic!("unexpected user data {other:?}"),
    };
    // Three elements of 16 bytes, as the host makes a structured view: raw.
    let data = [0u32; 12];
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(1),
        View::RawUav(2),
    ];
    let runs = run(&shader, &[[0; 4]], host_word, &views, Vec::new);
    for (label, sets) in &runs {
        assert_eq!(sets[0][1], [3, 0xCAFE_0000], "{label}");
    }
}
