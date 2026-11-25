//! `lane_uniform`'s tests (fork PR #36): first-lane reads on a 32-lane host.
//! A file of their own so that `lower.rs` stays under the asset policy's
//! 512 KiB file limit.

use super::{Parts, lower};
use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn first_lane_read_of_uniform_lds_load_lowers_on_wave32() {
    // Wave64 compute program lowered for a 32-lane host: inside a divergent
    // branch (`v0 < 1`), read LDS at a uniform address (`v1 = s16`), add a
    // scalar (`v8 = s17 + v2`), and read the first lane into `s17`.
    // When `ds_read_b32` reads from the uniform `v1`, `v8` is wave-uniform
    // and `v_readfirstlane_b32` needs no cross-half exchange. When it reads
    // from per-lane `v0`, the divergent read must still be refused.
    let make = |addr_vgpr: u32| -> Vec<u32> {
        vec![
            0xBE90_0380, // s_mov_b32 s16, 0
            0xBE91_0380, // s_mov_b32 s17, 0
            0xD760_0004, // v_readlane_b32 s4, v0, 0 (triggers wave64 exchange)
            0x0001_0100,
            0x7D88_0081, // v_cmp_gt_u32_e32 vcc, 1, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0xBF88_0005, // s_cbranch_execz -> restore exec
            0x7E02_0210, // v_mov_b32 v1, s16
            0xD8D8_0000, // ds_read_b32 v2, v{addr}
            0x0200_0000 | addr_vgpr,
            0x4A10_0411, // v_add_nc_u32 v8, s17, v2
            0x7E22_0508, // v_readfirstlane_b32 s17, v8
            0xBEFE_0406, // s_mov_b64 exec, s[6:7]
            0x8011_0411, // s_add_u32 s17, s17, s4 (keep s17 and s4 live)
            0x7E06_0211, // v_mov_b32 v3, s17
            0xD834_0000, // ds_write_b32 v0, v3
            0x0000_0300,
            S_ENDPGM,
        ]
    };
    let run = |words: &[u32]| {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15, // 512 bytes LDS
                threads: [128, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, true, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let prev = super::TEST_HOST_WAVE.replace(Some(32));
        let out = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        });
        super::TEST_HOST_WAVE.set(prev);
        out
    };

    if let Err(e) = run(&make(1)) {
        panic!("uniform ds_read_b32 + scalar add must be recognised as wave-uniform, got: {e}");
    }
    let err = run(&make(0))
        .err()
        .expect("per-lane ds_read_b32 must be refused")
        .to_string();
    assert!(
        err.contains("a first-lane read where not every lane runs"),
        "per-lane ds_read_b32 must still require all lanes, got: {err}"
    );
}

#[test]
fn lane_uniform_handles_v255_and_high_pair_operands() {
    // Regression test for operands ending at v255 (exclusive end 256):
    // 1. A single-word uniform v255 must still be recognised as wave-uniform.
    // 2. A 64-bit pair v[254:255] with uniform v254 and per-lane v255 shifted
    //    right by 32 copies v255 into v8 and must NOT be classified as uniform.
    let run = |body: &[u32]| {
        let mut words = vec![
            0xBE90_0380, // s_mov_b32 s16, 0
            0xBE91_0380, // s_mov_b32 s17, 0
            0xD760_0004, // v_readlane_b32 s4, v0, 0 (triggers wave64 exchange)
            0x0001_0100,
            0x7D88_0081,                           // v_cmp_gt_u32_e32 vcc, 1, v0
            0xBE86_246A,                           // s_and_saveexec_b64 s[6:7], vcc
            0xBF88_0000 | (body.len() as u32 + 1), // s_cbranch_execz -> restore exec
        ];
        words.extend_from_slice(body);
        words.extend_from_slice(&[
            0x7E22_0508, // v_readfirstlane_b32 s17, v8
            0xBEFE_0406, // s_mov_b64 exec, s[6:7]
            0x8011_0411, // s_add_u32 s17, s17, s4
            0x7E06_0211, // v_mov_b32 v3, s17
            0xD834_0000, // ds_write_b32 v0, v3
            0x0000_0300,
            S_ENDPGM,
        ]);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [128, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, true, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let prev = super::TEST_HOST_WAVE.replace(Some(32));
        let out = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        });
        super::TEST_HOST_WAVE.set(prev);
        out
    };

    // Uniform v255 -> uniform v8 -> v_readfirstlane_b32 lowers on wave32.
    let uniform_v255 = [
        0x7FFE_0210, // v_mov_b32 v255, s16
        0x4A11_FE11, // v_add_nc_u32 v8, s17, v255
    ];
    if let Err(e) = run(&uniform_v255) {
        panic!("uniform v255 read must be recognised as wave-uniform, got: {e}");
    }

    // Uniform v254 + per-lane v255 -> v_lshrrev_b64 v[8:9], 32, v[254:255] -> per-lane v8.
    let varying_high = [
        0x7FFC_0210, // v_mov_b32 v254, s16
        0x7FFE_0300, // v_mov_b32 v255, v0
        0xD700_0008, // v_lshrrev_b64 v[8:9], 32, v[254:255]
        0x0003_FCA0,
    ];
    let err = run(&varying_high)
        .err()
        .expect("varying v255 high word must not be classified as uniform")
        .to_string();
    assert!(
        err.contains("a first-lane read where not every lane runs"),
        "varying v255 high word must require all lanes, got: {err}"
    );
}
