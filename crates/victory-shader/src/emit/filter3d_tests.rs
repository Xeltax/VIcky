//! A level-0 sample of a 3D texture with a linear, clamping immediate
//! sampler, by the card and by hand (`VICTORY_FILTER_3D`, port of
//! readmet3xt-lab d246d1080; tests after its own): the same trilinear
//! values, and the same again with `VICTORY_COMPOSITE_GRADE_EDGE`'s clamp
//! of the composite's first grading lookup in front.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, StaticSampler, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// The immediate sampler of the colour-grading tables (PS_CompositeNMB:
/// clamp, linear, point mip), as `VICTORY_COMPOSITE_GRADE_EDGE` reviews it.
const GRADE_SAMPLER: [u32; 4] = [0x92, 0x00C0_0000, 0x1450_0000, 0];

/// Where the composite's first grading lookup is (`lower.rs`, the grade
/// edge clamp).
const GRADE_LOOKUP: usize = 0xefc;

/// The value a trilinear filter with clamping gives at the normalized
/// point `p` of an `n`^3 texture whose texel (x, y, z) holds (x, 10 y,
/// 100 z, 1). Infinite and not-a-number coordinates read the edge texel
/// (a not-a-number one the first), as the card's sampler does.
fn trilinear(n: u32, p: [f32; 3]) -> [f32; 4] {
    let at = |v: f32| {
        let t = v * n as f32 - 0.5;
        if t.is_nan() {
            0.0
        } else {
            t.clamp(0.0, (n - 1) as f32)
        }
    };
    let [x, y, z] = p.map(at);
    [x, 10.0 * y, 100.0 * z, 1.0]
}

fn texels(n: u32) -> Vec<[f32; 4]> {
    (0..n * n * n)
        .map(|i| {
            [
                (i % n) as f32,
                10.0 * (i / n % n) as f32,
                100.0 * (i / (n * n)) as f32,
                1.0,
            ]
        })
        .collect()
}

/// The program: the grading sampler, then for each point `image_sample_lz`
/// 3D of four channels there, stored; with `at` the first sample sits at
/// that byte of the code (`s_nop`s before it).
fn program(points: &[[f32; 3]], at: Option<usize>) -> NativeProgram {
    let mut words = vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF40C_0400,
        0xFA00_0000, // s_load_dwordx8 s[16:23], s[0:1], 0: the 3D SRV
        0xF408_0300,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 32: the output UAV
        0x7E1E_0280, // v_mov_b32 v15, 0: output byte offsets
    ];
    // s_mov_b32 s24..s27: the sampler words.
    for (k, w) in GRADE_SAMPLER.into_iter().enumerate() {
        words.extend([0xBE98_03FF + ((k as u32) << 16), w]);
    }
    for (i, p) in points.iter().enumerate() {
        for (reg, v) in p.iter().enumerate() {
            // v_mov_b32 v<reg>, literal
            words.extend([0x7E00_02FF | ((reg as u32) << 17), v.to_bits()]);
        }
        if i == 0
            && let Some(at) = at
        {
            assert!(4 * words.len() <= at, "the prologue is past {at:#x}");
            words.resize(at / 4, 0xBF80_0000); // s_nop 0
        }
        // image_sample_lz (0x27) 3D (dim 2), dmask xyzw, v[0:2] -> v[10:13],
        // resource s[16:23], sampler s[24:27].
        words.extend([0xF000_0F10 | (0x27 << 18), 0x00C4_0A00]);
        for ch in 0..4u32 {
            // buffer_store_dword v<10+ch>, v15, s[12:15], 0 offen offset:...
            words.extend([
                0xE070_1000 | ((i as u32 * 4 + ch) * 4),
                0x8003_000F | ((10 + ch) << 8),
            ]);
        }
    }
    words.push(0xBF81_0000); // s_endpgm
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: [(ResourceType::Srv, 0, 8), (ResourceType::Uav, 8, 4)]
            .map(|(kind, at, dwords)| UsedResource {
                kind,
                root: RootType::Table,
                table_offset_dwords: at,
                lower_bound: 0,
                stride: 0,
                dwords,
                user_data: 0,
                slot: 0,
            })
            .to_vec(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    }
}

/// `native` translated with `VICTORY_FILTER_3D` as `by_hand` and the grade
/// edge clamp as `edge`; the DXIL is validated when the validator is there.
fn translate(native: &NativeProgram, by_hand: bool, edge: bool) -> HostShader {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let was = super::lower::filter3d::TEST_FILTER_3D.replace(Some(by_hand));
    let lowered = lower::lower_with_grade_edge(
        &lower::Parts {
            native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        },
        edge,
    );
    super::lower::filter3d::TEST_FILTER_3D.set(was);
    let l = lowered.expect("translates");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        // (The tests' builder would leave the root signature's samplers out.)
        static_samplers: l
            .samplers
            .iter()
            .map(|&w| StaticSampler::from_words(w))
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
        half_views: l.half_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// Runs `shader` over an `n`^3 texture of [`texels`]: four floats a point.
fn run(host: &Host, shader: &HostShader, n: u32, points: usize) -> Vec<[f32; 4]> {
    let texels = texels(n);
    let (root, pso) = host.compute_pipeline(shader).unwrap();
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(
                    0,
                    Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                )],
                views: &[
                    View::Texture3D(n, n, n, &texels),
                    View::RawUav(4 * points as u32),
                ],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap();
    out[0]
        .chunks(4)
        .map(|c| [0, 1, 2, 3].map(|k| f32::from_bits(c[k])))
        .collect()
}

fn close(got: [f32; 4], want: [f32; 4]) -> bool {
    // The card's weights have 8 fractional bits.
    (0..4).all(|ch| (got[ch] - want[ch]).abs() <= 0.01 * (1.0 + want[ch].abs()))
}

/// The switch changes the program (eight reads and their weights for each
/// sample), and only with it on; the sampler stays bound either way.
#[test]
fn the_switch_takes_the_other_form() {
    let native = program(&[[0.3, 0.55, 0.8]], None);
    let card = translate(&native, false, false);
    let hand = translate(&native, true, false);
    assert_ne!(card.dxil, hand.dxil);
    assert!(hand.dxil.len() > card.dxil.len());
    assert_eq!(card.static_samplers, hand.static_samplers);
    assert_eq!(card.static_samplers[0].filter, 0x14, "linear, point mip");
    assert_eq!(card.static_samplers[0].address, [3, 3, 3], "clamp");
}

/// Both forms give the trilinear values, on WARP and on the card; the hand
/// form gives the edge texel for infinite and not-a-number coordinates
/// (the card's sampler is not asked: what it does with them is its own).
#[test]
fn by_hand_and_by_the_card_agree() {
    const N: u32 = 4;
    let finite = [
        [0.3f32, 0.55, 0.8],
        [0.125, 0.125, 0.125],
        [0.01, 0.99, 0.5],
        [0.7, 0.2, 0.33],
        [-0.4, 1.7, 0.0],
    ];
    let odd = [
        [f32::INFINITY, f32::NEG_INFINITY, 0.5],
        [f32::NAN, 0.5, f32::INFINITY],
    ];
    let points: Vec<[f32; 3]> = finite.iter().chain(&odd).copied().collect();
    let native = program(&points, None);
    let mut ran = 0;
    for (name, warp) in [("WARP", true), ("hardware", false)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {name} device; skipped");
            continue;
        };
        for by_hand in [false, true] {
            let got = run(&host, &translate(&native, by_hand, false), N, points.len());
            let checked = if by_hand { points.len() } else { finite.len() };
            for (i, p) in points.iter().enumerate().take(checked) {
                let want = trilinear(N, *p);
                assert!(
                    close(got[i], want),
                    "{name} by_hand={by_hand} at {p:?}: {:?} not {want:?}",
                    got[i]
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

/// `VICTORY_COMPOSITE_GRADE_EDGE`'s clamp (the composite's first grading
/// lookup, at machine offset 0xefc with the reviewed sampler, a 33^3 table)
/// and `VICTORY_FILTER_3D` compose: the clamp applies with its switch alone
/// (another program), and each of the four combinations gives the same
/// trilinear values with clamping, near and past the table's faces. With
/// both, the hand filter reads the clamped point and gives those values too
/// (the clamp moves no point to other texels or weights).
#[test]
fn the_grade_edge_clamp_and_the_hand_filter_compose() {
    const N: u32 = 33;
    let points = [
        [0.001f32, 0.5, 0.999],
        [0.0, 1.0, 0.25],
        [-0.2, 1.3, 0.6],
        [0.5 / 33.0, 32.5 / 33.0, 0.37],
        [0.62, 0.014, 0.986],
    ];
    let forms = [(false, false), (true, false), (false, true), (true, true)];
    // One program a point: only the lookup at 0xefc is the composite's.
    let shaders: Vec<[HostShader; 4]> = points
        .iter()
        .map(|p| {
            let native = program(&[*p], Some(GRADE_LOOKUP));
            forms.map(|(by_hand, edge)| translate(&native, by_hand, edge))
        })
        .collect();
    let s = &shaders[0];
    assert_ne!(s[0].dxil, s[2].dxil, "the edge clamp applies to the lookup");
    assert_ne!(s[1].dxil, s[3].dxil, "and in front of the hand filter");
    assert_ne!(s[0].dxil, s[1].dxil, "the hand filter applies");
    let mut ran = 0;
    for (name, warp) in [("WARP", true), ("hardware", false)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {name} device; skipped");
            continue;
        };
        for (p, four) in points.iter().zip(&shaders) {
            let want = trilinear(N, *p);
            for ((by_hand, edge), shader) in forms.iter().zip(four) {
                let got = run(&host, shader, N, 1)[0];
                assert!(
                    close(got, want),
                    "{name} by_hand={by_hand} edge={edge} at {p:?}: {got:?} not {want:?}"
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}
