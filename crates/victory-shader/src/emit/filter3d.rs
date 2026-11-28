//! `VICTORY_FILTER_3D=1` (opt-in): a level-0 sample of a 3D texture with
//! an immediate sampler that filters linearly and clamps is done in the
//! shader. Port of readmet3xt-lab d246d1080 ("colour-grading tables
//! filtered in the shader (experiment)"); read only, nothing from that tree
//! was run. A child module of the emitter so `lower.rs` stays under the
//! repository's file size limit.
//!
//! The composite pass (`PS_CompositeNMB`, `6318f669e4f17d65`) posterized
//! light in some runs there: every pixel of its output was exactly one entry
//! of its first 33^3 grading table run through the 64^3 ACES table, i.e. the
//! first table was read unfiltered (steps of 0.44 stops, green and magenta
//! where the channels step apart), with every static part right (sampler
//! linear, a Texture3D of the 33^3 RGBA16F table, smooth contents). With
//! the switch the eight texels around the point are read as they are
//! (`textureLoad`) and weighted by distance, taking the card's sampler out
//! of the path.
//!
//! With `VICTORY_COMPOSITE_GRADE_EDGE=1` as well, the same first lookup
//! (machine offset 0xefc of that shader, the same immediate sampler) has
//! its coordinates clamped to the edge texel centres first; the filter here
//! then reads those coordinates. The two compose: the hand filter clamps
//! the point to the first and last texel centres itself (as the sampler's
//! clamp does), so a point the edge clamp moved gives the same texels and
//! weights either way, and with both switches the lookup is the hand
//! filter's (the edge clamp adds nothing). The 64^3 ACES lookup after it
//! has the same sampler, so the switch filters it by hand too; the edge
//! clamp leaves it alone.

use super::{Attr, Bin, Cast, Lower, Source, V};

/// `VICTORY_FILTER_3D=1`: see the module. Off by default: an experiment
/// until a capture of the posterized sky says it is the cause. Part of the
/// shader cache key (a switch this crate reads).
pub(super) fn filter_3d_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_FILTER_3D.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILTER_3D").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`filter_3d_on`], for its thread only.
    pub(in crate::emit) static TEST_FILTER_3D: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether immediate sampler words filter linearly in both directions
/// (any mip filter) without anisotropy, comparison or min/max reduction,
/// and clamp on all three axes: what [`filter_3d_on`] does by hand.
fn linear_clamp_sampler(words: &[u32]) -> bool {
    let Some(&[w0, w1, w2, w3]) = words.get(..4) else {
        return false;
    };
    let s = super::super::sampler::StaticSampler::from_words([w0, w1, w2, w3]);
    s.filter & !1 == 0x14 && s.address == [3, 3, 3]
}

impl Lower<'_> {
    /// Whether the sample `name` of a 3D texture with sampler `ss` is done
    /// by hand ([`filter_3d_on`]): `image_sample_lz` with an immediate
    /// linear clamping sampler.
    pub(super) fn filters_3d(&self, name: &str, ss: &Source) -> bool {
        name == "image_sample_lz"
            && filter_3d_on()
            && matches!(ss, Source::Constant(words) if linear_clamp_sampler(words))
    }

    /// Trilinear by hand at level 0 of the 3D texture `h` at the normalized
    /// point `cs[..3]` ([`filter_3d_on`]): the texel centres around the
    /// point, their indices clamped to the texture (the sampler clamps),
    /// weighted by the point's place between them. The four filtered
    /// channels, and one of the reads to stand for the sample's result.
    pub(super) fn filter_3d(&mut self, h: V, cs: &[V]) -> ([V; 4], V) {
        // (When it is translated: a cached translation says nothing.)
        let hash = super::super::per_program::current();
        eprintln!(
            "FILTER-3D #{} a 3D lookup filtered in the shader (VICTORY_FILTER_3D)",
            hash.as_deref().unwrap_or("?")
        );
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let iu = self.s.m.undef(i32t);
        let fu = self.s.m.undef(f32t);
        let level = self.s.m.i32(0);
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, level]);
        let half = self.s.m.f32(0.5f32.to_bits());
        let one = self.s.m.f32(1.0f32.to_bits());
        let zero = self.s.m.f32(0);
        let (mut lo, mut hi, mut part) = ([iu; 3], [iu; 3], [fu; 3]);
        for k in 0..3 {
            let n = self.s.m.extract(c, d, k as u32);
            let n = self.s.m.cast(c, Cast::UiToFp, n, f32t);
            let last = self.s.m.bin(c, Bin::Sub, n, one);
            let t = self.s.m.bin(c, Bin::Mul, cs[k], n);
            let t = self.s.m.bin(c, Bin::Sub, t, half);
            // The place between texel centres clamped to the first and the
            // last centre (ours; d246d1080 clamps the indices alone): the
            // same weights for every finite point, as the sampler's clamp
            // gives, and an infinite or not-a-number coordinate reads the
            // edge texel as the card's sampler does (FMax and FMin return
            // the number beside a NaN) instead of weighting by a NaN.
            let t = self.binary(35, t, zero); // maximum
            let t = self.binary(36, t, last); // minimum
            let first = self.unary(27, t); // round down
            part[k] = self.s.m.bin(c, Bin::Sub, t, first);
            let next = self.s.m.bin(c, Bin::Add, first, one);
            let next = self.binary(36, next, last); // minimum
            lo[k] = self.s.m.cast(c, Cast::FpToSi, first, i32t);
            hi[k] = self.s.m.cast(c, Cast::FpToSi, next, i32t);
        }
        // The eight texels, x fastest.
        let mut texels = Vec::with_capacity(8);
        for n in 0..8 {
            let pick = |k: usize| if n >> k & 1 == 0 { lo[k] } else { hi[k] };
            let r = self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &[h, level, pick(0), pick(1), pick(2), iu, iu, iu],
            );
            texels.push(r);
        }
        let lerp = |l: &mut Self, a: V, b: V, t: V| {
            let d = l.s.m.bin(c, Bin::Sub, b, a);
            let d = l.s.m.bin(c, Bin::Mul, d, t);
            l.s.m.bin(c, Bin::Add, a, d)
        };
        let mut g = [fu; 4];
        for (ch, slot) in g.iter_mut().enumerate() {
            let mut v: Vec<V> = texels
                .iter()
                .map(|&r| self.s.m.extract(c, r, ch as u32))
                .collect();
            // Along x, then y, then z.
            for (k, &t) in part.iter().enumerate() {
                let step = 1 << k;
                for i in (0..8).step_by(2 * step) {
                    v[i] = lerp(self, v[i], v[i + step], t);
                }
            }
            *slot = v[0];
        }
        (g, texels[0])
    }
}
