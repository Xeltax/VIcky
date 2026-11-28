//! 16-bit halves moved through a table's typed buffer views as they are
//! (`VICTORY_FACE_D16=1`, opt-in): port of fork pull request #927 (head
//! d4b786fe1d, merged as 58de9bf4c; the NVIDIA tree's 77e13a974 has the
//! same code). The title keeps half floats, and indices beside them, in
//! buffers it views as `R16G16B16A16_UINT` and `R16_UINT`, and moves them
//! with `buffer_load_format_d16_*` / `buffer_store_format_d16_*`: on the
//! console each channel's 16 bits go as they are. Lowered by way of 32-bit
//! floats (load as float, `legacyF32ToF16`) through the title's integer view,
//! every value read came out 0 on NVIDIA: the network that drives the faces
//! (`compute_graph_ops`, then `ped_standard_head_cs`) computed zeros and no
//! mouth moved. With the switch, such an access moves the 16 bits as
//! unsigned integers and names its table place
//! ([`super::super::HostShader::half_views`]); the host shows a 16-bit float
//! or signed buffer view there as the unsigned twin of the same layout.
//! A format load of whole dwords whose result the program reads only one
//! half of, as a half float, moves 16-bit integers too (the head program's
//! reads of the network's output). Also the d16 channel packing of buffer
//! accesses, a child module of the emitter so `lower.rs` stays under the
//! repository's file size limit.

use std::collections::{BTreeSet, HashSet};

use super::{
    Attr, Bin, Fields, Inst, Lower, Operand, Reg, Role, RootType, Site, Source, V, Value, Why,
};

/// `VICTORY_FACE_D16=1`: 16-bit halves moved through a table's typed buffer
/// views keep their bits (see the module). Off by default until a picture
/// A/B confirms it on this tree. Part of the shader cache key (a switch
/// this crate reads).
pub(super) fn face_d16_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_FACE_D16.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FACE_D16").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`face_d16_on`], for its thread only.
    pub(in crate::emit) static TEST_FACE_D16: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The table places a program moves halves through.
#[derive(Default)]
pub(super) struct Halves {
    /// Named for the host ([`super::super::HostShader::half_views`]): root
    /// parameter, dword offset in the table.
    pub(super) named: BTreeSet<(u8, u32)>,
    /// [`Lower::places_read_as_halves`], found before the code is lowered.
    pub(super) only: BTreeSet<(u8, u32)>,
}

impl<'a> Lower<'a> {
    /// Whether every use of `value`, a typed buffer load's result, reads
    /// one 16-bit half of it as a half float: a `v_fma_mix*` source with
    /// its `op_sel_hi` bit set, or `v_cvt_f32_f16`. A program that takes a
    /// format load apart like that reads a 16-bit integer view on the
    /// console (the load gives the zero-extended integer, whose bits are a
    /// half; a float view's value it would not take apart). A use is a
    /// read as an operand ([`crate::ir::operand_reads`]): a later write of
    /// the register reads the value only to keep it in the lanes EXEC
    /// leaves alone, and a phi carries it on, its own uses counting. False
    /// without a use.
    pub(super) fn read_only_as_halves(&self, value: Value) -> bool {
        let p = self.x.p;
        let mut used = false;
        let mut seen = HashSet::from([value]);
        let mut todo = vec![value];
        while let Some(value) = todo.pop() {
            for b in &p.blocks {
                for phi in b.phis.iter().filter(|phi| phi.args.contains(&value)) {
                    if seen.insert(phi.value) {
                        todo.push(phi.value);
                    }
                }
                for i in &b.insts {
                    if !i.reads.iter().any(|r| r.1 == value) {
                        continue;
                    }
                    let operands = crate::ir::operand_reads(p, i);
                    let regs: Vec<Reg> = i
                        .reads
                        .iter()
                        .filter(|r| r.1 == value && operands.contains(&r.0))
                        .map(|r| r.0)
                        .collect();
                    if regs.is_empty() {
                        continue;
                    }
                    used = true;
                    let Fields::Valu(v) = &i.fields else {
                        return false;
                    };
                    match i.name() {
                        "v_cvt_f32_f16" => {}
                        "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                            let whole = v.src.iter().enumerate().any(|(k, op)| {
                                matches!(*op, Operand::Vector { reg, .. } if regs.contains(&Reg::V(reg)))
                                    && v.opsel_hi >> k & 1 == 0
                            });
                            if whole {
                                return false;
                            }
                        }
                        _ => return false,
                    }
                }
            }
        }
        used
    }

    /// Whether `inst` is a format load of whole dwords (not d16) whose
    /// every channel is read only as halves ([`Self::read_only_as_halves`]).
    pub(super) fn format_load_read_as_halves(&self, inst: &Inst) -> bool {
        let Fields::Buffer {
            data: Some(Operand::Vector { reg, .. }),
            ..
        } = inst.fields
        else {
            return false;
        };
        let name = inst.name();
        let count = match name
            .strip_prefix("buffer_load_format_")
            .or(name.strip_prefix("tbuffer_load_format_"))
        {
            Some("x") => 1,
            Some("xy") => 2,
            Some("xyz") => 3,
            Some("xyzw") => 4,
            _ => return false,
        };
        (0..count).all(|k| {
            inst.write(Reg::V(reg + k))
                .is_some_and(|v| self.read_only_as_halves(v))
        })
    }

    /// The table places ([`Self::table_place`]) every access through which
    /// moves halves: a d16 access, or a format load read only as halves
    /// ([`Self::format_load_read_as_halves`]). A place the program reaches
    /// otherwise too is left out (a format load of numbers beside one of
    /// halves, a raw or scalar load, a descriptor chosen by control flow),
    /// and so is every place of a table the program also picks from at run
    /// time: what the program makes of such a view is not halves alone.
    /// None without `VICTORY_FACE_D16=1` ([`face_d16_on`]): every access
    /// then takes the float path, as before this port.
    pub(super) fn places_read_as_halves(&self) -> BTreeSet<(u8, u32)> {
        let mut halves = BTreeSet::new();
        if !face_d16_on() {
            return halves;
        }
        let mut other = BTreeSet::new();
        let mut picked = HashSet::new();
        for d in &self.x.res.descriptors {
            if d.role == Role::Sampler {
                continue;
            }
            let inst = self
                .x
                .p
                .blocks
                .get(d.site.0)
                .and_then(|b| b.insts.get(d.site.1));
            let half = matches!(d.source, Source::Loaded { .. })
                && inst.is_some_and(|i| {
                    (i.name().starts_with("buffer_") && i.name().contains("_format_d16_"))
                        || self.format_load_read_as_halves(i)
                });
            for leaf in d.source.leaves() {
                let Source::Loaded { load, .. } = leaf else {
                    continue;
                };
                let Some(a) = self.scalar.get(load) else {
                    continue;
                };
                if a.dynamic.is_some() {
                    if let Some(rp) = self.x.native.layout.get(a.param)
                        && rp.root == RootType::Table
                    {
                        picked.insert(rp.slot);
                    }
                } else if let Some(place) = self.table_place(leaf) {
                    if half {
                        halves.insert(place);
                    } else {
                        other.insert(place);
                    }
                }
            }
        }
        halves.retain(|p| !other.contains(p) && !picked.contains(&p.0));
        halves
    }

    /// The table descriptor `src` is, as (root parameter, dword in the
    /// table), when the code loads it from a fixed place in a table: the
    /// load's own place, not the start of the range its used-resource
    /// entry describes (a range of several descriptors starts before most
    /// of them). `None` for a descriptor picked at run time, built in the
    /// code or chosen by control flow.
    pub(super) fn table_place(&self, src: &Source) -> Option<(u8, u32)> {
        let Source::Loaded { load, first } = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        (rp.root == RootType::Table && a.dynamic.is_none())
            .then(|| (rp.slot, a.offset / 4 + u32::from(*first)))
    }

    /// The component type of a typed buffer handle for the descriptor
    /// loaded at `site` (dwords `first..`) and accessed by `ui`, where the
    /// code reads the descriptor as data: unsigned integers (0x405) where
    /// the access moves halves, floats (0x409) otherwise.
    pub(super) fn typed_component(&self, site: Site, first: u8, ui: &Inst) -> u32 {
        let halves = self
            .table_place(&Source::Loaded { load: site, first })
            .is_some_and(|p| {
                self.half_places.only.contains(&p)
                    && (ui.name().contains("_format_d16_") || self.format_load_read_as_halves(ui))
            });
        if halves { 0x405 } else { 0x409 }
    }

    /// Whether the typed buffer access `inst` through `src` moves halves:
    /// sixteen bits a channel through a view of a table go as they are, as
    /// unsigned integers, and the host is told (the place is named), which
    /// shows the shader a 16-bit float view there as its unsigned twin. A
    /// format load of whole dwords whose every use reads one half of a
    /// result as a half float moves 16-bit integers too, where every
    /// access through its view does ([`Self::places_read_as_halves`]). A
    /// d16 access too only where every access does: a whole float read
    /// through the same place would see the unsigned twin.
    pub(super) fn moves_halves(&mut self, src: &Source, inst: &Inst, d16: bool) -> bool {
        let place = self.table_place(src).filter(|p| {
            self.half_places.only.contains(p) && (d16 || self.format_load_read_as_halves(inst))
        });
        if let Some(place) = place {
            self.half_places.named.insert(place);
        }
        place.is_some()
    }

    /// A typed buffer load of element `index` through `src`: four unsigned
    /// integers where it moves halves (`halves`), four floats otherwise.
    pub(super) fn typed_load(
        &mut self,
        site: Site,
        src: &Source,
        u: u32,
        index: V,
        halves: bool,
    ) -> Result<V, Why> {
        let c = self.cur;
        let iu = self.s.m.undef(self.s.t.i32);
        let (component, name, ret) = if halves {
            let i32t = self.s.t.i32;
            let ret = self.s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);
            (0x405, "bufferLoad.i32", ret)
        } else {
            (0x409, "bufferLoad.f32", self.s.t.res_ret)
        };
        let h = self.handle(site, src, false, (10 | u, component))?;
        Ok(self.s.op(c, name, Attr::ReadOnly, ret, 68, &[h, index, iu]))
    }

    /// The `count` channels a d16 store writes, from `regs` (two per VGPR,
    /// low half first), as the typed store takes them: through a view of a
    /// table that moves halves (`halves`) each half's bits as they are.
    /// Reached another way:
    /// shortcut: d16 data is taken as f16 (integer formats would keep 16
    /// bits), widened for the 32-bit float store.
    pub(super) fn d16_channels(&mut self, regs: &[V], count: u32, halves: bool) -> Vec<V> {
        let c = self.cur;
        let mut wide = Vec::new();
        for k in 0..count {
            let mut x = regs[(k / 2) as usize];
            if k % 2 == 1 {
                let k16 = self.s.m.i32(16);
                x = self.s.m.bin(c, Bin::LShr, x, k16);
            }
            if halves {
                let low = self.s.m.i32(0xffff);
                wide.push(self.s.m.bin(c, Bin::And, x, low));
                continue;
            }
            let f32t = self.s.t.f32;
            let f = self
                .s
                .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
            wide.push(self.as_int(f));
        }
        wide
    }

    /// Writes the VGPRs from `reg` on that a buffer load of `count` channels
    /// gives from its result `r` (under `guard`): with `d16`, two 16-bit
    /// channels per VGPR (an odd last high half 0), each a float's half
    /// (`legacyF32ToF16`) or, where the load moves halves (`halves`), the 16
    /// bits of each as they are; else one channel a VGPR (integers already
    /// where the load moved halves). Each VGPR is written as it is made, in
    /// the order the emitter had before this module.
    pub(super) fn buffer_result(
        &mut self,
        inst: &Inst,
        reg: u8,
        r: V,
        count: u32,
        (d16, halves): (bool, bool),
        guard: Option<V>,
    ) {
        let c = self.cur;
        if !d16 {
            for k in 0..count {
                let x = self.s.m.extract(c, r, k);
                let x = if halves { x } else { self.as_int(x) };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return;
        }
        let k16 = self.s.m.i32(16);
        let i32t = self.s.t.i32;
        for k in 0..count.div_ceil(2) {
            let mut x = None;
            for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                let f = self.s.m.extract(c, r, ch);
                let h = if halves {
                    let low = self.s.m.i32(0xffff);
                    self.s.m.bin(c, Bin::And, f, low)
                } else {
                    self.s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
                };
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
    }
}
