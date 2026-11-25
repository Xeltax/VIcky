//! A descriptor's dwords read as numbers ([`Lower::descriptor_data`]): the
//! console compiler computes sizes from descriptor bits (an image's width,
//! height and levels; a buffer's record count) instead of querying them,
//! and takes a buffer's address out of its descriptor to read the memory
//! behind it. The host's descriptors are not the console's, so those
//! fields are rebuilt the way the translator builds the resource itself:
//! from the host view at the same heap slot (its dimensions), and from the
//! title addresses the host keeps (`shader.rs` `USER_VIEW_ADDRESS`,
//! [`super::super::shader::SLOT_VIEW_ADDRESSES`]).
//!
//! A child module of the emitter (it shares its state), in a file of its own
//! so that `lower.rs` stays under the asset policy's file limit.

use super::{Attr, Bin, Inst, Lower, Reg, Tag, V, Val, Why};
use crate::ir::Value;
use crate::native::{ResourceType, RootType, UsedResource};
use crate::resource::{Site, Source};

/// The refusal of a descriptor load whose dwords are read as numbers and
/// whose kind nothing tells.
const ONLY_DATA: Why = "a descriptor read only as data";
/// What a dword the host has no value for reads as.
pub(super) const OTHER: Why = "a descriptor's address or format read as data";

/// Which buffer address dwords (0 and 1) [`Lower::view_words`] gives, where
/// the host can tell them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum AddressWords {
    /// None.
    No,
    /// Where the code computes with them.
    Computed,
    /// Whether the code computes with them or not: a descriptor nothing
    /// uses as one (`Lower::unused_descriptor_kind`), as a buffer whose
    /// address alone is read.
    Always,
}

impl<'a> Lower<'a> {
    /// The dwords, by place in the descriptor, that the host view the
    /// descriptor loaded at `site` names gives (`props` its kind, `image`
    /// the image kind, `first` the load's dword it starts at), rebuilt from
    /// the view's dimensions in the GFX10 layout: image dword 1 bits 30-31
    /// and dword 2 bits 0-13 width - 1, dword 2 bits 14-29 height - 1,
    /// dword 3 identity swizzle, last level at 16, type at 28, dword 4
    /// depth or layers - 1; buffer dword 2 the record count, and dwords 0
    /// and 1 as `address` says, where the host can tell them.
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn view_words(
        &mut self,
        inst: &Inst,
        site: Site,
        (code, first): (u8, u8),
        src: &Source,
        props: (u32, u32),
        image: Option<u32>,
        uav: bool,
        used: Option<&'a UsedResource>,
        address: AddressWords,
    ) -> Result<Vec<(u32, V)>, Why> {
        let h = self.handle(site, src, false, props)?;
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
        Ok(match image {
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
                // Its address too, as `address` says, where the host can
                // tell it.
                let computed = (0..2u8).any(|k| {
                    inst.write(Reg::S(code + first + k))
                        .is_some_and(|v| self.computed_with(v, 6))
                });
                let wanted = match address {
                    AddressWords::No => false,
                    AddressWords::Computed => computed,
                    AddressWords::Always => true,
                };
                if wanted
                    && first == 0
                    && let Some((lo, hi)) = self.view_address(site)
                {
                    words.extend([(0, lo), (1, hi)]);
                }
                words
            }
        })
    }

    /// Writes `words` (by place in the descriptor that starts at dword
    /// `first` of the `count` the load writes from SGPR `code`); the other
    /// dwords have no value (`why` says which), and where they meet
    /// numbers at a join the register is dead on that path.
    pub(super) fn write_words(
        &mut self,
        inst: &Inst,
        (code, first): (u8, u8),
        count: u32,
        words: &[(u32, V)],
        why: Why,
    ) {
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(why));
                        self.stale.insert(v);
                    }
                }
            }
        }
    }

    /// A descriptor load (`count` dwords into SGPRs from `code`) that no
    /// instruction uses as a descriptor: the code reads its dwords as
    /// numbers only (an image it only measures, a buffer whose record count
    /// it tests or whose address it walks), and that
    /// `Lower::unused_descriptor_kind` gave no kind (it takes eight dwords
    /// with a size read for an image, four with the record count read for a
    /// buffer). The table's range tells what
    /// the slot holds; the resource heap names no ranges, and the console
    /// compiler loads an image's descriptor whole (eight dwords) and a
    /// buffer's whole (four). Only dwords the code reads are given a value:
    /// a load nothing reads stays as it was.
    ///
    /// - An image's dwords 1-3 (width, height, levels) as for one in use
    ///   ([`Lower::view_words`]).
    /// - A buffer's address (dwords 0 and 1): a table entry at a fixed
    ///   offset asks the host when it binds (`view_address`); an entry of the
    ///   resource heap whose low dword is read reads the host's table of view
    ///   addresses by slot ([`Lower::heap_view_address`]). Its record count
    ///   (dword 2) from a table entry, from the view's size and the range's
    ///   stride.
    ///
    /// The other dwords have no value: reading them stops the program.
    // shortcut: an image nothing uses is measured as a 2D one (an array, 3D
    // or cube image gives its width and height all the same; its depth is
    // not given); one of the heap as a shader resource, not an unordered
    // one. A buffer is measured as a raw view (the host makes structured
    // views raw).
    pub(super) fn unused_descriptor(
        &mut self,
        inst: &Inst,
        site: Site,
        code: u8,
        count: u32,
    ) -> Result<(), Why> {
        let a = self.scalar.get(&site).copied();
        let root = a
            .and_then(|a| self.x.native.layout.get(a.param))
            .map(|p| p.root);
        let heap = root == Some(RootType::ResourceHeap);
        let src = Source::Loaded {
            load: site,
            first: 0,
        };
        let used = self.used_kind(&src);
        // Whether the slot holds an image (else a buffer).
        let image = match used {
            Some(u) if u.kind == ResourceType::Cbv => {
                self.stale.extend(inst.writes.iter().map(|w| w.1));
                return Err("a constant buffer's descriptor read as data");
            }
            Some(u) if matches!(u.kind, ResourceType::Srv | ResourceType::Uav) => {
                Some(u.dwords == 8)
            }
            None if heap && count == 8 => Some(true),
            None if heap && count == 4 => Some(false),
            _ => None,
        };
        // The load starts at a descriptor: slots are 32 bytes (a computed
        // term counts whole slots).
        let whole = a.is_some_and(|a| a.offset % 32 == 0)
            && matches!(root, Some(RootType::Table | RootType::ResourceHeap));
        let Some(image) = image.filter(|_| whole) else {
            if std::env::var_os("VICTORY_DESC_WHY").is_some() {
                eprintln!(
                    "DESC-WHY {:#x} {} scalar={a:?} lane_copied={}",
                    inst.machine.offset,
                    inst.name(),
                    self.x.res.lane_copied.contains(&site)
                );
            }
            return Err(ONLY_DATA);
        };
        let read = |l: &Self, k: u32| {
            k < count
                && inst
                    .write(Reg::S(code + k as u8))
                    .is_some_and(|v| l.read_anywhere(v))
        };
        let uav = used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let mut words = Vec::new();
        if image {
            if (1..4).any(|k| read(self, k)) {
                words = self.view_words(
                    inst,
                    site,
                    (code, 0),
                    &src,
                    (2 | u, 0x409),
                    Some(2),
                    uav,
                    used,
                    AddressWords::No,
                )?;
                words.retain(|w| (1..4).contains(&w.0));
            }
        } else {
            if used.is_some() && read(self, 2) {
                words = self.view_words(
                    inst,
                    site,
                    (code, 0),
                    &src,
                    (11 | u, 0),
                    None,
                    uav,
                    used,
                    AddressWords::No,
                )?;
            }
            // The address: a table entry's is asked for by a load of at most
            // four dwords whether read or not (as before this file); a heap
            // entry's only where its low dword is read, since four dwords
            // of the heap read for their second alone may be an image's
            // (its width's low bits), which the table has no value for.
            let address = if heap {
                if read(self, 0) {
                    Some(self.heap_view_address(site)?)
                } else {
                    None
                }
            } else if count <= 4 || read(self, 0) || read(self, 1) {
                self.view_address(site)
            } else {
                None
            };
            if let Some((lo, hi)) = address {
                words.extend([(0, lo), (1, hi)]);
            }
        }
        self.write_words(inst, (code, 0), count, &words, OTHER);
        Ok(())
    }

    /// The title address (low dword, bits from 32 up) of the buffer view
    /// the descriptor load at `site` reads from the resource heap, at an
    /// index the code computes: from the host's table of view addresses by
    /// title slot ([`super::super::shader::SLOT_VIEW_ADDRESSES`]), read in
    /// the current block. (The console's descriptor holds more in the
    /// second dword, the stride from bit 16 up; code that takes the address
    /// masks those away.)
    pub(super) fn heap_view_address(&mut self, site: Site) -> Result<(V, V), Why> {
        use super::super::shader::{SLOT_STRIDES_COUNT, SLOT_VIEW_ADDRESSES};
        let c = self.cur;
        // The table is by title slot: the index before any heap map.
        let (slot, _) = self.title_heap_index(site, 0, false)?;
        // Inside the table whatever the index (lanes that do not run
        // compute one too): a root view has no bounds of its own.
        let i32t = self.s.t.i32;
        let last = self.s.m.i32(SLOT_STRIDES_COUNT - 1);
        let slot = self
            .s
            .op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[slot, last]);
        self.s.m.tag(slot, Tag::Clamp);
        let three = self.s.m.i32(3);
        let byte = self.s.m.bin(c, Bin::Shl, slot, three);
        let h = self.s.root_srv_handle(SLOT_VIEW_ADDRESSES);
        let (iu, both, align) = (self.s.m.undef(i32t), self.s.m.i8(3), self.s.m.i32(4));
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, byte, iu, both, align],
        );
        let (lo, hi) = (self.s.m.extract(c, r, 0), self.s.m.extract(c, r, 1));
        Ok((self.as_int(lo), self.as_int(hi)))
    }

    /// Whether any instruction or join reads `v`.
    fn read_anywhere(&self, v: Value) -> bool {
        self.x.p.blocks.iter().any(|b| {
            b.phis.iter().any(|p| p.args.contains(&v))
                || b.insts.iter().any(|i| i.reads.iter().any(|r| r.1 == v))
        })
    }
}
