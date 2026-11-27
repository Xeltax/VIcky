//! A diagnostic of the clean-up passes ([`super::opt`]): the module before
//! them and after them run side by side on made-up threads, and what each
//! one does to the world (stores, atomics, barriers, outputs, discards, its
//! end) is compared. A difference is a pass that changed what the program
//! does.
//!
//! One thread is run, not a wave: what a pass may get wrong across lanes
//! (which lanes a wave operation sees) is not seen here. Within the thread
//! the run assumes no more than the passes may: every call without an
//! effect gives the same result for the same arguments, a read of memory
//! gives the same result until something with an effect runs (another
//! thread's store may land at a barrier or any other call with an effect),
//! and shared memory holds what this thread stored until then.
//!
//! `VICTORY_SHADER_OPT_CHECK=<n>` runs each translated program on `n`
//! threads (seeds) and writes one `SHADER-OPT-CHECK` line per program to
//! standard error: `differs` and where, or `same` and how the runs ended
//! (with the program's hash: nothing of the program itself). Meant for
//! `victory translate`.

use std::collections::HashMap;

use super::opt::{Effect, effect};
use super::{Bin, Cast, Const, Module, Op, Rmw, Term, Ty, TypeDef, V, pred};

/// Block visits after which a run stops (both runs stop at the same visit:
/// the passes leave every block where it was).
const VISITS: usize = 20_000;

/// A value: one scalar's bits, or the fields of a structure.
#[derive(Debug, Clone, PartialEq)]
enum Val {
    S(u64),
    Agg(Vec<u64>),
}

impl Val {
    fn bits(&self) -> u64 {
        match self {
            Val::S(x) => *x,
            Val::Agg(f) => f.iter().fold(0x51_7cc1_b727_220a_95, |h, &x| mix(h ^ x)),
        }
    }
}

/// One thing a run did to the world: what (a function's name, `store`,
/// `ret`...) and with which values.
pub(crate) type Effects = Vec<(String, Vec<u64>)>;

/// SplitMix64's finaliser.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn hash_str(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 { !0 } else { (1 << bits) - 1 }
}

fn signed(bits: u32, x: u64) -> i64 {
    if bits < 64 && x >> (bits - 1) & 1 != 0 {
        (x | !mask(bits)) as i64
    } else {
        x as i64
    }
}

/// Whether float predicate `p` holds for `x` and `y` (bits 0-2: equal,
/// greater, less; bit 3: or unordered).
fn holds_f(p: u8, x: f64, y: f64) -> bool {
    if x.is_nan() || y.is_nan() {
        p & 8 != 0
    } else {
        (p & 1 != 0 && x == y) || (p & 2 != 0 && x > y) || (p & 4 != 0 && x < y)
    }
}

fn holds_i(p: u8, bits: u32, x: u64, y: u64) -> bool {
    let (sx, sy) = (signed(bits, x), signed(bits, y));
    match p {
        pred::EQ => x == y,
        pred::NE => x != y,
        pred::IUGT => x > y,
        pred::IUGE => x >= y,
        pred::IULT => x < y,
        pred::IULE => x <= y,
        pred::SGT => sx > sy,
        pred::SGE => sx >= sy,
        pred::SLT => sx < sy,
        _ => sx <= sy,
    }
}

/// One run of a module.
struct Run<'a> {
    m: &'a Module,
    seed: u64,
    vals: Vec<Option<Val>>,
    effects: Effects,
    /// Bumped by every instruction with an effect: memory reads made
    /// before give no word on reads after.
    epoch: u64,
    /// Shared memory this thread stored since the last effect.
    lds: HashMap<u64, u64>,
}

impl Run<'_> {
    fn bits_of(&self, ty: Ty) -> u32 {
        match &self.m.types[ty.0 as usize] {
            TypeDef::Int(b) => *b,
            TypeDef::Double => 64,
            TypeDef::Ptr(..) => 64,
            _ => 32,
        }
    }

    /// A made-up value of type `ty` from the hash `h`: small integers and
    /// numbers near 1 as often as wide ones, so comparisons go both ways.
    fn make(&self, ty: Ty, h: u64) -> Val {
        let h = mix(h ^ self.seed);
        match &self.m.types[ty.0 as usize] {
            TypeDef::Int(1) => Val::S(h & 1),
            TypeDef::Int(b) => Val::S(
                match h % 4 {
                    0 => (h >> 8) % 8,
                    1 => (h >> 8) % 300,
                    _ => h >> 3,
                } & mask(*b),
            ),
            TypeDef::Float => {
                let f = match h % 8 {
                    0 => [0.0f32, 1.0, -1.0, 0.5, 2.0, -0.0, 4.0, 0.25][(h >> 8) as usize % 8],
                    _ => ((h >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0) as f32,
                };
                Val::S(u64::from(f.to_bits()))
            }
            TypeDef::Double => {
                Val::S(((h >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0).to_bits())
            }
            TypeDef::Struct(_, fields) => Val::Agg(
                fields
                    .iter()
                    .enumerate()
                    .map(|(n, &f)| self.make(f, mix(h ^ (n as u64 + 1))).bits())
                    .collect(),
            ),
            _ => Val::S(0),
        }
    }

    fn get(&self, v: V) -> Result<Val, String> {
        Ok(match v {
            V::Inst(i) => self.vals[i as usize]
                .clone()
                .ok_or_else(|| format!("instruction {i} read before it ran"))?,
            V::Const(c) => self.constant(c),
            // Each global at its own address.
            V::Global(g) => Val::S((u64::from(g) + 1) << 40),
            V::Func(_) => Val::S(0),
        })
    }

    fn constant(&self, c: u32) -> Val {
        let (ty, k) = &self.m.consts[c as usize];
        match k {
            Const::Int(x) => Val::S(*x as u64 & mask(self.bits_of(*ty))),
            Const::Float(bits) => Val::S(u64::from(*bits)),
            Const::Undef => Val::S(0),
            Const::Aggregate(fields) => {
                Val::Agg(fields.iter().map(|&f| self.constant(f).bits()).collect())
            }
        }
    }

    fn scalar(&self, v: V) -> Result<u64, String> {
        Ok(self.get(v)?.bits())
    }

    fn size(&self, ty: Ty) -> u64 {
        match &self.m.types[ty.0 as usize] {
            TypeDef::Int(b) => u64::from(*b).div_ceil(8),
            TypeDef::Float => 4,
            TypeDef::Double | TypeDef::Ptr(..) => 8,
            TypeDef::Array(n, e) => u64::from(*n) * self.size(*e),
            TypeDef::Struct(_, f) => f.iter().map(|&t| self.size(t)).sum(),
            _ => 0,
        }
    }

    /// Something with an effect ran: memory may hold anything new.
    fn effect(&mut self, what: String, args: Vec<u64>) {
        self.effects.push((what, args));
        self.epoch += 1;
        self.lds.clear();
    }

    fn load(&self, addr: u64) -> u64 {
        // A constant table (a global with an initialiser) reads the same
        // whatever runs; shared memory what this thread stored, else what
        // the epoch holds.
        let g = (addr >> 40).wrapping_sub(1) as usize;
        if self.m.globals.get(g).is_some_and(|g| g.init.is_some()) {
            return mix(addr ^ 0xc0de);
        }
        match self.lds.get(&addr) {
            Some(&x) => x,
            None => mix(addr ^ mix(self.epoch ^ 0x1d5)) & 0xffff_ffff,
        }
    }

    fn bin(&self, ty: Ty, op: Bin, x: u64, y: u64) -> u64 {
        match self.m.types[ty.0 as usize] {
            TypeDef::Int(bits) => {
                let (x, y) = (x & mask(bits), y & mask(bits));
                let r = match op {
                    Bin::Add => x.wrapping_add(y),
                    Bin::Sub => x.wrapping_sub(y),
                    Bin::Mul => x.wrapping_mul(y),
                    Bin::UDiv => x.checked_div(y).unwrap_or(0),
                    Bin::URem => x.checked_rem(y).unwrap_or(0),
                    Bin::SDiv => signed(bits, x).checked_div(signed(bits, y)).unwrap_or(0) as u64,
                    Bin::And => x & y,
                    Bin::Or => x | y,
                    Bin::Xor => x ^ y,
                    // A shift by the width or more has no defined result:
                    // any made-up one, the same in both runs.
                    Bin::Shl if y >= u64::from(bits) => 0x5eed,
                    Bin::LShr | Bin::AShr if y >= u64::from(bits) => 0x5eed,
                    Bin::Shl => x << y,
                    Bin::LShr => x >> y,
                    Bin::AShr => (signed(bits, x) >> y) as u64,
                };
                r & mask(bits)
            }
            TypeDef::Double => {
                let (x, y) = (f64::from_bits(x), f64::from_bits(y));
                let r = match op {
                    Bin::Add => x + y,
                    Bin::Sub => x - y,
                    Bin::Mul => x * y,
                    _ => x / y,
                };
                r.to_bits()
            }
            _ => {
                let (x, y) = (f32::from_bits(x as u32), f32::from_bits(y as u32));
                let r = match op {
                    Bin::Add => x + y,
                    Bin::Sub => x - y,
                    Bin::Mul => x * y,
                    _ => x / y,
                };
                u64::from(r.to_bits())
            }
        }
    }

    fn cast(&self, op: Cast, from: Ty, to: Ty, x: u64) -> u64 {
        let float = |t: Ty, x: u64| match self.m.types[t.0 as usize] {
            TypeDef::Double => f64::from_bits(x),
            _ => f64::from(f32::from_bits(x as u32)),
        };
        let to_float = |x: f64| match self.m.types[to.0 as usize] {
            TypeDef::Double => x.to_bits(),
            _ => u64::from((x as f32).to_bits()),
        };
        let (fb, tb) = (self.bits_of(from), self.bits_of(to));
        match op {
            Cast::Trunc | Cast::ZExt | Cast::Bitcast => x & mask(tb),
            Cast::SExt => signed(fb, x & mask(fb)) as u64 & mask(tb),
            Cast::FpToUi => (float(from, x) as u64) & mask(tb),
            Cast::FpToSi => (float(from, x) as i64 as u64) & mask(tb),
            Cast::UiToFp => to_float((x & mask(fb)) as f64),
            Cast::SiToFp => to_float(signed(fb, x & mask(fb)) as f64),
            Cast::FpExt => to_float(float(from, x)),
        }
    }

    fn rmw(op: Rmw, old: u64, v: u64) -> u64 {
        let (o, w) = (old as u32, v as u32);
        u64::from(match op {
            Rmw::Xchg => w,
            Rmw::Add => o.wrapping_add(w),
            Rmw::Sub => o.wrapping_sub(w),
            Rmw::And => o & w,
            Rmw::Or => o | w,
            Rmw::Xor => o ^ w,
            Rmw::Max => (o as i32).max(w as i32) as u32,
            Rmw::Min => (o as i32).min(w as i32) as u32,
            Rmw::UMax => o.max(w),
            Rmw::UMin => o.min(w),
        })
    }

    fn inst(&mut self, i: u32) -> Result<Val, String> {
        let m = self.m;
        let inst = &m.insts[i as usize];
        let ty = inst.ty;
        Ok(match &inst.op {
            Op::Bin(op, a, b) => Val::S(self.bin(ty, *op, self.scalar(*a)?, self.scalar(*b)?)),
            Op::Cmp(p, a, b) => {
                let t = m.type_of(*a);
                let (x, y) = (self.scalar(*a)?, self.scalar(*b)?);
                let r = if *p <= pred::TRUE {
                    let f = |x: u64| match m.types[t.0 as usize] {
                        TypeDef::Double => f64::from_bits(x),
                        _ => f64::from(f32::from_bits(x as u32)),
                    };
                    holds_f(*p, f(x), f(y))
                } else {
                    let bits = self.bits_of(t);
                    holds_i(*p, bits, x & mask(bits), y & mask(bits))
                };
                Val::S(u64::from(r))
            }
            Op::Select(c, t, f) => {
                if self.scalar(*c)? & 1 != 0 {
                    self.get(*t)?
                } else {
                    self.get(*f)?
                }
            }
            Op::Cast(op, x) => {
                let from = m.type_of(*x);
                Val::S(self.cast(*op, from, ty, self.scalar(*x)?))
            }
            Op::Extract(a, n) => match self.get(*a)? {
                Val::Agg(f) => Val::S(*f.get(*n as usize).ok_or("a field past the end")?),
                Val::S(x) => Val::S(mix(x ^ u64::from(*n))),
            },
            Op::Gep(pointee, p, idx) => {
                let mut addr = self.scalar(*p)?;
                let mut t = *pointee;
                for (k, &v) in idx.iter().enumerate() {
                    let n = self.scalar(v)?;
                    let n = signed(self.bits_of(m.type_of(v)), n) as u64;
                    if k == 0 {
                        addr = addr.wrapping_add(n.wrapping_mul(self.size(t)));
                        continue;
                    }
                    match &m.types[t.0 as usize] {
                        TypeDef::Array(_, e) => {
                            t = *e;
                            addr = addr.wrapping_add(n.wrapping_mul(self.size(t)));
                        }
                        TypeDef::Struct(_, fields) => {
                            let fields = fields.clone();
                            let at = n as usize;
                            addr += fields[..at.min(fields.len())]
                                .iter()
                                .map(|&f| self.size(f))
                                .sum::<u64>();
                            t = *fields.get(at).ok_or("a field past the end")?;
                        }
                        _ => addr = addr.wrapping_add(n),
                    }
                }
                Val::S(addr)
            }
            Op::Load(p) => Val::S(self.load(self.scalar(*p)?)),
            Op::Store(p, v) => {
                let (p, v) = (self.scalar(*p)?, self.scalar(*v)?);
                self.effect("store".into(), vec![p, v]);
                self.lds.insert(p, v);
                Val::S(0)
            }
            Op::AtomicRmw(op, p, v) => {
                let (p, v) = (self.scalar(*p)?, self.scalar(*v)?);
                let old = self.load(p);
                let op = *op;
                self.effect(format!("atomicrmw {}", op as u8), vec![p, v, old]);
                self.lds.insert(p, Self::rmw(op, old, v));
                Val::S(old)
            }
            Op::CmpXchg(p, want, new) => {
                let (p, want, new) = (self.scalar(*p)?, self.scalar(*want)?, self.scalar(*new)?);
                let old = self.load(p);
                let swapped = old & 0xffff_ffff == want & 0xffff_ffff;
                self.effect("cmpxchg".into(), vec![p, want, new, old]);
                self.lds.insert(p, if swapped { new } else { old });
                Val::Agg(vec![old, u64::from(swapped)])
            }
            Op::Call(f, args) => {
                let func = &m.funcs[*f as usize];
                let name = func.name.clone();
                let class = effect(func);
                let mut a = Vec::with_capacity(args.len());
                for &v in args {
                    a.push(self.scalar(v)?);
                }
                let h = a.iter().fold(hash_str(&name), |h, &x| mix(h ^ x));
                match class {
                    Effect::Writes => {
                        let r = self.make(ty, h ^ mix(self.epoch));
                        let mut seen = a.clone();
                        if !matches!(m.types[ty.0 as usize], TypeDef::Void) {
                            seen.push(r.bits());
                        }
                        self.effect(name, seen);
                        r
                    }
                    Effect::Reads => self.make(ty, h ^ mix(self.epoch)),
                    Effect::Pure | Effect::Lanes | Effect::Local => self.make(ty, h),
                }
            }
            Op::Phi(_) => return Err("a join among a block's instructions".into()),
        })
    }

    fn run(mut self) -> Effects {
        if let Err(e) = self.go() {
            self.effects.push((format!("error: {e}"), Vec::new()));
        }
        self.effects
    }

    fn go(&mut self) -> Result<(), String> {
        let m = self.m;
        let (mut b, mut from) = (0usize, usize::MAX);
        for _ in 0..VISITS {
            let blk = &m.blocks[b];
            // Joins take their values together, before any is assigned. A
            // join without a value for the edge taken has none (the lowering
            // leaves some that nothing reads, and the writer drops them): a
            // read of it is the error.
            let mut joined = Vec::with_capacity(blk.phis.len());
            for &p in &blk.phis {
                let Op::Phi(incoming) = &m.insts[p as usize].op else {
                    return Err("a join that is no join".into());
                };
                let v = match incoming.iter().find(|x| x.1 == from) {
                    Some(x) => self.get(x.0).ok(),
                    None => None,
                };
                joined.push((p, v));
            }
            for (p, x) in joined {
                self.vals[p as usize] = x;
            }
            for n in 0..m.blocks[b].body.len() {
                let i = m.blocks[b].body[n];
                let r = self.inst(i)?;
                self.vals[i as usize] = Some(r);
            }
            from = b;
            b = match m.blocks[b].term.ok_or("a block without a terminator")? {
                Term::Ret => {
                    self.effects.push(("ret".into(), Vec::new()));
                    return Ok(());
                }
                Term::Br(t) => t,
                Term::CondBr(c, t, f) => {
                    if self.scalar(c)? & 1 != 0 {
                        t
                    } else {
                        f
                    }
                }
            };
        }
        self.effects.push(("still running".into(), Vec::new()));
        Ok(())
    }
}

/// What `m` does to the world on the thread `seed` makes up.
pub(crate) fn effects(m: &Module, seed: u64) -> Effects {
    Run {
        m,
        seed: mix(seed.wrapping_add(0x9e37_79b9_7f4a_7c15)),
        vals: vec![None; m.insts.len()],
        effects: Vec::new(),
        epoch: 0,
        lds: HashMap::new(),
    }
    .run()
}

/// How `before` and `after` compared on the threads `0..seeds`: the first
/// difference (its seed, the number of effects both made before it, and
/// what differs: names, never values), else `None`; and how the runs of
/// `before` ended (`ret`, cut at the visit limit, or an error the module
/// makes this run meet: a value read before it ran, say), for how much
/// of the program the check saw.
pub(crate) fn compare(before: &Module, after: &Module, seeds: u64) -> (Option<String>, String) {
    let mut ended = [0usize; 3];
    let mut effects_seen = 0;
    let mut first_error = None;
    for seed in 0..seeds {
        let (x, y) = (effects(before, seed), effects(after, seed));
        let end = x.last().map_or("", |e| e.0.as_str());
        match end {
            "ret" => ended[0] += 1,
            "still running" => ended[1] += 1,
            e => {
                ended[2] += 1;
                first_error.get_or_insert_with(|| e.to_owned());
            }
        }
        effects_seen += x.len();
        if x == y {
            continue;
        }
        let at = x.iter().zip(&y).take_while(|(a, b)| a == b).count();
        let say = |e: Option<&(String, Vec<u64>)>| e.map_or("nothing".to_owned(), |e| e.0.clone());
        let what = match (x.get(at), y.get(at)) {
            (Some(a), Some(b)) if a.0 == b.0 => format!("{} with other values", a.0),
            (a, b) => format!("{} became {}", say(a), say(b)),
        };
        return (
            Some(format!(
                "seed={seed} effects_alike={at} of {} then {what}",
                x.len()
            )),
            String::new(),
        );
    }
    let mut tally = format!(
        "ret={} cut={} error={} effects={effects_seen}",
        ended[0], ended[1], ended[2]
    );
    if let Some(e) = first_error {
        tally.push_str(&format!(" ({e})"));
    }
    (None, tally)
}

/// [`compare`]'s difference alone.
#[cfg(test)]
pub(crate) fn first_difference(before: &Module, after: &Module, seeds: u64) -> Option<String> {
    compare(before, after, seeds).0
}

/// The switch: it changes no translation, so the translator's fingerprint
/// leaves it out (`emit::translation_fingerprint`).
pub(crate) const SWITCH: &str = "VICTORY_SHADER_OPT_CHECK";

/// `VICTORY_SHADER_OPT_CHECK`'s value: how many threads to run each
/// program on (0: no check).
pub(crate) fn seeds() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var(SWITCH)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

#[cfg(test)]
mod tests {
    use super::super::Attr;
    use super::super::opt::{ALL, DEAD, FACTS, FOLD, LOADS, REUSE};
    use super::*;

    /// A read, a store, the same read again: one read is not the other.
    fn read_store_read() -> (Module, usize, V, V) {
        let mut m = Module::default();
        let i32t = m.int(32);
        let void = m.void();
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        m.ptr(fty);
        let reads = m.function("dx.op.bufferLoad.i32", fty, Some(Attr::ReadOnly), false);
        let writes = m.function("dx.op.bufferStore.i32", fty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        m.function("main", main_ty, None, true);
        let b = m.new_block();
        let one = m.i32(1);
        let first = m.call(b, reads, vec![one, one]);
        m.call(b, writes, vec![first, one]);
        let again = m.call(b, reads, vec![one, one]);
        m.call(b, writes, vec![again, one]);
        m.terminate(b, Term::Ret);
        (m, b, first, again)
    }

    #[test]
    fn the_passes_leave_what_a_program_does() {
        let (m, ..) = read_store_read();
        for passes in [DEAD, FOLD, REUSE, FACTS, LOADS, ALL] {
            let mut after = m.clone();
            after.optimise(passes);
            assert_eq!(first_difference(&m, &after, 8), None, "passes {passes}");
        }
    }

    #[test]
    fn a_read_taken_across_a_store_is_seen() {
        // What an unsound reuse would make: the second read is the first.
        let (m, b, first, again) = read_store_read();
        let mut wrong = m.clone();
        let V::Inst(a) = again else { panic!() };
        let blk = &mut wrong.blocks[b];
        blk.body.retain(|&i| i != a);
        for &i in &blk.body {
            if let Op::Call(_, args) = &mut wrong.insts[i as usize].op {
                for x in args.iter_mut().filter(|x| **x == again) {
                    *x = first;
                }
            }
        }
        let found = first_difference(&m, &wrong, 4).expect("a difference");
        assert!(found.contains("with other values"), "{found}");
    }
}
