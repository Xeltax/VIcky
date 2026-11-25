//! A host-side check of the ray-tracing structures in title memory
//! (`VICTORY_BVH_CHECK=<frames>`, diagnostics): every that many frames the
//! addresses named since the last check by the dispatches of shaders that
//! read title memory by address are tried as structures, the ones found
//! are walked from their root, and counts are logged. Counts only: no
//! bytes of a structure reach the log.
//!
//! The node formats are those of `victory_shader::bvh`. A structure begins
//! with a header at the address a dispatch names; node pointers count from
//! that address, and the dword at byte [`ROOT`] of the header is the root
//! pointer.
//!
//! Where this comes from: measured in the retail title on 2026-11-25, not
//! from any document. Box nodes that fit together (a parent's two box-node
//! children found where one base puts them, each inside the box the parent
//! holds for it) were looked for in 1.3 GB of the copies shaders had
//! touched, without assuming a base; the bases that came out were
//! addresses dispatches name, and in the structures with one root and
//! nothing wrong the dword at byte 48 was that root's pointer. Walking by
//! this rule from the named addresses then gave 181 structures, 126 of
//! them with nothing wrong, the largest with 10,826 box nodes and 26,733
//! triangles.
//!
//! shortcut: title memory is read as it is when the check runs, through the
//! kernel (a page that is not there reads zeros). What the builder stored
//! this frame may not be copied back yet, so a structure it is rebuilding
//! may be seen half old and half new; with `VICTORY_BVH_QUARANTINE=1`
//! nothing is copied back and the check sees the title's own bytes only.
//! shortcut: instance nodes are counted and not followed (their layout is
//! not in `bvh.rs`); a bottom-level structure is checked when a dispatch
//! names it.

use std::collections::HashSet;
use std::sync::Mutex;

use victory_shader::bvh;

/// A node's size and the unit of a node pointer's offset.
const NODE: u64 = 64;
/// The byte of a structure's header that holds its root pointer.
const ROOT: u64 = 48;
/// Nodes walked in one structure before the walk gives up.
const MOST_NODES: usize = 1 << 20;
/// Addresses kept between two checks.
const MOST_NAMED: usize = 1 << 18;
/// A window of title memory outside buffers (`memory.rs` `CELL`).
const WINDOW: u64 = 4 << 20;
/// A structure whose nodes lie further apart than this has a pointer that
/// leaves it: the largest measured was 3 MB.
const FAR: u64 = 256 << 20;

/// What a walk of one structure found.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct Counts {
    /// Box nodes with 16-bit and with 32-bit bounds, triangle nodes, the
    /// triangles the pointers name in them, instance nodes.
    pub box16: u64,
    pub box32: u64,
    pub triangle_nodes: u64,
    pub triangles: u64,
    pub instances: u64,
    /// Children whose box is not finite or has a minimum above its maximum:
    /// no ray enters such a box, so the child is switched off and whatever
    /// its pointer names is not walked. Not counted as wrong.
    pub closed: u64,
    /// Pointers of kind 7 (procedural: the title has none) and pointers to
    /// a node whose 64 bytes are all zero.
    pub kind7: u64,
    pub zero_nodes: u64,
    /// Triangles with a vertex that is not finite.
    pub bad_triangles: u64,
    /// Children (box nodes by the union of their boxes, triangles by their
    /// vertices) that reach outside the box their parent holds for them.
    pub outside: u64,
    /// Pointers to a node already walked: a node with two parents, or a
    /// cycle.
    pub shared: u64,
    /// The deepest pointer followed (the root is 1).
    pub depth: u32,
    /// The lowest and highest node address reached.
    pub first: u64,
    pub last: u64,
    /// 4 MiB windows holding a node reached.
    pub windows: u64,
    /// The walk stopped at its limit of nodes.
    pub cut: bool,
}

impl Counts {
    /// What a structure built right has none of.
    pub(crate) fn wrong(&self) -> u64 {
        self.kind7 + self.zero_nodes + self.bad_triangles + self.outside + self.shared
    }

    fn nodes(&self) -> u64 {
        self.box16 + self.box32 + self.triangle_nodes + self.instances
    }

    /// Bytes from the lowest node reached to the end of the highest.
    fn reach(&self) -> u64 {
        if self.nodes() == 0 {
            0
        } else {
            self.last - self.first + NODE
        }
    }

    fn add(&mut self, c: &Counts) {
        self.box16 += c.box16;
        self.box32 += c.box32;
        self.triangle_nodes += c.triangle_nodes;
        self.triangles += c.triangles;
        self.instances += c.instances;
        self.closed += c.closed;
        self.kind7 += c.kind7;
        self.zero_nodes += c.zero_nodes;
        self.bad_triangles += c.bad_triangles;
        self.outside += c.outside;
        self.shared += c.shared;
        self.depth = self.depth.max(c.depth);
        self.windows += c.windows;
        self.cut |= c.cut;
    }
}

type Box6 = [f32; 6];

fn words(read: &mut dyn FnMut(u64, &mut [u8]), at: u64, bytes: usize) -> [u32; 28] {
    let mut raw = [0u8; 112];
    read(at, &mut raw[..bytes.min(112)]);
    std::array::from_fn(|i| {
        u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]])
    })
}

fn ordered(b: &Box6) -> bool {
    b.iter().all(|v| v.is_finite()) && (0..3).all(|a| b[a] <= b[3 + a])
}

/// Whether `inner` lies in `outer`, with room for 16-bit bounds' rounding.
fn within(outer: &Box6, inner: &Box6) -> bool {
    (0..3).all(|a| {
        let room = 2e-3 * outer[a].abs().max(outer[3 + a].abs()) + 1e-4;
        inner[a] >= outer[a] - room && inner[3 + a] <= outer[3 + a] + room
    })
}

/// The boxes of a box node of `kind`, and their union over the children
/// that are there and not switched off.
fn boxes(w: &[u32; 28], kind: u32) -> ([Box6; 4], Option<Box6>) {
    let all = if kind == 5 {
        bvh::boxes32(w)
    } else {
        bvh::boxes16(w)
    };
    let mut union: Option<Box6> = None;
    for (n, b) in all.iter().enumerate() {
        if w[n] == bvh::NONE || !ordered(b) {
            continue;
        }
        union = Some(match union {
            None => *b,
            Some(u) => std::array::from_fn(|i| {
                if i < 3 {
                    u[i].min(b[i])
                } else {
                    u[i].max(b[i])
                }
            }),
        });
    }
    (all, union)
}

/// The box around triangle `k` of a triangle node; None if a vertex is not
/// finite.
fn triangle_box(w: &[u32; 28], k: u32) -> Option<Box6> {
    let mut b = [
        f32::INFINITY,
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for vertex in bvh::TRIANGLE_VERTICES[(k & 3) as usize] {
        for a in 0..3 {
            let v = f32::from_bits(w[3 * vertex as usize + a]);
            if !v.is_finite() {
                return None;
            }
            b[a] = b[a].min(v);
            b[3 + a] = b[3 + a].max(v);
        }
    }
    Some(b)
}

fn node_bytes(kind: u32) -> usize {
    if kind == 5 { 112 } else { 64 }
}

/// Walks the structure whose node pointers count from `base`, from the
/// pointer `root`, over `most` nodes at most.
pub(crate) fn walk(
    read: &mut dyn FnMut(u64, &mut [u8]),
    base: u64,
    root: u32,
    most: usize,
) -> Counts {
    let mut c = Counts {
        first: u64::MAX,
        ..Counts::default()
    };
    let mut seen: HashSet<u64> = HashSet::new();
    let mut triangle_nodes: HashSet<u64> = HashSet::new();
    let mut windows: HashSet<u64> = HashSet::new();
    let mut stack: Vec<(u32, u32, Option<Box6>)> = vec![(root, 1, None)];
    while let Some((pointer, depth, held)) = stack.pop() {
        if seen.len() + triangle_nodes.len() >= most {
            c.cut = true;
            break;
        }
        let kind = pointer & 7;
        let at = base.wrapping_add(u64::from(pointer >> 3) * NODE);
        c.depth = c.depth.max(depth);
        if kind == 7 {
            c.kind7 += 1;
            continue;
        }
        match kind {
            4 | 5 => {
                if !seen.insert(at) {
                    c.shared += 1;
                    continue;
                }
                let w = words(read, at, node_bytes(kind));
                if w.iter().take(16).all(|&d| d == 0) {
                    c.zero_nodes += 1;
                    continue;
                }
                if kind == 5 {
                    c.box32 += 1;
                } else {
                    c.box16 += 1;
                }
                let (all, union) = boxes(&w, kind);
                if let (Some(held), Some(union)) = (held, union)
                    && !within(&held, &union)
                {
                    c.outside += 1;
                }
                for n in 0..4 {
                    if w[n] == bvh::NONE {
                        continue;
                    }
                    if !ordered(&all[n]) {
                        c.closed += 1;
                        continue;
                    }
                    stack.push((w[n], depth + 1, Some(all[n])));
                }
            }
            6 => {
                if seen.insert(at) {
                    c.instances += 1;
                } else {
                    c.shared += 1;
                    continue;
                }
            }
            k => {
                let w = words(read, at, 64);
                if w.iter().take(16).all(|&d| d == 0) {
                    c.zero_nodes += 1;
                    continue;
                }
                if triangle_nodes.insert(at) {
                    c.triangle_nodes += 1;
                }
                c.triangles += 1;
                match triangle_box(&w, k) {
                    None => c.bad_triangles += 1,
                    Some(b) => {
                        if held.is_some_and(|held| !within(&held, &b)) {
                            c.outside += 1;
                        }
                    }
                }
            }
        }
        c.first = c.first.min(at);
        c.last = c.last.max(at);
        windows.insert(at / WINDOW);
    }
    c.windows = windows.len() as u64;
    if c.first == u64::MAX {
        c.first = 0;
    }
    c
}

/// The structure whose header is at `at`, if there is one: its root
/// pointer and its walk. A header's root is a box node, and a structure
/// has two box nodes at least (one box node with triangles under it is
/// what many a table of floats looks like).
pub(crate) fn structure(read: &mut dyn FnMut(u64, &mut [u8]), at: u64) -> Option<(u32, Counts)> {
    let mut raw = [0u8; 4];
    read(at + ROOT, &mut raw);
    let root = u32::from_le_bytes(raw);
    if !matches!(root & 7, 4 | 5) || root >> 3 == 0 || root == bvh::NONE {
        return None;
    }
    let c = walk(read, at, root, MOST_NODES);
    (c.box16 + c.box32 >= 2).then_some((root, c))
}

/// `VICTORY_BVH_CHECK=<frames>`: how many frames lie between two checks
/// (0 or unset: no check).
fn every() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_BVH_CHECK")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

static NAMED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);

/// What a dispatch of a shader that reads title memory by address names
/// (`memory::ready`).
pub(crate) fn named(addresses: &[u64], constants: &[u32]) {
    if every() == 0 {
        return;
    }
    let mut guard = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    let set = guard.get_or_insert_with(HashSet::new);
    let mut add = |a: u64| {
        if set.len() < MOST_NAMED {
            set.insert(a);
        }
    };
    addresses.iter().for_each(|&a| add(a));
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            add(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            add(u64::from(c) << 8);
        }
    }
}

/// At the first dispatch of `frame`: the check, when one is due.
pub(crate) fn frame(frame: u64) {
    let n = every();
    if n == 0 || !frame.is_multiple_of(n) {
        return;
    }
    let Some(named) = NAMED.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let started = std::time::Instant::now();
    let mut read = |at: u64, to: &mut [u8]| {
        to.fill(0);
        crate::host::copy_mapped(to.as_mut_ptr(), at as *const u8, to.len(), false);
    };
    let mut walked: Vec<(u64, u32, Counts)> = named
        .iter()
        .filter_map(|&at| structure(&mut read, at).map(|(root, c)| (at, root, c)))
        .collect();
    // Two kinds, told apart by what they hold: structures of instances
    // (top level) and structures of triangles.
    let (mut top, mut bottom) = (Counts::default(), Counts::default());
    let (mut tops, mut bottoms, mut wrong, mut far, mut reach) = (0u32, 0u32, 0u32, 0u32, 0u64);
    for (_, _, c) in &walked {
        if c.instances > c.triangles {
            top.add(c);
            tops += 1;
        } else {
            bottom.add(c);
            bottoms += 1;
        }
        wrong += u32::from(c.wrong() != 0);
        if c.reach() > FAR {
            far += 1;
        } else {
            reach += c.reach();
        }
    }
    tracing::info!(
        frame,
        named = named.len(),
        structures = walked.len(),
        with_wrong = wrong,
        reaching_far = far,
        others_reach_mb = format_args!("{:.1}", reach as f64 / 1048576.0),
        ms = started.elapsed().as_millis() as u64,
        "ray-tracing structures in title memory, walked from the addresses dispatches named (VICTORY_BVH_CHECK)"
    );
    for (what, n, c) in [
        ("of instances", tops, &top),
        ("of triangles", bottoms, &bottom),
    ] {
        tracing::info!(
            frame,
            structures = n,
            box16 = c.box16,
            box32 = c.box32,
            triangle_nodes = c.triangle_nodes,
            triangles = c.triangles,
            instances = c.instances,
            closed = c.closed,
            kind7 = c.kind7,
            zero_nodes = c.zero_nodes,
            bad_triangles = c.bad_triangles,
            outside = c.outside,
            shared = c.shared,
            deepest = c.depth,
            cut = c.cut,
            "ray-tracing structures {what}, counts over all of them"
        );
    }
    // The ones with most wrong, a few.
    walked.sort_unstable_by_key(|(_, _, c)| std::cmp::Reverse(c.wrong()));
    for (at, root, c) in walked.iter().take(6).filter(|(_, _, c)| c.wrong() != 0) {
        tracing::info!(
            frame,
            at = format_args!("{at:#x}"),
            root = format_args!("{root:#x}"),
            reach_mb = format_args!("{:.1}", c.reach() as f64 / 1048576.0),
            counts = ?c,
            "ray-tracing structure with something wrong"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A made-up structure in a byte vector at address `AT`: a header of
    /// two nodes' size with the root pointer at byte 48, a 32-bit root at
    /// step 2 with a 16-bit box node and a triangle under it, and a
    /// triangle under the 16-bit node.
    const AT: u64 = 0x10_0000;

    fn put_f32(m: &mut [u8], at: usize, v: f32) {
        m[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn put_u32(m: &mut [u8], at: usize, v: u32) {
        m[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn half(v: f32) -> u16 {
        // Enough for the small whole numbers used here.
        let bits = v.to_bits();
        let sign = (bits >> 16) & 0x8000;
        if v == 0.0 {
            return sign as u16;
        }
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        (sign | (exponent as u32) << 10 | (bits >> 13) & 0x3ff) as u16
    }

    fn triangle(m: &mut [u8], node: usize, v: [[f32; 3]; 3]) {
        for (n, p) in v.iter().enumerate() {
            for a in 0..3 {
                put_f32(m, node + 12 * n + 4 * a, p[a]);
            }
        }
        put_u32(m, node + 60, 0b1001);
    }

    fn made() -> Vec<u8> {
        let mut m = vec![0u8; 64 * 16];
        m[..8].copy_from_slice(b"headerxx");
        put_u32(&mut m, ROOT as usize, 2 << 3 | 5);
        // Root: a 32-bit box node at step 2 (bytes 128..256).
        let root = 128;
        put_u32(&mut m, root, 4 << 3 | 4);
        put_u32(&mut m, root + 4, 5 << 3);
        put_u32(&mut m, root + 8, bvh::NONE);
        put_u32(&mut m, root + 12, bvh::NONE);
        for (n, b) in [
            [0.0, 0.0, 0.0, 4.0, 4.0, 4.0f32],
            [8.0, 0.0, 0.0, 9.0, 1.0, 1.0],
        ]
        .iter()
        .enumerate()
        {
            for (i, v) in b.iter().enumerate() {
                put_f32(&mut m, root + 16 + 24 * n + 4 * i, *v);
            }
        }
        // A 16-bit box node at step 4 with one triangle child at step 6.
        let inner = 256;
        put_u32(&mut m, inner, 6 << 3);
        for n in 1..4 {
            put_u32(&mut m, inner + 4 * n, bvh::NONE);
        }
        for (i, v) in [1.0, 1.0, 1.0, 2.0, 2.0, 2.0f32].iter().enumerate() {
            m[inner + 16 + 2 * i..inner + 18 + 2 * i].copy_from_slice(&half(*v).to_le_bytes());
        }
        triangle(
            &mut m,
            320,
            [[8.0, 0.0, 0.0], [9.0, 0.0, 0.0], [8.0, 1.0, 1.0]],
        );
        triangle(
            &mut m,
            384,
            [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 2.0]],
        );
        m
    }

    fn reader(m: &[u8]) -> impl FnMut(u64, &mut [u8]) + '_ {
        move |at, to| {
            to.fill(0);
            let Some(from) = at.checked_sub(AT).map(|v| v as usize) else {
                return;
            };
            if from < m.len() {
                let n = to.len().min(m.len() - from);
                to[..n].copy_from_slice(&m[from..from + n]);
            }
        }
    }

    #[test]
    fn a_structure_is_found_by_its_header_and_walked() {
        let m = made();
        let mut read = reader(&m);
        let (root, c) = structure(&mut read, AT).expect("a structure");
        assert_eq!(root, 2 << 3 | 5);
        assert_eq!(
            (c.box32, c.box16, c.triangle_nodes, c.triangles, c.instances),
            (1, 1, 2, 2, 0)
        );
        assert_eq!(c.wrong(), 0, "{c:?}");
        assert_eq!(c.depth, 3);
        assert_eq!(
            (c.first, c.last, c.windows, c.reach()),
            (AT + 128, AT + 384, 1, 320)
        );
    }

    #[test]
    fn counts_what_is_wrong() {
        let mut m = made();
        // The inner triangle leaves its box; the root's second child names
        // a node of zeros; its third is switched off (a box whose minimum
        // is above its maximum) and its pointer, of kind 7, is not
        // followed; its fourth names the inner node again.
        put_f32(&mut m, 384, 50.0);
        put_u32(&mut m, 128 + 4, 9 << 3);
        put_u32(&mut m, 128 + 8, 7 << 3 | 7);
        for (i, v) in [3.0, 0.0, 0.0, 1.0, 1.0, 1.0f32].iter().enumerate() {
            put_f32(&mut m, 128 + 16 + 48 + 4 * i, *v);
        }
        put_u32(&mut m, 128 + 12, 4 << 3 | 4);
        for (i, v) in [0.0, 0.0, 0.0, 4.0, 4.0, 4.0f32].iter().enumerate() {
            put_f32(&mut m, 128 + 16 + 72 + 4 * i, *v);
        }
                let c = walk(&mut reader(&m), AT, 2 << 3 | 5, MOST_NODES);
        assert_eq!(
            (c.outside, c.zero_nodes, c.kind7, c.closed, c.shared),
            (1, 1, 0, 1, 1),
            "{c:?}"
        );
        assert_eq!(c.wrong(), 3);
        // The same child open: its pointer of kind 7 counts.
        for (i, v) in [0.0, 0.0, 0.0, 1.0, 1.0, 1.0f32].iter().enumerate() {
            put_f32(&mut m, 128 + 16 + 48 + 4 * i, *v);
        }
        let c = walk(&mut reader(&m), AT, 2 << 3 | 5, MOST_NODES);
        assert_eq!((c.kind7, c.closed, c.wrong()), (1, 0, 4), "{c:?}");
    }

    #[test]
    fn memory_without_a_header_is_no_structure() {
        let zeros = vec![0u8; 64 * 32];
        assert_eq!(structure(&mut reader(&zeros), AT), None);
        let noise: Vec<u8> = (0..64 * 32u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        assert_eq!(structure(&mut reader(&noise), AT), None);
        // A header whose root is a triangle, and one box node alone.
        let mut m = made();
        put_u32(&mut m, ROOT as usize, 5 << 3);
        assert_eq!(structure(&mut reader(&m), AT), None);
        put_u32(&mut m, ROOT as usize, 4 << 3 | 4);
        assert_eq!(structure(&mut reader(&m), AT), None);
    }

    #[test]
    fn a_cycle_ends_and_a_walk_stops_at_its_limit() {
        let mut m = made();
        // The inner node's child is the root.
        put_u32(&mut m, 256, 2 << 3 | 5);
        let c = walk(&mut reader(&m), AT, 2 << 3 | 5, MOST_NODES);
        assert_eq!(c.shared, 1);
        assert!(!c.cut);
        let c = walk(&mut reader(&made()), AT, 2 << 3 | 5, 2);
        assert!(c.cut);
    }
}
