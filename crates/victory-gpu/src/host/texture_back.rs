//! The host list's side of the texture copy-back ([`crate::texback`]): the
//! textures of the kinds copied back that a list's shaders write are noted
//! as the draw or dispatch is prepared, and copied into read-back buffers
//! when the list closes; their texels land in title memory once the list has
//! run (`read_backs`). After fxceless/victory-fork #999 (`f5c03f3b0f`) and
//! #1376 (`86d446b74f`); a module of its own to keep `host.rs` small.
//! Also: the landing of a flight's read-backs, and the copy-backs' share of
//! the waits of submissions sent ahead (`crate::texback::STATS`).

use super::HostList;
use crate::texback::{STATS, SETTLES, SUPERSEDED, WAITED_US, WENT_AHEAD};
use std::sync::atomic::Ordering::Relaxed;

/// After lists have run, in order (the lists of one flight, or one list):
/// their read-backs ([`super::read_backs`]), each list's texture copy-backs
/// first. With `VICTORY_TEXTURE_BACK_LEAN` a copy that a later one of the
/// same memory among `lists` supersedes is left out: what tells the title
/// that the card is done (the flight's bottom-of-pipe writes, the signals
/// behind it) comes after all of them.
pub(super) fn read_backs<'a>(lists: impl Iterator<Item = &'a HostList> + Clone) {
    let superseded = if crate::texback::lean() { superseded(lists.clone()) } else { Vec::new() };
    for (l, list) in lists.enumerate() {
        for (b, back) in list.deferred().texture_backs.iter().enumerate() {
            if superseded.contains(&(l, b)) {
                STATS[SUPERSEDED].fetch_add(1, Relaxed);
                continue;
            }
            crate::texback::land(back);
        }
        super::read_backs(list);
    }
}

/// The copies (list, copy) among `lists` that a later copy of the same
/// memory supersedes ([`superseded_of`]).
fn superseded<'a>(lists: impl Iterator<Item = &'a HostList>) -> Vec<(usize, usize)> {
    let copies: Vec<CopyBack> = lists
        .enumerate()
        .flat_map(|(l, list)| {
            let deferred = list.deferred();
            deferred.texture_backs.iter().enumerate().map(|(b, back)| ((l, b), back.range(), back.fresh())).collect::<Vec<_>>()
        })
        .collect();
    superseded_of(&copies)
}

/// A copy back, in landing order: (list, copy), the title memory it writes
/// (address, end), and whether its allocation is still the one it was
/// recorded for.
type CopyBack = ((usize, usize), (usize, usize), bool);

/// The copies a later copy of exactly the same memory supersedes, one whose
/// allocation is still the one it was recorded for (a stale one lands
/// nothing: the copy before it still does).
fn superseded_of(copies: &[CopyBack]) -> Vec<(usize, usize)> {
    let mut later = std::collections::HashSet::new();
    let mut out = Vec::new();
    for &(at, range, fresh) in copies.iter().rev() {
        if later.contains(&range) {
            out.push(at);
        } else if fresh {
            later.insert(range);
        }
    }
    out
}

/// Whether preparing `runs` reads memory texture copy-backs in the air
/// write.
fn reads(runs: &[&HostList]) -> bool {
    let ranges = {
        let line = super::line();
        if line.texture_written.is_empty() {
            return false;
        }
        line.texture_written.clone()
    };
    super::blocked_by(&ranges, runs, false).is_some()
}

/// A submission sent ahead waits for flights before it is prepared
/// ([`super::late_blocked`]): if it reads memory their texture copy-backs
/// write (and that memory holds it back: not with
/// `VICTORY_TEXTURE_BACK_AHEAD`), when its wait began, for [`waited`].
pub(super) fn reading(runs: &[&HostList]) -> Option<std::time::Instant> {
    (crate::texback::holding() && reads(runs)).then(std::time::Instant::now)
}

/// The wait [`reading`] found is over: counted for the copy-back.
pub(super) fn waited(began: Option<std::time::Instant>) {
    if let Some(began) = began {
        STATS[SETTLES].fetch_add(1, Relaxed);
        STATS[WAITED_US].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
    }
}

/// With `VICTORY_TEXTURE_BACK_AHEAD`: a submission sent ahead without a
/// wait that reads memory texture copy-backs in the air write.
pub(super) fn went_ahead(runs: &[&HostList]) {
    if !crate::texback::holding() && reads(runs) {
        STATS[WENT_AHEAD].fetch_add(1, Relaxed);
    }
}

impl HostList {
    /// A draw or dispatch being prepared writes title texture `resource`
    /// through unordered access: if it is of a kind copied back
    /// ([`crate::texback`]), the list copies it when it closes.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        // A target whose ticket went stale (the title reused or remapped the
        // memory it tiles into) must be re-registered, or the read-back
        // would be dropped as one of memory the texture no longer holds.
        let existing = deferred
            .back_wanted
            .iter_mut()
            .find(|target| target.owner() == resource as usize);
        match existing {
            Some(entry) if !entry.fresh() => {
                if let Some(target) = crate::texback::Target::new(resource) {
                    *entry = target;
                }
            }
            None => deferred.back_wanted.extend(crate::texback::Target::new(resource)),
            Some(_) => {}
        }
    }

    /// Before the list closes: the textures its shaders wrote that are copied
    /// back go into read-back buffers ([`crate::texback::record`]). Each is
    /// left in the state the list's own recording left it in, not in
    /// COPY_SOURCE: a later list (or a bindless use) starts from that state
    /// (after fxceless/victory-fork PR #1170, BradRoland).
    pub(super) fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            let copy = target.copy.clone();
            // The state this recording leaves, including any title barriers
            // after the UAV write. Other lists can be recorded out of order.
            let last = self
                .states
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&(std::sync::Arc::as_ptr(&copy) as usize))
                .map(|(_, last, _)| *last);
            // On first use the submission prelude establishes COPY_SOURCE.
            let needs_barrier = last.is_none_or(|state| state & COPY_SOURCE == 0);
            if needs_barrier {
                self.settle(&copy, COPY_SOURCE);
            }
            let back = crate::texback::record(&self.list, target);
            if needs_barrier && let Some(last) = last {
                self.settle(&copy, last);
            }
            if let Some(back) = back {
                self.deferred().texture_backs.push(back);
            }
        }
    }

    /// For tests: whether the list's read-back wants a *fresh* target for
    /// `resource` — one whose ticket has not gone stale. A stale target is
    /// one the title's memory reuse invalidated, which the list must
    /// re-register on the next write of `resource`.
    #[cfg(test)]
    pub(crate) fn back_wanted_fresh(&self, resource: crate::object::This) -> bool {
        self.deferred()
            .back_wanted
            .iter()
            .any(|t| t.owner() == resource as usize && t.fresh())
    }

    /// For tests: how many texture read-backs the list recorded at its close.
    #[cfg(test)]
    pub(crate) fn texture_back_count(&self) -> usize {
        self.deferred().texture_backs.len()
    }
}

#[cfg(test)]
mod lean_tests {
    use super::superseded_of;

    /// Only an earlier copy of the very same memory is left out, and only
    /// for a later copy that still lands.
    #[test]
    fn a_later_fresh_copy_of_the_same_memory_supersedes() {
        let (a, b, c) = ((0x1000, 0x2000), (0x2000, 0x3000), (0x1000, 0x1800));
        let copies = [((0, 0), a, true), ((0, 1), b, true), ((1, 0), a, true), ((1, 1), c, true), ((2, 0), b, false)];
        let mut out = superseded_of(&copies);
        out.sort_unstable();
        // `a` of list 0 by list 1's; `b` of list 0 stays (list 2's is stale);
        // `c` overlaps `a` but is not the same memory.
        assert_eq!(out, [(0, 0)]);
        assert!(superseded_of(&copies[..1]).is_empty());
    }
}
