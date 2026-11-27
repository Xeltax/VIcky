//! VIcky: what `VICTORY_BVH_INPUT_CHECK=1` (diagnostics, `crate::draw`'s
//! `bvh_input`) reads of a copy: its bytes against title memory, and what
//! the copy knows of them. Reads only.

use super::Shadow;
use crate::watch::PAGE;

/// What a copy's range holds against title memory.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Compared {
    /// Bytes compared (the range cut to the copy), and how many differ.
    pub(crate) bytes: u64,
    pub(crate) differing: u64,
    /// The offset in the range of the first byte that differs.
    pub(crate) first: Option<u64>,
}

/// What a copy knows of a range ([`Shadow::input_state`]).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct State {
    /// The copy's own number ([`Shadow::id`]).
    pub(crate) id: u64,
    /// Whether the copy is in the card's memory (its host side is staging)
    /// or is title memory in place.
    pub(crate) carded: bool,
    pub(crate) in_place: bool,
    /// Bytes of the range a submission not yet copied back may be writing:
    /// there the copy is newer than title memory.
    pub(crate) pending: u64,
    /// Pages of the range the copy never took, and the lowest and highest
    /// asking (write-watch number) that took the others (0: none did).
    pub(crate) never: u64,
    pub(crate) pages: u64,
    pub(crate) taken: [u64; 2],
    /// The asking that stands, if the range lies in one asked ahead.
    pub(crate) asking: Option<u64>,
}

impl Shadow {
    /// `[address, address + len)` of the copy's host side against title
    /// memory now, a part at a time (a page of title memory that is not
    /// there reads zeros). A copy in place is title memory: nothing differs.
    pub(crate) fn input_compare(&self, address: u64, len: u64) -> Compared {
        const PART: usize = 1 << 16;
        let mut out = Compared::default();
        if self.place.is_some() || address < self.start {
            return out;
        }
        let len = len.min(self.left(address));
        let mut title = vec![0u8; PART];
        let mut held = vec![0u8; PART];
        let mut at = 0u64;
        while at < len {
            let n = (len - at).min(PART as u64) as usize;
            title[..n].fill(0);
            crate::host::copy_mapped(title.as_mut_ptr(), (address + at) as *const u8, n, false);
            // SAFETY: `[address + at, address + at + n)` lies inside the
            // buffer, so inside the copy's mapping.
            unsafe {
                std::ptr::copy_nonoverlapping(self.data.add((address + at - self.start) as usize), held.as_mut_ptr(), n);
            }
            out.bytes += n as u64;
            if title[..n] != held[..n] {
                let mut first = None;
                for (i, (a, b)) in title[..n].iter().zip(&held[..n]).enumerate() {
                    if a != b {
                        first.get_or_insert(at + i as u64);
                        out.differing += 1;
                    }
                }
                out.first = out.first.or(first);
            }
            at += n as u64;
        }
        out
    }

    /// What the copy knows of `[address, address + len)`.
    pub(crate) fn input_state(&self, address: u64, len: u64) -> State {
        let len = len.min(self.left(address));
        let (from, to) = (address as usize, (address + len) as usize);
        let mut state = State { id: self.id, carded: self.card.is_some(), in_place: self.place.is_some(), ..State::default() };
        state.pending = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|&(a, e)| e.min(to).saturating_sub(a.max(from)) as u64)
            .sum();
        let base = self.start as usize / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut lowest = u64::MAX;
        for page in from / PAGE..to.div_ceil(PAGE) {
            state.pages += 1;
            match copied.get(page - base) {
                Some(&at) if at != 0 => {
                    lowest = lowest.min(at);
                    state.taken[1] = state.taken[1].max(at);
                }
                _ => state.never += 1,
            }
        }
        state.taken[0] = if lowest == u64::MAX { 0 } else { lowest };
        state.asking = crate::watch::ahead_now(from, to);
        state
    }
}
