//! Lot 4's probes P4 and P5 (`VICTORY_LOT4_PROBE=1`, `crate::lot4`), on the
//! title's views of its physical pages.
//!
//! P4, each physical page placed at its first mapping, simulated: today a
//! page's place in its section is fixed when the title allocates it
//! (`make_pages`), and the title maps its pages in its own order, so a
//! range of addresses shows pages from scattered places, which in place
//! refuses (`Cover::Scattered`: the constants heap in V3, windows over view
//! memory). Beside the real table, a page here gets a place at its first
//! mapping: its address (a page whose address another page took first gets
//! a place of its own elsewhere; the place is kept when the page is mapped
//! again, at another address or after an unmap, as the real scheme would
//! keep it). A range whose pages' places follow its addresses would be one
//! run of one section. Behaviour unchanged: only counted.
//!
//! P5, the alias census: physical pages by how many addresses show them,
//! over all and over the blocks under live host copies.

// (Called from diagnostics builds only; the lock helpers are used in all.)
#![cfg_attr(not(feature = "diagnostics"), allow(dead_code))]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;

use super::{Cover, K64, cover_in, views};

/// The simulated places: by page (section, offset of its 64 KiB), its place;
/// the places taken; the next place for a page whose address was taken;
/// pages placed at their address, and elsewhere.
#[derive(Default, Clone)]
struct Lazy {
    place: HashMap<(usize, u64), u64>,
    taken: HashSet<u64>,
    spill: u64,
    at_address: u64,
    elsewhere: u64,
}

/// Places of pages whose address another page took first: far above any
/// address.
const SPILL: u64 = 1 << 62;

static LAZY: Mutex<Option<Lazy>> = Mutex::new(None);

/// The runtime mapped `[at, at + len)` as a view of `section` from
/// `offset` (`super::title_memory_view`): its pages mapped for the first
/// time get their places.
pub(crate) fn mapped(at: usize, len: usize, section: usize, offset: u64) {
    if !crate::lot4::on() {
        return;
    }
    let mut lazy = LAZY.lock().unwrap_or_else(|e| e.into_inner());
    place(lazy.get_or_insert_with(Lazy::default), at, len, section, offset);
}

fn place(lazy: &mut Lazy, at: usize, len: usize, section: usize, offset: u64) {
    for k in 0..len.div_ceil(K64) {
        let page = (section, offset + (k * K64) as u64);
        if lazy.place.contains_key(&page) {
            continue;
        }
        let address = (at + k * K64) as u64;
        let place = if lazy.taken.insert(address) {
            lazy.at_address += 1;
            address
        } else {
            lazy.elsewhere += 1;
            lazy.spill += K64 as u64;
            SPILL + lazy.spill
        };
        lazy.place.insert(page, place);
    }
}

/// `section` was closed (`super::title_memory_section_closed`): its pages
/// and their places go with it.
pub(crate) fn section_closed(section: usize) {
    if !crate::lot4::on() {
        return;
    }
    let mut lazy = LAZY.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(lazy) = lazy.as_mut() {
        let gone: Vec<(usize, u64)> = lazy.place.keys().filter(|p| p.0 == section).copied().collect();
        for page in gone {
            if let Some(place) = lazy.place.remove(&page) {
                lazy.taken.remove(&place);
            }
        }
    }
}

/// Bytes of the sorted, disjoint `ranges` that views of physical pages hold.
pub(crate) fn view_bytes(ranges: &[(u64, u64)]) -> u64 {
    let v = views();
    let mut bytes = 0u64;
    for &(from, to) in ranges {
        let (from, to) = (from as usize, to as usize);
        let first = v.views.range(..=from).next_back().filter(|(base, view)| **base + view.0 > from).map(|(base, _)| *base);
        let start = first.unwrap_or(from);
        for (&base, &(len, _, _)) in v.views.range(start..to) {
            let (lo, hi) = (base.max(from), (base + len).min(to));
            if lo < hi {
                bytes += (hi - lo) as u64;
            }
        }
    }
    bytes
}

/// How `[at, at + len)` lies in the views: `None` if no view holds its first
/// byte; else today's cover, its runs of a section today (pages whose
/// places follow one another), and whether the simulated places make it one
/// run.
fn placement(views: &BTreeMap<usize, (usize, usize, u64)>, lazy: &Lazy, at: usize, len: usize) -> Option<(Cover, usize, bool)> {
    let today = cover_in(views, at, len);
    if today == Cover::None {
        return None;
    }
    let end = at + len.max(1);
    let mut page = at / K64 * K64;
    let (mut runs, mut next) = (0usize, None);
    let (mut delta, mut one) = (None, true);
    while page < end {
        let Some((&base, &(bytes, section, offset))) = views.range(..=page).next_back().filter(|(base, view)| **base + view.0 > page) else {
            one = false;
            // A hole: the run starts again after it.
            next = None;
            page += K64;
            continue;
        };
        let view_end = (base + bytes).min(end);
        while page < view_end {
            let off = offset + (page - base) as u64;
            if next != Some((section, off)) {
                runs += 1;
            }
            next = Some((section, off + K64 as u64));
            match lazy.place.get(&(section, off)) {
                Some(&place) => {
                    let d = i128::from(place) - page as i128;
                    match delta {
                        None => delta = Some(d),
                        Some(was) if was != d => one = false,
                        _ => {}
                    }
                }
                None => one = false,
            }
            page += K64;
        }
    }
    Some((today, runs, one))
}

/// Per kind of range: ranges over view memory and their bytes; bytes one
/// run today, scattered, with holes; bytes one run under the simulated
/// places, and of the bytes not one run today those that would be.
#[derive(Default)]
struct Tally {
    ranges: u64,
    bytes: u64,
    run: u64,
    scattered: u64,
    holes: u64,
    lazy_run: u64,
    lazy_gained: u64,
}

impl Tally {
    fn add(&mut self, bytes: u64, (today, _, one): (Cover, usize, bool)) {
        self.ranges += 1;
        self.bytes += bytes;
        match today {
            Cover::Run(..) => self.run += bytes,
            Cover::Scattered => self.scattered += bytes,
            _ => self.holes += bytes,
        }
        if one {
            self.lazy_run += bytes;
            if !matches!(today, Cover::Run(..)) {
                self.lazy_gained += bytes;
            }
        }
    }

    fn text(&self, name: &str) -> String {
        let mb = |b: u64| b as f64 / (1 << 20) as f64;
        let refused = self.bytes - self.run;
        format!(
            "{name}: {} over view memory, {:.0} MB: today one run {:.0} MB, scattered {:.0} MB, with holes {:.0} MB; \
             with the places simulated one run {:.0} MB, of the {:.0} MB not one run today {:.0} MB ({:.1} %)",
            self.ranges,
            mb(self.bytes),
            mb(self.run),
            mb(self.scattered),
            mb(self.holes),
            mb(self.lazy_run),
            mb(refused),
            mb(self.lazy_gained),
            crate::lot4::pct(self.lazy_gained, refused),
        )
    }
}

/// P4's line, over the live host copies `copies` (start, bytes, in the
/// card's memory, a window): the windows, and the title buffers of the
/// constants heap's size (`VICTORY_INPLACE_CONSTANTS_MIN_MB`, 64 MB) and
/// more, each range once.
pub(crate) fn placement_line(copies: &[(u64, u64, bool, bool)]) {
    let views = views().views.clone();
    let lazy = LAZY.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default();
    let mut seen = HashSet::new();
    let (mut windows, mut big) = (Tally::default(), Tally::default());
    let mut largest: Vec<(u64, u64, Cover, usize, bool)> = Vec::new();
    for &(start, len, _, window) in copies {
        if !seen.insert((start, len, window)) || (!window && len < super::constants_min()) {
            continue;
        }
        let Some(p) = placement(&views, &lazy, start as usize, len as usize) else {
            continue;
        };
        if window {
            windows.add(len, p);
        } else {
            big.add(len, p);
            largest.push((len, start, p.0, p.1, p.2));
        }
    }
    largest.sort_by_key(|l| std::cmp::Reverse(l.0));
    let listed: Vec<String> = largest
        .iter()
        .take(3)
        .map(|&(len, start, today, runs, one)| {
            format!(
                "{:.0} MB at {start:#x}: today {} in {runs} runs, simulated {}",
                len as f64 / (1 << 20) as f64,
                match today {
                    Cover::Run(..) => "one run",
                    Cover::Scattered => "scattered",
                    _ => "with holes",
                },
                if one { "one run" } else { "not one run" }
            )
        })
        .collect();
    tracing::info!(
        "lot4 probe P4: each physical page placed at its first mapping (simulated, behaviour unchanged): {} pages placed \
         at their address, {} elsewhere (their address taken first); {}; {}; kill under half of the bytes not one run \
         today; largest title buffers: {}",
        lazy.at_address,
        lazy.elsewhere,
        windows.text("windows"),
        big.text("title buffers of the constants heap's size and more"),
        if listed.is_empty() { "none over view memory".into() } else { listed.join("; ") },
    );
}

/// P5's line: physical pages by how many addresses show them (1, 2, 3, 4 or
/// more), over all and over the blocks `under` live host copies (sorted,
/// disjoint), with the bytes shown at more than one address; and the
/// sections of physical pages.
pub(crate) fn alias_line(under: &[(u64, u64)]) {
    let v = views();
    let (mut all, mut copied) = ([0u64; 4], [0u64; 4]);
    for addresses in v.pages.values() {
        let n = addresses.len().clamp(1, 4) - 1;
        all[n] += 1;
        if addresses.iter().any(|&a| crate::lot4::covered_by(under, a as u64)) {
            copied[n] += 1;
        }
    }
    let (sections, bytes, committed) = v.sections.values().fold((0u64, 0u64, 0u64), |(n, b, c), &(bytes, com)| (n + 1, b + bytes, c + u64::from(com)));
    drop(v);
    let mb = |pages: u64| (pages * K64 as u64) as f64 / (1 << 20) as f64;
    let text = |c: &[u64; 4]| {
        format!(
            "{} pages: at one address {}, two {}, three {}, four or more {}; {:.0} MB shown at more than one address",
            c.iter().sum::<u64>(),
            c[0],
            c[1],
            c[2],
            c[3],
            mb(c[1] + c[2] + c[3]),
        )
    };
    tracing::info!(
        "lot4 probe P5: physical pages mapped now: {}; of them under live host copies: {}; sections of physical pages: \
         {sections}, {:.0} MB ({committed} made committed)",
        text(&all),
        text(&copied),
        bytes as f64 / (1 << 20) as f64,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pages mapped in address order at their first mapping make one run,
    /// whatever their places in their sections; a page whose address
    /// another took first does not.
    #[test]
    fn first_mappings_in_address_order_make_one_run() {
        let k = K64;
        let base = 0x1000_0000usize;
        let mut lazy = Lazy::default();
        // Four pages of section 7 at scattered offsets, mapped at four
        // addresses in a row, one view each.
        let offsets = [5u64, 2, 9, 0].map(|o| o * k as u64);
        let mut views = BTreeMap::new();
        for (i, &o) in offsets.iter().enumerate() {
            views.insert(base + i * k, (k, 7usize, o));
            place(&mut lazy, base + i * k, k, 7, o);
        }
        let (today, runs, one) = placement(&views, &lazy, base, 4 * k).unwrap();
        assert_eq!(today, Cover::Scattered);
        assert_eq!(runs, 4);
        assert!(one, "simulated: one run");
        // A page first mapped elsewhere keeps that place when mapped here.
        let mut lazy2 = Lazy::default();
        place(&mut lazy2, 0x2000_0000, k, 7, offsets[1]);
        for (i, &o) in offsets.iter().enumerate() {
            place(&mut lazy2, base + i * k, k, 7, o);
        }
        assert!(!placement(&views, &lazy2, base, 4 * k).unwrap().2);
        // Another page first at the same address goes elsewhere.
        let mut lazy3 = Lazy::default();
        place(&mut lazy3, base, k, 8, 0);
        place(&mut lazy3, base, k, 7, offsets[0]);
        assert_eq!((lazy3.at_address, lazy3.elsewhere), (1, 1));
        // Not view memory: nothing.
        assert!(placement(&views, &lazy, 0x5000_0000, k).is_none());
    }

    /// One view of one run of a section is one run both ways.
    #[test]
    fn a_run_today_stays_one() {
        let k = K64;
        let mut views = BTreeMap::new();
        views.insert(0x4000_0000usize, (8 * k, 3usize, 16 * k as u64));
        let mut lazy = Lazy::default();
        place(&mut lazy, 0x4000_0000, 8 * k, 3, 16 * k as u64);
        let (today, runs, one) = placement(&views, &lazy, 0x4000_0000 + k, 4 * k).unwrap();
        assert!(matches!(today, Cover::Run(3, _)));
        assert_eq!(runs, 1);
        assert!(one);
        let mut t = Tally::default();
        t.add(4 * k as u64, (today, runs, one));
        assert_eq!((t.run, t.lazy_run, t.lazy_gained), (4 * k as u64, 4 * k as u64, 0));
    }
}
