//! VIcky (`VICTORY_CARD_DIRTY_LIST=1|alt|check`; off by default): the
//! copies in the card's memory whose CPU side changed since the last take
//! ([`super::card_dirty`]) are kept in a list as they change, and the take
//! visits those instead of every copy.
//!
//! Profile PF3 (lever 3): `card_dirty` walked every card copy at each of
//! some nine submissions a frame, a `retain` over all their weak references
//! that took each copy's lock, 5.5 ms a frame of the submitting thread,
//! nearly all of it finding nothing to take.
//!
//! A copy goes on the list under its notes' lock ([`super::DirtyNotes`]),
//! put there by the change that finds it off the list ([`Listing`]). The
//! take empties the list and, for each copy on it, takes its notes and
//! marks it off the list in the same lock. So a change noted before a take
//! began is on the list that take empties, and one noted after its copy was
//! taken puts it on the list again: the take finds what the walk found, in
//! another order. A copy is known to its notes once it is among the copies
//! in the card's memory ([`carded`]); one with notes by then is listed
//! there.
//!
//! `check`: the list is taken, then every copy walked as before; a copy the
//! walk finds with notes and off the list is counted (and taken, as the walk
//! took it). `alt`: twenty seconds the walk, twenty the list, in turn, each
//! change said in the log; the list is kept in both.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::{CARDED, DirtyNotes, PatchPage, RELAY_FETCHED, Shadow, joined};

/// Per copy in the card's memory, in its notes ([`DirtyNotes`]): the copy
/// itself, once it is among the copies in the card's memory with the list
/// kept ([`carded`]), and whether it is on the list.
#[derive(Default)]
pub(super) struct Listing {
    me: Option<Weak<Shadow>>,
    listed: bool,
}

impl Listing {
    /// Changes were noted, under the notes' lock: the copy to put on the
    /// list ([`list`]) if it is known and not on it yet.
    pub(super) fn to_list(&mut self) -> Option<Weak<Shadow>> {
        if self.listed {
            return None;
        }
        let me = self.me.clone()?;
        self.listed = true;
        Some(me)
    }
}

/// The copies with changes not taken yet, each once ([`Listing::listed`]).
/// Locked inside a copy's notes' lock, never around it.
static LIST: Mutex<Vec<Weak<Shadow>>> = Mutex::new(Vec::new());

/// `VICTORY_CARD_DIRTY_LIST`: 0 off, 1 on, 2 in turn, 3 check.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_CARD_DIRTY_LIST").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            Ok("check") => 3,
            _ => return 0,
        };
        tracing::info!(
            "copies in the card's memory whose CPU side changed are kept in a list as they change, and taken from it \
             before each submission instead of walking every copy (VICTORY_CARD_DIRTY_LIST={})",
            match mode {
                1 => "1",
                2 => "alt: twenty seconds the walk, twenty the list",
                _ => "check: the list, then the walk over every copy, counting copies with changes the list did not hold",
            }
        );
        mode
    })
}

/// Whether the list is kept (any mode; in both phases of `alt`).
pub(super) fn kept() -> bool {
    mode() != 0
}

/// Whether this take uses the list (`alt`: by the phase, each change said
/// with the counts of the phase before).
fn listed_take() -> bool {
    match mode() {
        0 => false,
        2 => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: AtomicU8 = AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                let [takes, us, looked, found] = [0, 1, 2, 3].map(|i| PHASE[i].swap(0, Ordering::Relaxed));
                let per = |n: u64| n as f64 / takes.max(1) as f64;
                tracing::info!(
                    "copies in the card's memory with changes are taken from their list from now \
                     (VICTORY_CARD_DIRTY_LIST=alt): {}; since the change before: {takes} takes, {:.1} us a take, \
                     {:.1} copies looked at a take, {:.1} with something to send",
                    if on { "ON" } else { "off, every copy walked as usual (the list is kept)" },
                    per(us),
                    per(looked),
                    per(found)
                );
            }
            on
        }
        _ => true,
    }
}

/// Since the card line before ([`clause`]): takes, their microseconds,
/// copies looked at, copies with something to send, and (`check`) copies
/// with changes the list did not hold.
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// The same four first, since the last change of `alt`'s phase.
static PHASE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Puts a copy on the list ([`Listing::to_list`] said so).
pub(super) fn list(me: Weak<Shadow>) {
    LIST.lock().unwrap_or_else(|e| e.into_inner()).push(me);
}

/// A copy joins the copies in the card's memory: with the list kept, it is
/// known to its notes from now, and listed if it has notes already.
pub(super) fn carded(shadow: &Arc<Shadow>) {
    CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(shadow));
    if !kept() {
        return;
    }
    let Some(card) = shadow.card.as_ref() else {
        return;
    };
    let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
    dirty.listing.me = Some(Arc::downgrade(shadow));
    if (!dirty.ranges.is_empty() || !dirty.patches.is_empty())
        && let Some(me) = dirty.listing.to_list()
    {
        list(me);
    }
}

/// [`super::card_dirty`] with the list kept: from the list, or (`alt`'s
/// other phase) by the walk; with `check`, the walk after the list.
pub(super) fn card_dirty(
    sent_as: u64,
    patches: &mut Vec<(Arc<Shadow>, Vec<PatchPage>)>,
) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let started = std::time::Instant::now();
    let mut out = Vec::new();
    let mut missed = 0u64;
    let looked = if listed_take() {
        let batch = std::mem::take(&mut *LIST.lock().unwrap_or_else(|e| e.into_inner()));
        let fetched = RELAY_FETCHED.load(Ordering::Acquire);
        let mut looked = 0u64;
        for weak in batch {
            let Some(shadow) = weak.upgrade() else {
                continue;
            };
            looked += 1;
            let taken = shadow.card.as_ref().map(|c| {
                c.take_dirty(sent_as, fetched, &mut |d: &mut DirtyNotes| {
                    d.listing.listed = false;
                    true
                })
            });
            let (ranges, patched) = taken.unwrap_or_default();
            super::card_taken(shadow, ranges, patched, patches, &mut out);
        }
        if mode() == 3 {
            let (mut more, mut more_patches) = (Vec::new(), Vec::new());
            super::card_dirty_walk(sent_as, &mut more_patches, &mut more, |d: &mut DirtyNotes| {
                let missing = !d.listing.listed && (!d.ranges.is_empty() || !d.patches.is_empty());
                missed += u64::from(missing);
                missing
            });
            merge(&mut out, more, patches, more_patches);
        }
        looked
    } else {
        super::card_dirty_walk(sent_as, patches, &mut out, |_: &mut DirtyNotes| true)
    };
    let us = started.elapsed().as_micros() as u64;
    let found = out.len() as u64;
    for (i, n) in [1, us, looked, found].into_iter().enumerate() {
        STATS[i].fetch_add(n, Ordering::Relaxed);
        PHASE[i].fetch_add(n, Ordering::Relaxed);
    }
    STATS[4].fetch_add(missed, Ordering::Relaxed);
    out
}

/// `check`: what the walk took after the list (copies the list did not
/// hold) joined with what the list gave, a copy once in each.
fn merge(
    out: &mut Vec<(Arc<Shadow>, Vec<(usize, usize)>)>,
    more: Vec<(Arc<Shadow>, Vec<(usize, usize)>)>,
    patches: &mut Vec<(Arc<Shadow>, Vec<PatchPage>)>,
    more_patches: Vec<(Arc<Shadow>, Vec<PatchPage>)>,
) {
    for (shadow, ranges) in more {
        match out.iter_mut().find(|(s, _)| Arc::ptr_eq(s, &shadow)) {
            Some((_, had)) => {
                had.extend(ranges);
                *had = joined(std::mem::take(had));
            }
            None => out.push((shadow, ranges)),
        }
    }
    for (shadow, pages) in more_patches {
        match patches.iter_mut().find(|(s, _)| Arc::ptr_eq(s, &shadow)) {
            Some((_, had)) => had.extend(pages),
            None => patches.push((shadow, pages)),
        }
    }
}

/// The card line's words on the list (empty without it), since the line
/// before.
pub(crate) fn clause() -> String {
    if !kept() {
        return String::new();
    }
    let [takes, us, looked, found, missed] = [0, 1, 2, 3, 4].map(|i| STATS[i].swap(0, Ordering::Relaxed));
    let per = |n: u64| n as f64 / takes.max(1) as f64;
    format!(
        "; copies with changes taken from their list (VICTORY_CARD_DIRTY_LIST={}): {takes} takes since the line before, \
         {:.1} us a take, {:.1} copies looked at a take, {:.1} with something to send{}",
        match mode() {
            1 => "1",
            2 => "alt",
            _ => "check",
        },
        per(us),
        per(looked),
        per(found),
        if mode() == 3 {
            format!(", {missed} copies with changes the list did not hold (check)")
        } else {
            String::new()
        }
    )
}

#[cfg(test)]
mod tests {
    use super::Listing;
    use std::sync::Weak;

    /// A copy goes on the list once until it is taken, and never before it
    /// is known to its notes.
    #[test]
    fn a_copy_is_listed_once_until_taken() {
        let mut listing = Listing::default();
        assert!(listing.to_list().is_none(), "not known yet");
        assert!(!listing.listed);
        listing.me = Some(Weak::new());
        assert!(listing.to_list().is_some());
        assert!(listing.to_list().is_none(), "on the list already");
        assert!(listing.to_list().is_none());
        // The take marks it off the list in the notes' lock.
        listing.listed = false;
        assert!(listing.to_list().is_some());
    }
}
