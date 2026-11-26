//! VIcky: a submission's refreshes of host copies by part, and
//! `VICTORY_REFRESH_FRONT=1` (off by default), which brings each copy up to
//! date once per part, ahead of the part's steps. After codec-d/cool_fox
//! dc2a3ad4 (`refresh_ahead`, `written_spans`) and 37332f4d (`refresh_front`,
//! `front_spans`); theirs runs on threads, this one on the submitting thread.
//!
//! A part is the run of a submission's steps, across its lists, between two
//! steps that write title memory: a title buffer copy (`Step::Copy`) and
//! work that may write it (`Step::Run`). Inside a part nothing of ours
//! writes title memory: fills write upload memory, shader writes take a
//! snapshot of the copy, textures refresh their staging.
//!
//! Without the switch a refresh step is merged per copy into the
//! submission's batch, and done when a step gives that copy to shaders
//! (`Batch::flush_one`), or at the part's end for all (`Batch::flush`). A
//! list of dispatches records "refresh this range, shaders write it" for
//! each dispatch, so one copy can be brought up to date many times in a
//! part, each an asking of the journal or the guards.
//!
//! With it, at the start of each part, every copy the part refreshes is
//! brought up to date once, over the span of its refreshes that come before
//! the part first gives it to shaders; the part's refresh steps inside that
//! span are then skipped. What a part refreshes of a copy after giving it to
//! its shaders is left where it stands (in the batch, done around the ranges
//! the shaders write), as without the switch.
//!
//! The rule is `VICTORY_REFRESH_ONCE`'s (on with this switch: a range
//! brought up to date in a submission is not brought again until a step of
//! ours writes title memory there), and `VICTORY_ASK_ONCE`'s (on by
//! default): what the title's other threads write while a submission is
//! being prepared is taken by the next submission. A refresh comes earlier
//! than its step, but never across a step that writes title memory: each
//! part is brought up to date only once the step before it has run (a title
//! copy notes its bytes in the journal, `crate::watch::wrote`; other work
//! ends the submission's one asking and clears what counts as up to date).
//!
//! `VICTORY_REFRESH_FRONT_CHECK=1` (diagnostics) does one part in 16 both
//! ways: the part is brought up to date ahead, and each of its refresh steps
//! that the front covers is done again at its place, the copy's bytes
//! compared before and after.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

use super::{Batch, HostList, Refreshed, Step};
use crate::shadow::Shadow;

/// `VICTORY_REFRESH_FRONT=1`.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_FRONT").is_some_and(|v| v == "1"))
}

/// `VICTORY_REFRESH_FRONT_CHECK=1`, with the switch.
#[cfg(feature = "diagnostics")]
fn check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    on() && *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_FRONT_CHECK").is_some_and(|v| v == "1"))
}

/// What a step is to the refreshes by part ([`parts`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Touch {
    /// A copy (by address) is brought up to date over `[from, to)`.
    Refresh(usize, usize, usize),
    /// The copy is given to shaders.
    Write(usize),
    /// Title memory is written: the part ends.
    Title,
}

/// A copy refreshed in a part: the span of its refreshes before the part
/// first gave it to shaders (`from >= to`: none), and whether the part
/// refreshed it after that too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PartCopy {
    pub(super) copy: usize,
    pub(super) from: usize,
    pub(super) to: usize,
    pub(super) late: bool,
}

impl PartCopy {
    /// Whether something of it is brought up to date ahead of the part.
    fn ahead(&self) -> bool {
        self.from < self.to
    }
}

/// The copies each part of `steps` refreshes, in the order they are first
/// met (refreshed or given to shaders); a copy given to shaders and never
/// refreshed in a part is left out.
pub(super) fn parts(steps: impl IntoIterator<Item = Touch>) -> Vec<Vec<PartCopy>> {
    let mut parts = Vec::new();
    let mut part: Vec<PartCopy> = Vec::new();
    // Whether the part gave each of `part` (by place) to shaders yet.
    let mut given: Vec<bool> = Vec::new();
    let mut at: HashMap<usize, usize> = HashMap::new();
    let mut last = usize::MAX;
    let finish = |parts: &mut Vec<Vec<PartCopy>>, part: &mut Vec<PartCopy>| {
        part.retain(|c| c.ahead() || c.late);
        parts.push(std::mem::take(part));
    };
    for step in steps {
        match step {
            Touch::Refresh(copy, from, to) => {
                // Most come several times on end.
                let i = if part.get(last).is_some_and(|c| c.copy == copy) {
                    last
                } else {
                    *at.entry(copy).or_insert_with(|| {
                        part.push(PartCopy { copy, from: usize::MAX, to: 0, late: false });
                        given.push(false);
                        part.len() - 1
                    })
                };
                last = i;
                let c = &mut part[i];
                if given[i] {
                    c.late = true;
                } else {
                    (c.from, c.to) = (c.from.min(from), c.to.max(to));
                }
            }
            Touch::Write(copy) => {
                let i = *at.entry(copy).or_insert_with(|| {
                    part.push(PartCopy { copy, from: usize::MAX, to: 0, late: false });
                    given.push(false);
                    part.len() - 1
                });
                given[i] = true;
            }
            Touch::Title => {
                finish(&mut parts, &mut part);
                given.clear();
                at.clear();
                last = usize::MAX;
            }
        }
    }
    finish(&mut parts, &mut part);
    parts
}

/// A submission's parts ([`parts`]) as its lists' steps give them, and the
/// copies they name (kept only when they are to be brought up to date).
pub(super) struct Walk {
    pub(super) parts: Vec<Vec<PartCopy>>,
    copies: HashMap<usize, Arc<Shadow>>,
}

/// Reads the steps of `runs`, in the order they will run.
pub(super) fn walk(runs: &[&HostList], keep: bool) -> Walk {
    let mut copies: HashMap<usize, Arc<Shadow>> = HashMap::new();
    let mut touches = Vec::new();
    for list in runs {
        let mut last = 0;
        for step in &list.deferred().copies {
            touches.push(match step {
                Step::Refresh(shadow, address, len) => {
                    let key = Arc::as_ptr(shadow) as usize;
                    if keep && key != last {
                        copies.entry(key).or_insert_with(|| shadow.clone());
                        last = key;
                    }
                    Touch::Refresh(key, *address, *address + *len)
                }
                Step::GpuWrite(shadow, ..) => Touch::Write(Arc::as_ptr(shadow) as usize),
                Step::Copy(..) | Step::Run(_) => Touch::Title,
                Step::Fill(..) | Step::Host(_) | Step::Texture(_) | Step::HeapPass(_) => continue,
            });
        }
    }
    Walk { parts: parts(touches), copies }
}

/// Diagnostics: the frame line's (copy, part) pairs of a submission;
/// `views` are the copies (by address) its one asking found to be view
/// memory the guards hold.
#[cfg(feature = "diagnostics")]
pub(super) fn count(parts: &[Vec<PartCopy>], views: &[usize]) {
    use crate::copystat::{C, add};
    let views: std::collections::HashSet<usize> = views.iter().copied().collect();
    let (mut named, mut pairs, mut viewed, mut late) = (0, 0, 0, 0);
    for part in parts.iter().filter(|p| !p.is_empty()) {
        named += 1;
        for c in part {
            pairs += 1;
            viewed += u64::from(views.contains(&c.copy));
            late += u64::from(c.late);
        }
    }
    add(C::FrontParts, named);
    add(C::FrontPairs, pairs);
    add(C::FrontPairsViews, viewed);
    add(C::FrontPairsLate, late);
}

impl Batch {
    /// Before the steps of a submission: its parts as [`walk`] found them
    /// (with the copies kept), and the first part's copies brought up to
    /// date.
    pub(super) fn front_begin(&mut self, walk: Walk) {
        let Walk { parts, copies } = walk;
        self.front_parts = parts
            .into_iter()
            .rev()
            .map(|part| {
                part.into_iter()
                    .filter(PartCopy::ahead)
                    .filter_map(|c| Some((copies.get(&c.copy)?.clone(), c.from, c.to)))
                    .collect()
            })
            .collect();
        self.front_next();
    }

    /// At the start of the submission, and after each step that wrote title
    /// memory: the next part's copies brought up to date, each once over its
    /// span, and noted so that the part's refresh steps inside it are
    /// skipped ([`Batch::front_covers`]). A span inside what is up to date in
    /// the submission already, and not written by a step of ours since, is
    /// left alone.
    pub(super) fn front_next(&mut self) {
        self.front.clear();
        let Some(part) = self.front_parts.pop() else {
            return;
        };
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
            static TURN: AtomicU64 = AtomicU64::new(0);
            self.front_checked = check() && TURN.fetch_add(1, Relaxed).is_multiple_of(16);
            if self.front_checked {
                CHECKS[PARTS].fetch_add(1, Relaxed);
            }
        }
        let mut due = Vec::with_capacity(part.len());
        for (shadow, from, to) in part {
            let key = Arc::as_ptr(&shadow) as usize;
            self.front.insert(key, (from, to));
            if self.current.get(&key).is_some_and(|&(_, a, b)| a <= from && to <= b) {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::FrontKept, 1);
                continue;
            }
            due.push((shadow, from, to));
        }
        // One write-watch asking for the copies that lie close together, as
        // a flush makes (none while the submission's one asking stands).
        let ahead = due.len() > 1;
        if ahead {
            let started = std::time::Instant::now();
            let ranges: Vec<(usize, usize)> = due.iter().filter(|r| !r.0.unwatched()).map(|r| (r.1, r.2)).collect();
            crate::watch::ask_ahead(&ranges);
            let us = started.elapsed().as_micros() as u64;
            super::preparing(0, us);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::AskAheadUs, us);
        }
        for (shadow, from, to) in due {
            Self::update(&shadow, from, to, Refreshed::Front);
            Self::brought(&mut self.current, &shadow, from, to);
        }
        if ahead {
            crate::watch::ask_ahead_done();
        }
    }

    /// Whether the refresh of `[address, address + len)` of `shadow`'s copy
    /// was brought up to date ahead of the part's steps (and is skipped). In
    /// a part being checked it is done again here, compared.
    pub(super) fn front_covers(&self, shadow: &Arc<Shadow>, address: usize, len: usize) -> bool {
        let Some(&(from, to)) = self.front.get(&(Arc::as_ptr(shadow) as usize)) else {
            return false;
        };
        if address < from || address + len > to {
            return false;
        }
        #[cfg(feature = "diagnostics")]
        if self.front_checked {
            check_at_step(shadow, address, len);
        }
        true
    }

    /// The submission's steps have all run: every part was met.
    pub(super) fn front_end(&mut self) {
        self.front.clear();
        if self.front_parts.is_empty() {
            return;
        }
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            if CHECKS[UNMET].fetch_add(1, Relaxed) < 4 {
                tracing::warn!(
                    left = self.front_parts.len(),
                    "VICTORY_REFRESH_FRONT: a submission's steps ended before all the parts its walk found were met"
                );
            }
        }
        self.front_parts.clear();
    }
}

/// Diagnostics (`VICTORY_REFRESH_FRONT_CHECK=1`), since the start: parts
/// checked, refresh steps done again at their place, bytes compared, bytes
/// that differed, steps that found any, and submissions whose parts were
/// not all met ([`Batch::front_end`]).
#[cfg(feature = "diagnostics")]
static CHECKS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
#[cfg(feature = "diagnostics")]
const PARTS: usize = 0;
#[cfg(feature = "diagnostics")]
const STEPS: usize = 1;
#[cfg(feature = "diagnostics")]
const COMPARED: usize = 2;
#[cfg(feature = "diagnostics")]
const DIFFERED: usize = 3;
#[cfg(feature = "diagnostics")]
const STEPS_DIFFERING: usize = 4;
#[cfg(feature = "diagnostics")]
const UNMET: usize = 5;

/// The most bytes of a refresh step compared ([`check_at_step`]).
#[cfg(feature = "diagnostics")]
const CHECK_MOST: usize = 256 << 10;

/// A refresh step the front covered, done again at its place: what the
/// update changed in the copy (title memory written since the front by the
/// title's other threads, which the next submission takes without the
/// switch too, or something the front missed).
#[cfg(feature = "diagnostics")]
fn check_at_step(shadow: &Shadow, address: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let (compared, differed) = shadow.update_compared(address, len, CHECK_MOST);
    CHECKS[STEPS].fetch_add(1, Relaxed);
    CHECKS[COMPARED].fetch_add(compared as u64, Relaxed);
    if differed > 0 {
        CHECKS[DIFFERED].fetch_add(differed as u64, Relaxed);
        let n = CHECKS[STEPS_DIFFERING].fetch_add(1, Relaxed) + 1;
        if n <= 16 || n.is_power_of_two() {
            tracing::info!(
                n,
                copy = shadow.id(),
                address = format_args!("{address:#x}"),
                len,
                differed,
                compared,
                "VICTORY_REFRESH_FRONT_CHECK: a refresh step done again at its place changed bytes the front had brought"
            );
        }
    }
}

/// The copies line's words on the check, if it is on.
#[cfg(feature = "diagnostics")]
pub(crate) fn check_clause() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    if !check() {
        return String::new();
    }
    let c: Vec<u64> = CHECKS.iter().map(|a| a.load(Relaxed)).collect();
    format!(
        "; check since the start (VICTORY_REFRESH_FRONT_CHECK=1): {} parts, {} refresh steps done again at their \
         place, {:.1} MB compared, {} bytes differed in {} steps; {} submissions with parts not met",
        c[PARTS],
        c[STEPS],
        c[COMPARED] as f64 / 1e6,
        c[DIFFERED],
        c[STEPS_DIFFERING],
        c[UNMET],
    )
}

#[cfg(test)]
mod tests {
    use super::Touch::{Refresh, Title, Write};
    use super::{PartCopy, parts};

    fn copy(copy: usize, from: usize, to: usize, late: bool) -> PartCopy {
        PartCopy { copy, from, to, late }
    }

    #[test]
    fn a_part_brings_each_copy_up_to_its_first_write_by_shaders() {
        let found = parts([
            Refresh(1, 10, 20),
            Refresh(1, 12, 25),
            Refresh(2, 0, 5),
            Write(1),
            // After the part gave copy 1 to shaders: left to the batch.
            Refresh(1, 0, 100),
            Refresh(2, 30, 40),
            Write(2),
            Refresh(1, 200, 300),
        ]);
        assert_eq!(found, [vec![copy(1, 10, 25, true), copy(2, 0, 40, false)]]);
    }

    #[test]
    fn a_step_that_writes_title_memory_starts_a_part() {
        let found = parts([
            Refresh(1, 10, 20),
            Write(1),
            Title,
            // The next part is brought up to date again, its own way.
            Refresh(1, 30, 40),
            Refresh(1, 0, 5),
            Title,
            Title,
            Write(3),
            Refresh(3, 1, 2),
            Refresh(4, 7, 9),
        ]);
        assert_eq!(
            found,
            [
                vec![copy(1, 10, 20, false)],
                vec![copy(1, 0, 40, false)],
                vec![],
                // Copy 3 was given to shaders before any refresh: nothing of it ahead.
                vec![copy(3, usize::MAX, 0, true), copy(4, 7, 9, false)],
            ]
        );
    }

    #[test]
    fn copies_only_written_are_left_out_and_order_is_kept() {
        let found = parts([Write(9), Refresh(5, 0, 1), Refresh(3, 4, 6), Refresh(5, 8, 9), Write(7)]);
        assert_eq!(found, [vec![copy(5, 0, 9, false), copy(3, 4, 6, false)]]);
        assert_eq!(parts([]), [Vec::<PartCopy>::new()]);
    }

    #[test]
    fn the_span_ahead_covers_every_refresh_before_the_first_write() {
        // Whatever the order, each refresh before its copy's first write in
        // the part lies inside the span ahead, and each one after it is
        // marked late (so nothing is skipped that was not brought ahead).
        let steps = [
            Refresh(1, 50, 60),
            Refresh(2, 5, 6),
            Refresh(1, 0, 10),
            Refresh(2, 100, 200),
            Write(2),
            Refresh(1, 70, 80),
            Refresh(2, 0, 1),
            Write(1),
            Refresh(1, 90, 95),
        ];
        let found = parts(steps);
        let mut given = std::collections::HashSet::new();
        for step in steps {
            match step {
                Refresh(c, from, to) => {
                    let p = found[0].iter().find(|p| p.copy == c).unwrap();
                    if given.contains(&c) {
                        assert!(p.late);
                    } else {
                        assert!(p.from <= from && to <= p.to);
                    }
                }
                Write(c) => {
                    given.insert(c);
                }
                Title => unreachable!(),
            }
        }
    }
}
