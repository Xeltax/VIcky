//! VIcky: a submission's refreshes of host copies by part (diagnostics: the
//! copies line's pairs). After codec-d/cool_fox dc2a3ad4 (`written_spans`)
//! and 37332f4d (`front_spans`).
//!
//! A part is the run of a submission's steps, across its lists, between two
//! steps that write title memory: a title buffer copy (`Step::Copy`) and
//! work that may write it (`Step::Run`). Inside a part nothing of ours
//! writes title memory: fills write upload memory, shader writes take a
//! snapshot of the copy, textures refresh their staging.
//!
//! A refresh step is merged per copy into the submission's batch, and done
//! when a step gives that copy to shaders (`Batch::flush_one`), or at the
//! part's end for all (`Batch::flush`). A list of dispatches records
//! "refresh this range, shaders write it" for each dispatch, so one copy
//! can be brought up to date several times in a part, each an asking of the
//! journal or the guards. The distinct (copy, part) pairs a frame are the
//! fewest updates a refresh per copy and part would need.

use std::collections::HashMap;
use std::sync::Arc;

use super::{HostList, Step};

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
    /// Whether something of it is refreshed before the part gives it to
    /// shaders.
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

/// A submission's parts ([`parts`]) as the steps of `runs` give them, in
/// the order they will run.
pub(super) fn walk(runs: &[&HostList]) -> Vec<Vec<PartCopy>> {
    let mut touches = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            touches.push(match step {
                Step::Refresh(shadow, address, len) => {
                    Touch::Refresh(Arc::as_ptr(shadow) as usize, *address, *address + *len)
                }
                Step::GpuWrite(shadow, ..) => Touch::Write(Arc::as_ptr(shadow) as usize),
                Step::Copy(..) | Step::Run(_) => Touch::Title,
                Step::Fill(..) | Step::Host(_) | Step::Texture(_) | Step::HeapPass(_) => continue,
            });
        }
    }
    parts(touches)
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

#[cfg(test)]
mod tests {
    use super::Touch::{Refresh, Title, Write};
    use super::{PartCopy, parts};

    fn copy(copy: usize, from: usize, to: usize, late: bool) -> PartCopy {
        PartCopy { copy, from, to, late }
    }

    #[test]
    fn a_part_spans_each_copy_up_to_its_first_write_by_shaders() {
        let found = parts([
            Refresh(1, 10, 20),
            Refresh(1, 12, 25),
            Refresh(2, 0, 5),
            Write(1),
            // After the part gave copy 1 to shaders.
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
        // the part lies inside the span, and each one after it is marked
        // late.
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
