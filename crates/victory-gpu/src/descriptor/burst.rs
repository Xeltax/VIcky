//! VIcky (diagnostics builds): the heap sync and the wait at its join by
//! the submission's place in the burst (lever 2, step 0, of
//! fable-2026-11-27-pf3.md).
//!
//! Profile PF3: the submitting thread waits some 30 ms a frame at the join
//! of the heap sync beside it (`super::beside`), although the sync's own
//! time (57-63 ms a frame) is less than the thread's work before the join
//! (about 68 ms). The join is submission by submission: each waits for what
//! its own sync has left when the thread's work on it is done. The guess to
//! test: the burst's first sync carries most of the slots, those written
//! during a whole frame of recording, and the later ones only what was
//! written since the submission before.
//!
//! Counted by the submission's place since the last present (1st, 2nd, ...
//! up to [`PLACES`], the last with all later ones; the present's own
//! submission, its copy to the window, apart): the sync's own time, the
//! slots it mirrored (those it found written and those `VICTORY_MIRROR_TRY`
//! left to it), and how long the submitting thread waited at the join.
//! Without `VICTORY_HEAP_SYNC_BESIDE=1` the sync runs on the submitting
//! thread, and all of its time counts as waited. One line per frame-rate
//! line, next to the sync's ([`line`]), per frame.
//!
//! A submission's place is one more than that of the submission before, and
//! 1st after the present's own (`crate::host`'s present marks it,
//! [`present_own`]) or, if a present passed without one, after that present
//! (`crate::queue::presented`). Submissions follow one another under the
//! submission lock, so each place is counted once.

#[cfg(feature = "diagnostics")]
use std::sync::Mutex;
#[cfg(feature = "diagnostics")]
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Places counted one by one; the last of them holds all later ones.
const PLACES: usize = 17;

/// By place, and the present's own last: submissions, the sync's
/// microseconds, slots mirrored, and microseconds waited at the join.
struct State {
    /// The place of the next submission, from 0.
    place: usize,
    /// Presents so far at the submission before.
    frame: u64,
    rows: [[u64; 4]; PLACES + 1],
}

impl State {
    const fn new() -> State {
        State { place: 0, frame: 0, rows: [[0; 4]; PLACES + 1] }
    }

    /// A submission ended, `frame` presents so far: the present's own
    /// (`own`) or the next in its burst, its sync having taken `us`
    /// microseconds and mirrored `slots`, and the submitting thread having
    /// waited `waited` microseconds at its join.
    fn note(&mut self, own: bool, frame: u64, [us, slots, waited]: [u64; 3]) {
        let row = if own {
            self.place = 0;
            self.frame = frame;
            PLACES
        } else {
            if self.frame != frame {
                self.frame = frame;
                self.place = 0;
            }
            let row = self.place.min(PLACES - 1);
            self.place += 1;
            row
        };
        for (total, n) in self.rows[row].iter_mut().zip([1, us, slots, waited]) {
            *total += n;
        }
    }

    /// The line per frame over `frames`, the counts taken; `None` if no
    /// submission was noted since.
    fn take(&mut self, frames: f64) -> Option<String> {
        let rows = std::mem::replace(&mut self.rows, [[0; 4]; PLACES + 1]);
        if rows.iter().all(|row| row[0] == 0) {
            return None;
        }
        let frames = frames.max(1.0);
        let per = |n: u64| n as f64 / frames;
        let places: Vec<String> = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row[0] > 0)
            .map(|(i, &[_, us, slots, waited])| {
                format!(
                    "{} {:.2} ms / {:.0} slots / {:.2} ms waited",
                    place_name(i),
                    per(us) / 1000.0,
                    per(slots),
                    per(waited) / 1000.0
                )
            })
            .collect();
        let counts: Vec<String> = rows.iter().map(|row| format!("{:.2}", per(row[0]))).collect();
        Some(format!(
            "heap sync by place in the burst: {} (per frame over {frames:.0} frames: the sync's own ms, the slots it \
             mirrored with those VICTORY_MIRROR_TRY left to it, and the submitting thread's ms waited at the join, all \
             of the sync's ms when it runs on that thread; submissions a frame by place, 1st to {} then the present's \
             own: {})",
            places.join("; "),
            place_name(PLACES - 1),
            counts.join(" ")
        ))
    }
}

/// The name of row `i` (place `i + 1`).
fn place_name(i: usize) -> String {
    if i == PLACES {
        return "the present's own".to_owned();
    }
    let n = i + 1;
    let suffix = match (n % 10, n % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    if i == PLACES - 1 { format!("{n}{suffix} and later") } else { format!("{n}{suffix}") }
}

#[cfg(feature = "diagnostics")]
static STATE: Mutex<State> = Mutex::new(State::new());

/// The last sync's microseconds and slots mirrored ([`synced`]), taken by
/// its submission ([`submitted`]).
#[cfg(feature = "diagnostics")]
static LAST: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's submissions now are the present's own ([`present_own`]).
    static OWN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The heap sync ended (`super::sync_held`): it took `us` microseconds
/// and mirrored `slots`.
#[cfg(feature = "diagnostics")]
pub(crate) fn synced(us: u64, slots: u64) {
    LAST[0].store(us, Relaxed);
    LAST[1].store(slots, Relaxed);
}

/// A submission's sync is over: joined after the submitting thread waited
/// `joined` for it (`beside`), or done on that thread.
#[cfg(feature = "diagnostics")]
pub(crate) fn submitted(beside: bool, joined: std::time::Duration) {
    let (us, slots) = (LAST[0].swap(0, Relaxed), LAST[1].swap(0, Relaxed));
    let waited = if beside { joined.as_micros() as u64 } else { us };
    STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note(OWN.get(), crate::queue::presented(), [us, slots, waited]);
}

/// While the guard lives, this thread's submissions are the present's own
/// (its copy to the window).
#[cfg(feature = "diagnostics")]
pub(crate) fn present_own() -> Own {
    OWN.set(true);
    Own
}

/// See [`present_own`].
#[cfg(feature = "diagnostics")]
pub(crate) struct Own;

#[cfg(feature = "diagnostics")]
impl Drop for Own {
    fn drop(&mut self) {
        OWN.set(false);
    }
}

/// The line next to the heap sync's (`super::sync_line`), per frame over
/// `frames`; none if no submission was noted since the last.
#[cfg(feature = "diagnostics")]
pub(crate) fn line(frames: f64) {
    let text = STATE.lock().unwrap_or_else(|e| e.into_inner()).take(frames);
    if let Some(text) = text {
        tracing::info!("{text}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Places count from the present's own, or from a present that had
    /// none; past the last place all go to it.
    #[test]
    fn submissions_are_counted_by_their_place_since_the_present() {
        let mut s = State::new();
        s.note(false, 1, [100, 10, 50]);
        s.note(false, 1, [200, 20, 0]);
        s.note(true, 2, [7, 1, 7]);
        s.note(false, 2, [300, 30, 60]);
        assert_eq!(s.rows[0], [2, 400, 40, 110]);
        assert_eq!(s.rows[1], [1, 200, 20, 0]);
        assert_eq!(s.rows[PLACES], [1, 7, 1, 7]);
        // A present without one of its own: 1st again.
        s.note(false, 3, [1, 1, 1]);
        assert_eq!(s.rows[0][0], 3);
        for _ in 0..30 {
            s.note(false, 3, [0, 0, 0]);
        }
        assert_eq!(s.rows[PLACES - 1][0], 30 - (PLACES as u64 - 2));
        assert_eq!(s.rows.iter().map(|row| row[0]).sum::<u64>(), 35);
    }

    /// The line: per frame, places with submissions only, and the counts
    /// taken.
    #[test]
    fn the_line_is_per_frame_and_taken() {
        let mut s = State::new();
        assert!(s.take(1.0).is_none());
        s.note(false, 1, [4000, 100, 2000]);
        s.note(false, 1, [1000, 10, 0]);
        s.note(true, 2, [500, 2, 500]);
        let text = s.take(2.0).expect("a line");
        assert!(text.starts_with("heap sync by place in the burst: 1st 2.00 ms / 50 slots / 1.00 ms waited; 2nd 0.50 ms"), "{text}");
        assert!(text.contains("the present's own 0.25 ms / 1 slots / 0.25 ms waited"), "{text}");
        assert!(!text.contains("3rd"), "{text}");
        assert!(s.take(2.0).is_none());
    }

    #[test]
    fn places_have_their_names() {
        let names: Vec<String> = [0, 1, 2, 3, 10, 11, 12, 13, 15, 16, PLACES].into_iter().map(place_name).collect();
        assert_eq!(
            names,
            ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "14th", "16th", "17th and later", "the present's own"]
        );
    }
}
