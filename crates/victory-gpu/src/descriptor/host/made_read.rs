//! VIcky, structural stage 4's counter (`VICTORY_MADE_READ`; counted always,
//! no behaviour change; `VICTORY_MADE_READ=0` leaves its line out).
//!
//! The title's recording threads wait some 83.5 ms a frame, summed, for a
//! heap's slot state held alone (wall2, profile PB1): `HostHeap::mirror` >
//! `made_alone` > `held_alone`, 67 ms of it from `draw::State::bind_attempt`
//! and 16.8 from `descriptor::mirror` > `mirror_tried`. Stage 4
//! (`VICTORY_MADE_READ=1|alt|check`, not written yet) would take the slot
//! state shared first and return the view when the slot was made from the
//! record already, alone only to make it. This counts, per caller, how many
//! of the `mirror` calls took that path (the slot's host slot there without
//! taking a pool slot, its key the record's: `Count::Same`), so could have
//! been answered under a shared lock. Stage 4 goes ahead if that is 70 % or
//! more of the draws' calls.
//!
//! The callers are told apart by a mark on the thread ([`As`]), set in
//! `bind_attempt` and in `descriptor::mirror`; the heap sync's makings and
//! the slots made for textures that waited count as "other". Only
//! `HostHeap::mirror` is counted: the paths that hold the state for several
//! slots (`mirror_due`, `mirror_chunk`) or try it (`VICTORY_MIRROR_TRY`'s
//! free path) do not wait there one slot at a time.

use std::cell::Cell;
use std::sync::OnceLock;

/// Who calls `HostHeap::mirror`, for the counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Caller {
    /// The heap sync, the slots that waited for a texture, the rest (a
    /// thread without a mark).
    #[allow(dead_code)]
    Other = 0,
    /// A draw reading its tables (`draw::State::bind_attempt`).
    Draw = 1,
    /// The title's view calls and descriptor copies (`descriptor::mirror`,
    /// through `mirror_tried`).
    Call = 2,
}

thread_local! {
    /// The caller this thread's `mirror` calls count for.
    static CALLER: Cell<u8> = const { Cell::new(0) };
    /// Whether the making of the `mirror` call under way only looked.
    static SAME: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// Tests: with `Some`, what this thread counts is added here too.
    static TEST_COUNTS: std::cell::RefCell<Option<[u64; 6]>> = const { std::cell::RefCell::new(None) };
}

/// While alive, this thread's `mirror` calls count as `caller`'s.
pub(crate) struct As(u8);

impl As {
    pub(crate) fn enter(caller: Caller) -> Self {
        As(CALLER.replace(caller as u8))
    }
}

impl Drop for As {
    fn drop(&mut self) {
        CALLER.set(self.0);
    }
}

/// Per caller: calls, and those that found the slot made from the record.
static COUNTS: crate::perthread::Counters<6> = crate::perthread::Counters::new();

/// Before a `mirror` call's making (the slot state held): the caller.
#[inline]
pub(crate) fn begin() -> u8 {
    SAME.set(false);
    CALLER.get()
}

/// In the making: the slot was made from the record already, and nothing
/// was done but looking (`HostHeap::mirror_view`'s `Count::Same`).
#[inline]
pub(crate) fn same() {
    SAME.set(true);
}

/// After the making: the call counted for `caller` ([`begin`]).
#[inline]
pub(crate) fn end(caller: u8) {
    let at = usize::from(caller.min(2)) * 2;
    let same = SAME.get();
    COUNTS.add(at, 1);
    if same {
        COUNTS.add(at + 1, 1);
    }
    #[cfg(test)]
    TEST_COUNTS.with_borrow_mut(|counts| {
        if let Some(counts) = counts {
            counts[at] += 1;
            counts[at + 1] += u64::from(same);
        }
    });
}

/// Whether the line is said (`VICTORY_MADE_READ=0`: not; `count`, `1` or
/// unset: said).
fn said() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MADE_READ").is_none_or(|v| v != "0"))
}

/// With the heap sync's line, once a frame-rate line: per frame over
/// `frames`, the calls by caller and the share that only looked.
pub(crate) fn line(frames: f64) {
    let counts = COUNTS.take();
    if !said() || counts.iter().all(|&n| n == 0) {
        return;
    }
    tracing::info!("{}", text(&counts, frames));
}

/// The line's text (see [`line`]).
fn text(c: &[u64; 6], frames: f64) -> String {
    let frames = frames.max(1.0);
    let share = |calls: u64, same: u64| if calls == 0 { 0.0 } else { 100.0 * same as f64 / calls as f64 };
    let part = |at: usize| format!("{:.0} calls, {:.1} %", c[at] as f64 / frames, share(c[at], c[at + 1]));
    format!(
        "the heap's slot state taken alone for one slot (HostHeap::mirror; stage 4's counter, VICTORY_MADE_READ): per \
         frame, draws (draw::State::bind_attempt) {} of them found their slot made from the record already, so could \
         have been answered under a shared lock; the title's view calls and copies (descriptor::mirror > mirror_tried) \
         {}; other callers (the heap sync, slots that waited for a texture) {}; draws and calls together {:.1} % \
         (stage 4 goes ahead if the draws' share is 70 % or more)",
        part(2),
        part(4),
        part(0),
        share(c[2] + c[4], c[3] + c[5]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A call counts for the caller marked on its thread, nested marks give
    /// back the outer one, and only a making that took the looking path
    /// counts as found.
    #[test]
    fn calls_count_by_caller_and_path() {
        let call = |same_path: bool| {
            let caller = begin();
            if same_path {
                same();
            }
            end(caller);
        };
        let c = std::thread::scope(|s| {
            s.spawn(|| {
                TEST_COUNTS.set(Some([0; 6]));
                call(true);
                {
                    let _draw = As::enter(Caller::Draw);
                    call(true);
                    call(true);
                    call(false);
                    {
                        let _call = As::enter(Caller::Call);
                        call(false);
                    }
                    call(true);
                }
                call(false);
                TEST_COUNTS.take().expect("set above")
            })
            .join()
            .expect("the counting thread")
        });
        assert_eq!(c, [2, 1, 4, 3, 1, 0], "other, draw, call: calls and found");
        let text = text(&c, 2.0);
        assert!(text.contains("draws (draw::State::bind_attempt) 2 calls, 75.0 %"), "{text}");
        assert!(text.contains("together 60.0 %"), "{text}");
    }
}
