//! VIcky: at the relay's late stage 3 a present waits until the flights
//! that copy textures back into title memory ([`crate::texback`]) have
//! landed, and for nothing else, before the title goes on
//! (`VICTORY_LATE_WATER=1`; off by default). `VICTORY_LATE_WATER=alt`:
//! twenty seconds without, twenty with, in turn, each change said in the
//! log (`(VICTORY_LATE_WATER=alt): ON` / `off`) with the counts of the
//! phase before.
//!
//! # Why
//!
//! With the copy-back on, the whole water surface vanished for single
//! frames at stage 3 and not at stage 2 (CuteBooty, fxceless/victory-fork
//! #863: 1 of 12 samples of a video at 2 a second, more while the boat
//! moved; none in about 5 minutes at stage 2). The two stages differ at the
//! present only ([`super::late_free_present`]). At stage 2 the present
//! settles every flight: twice in `crate::queue`'s present, and once more
//! as the present's copy is a submission that is waited for, which comes
//! after the present's forced run of what the queues still held. When the
//! call returns, everything the frame's shaders wrote is in title memory.
//! At stage 3 the copy is a flight too and the call returns at once: the
//! frame's flights land beside the title's next frame. Waiting for them is
//! what `VICTORY_LATE_WAIT=2` costs in the boat scene: 14-29 ms a frame,
//! median 23, in "settled for: a synchronous submission or present 1.0"
//! (run V4), the card's tail of the frame and its landing.
//!
//! What lands then that the title reads without a wait we can see is the
//! copy-back: the ocean's 256x256 height and displacement maps and its
//! min/max planes. The surface needs the maps: with only the two min/max
//! kinds copied back the boat floats and most of the surface around it is
//! not drawn (#863). The title's CPU reads them (about 30 code addresses,
//! the read watch of vx-layer `eee5667d`). Our own reads of that memory are
//! ordered already: the copy-backs' ranges are in the flights' written
//! ranges, so a texture, a title buffer copy or a fill that reads them
//! waits for their flight ([`super::late_blocked`]). The title's own loads
//! are not.
//!
//! # What the console guarantees, and why this wait is enough
//!
//! On the console the card's stores are in the one memory the CPU reads as
//! soon as the card has run them: nothing lands afterwards. What the title
//! is promised is order: a fence value or a bottom-of-pipe label it sees
//! means every store before it is there. The relay keeps that promise at
//! every stage: a flight's copy-backs land before the writes and signals
//! behind it. The present promises nothing about the card. But there a
//! pass's stores are in memory as soon as the card gets to it, with nothing
//! to land after, so what a frame's compute passes wrote (the ocean's wave
//! passes, whose maps its drawing samples) is there long before the
//! title's next frame reads it, and a title can lean on that without a
//! fence. Here at stage 3 a flight lands when the card and the relay get
//! to it, up to a frame late, so the next frame can read the frame
//! before's maps, or read them while the relay writes them (a landing
//! tiles a map and writes it whole).
//!
//! The wait gives back what stage 2 gave the title for the textures copied
//! back. When the present returns, every copy-back of the frame is in title
//! memory. Flights land in order, so everything before the last flight
//! that copies back has landed too: merges, read-backs, bottom-of-pipe
//! writes and the signals behind them. What it leaves to land beside the
//! next frame is the frame's tail behind that flight and the present's own
//! copy. Stage 2 waited for those as well; that is what this saves. With
//! `VICTORY_FENCE_SENT=1` the next frame's beginning value is published
//! when its submission is sent, which is after this wait: the copy-backs
//! of the frame before are in title memory by then.
//!
//! The wait stands after the present has sent everything: the forced run
//! of what the queues still held (`VICTORY_WAIT_FORCE`, on by default;
//! with `VICTORY_ASYNC_SUBMIT=1` also what its thread had taken), then the
//! present's copy. So every list of the frame that copies back is in the
//! air or landed. At the point where stage 2 first settles, a list still
//! held in a queue's line behind a wait would not be. CuteBooty tried a
//! wait of this kind (#863: 6-14 ms a frame by its counter, the dropouts
//! rare but not gone, more with `VICTORY_ASYNC_SUBMIT=1`); where it stood
//! is not said.
//!
//! Not covered: a present sent behind the queue's line
//! (`VICTORY_ASYNC_PRESENT=1`) returns before the frame is sent; with
//! `VICTORY_WAIT_FORCE=0` what a queue still holds at a present is not sent
//! (stage 2 does not settle it either). If the surface still vanishes with
//! the switch, the title reads something of the frame's tail as well, and
//! `VICTORY_LATE_WAIT=2` is the cure again. Without the relay
//! (`VICTORY_LATE_WAIT=3 VICTORY_RELAY=0`) nothing lands flight by flight:
//! a present with such a flight in the air settles every flight, as at
//! stage 2.
//!
//! The waits count as settles for "a synchronous submission or present" on
//! the late-wait line, and on the frame line's `late water` clause.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

/// `VICTORY_LATE_WATER`: 0 off, 1 on, 2 in turn (`alt`).
fn parse(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("1") => 1,
        Some("alt") => 2,
        _ => 0,
    }
}

/// [`parse`] of the switch, said once in the log.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = parse(std::env::var("VICTORY_LATE_WATER").ok().as_deref());
        if mode != 0 && super::late_free_present() {
            tracing::info!(
                "presents wait until the flights that copy textures back into title memory have landed, and only for them (VICTORY_LATE_WATER=1; alt: in turn)"
            );
        } else if mode != 0 {
            tracing::info!(
                stage = super::late_stage(),
                "VICTORY_LATE_WATER has nothing to do: it acts at late stage 3, and at this stage the present settles every flight"
            );
        }
        mode
    })
}

/// Since the last frame line ([`summary`]), and since the last change of
/// `alt`'s phase: presents looked at, presents that found flights copying
/// textures back in the air, presents that waited for them, microseconds
/// waited.
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static PHASE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn count(i: usize, n: u64) {
    COUNTS[i].fetch_add(n, Relaxed);
    PHASE[i].fetch_add(n, Relaxed);
}

/// Whether presents wait now: always with `1`; with `alt` twenty seconds
/// off, twenty on, in turn, each change logged with the counts of the
/// phase before.
fn on_now(mode: u8) -> bool {
    if mode == 1 {
        return true;
    }
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    let seconds = EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs();
    let on = (seconds / 20) % 2 == 1;
    static LAST: AtomicU8 = AtomicU8::new(2);
    if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
        let [presents, found, waited, us] = [0, 1, 2, 3].map(|i| PHASE[i].swap(0, Relaxed));
        tracing::info!(
            "presents wait for the flights that copy textures back from now (VICTORY_LATE_WATER=alt): {}; since the change before: {presents} presents, {found} of them with such flights in the air, {waited} waited for them, {:.1} ms waiting in all",
            if on { "ON" } else { "off" },
            us as f64 / 1e3
        );
    }
    on
}

/// The flights among `items` that copy textures back.
pub(super) fn carrying(items: &[super::Late]) -> usize {
    items
        .iter()
        .filter(|item| matches!(item, super::Late::Run(flight) if !flight.texture_back_ranges.is_empty()))
        .count()
}

/// At the end of the title's present, once everything is sent (the queues'
/// forced run, the present's copy): with the switch, at stage 3, waits
/// until no flight that copies textures back is in the air or being
/// landed.
pub(crate) fn at_present() {
    let mode = mode();
    if mode == 0 || !super::late_free_present() {
        return;
    }
    let on = on_now(mode);
    count(0, 1);
    let relay = super::relay();
    let found = {
        let line = super::line();
        // (Without the relay a settle takes the whole line at once, and
        // its flights are not counted while it lands them.)
        line.backs != 0 || (!relay && line.settling)
    };
    if !found {
        return;
    }
    count(1, 1);
    if !on {
        return;
    }
    let waiting = std::time::Instant::now();
    if relay {
        // Flights land in order: the last one that copies back, and every
        // one before it.
        super::relay_wait(super::WHY_SYNC, |line| line.backs == 0);
    } else {
        super::settle_now(super::WHY_SYNC);
    }
    count(2, 1);
    count(3, waiting.elapsed().as_micros() as u64);
}

/// The frame line's clause, over `frames`; empty without the switch.
pub(super) fn summary(frames: f64) -> String {
    let mode = mode();
    let [presents, found, waited, us] = [0, 1, 2, 3].map(|i| COUNTS[i].swap(0, Relaxed));
    if mode == 0 {
        return String::new();
    }
    if !super::late_free_present() {
        return "; late water idle (VICTORY_LATE_WATER: the present settles every flight at this late stage)".into();
    }
    let frames = frames.max(1.0);
    let now = match mode {
        1 => "1",
        _ if on_now(mode) => "alt, ON now",
        _ => "alt, off now",
    };
    format!(
        "; late water (VICTORY_LATE_WATER={now}): {:.2} settles a frame it caused, {:.1} ms a frame waiting; {:.2} presents a frame, {:.2} with flights copying textures back in the air",
        waited as f64 / frames,
        us as f64 / 1e3 / frames,
        presents as f64 / frames,
        found as f64 / frames
    )
}

#[cfg(test)]
mod tests {
    use super::super::{Flight, Late};
    use super::*;

    #[test]
    fn the_switch_is_one_or_alt() {
        assert_eq!(parse(None), 0);
        assert_eq!(parse(Some("1")), 1);
        assert_eq!(parse(Some(" 1 ")), 1);
        assert_eq!(parse(Some("alt")), 2);
        assert_eq!(parse(Some("0")), 0);
        assert_eq!(parse(Some("2")), 0);
    }

    /// Only flights with copy-back ranges count; a flight without them,
    /// and what follows flights in the line, do not.
    #[test]
    fn only_flights_that_copy_textures_back_are_counted() {
        let flight = |number: u64, ranges: Vec<(usize, usize)>| {
            Late::Run(Flight {
                number,
                fence: number,
                lists: Vec::new(),
                order: Vec::new(),
                card: None,
                back: Vec::new(),
                texture_back_ranges: ranges,
            })
        };
        assert_eq!(carrying(&[]), 0);
        let line = [
            flight(1, Vec::new()),
            Late::Signal(0x1000, 1),
            flight(2, vec![(0x10000, 0x50000)]),
            flight(3, Vec::new()),
            flight(4, vec![(0x60000, 0x70000), (0x80000, 0x88000)]),
            Late::Signal(0x1000, 2),
        ];
        assert_eq!(carrying(&line), 2);
        assert_eq!(carrying(&line[..2]), 0);
        assert_eq!(carrying(&line[3..]), 1);
    }
}
