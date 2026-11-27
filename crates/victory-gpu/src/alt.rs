//! VIcky: opt-in switches with an `alt` mode, for the ports of lot 4
//! (`VICTORY_SPANS_QUICK`, `VICTORY_SYNC_FINE`, `VICTORY_CARD_AWAKE`,
//! `VICTORY_HELPER_PRIORITY`, `VICTORY_WIDE_THREADS`, `VICTORY_JOIN_COPIES`;
//! the lock levers `VICTORY_BUFFERS_STRIPES` and `VICTORY_MADE_TURNS`;
//! `VICTORY_TEXTURE_BACK_ASYNC` uses texback's own, of the same form).
//!
//! Unset or `0`: off (the default, as before). `1`: on. `alt`: twenty
//! seconds off and twenty on, in turn from the first asking, each change
//! said in the log as `<what> from now (<NAME>=alt): ON` or `... : off, as
//! usual` (the form `xcl-notes/alt-fps.sh` and the other `alt` switches,
//! `VICTORY_LATE_WATER`, `VICTORY_TEXTURE_BACK_LEAN`, use). Also the PF5b
//! levers `VICTORY_HEAP_SYNC_NEED` and `VICTORY_MADE_PASS_FIRST`; the second
//! follows the first's phases when both are `alt` ([`Alt::following`]).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
use std::time::Instant;

/// An opt-in switch read once from the environment.
pub(crate) struct Alt {
    name: &'static str,
    what: &'static str,
    mode: OnceLock<u8>,
    epoch: OnceLock<Instant>,
    /// The phase last said: 0 off, 1 on, 2 none yet.
    last: AtomicU8,
    /// VIcky ([`Alt::following`]): with `alt` here and in this switch too,
    /// this one's phases are that one's (both change at once).
    follows: Option<&'static Alt>,
}

impl Alt {
    pub(crate) const fn new(name: &'static str, what: &'static str) -> Self {
        Self { name, what, mode: OnceLock::new(), epoch: OnceLock::new(), last: AtomicU8::new(2), follows: None }
    }

    /// VIcky: a switch whose `alt` phases are those of `leader` when that
    /// one is set to `alt` too (two levers judged together, e.g.
    /// `VICTORY_MADE_PASS_FIRST` with `VICTORY_HEAP_SYNC_NEED`); with the
    /// leader at `1` or unset, phases of its own, as [`Alt::new`]'s.
    pub(crate) const fn following(name: &'static str, what: &'static str, leader: &'static Alt) -> Self {
        Self {
            name,
            what,
            mode: OnceLock::new(),
            epoch: OnceLock::new(),
            last: AtomicU8::new(2),
            follows: Some(leader),
        }
    }

    /// Where the phases count from: the first asking, or the leader's.
    fn epoch(&self) -> Instant {
        match self.follows {
            Some(leader) if leader.mode() == 2 => leader.epoch(),
            _ => *self.epoch.get_or_init(Instant::now),
        }
    }

    /// 0 off, 1 on, 2 `alt`.
    pub(crate) fn mode(&self) -> u8 {
        *self.mode.get_or_init(|| {
            let mode = match std::env::var(self.name).as_deref().map(str::trim) {
                Ok("1") => 1,
                Ok("alt") => 2,
                _ => 0,
            };
            if mode == 1 {
                tracing::info!("{} ({}=1)", self.what, self.name);
            }
            mode
        })
    }

    /// Whether the switch is set at all (`1` or `alt`).
    pub(crate) fn set(&self) -> bool {
        self.mode() != 0
    }

    /// Whether it is on now; with `alt`, the phase of now, each change said.
    pub(crate) fn on(&self) -> bool {
        match self.mode() {
            0 => false,
            1 => true,
            _ => {
                let on = (self.epoch().elapsed().as_secs() / 20) % 2 == 1;
                // Looked at before it is swapped: switches asked on every
                // lookup (`VICTORY_BUFFERS_STRIPES`, `VICTORY_MADE_TURNS`)
                // would otherwise write this line from every thread, every
                // time, which is the very sharing they take away.
                if self.last.load(Relaxed) != u8::from(on) && self.last.swap(u8::from(on), Relaxed) != u8::from(on) {
                    tracing::info!("{} from now ({}=alt): {}", self.what, self.name, if on { "ON" } else { "off, as usual" });
                }
                on
            }
        }
    }

    /// Its state, for a frame line: `off`, `on`, `alt, ON`, `alt, off`.
    pub(crate) fn said(&self) -> &'static str {
        match (self.mode(), self.last.load(Relaxed)) {
            (0, _) => "off",
            (1, _) => "on",
            (_, 1) => "alt, ON",
            _ => "alt, off",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A switch set as `mode`, its phases counted from `ago` before now.
    fn set(alt: &Alt, mode: u8, ago: std::time::Duration) {
        alt.mode.set(mode).expect("a fresh switch");
        alt.epoch.set(Instant::now() - ago).expect("a fresh switch");
    }

    /// With both at `alt`, the follower's phases are the leader's (both
    /// change at once); with the leader at `1`, the follower keeps its own.
    #[test]
    fn a_follower_takes_its_leader_s_phases_when_both_alternate() {
        static LEADER: Alt = Alt::new("VICTORY_TEST_ALT_LEADER", "test leader");
        static FOLLOWER: Alt = Alt::following("VICTORY_TEST_ALT_FOLLOWER", "test follower", &LEADER);
        static FIXED: Alt = Alt::new("VICTORY_TEST_ALT_FIXED", "test leader at 1");
        static OWN: Alt = Alt::following("VICTORY_TEST_ALT_OWN", "test follower of a fixed leader", &FIXED);
        // The leader 25 s in (its ON phase), the follower 5 s in (its own
        // would be off): the follower is ON with it.
        set(&LEADER, 2, std::time::Duration::from_secs(25));
        set(&FOLLOWER, 2, std::time::Duration::from_secs(5));
        assert!(LEADER.on());
        assert!(FOLLOWER.on(), "the follower kept its own phases");
        assert_eq!(FOLLOWER.epoch(), LEADER.epoch());
        // A leader at `1`: the follower's own phases (5 s in: off).
        set(&FIXED, 1, std::time::Duration::from_secs(25));
        set(&OWN, 2, std::time::Duration::from_secs(5));
        assert!(FIXED.on());
        assert!(!OWN.on(), "a follower of a leader at 1 took its phases");
    }
}
