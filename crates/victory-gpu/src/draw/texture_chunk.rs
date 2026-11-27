//! VIcky: the pass over textures picked by index (`State::heap_textures`)
//! makes its due slots a chunk at a time under one lock of the heap's slot
//! state (`VICTORY_TEXTURE_PASS_CHUNK=1|alt`; off by default).
//!
//! Profile PF3 (fable-2026-11-27-pf3.md): the pass waited some 77 ms a
//! frame for `HostHeap::made`, one write lock for each due slot it made
//! (`HostHeap::mirror`), for about 1.5 ms of work under it, while the heap
//! sync and the pass over buffer views held the lock in chunks of 64.
//!
//! With the switch the pass takes its due slots [`CHUNK`] at a time, in the
//! order the queue gives them, as before. For each chunk:
//! 1. the looks that come before a slot is made, slot by slot as before
//!    (a texture uploaded whole and not changed since, memory not mapped or
//!    no more of it than before, a texture of the host's own, no room on
//!    the card): a slot they settle is not made;
//! 2. the others are made together, `made` locked once for all of them
//!    (`HostHeap::mirror_chunk`, as the pass over buffer views makes its
//!    slots): each slot's making is the one `HostHeap::mirror` does for a
//!    draw;
//! 3. their uploads, in order, as before.
//!
//! The bytes and the time of a pass are tested before each chunk, not
//! before each slot: a pass may go up to 63 slots past its budget. The card
//! asked again for room after 32 MiB of uploads tells the next chunk's
//! looks, not the rest of this chunk's. The same slots are made, from the
//! same records, in the same order; a slot's look and its making no longer
//! follow each other at once (the looks read the title's record and memory,
//! the making reads the record again under the lock, as it did).
//!
//! `VICTORY_TEXTURE_PASS_CHUNK=alt`: twenty seconds without, twenty with,
//! in turn from the first pass, each change said in the log with what the
//! passes took in the phase before.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

/// Due slots looked at, and those of them made under one lock.
pub(super) const CHUNK: usize = 64;

/// What the looks at a due slot before its making say
/// (`State::texture_look`).
pub(super) enum Look {
    /// Settled without a making (nothing to upload now, or held back).
    Skip,
    /// Made and uploaded, with the bytes of its placed texture mapped if
    /// not all of it is.
    Go(Option<u64>),
    /// The card has no room and the pass has refused enough: it ends.
    Stop,
}

/// What the switch asks for: 0 off, 1 on, 2 in turn.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_TEXTURE_PASS_CHUNK").as_deref() {
        Ok("1") => {
            tracing::info!(
                "the pass over textures picked by index makes its due slots {CHUNK} at a time, under one lock of the \
                 heap's slot state (VICTORY_TEXTURE_PASS_CHUNK=1)"
            );
            1
        }
        Ok("alt") => {
            tracing::info!(
                "the pass over textures picked by index makes its due slots {CHUNK} at a time in turn, twenty seconds \
                 without and twenty with (VICTORY_TEXTURE_PASS_CHUNK=alt)"
            );
            2
        }
        _ => 0,
    })
}

/// Whether this pass makes its slots a chunk at a time ([`mode`]; with
/// `alt`, as the phase says now).
pub(super) fn on() -> bool {
    match mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: AtomicU8 = AtomicU8::new(2);
            if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
                let [passes, us, chunks, made] = PHASE.each_ref().map(|c| c.swap(0, Relaxed));
                tracing::info!(
                    "the pass over textures picked by index makes its due slots a chunk at a time from now \
                     (VICTORY_TEXTURE_PASS_CHUNK=alt): {}; since the change before: {passes} passes, {:.2} ms a pass, \
                     {made} slots made, {chunks} chunks",
                    if on { "ON" } else { "off, one lock a slot as usual" },
                    us as f64 / 1000.0 / passes.max(1) as f64,
                );
            }
            on
        }
    }
}

/// With `alt`, since the last change: passes, their microseconds, chunks
/// made under one lock, and slots made.
static PHASE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A pass took `us` microseconds (with `alt`).
pub(super) fn pass(us: u64) {
    if mode() == 2 {
        PHASE[0].fetch_add(1, Relaxed);
        PHASE[1].fetch_add(us, Relaxed);
    }
}

/// `made` slots were made (one chunk under one lock, or one at a time).
pub(super) fn made(made: usize, chunk: bool) {
    if mode() == 2 && made > 0 {
        if chunk {
            PHASE[2].fetch_add(1, Relaxed);
        }
        PHASE[3].fetch_add(made as u64, Relaxed);
    }
}
