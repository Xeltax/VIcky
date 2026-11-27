//! VIcky: counters that every thread adds to, kept per thread (lot 4, idea
//! C5 of `fable-2026-11-27-sources.md`).
//!
//! The diagnostics counters were arrays of adjacent shared atomics: eight to
//! a cache line, each add a locked add, from the title's recording threads,
//! the heap sync, the pass over buffer views and the submitting thread at
//! once. Every add took the line from the core that added last, across the
//! two CCDs of the test machine as often as not, and the fps runs use the
//! diagnostics build. The allocator's counters (two locked adds per
//! allocation and per release, on one line) were the busiest of all.
//!
//! Here each thread adds to a block of its own, aligned to 128 bytes so that
//! no two threads' blocks share a line (or a pair of lines, for the
//! adjacent-line prefetcher). Only the owner writes a block, with a plain
//! load and store: no locked instruction, and the line stays in the owner's
//! cache until a reader sums it, once per log line. Totals are sums over
//! the blocks (wrapping: a thread may take away what another added), so they
//! are the same as the shared counters gave, add for add.
//!
//! A thread gets its block at its first add, for every table at once
//! ([`slot`]), and keeps it until the process ends (a block outlives its
//! thread; its counts stay in the sums). Past [`SLOTS`] threads the rest
//! share one more block, written with locked adds as before.

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

/// Threads with a block of their own.
pub(crate) const SLOTS: usize = 512;

/// Threads that took a block so far (the ones past [`SLOTS`] included).
static CLAIMED: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// This thread's block in every table ([`SLOTS`]: the shared one);
    /// `usize::MAX` until its first add. A const thread local without a
    /// destructor: no registration, no allocation (the allocator's
    /// counters use it).
    static SLOT: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// This thread's block, taken at its first call.
#[inline]
pub(crate) fn slot() -> usize {
    // (`try_with`: a thread being torn down after its thread locals adds to
    // the shared block.)
    SLOT.try_with(|s| {
        let at = s.get();
        if at != usize::MAX { at } else { claim(s) }
    })
    .unwrap_or(SLOTS)
}

#[cold]
fn claim(s: &Cell<usize>) -> usize {
    let at = CLAIMED.fetch_add(1, Relaxed).min(SLOTS);
    s.set(at);
    at
}

/// Threads that took a block so far: (with a block of their own, sharing
/// the last).
pub(crate) fn threads() -> (usize, usize) {
    let n = CLAIMED.load(Relaxed);
    (n.min(SLOTS), n.saturating_sub(SLOTS))
}

/// One thread's `N` counters, on lines of their own.
#[repr(align(128))]
struct Block<const N: usize>([AtomicU64; N]);

impl<const N: usize> Block<N> {
    const ZERO: Self = Block([const { AtomicU64::new(0) }; N]);
}

/// `N` counters kept per thread ([`slot`]).
pub(crate) struct Counters<const N: usize> {
    blocks: [Block<N>; SLOTS + 1],
    /// The totals [`Counters::take`] returned up to.
    taken: Mutex<[u64; N]>,
}

impl<const N: usize> Counters<N> {
    pub(crate) const fn new() -> Self {
        Counters { blocks: [Block::<N>::ZERO; SLOTS + 1], taken: Mutex::new([0; N]) }
    }

    /// This thread's counter `i`, and whether only this thread writes it.
    #[inline]
    fn mine(&self, i: usize) -> (&AtomicU64, bool) {
        let at = slot();
        (&self.blocks[at].0[i], at < SLOTS)
    }

    /// Adds `n` to counter `i`.
    #[inline]
    pub(crate) fn add(&self, i: usize, n: u64) {
        let (c, own) = self.mine(i);
        if own {
            // Only this thread writes it: no locked add.
            c.store(c.load(Relaxed).wrapping_add(n), Relaxed);
        } else {
            c.fetch_add(n, Relaxed);
        }
    }

    /// Takes `n` from counter `i` (the totals wrap: what one thread takes,
    /// another may have added).
    #[inline]
    pub(crate) fn sub(&self, i: usize, n: u64) {
        let (c, own) = self.mine(i);
        if own {
            c.store(c.load(Relaxed).wrapping_sub(n), Relaxed);
        } else {
            c.fetch_sub(n, Relaxed);
        }
    }

    /// The blocks written so far.
    fn used(&self) -> &[Block<N>] {
        let (own, past) = threads();
        // The shared block is the last; it is summed only once it is used.
        let end = if past > 0 { SLOTS + 1 } else { own };
        &self.blocks[..end]
    }

    /// Counter `i` summed over every thread, since the start.
    pub(crate) fn total(&self, i: usize) -> u64 {
        self.used().iter().fold(0u64, |sum, b| sum.wrapping_add(b.0[i].load(Relaxed)))
    }

    /// Every counter summed over every thread, since the start.
    pub(crate) fn totals(&self) -> [u64; N] {
        let mut sums = [0u64; N];
        for b in self.used() {
            for (sum, c) in sums.iter_mut().zip(&b.0) {
                *sum = sum.wrapping_add(c.load(Relaxed));
            }
        }
        sums
    }

    /// Every counter's total since the last call (what a swap to zero of
    /// each shared counter gave).
    pub(crate) fn take(&self) -> [u64; N] {
        let mut taken = self.taken.lock().unwrap_or_else(|e| e.into_inner());
        let now = self.totals();
        let out = std::array::from_fn(|i| now[i].wrapping_sub(taken[i]));
        *taken = now;
        out
    }

    /// Counter `i` of every thread's block in use, one value each (for a
    /// maximum over threads, not a sum).
    pub(crate) fn each(&self, i: usize) -> impl Iterator<Item = u64> + '_ {
        self.used().iter().map(move |b| b.0[i].load(Relaxed))
    }

    /// This thread's counter `i` set to `v`; on the shared block, `v` is
    /// kept only where `keep(old, v)` says so (a compare and swap).
    #[inline]
    pub(crate) fn set_if(&self, i: usize, v: u64, keep: impl Fn(u64, u64) -> bool) {
        let (c, own) = self.mine(i);
        if own {
            c.store(v, Relaxed);
        } else {
            let _ = c.fetch_update(Relaxed, Relaxed, |old| keep(old, v).then_some(v));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sums over threads match what was added, with adds and takes from
    /// several threads at once, and `take` gives each add once.
    #[test]
    fn sums_match_under_several_threads() {
        static C: Counters<4> = Counters::new();
        const THREADS: u64 = 12;
        const ADDS: u64 = 100_000;
        let before = C.totals();
        std::thread::scope(|s| {
            for t in 0..THREADS {
                s.spawn(move || {
                    for k in 0..ADDS {
                        C.add(0, 1);
                        C.add(1, t + 1);
                        C.add(2, k);
                        // One thread's add, another's take: the sum is 0.
                        if t % 2 == 0 { C.add(3, 5) } else { C.sub(3, 5) }
                    }
                });
            }
        });
        let after = C.totals();
        let got: [u64; 4] = std::array::from_fn(|i| after[i].wrapping_sub(before[i]));
        assert_eq!(got[0], THREADS * ADDS);
        assert_eq!(got[1], (1..=THREADS).sum::<u64>() * ADDS);
        assert_eq!(got[2], THREADS * (ADDS * (ADDS - 1) / 2));
        assert_eq!(got[3], 0);
        assert_eq!(C.total(0).wrapping_sub(before[0]), THREADS * ADDS);
    }

    /// `take` hands each add out once, while threads go on adding.
    #[test]
    fn takes_while_adding_lose_nothing() {
        static C: Counters<1> = Counters::new();
        let _ = C.take();
        let stop = std::sync::atomic::AtomicBool::new(false);
        let taken = std::thread::scope(|s| {
            let adders: Vec<_> = (0..6)
                .map(|_| {
                    s.spawn(|| {
                        let mut n = 0u64;
                        while !stop.load(Relaxed) {
                            C.add(0, 3);
                            n += 3;
                        }
                        n
                    })
                })
                .collect();
            let mut taken = 0u64;
            for _ in 0..200 {
                taken += C.take()[0];
                std::thread::yield_now();
            }
            stop.store(true, Relaxed);
            let added: u64 = adders.into_iter().map(|a| a.join().unwrap()).sum();
            (taken + C.take()[0], added)
        });
        assert_eq!(taken.0, taken.1);
    }

    /// The largest of every thread's own value.
    #[test]
    fn each_thread_sets_its_own() {
        static C: Counters<1> = Counters::new();
        std::thread::scope(|s| {
            for t in 1..=5u64 {
                s.spawn(move || C.set_if(0, t * 10, |old, v| v > old));
            }
        });
        assert_eq!(C.each(0).max(), Some(50));
    }

    /// Micro-benchmark (`cargo test -p victory-gpu --release perthread -- --ignored --nocapture`):
    /// the same adds from 1, 2, 4, 8 and 12 threads, to adjacent shared
    /// atomics as before and to per-thread blocks; nanoseconds an add.
    #[test]
    #[ignore = "a benchmark: run alone, with --nocapture"]
    fn bench_shared_against_per_thread() {
        const ADDS: u64 = 4_000_000;
        static SHARED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
        static OWN: Counters<8> = Counters::new();
        let run = |threads: u64, each: &(dyn Fn(u64) + Sync)| {
            let started = std::time::Instant::now();
            std::thread::scope(|s| {
                for t in 0..threads {
                    s.spawn(move || {
                        for k in 0..ADDS {
                            each((t + k) % 4);
                        }
                    });
                }
            });
            started.elapsed().as_nanos() as f64 / ADDS as f64
        };
        for threads in [1, 2, 4, 8, 12] {
            let shared = run(threads, &|i| {
                SHARED[i as usize].fetch_add(1, Relaxed);
            });
            let own = run(threads, &|i| OWN.add(i as usize, 1));
            println!(
                "perthread bench: {threads:>2} threads, {ADDS} adds each: shared atomics {shared:6.2} ns an add (wall, per \
                 thread's add), per-thread blocks {own:6.2} ns ({:.1}x)",
                shared / own
            );
        }
        let shared: u64 = SHARED.iter().map(|c| c.load(Relaxed)).sum();
        assert_eq!(shared, OWN.totals().iter().sum::<u64>());
    }
}
