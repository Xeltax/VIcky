//! VIcky (`VICTORY_BUFFERS_STRIPES=1|alt`, roadmap A1): a table that is read
//! far more often than it is written, behind striped locks.
//!
//! The tables of host copies (`crate::shadow`'s `BUFFERS`, the title
//! buffers, and `LOOSE`, the windows of memory outside them) are looked up
//! for every buffer view a slot is made of, every root view, vertex, index
//! and argument buffer of a draw, every buffer slot the pass over buffer
//! views brings up to date: P7 counted 289,000 takes of `BUFFERS` a frame in
//! the prologue, a quarter of them after a wait, 72 ms waited, and every
//! lookup that `BUFFERS` does not answer takes `LOOSE` next. A lookup only
//! reads the table (a range walk and a reference count); the table changes
//! when the title makes or releases a buffer, and when a window is made or
//! let go. One mutex made all of them take turns.
//!
//! Here a reader with the switch on takes the read lock of one stripe, its
//! thread's (`crate::perthread::slot`, the block the per-thread counters
//! give it), each stripe on lines of its own: readers on different stripes
//! touch no line in common. A reader with the switch off takes the whole
//! lock alone, as the plain mutex was taken. A writer takes the whole lock
//! and then, once the switch is set at all, every stripe's write lock in
//! order: it waits for the readers of each stripe to be done, and none
//! comes in until it is. With the switch unset no reader takes a stripe,
//! so writers leave them alone and the whole lock is the old mutex.
//!
//! Lock order: the whole lock, then the stripes in rising order. A reader
//! holds one lock (the whole lock or one stripe) and takes neither again
//! while it holds it; it may take its own locks inside (the copies made on
//! first use are made under `crate::shadow`'s `MAKING`).

use std::cell::UnsafeCell;
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use std::time::Instant;

use crate::lot4::locks::Lock;

/// Stripes per table. A thread's stripe is its per-thread block modulo
/// this: the threads that look copies up (the title's recording threads,
/// the heap sync, the pass over buffer views and its threads, the
/// submitting thread) are some 10 to 20, so most have a stripe to
/// themselves. A writer takes all of them.
pub(crate) const STRIPES: usize = 32;

/// One stripe's lock, on lines of its own (two, for the adjacent-line
/// prefetcher).
#[repr(align(128))]
struct Stripe(RwLock<()>);

/// `T` behind a whole lock and [`STRIPES`] read locks.
pub(crate) struct Striped<T> {
    whole: Mutex<()>,
    stripes: [Stripe; STRIPES],
    data: UnsafeCell<T>,
    /// P7's kinds (`VICTORY_LOT4_PROBE`): held alone (a reader with the
    /// switch off), read on a stripe, written.
    kinds: [Lock; 3],
}

// SAFETY: `data` is read (`&T`) only under the whole lock or under one
// stripe's read lock, and written (`&mut T`) only under the whole lock and
// every stripe's write lock, or under the whole lock alone while no reader
// takes a stripe (`Striped::write`'s `stripes`). Readers on several threads
// share `&T` (so `T: Sync`); a writer has it alone (so `T: Send`).
unsafe impl<T: Send + Sync> Sync for Striped<T> {}
// SAFETY: as above; the locks are `Send`.
unsafe impl<T: Send> Send for Striped<T> {}

impl<T> Striped<T> {
    pub(crate) const fn new(data: T, kinds: [Lock; 3]) -> Self {
        Self {
            whole: Mutex::new(()),
            stripes: [const { Stripe(RwLock::new(())) }; STRIPES],
            data: UnsafeCell::new(data),
            kinds,
        }
    }

    /// The table read: on this thread's stripe (`striped`), or with the
    /// whole lock held alone, as the plain mutex was.
    pub(crate) fn read(&self, striped: bool) -> Reading<'_, T> {
        crate::lot4::locks::read_by(self.kinds[0]);
        let held = if striped {
            let stripe = &self.stripes[crate::perthread::slot() % STRIPES].0;
            Held::Stripe(crate::lot4::locks::read(self.kinds[1], stripe))
        } else {
            Held::Whole(crate::lot4::locks::mutex(self.kinds[0], &self.whole))
        };
        Reading { table: self, held }
    }

    /// The table written: the whole lock, then (with `stripes`: whenever a
    /// reader may hold one, i.e. whenever the switch is set at all) every
    /// stripe's write lock.
    pub(crate) fn write(&self, stripes: bool) -> Writing<'_, T> {
        let probe = crate::lot4::on();
        let mut began = None;
        let whole = if probe {
            match self.whole.try_lock() {
                Ok(held) => held,
                Err(TryLockError::Poisoned(e)) => e.into_inner(),
                Err(TryLockError::WouldBlock) => {
                    began = Some(Instant::now());
                    self.whole.lock().unwrap_or_else(|e| e.into_inner())
                }
            }
        } else {
            self.whole.lock().unwrap_or_else(|e| e.into_inner())
        };
        let held = self.stripes_written(stripes, probe, &mut began);
        self.written(probe, began);
        Writing { table: self, _stripes: held, _whole: whole }
    }

    /// Every stripe's write lock, in order, if `stripes`.
    fn stripes_written(&self, stripes: bool, probe: bool, began: &mut Option<Instant>) -> Vec<RwLockWriteGuard<'_, ()>> {
        if !stripes {
            return Vec::new();
        }
        self.stripes
            .iter()
            .map(|stripe| {
                if probe {
                    match stripe.0.try_write() {
                        Ok(held) => return held,
                        Err(TryLockError::Poisoned(e)) => return e.into_inner(),
                        Err(TryLockError::WouldBlock) => {
                            began.get_or_insert_with(Instant::now);
                        }
                    }
                }
                stripe.0.write().unwrap_or_else(|e| e.into_inner())
            })
            .collect()
    }

    /// P7: a write taken, after a wait since `began` if any lock of it was
    /// held elsewhere.
    fn written(&self, probe: bool, began: Option<Instant>) {
        if !probe {
            return;
        }
        match began {
            Some(began) => crate::lot4::locks::waited(self.kinds[2], began.elapsed().as_nanos() as u64),
            None => crate::lot4::locks::tried(self.kinds[2], true),
        }
    }

    /// Whether this thread could take its stripe's read lock now, without
    /// waiting (tests).
    #[cfg(test)]
    fn stripe_free(&self) -> bool {
        self.stripes[crate::perthread::slot() % STRIPES].0.try_read().is_ok()
    }
}

/// What a [`Reading`] holds.
enum Held<'a> {
    Whole(MutexGuard<'a, ()>),
    Stripe(RwLockReadGuard<'a, ()>),
}

/// The table read ([`Striped::read`]).
pub(crate) struct Reading<'a, T> {
    table: &'a Striped<T>,
    held: Held<'a>,
}

impl<'a, T> Reading<'a, T> {
    /// The table written from this reading. Held alone, the whole lock is
    /// kept and the stripes taken after it (with `stripes`): no writer came
    /// in between. On a stripe, the stripe is let go first and the table
    /// written as [`Striped::write`] writes it: another writer may come in
    /// between, so what the reader found is to be looked for again.
    pub(crate) fn upgrade(self, stripes: bool) -> Writing<'a, T> {
        let table = self.table;
        match self.held {
            Held::Whole(whole) => {
                let probe = crate::lot4::on();
                let mut began = None;
                let held = table.stripes_written(stripes, probe, &mut began);
                table.written(probe, began);
                Writing { table, _stripes: held, _whole: whole }
            }
            Held::Stripe(stripe) => {
                drop(stripe);
                table.write(stripes)
            }
        }
    }
}

impl<T> std::ops::Deref for Reading<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: the whole lock or a stripe's read lock is held: no writer
        // holds the table (`Striped`'s Sync).
        unsafe { &*self.table.data.get() }
    }
}

/// The table written ([`Striped::write`]).
pub(crate) struct Writing<'a, T> {
    table: &'a Striped<T>,
    // Let go in this order: the stripes, then the whole lock.
    _stripes: Vec<RwLockWriteGuard<'a, ()>>,
    _whole: MutexGuard<'a, ()>,
}

impl<T> std::ops::Deref for Writing<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: held alone (`Striped`'s Sync).
        unsafe { &*self.table.data.get() }
    }
}

impl<T> std::ops::DerefMut for Writing<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: held alone (`Striped`'s Sync).
        unsafe { &mut *self.table.data.get() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

    /// A writer waits for a reader on a stripe; readers on stripes and
    /// alone read together; an upgrade from alone keeps the whole lock.
    #[test]
    fn writers_exclude_striped_readers() {
        let table: &'static Striped<Vec<u64>> = Box::leak(Box::new(Striped::new(vec![1], [Lock::Buffers, Lock::BuffersRead, Lock::BuffersWrite])));
        let reading = table.read(true);
        // Another reader alone reads beside the striped one.
        let alone = std::thread::spawn(move || table.read(false).len()).join().unwrap();
        assert_eq!(alone, 1);
        let written = Arc::new(AtomicBool::new(false));
        let writer = {
            let written = written.clone();
            std::thread::spawn(move || {
                table.write(true).push(2);
                written.store(true, Relaxed);
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!written.load(Relaxed), "the writer waited for the reader on its stripe");
        assert_eq!(*reading, vec![1]);
        drop(reading);
        writer.join().unwrap();
        assert!(written.load(Relaxed));
        assert_eq!(*table.read(true), vec![1, 2]);
        let mut up = table.read(false).upgrade(true);
        up.push(3);
        drop(up);
        assert_eq!(*table.read(false), vec![1, 2, 3]);
        assert!(table.stripe_free(), "every stripe let go");
        // Without stripes (the switch unset): the whole lock alone.
        table.write(false).push(4);
        assert_eq!(table.read(false).len(), 4);
    }

    /// The same walk as `crate::shadow`'s lookup, over a table of the title
    /// buffers' shape: under the plain mutex and on stripes, the same key.
    fn innermost(map: &BTreeMap<(u64, usize), u64>, address: u64, end: u64) -> Option<(u64, usize)> {
        map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| **e >= end).map(|(k, _)| *k)
    }

    /// Readers on stripes and alone, and writers, on 12 threads: every
    /// read sees a table one writer left whole (a writer changes two
    /// entries that must stay equal), and the final table is the writes'.
    #[test]
    fn twelve_threads_mixed_modes() {
        static PAIRS: Striped<BTreeMap<u64, u64>> = Striped::new(BTreeMap::new(), [Lock::Loose, Lock::LooseRead, Lock::LooseWrite]);
        const WRITES: u64 = 2_000;
        let stop = Arc::new(AtomicBool::new(false));
        let reads = Arc::new(AtomicU64::new(0));
        let mut threads = Vec::new();
        for t in 0..10 {
            let (stop, reads) = (stop.clone(), reads.clone());
            threads.push(std::thread::spawn(move || {
                let mut n = 0u64;
                while !stop.load(Relaxed) {
                    let table = PAIRS.read(t % 3 != 0);
                    let (a, b) = (table.get(&1).copied(), table.get(&2).copied());
                    assert_eq!(a, b, "a write seen half done");
                    n += 1;
                }
                reads.fetch_add(n, Relaxed);
            }));
        }
        let writers: Vec<_> = (0..2)
            .map(|w| {
                std::thread::spawn(move || {
                    for i in 0..WRITES {
                        let mut table = if i % 2 == 0 { PAIRS.write(true) } else { PAIRS.read(w == 0).upgrade(true) };
                        let v = table.get(&1).copied().unwrap_or(0) + 1;
                        table.insert(1, v);
                        std::hint::spin_loop();
                        table.insert(2, v);
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        stop.store(true, Relaxed);
        for t in threads {
            t.join().unwrap();
        }
        let table = PAIRS.read(false);
        assert_eq!((table.get(&1), table.get(&2)), (Some(&(2 * WRITES)), Some(&(2 * WRITES))));
        assert!(reads.load(Relaxed) > 0);
    }

    /// Same answers alone and on stripes, over random tables and asks.
    #[test]
    fn same_lookups_alone_and_striped() {
        let table: &'static Striped<BTreeMap<(u64, usize), u64>> =
            Box::leak(Box::new(Striped::new(BTreeMap::new(), [Lock::Buffers, Lock::BuffersRead, Lock::BuffersWrite])));
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..20 {
            {
                let mut map = table.write(true);
                for _ in 0..200 {
                    let start = 0x1000_0000 + (next() % 0x400) * 0x1_0000;
                    let len = (1 + next() % 64) * 0x1000;
                    map.insert((start, next() as usize % 8), start + len);
                }
                if round % 3 == 2 {
                    let keys: Vec<_> = map.keys().copied().filter(|k| k.1 % 3 == 0).collect();
                    for k in keys {
                        map.remove(&k);
                    }
                }
            }
            for _ in 0..2_000 {
                let address = 0x1000_0000 + next() % 0x400_0000;
                let end = address + 1 + next() % 0x2000;
                let alone = innermost(&table.read(false), address, end);
                let striped = innermost(&table.read(true), address, end);
                assert_eq!(alone, striped);
            }
        }
    }

    /// Micro-benchmark: 1, 4, 8 and 12 threads looking up a table of the
    /// title buffers' shape (6,000 buffers, the innermost one holding an
    /// address found as `crate::shadow`'s lookup finds it, then a reference
    /// count taken), under one mutex as before and on stripes, with one
    /// write in 1,000 lookups:
    /// `cargo test -p victory-gpu --profile diag --lib stripes::tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "a benchmark: run alone, with --nocapture"]
    fn bench_lookups() {
        const BUFFERS: u64 = 6_000;
        const LOOKS: u64 = 400_000;
        let fill = |map: &mut BTreeMap<(u64, usize), (u64, Arc<u64>)>| {
            for i in 0..BUFFERS {
                let start = 0x1_0000_0000 + i * 0x2_0000;
                map.insert((start, i as usize), (start + 0x1_8000, Arc::new(i)));
            }
        };
        let walk = |map: &BTreeMap<(u64, usize), (u64, Arc<u64>)>, address: u64| {
            map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.0 >= address + 256).map(|(_, e)| e.1.clone())
        };
        let plain: &'static Mutex<BTreeMap<(u64, usize), (u64, Arc<u64>)>> = Box::leak(Box::new(Mutex::new(BTreeMap::new())));
        fill(&mut plain.lock().unwrap());
        let striped: &'static Striped<BTreeMap<(u64, usize), (u64, Arc<u64>)>> =
            Box::leak(Box::new(Striped::new(BTreeMap::new(), [Lock::Buffers, Lock::BuffersRead, Lock::BuffersWrite])));
        fill(&mut striped.write(true));
        let run = |threads: u64, striped_on: bool| {
            let began = Instant::now();
            let waits = Arc::new(AtomicU64::new(0));
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let waits = waits.clone();
                    std::thread::spawn(move || {
                        let mut seed = 0x2545_F491_4F6C_DD1Du64 ^ (t << 32);
                        let (mut found, mut waited) = (0u64, 0u64);
                        for n in 0..LOOKS {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            let address = 0x1_0000_0000 + seed % (BUFFERS * 0x2_0000);
                            let write = n % 1_000 == 999;
                            let at = Instant::now();
                            if striped_on {
                                if write {
                                    let mut map = striped.write(true);
                                    let key = (0x1_0000_0000 + (seed % BUFFERS) * 0x2_0000, (seed % BUFFERS) as usize);
                                    if let Some(e) = map.get_mut(&key) {
                                        e.0 = key.0 + 0x1_8000;
                                    }
                                    waited += at.elapsed().as_nanos() as u64;
                                } else {
                                    let map = striped.read(true);
                                    waited += at.elapsed().as_nanos() as u64;
                                    found += u64::from(walk(&map, address).is_some());
                                }
                            } else {
                                let mut map = plain.lock().unwrap();
                                waited += at.elapsed().as_nanos() as u64;
                                if write {
                                    let key = (0x1_0000_0000 + (seed % BUFFERS) * 0x2_0000, (seed % BUFFERS) as usize);
                                    if let Some(e) = map.get_mut(&key) {
                                        e.0 = key.0 + 0x1_8000;
                                    }
                                } else {
                                    found += u64::from(walk(&map, address).is_some());
                                }
                            }
                        }
                        waits.fetch_add(waited, Relaxed);
                        found
                    })
                })
                .collect();
            let found: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
            let wall = began.elapsed().as_nanos() as f64;
            (wall / LOOKS as f64, waits.load(Relaxed) as f64 / (threads * LOOKS) as f64, found)
        };
        for threads in [1u64, 4, 8, 12] {
            for _ in 0..2 {
                let (wall_plain, lock_plain, _) = run(threads, false);
                let (wall_striped, lock_striped, _) = run(threads, true);
                println!(
                    "stripes bench, {threads:>2} threads: one mutex {wall_plain:>7.1} ns a lookup per thread (wall), \
                     {lock_plain:>6.1} ns taking the lock; on stripes {wall_striped:>7.1} ns, {lock_striped:>6.1} ns; {:.1}x",
                    wall_plain / wall_striped
                );
            }
        }
    }
}
