//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed. A view heap the
//! host caps below the title's slots keeps that order only for an identity
//! window of them; past it each title slot that holds a view is made in a
//! pool slot, which translated shaders look up in a heap map ([`MapLayout`]).

use std::sync::Mutex;
use std::sync::atomic::AtomicU64;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::left;
use super::slot_map::SlotMap;
use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) mod made_first;
pub(crate) mod made_read;
pub(crate) mod note_kind;
pub(crate) mod par_mirror;
pub(crate) mod sync_lean;
pub(crate) mod tables_left;

/// The next host heap's identity ([`HostHeap::release`]).
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub(crate) struct HostHeap {
    /// This host heap's identity, recorded with the dynamic slots a list
    /// takes from it and checked when they are given back
    /// ([`HostHeap::release`]): a title heap can be let go before the lists
    /// that used it are reset, and a new one made at the same address; the
    /// old lists' reset must not free the new heap's slots. Port of
    /// trustno1 47747d62 (fxceless #1324).
    id: u64,
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's at their own index (in a view heap with
    /// a heap map, the identity window); view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots from `dynamic_first`: [`DYNAMIC`] in view heaps, none
    /// in sampler heaps. (Upstream falls back to 2^16 on a capped host heap;
    /// ours keeps 2^18 there: 2^16 and 2^17 both ran out in this title.)
    dynamic: u32,
    /// The first dynamic slot: `count`, or in a view heap with a heap map
    /// the one after the map's slot ([`MapLayout`]).
    dynamic_first: u32,
    /// The title's own slot count (`count` may be fewer).
    title_count: u32,
    sampler: bool,
    /// The record (view index and generation) each host slot of the
    /// title's slots was last made from, and the heap map's bookkeeping.
    /// Held alone (its write lock) by all but the threads of the pass over
    /// buffer views with VIcky's `VICTORY_HEAP_PASS_THREADS`, which hold it
    /// together (its read lock; [`Made`]).
    made: std::sync::RwLock<Made>,
    /// The heap map of a view heap capped below the title's slots
    /// ([`MapLayout`]); `None` for the others.
    map: Option<HeapMap>,
    /// The title heap's first descriptor, as the first slot mirrored past
    /// the window gives it (0 until then): for [`HostHeap::reclaim`].
    title_start: AtomicU64,
    /// When [`HostHeap::reclaim`] last ran (`crate::host::now_ms`;
    /// `u64::MAX`: never).
    reclaimed: AtomicU64,
    /// A fingerprint of the record each title slot [`HostHeap::mirror`]
    /// makes ([`HostHeap::mirrored`]) held when the pass over records
    /// copied in as bytes last looked at it ([`HostHeap::mirror_copied`];
    /// 0: never looked at). Made on first use.
    copied: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// VIcky (`VICTORY_HEAP_KEPT_SPLIT=1`, [`super::kept`]): the slots below
    /// `dynamic_first` and the dynamic ones each behind a lock of their own.
    kept: super::kept::Holds<ID3D12Resource>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// How many dynamic slots are free (changed under `next`'s lock): with
    /// fewer than a table needs, the search is not made. Once the world
    /// had taken them all, every later table of the frame searched every
    /// slot before it was refused.
    free: std::sync::atomic::AtomicU32,
    /// The shortest table that found no run of free slots since slots were
    /// last given back (changed under `next`'s lock; `u32::MAX`: none).
    /// Until something is given back the free runs only shrink, so a table
    /// of that length or longer is refused without the search. Enough
    /// slots were free, but in pieces, and every such table walked the
    /// whole ring before it was refused: 0.33 ms each with 131,072 slots,
    /// 1.0 s of the recording threads in a frame that refused 3,143 (run
    /// 45), under the lock all of them need. (A free run that lies across
    /// the place the search starts from is seen by neither.)
    too_long: std::sync::atomic::AtomicU32,
    /// View heaps: the bytes per element of the buffer view at each title
    /// slot, for shaders that pick a buffer by an index they compute
    /// (`victory_shader::emit::SLOT_STRIDES`): an upload buffer of
    /// `SLOT_STRIDES_COUNT` dwords, its GPU address and its mapping.
    strides: Option<(ID3D12Resource, u64, usize)>,
    /// Title slots holding buffer views. Such a view gets its host slot
    /// (and its memory a host copy) only when something reads it: a draw
    /// through a table, or [`HostHeap::buffer_slots`] for shaders that pick
    /// buffers from the whole heap.
    buffers: Mutex<std::collections::BTreeSet<u32>>,
    /// Of those and of the slots that left the set: the ones written since
    /// [`HostHeap::buffer_changes`] last took them.
    changed: Mutex<std::collections::HashSet<u32>>,
    /// Title slots holding a view of a texture the title may fill itself (a
    /// placed texture, a texture object), with the frame from which they
    /// are due: such a texture is uploaded when a draw reads it through a
    /// table; one a shader picks by a computed index is brought up to date
    /// from here ([`HostHeap::texture_slots_due`]).
    textures: Mutex<TextureQueue>,
    /// Per title slot, the frame (plus one; 0: never) in which a draw last
    /// read it through a table, or made a table it lies in (with
    /// [`TABLED_READ`] set if the draw read it). The draw that
    /// reads such a slot brings its texture up to date itself and the slot
    /// then leaves `textures`; a write of a slot that was a table's within
    /// the last [`TABLED_FRAMES`] frames is due only [`TABLED_GRACE`]
    /// frames later, by when its draw has read it. A title that builds a
    /// table per draw writes thousands of texture views a frame into slots
    /// only tables read (the title: some 2,500 a frame in its
    /// opening cutscene, a few hundred textures over and over), and every
    /// one of them stood in the queue of textures picked by index, ahead
    /// of or behind the ones shaders do pick by index.
    /// VIcky (`VICTORY_TABLE_BOUND`, [`left`]): with the switch, the tables
    /// made afresh past the identity window mark here too; with `alt`,
    /// nothing acts on the marks in its off phase ([`left::live`]).
    tabled: Vec<std::sync::atomic::AtomicU32>,
    /// VIcky (`VICTORY_TABLE_BOUND`, [`left`]; empty without the switch):
    /// per title slot, the frame (plus one; 0: never) in which a draw last
    /// bound a table it lies in, whatever it holds: buffer marks, kept
    /// apart from `tabled` (after myokex c3723f3, as 7b66261e had them).
    /// Only counted (the sync's records and the pass's slots they would
    /// leave to the draws, stage 2's question); nothing reads them else.
    buffer_marks: Vec<std::sync::atomic::AtomicU32>,
    /// VIcky (`VICTORY_TABLE_BOUND=alt`, [`HostHeap::table_bound_flip`]):
    /// whether the marks of `tabled` acted ([`left::live`]) at this heap's
    /// last heap sync.
    bound_live: std::sync::atomic::AtomicBool,
    /// VIcky (`VICTORY_TABLES_LEFT`, stage 2, [`tables_left`]; empty without
    /// the switch): the slots left to the draws and not noted since, the
    /// records waiting their grace, the holds of host copies, and the
    /// check's model.
    left: left::HeapLeft,
    /// The title memory each placed texture view of this heap describes,
    /// by slot and by address ([`Spans`]): which slots a newer view of
    /// other memory there makes stale, and which to look at again when the
    /// runtime maps, unmaps or writes memory ([`remapped`]).
    placed: std::sync::Arc<Mutex<Spans>>,
    /// VIcky (`VICTORY_SPANS_QUICK`, [`spans_quick`]; empty without it): a
    /// bit per title slot that may be in `placed` (put there under its lock
    /// when the slot is noted there, taken out when it is forgotten there;
    /// a slot another's view made stale keeps its bit, which says only
    /// "may"). A write of no placed texture view to a slot without its bit
    /// needs nothing of `placed`.
    spanned: Vec<std::sync::atomic::AtomicU64>,
    /// VIcky (`VICTORY_HEAP_BUFFERS_SKIP_SAME=1`; empty without it): per
    /// title slot, whether the last [`HostHeap::note_buffer`] of the slot
    /// was of a plain buffer view ([`plain_buffer`]). Read and written
    /// under `made`'s lock, as every call of `note_buffer` is.
    noted: Vec<std::sync::atomic::AtomicBool>,
    /// VIcky ([`note_kind`]; empty in sampler heaps): per title slot, what
    /// its last note left it as: a plain buffer view's, bare (in none of
    /// the sets: no view, or a view neither of a buffer nor of a texture),
    /// or another. Read and written under `made`'s lock, as `noted`.
    kinds: Vec<std::sync::atomic::AtomicU8>,
    /// VIcky (with `noted`): what `strides` holds for each slot, so that a
    /// stride that is already there is not written again. `strides` is
    /// write-combined memory: a write there, followed by the next lock
    /// taken, cost about 0.1 us a slot (the bench in `fast_tests`).
    strides_held: Vec<std::sync::atomic::AtomicU32>,
    /// VIcky (`VICTORY_MIRROR_TRY`, [`super::mirror_try`]): the copied
    /// slots [`HostHeap::mirror_copy`] left to the next drain.
    deferred: super::mirror_try::Deferred,
    /// VIcky (`VICTORY_PARALLEL_MIRROR`, [`par_mirror`]): `made` in pieces
    /// for the threads that make slots together, slot `i` made with piece
    /// `i % par_mirror::PIECES` locked.
    pieces: par_mirror::Pieces,
    /// VIcky (`VICTORY_MADE_TURNS`, [`MADE_TURNS`]): the threads waiting
    /// for `made` that the chunk makers let in between two slots.
    turns: Turns,
}

/// The spans of every view heap made ([`HostHeap::placed`]), for the
/// texture code, which has no heap at hand ([`next_start`],
/// [`first_viewed`]).
static PLACED_SPANS: Mutex<Vec<std::sync::Weak<Mutex<Spans>>>> = Mutex::new(Vec::new());

/// Where the next texture placed inside `[start, end)` starts, past the
/// first bytes of the one at `start` ([`HEAD`]): what is there of the
/// texture at `start` ends there, whatever its full size. `None`: no view
/// in a slot shaders pick by index describes memory in that range.
pub(crate) fn next_start(start: u64, end: u64) -> Option<u64> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*spans).next_start(start, end))
        .min()
}

/// The lowest level that a view of the placed texture at `start` shows,
/// over the slots shaders pick by index that hold one now. The title
/// views a texture from the first level it holds. `None`: no such slot.
pub(crate) fn first_viewed(start: u64) -> Option<u32> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*spans).first_viewed(start))
        .min()
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose first level with memory changed since a texture pass last took
/// them ([`HostHeap::texture_views_remade`]).
static RESHOWN: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// The first level with memory of the placed texture at `start` changed:
/// its slots are made again at the next texture pass.
pub(crate) fn levels_changed(start: u64) {
    RESHOWN.lock().unwrap_or_else(|e| e.into_inner()).push(start);
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose last upload left a held level without tiles (the card had no
/// room, or no heap could be had), and when (`crate::host::moment`). A
/// texture pass makes their slots due again once the card has room
/// ([`HostHeap::texture_slots_waiting`]).
static WAITING: Mutex<std::collections::BTreeMap<u64, u64>> = Mutex::new(std::collections::BTreeMap::new());

/// The last upload of the placed texture at `start` left a held level
/// without tiles. A texture picked by index is uploaded once and then
/// only when its memory is found changed, and its staging buffer goes
/// with the upload: without this the level stayed empty (the texture
/// blurred, or black if no level had tiles) for as long as the title left
/// its memory alone, where a texture refused for want of room is tried
/// again.
pub(crate) fn levels_waiting(start: u64) {
    WAITING.lock().unwrap_or_else(|e| e.into_inner()).insert(start, crate::host::moment());
}

/// Host-slot state behind one lock: every write of a host slot for a title
/// slot takes it, so a pool slot is taken, made and published (or given
/// back) at once.
///
/// VIcky (`VICTORY_HEAP_PASS_THREADS`): everyone holds it alone (the write
/// lock of [`HostHeap::made`]) as before, but the threads of the pass over
/// buffer views, which hold it together (its read lock), each for title
/// slots none of the others makes ([`HostHeap::mirror_shared`]). What the
/// making of one title slot changes here is its host slot's key and its
/// heap map entry, which only the thread making that slot touches (atomics,
/// for the threads' sake: under the write lock they are plain loads and
/// stores), and the heap map's pool and the cache, each behind a lock of
/// its own taken inside this one. A pool slot given back goes through the
/// pool's lock before another thread can take it, its key zeroed first.
struct Made {
    /// The record (view index and generation) each host slot of the title's
    /// (`count`, and in a heap with a heap map its pool's) was last made
    /// from; 0 for none.
    keys: Vec<std::sync::atomic::AtomicU64>,
    /// Which pool slot each title slot past the identity window holds (a
    /// view heap with a heap map).
    map: Option<SlotMap>,
    /// VIcky (`VICTORY_HEAP_SYNC_CACHE`): buffer views made before, by
    /// record ([`RecordCache`]).
    cache: Mutex<CacheSlot>,
}

/// [`Made::cache`]: made on first use, and only tried once.
#[derive(Default)]
struct CacheSlot {
    table: Option<RecordCache>,
    tried: bool,
}

/// [`HostHeap::made`] locked by the pass over buffer views
/// ([`HostHeap::made_counted`]): on a pass's thread, how long it was held is
/// counted when it goes.
struct MadeCounted<'a> {
    made: MadeHeld<'a>,
    held: Option<std::time::Instant>,
}

/// [`HostHeap::made`] held alone, or with the other threads of the pass over
/// buffer views ([`Made`]).
enum MadeHeld<'a> {
    Alone(std::sync::RwLockWriteGuard<'a, Made>),
    Shared(std::sync::RwLockReadGuard<'a, Made>),
    /// VIcky (`VICTORY_MADE_TURNS`): let go for a moment (handed to a
    /// waiting thread), never looked through.
    Gone,
}

impl<'a> MadeCounted<'a> {
    /// `made` taken back alone after a turn (`VICTORY_MADE_TURNS`), its
    /// hold counted from now as [`HostHeap::made_counted`] counts it.
    fn alone(made: std::sync::RwLockWriteGuard<'a, Made>) -> Self {
        MadeCounted { made: MadeHeld::Alone(made), held: crate::draw::heap_pass::parts::counting().then(std::time::Instant::now) }
    }
}

impl Drop for MadeCounted<'_> {
    fn drop(&mut self) {
        if let Some(held) = self.held.take() {
            crate::draw::heap_pass::parts::held(held.elapsed());
        }
    }
}

impl std::ops::Deref for MadeCounted<'_> {
    type Target = Made;

    fn deref(&self) -> &Made {
        match &self.made {
            MadeHeld::Alone(made) => made,
            MadeHeld::Shared(made) => made,
            MadeHeld::Gone => unreachable!("made looked through while let go for a turn"),
        }
    }
}

impl Made {
    /// The host slot title slot `title` is made in, if it has one (none is
    /// taken here): its own below the window, its pool slot past it.
    fn host(&self, title: u32) -> Option<u32> {
        match &self.map {
            Some(map) => map.host(title),
            None => ((title as usize) < self.keys.len()).then_some(title),
        }
    }

    /// The record host slot `host` was last made from (0: none).
    fn key(&self, host: u32) -> u64 {
        self.keys[host as usize].load(std::sync::atomic::Ordering::Relaxed)
    }

    /// [`Made::key`], `None` past the host slots it has keys for.
    fn key_of(&self, host: u32) -> Option<u64> {
        self.keys.get(host as usize).map(|k| k.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn set_key(&self, host: u32, key: u64) {
        self.keys[host as usize].store(key, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The made state held alone, poisoned or not.
fn held_alone(made: &std::sync::RwLock<Made>) -> std::sync::RwLockWriteGuard<'_, Made> {
    crate::lot4::locks::write(crate::lot4::locks::Lock::Made, made)
}

/// VIcky (roadmap A2): `VICTORY_MADE_TURNS=1|alt` (off by default).
///
/// The heap sync makes the slots it finds written in chunks of
/// [`MIRROR_CHUNK`] and the pass over buffer views in chunks of
/// [`PASS_CHUNK`], each chunk with `made` held alone (some 35-45 us in
/// the probe run); between two chunks the maker lets `made` go and takes it
/// again at once. The title's recording threads take `made` for one slot at a
/// time (each view it makes or descriptor it copies, `HostHeap::mirror`),
/// and the standard lock lets nobody in line first: a thread that waits
/// sleeps, and the maker, still running, takes the lock back before the
/// sleeper wakes. P7 counted 982 waits a frame, 165 ms waited, some 170 us
/// each (the pass's own, 188 us each): four to five chunks' worth.
///
/// With the switch on, a thread that takes `made` for a few slots and
/// finds it held says so ([`Turns::waiting`]) and spins for up to
/// [`TURN_SPIN`] before it sleeps. A chunk maker looks at that count
/// before each slot; when someone waits, its chunk ends there (its keeps
/// and marks made, as at a chunk's end), it lets `made` go, waits up to
/// [`TURN_WAIT`] for a waiting thread to have taken it ([`Turns::served`])
/// and takes it again. What each slot's making does is the same, and the
/// chunks are cut where a waiting thread comes in: as if the chunks were
/// smaller there. The chunk makers do not say when they wait (the sync and
/// the pass would otherwise hand `made` to each other slot by slot).
/// `alt`: twenty seconds as before, twenty with the turns, in turn.
pub(crate) static MADE_TURNS: crate::alt::Alt =
    crate::alt::Alt::new("VICTORY_MADE_TURNS", "the heap's slot state goes to a thread waiting for it between a chunk's slots");

/// How long a thread that waits for `made` ([`HostHeap::made_alone`])
/// spins before it sleeps: a chunk maker lets it in after its slot (some
/// 0.5-0.7 us, more when the slot's lookups wait).
const TURN_SPIN: std::time::Duration = std::time::Duration::from_micros(30);

/// How long a chunk maker that let `made` go waits for a waiting thread to
/// take it, and spins for it back, before it sleeps for it.
const TURN_WAIT: std::time::Duration = std::time::Duration::from_micros(100);

#[cfg(test)]
thread_local! {
    /// Tests: whether this thread takes turns, whatever the switch.
    pub(crate) static TEST_TURNS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether `made` goes to its waiting threads between a chunk's slots now.
fn turns_now() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_TURNS.get() {
        return on;
    }
    MADE_TURNS.on()
}

/// VIcky (`VICTORY_MADE_TURNS`): the threads waiting for one heap's `made`
/// that its chunk makers let in, on lines of their own. Written only by
/// threads that find `made` held; the chunk makers read `waiting` before
/// each slot.
#[derive(Default)]
#[repr(align(128))]
struct Turns {
    /// Threads waiting for `made` now ([`HostHeap::made_alone`]).
    waiting: std::sync::atomic::AtomicU32,
    /// Waiting threads that got it, since the start.
    served: std::sync::atomic::AtomicU32,
    /// VIcky (`VICTORY_MADE_PASS_FIRST`, [`made_first`]): the pass over
    /// buffer views waiting for `made` now (its chunks made alone,
    /// [`HostHeap::made_pass_first`]), and its waits that got it, since the
    /// start. Only the heap sync's chunks look at these.
    pass_waiting: std::sync::atomic::AtomicU32,
    pass_served: std::sync::atomic::AtomicU32,
}

/// Tests: `made` handed over between a chunk's slots, since the start.
#[cfg(test)]
pub(crate) static TURNS_TAKEN: AtomicU64 = AtomicU64::new(0);

/// `made`'s write lock tried once, poisoned or not.
fn made_tried(made: &std::sync::RwLock<Made>) -> Option<std::sync::RwLockWriteGuard<'_, Made>> {
    match made.try_write() {
        Ok(made) => Some(made),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(e.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

impl HostHeap {
    /// `made` held alone by a thread that makes one slot or a few (a view
    /// the title made or copied, a texture's slot, the diagnostics' looks):
    /// [`held_alone`]; with [`MADE_TURNS`] on, a wait is said to the chunk
    /// makers ([`Turns::waiting`]), which let it in after their slot, and is
    /// spent spinning for up to [`TURN_SPIN`] before sleeping.
    fn made_alone(&self) -> std::sync::RwLockWriteGuard<'_, Made> {
        use std::sync::atomic::Ordering::SeqCst;
        if !turns_now() {
            return held_alone(&self.made);
        }
        if let Some(made) = made_tried(&self.made) {
            crate::lot4::locks::tried(crate::lot4::locks::Lock::Made, true);
            return made;
        }
        let began = std::time::Instant::now();
        self.turns.waiting.fetch_add(1, SeqCst);
        let mut got = None;
        while got.is_none() && began.elapsed() < TURN_SPIN {
            for _ in 0..32 {
                std::hint::spin_loop();
            }
            got = made_tried(&self.made);
        }
        let made = got.unwrap_or_else(|| self.made.write().unwrap_or_else(|e| e.into_inner()));
        self.turns.served.fetch_add(1, SeqCst);
        self.turns.waiting.fetch_sub(1, SeqCst);
        crate::lot4::locks::waited(crate::lot4::locks::Lock::Made, began.elapsed().as_nanos() as u64);
        made
    }

    /// Whether a chunk maker holding `made` (with the turns `on`) lets a
    /// waiting thread in now.
    #[inline]
    fn turn_wanted(&self, on: bool) -> bool {
        on && self.turns.waiting.load(std::sync::atomic::Ordering::Relaxed) > 0
    }

    /// A chunk maker lets `made` go to the threads waiting for it
    /// ([`HostHeap::turn_wanted`]) and takes it back ([`HostHeap::turn_back`]).
    fn turn_over<'a>(&'a self, made: std::sync::RwLockWriteGuard<'a, Made>) -> std::sync::RwLockWriteGuard<'a, Made> {
        let began = std::time::Instant::now();
        let served = self.turns.served.load(std::sync::atomic::Ordering::SeqCst);
        drop(made);
        self.turn_back(served, began)
    }

    /// The second half of [`HostHeap::turn_over`], `made` let go at `began`
    /// with `served` waiting threads served then: waits up to [`TURN_WAIT`]
    /// for one more to be served, spins for `made` back for the rest of it,
    /// then sleeps for it. Counted as a turn (P7, `MadeTurn`).
    fn turn_back(&self, served: u32, began: std::time::Instant) -> std::sync::RwLockWriteGuard<'_, Made> {
        use std::sync::atomic::Ordering::SeqCst;
        while self.turns.served.load(SeqCst) == served && self.turns.waiting.load(SeqCst) > 0 && began.elapsed() < TURN_WAIT {
            std::hint::spin_loop();
        }
        let mut made = None;
        while made.is_none() && began.elapsed() < TURN_WAIT {
            for _ in 0..32 {
                std::hint::spin_loop();
            }
            made = made_tried(&self.made);
        }
        let made = made.unwrap_or_else(|| self.made.write().unwrap_or_else(|e| e.into_inner()));
        #[cfg(test)]
        TURNS_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::lot4::locks::waited(crate::lot4::locks::Lock::MadeTurn, began.elapsed().as_nanos() as u64);
        made
    }
}

thread_local! {
    /// VIcky (`VICTORY_HEAP_PASS_THREADS`): the title slots past the
    /// identity window that a thread of the pass over buffer views found
    /// no pool slot for, the pool used up ([`HostHeap::take_room_wanted`]).
    static ROOM_WANTED: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The heap map of a view heap with a [`MapLayout`]: an upload buffer of one
/// entry per title slot from the identity window on ([`slot_map::entry`]:
/// the bitwise not of its host slot, 0 for none), persistently mapped, and
/// read by translated shaders through a raw view at the layout's map slot
/// (`victory_shader::emit::set_view_heap_map`).
///
/// [`slot_map::entry`]: super::slot_map::entry
struct HeapMap {
    buffer: ID3D12Resource,
    /// Where the entries are mapped.
    at: usize,
}

/// The title's view slots in a view heap whose host heap is capped below
/// them, with a heap map: title slots `0..window` at their own index, as
/// tables made in place need; `pool` slots for the title slots past the
/// window that hold views (looked up in the map by translated shaders);
/// the map's raw view at `slot`; then the fork's [`DYNAMIC`] dynamic slots
/// and its null guard slot last, where they were without a map.
///
/// On a capped host this layout uses 1,000,000 shader-visible views
/// ([`PORTABLE_VIEWS`]); the title's has 1,245,184 slots, some
/// 650,000-700,000 of them holding views, about 335,000 of those past the
/// 737,855 title slots a capped heap kept in place (their bindless reads
/// gave the null slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MapLayout {
    window: u32,
    pool: u32,
    slot: u32,
}

impl MapLayout {
    /// The layout for `title_count` title slots, an identity window of up
    /// to `window` slots and the map at `slot`.
    fn new(title_count: u32, window: u32, slot: u32) -> Self {
        let window = window.min(slot).min(title_count);
        MapLayout {
            window,
            pool: (slot - window).min(title_count - window),
            slot,
        }
    }

    /// The first dynamic slot.
    fn dynamic_first(self) -> u32 {
        self.slot + 1
    }

    /// The null guard slot, the heap's last: the limit translated shaders
    /// clamp view indices to.
    fn guard(self) -> u32 {
        self.dynamic_first() + DYNAMIC
    }

    /// The heap's slots.
    fn size(self) -> u32 {
        self.guard() + 1
    }
}

/// Bytes per title view descriptor.
const TITLE_SLOT_BYTES: u64 = 32;

/// The identity window of a heap map without `VICTORY_HEAP_WINDOW`: the
/// title's tables and its first two 64Ki runs of views lie below it, its
/// first empty run (110,940..327,680) spans it.
const WINDOW: u32 = 1 << 17;

/// The heap map's slot in a capped view heap: right before the dynamic
/// slots, so that those and the guard slot stay where they were
/// (`PORTABLE_VIEWS - DYNAMIC - 1` on, the guard at `PORTABLE_VIEWS - 1`).
/// No view index reaches it: one below the window is its own, one past it
/// is a pool slot or the guard, and tables made in place lie below the
/// window.
const MAP_SLOT: u32 = PORTABLE_VIEWS - DYNAMIC - 2;

/// The identity window: [`WINDOW`], or `VICTORY_HEAP_WINDOW` (a smaller one
/// maps more slots, a larger one fewer; at most [`MAP_SLOT`]).
fn window() -> u32 {
    std::env::var("VICTORY_HEAP_WINDOW")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(WINDOW)
        .min(MAP_SLOT)
}

/// Whether a capped view heap gets a heap map; `VICTORY_HEAP_MAP=0` keeps
/// the fork's cap (views past it read the null slot).
fn heap_map_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_MAP").is_none_or(|v| v != "0"))
}

/// The heap map translated shaders read: the identity window and the
/// map's slot ([`probe_view_limit`] sets it before the first translation).
fn heap_map() -> Option<(u32, u32)> {
    // Tests: only the thread's own map. `HostHeap::new` probes the host in
    // tests too, which on a capped card sets the translator's map for the
    // whole process.
    #[cfg(test)]
    return TEST_MAP.get();
    #[cfg(not(test))]
    victory_shader::emit::view_heap_map()
}

/// VIcky (diagnostics): the heap map's pool (title slots mapped, pool
/// slots free and waiting, refusals), each time the title slots it holds
/// pass another 64Ki.
#[cfg(feature = "diagnostics")]
fn note_pool(made: &Made) {
    use std::sync::atomic::Ordering::Relaxed;
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let Some(stats) = made.map.as_ref().map(SlotMap::stats) else {
        return;
    };
    let step = stats.mapped >> 16;
    if step > LOGGED.load(Relaxed) {
        LOGGED.store(step, Relaxed);
        tracing::info!(?stats, "heap map pool");
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: the heap map (window, slot) of the view heaps this thread
    /// makes, as [`probe_view_limit`] sets one on a capped host, without
    /// reaching other tests' heaps.
    pub(crate) static TEST_MAP: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// Title memory by the placed texture views that describe it: per span of
/// memory (its start, its end at full size) the title slots holding a
/// view of a texture that takes exactly that span. A console title need
/// not clear a descriptor when it streams a texture out, so a newer view
/// that cannot be there together with an older one is all that says the
/// older one is stale ([`Rule`] says when that is). Views of one span with
/// different descriptions (the formats of one family, say) are one
/// texture's.
///
/// Spans may overlap: a texture's smallest levels lie at the start of its
/// memory, and a title that holds only those puts the next texture where
/// the larger ones would be (the title: three 512x256 textures
/// of 0x2c000 bytes each whose starts are 0x2000 and 0x4000 apart, all
/// three read by the same frame's draws, run 188). What is there of a
/// texture ends where the next one starts.
/// No host heap is needed, so it is tested without one.
#[derive(Default)]
pub(crate) struct Spans {
    /// When a newer view makes an older one stale.
    rule: Rule,
    by_start: std::collections::BTreeMap<u64, (u64, std::collections::BTreeSet<u32>)>,
    by_slot: std::collections::HashMap<u32, u64>,
    /// The level each slot's view starts at ([`Spans::shows`]).
    first: std::collections::HashMap<u32, u32>,
    /// The bytes a placed texture takes, by resource description: worked
    /// out once for each (a title writes the views of a few thousand
    /// descriptions, thousands of times a frame).
    sizes: std::collections::HashMap<[u32; 14], Option<u64>>,
}

/// When a view the title writes makes an older view of other memory stale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Rule {
    /// When it starts where the older one starts: at the same address
    /// with another length, or within the older one's first [`HEAD`]
    /// bytes (or the older within its own). No two textures share the
    /// block of their smallest levels. A view that only reaches into the
    /// address range of an older texture's larger levels says nothing.
    #[default]
    Start,
    /// Whenever their spans at full size overlap (the rule until run
    /// 188, where it made null the slots of textures packed as above:
    /// each view written took its neighbours out of the queue and off the
    /// card, for good). `VICTORY_STALE_TEXTURES=overlap`.
    Overlap,
    /// Never; a view that would be stale under [`Rule::Overlap`] is left
    /// out of the spans instead. `VICTORY_STALE_TEXTURES=0`, as before.
    Never,
}

/// The first bytes of a placed texture that are its own under
/// [`Rule::Start`]: the 4 KiB block its smallest levels share. (Of 129
/// pairs of overlapping neighbours in the listings of runs 188 and 189,
/// none starts closer than 0x2000 to the one before it.)
const HEAD: u64 = 4096;

impl Spans {
    /// The bytes of title memory a placed texture with the resource
    /// description `desc` takes (`crate::resource::placed_bytes`).
    fn bytes(&mut self, desc: &[u32; 14]) -> Option<u64> {
        if let Some(&known) = self.sizes.get(desc) {
            return known;
        }
        let bytes = crate::resource::placed_bytes(desc);
        self.sizes.insert(*desc, bytes);
        bytes
    }

    /// The starts of the spans that overlap `[start, end)`, each taken to
    /// end where the next one starts: the last span that starts before
    /// `start`, if it reaches past it at full size, and every span that
    /// starts inside.
    fn overlapping(&self, start: u64, end: u64) -> Vec<u64> {
        let before = self
            .by_start
            .range(..start)
            .next_back()
            .filter(|(_, span)| span.0 > start)
            .map(|(at, _)| *at);
        before.into_iter().chain(self.by_start.range(start..end).map(|(at, _)| *at)).collect()
    }

    /// Slot `index` holds a view of a texture in `[start, end)`. `newer`:
    /// the title wrote it now, so the slots whose views cannot be there
    /// with it ([`Rule`]) are stale: they are returned, and forgotten.
    /// Otherwise (a draw made the slot from what it held), and under
    /// [`Rule::Never`], a view that clashes with other spans is left out.
    fn note(&mut self, index: u32, start: u64, end: u64, newer: bool) -> Vec<u32> {
        let same = |span: Option<&(u64, std::collections::BTreeSet<u32>)>| span.is_some_and(|s| s.0 == end);
        if self.by_slot.get(&index) == Some(&start) && same(self.by_start.get(&start)) {
            return Vec::new();
        }
        self.forget(index);
        if end <= start {
            return Vec::new();
        }
        let rule = self.rule;
        // The spans the view cannot be there with.
        let clashing: Vec<u64> = match rule {
            Rule::Start => self
                .by_start
                .range(start.saturating_sub(HEAD - 1)..start.saturating_add(HEAD))
                .map(|(at, _)| *at)
                .collect(),
            Rule::Overlap | Rule::Never => self.overlapping(start, end),
        };
        let mut stale = Vec::new();
        for at in clashing {
            if at == start && same(self.by_start.get(&at)) {
                continue;
            }
            if !newer || rule == Rule::Never {
                return stale;
            }
            if let Some((old_end, slots)) = self.by_start.remove(&at) {
                // The first of them are said with both spans, to tell a
                // replacement from a neighbour in one launch's log.
                static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
                    tracing::info!(
                        slot = index,
                        start = format_args!("{start:#x}"),
                        end = format_args!("{end:#x}"),
                        stale_start = format_args!("{at:#x}"),
                        stale_end = format_args!("{old_end:#x}"),
                        stale_slots = slots.len(),
                        rule = ?rule,
                        "a placed texture view makes older views of other memory stale"
                    );
                }
                for slot in slots {
                    self.by_slot.remove(&slot);
                    stale.push(slot);
                }
            }
        }
        self.by_start.entry(start).or_insert_with(|| (end, Default::default())).1.insert(index);
        self.by_slot.insert(index, start);
        stale
    }

    /// The view in slot `index` starts at level `first` of its texture.
    fn shows(&mut self, index: u32, first: u32) {
        self.first.insert(index, first);
    }

    /// The lowest level a view of the texture at `start` starts at.
    fn first_viewed(&self, start: u64) -> Option<u32> {
        self.by_start.get(&start)?.1.iter().filter_map(|slot| self.first.get(slot)).min().copied()
    }

    /// The first span that starts inside `[start, end)` past the first
    /// [`HEAD`] bytes of it.
    fn next_start(&self, start: u64, end: u64) -> Option<u64> {
        self.by_start.range(start.saturating_add(HEAD)..end).next().map(|(at, _)| *at)
    }

    /// The slots that hold a view of the texture at `start`.
    fn slots_at(&self, start: u64) -> Vec<u32> {
        self.by_start.get(&start).map(|span| span.1.iter().copied().collect()).unwrap_or_default()
    }

    /// Slot `index` holds no placed texture view any more.
    fn forget(&mut self, index: u32) {
        self.first.remove(&index);
        let Some(start) = self.by_slot.remove(&index) else {
            return;
        };
        if let Some(span) = self.by_start.get_mut(&start) {
            span.1.remove(&index);
            if span.1.is_empty() {
                self.by_start.remove(&start);
            }
        }
    }

    /// The slots whose texture's memory overlaps `[start, end)`.
    fn slots_in(&self, start: u64, end: u64) -> Vec<u32> {
        self.overlapping(start, end)
            .iter()
            .filter_map(|at| self.by_start.get(at))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }

    /// The slots of the textures that hold bytes of `[start, end)`, for a
    /// write there: as [`Spans::slots_in`], but when a texture starts at
    /// `start` itself the one before it is left out (it ends where that
    /// one starts, and a request that fills a texture begins at its first
    /// byte: `slots_in` would name its neighbour with every such write).
    fn slots_holding(&self, start: u64, end: u64) -> Vec<u32> {
        if end <= start {
            return Vec::new();
        }
        let before = if self.by_start.contains_key(&start) {
            None
        } else {
            self.by_start.range(..start).next_back().filter(|(_, span)| span.0 > start).map(|(_, span)| span)
        };
        before
            .into_iter()
            .chain(self.by_start.range(start..end).map(|(_, span)| span))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }
}

/// Title memory written behind the guards' back whose bytes are there
/// (DirectStorage's reads into physical pages, at the addresses the title
/// has them mapped; `crate::watch::title_memory_written`), since a texture
/// pass last took the ranges ([`page_written`]). Kept apart from
/// [`REMAPPED`] so that what such writes do is counted by itself.
static PAGE_WRITTEN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Since the start: such ranges kept for a pass; ranges dropped because no
/// pass took them; slots of placed textures that held bytes of them when a
/// pass took them; of those, the slots whose texture had been uploaded
/// before the write (out of the queue, or in it after an upload): they are
/// the ones that kept the bytes from before, and the ones made due when
/// `VICTORY_PAGE_WRITES` asks for it.
pub(crate) static PAGE_WRITES: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Frames from the pass that takes a written range to the look at its
/// slots: the levels of one texture arrive in a few requests, and a look
/// two frames on takes those of the next frame with it.
const PAGE_WRITE_AFTER: u64 = 2;

/// The bytes of a write the guards do not see are in title memory `[at,
/// at + len)`: the placed textures there changed with no descriptor
/// written and nothing mapped. Kept for the next texture pass
/// ([`HostHeap::texture_slots_page_written`]), as [`remapped`] keeps its
/// ranges; locks one mutex and calls nothing.
pub(crate) fn page_written(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    PAGE_WRITES[0].fetch_add(1, Relaxed);
    if let Some(last) = ranges.last_mut()
        && last.1 == start
    {
        last.1 = end;
        return;
    }
    if ranges.len() >= MOST {
        PAGE_WRITES[1].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end));
}

/// Title memory the runtime mapped, unmapped or is about to write since a
/// texture pass last took the ranges ([`remapped`]).
static REMAPPED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// The lowest start and the highest end of any span of [`Spans`] so far
/// (the start is `u64::MAX` until there is one): what the runtime does to
/// memory outside is none of the texture pass's business.
/// VIcky: whether a slot's write of no placed texture view leaves the
/// spans of placed textures (`HostHeap::placed`) and their lock alone when
/// the slot is not in them (`VICTORY_SPANS_QUICK=1|alt`; off by default,
/// every write takes the lock, as before). Port of Belobug a284ca74, where
/// it is on by default: every record the heap sync mirrors went by that
/// one lock of the heap, from all its threads, with the title's own threads
/// and the texture passes (4 of 16 ms of the sync's wait there, run
/// city4). What is done is the same: forgetting a slot that is not there
/// does nothing. Here `VICTORY_HEAP_NOTE_KIND` already spares the lock for
/// notes from and to plain buffer views and bare slots; this spares it for
/// the other writes of slots that hold no placed texture view (views of
/// texture objects, constant views, typed and structured buffer views).
pub(crate) static SPANS_QUICK: crate::alt::Alt = crate::alt::Alt::new(
    "VICTORY_SPANS_QUICK",
    "writes of heap slots of no placed texture view leave the spans' lock alone when the slot is not in them",
);

/// Since the last sync line, with [`SPANS_QUICK`] set: writes that left the
/// spans alone, writes that took their lock.
pub(crate) static SPANS_QUICK_COUNTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

static SPANNED: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(u64::MAX),
    std::sync::atomic::AtomicU64::new(0),
];

/// Since the start: slots whose view a newer view of other memory there
/// made stale; ranges of memory noted as mapped, unmapped or written;
/// ranges dropped because no pass took them; slots made due by such
/// ranges; and made slots whose memory was found unmapped (made null).
pub(crate) static STALE: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// When a view the title writes makes older views of other memory stale
/// ([`Rule`]): where it starts in the block of their smallest levels, by
/// default; `VICTORY_STALE_TEXTURES=0`: never, as before;
/// `VICTORY_STALE_TEXTURES=overlap`: wherever their spans overlap, the rule
/// until run 188.
fn stale_rule() -> Rule {
    static RULE: std::sync::OnceLock<Rule> = std::sync::OnceLock::new();
    *RULE.get_or_init(|| match std::env::var("VICTORY_STALE_TEXTURES").as_deref() {
        Ok("0") => Rule::Never,
        Ok("overlap") => Rule::Overlap,
        _ => Rule::Start,
    })
}

/// The runtime mapped or unmapped title memory `[at, at + len)`, or is
/// about to write it through the kernel (`crate::watch`): the placed
/// textures there may have arrived, gone, or changed with no descriptor
/// written. The range is kept for the next texture pass
/// ([`HostHeap::texture_slots_remapped`]). Called with the runtime's own
/// locks held: this locks one mutex of its own and calls nothing.
pub(crate) fn remapped(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = REMAPPED.lock().unwrap_or_else(|e| e.into_inner());
    STALE[1].fetch_add(1, Relaxed);
    // Pages are mapped one after the other.
    if let Some(last) = ranges.last_mut()
        && last.1 == start
    {
        last.1 = end;
        return;
    }
    if ranges.len() >= MOST {
        // No pass runs (no shader that picks textures by index is bound):
        // the slots there are looked at when they are due, as before.
        STALE[2].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end));
}

/// Frames for which a slot a draw read through a table counts as a table's
/// slot ([`HostHeap::tabled`]).
const TABLED_FRAMES: u64 = 240;

/// Frames a write of a table's slot waits in the queue for the draw that
/// reads it, before the slot is taken for one shaders pick by index.
/// Frames as `crate::frame::current` counts them: in its opening cutscene
/// The title begins two or three of them per frame it presents
/// (435 begun in the 353 s of run 52's, for some 140 presented).
const TABLED_GRACE: u64 = 8;

/// In [`HostHeap::tabled`]: a draw read the slot in that frame.
const TABLED_READ: u32 = 1 << 31;

/// Since the start: writes of texture views into slots that count as a
/// table's (due later), and slots that left the queue because a draw read
/// them through a table.
pub(crate) static TABLED_WRITES: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether slots draws read through tables leave the queue of textures
/// picked by index; `VICTORY_TABLED_TEXTURES=1` keeps them in it as
/// before.
fn tabled_out() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLED_TEXTURES").is_none_or(|v| v != "1"))
}

/// What the queue of textures picked by index knows of a title slot.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Waiting {
    /// The frame from which the slot is due.
    due: u64,
    /// The frame it was written in.
    written: u64,
    /// Which write of the queue put it here: the pass hands it back with
    /// what it did, so a slot written again meanwhile stays as written.
    write: u64,
    /// Looks in a row that found nothing to upload (its memory not mapped,
    /// or no more of it than at the last upload).
    looks: u8,
    /// Bytes of its memory that were mapped when it was last uploaded
    /// without all of it there; 0: not uploaded yet.
    mapped: u64,
    /// For a slot uploaded whole and looked at again
    /// ([`TextureQueue::recheck`]): where its memory is (address, bytes),
    /// the fingerprint it had at the upload, how many looks found it
    /// unchanged, and the frame of the upload. A fingerprint of 0: none.
    /// VIcky (`VICTORY_RECHECK_PARTLY`): also for a slot uploaded in part
    /// (`mapped` not 0, [`TextureQueue::recheck_partly`]).
    checked: (u64, u64, u64, u8, u64),
    /// The ranges of its memory the fingerprint reads at the start of its
    /// mips (`crate::texture::staged_memory`), with `checked` (262625e33).
    mips: Option<Mips>,
    /// VIcky (`VICTORY_DS_HOLD`): looks held back because a DirectStorage
    /// write was pending under its memory ([`TextureQueue::ds_hold`]).
    holds: u8,
}

impl Waiting {
    /// A slot written in `frame`, write `write` of the queue, due from `due`,
    /// nothing known of it yet.
    fn new(due: u64, frame: u64, write: u64) -> Waiting {
        Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0), mips: None, holds: 0 }
    }
}

/// Ranges of a texture's memory (offsets, bytes) read for its fingerprint.
pub(crate) type Mips = std::sync::Arc<[(u64, u64)]>;

/// The queue of title slots whose texture view waits to be brought up to
/// date for shaders that pick textures by index: which are due, and in
/// which order (no host heap needed, so it is tested without one).
#[derive(Default)]
pub(crate) struct TextureQueue {
    slots: std::collections::BTreeMap<u32, Waiting>,
    writes: u64,
    /// Slots held back because the card had no room for the host copy
    /// they need ([`TextureQueue::hold`]), each with the write it was
    /// held as.
    held: std::collections::BTreeMap<u32, u64>,
}

impl TextureQueue {
    /// A texture view was written into slot `index` in `frame`, due from
    /// frame `due`. `seen`: it was not written, a draw or the pass made
    /// the host slot from a view the queue may hold already, which then
    /// stays as it is.
    fn written(&mut self, index: u32, frame: u64, due: u64, seen: bool) {
        if seen && self.slots.contains_key(&index) {
            return;
        }
        self.writes += 1;
        let write = self.writes;
        self.slots.insert(index, Waiting::new(due, frame, write));
    }

    fn remove(&mut self, index: u32) {
        self.slots.remove(&index);
    }

    /// Up to `max` slots due at `frame`, each with its write number, in
    /// the order to serve them:
    /// - first the ones looked at before (their memory was not mapped, or
    ///   only part of it): the look is a query, and memory that has
    ///   arrived since is what the title streamed in last;
    /// - then the others, the newest write first and the oldest next, in
    ///   turns: slot order served the low slots of a backlog of
    ///   25,000-40,000 while the textures of the shot on screen waited
    ///   (run 52: 98% of the drawn pixels had no colour texture in the
    ///   first 3D frames).
    fn due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        let mut again: Vec<(u64, u32, u64)> = Vec::new();
        let mut fresh: Vec<(u64, u32, u64)> = Vec::new();
        for (&index, w) in self.slots.iter().filter(|(_, w)| w.due <= frame) {
            let class = if w.looks > 0 || w.mapped > 0 { &mut again } else { &mut fresh };
            class.push((w.written, index, w.write));
        }
        // By the number of the write, counted over the run.
        let newest_first = |a: &(u64, u32, u64), b: &(u64, u32, u64)| b.2.cmp(&a.2);
        again.sort_unstable_by(newest_first);
        fresh.sort_unstable_by(newest_first);
        let mut order: Vec<(u32, u64)> = again.iter().map(|&(_, index, write)| (index, write)).collect();
        let (mut new, mut old) = (0, fresh.len());
        while new < old && order.len() < max {
            order.push((fresh[new].1, fresh[new].2));
            new += 1;
            if new < old && order.len() < max {
                old -= 1;
                order.push((fresh[old].1, fresh[old].2));
            }
        }
        order.truncate(max);
        order
    }

    /// The entry of `index` if it is still the write the pass was handed.
    fn same(&mut self, index: u32, write: u64) -> Option<&mut Waiting> {
        self.slots.get_mut(&index).filter(|w| w.write == write)
    }

    /// Slot `index` is up to date: out of the queue until written again.
    fn done(&mut self, index: u32, write: u64) {
        if self.same(index, write).is_some() {
            self.slots.remove(&index);
        }
    }

    /// Nothing to upload at slot `index` now: looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    fn later(&mut self, index: u32, write: u64, frame: u64, most: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = frame + (1u64 << w.looks.min(16)).min(most.max(1));
            w.looks = w.looks.saturating_add(1);
        }
    }

    /// The card has no room for the host copy slot `index` needs: it is
    /// not handed out again before frame `due`, or before the card has
    /// room again ([`TextureQueue::room`]). Nothing else of it changes: it
    /// is not a slot "waiting for its memory", and when it is due it has
    /// its old place among the slots never looked at. (It stayed due, so
    /// every pass on a full card looked at the same few hundred slots
    /// again and stopped there, and the slots behind them were never
    /// reached: the ones whose host copy exists and only wants bringing up
    /// to date, and the ones whose memory went, which is what lets host
    /// copies go and makes room.)
    fn hold(&mut self, index: u32, write: u64, due: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = w.due.max(due);
            self.held.insert(index, write);
        }
    }

    /// The card has room again at `frame`: the slots held back for lack of
    /// it are due now, as if they had never been held. Returns how many.
    fn room(&mut self, frame: u64) -> usize {
        let held = std::mem::take(&mut self.held);
        let mut back = 0;
        for (index, write) in held {
            if let Some(w) = self.same(index, write) {
                w.due = w.due.min(frame);
                back += 1;
            }
        }
        back
    }

    /// Slot `index` was uploaded with `mapped` bytes of its memory there,
    /// not all of it: looked at again from frame `due`.
    fn partly(&mut self, index: u32, write: u64, due: u64, mapped: u64) {
        if let Some(w) = self.same(index, write) {
            // VIcky: the fingerprint of an earlier upload in part
            // ([`TextureQueue::recheck_partly`], only ever noted with
            // `VICTORY_RECHECK_PARTLY`) is not this upload's.
            if w.mapped > 0 && w.checked.2 != 0 {
                (w.checked, w.mips) = ((0, 0, 0, 0, 0), None);
            }
            (w.due, w.looks, w.mapped) = (due, 0, mapped.max(1));
        }
    }

    /// Slot `index` was uploaded whole from memory `at` (address, bytes)
    /// with fingerprint `print`: looked at again from frame `due`, for a
    /// title that fills a texture's memory after it wrote the view (the
    /// `looks`-th such look; `uploaded` is the upload's frame).
    /// `mips`: the ranges the fingerprint read besides its windows.
    #[allow(clippy::too_many_arguments)]
    fn recheck(
        &mut self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        mips: Option<Mips>,
        print: u64,
        looks: u8,
        uploaded: u64,
    ) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 1, 0);
            w.checked = (at.0, at.1, print, looks, uploaded);
            w.mips = mips;
        }
    }

    /// VIcky (`VICTORY_RECHECK_PARTLY`): slot `index` was uploaded with
    /// `mapped` bytes of its memory there, not all ([`TextureQueue::partly`],
    /// whose part this keeps), and that memory had fingerprint `print`
    /// (`at`, `mips` and `uploaded` as for [`TextureQueue::recheck`]): its
    /// looks also ask the fingerprint, so bytes that land in pages mapped
    /// already are uploaded too, not only more pages mapped.
    #[allow(clippy::too_many_arguments)]
    fn recheck_partly(
        &mut self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        mips: Option<Mips>,
        print: u64,
        uploaded: u64,
        mapped: u64,
    ) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 0, mapped.max(1));
            w.checked = (at.0, at.1, print, 0, uploaded);
            w.mips = mips;
        }
    }

    /// What [`TextureQueue::recheck`] or [`TextureQueue::recheck_partly`]
    /// noted of slot `index`, if it is waiting for such a look, with the
    /// ranges the fingerprint read and the bytes that were mapped at the
    /// upload (0: it was uploaded whole).
    fn checked(&self, index: u32) -> Option<((u64, u64, u64, u8, u64), Option<Mips>, u64)> {
        self.slots.get(&index).filter(|w| w.checked.2 != 0).map(|w| (w.checked, w.mips.clone(), w.mapped))
    }

    /// VIcky (`VICTORY_DS_HOLD`): a DirectStorage write is pending under slot
    /// `index`'s memory: not handed out before frame `due`, at most `most`
    /// times for this write of the slot. False once it was held that often
    /// (it is uploaded now, so that nothing waits for good); true also for a
    /// slot written again meanwhile (the new write is due as written).
    fn ds_hold(&mut self, index: u32, write: u64, due: u64, most: u8) -> bool {
        let Some(w) = self.same(index, write) else {
            return true;
        };
        if w.holds >= most {
            return false;
        }
        w.holds += 1;
        w.due = w.due.max(due);
        true
    }

    /// The memory slot `index`'s texture lies in was mapped, unmapped or
    /// written in `frame`: the slot is looked at from frame `due` as one
    /// written now, whatever was known of it (a pass at work on it does
    /// not take it out).
    fn changed(&mut self, index: u32, frame: u64, due: u64) {
        self.writes += 1;
        let write = self.writes;
        let due = self.slots.get(&index).map_or(due, |w| w.due.min(due));
        self.slots.insert(index, Waiting::new(due, frame, write));
    }

    /// Bytes of slot `index`'s memory that were mapped at its last upload
    /// (0: not uploaded, or all of it was).
    fn mapped(&self, index: u32) -> u64 {
        self.slots.get(&index).map_or(0, |w| w.mapped)
    }

    /// For the frame line: slots in the queue; of them due at `frame`;
    /// waiting for their memory (looked at, never uploaded); uploaded in
    /// part; and the frames the oldest due one has waited since its write.
    fn census(&self, frame: u64) -> [u64; 5] {
        let mut c = [self.slots.len() as u64, 0, 0, 0, 0];
        for w in self.slots.values() {
            if w.mapped > 0 {
                c[3] += 1;
            } else if w.looks > 0 {
                c[2] += 1;
            }
            if w.due <= frame {
                c[1] += 1;
                c[4] = c[4].max(frame.saturating_sub(w.written));
            }
        }
        c
    }
}

/// Whether `view` is of a texture the title may fill itself: a texture it
/// placed in memory, or a texture object.
pub(crate) fn is_texture_view(view: &View) -> bool {
    match view {
        View::Placed(false, resource, _) => resource[0] != 1,
        View::Shader(r, _) => crate::resource::buffer_bytes(*r as crate::object::This).is_none(),
        _ => false,
    }
}

/// Buffer views left null because the host would refuse them
/// ([`buffer_view_ok`]), since the start.
pub(crate) static REFUSED_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the host device is gone, and why (the first time it is asked
/// after that): for passes that make many views, to name the one that did
/// it.
pub(crate) fn device_removed() -> Option<String> {
    let host = host()?;
    // SAFETY: a plain query.
    let reason = unsafe { host.device.GetDeviceRemovedReason() };
    reason.err().map(|e| e.to_string())
}

/// Writes of title view-heap slots with a placed texture view and with a
/// view of a texture object, since the start (diagnostic counts).
pub(crate) static PLACED_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static OBJECT_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes per element of a buffer view, as the console's descriptor holds
/// them for indexed reads (structured: the stride; raw: 1, the index is the
/// byte; typed: the format's); `None` if `view` is not a buffer view.
fn view_stride(view: &View) -> Option<u32> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => {
            (d[..10].try_into().ok()?, *unordered)
        }
        _ => return None,
    };
    // SAFETY: buffer views' union members.
    let (stride, raw, format) = unsafe {
        if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, d.Format)
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, d.Format)
        }
    };
    Some(if stride > 0 {
        stride
    } else if raw {
        1
    } else {
        crate::resource::element(format.0 as u32).map_or(0, |(bytes, _)| bytes as u32)
    })
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The size of every host view heap on a host that takes more than
/// [`PORTABLE_VIEWS`] (AMD): the title's heap (a little over 1,000,000
/// views in the retail game) plus [`DYNAMIC`] fits with room to spare, and
/// the last slot is the null slot every view index at or past it reads.
const UNCAPPED_VIEWS: u32 = 1 << 21;

/// Picks, once, the null slot at the end of every host view heap and tells
/// the translator now (`set_view_heap_limit`) instead of when the title's
/// heap is made: a shader keeps the limit it was translated with, and the
/// title makes 25 pipelines (their shaders translated at once) before it
/// makes its view heap. Called before a shader is translated and before a
/// view heap is made.
///
/// Every host clamps. A host that caps view heaps at [`PORTABLE_VIEWS`]
/// (NVIDIA) hung on a view index past the heap; one that does not (AMD
/// RDNA3, RX 7900 XT) took a page fault and removed the device on entering
/// the world: a compute program read view slot 16,777,088 (0xffff80) of a
/// heap of 1,507,328, past its end. Unclamped, that index is undefined on
/// any host.
///
/// A capped host also gets a heap map (VIcky; unless `VICTORY_HEAP_MAP=0`):
/// the translator looks bindless views past the identity window up in it
/// (`set_view_heap_map`), and from then on every view heap of more title
/// slots than the window is laid out for it ([`MapLayout`], whatever its
/// size: the map's slot is fixed in the translations; its guard slot is
/// this null slot). The probe asks for [`UNCAPPED_VIEWS`] (2^21, the
/// translator's whole index range), so a host that takes 1,000,001 views
/// but not the title's heap (an RTX 4060 Laptop) gets the map too.
pub(crate) fn probe_view_limit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(host) = host() else { return };
        let fits = |count: u32| {
            // SAFETY: plain heap creation; the heap is released at once.
            let made: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            };
            made.is_ok()
        };
        let views = if fits(UNCAPPED_VIEWS) {
            UNCAPPED_VIEWS
        } else {
            PORTABLE_VIEWS
        };
        tracing::info!(
            views,
            "host view heaps hold {views} views; view indices are clamped to the null slot at the end from the first shader on"
        );
        victory_shader::emit::set_view_heap_limit(views - 1);
        if views == PORTABLE_VIEWS {
            set_capped_heap_map();
        }
    });
}

/// Sets the translator's heap map for a host view heap capped at
/// [`PORTABLE_VIEWS`], unless `VICTORY_HEAP_MAP=0`.
fn set_capped_heap_map() {
    if !heap_map_on() {
        return;
    }
    let window = window();
    victory_shader::emit::set_view_heap_map(Some((window, MAP_SLOT)));
    tracing::info!(
        window,
        map_slot = MAP_SLOT,
        "heap map: bindless views past the identity window are looked up (VICTORY_HEAP_MAP=0: off)"
    );
}

/// The offline warm-up (`victory translate --warm`, no device): translated
/// with the clamp of a capped host (`VICTORY_VIEW_LIMIT=999999`, which the
/// run's log gives), shaders get the heap map a run on that host has too,
/// so that the run finds them under its own cache key. Does nothing for
/// any other limit, or once a map is set.
pub(crate) fn offline_heap_map() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if victory_shader::emit::view_heap_limit() == PORTABLE_VIEWS - 1
            && victory_shader::emit::view_heap_map().is_none()
        {
            set_capped_heap_map();
        }
    });
}

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out in the world ("no free dynamic descriptor slots", run 11;
/// upstream PR #336 takes 2^18). On a host heap capped at 1,000,000 views
/// (NVIDIA) these come out of the title's slots, whose highest seen is
/// about 720,000: 2^17 left 868,000 of them, and ran out in the opening
/// cutscene (a frame asks for up to 16,000 tables while the frame before
/// still holds its own; a draw refused its table read slots never made
/// for it, and the card stopped within frames). 2^18 leaves 737,000; a
/// table past them is made afresh in these slots like any other.
pub(crate) const DYNAMIC: u32 = 1 << 18;

// The heap map leaves the fork's dynamic slots and guard slot in place.
const _: () = assert!(MAP_SLOT + 1 + DYNAMIC == PORTABLE_VIEWS - 1);

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        let extra = if sampler { 0 } else { DYNAMIC };
        let title_count = count;
        // A view heap ends at the null slot shaders clamp view indices to
        // (`probe_view_limit`), whatever the title's own count: the title's
        // slots, then the dynamic ones, then null up to and with that slot.
        // Also whether the translator looks bindless views up in a heap map,
        // as it does from the first translation on.
        if !sampler {
            probe_view_limit();
        }
        let null = if sampler {
            0
        } else {
            victory_shader::emit::view_heap_limit()
        };
        let layout = (!sampler)
            .then(heap_map)
            .flatten()
            .filter(|&(window, _)| count > window)
            .map(|(window, slot)| MapLayout::new(count, window, slot));
        let (heap, count, size) = if let Some(l) = layout {
            // The map is set on a capped host only, whose null slot is the
            // layout's guard (`MAP_SLOT + 1 + DYNAMIC`).
            #[cfg(not(test))]
            debug_assert_eq!(l.guard(), null, "the heap map's guard is the clamp's null slot");
            tracing::warn!(
                count,
                window = l.window,
                pool = l.pool,
                map_slot = l.slot,
                "host view heap with a heap map: title slots past the window that hold views get pool slots"
            );
            match make(l.size()) {
                Ok(heap) => (heap, l.window, l.size()),
                Err(e) => {
                    tracing::warn!(error = %e, count, "no host view heap with a heap map");
                    return None;
                }
            }
        } else if null == 0 {
            match make(count + extra) {
                Ok(heap) => (heap, count, count + extra),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else {
            let size = null + 1;
            let fitted = if count + extra <= null {
                count
            } else {
                // shortcut: descriptors past the host's limit are left out; a title that reaches
                // them needs remapping. They read the null slot instead.
                tracing::warn!(count, "host view heap capped at {size}; views past it read a null slot");
                null - extra
            };
            match make(size) {
                Ok(heap) => (heap, fitted, size),
                Err(e) => {
                    tracing::warn!(error = %e, count, size, "no host view heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(size) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        // The heap map: an entry per title slot past the window, zeroed (no
        // slot: the guard).
        let map = layout.and_then(|l| {
            let bytes = u64::from((title_count - l.window).max(1)) * 4;
            let Some((buffer, at)) = host.upload_buffer(bytes) else {
                tracing::warn!(bytes, "no heap map: title slots past the window read the null slot");
                return None;
            };
            // SAFETY: the buffer's whole mapping.
            unsafe { std::ptr::write_bytes(at, 0, bytes as usize) };
            crate::host::name(&buffer, || "heap map".to_owned());
            Some(HeapMap { buffer, at: at as usize })
        });
        let strides = (!sampler)
            .then(|| {
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_COUNT) * 4;
                let (buffer, data) = host.upload_buffer(bytes)?;
                // SAFETY: the buffer's whole mapping; a live buffer.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0, bytes as usize);
                    buffer.GetGPUVirtualAddress()
                };
                crate::host::name(&buffer, || "element strides by view heap slot".to_owned());
                Some((buffer, gpu, data as usize))
            })
            .flatten();
        // Title-keyed state covers the title's slots past the window too.
        let (pool, dynamic_first, titled) = match layout {
            Some(l) => (l.pool, l.dynamic_first(), title_count),
            None => (0, count, count),
        };
        let h = HostHeap {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            dynamic_first,
            title_count,
            sampler,
            made: std::sync::RwLock::new(Made {
                keys: (0..count + pool).map(|_| AtomicU64::new(0)).collect(),
                map: map
                    .is_some()
                    .then(|| SlotMap::new(count, title_count, count, pool)),
                cache: Mutex::new(CacheSlot::default()),
            }),
            map,
            copied: Mutex::new(Vec::new()),
            title_start: AtomicU64::new(0),
            reclaimed: AtomicU64::new(u64::MAX),
            kept: super::kept::Holds::new(dynamic_first + extra, super::kept::on().then_some(dynamic_first)),
            next: Mutex::new((0, vec![false; DYNAMIC as usize])),
            free: std::sync::atomic::AtomicU32::new(DYNAMIC),
            too_long: std::sync::atomic::AtomicU32::new(u32::MAX),
            strides,
            buffers: Mutex::default(),
            changed: Mutex::default(),
            textures: Mutex::default(),
            tabled: (0..if sampler { 0 } else { titled })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            buffer_marks: (0..if sampler || !left::set() { 0 } else { titled })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            bound_live: std::sync::atomic::AtomicBool::new(left::live()),
            left: left::HeapLeft::new(titled, sampler),
            placed: {
                let spans = std::sync::Arc::<Mutex<Spans>>::default();
                if !sampler {
                    PLACED_SPANS
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(std::sync::Arc::downgrade(&spans));
                }
                spans
            },
            noted: (0..if sampler || !skip_same() { 0 } else { titled })
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect(),
            spanned: (0..if sampler || !(SPANS_QUICK.set() || cfg!(test)) { 0 } else { (titled as usize).div_ceil(64) })
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            // (A slot never noted is in none of the sets: bare.)
            kinds: (0..if sampler { 0 } else { titled })
                .map(|_| std::sync::atomic::AtomicU8::new(note_kind::BARE))
                .collect(),
            // (The table starts zeroed.)
            strides_held: (0..if sampler || !skip_same() {
                0
            } else {
                titled.min(victory_shader::emit::SLOT_STRIDES_COUNT)
            })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            deferred: Default::default(),
            pieces: par_mirror::Pieces::new(),
            turns: Turns::default(),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, size) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        if let Some(l) = layout {
            // The map's raw view (`raw_at` holds the buffer), or one that
            // reads zeros: every lookup then gives the guard slot.
            match &h.map {
                Some(m) => h.raw_at(l.slot, &m.buffer, 0, u64::from(title_count - l.window) * 4),
                None => h.null_raw_at(l.slot),
            }
        }
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicAsked, 1);
        let mut guard = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Next, &self.next);
        let (next, taken) = &mut *guard;
        let enough = self.free.load(std::sync::atomic::Ordering::Relaxed) >= n
            && n < self.too_long.load(std::sync::atomic::Ordering::Relaxed);
        let n = n as usize;
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let starts = (cursor..end).chain(0..second_end);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        // Not capped at `DYNAMIC`: that would stop the scan at the cursor
        // again and miss the runs crossing it.
        for at in starts.take(if enough { usize::MAX } else { 0 }) {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            if enough {
                self.too_long.fetch_min(n as u32, std::sync::atomic::Ordering::Relaxed);
            }
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::DynamicRefused, 1);
                add(C::DynamicNs, started.elapsed().as_nanos() as u64);
            }
            return None;
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicNs, started.elapsed().as_nanos() as u64);
        taken[first..first + n].fill(true);
        self.free.fetch_sub(n as u32, std::sync::atomic::Ordering::Relaxed);
        *next = ((first + n) % DYNAMIC as usize) as u32;
        Some(self.dynamic_first + first as u32)
    }

    /// The title slots this heap mirrors at their own index (tables read in
    /// place reach only these: one that reaches past them is made afresh);
    /// fewer than the title's when the host caps its size, and with a heap
    /// map the identity window.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The host slot title slot `title` is made in (`made` locked): its own
    /// index below `count`; with a heap map past it, its pool slot, taken
    /// now if it has none (then `true`: [`HostHeap::publish`] writes its
    /// map entry once the slot is made). `None` past the title's slots, or
    /// with the pool used up even after [`HostHeap::reclaim`] (the title
    /// slot then reads the guard slot).
    fn host_slot(&self, made: &Made, title: u32, alone: bool) -> Option<(u32, bool)> {
        if title < self.count {
            return Some((title, false));
        }
        if title >= self.title_count {
            return None;
        }
        crate::draw::heap_pass::parts::count(crate::draw::heap_pass::parts::Count::Mapped);
        let map = made.map.as_ref()?;
        if let Some(host) = map.host(title) {
            return Some((host, false));
        }
        if let Some(host) = map.assign(title, crate::host::submissions().1) {
            return Some((host, true));
        }
        // VIcky (`VICTORY_HEAP_PASS_THREADS`): a thread of the pass over
        // buffer views gives back no other title slot's pool slot (another
        // thread may be making that slot); the slot is made once the
        // pass's threads are done ([`HostHeap::make_room`]).
        if !alone {
            ROOM_WANTED.with_borrow_mut(|wanted| wanted.push(title));
            return None;
        }
        if self.reclaim(made) > 0
            && let Some(host) = map.assign(title, crate::host::submissions().1)
        {
            return Some((host, true));
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(title, "heap map pool used up: title slots past the window read the null slot");
        });
        None
    }

    /// Writes title slot `title`'s heap map entry (`made` locked), after
    /// its pool slot was made.
    fn publish(&self, made: &Made, title: u32) {
        let (Some(map), Some(m)) = (&made.map, &self.map) else {
            return;
        };
        let Some(i) = title.checked_sub(map.window()).filter(|&i| i < map.len()) else {
            return;
        };
        // The descriptor first, then the entry naming it.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // SAFETY: entry `i` of the mapped heap map.
        unsafe { (m.at as *mut u32).add(i as usize).write_volatile(map.entry(title)) };
    }

    /// Gives back title slot `title`'s pool slot (`made` locked), if it has
    /// one: its map entry is zeroed (the guard slot), and the slot is
    /// handed out again once the submissions that may have read the old
    /// entry are done. Its view stays, held, until it is made again.
    fn unmap(&self, made: &Made, title: u32) {
        let (Some(map), Some(m)) = (made.map.as_ref(), &self.map) else {
            return;
        };
        let Some(host) = map.host(title).filter(|_| title >= map.window()) else {
            return;
        };
        // (The key before the slot is given back: once it is, a thread
        // making another title slot may take it, `VICTORY_HEAP_PASS_THREADS`.)
        made.set_key(host, 0);
        let i = title - map.window();
        // SAFETY: entry `i` of the mapped heap map (`title` is mapped).
        unsafe { (m.at as *mut u32).add(i as usize).write_volatile(0) };
        // The entry is out before the submissions running are counted.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        map.free(title, crate::host::submissions().0);
    }

    /// Title slot `title` holds no view (`made` locked): its host slot reads
    /// nothing and leaves the sets of the slots shaders pick by index (as
    /// the fork did), and a pool slot is given back. `for_draw`: the slot
    /// leaves the sets even if its host slot was never made. `marks` as for
    /// [`HostHeap::note_buffer_marking`].
    fn empty_slot(&self, made: &Made, title: u32, for_draw: bool, marks: bool) {
        use crate::draw::heap_pass::parts::{Part, enter};
        match made.host(title) {
            Some(host) if made.key(host) != 0 => {
                made.set_key(host, 0);
                {
                    let _create = enter(Part::Create);
                    self.make_null(host);
                }
                let _note = enter(Part::Note);
                let _ = self.note_buffer_marking(title, None, false, marks);
            }
            // Left for later and stale since: out of the sets and the stride table.
            _ if for_draw => {
                let _note = enter(Part::Note);
                let _ = self.note_buffer_marking(title, None, false, marks);
            }
            _ => {}
        }
        let _map = enter(Part::Map);
        self.unmap(made, title);
    }

    /// With the heap map's pool used up (`made` locked): gives back the pool
    /// slots of title slots that hold no live view any more (a view of a
    /// released resource, or no view), which nothing else finds: the fork
    /// learns of a slot's view only from the title's descriptor writes and
    /// copies, and a title that releases a resource need not write its
    /// slots again. At most once every two seconds; the slots given back
    /// are handed out once the submissions running now are done. Returns
    /// how many.
    fn reclaim(&self, made: &Made) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        const EVERY_MS: u64 = 2000;
        let now = crate::host::now_ms();
        let last = self.reclaimed.load(Relaxed);
        if last != u64::MAX && now < last.saturating_add(EVERY_MS) {
            return 0;
        }
        self.reclaimed.store(now, Relaxed);
        let start = self.title_start.load(Relaxed);
        let Some(map) = made.map.as_ref().filter(|_| start != 0) else {
            return 0;
        };
        let started = std::time::Instant::now();
        let dead: Vec<u32> = map
            .mapped()
            .map(|(title, _)| title)
            .filter(|&title| {
                let handle = start + u64::from(title) * TITLE_SLOT_BYTES;
                // SAFETY: `title < title_count`, a slot of the live title heap.
                let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
                tag != VIEW_TAG || view_at(i, generation).is_none()
            })
            .collect();
        for &title in &dead {
            self.empty_slot(made, title, true, true);
        }
        // The first passes at warn level, the rest at debug.
        static PASSES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let (given_back, ms) = (dead.len(), started.elapsed().as_millis());
        if PASSES.fetch_add(1, Relaxed) < 16 {
            tracing::warn!(given_back, ms, "heap map pool used up: pool slots of title slots without a live view given back");
        } else {
            tracing::debug!(given_back, ms, "heap map pool used up: pool slots of title slots without a live view given back");
        }
        dead.len()
    }

    /// The GPU address of the table of element strides by title slot (view
    /// heaps).
    pub(crate) fn strides_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|s| s.1)
    }

    /// The table's buffer, for a list to hold.
    pub(crate) fn strides_buffer(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|s| &s.0)
    }

    /// Up to `max` title slots whose texture view is due at `frame`, each
    /// with its write number, in the order to serve them
    /// ([`TextureQueue::due`]), and the queue's counts
    /// ([`TextureQueue::census`]).
    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> (Vec<(u32, u64)>, [u64; 5]) {
        let textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        (if max == 0 { Vec::new() } else { textures.due(frame, max) }, textures.census(frame))
    }

    /// Every title slot of the queue that is due at `frame`, and whether
    /// it was uploaded before (in part, or whole and waiting for another
    /// look): for the card accounting line
    /// (`crate::resource::card_accounting`).
    pub(crate) fn texture_slots_all_due(&self, frame: u64) -> Vec<(u32, bool)> {
        let textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        textures
            .slots
            .iter()
            .filter(|(_, w)| w.due <= frame)
            .map(|(&index, w)| (index, w.mapped > 0 || w.checked.2 != 0))
            .collect()
    }

    /// The texture view of title slot `index` (as handed out with `write`)
    /// was brought up to date: due again from frame `again` (uploaded with
    /// that many bytes of its memory there, not all), or not until the
    /// slot is written again.
    pub(crate) fn texture_slot_done(&self, index: u32, write: u64, again: Option<(u64, u64)>) {
        let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        match again {
            Some((frame, mapped)) => textures.partly(index, write, frame, mapped),
            None => textures.done(index, write),
        }
    }

    /// Title slot `index` was uploaded whole: see [`TextureQueue::recheck`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slot_recheck(
        &self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        mips: Option<Mips>,
        print: u64,
        looks: u8,
        uploaded: u64,
    ) {
        let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        textures.recheck(index, write, due, at, mips, print, looks, uploaded);
    }

    /// Title slot `index` was uploaded in part: see
    /// [`TextureQueue::recheck_partly`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slot_recheck_partly(
        &self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        mips: Option<Mips>,
        print: u64,
        uploaded: u64,
        mapped: u64,
    ) {
        let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        textures.recheck_partly(index, write, due, at, mips, print, uploaded, mapped);
    }

    /// See [`TextureQueue::checked`].
    pub(crate) fn texture_slot_checked(&self, index: u32) -> Option<((u64, u64, u64, u8, u64), Option<Mips>, u64)> {
        crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures).checked(index)
    }

    /// See [`TextureQueue::ds_hold`].
    pub(crate) fn texture_slot_ds_hold(&self, index: u32, write: u64, due: u64, most: u8) -> bool {
        crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures).ds_hold(index, write, due, most)
    }

    /// VIcky (`VICTORY_STALE_SLOTS`): whether title slot `index` is in the
    /// queue (due, or waiting for a later look): something will look at it
    /// again.
    pub(crate) fn texture_slot_queued(&self, index: u32) -> bool {
        crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures).slots.contains_key(&index)
    }

    /// Nothing to upload at title slot `index` now (its memory is not
    /// mapped, or no more of it than before): looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, frame: u64, most: u64) {
        let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        textures.later(index, write, frame, most);
    }

    /// See [`TextureQueue::hold`].
    pub(crate) fn texture_slot_held(&self, index: u32, write: u64, due: u64) {
        crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures).hold(index, write, due);
    }

    /// See [`TextureQueue::room`].
    pub(crate) fn texture_slots_room(&self, frame: u64) -> usize {
        crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures).room(frame)
    }

    /// Bytes of title slot `index`'s memory that were mapped when it was
    /// last uploaded in part (0: not uploaded, or whole).
    pub(crate) fn texture_slot_mapped(&self, index: u32) -> u64 {
        crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures).mapped(index)
    }

    /// Takes the ranges of title memory the runtime mapped, unmapped or
    /// wrote since the last call ([`remapped`]) and makes the slots whose
    /// placed texture lies there due from the next frame (by when a write
    /// under way has landed): memory that arrived is uploaded, memory that
    /// went makes the slot null ([`HostHeap::texture_slot_gone`]), and a
    /// texture streamed into the place of another of the same description
    /// is uploaded although no descriptor says so. Slots tables read are
    /// left to their draws (VIcky: not with `VICTORY_TABLE_BOUND_FRESH`,
    /// [`HostHeap::leave_tabled`]). Returns how many slots it made due.
    /// shortcut: the ranges are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    pub(crate) fn texture_slots_remapped(&self, frame: u64) -> usize {
        self.texture_slots_page_written(frame);
        let ranges = std::mem::take(&mut *REMAPPED.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        self.slots_remapped_in(&ranges, frame)
    }

    /// [`HostHeap::texture_slots_remapped`] for `ranges` (start, end).
    fn slots_remapped_in(&self, ranges: &[(u64, u64)], frame: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let mut slots: Vec<u32> = {
            let placed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*self.placed);
            ranges.iter().flat_map(|&(start, end)| placed.slots_in(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        self.leave_tabled(&mut slots, frame, crate::texstale::Fresh::Remapped);
        let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
        }
        STALE[3].fetch_add(slots.len() as u64, Relaxed);
        slots.len()
    }

    /// The due-again passes' rule for slots tables read lately
    /// ([`HostHeap::tabled_lately`]): left to their draws (taken out of
    /// `slots`), as stage 1 first had it; VIcky, with
    /// `VICTORY_TABLE_BOUND_FRESH` ([`left::fresh`]), kept and made due as
    /// any slot (the pass makes such a slot from fresh bytes, or finds its
    /// texture staged by the draws and leaves it to them), and counted as
    /// `why` ([`crate::texstale::Fresh`]).
    fn leave_tabled(&self, slots: &mut Vec<u32>, frame: u64, why: crate::texstale::Fresh) {
        if left::fresh() {
            let tabled = slots.iter().filter(|&&slot| self.tabled_lately(slot, frame)).count();
            crate::texstale::fresh_count(why, tabled as u64);
            return;
        }
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
    }

    /// Takes the ranges of title memory written behind the guards' back
    /// since the last call ([`page_written`]) and finds the slots whose
    /// placed texture holds bytes of them ([`Spans::slots_holding`]). A
    /// slot whose texture was uploaded before the write shows the bytes
    /// from before: with `VICTORY_PAGE_WRITES` (`crate::watch::page_writes`,
    /// bit 2) it is due again [`PAGE_WRITE_AFTER`] frames on, as memory
    /// mapped under it would make it ([`HostHeap::texture_slots_remapped`]);
    /// its host copy exists, so the upload needs no room on the card. A
    /// slot still waiting for its first upload is left as it is: that
    /// upload takes the bytes that are there by then. Slots tables read
    /// are left to their draws (VIcky: not with `VICTORY_TABLE_BOUND_FRESH`,
    /// [`HostHeap::leave_tabled`], which also has the textures the draws
    /// keep staged there read again, `crate::texture::restage_in`). Counted
    /// either way; returns how many slots it made due.
    fn texture_slots_page_written(&self, frame: u64) -> usize {
        let ranges = std::mem::take(&mut *PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        // VIcky (`VICTORY_TABLE_BOUND_FRESH`, [`left::fresh_live`]): the
        // textures draws keep staged under these writes are read again
        // whole at the next submission that samples them, and uploaded by
        // its list: the check at submission finds such a write only through
        // the guards it opens, a partly mapped texture is read at most every
        // 30 frames, and a slot the draws read is in no queue.
        if left::fresh_live() {
            let restaged = crate::texture::restage_in(&ranges);
            crate::texstale::fresh_count(crate::texstale::Fresh::Restaged, restaged as u64);
        }
        self.slots_written_under(&ranges, frame, crate::watch::page_writes() & 2 != 0)[2]
    }

    /// [`HostHeap::texture_slots_page_written`] for `ranges` (start, end):
    /// the slots that held bytes of them, those of them uploaded before,
    /// and those made due (the latter, if `due`).
    fn slots_written_under(&self, ranges: &[(u64, u64)], frame: u64, due: bool) -> [usize; 3] {
        use std::sync::atomic::Ordering::Relaxed;
        let mut slots: Vec<u32> = {
            let placed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*self.placed);
            ranges.iter().flat_map(|&(start, end)| placed.slots_holding(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        self.leave_tabled(&mut slots, frame, crate::texstale::Fresh::PageWritten);
        let held = slots.len();
        let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        // Uploaded before: out of the queue, or in it after an upload (in
        // part, or whole and being looked at again).
        slots.retain(|slot| textures.slots.get(slot).is_none_or(|w| w.mapped > 0 || w.checked.2 != 0));
        PAGE_WRITES[2].fetch_add(held as u64, Relaxed);
        PAGE_WRITES[3].fetch_add(slots.len() as u64, Relaxed);
        // VIcky (`VICTORY_STALE_SLOTS`): their uploads had a write after them.
        crate::texstale::written_under(&slots);
        if !due {
            return [held, slots.len(), 0];
        }
        for &slot in &slots {
            textures.changed(slot, frame, frame + PAGE_WRITE_AFTER);
        }
        [held, slots.len(), slots.len()]
    }

    /// Whether host slot `index` is made from the very record the title
    /// has at `handle` now. Its host texture then exists (a slot keeps what
    /// it views), so bringing that texture up to date takes no memory of
    /// the card.
    pub(crate) fn texture_slot_made_from(&self, index: u32, handle: u64) -> bool {
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        if tag != VIEW_TAG {
            return false;
        }
        // And the slot holds a host resource: a slot is also noted as made
        // from a view the host could not make (`make` gave nothing, the
        // slot is null and keeps nothing). The host copy of such a
        // texture is held by no slot, may have been let go, and would be
        // made anew here, with the card full. `made` first, then `kept`,
        // as `mirror` takes them.
        // (With a heap map, title slot `index` past the window is made in
        // its pool slot, if it holds one.)
        let made = self.made_alone();
        let Some(host) = made.host(index) else {
            return false;
        };
        made.key_of(host).is_some_and(|made| made != 0 && made == key)
            && self.kept.with(host, |slot| slot.is_some_and(Option::is_some))
    }

    /// Levels on demand (`crate::tiles`): makes again the host slots of the
    /// placed textures whose first level with memory changed since the
    /// last call ([`levels_changed`]), each from the view it was made
    /// from, so that it shows the texture from that level ([`make`]). Only
    /// slots that are made; a slot draws read through a table is made by
    /// its draw. Returns how many it made.
    /// shortcut: the starts are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    /// Levels on demand (`crate::tiles`): the card has room, so the slots
    /// of up to [`Self::WAITING_A_PASS`] placed textures whose upload left
    /// a held level without tiles at least a second ago
    /// ([`levels_waiting`]) are due again from the next frame, as memory
    /// mapped under them would make them
    /// ([`HostHeap::texture_slots_remapped`]): the upload then gives the
    /// level its tiles. To be called only while the card has room: without
    /// it the upload would give nothing again. Slots tables read are left
    /// to their draws (`texture.rs` `upload` asks again for those; VIcky:
    /// not with `VICTORY_TABLE_BOUND_FRESH`, [`HostHeap::leave_tabled`]). Returns
    /// how many slots it made due.
    /// shortcut: kept in one place for all heaps, as the starts below.
    pub(crate) fn texture_slots_waiting(&self, frame: u64) -> usize {
        if self.sampler {
            return 0;
        }
        let starts: Vec<u64> = {
            let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
            if waiting.is_empty() {
                return 0;
            }
            let starts: Vec<u64> = waiting
                .iter()
                .filter(|&(_, &since)| crate::host::long_ago(since, 1000, 2))
                .map(|(&start, _)| start)
                .take(Self::WAITING_A_PASS)
                .collect();
            for start in &starts {
                waiting.remove(start);
            }
            starts
        };
        if starts.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*self.placed);
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        self.leave_tabled(&mut slots, frame, crate::texstale::Fresh::Waiting);
        let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
        }
        crate::tiles::STATS[10].fetch_add(slots.len() as u64, std::sync::atomic::Ordering::Relaxed);
        slots.len()
    }

    /// Textures a pass asks again for their levels
    /// ([`HostHeap::texture_slots_waiting`]): the first ones given tiles
    /// may take the room the others were asked for, and an upload that
    /// gives nothing is a whole texture staged and copied for nothing.
    const WAITING_A_PASS: usize = 128;

    pub(crate) fn texture_views_remade(&self) -> usize {
        if self.sampler {
            return 0;
        }
        let starts = std::mem::take(&mut *RESHOWN.lock().unwrap_or_else(|e| e.into_inner()));
        if starts.is_empty() {
            return 0;
        }
        let slots: Vec<u32> = {
            let placed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*self.placed);
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        let made = self.made_alone();
        let mut remade = 0;
        for slot in slots {
            // Title slot `slot`'s host slot (with a heap map, past the
            // window its pool slot, if it holds one).
            let Some(host) = made.host(slot) else {
                continue;
            };
            let Some(key) = made.key_of(host) else {
                continue;
            };
            if key == 0 {
                continue;
            }
            let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
                continue;
            };
            if let Some(kept) = make(&view, self.cpu(host)) {
                self.keep(host, kept);
                remade += 1;
            }
        }
        remade
    }

    /// The memory of the texture title slot `index` views is not mapped
    /// (any more): a host slot made from it becomes null again, so that
    /// the host copy can be let go (`crate::resource::let_go_idle`; a slot
    /// keeps what it views). True if the slot was made.
    /// (With a heap map, a title slot past the window keeps its pool slot:
    /// it still holds the view, made again once its memory is mapped.)
    pub(crate) fn texture_slot_gone(&self, index: u32) -> bool {
        let made = self.made_alone();
        if made
            .host(index)
            .and_then(|host| made.key_of(host))
            .is_none_or(|key| key == 0)
        {
            return false;
        }
        self.unmake(&made, index);
        STALE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Makes title slot `index`'s host slot null again, as it is before its
    /// view is made: a null view of the kind it held.
    fn unmake(&self, made: &Made, index: u32) {
        // VIcky (`VICTORY_PARALLEL_MIRROR`): on a thread that makes slots
        // with others, a slot of another piece than the one being made is
        // unmade once that piece is let go of.
        if self.par_stale(index) {
            return;
        }
        let Some(host) = made.host(index) else {
            return;
        };
        let Some(key) = made.key_of(host) else {
            return;
        };
        made.set_key(host, 0);
        if key == 0 {
            return;
        }
        match view_at(key as u32, (key >> 32) as u32) {
            Some(view) => null_for(&view, self.cpu(host)),
            None => null_view(self.cpu(host)),
        }
        self.keep(host, None);
    }

    /// Title slot `index` is a table's ([`HostHeap::tabled`]). `read`: a
    /// draw read it through the table and brought what it views up to
    /// date, so the slot leaves the queue of textures picked by index; the
    /// caller makes the host slot in place ([`HostHeap::mirror`]), as the
    /// pass would have, for a shader that picks it by index all the same.
    /// Returns whether this was the slot's first read in this frame.
    pub(crate) fn texture_slot_tabled(&self, index: u32, read: bool) -> bool {
        self.tabled_in(index, read, crate::frame::current())
    }

    /// [`HostHeap::texture_slot_tabled`] in frame `frame`.
    fn tabled_in(&self, index: u32, read: bool, frame: u64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(stamp) = self.tabled.get(index as usize) else {
            return false;
        };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        let old = stamp.load(Relaxed);
        if !read {
            // VIcky (`VICTORY_TABLE_BOUND_FRESH`, [`left::fresh`]): a slot a
            // table only spans is made by no draw (none reads it through the
            // table, none uploads its texture): no mark, so it keeps the
            // rules of a slot picked by index, as without the switch.
            if left::fresh() {
                return false;
            }
            if old & !TABLED_READ != now {
                stamp.store(now, Relaxed);
            }
            return false;
        }
        // Once per slot and frame: tables are read by every draw.
        if old != now | TABLED_READ {
            stamp.store(now | TABLED_READ, Relaxed);
            // (VIcky, `VICTORY_TABLE_BOUND=alt`: marked in its off phase
            // too, but left in the queue, [`left::live`].)
            if tabled_out() && left::live() {
                let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
                if textures.slots.remove(&index).is_some() {
                    TABLED_WRITES[1].fetch_add(1, Relaxed);
                }
            }
            return true;
        }
        false
    }

    /// Whether title slot `index` was a table's within the last
    /// [`TABLED_FRAMES`] frames before `frame` (never, if such slots are
    /// to be queued as any other; VIcky: nor in the off phase of
    /// `VICTORY_TABLE_BOUND=alt`, [`left::live`]).
    fn tabled_lately(&self, index: u32, frame: u64) -> bool {
        tabled_out() && left::live() && self.tabled_marked(index, frame)
    }

    /// Whether title slot `index` holds a mark of a table set within the
    /// last [`TABLED_FRAMES`] frames before `frame`, whether marks act now
    /// or not.
    fn tabled_marked(&self, index: u32, frame: u64) -> bool {
        let stamp = self.tabled_stamp(index) & !TABLED_READ;
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        stamp != 0 && u64::from(now.wrapping_sub(stamp)) <= TABLED_FRAMES
    }

    /// What `tabled` holds for title slot `index`; 0: never a table's.
    fn tabled_stamp(&self, index: u32) -> u32 {
        self.tabled
            .get(index as usize)
            .map_or(0, |s| s.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The queue's part of a write of title slot `index` in `frame`:
    /// `texture`, whether it now holds a view of a texture the title may
    /// fill; `seen` as for [`HostHeap::note_buffer`].
    fn queue_texture(&self, index: u32, texture: bool, seen: bool, frame: u64) {
        // (VIcky, `VICTORY_TABLE_BOUND=alt`: in its off phase as if no
        // table marked the slot, [`left::live`].)
        let stamp = if tabled_out() && left::live() { self.tabled_stamp(index) } else { 0 };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        // A table's slot lately: the draw that reads it uploads its
        // texture, so its write waits for that draw.
        let lately = stamp != 0 && u64::from(now.wrapping_sub(stamp & !TABLED_READ)) <= TABLED_FRAMES;
        // Made now by a draw of this frame that read it through a table:
        // up to date.
        let read_now = seen && stamp == now | TABLED_READ;
        let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        if texture && !read_now {
            let due = if lately && !seen {
                TABLED_WRITES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                frame + TABLED_GRACE
            } else {
                0
            };
            textures.written(index, frame, due, seen);
        } else {
            textures.remove(index);
        }
    }

    /// VIcky (`VICTORY_TABLE_BOUND`, [`left`]): title slot `index` lies in a
    /// table a draw binds, whatever it holds: its buffer mark
    /// ([`HostHeap::buffer_marks`]), only counted. Nothing without the
    /// switch.
    pub(crate) fn slot_tabled(&self, index: u32) {
        self.slot_tabled_in(index, crate::frame::current());
    }

    /// [`HostHeap::slot_tabled`] in `frame`.
    pub(crate) fn slot_tabled_in(&self, index: u32, frame: u64) {
        if let Some(stamp) = self.buffer_marks.get(index as usize) {
            left::mark(stamp, frame);
        }
    }

    /// VIcky ([`left`]): whether a draw's table claimed title slot `index`
    /// within the last [`left::FRAMES`] frames before `frame` (its buffer
    /// mark; stage 2 would leave it to the draws). Never without the switch.
    pub(crate) fn buffer_marked_lately(&self, index: u32, frame: u64) -> bool {
        self.buffer_marks
            .get(index as usize)
            .is_some_and(|stamp| left::lately(stamp.load(std::sync::atomic::Ordering::Relaxed), frame))
    }

    /// VIcky ([`left`]'s line): the pass over buffer views of `frame` was
    /// given `slots`; counted, with those that carry a buffer mark.
    pub(crate) fn count_pass_marks(&self, slots: &[u32], frame: u64) {
        left::count(left::C::PassSlots, slots.len() as u64);
        if self.buffer_marks.is_empty() {
            return;
        }
        let marked = slots.iter().filter(|&&index| self.buffer_marked_lately(index, frame)).count();
        left::count(left::C::PassMarked, marked as u64);
    }

    /// VIcky ([`left`]'s line): a slot the heap sync found new, as its
    /// making begins (`made` held): title slot `index`, its record `key`
    /// and view (`None`: no view of ours). Counted by the first rule of
    /// stages 2 and 3 that would take it: a buffer mark would leave it to
    /// the draws; stage 2's grace would hold it (a constant buffer or
    /// placed view, named whole in its record); no view, into a slot that
    /// shows none, or one; else the view made, by kind.
    fn count_sync_rule(&self, made: &Made, index: u32, key: u64, view: Option<&View>) {
        use left::C;
        let generation = (key >> 32) as u32;
        let rule = match view {
            Some(_) if self.buffer_marked_lately(index, crate::frame::current()) => C::WouldLeave,
            Some(_) if matches!(generation >> super::KIND_SHIFT, super::CONSTANT | super::PLACED) => C::WouldHold,
            Some(view) if is_texture_view(view) => C::TexturesMade,
            Some(view) if view.is_buffer() => C::BuffersMade,
            Some(_) => C::OthersMade,
            None if made.host(index).and_then(|host| made.key_of(host)).is_some_and(|shown| shown != 0) => C::NoViewShown,
            None => C::NoViewNone,
        };
        left::count_sync(rule);
    }

    /// VIcky (`VICTORY_TABLE_BOUND=alt`, [`left`]), at each heap sync before
    /// its pages (`super::walk_heap`; title slot `i` at `start + i * step`):
    /// at a flip of the phase to off, the slots with a live mark of a table
    /// are swept once ([`HostHeap::table_bound_sweep`]), so that the off
    /// phase starts from what it would have had. Nothing otherwise. Returns
    /// what the sweep did, if one ran.
    pub(crate) fn table_bound_flip(&self, (start, step): (u64, u64)) -> Option<[usize; 3]> {
        if self.sampler || self.tabled.is_empty() {
            return None;
        }
        // (Only the heap sync, one at a time, changes it: written only when
        // the phase changed.)
        let live = left::live();
        if self.bound_live.load(std::sync::atomic::Ordering::Relaxed) == live {
            return None;
        }
        self.bound_live.store(live, std::sync::atomic::Ordering::Relaxed);
        // VIcky (`VICTORY_STALE_SLOTS`): no slot is made by a draw in the
        // off phase; the ones watched as such go back to the pass below.
        if !live {
            crate::texstale::table_watch_clear();
        }
        (!live && tabled_out()).then(|| self.table_bound_sweep((start, step), crate::frame::current()))
    }

    /// [`HostHeap::table_bound_flip`]'s sweep in `frame`: every title slot
    /// with a mark of a table set within the last [`TABLED_FRAMES`] frames
    /// ([`HostHeap::tabled_marked`]). Its record is read through the
    /// kernel, a sync's page at a time (the slots of a page the title
    /// unmapped are skipped, as the sync skips them), and noted as seen
    /// ([`HostHeap::copied`]). A slot whose host slot is not made from it
    /// (the sync left it to the draws) is mirrored as the sync mirrors
    /// ([`HostHeap::mirror_due`]); one its draw made from it already, if it
    /// holds a texture view, is noted again as the sync notes a new record
    /// ([`HostHeap::note_buffer`]): its texture goes back in the queue of
    /// textures picked by index, as written now, and its memory in the
    /// spans of placed textures. Returns the slots swept, those mirrored,
    /// and the textures put back in the queue.
    fn table_bound_sweep(&self, (start, step): (u64, u64), frame: u64) -> [usize; 3] {
        let began = std::time::Instant::now();
        let mirrored = self.mirrored();
        let marked: Vec<u32> = (0..(self.tabled.len() as u32).min(mirrored))
            .filter(|&index| self.tabled_marked(index, frame))
            .collect();
        // (index, handle, the record's first words, its print)
        let mut found: Vec<(u32, u64, [u32; 3], u64)> = Vec::with_capacity(marked.len());
        let (mut bytes, mut unread) = (Vec::new(), 0);
        let page = crate::watch::PAGE as u64;
        let handle_of = |index: u32| start + u64::from(index) * step;
        let mut i = 0;
        while i < marked.len() {
            let first = handle_of(marked[i]);
            let j = i + marked[i..].iter().take_while(|&&index| handle_of(index) / page == first / page).count();
            bytes.resize((handle_of(marked[j - 1]) + step - first) as usize, 0);
            if crate::host::read_into(first as *const u8, &mut bytes) {
                for &index in &marked[i..j] {
                    let at = (handle_of(index) - first) as usize;
                    let record = &bytes[at..at + step as usize];
                    let word = |w: usize| u32::from_le_bytes([0, 1, 2, 3].map(|b| record.get(w * 4 + b).copied().unwrap_or(0)));
                    found.push((index, handle_of(index), [word(0), word(1), word(2)], record_print(record) | 1));
                }
            } else {
                unread += j - i;
            }
            i = j;
        }
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if copied.is_empty() {
                copied.resize(mirrored as usize, 0);
            }
            for &(index, _, _, print) in &found {
                if let Some(last) = copied.get_mut(index as usize) {
                    *last = print;
                }
            }
        }
        let (made_by_draws, due): (Vec<_>, Vec<_>) = {
            let made = self.made_alone();
            found.into_iter().partition(|&(index, _, [tag, i, generation], _)| {
                tag == VIEW_TAG
                    && made.host(index).and_then(|host| made.key_of(host)) == Some(u64::from(generation) << 32 | u64::from(i))
            })
        };
        let due: Vec<(u32, u64)> = due.into_iter().map(|(index, handle, _, _)| (index, handle)).collect();
        self.mirror_due(&due);
        let mut queued = 0;
        for chunk in made_by_draws.chunks(MIRROR_CHUNK) {
            let made = self.made_alone();
            for &(index, _, [_, i, generation], _) in chunk {
                let Some(view) = view_at(i, generation).filter(is_texture_view) else {
                    continue;
                };
                for stale in self.note_buffer(index, Some(&view), false) {
                    if stale != index {
                        self.unmake(&made, stale);
                    }
                }
                queued += 1;
            }
        }
        tracing::info!(
            "VICTORY_TABLE_BOUND=alt, at the flip to off: {} title slots with a live mark of a table swept in {:.1} ms: {} \
             mirrored as the heap sync would have mirrored them (it had left them to the draws), {} made by their draws \
             already ({queued} textures put back in the queue of textures picked by index), {unread} on pages that could \
             not be read",
            marked.len(),
            began.elapsed().as_secs_f64() * 1000.0,
            due.len(),
            made_by_draws.len(),
        );
        [marked.len(), due.len(), queued]
    }

    /// The title slots that hold buffer views.
    pub(crate) fn buffer_slots(&self) -> Vec<u32> {
        let buffers = crate::lot4::locks::mutex(crate::lot4::locks::Lock::HeapBuffers, &self.buffers);
        buffers.iter().copied().collect()
    }

    /// What `f` takes of the title slots that hold buffer views, in order,
    /// under the lock of that set (VIcky, `VICTORY_HEAP_PASS_SPREAD`: a
    /// sixteenth of them, `crate::draw::heap_pass::window`).
    pub(crate) fn with_buffer_slots<R>(&self, f: impl FnOnce(&std::collections::BTreeSet<u32>) -> R) -> R {
        f(&crate::lot4::locks::mutex(crate::lot4::locks::Lock::HeapBuffers, &self.buffers))
    }

    /// The title slots whose buffer view was written, or went, since the
    /// last call.
    pub(crate) fn buffer_changes(&self) -> Vec<u32> {
        let mut changed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Changed, &self.changed);
        std::mem::take(&mut *changed).into_iter().collect()
    }

    /// Records what title slot `index` now holds for shaders that pick
    /// buffers from the heap: its element stride in the table, and whether
    /// it is a buffer view at all.
    /// `seen`: the slot was not written now, a draw or a pass made its
    /// host slot from what it held.
    /// Returns the title slots whose placed texture view the write made
    /// stale ([`Spans`]): out of the queue of textures already, their host
    /// slots are the caller's to make null (it holds `made`).
    fn note_buffer(&self, index: u32, view: Option<&View>, seen: bool) -> Vec<u32> {
        self.note_buffer_marking(index, view, seen, true)
    }

    /// [`HostHeap::note_buffer`]; `marks`: whether the note of a buffer
    /// slot (or of one that leaves the buffer slots) marks it as written
    /// since the pass over buffer views last looked
    /// ([`HostHeap::buffer_changes`]). VIcky (`VICTORY_HEAP_PASS_ONE_LOOK=1`,
    /// `crate::draw::heap_pass::one_look`): not when that pass makes a slot
    /// it looks at. The pass keeps the range of the view it makes there and
    /// then; the mark only had its next pass look at the slot a second time
    /// and find it made. A write of the title after the making marks the
    /// slot as before (its note is the sync's, `marks` set), and so does a
    /// draw's making.
    fn note_buffer_marking(&self, index: u32, view: Option<&View>, seen: bool, marks: bool) -> Vec<u32> {
        self.note_buffer_into(index, view, seen, marks, None)
    }

    /// [`HostHeap::note_buffer_marking`]; with `sink` (VIcky,
    /// `VICTORY_HEAP_SYNC_LEAN`, [`sync_lean`]) its marks go there, for the
    /// end of the chunk the caller is at, as [`KeptLater`] takes them.
    fn note_buffer_into(
        &self,
        index: u32,
        view: Option<&View>,
        seen: bool,
        marks: bool,
        mut sink: Option<&mut Vec<u32>>,
    ) -> Vec<u32> {
        if self.sampler {
            return Vec::new();
        }
        // VIcky ([`note_kind`]): what the slot's last note left it as, and
        // what this one makes it.
        let (was, now) = (self.kind(index), note_kind::of(view));
        #[cfg(feature = "diagnostics")]
        note_kind::noted(was, now);
        // VIcky (`VICTORY_HEAP_BUFFERS_SKIP_SAME=1`): the slot's last note
        // was of a plain buffer view and this one is too. That note took
        // the slot out of the queue of textures and out of the spans of
        // placed textures, and put it in the set of buffer slots; only a
        // note of another kind of view puts it back in the first two or
        // takes it out of the third. What is left to do is what differs:
        // the stride, and the mark that the slot was written since the
        // pass over buffer views last looked.
        let plain = !self.noted.is_empty() && now == note_kind::PLAIN;
        if plain && self.noted_plain(index) {
            self.note_stride(index, view.and_then(view_stride));
            self.mark_into(index, marks, sink);
            FAST.add(Fast::NotesSkipped as usize, 1);
            if check_now() {
                self.check_noted(index);
            }
            return Vec::new();
        }
        // VIcky (`VICTORY_HEAP_NOTE_KIND`): from and to a plain buffer
        // view's or a bare slot, what differs alone.
        if self.note_by_kind(index, view, (was, now), marks, sink.as_deref_mut()) {
            return Vec::new();
        }
        // Measured for the texture side of the same gap (textures a shader
        // picks by a computed index are neither made, if placed, nor
        // uploaded): slots written with a placed texture view, and with a
        // view of a texture object.
        let texture = match view {
            Some(View::Placed(false, resource, _)) if resource[0] != 1 => {
                PLACED_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            Some(View::Shader(r, _)) if crate::resource::buffer_bytes(*r as crate::object::This).is_none() => {
                OBJECT_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            _ => false,
        };
        self.queue_texture(index, texture, seen, crate::frame::current());
        // The memory a placed texture view describes. Not for a slot
        // tables read (thousands of writes a frame, each read by its
        // draw): no pass looks at it, and the same texture's view in a
        // slot shaders pick by index says what it says.
        // (VIcky, `VICTORY_TABLE_BOUND_FRESH`, [`left::fresh`]: kept for
        // them too, so the passes over memory mapped, written or waiting
        // find them: the draws' slots past the identity window are those
        // a shader may also pick by index, and the notes of a slot the sync
        // leaves to the draws are a draw's making in place, a few thousand
        // a frame, which take the spans' lock either way unless
        // `VICTORY_SPANS_QUICK` spares it.)
        let placed_view = match view {
            Some(View::Placed(false, resource, desc))
                if resource[0] != 1 && (left::fresh() || !self.tabled_lately(index, crate::frame::current())) =>
            {
                // (And the level the view starts at: word 4 of a texture
                // view's description, after format, dimension, mapping
                // and a pad.)
                Some((resource, u64::from(desc[10]) | u64::from(desc[11]) << 32, desc[4]))
            }
            _ => None,
        };
        // VIcky (`VICTORY_SPANS_QUICK`, port of Belobug a284ca74): a write of
        // no placed texture view to a slot that is not in the spans leaves
        // their lock alone (forgetting a slot that is not there does
        // nothing). Whether the slot's bit is set changes only here, under
        // the slot's own lock (`made`, or its piece with
        // `VICTORY_PARALLEL_MIRROR`), as every write of the slot.
        let quick = !self.spanned.is_empty() && placed_view.is_none() && !self.in_spans(index) && SPANS_QUICK.on();
        let stale = if quick {
            SPANS_QUICK_COUNTS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Vec::new()
        } else {
            use std::sync::atomic::Ordering::Relaxed;
            if !self.spanned.is_empty() {
                SPANS_QUICK_COUNTS[1].fetch_add(1, Relaxed);
            }
            let mut placed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*self.placed);
            let span = placed_view.and_then(|(resource, start, first)| {
                Some((start, start.saturating_add(placed.bytes(resource)?), first))
            });
            match span {
                Some((start, end, first)) => {
                    SPANNED[0].fetch_min(start, Relaxed);
                    SPANNED[1].fetch_max(end, Relaxed);
                    placed.rule = stale_rule();
                    self.set_spanned(index, true);
                    let stale = placed.note(index, start, end, !seen);
                    placed.shows(index, first);
                    stale
                }
                None => {
                    placed.forget(index);
                    self.set_spanned(index, false);
                    Vec::new()
                }
            }
        };
        if !stale.is_empty() {
            STALE[0].fetch_add(stale.len() as u64, std::sync::atomic::Ordering::Relaxed);
            let mut textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
            for &slot in &stale {
                textures.remove(slot);
            }
        }
        let stride = view.and_then(view_stride);
        self.note_stride(index, stride);
        let mut buffers = crate::lot4::locks::mutex(crate::lot4::locks::Lock::HeapBuffers, &self.buffers);
        let was = if stride.is_some() {
            buffers.insert(index);
            true
        } else {
            buffers.remove(&index)
        };
        if was {
            self.mark_into(index, marks, sink);
        }
        drop(buffers);
        // (A note of any other kind, or of none, makes the next one a full
        // one again.)
        self.set_noted(index, plain);
        self.set_kind(index, now);
        stale
    }

    /// VIcky (`VICTORY_SPANS_QUICK`): whether title slot `index` may be in
    /// the spans of placed textures ([`HostHeap::spanned`]; a slot past
    /// the set may).
    fn in_spans(&self, index: u32) -> bool {
        let bit = 1u64 << (index % 64);
        self.spanned
            .get(index as usize / 64)
            .is_none_or(|w| w.load(std::sync::atomic::Ordering::Acquire) & bit != 0)
    }

    /// VIcky (`VICTORY_SPANS_QUICK`): title slot `index` is (`true`) or is
    /// no longer in the spans; under the lock of `placed`.
    fn set_spanned(&self, index: u32, there: bool) {
        let bit = 1u64 << (index % 64);
        if let Some(w) = self.spanned.get(index as usize / 64) {
            if there {
                w.fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
            } else {
                w.fetch_and(!bit, std::sync::atomic::Ordering::AcqRel);
            }
        }
    }

    /// [`HostHeap::mark_changed`], or with `sink` (and `marks`) the mark
    /// left there ([`HostHeap::note_buffer_into`]).
    fn mark_into(&self, index: u32, marks: bool, sink: Option<&mut Vec<u32>>) {
        match sink {
            Some(sink) if marks => sink.push(index),
            _ => self.mark_changed(index, marks),
        }
    }

    /// Marks title slot `index` as written since the pass over buffer views
    /// last looked; without `marks` (the pass's own making,
    /// [`HostHeap::note_buffer_marking`]), counts the mark spared if the
    /// slot was not marked already ([`PASS_MARKS_SPARED`]).
    fn mark_changed(&self, index: u32, marks: bool) {
        // VIcky (`VICTORY_HEAP_KEPT_SPLIT`): in a chunk of `mirror_due`,
        // made with the chunk's others at its end ([`KeptLater`]).
        if marks && self.later(index, Later::Mark).is_ok() {
            return;
        }
        let mut changed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Changed, &self.changed);
        if marks {
            changed.insert(index);
        } else if !changed.contains(&index) {
            PASS_MARKS_SPARED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Writes title slot `index`'s element stride into the table shaders
    /// read it from (0: no buffer view).
    fn note_stride(&self, index: u32, stride: Option<u32>) {
        // (With `VICTORY_HEAP_BUFFERS_SKIP_SAME`: only a stride that is not
        // the one there.)
        if let Some(held) = self.strides_held.get(index as usize) {
            // (Load and store, under `made`'s lock: no locked instruction,
            // which would empty the write-combining buffers.)
            if held.load(std::sync::atomic::Ordering::Relaxed) == stride.unwrap_or(0) {
                return;
            }
            held.store(stride.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
        }
        if let Some((_, _, data)) = &self.strides
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            // SAFETY: inside the mapping; shaders read it while we write (a
            // dword, either value is one the slot had).
            unsafe { (*data as *mut u32).add(index as usize).write_volatile(stride.unwrap_or(0)) };
        }
    }

    /// Whether title slot `index`'s last note was of a plain buffer view
    /// ([`HostHeap::noted`]; never without the switch).
    fn noted_plain(&self, index: u32) -> bool {
        self.noted
            .get(index as usize)
            .is_some_and(|n| n.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn set_noted(&self, index: u32, plain: bool) {
        if let Some(n) = self.noted.get(index as usize) {
            n.store(plain, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// `VICTORY_HEAP_SYNC_CHECK=1`: what a full note of title slot `index`
    /// would have found to do beside the stride and the mark, where the
    /// shortcut of `VICTORY_HEAP_BUFFERS_SKIP_SAME` did nothing: the slot
    /// in the queue of textures, in the spans of placed textures, or not in
    /// the set of buffer slots. Each is counted as a difference.
    fn check_noted(&self, index: u32) {
        let queued = self
            .textures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .slots
            .contains_key(&index);
        let spanned = {
            let placed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*self.placed);
            placed.by_slot.contains_key(&index) || placed.first.contains_key(&index)
        };
        let listed = self
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&index);
        checked(Check::Notes, !queued && !spanned && listed, || {
            format!("title slot {index}: queued as a texture {queued}, in the spans {spanned}, a buffer slot {listed}")
        });
    }

    /// This host heap's identity ([`HostHeap::release`]).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned), if they were taken from this heap (`id`, its
    /// [`HostHeap::id`] then); a release naming another heap frees nothing.
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, id: u64, first: u32, n: u32) {
        if id != self.id {
            return;
        }
        let Some(at) = first.checked_sub(self.dynamic_first) else {
            return;
        };
        let mut guard = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Next, &self.next);
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            let given = slots.iter().filter(|taken| **taken).count() as u32;
            self.free.fetch_add(given, std::sync::atomic::Ordering::Relaxed);
            self.too_long.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let _keep = crate::draw::heap_pass::parts::enter(crate::draw::heap_pass::parts::Part::Keep);
        // VIcky (`VICTORY_HEAP_KEPT_SPLIT`): a mirrored slot's, in a chunk of
        // `mirror_due`, made with the chunk's others at its end
        // ([`KeptLater`]).
        let resource = match self.later(index, Later::Keep(resource)) {
            Ok(()) | Err(Later::Mark) => return,
            Err(Later::Keep(resource)) => resource,
        };
        // A running submission may still read the old resource through
        // this slot.
        self.kept.replace(index, resource, retire_held);
    }

    /// VIcky (`VICTORY_HEAP_KEPT_SPLIT`): `what`, for slot `index` (a host
    /// slot's keep, a title slot's mark), noted for the end of the chunk
    /// [`HostHeap::mirror_due`] is at on this thread, if it is at one of
    /// this heap's and `what` is a mark or a mirrored slot's keep; else
    /// given back, to be made now.
    fn later(&self, index: u32, what: Later) -> Result<(), Later> {
        if !self.kept.split() || (matches!(what, Later::Keep(_)) && !self.kept.low(index)) {
            return Err(what);
        }
        // VIcky (`VICTORY_PARALLEL_MIRROR`, [`par_mirror::keep_now`]): on a
        // thread that makes slots with others, a stale or pool slot that
        // comes to hold nothing lets go at once (another thread may make it
        // again, and keep that, before this chunk ends).
        if matches!(what, Later::Keep(None)) && par_mirror::keep_now(self, index) {
            return Err(what);
        }
        let me = std::ptr::from_ref(self) as usize;
        KEPT_LATER.with_borrow_mut(|later| match later {
            Some(later) if later.heap == me => {
                match what {
                    Later::Keep(resource) => later.keeps.push((index, resource)),
                    Later::Mark => later.marks.push(index),
                }
                Ok(())
            }
            _ => Err(what),
        })
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        // A view's range has to lie inside the buffer: a description that runs
        // past it is not a valid call, and what a refused creation leaves in
        // the slot is undefined rather than "whatever was there" -- enough to
        // keep such a description away from a slot shaders read by index.
        let Some(range) = raw_view_range(buffer, offset, len, "SRV") else {
            self.null_raw_at(index);
            return;
        };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: range.first,
                    NumElements: range.elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view that reads
    /// zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, None);
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// `len` bytes of `buffer` from byte `offset` (a multiple of 16): a
    /// root UAV with a size, whose accesses past it do nothing instead of
    /// reaching whatever the host placed after the buffer.
    pub(crate) fn raw_uav_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        // As `raw_at`: the range inside the buffer, or nothing at all. What a
        // refused description leaves in the slot is undefined: it may leave the
        // slot unusable, and a dispatch reading it then returns wrong bytes or
        // faults. Not observed on this machine; recorded as a possibility.
        let Some(range) = raw_view_range(buffer, offset, len, "UAV") else {
            self.null_raw_uav_at(index);
            return;
        };
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: range.first,
                    NumElements: range.elements,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device.CreateUnorderedAccessView(
                buffer,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// nothing: reads give zeros, writes are dropped.
    pub(crate) fn null_raw_uav_at(&self, index: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a slot of our heap; a null resource is allowed with a desc.
        unsafe {
            host.device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(index) * u64::from(self.step),
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// The title slots [`HostHeap::mirror`] makes host slots for: with a
    /// heap map all the title's (past the identity window in pool slots),
    /// else the `count` mirrored at their own index.
    pub(crate) fn mirrored(&self) -> u32 {
        if self.map.is_some() {
            self.title_count
        } else {
            self.count
        }
    }

    /// For a slot on a page of the title's heap that changed with no call
    /// (`super::sync_heaps`: the title copies records into its heap as
    /// bytes): [`HostHeap::mirror`], if the `bytes` bytes of the record at
    /// `handle` are not the ones this last saw there. True if it did.
    ///
    /// The page says only that something on it was written, and `mirror`
    /// by itself would take every buffer or placed-texture record on it for
    /// newly written each time (it leaves such a slot unmade until a draw
    /// or a pass uses it): the texture queue's entry for the slot would
    /// start again, and the buffer pass would look at 2,048 slots for one
    /// that changed.
    /// `record`: the record's bytes as the caller read them (through the
    /// kernel, which fails on a page the title unmapped meanwhile where a
    /// read in place would fault: the title unmaps heap pages too).
    pub(crate) fn mirror_copied(&self, index: u32, handle: u64, record: &[u8]) -> bool {
        // (With a heap map, title slots past the identity window too: they
        // are made in pool slots, `HostHeap::mirror`.)
        if index >= self.mirrored() {
            return false;
        }
        // A slot draws read through a table lately is left to its draws,
        // as the passes over slots picked by index leave it
        // (`texture_slots_remapped`, `slots_written_under`): the title
        // writes its per-draw tables into the same heap every frame, and
        // each draw makes what its table shows in a slot of its own. Those
        // were most of the 26,000 to 43,000 records a frame this pass
        // mirrored in run 200 and in the play session after it, each then
        // made and brought up to date once more by the pass over buffer
        // views (some 50 ms a frame). `VICTORY_HEAP_SYNC_TABLES=1`: they
        // are mirrored too, as in run 200.
        if !sync_tables() && self.tabled_lately(index, crate::frame::current()) {
            COPIED_TABLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        let n = record.len();
        // Never 0, which is "never looked at". (A multiply-and-fold over
        // the record's words: BLAKE3 here was most of the 35 to 42 ms a
        // frame this pass took over some 250,000 to 300,000 records, run
        // 200.)
        let print = record_print(record) | 1;
        // VIcky (`VICTORY_TABLES_LEFT`, stage 2, [`tables_left`]).
        let (frame, mut waits, mut retire) = (crate::frame::current(), Vec::new(), Vec::new());
        let now = {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if copied.is_empty() {
                copied.resize(self.mirrored() as usize, 0);
            }
            let Some(last) = copied.get_mut(index as usize) else {
                return false;
            };
            // A slot never looked at that holds nothing: as it was.
            if *last == print || (*last == 0 && record[..n].iter().all(|&b| b == 0)) {
                return false;
            }
            // (VIcky, stage 2: a buffer or constant view in a slot a draw's
            // table read lately is left to the draws, its print not noted;
            // with the grace, a new constant or placed buffer record
            // elsewhere waits, its print noted.)
            let (note, now) = self.left_found(index, words_of(record), frame, &mut waits, &mut retire);
            if note {
                *last = print;
            }
            now
        };
        self.left_after(frame, waits, retire);
        if !now {
            return false;
        }
        // (VIcky, [`left`]'s line: counted by rule.)
        let _making = left::SyncMaking::enter();
        self.mirror(index, handle, false);
        true
    }

    /// [`HostHeap::mirror_copied`] for every record of one changed page of
    /// the title's heap (`super::sync_heaps`): title slots `slots`, the
    /// record of slot `i` at title address `start + i * step`, and `page`
    /// the bytes read from title address `from` on. Returns the records
    /// looked at and those mirrored.
    ///
    /// The same records are mirrored as one by one. What differs is the
    /// work for the others, which are nearly all of them (some 330,000
    /// looked at a frame for 14,000 mirrored, 16 ms, run 225): the lock of
    /// the fingerprints is taken once for the page and not once for each
    /// record, and a record whose fingerprint is the one last seen is done
    /// with before the question whether its slot was a table's lately. So
    /// the count of slots "left to the draws" is now of records that
    /// changed, not of every record on the page. The slots to mirror are
    /// made after the page's walk, in slot order as before.
    pub(crate) fn mirror_copied_page(
        &self,
        slots: std::ops::Range<u64>,
        (start, step): (u64, u64),
        from: usize,
        page: &[u8],
    ) -> (u64, u64) {
        let (looked, due) = self.copied_due(slots, (start, step), from, page);
        // (VIcky, [`left`]'s line: counted by rule.)
        let _making = left::SyncMaking::enter();
        for &(index, handle) in &due {
            self.mirror(index, handle, false);
        }
        (looked, due.len() as u64)
    }

    /// The first part of [`HostHeap::mirror_copied_page`], which nothing
    /// in the second part changes: the records of the page looked at, and
    /// the slots to mirror (title slot, handle), in slot order. Their
    /// prints are noted as seen.
    pub(crate) fn copied_due(
        &self,
        slots: std::ops::Range<u64>,
        at: (u64, u64),
        from: usize,
        page: &[u8],
    ) -> (u64, Vec<(u32, u64)>) {
        self.copied_due_with(slots, at, from, page, None)
    }

    /// [`HostHeap::copied_due`]; with `records` (VIcky,
    /// `VICTORY_HEAP_SYNC_LEAN`, [`sync_lean`]) also each due slot's record
    /// (its first three words) as the page held it, in the same order.
    pub(crate) fn copied_due_with(
        &self,
        slots: std::ops::Range<u64>,
        (start, step): (u64, u64),
        from: usize,
        page: &[u8],
        mut records: Option<&mut Vec<[u32; 3]>>,
    ) -> (u64, Vec<(u32, u64)>) {
        let (frame, tables) = (crate::frame::current(), sync_tables());
        let mut looked = 0u64;
        let mut due: Vec<(u32, u64)> = Vec::new();
        // VIcky (`VICTORY_TABLES_LEFT`, stage 2, [`tables_left`]): the records
        // that wait their grace, and the holds let go for another record.
        let (mut waits, mut retire) = (Vec::new(), Vec::new());
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if copied.is_empty() {
                copied.resize(self.mirrored() as usize, 0);
            }
            for index in slots {
                let handle = start + index * step;
                // A record that straddles the range's ends is left to the
                // range that holds its first byte.
                let Some(at) = (handle as usize).checked_sub(from) else {
                    continue;
                };
                let Some(record) = page.get(at..at + step as usize) else {
                    continue;
                };
                looked += 1;
                // None past the host heap's slots, as `mirror_copied`.
                let Some(last) = copied.get_mut(index as usize) else {
                    continue;
                };
                // Never 0, which is "never looked at".
                let print = record_print(record) | 1;
                // The record last seen, or a slot never looked at that
                // holds nothing: as it was.
                if *last == print || (*last == 0 && record.iter().all(|&b| b == 0)) {
                    continue;
                }
                // A table's slot lately: left to its draws, and looked at
                // again next time.
                if !tables && self.tabled_lately(index as u32, frame) {
                    COPIED_TABLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                // (VIcky, stage 2: a buffer or constant view in a slot a
                // draw's table read lately is left to the draws and looked
                // at again next time; with the grace, a new constant or
                // placed buffer record elsewhere waits, its print noted.)
                let words = words_of(record);
                let (note, now) = self.left_found(index as u32, words, frame, &mut waits, &mut retire);
                if note {
                    *last = print;
                }
                if !now {
                    continue;
                }
                due.push((index as u32, handle));
                if let Some(records) = records.as_deref_mut() {
                    records.push(words);
                }
            }
        }
        self.left_after(frame, waits, retire);
        (looked, due)
    }

    /// The second part of [`HostHeap::mirror_copied_page`] for the slots
    /// [`HostHeap::copied_due`] gave (`VICTORY_HEAP_SYNC_THREADS`): each
    /// [`HostHeap::mirror`]ed in the order given, with `made` locked once
    /// for every [`MIRROR_CHUNK`] slots instead of once for each. What each
    /// slot's mirror does is the same; only other threads that want the
    /// lock (a draw reading a table, the pass over buffer views) wait for
    /// a chunk where they waited for a slot, and no longer take turns with
    /// this thread slot by slot.
    ///
    /// The host resources the chunk's slots let go of are retired together
    /// once the chunk is done ([`retire_held`]): the title's ring of
    /// buffer views had every slot let go of one each frame, each then a
    /// box of its own in the list of retired objects, under its lock.
    pub(crate) fn mirror_due(&self, due: &[(u32, u64)]) {
        // VIcky (`VICTORY_MADE_TURNS`): asked once a call.
        let turns = turns_now();
        // VIcky (`VICTORY_MADE_PASS_FIRST`, [`made_first`]): asked once a
        // call (never on the pass's own thread).
        let hand = made_first::hand_now();
        // (Slots made since the last hand-over: at least a chunk's between
        // two, `made_first::PASS_TURN_EVERY`.)
        let mut since = made_first::PASS_TURN_EVERY;
        for chunk in due.chunks(MIRROR_CHUNK) {
            RETIRE_LATER.set(Some(Vec::new()));
            {
                let mut made = held_alone(&self.made);
                // VIcky (`VICTORY_HEAP_KEPT_SPLIT`): the chunk's keeps of
                // mirrored slots and its marks made together at its end,
                // before `made` goes.
                let mut later = KeptLater::begin(self);
                for &(index, handle) in chunk {
                    // VIcky (`VICTORY_MADE_TURNS`): a thread waits for
                    // `made`: the chunk ends here (its keeps and marks
                    // made) and goes on once that thread had it. VIcky
                    // (`VICTORY_MADE_PASS_FIRST`): the pass over buffer
                    // views first, if it waits.
                    if since >= made_first::PASS_TURN_EVERY && self.pass_wanted(hand) {
                        drop(later);
                        made = self.pass_turn_over(made);
                        later = KeptLater::begin(self);
                        since = 0;
                    } else if self.turn_wanted(turns) {
                        drop(later);
                        made = self.turn_over(made);
                        later = KeptLater::begin(self);
                    }
                    since += 1;
                    if let Some(record) = self.record_at(index, handle, false) {
                        self.mirror_made(&made, true, (index, handle, false, true), record, &mut Lookups::each());
                    }
                }
                drop(later);
                drop(made);
            }
            if let Some(held) = RETIRE_LATER.take().filter(|held| !held.is_empty()) {
                crate::host::retire(held);
            }
        }
    }

    /// [`HostHeap::mirror_with`] for each of `slots` (title slot, handle),
    /// with `made` locked once for all of them (the pass over buffer views
    /// with `VICTORY_HEAP_SYNC_CACHE`, a chunk at a time): what each
    /// slot's mirror does is the same, other threads wait for the chunk
    /// where they waited for a slot. `lookups` has one for each slot;
    /// `marks` as for [`HostHeap::mirror_with`].
    pub(crate) fn mirror_chunk(&self, slots: &[(u32, u64)], lookups: &mut [Lookups], marks: bool) -> Vec<Option<View>> {
        self.mirror_held(true, slots, lookups, marks)
    }

    /// VIcky (`VICTORY_HEAP_PASS_THREADS`): [`HostHeap::mirror_chunk`] for
    /// one of the threads of the pass over buffer views, with `made` held
    /// together with the pass's other threads (its read lock), which make
    /// other title slots ([`Made`]). What each slot's making does is the
    /// same, but that a pool slot is not won by giving back others'
    /// ([`HostHeap::reclaim`]): the title slots that wanted one are noted
    /// for [`HostHeap::make_room`] ([`take_room_wanted`]).
    pub(crate) fn mirror_shared(&self, slots: &[(u32, u64)], lookups: &mut [Lookups], marks: bool) -> Vec<Option<View>> {
        self.mirror_held(false, slots, lookups, marks)
    }

    /// [`HostHeap::mirror_chunk`] with `made` held `alone` or with the
    /// pass's other threads.
    fn mirror_held(&self, alone: bool, slots: &[(u32, u64)], lookups: &mut [Lookups], marks: bool) -> Vec<Option<View>> {
        use crate::draw::heap_pass::parts::{Part, enter};
        if !alone {
            // VIcky (`VICTORY_PARALLEL_MIRROR`, [`par_mirror`]): each slot
            // made with its piece locked.
            return self.pass_job(|pieces| self.mirror_shared_in(pieces, slots, lookups, marks));
        }
        // VIcky (`VICTORY_MADE_TURNS`): asked once a chunk.
        let turns = turns_now();
        let mut made = self.made_counted(alone);
        slots
            .iter()
            .zip(lookups)
            .map(|(&(index, handle), lookups)| {
                // VIcky (`VICTORY_MADE_TURNS`): a thread waits for `made`:
                // it goes to that thread before this slot.
                if self.turn_wanted(turns) {
                    let began = std::time::Instant::now();
                    let served = self.turns.served.load(std::sync::atomic::Ordering::SeqCst);
                    drop(std::mem::replace(&mut made, MadeCounted { made: MadeHeld::Gone, held: None }));
                    made = MadeCounted::alone(self.turn_back(served, began));
                }
                let record = {
                    let _read = enter(Part::Read);
                    self.record_at(index, handle, true)?
                };
                let _decide = enter(Part::Decide);
                self.mirror_made(&made, alone, (index, handle, true, marks), record, lookups)
            })
            .collect()
    }

    /// [`HostHeap::mirror_held`] with `made` held with the pass's other
    /// threads; with `pieces`, each slot made with its piece locked
    /// ([`HostHeap::in_piece`]).
    fn mirror_shared_in(&self, pieces: bool, slots: &[(u32, u64)], lookups: &mut [Lookups], marks: bool) -> Vec<Option<View>> {
        use crate::draw::heap_pass::parts::{Part, enter};
        let made = self.made_counted(false);
        slots
            .iter()
            .zip(lookups)
            .map(|(&(index, handle), lookups)| {
                let record = {
                    let _read = enter(Part::Read);
                    self.record_at(index, handle, true)?
                };
                let _decide = enter(Part::Decide);
                let mut make = || self.mirror_made(&made, false, (index, handle, true, marks), record, lookups);
                if pieces { self.in_piece(&made, index, make) } else { make() }
            })
            .collect()
    }

    /// VIcky (`VICTORY_HEAP_PASS_THREADS`): once the threads of a pass over
    /// buffer views are done, with `made` held alone, the slots (title slot,
    /// handle) they found no pool slot for: if the pool slots of title slots
    /// without a live view are given back now ([`HostHeap::reclaim`], at
    /// most once every two seconds), they are made as the pass makes them,
    /// as the pass on one thread made them when the first of them gave
    /// those back. Returns how many were made.
    pub(crate) fn make_room(&self, slots: &[(u32, u64)], marks: bool) -> usize {
        if slots.is_empty() {
            return 0;
        }
        let made = held_alone(&self.made);
        if self.reclaim(&made) == 0 {
            return 0;
        }
        for &(index, handle) in slots {
            if let Some(record) = self.record_at(index, handle, true) {
                self.mirror_made(&made, true, (index, handle, true, marks), record, &mut Lookups::kept());
            }
        }
        slots.len()
    }

    /// `made` locked by the pass over buffer views, `alone` or with the
    /// pass's other threads: on a pass's thread, the wait for the lock and
    /// how long it is held are counted (`crate::draw::heap_pass::parts`).
    fn made_counted(&self, alone: bool) -> MadeCounted<'_> {
        use crate::draw::heap_pass::parts;
        use crate::lot4::locks::{Lock, tried as counted, waited};
        let lock = if alone { Lock::Made } else { Lock::MadeShared };
        let tried = if alone {
            match self.made.try_write() {
                Ok(made) => Some(MadeHeld::Alone(made)),
                Err(std::sync::TryLockError::Poisoned(e)) => Some(MadeHeld::Alone(e.into_inner())),
                Err(std::sync::TryLockError::WouldBlock) => None,
            }
        } else {
            match self.made.try_read() {
                Ok(made) => Some(MadeHeld::Shared(made)),
                Err(std::sync::TryLockError::Poisoned(e)) => Some(MadeHeld::Shared(e.into_inner())),
                Err(std::sync::TryLockError::WouldBlock) => None,
            }
        };
        let made = match tried {
            Some(made) => {
                counted(lock, true);
                made
            }
            None => {
                let _wait = parts::enter(parts::Part::Wait);
                let began = std::time::Instant::now();
                let made = if alone {
                    // VIcky (`VICTORY_MADE_PASS_FIRST`, [`made_first`]): on
                    // the pass's thread, it says it waits and spins for
                    // `made` before it sleeps (the sync's chunks let it in).
                    if parts::counting() && made_first::spin_now() {
                        MadeHeld::Alone(self.made_pass_first(began))
                    } else {
                        MadeHeld::Alone(self.made.write().unwrap_or_else(|e| e.into_inner()))
                    }
                } else {
                    MadeHeld::Shared(self.made.read().unwrap_or_else(|e| e.into_inner()))
                };
                waited(lock, began.elapsed().as_nanos() as u64);
                parts::waited(began.elapsed());
                made
            }
        };
        MadeCounted { made, held: parts::counting().then(std::time::Instant::now) }
    }

    /// Makes title slot `index`'s host slot (its own, or with a heap map
    /// past the identity window its pool slot) from the record the title
    /// has at `handle` if it changed since, and returns the view recorded
    /// there. Buffer views are made only for draws (`for_draw`), which also
    /// copy their data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        let record = self.record_at(index, handle, for_draw)?;
        let made = self.made_alone();
        // VIcky (stage 4's counter, [`made_read`]): whether the making only
        // looked, by caller.
        let caller = made_read::begin();
        let view = self.mirror_made(&made, true, (index, handle, for_draw, true), record, &mut Lookups::each());
        made_read::end(caller);
        view
    }

    /// [`HostHeap::mirror`] (not for a draw) of a descriptor the title
    /// copied into title slot `index` at `handle` (`super::copy_simple`).
    /// VIcky (`VICTORY_MIRROR_TRY`, [`super::mirror_try`]): made now if
    /// `made` is free; else left on this heap's list for the next drain
    /// ([`HostHeap::drain_deferred`]), and with `check` made now all the
    /// same, waiting for the lock. Without the switch, `mirror` itself.
    pub(crate) fn mirror_copy(&self, index: u32, handle: u64) {
        self.mirror_tried(index, handle, false);
    }

    /// [`HostHeap::mirror`] (not for a draw) of a view the title made by a
    /// call (`Create*View`) in title slot `index` at `handle`
    /// (`super::record`). VIcky (`VICTORY_MIRROR_TRY=2`, `check2`): as
    /// [`HostHeap::mirror_copy`], counted apart.
    pub(crate) fn mirror_created(&self, index: u32, handle: u64) {
        self.mirror_tried(index, handle, true);
    }

    /// [`HostHeap::mirror_copy`], or with `created`
    /// [`HostHeap::mirror_created`].
    fn mirror_tried(&self, index: u32, handle: u64, created: bool) {
        use super::mirror_try::{Count, Mode, Slot, count, mode, record_key};
        let check = match mode() {
            Mode::Off => {
                self.mirror(index, handle, false);
                return;
            }
            Mode::Try => false,
            Mode::Check => true,
        };
        let Some(record) = self.record_at(index, handle, false) else {
            return;
        };
        let free = match self.made.try_write() {
            Ok(made) => Some(made),
            Err(std::sync::TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
        crate::lot4::locks::tried(crate::lot4::locks::Lock::Made, free.is_some());
        if let Some(made) = free {
            count(if created { Count::CreatedAtCall } else { Count::MadeAtCall }, 1);
            self.mirror_made(&made, true, (index, handle, false, true), record, &mut Lookups::each());
            return;
        }
        if check {
            // Made as before, then left with the record it was made from.
            let made = self.made_alone();
            self.mirror_made(&made, true, (index, handle, false, true), record, &mut Lookups::each());
        }
        let key = if check { record_key(record) } else { 0 };
        if self.deferred.push(Slot { index, handle, key }) {
            count(if created { Count::CreatedLeft } else { Count::Left }, 1);
        } else if !check {
            // The list is full: made now, waiting, as before.
            self.mirror(index, handle, false);
        }
    }

    /// VIcky (`VICTORY_MIRROR_TRY`): the slots [`HostHeap::mirror_copy`] (and
    /// [`HostHeap::mirror_created`]) left on this heap's list, made now in
    /// the order they were left, through
    /// [`HostHeap::mirror_due`] as the heap sync makes the slots it finds
    /// written, each from its record as it is now. A slot whose record can
    /// no longer be read (the title unmapped its page) is not made. With
    /// `check`, the slots whose record is not the one made at the call are
    /// counted. `by_sync`: drained by the heap sync, else by a pass over
    /// the heap (for the counts). Returns how many were made.
    pub(crate) fn drain_deferred(&self, by_sync: bool) -> usize {
        use super::mirror_try::{Count, Mode, count, mode, readable, record_key};
        let slots = self.deferred.take();
        if slots.is_empty() {
            return 0;
        }
        let check = mode() == Mode::Check;
        let (mut due, mut gone, mut differing) = (Vec::with_capacity(slots.len()), 0, 0);
        // The page last found readable: a copy's slots lie side by side.
        let mut page_read = None;
        for slot in &slots {
            let pages = (slot.handle >> 12, (slot.handle + 11) >> 12);
            if page_read != Some(pages) {
                if !readable(slot.handle) {
                    gone += 1;
                    continue;
                }
                page_read = Some(pages);
            }
            if check && record_key(super::record_words(slot.handle)) != slot.key {
                differing += 1;
            }
            due.push((slot.index, slot.handle));
        }
        self.mirror_due(&due);
        count(if by_sync { Count::DrainedBySync } else { Count::DrainedByPass }, due.len() as u64);
        count(Count::Gone, gone);
        if check {
            count(Count::Compared, due.len() as u64);
            count(Count::Differing, differing);
        }
        due.len()
    }

    /// Tests: `f` run with `made` held alone by this thread.
    #[cfg(test)]
    pub(crate) fn with_made_held<R>(&self, f: impl FnOnce() -> R) -> R {
        let made = held_alone(&self.made);
        let out = f();
        drop(made);
        out
    }

    /// [`HostHeap::mirror`] for a draw, with the host copies its making
    /// looks up kept in `lookups`, for the caller to ask again
    /// ([`buffer_copy_with`]). `marks`: whether its making marks the slot as
    /// written for the pass over buffer views (false: that pass's own making
    /// with `VICTORY_HEAP_PASS_ONE_LOOK=1`, [`HostHeap::note_buffer_marking`]).
    pub(crate) fn mirror_with(&self, index: u32, handle: u64, lookups: &mut Lookups, marks: bool) -> Option<View> {
        use crate::draw::heap_pass::parts::{Part, enter};
        let record = {
            let _read = enter(Part::Read);
            self.record_at(index, handle, true)?
        };
        let made = self.made_counted(true);
        let _decide = enter(Part::Decide);
        self.mirror_made(&made, true, (index, handle, true, marks), record, lookups)
    }

    /// The part of [`HostHeap::mirror`] before `made` is locked: `None` if
    /// the slot has no host slot at all; else the record's three words.
    fn record_at(&self, index: u32, handle: u64, for_draw: bool) -> Option<[u32; 3]> {
        if index >= self.mirrored() {
            crate::draw::heap_pass::parts::count(crate::draw::heap_pass::parts::Count::Past);
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        if self.map.is_some() && index >= self.count {
            // Where the title's descriptors start, for `reclaim`.
            let start = handle.wrapping_sub(u64::from(index) * TITLE_SLOT_BYTES);
            let _ = self.title_start.compare_exchange(
                0,
                start,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        Some(unsafe { (handle as *const [u32; 3]).read_unaligned() })
    }

    /// The rest of [`HostHeap::mirror`], with `made` locked by the caller
    /// (one lock for several slots: [`HostHeap::mirror_due`],
    /// [`HostHeap::mirror_chunk`]): title slot `index` at `handle`, which
    /// held `record` when [`HostHeap::record_at`] read it; `marks` as for
    /// [`HostHeap::mirror_with`]. `alone`: `made` is held alone, not with
    /// the other threads of the pass over buffer views
    /// ([`HostHeap::mirror_shared`]).
    fn mirror_made(
        &self,
        made: &Made,
        alone: bool,
        slot: (u32, u64, bool, bool),
        [tag, i, generation]: [u32; 3],
        lookups: &mut Lookups,
    ) -> Option<View> {
        use crate::draw::heap_pass::parts::{Part, enter};
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let view = {
            let _view = enter(Part::View);
            (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten()
        };
        self.mirror_view(made, alone, slot, key, view, lookups)
    }

    /// [`HostHeap::mirror_made`] with the record's view (`key` its record)
    /// looked up by the caller: VIcky (`VICTORY_PARALLEL_MIRROR`,
    /// [`par_mirror`]) looks up a chunk's views at once.
    fn mirror_view(
        &self,
        made: &Made,
        alone: bool,
        (index, handle, for_draw, marks): (u32, u64, bool, bool),
        key: u64,
        view: Option<View>,
        lookups: &mut Lookups,
    ) -> Option<View> {
        use crate::draw::heap_pass::parts::{Count, Part, count, enter};
        // VIcky (`VICTORY_LOT4_PROBE`, P2): the sync's slots by kind (also
        // those the parallel mirror's threads make).
        #[cfg(feature = "diagnostics")]
        if !for_draw {
            crate::lot4::sync_slot(view.as_ref());
        }
        // VIcky ([`left`]'s line): the heap sync's slots, by rule.
        if !for_draw && left::in_sync_making() {
            self.count_sync_rule(made, index, key, view.as_ref());
        }
        // VIcky (`VICTORY_TABLES_LEFT=check`, [`tables_left`]): the check's
        // model of what the switch would show (nothing without it).
        self.check_making(made, index, key, view.is_some(), for_draw);
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            count(Count::NotOurs);
            #[cfg(feature = "diagnostics")]
            note_kind::slot(note_kind::Slot::NotOurs);
            self.empty_slot(made, index, for_draw, marks);
            return None;
        };
        let host = {
            let _map = enter(Part::Map);
            self.host_slot(made, index, alone)
        };
        let Some((host, taken)) = host else {
            // Past the window with the pool used up: the slot reads the
            // guard. What the title wrote is noted all the same (the sets
            // of slots shaders pick by index retry it).
            count(Count::NoHost);
            if !for_draw {
                for stale in self.note_buffer(index, Some(&view), false) {
                    if stale != index {
                        self.unmake(made, stale);
                    }
                }
            }
            return Some(view);
        };
        if taken {
            count(Count::Taken);
        }
        if taken || made.key(host) != key {
            // Views of other textures in the memory this one describes
            // are stale: their host slots read nothing from here on, and
            // let go of the host copies they held.
            let stale = {
                let _note = enter(Part::Note);
                self.note_buffer_marking(index, Some(&view), for_draw, marks)
            };
            for stale in stale {
                if stale != index {
                    self.unmake(made, stale);
                }
            }
            // A view of a texture object that has no host copy yet waits
            // for it, as buffer views wait for a draw: the slot is made
            // when the texture is first used (`super::make_waiting`), or
            // here when a draw reads the slot through a table.
            let waiting = if for_draw { None } else { view.waiting_texture() };
            if let Some(texture) = waiting {
                super::wait_for(texture, handle);
            }
            let later = !for_draw && (view.is_buffer() || waiting.is_some());
            // Another view of the placed texture the slot shows already (a
            // title that streams a texture in writes its view again with
            // more mips each time some arrive): the slot goes on showing
            // the old view until a draw or the texture pass makes the new
            // one and uploads what it shows. Made null here, the texture
            // read as nothing for as long as it waited in the queue, every
            // time it got sharper.
            let old = made.key(host);
            // (With `VICTORY_HEAP_BUFFERS_SKIP_SAME=1` the old view is not
            // looked up for a view that is no placed texture's: no old view
            // is the same placed texture as it.)
            if later
                && keep_views()
                && (!skip_same() || placed_texture_view(&view))
                // (No record is 0: generations start at 1. The table's lock
                // not taken for it, VIcky.)
                && old != 0
                && view_at(old as u32, (old >> 32) as u32)
                    .is_some_and(|old| same_placed_texture(&old, &view))
            {
                KEPT_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(view);
            }
            made.set_key(host, if later { 0 } else { key });
            let kept = if later {
                None
            } else {
                self.make_slot(made, alone, &view, key, host, lookups)
            };
            if kept.is_none() && !later {
                count(Count::Unmade);
            }
            if kept.is_none() && !self.sampler {
                let _create = enter(Part::Create);
                null_for(&view, self.cpu(host));
            }
            self.keep(host, kept.flatten());
            if taken {
                let _map = enter(Part::Map);
                self.publish(made, index);
                #[cfg(feature = "diagnostics")]
                note_pool(made);
            }
        } else {
            count(Count::Same);
            made_read::same();
            #[cfg(feature = "diagnostics")]
            note_kind::slot(note_kind::Slot::Same);
        }
        Some(view)
    }

    /// Makes host slot `host` from `view` (record `key`), as [`make`]
    /// does; with `VICTORY_HEAP_SYNC_CACHE`, a buffer view made before is
    /// copied from the cache ([`RecordCache`]) when it would be made on the
    /// same host copies.
    /// (`alone` as for [`HostHeap::mirror_made`].)
    fn make_slot(&self, made: &Made, alone: bool, view: &View, key: u64, host: u32, lookups: &mut Lookups) -> Option<Kept> {
        let at = self.cpu(host);
        let size = cache_size();
        if size == 0 || self.sampler {
            return made_otherwise(view, at, lookups);
        }
        let Some(extent) = cacheable(view) else {
            FAST.add(Fast::NotCacheable as usize, 1);
            return made_otherwise(view, at, lookups);
        };
        // VIcky (`VICTORY_HEAP_PASS_THREADS`): the threads of the pass take
        // the cache's lock in turn, so the copies its making asks for are
        // looked up before it (the same asks, kept in `lookups`).
        if !alone && lookups.kept {
            let _ = shown_copy(view, extent.0, extent.1, lookups);
        }
        let mut cache = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Cache, &made.cache);
        if !cache.tried {
            cache.tried = true;
            cache.table = RecordCache::new(size, self.step);
        }
        let Some(table) = cache.table.as_mut() else {
            drop(cache);
            return made_otherwise(view, at, lookups);
        };
        table.make(view, key, extent, at, lookups)
    }

    /// VIcky (`VICTORY_HEAP_PASS_THREADS_CHECK`): of each of `slots`, the
    /// record its host slot was last made from (0: none, or no host slot)
    /// and the host resource the slot holds (as a raw pointer).
    pub(crate) fn slots_seen(&self, slots: &[u32]) -> Vec<(u64, Option<usize>)> {
        slots
            .chunks(1024)
            .flat_map(|chunk| {
                let made = self.made_alone();
                let hosts: Vec<Option<u32>> = chunk.iter().map(|&index| made.host(index)).collect();
                let kept = self.kept.both();
                hosts
                    .into_iter()
                    .map(|host| {
                        let Some(host) = host else {
                            return (0, None);
                        };
                        let shown = kept.get(host).and_then(Option::as_ref);
                        (made.key_of(host).unwrap_or(0), shown.map(|b| windows::core::Interface::as_raw(b) as usize))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Whether title slot `index`'s host slot is backed by a made host
    /// descriptor (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        let made = held_alone(&self.made);
        made.host(index)
            .and_then(|host| made.key_of(host))
            .is_some_and(|key| key != 0)
    }

    /// The host slot title slot `index` is made in, if it has one (tests).
    #[cfg(test)]
    pub(crate) fn host_of(&self, index: u32) -> Option<u32> {
        held_alone(&self.made).host(index)
    }

    /// The host resource title slot `index`'s host slot shows, as a raw
    /// pointer; `None` if it shows none (tests).
    #[cfg(test)]
    pub(crate) fn shown_raw(&self, index: u32) -> Option<usize> {
        let host = held_alone(&self.made).host(index)?;
        self.kept
            .with(host, |slot| slot?.as_ref().map(|b| windows::core::Interface::as_raw(b) as usize))
    }

    /// Whether title slot `index`'s host slot shows `buffer`; `None` if it
    /// shows no resource (diagnostics: a buffer view's slot must show the
    /// host copy that is brought up to date for it).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn shows(&self, index: u32, buffer: &ID3D12Resource) -> Option<bool> {
        let made = match self.made.try_write() {
            Ok(made) => made,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                crate::draw::heap_pass::parts::count(crate::draw::heap_pass::parts::Count::ShowsWaited);
                self.made_alone()
            }
        };
        let host = made.host(index);
        drop(made);
        let host = host?;
        self.kept.with(host, |slot| {
            let shown = slot?.as_ref()?;
            Some(windows::core::Interface::as_raw(shown) == windows::core::Interface::as_raw(buffer))
        })
    }

    /// VIcky (`VICTORY_HEAP_PASS_THREADS`, diagnostics): of `slots` (title
    /// slot, the buffer its range is brought up to date in), how many show
    /// another buffer ([`HostHeap::shows`] false), for a thread of the pass:
    /// `made` held with the other threads, once for all of them.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn shows_elsewhere(&self, slots: &[(u32, &ID3D12Resource)]) -> u64 {
        if slots.is_empty() {
            return 0;
        }
        let hosts: Vec<Option<u32>> = {
            let made = self.made.read().unwrap_or_else(|e| e.into_inner());
            slots.iter().map(|&(index, _)| made.host(index)).collect()
        };
        let kept = self.kept.both();
        slots
            .iter()
            .zip(hosts)
            .filter(|&(&(_, buffer), host)| {
                host.and_then(|host| kept.get(host)?.as_ref())
                    .is_some_and(|shown| windows::core::Interface::as_raw(shown) != windows::core::Interface::as_raw(buffer))
            })
            .count() as u64
    }

    /// What the queue of textures picked by index knows of title slot
    /// `index` at `frame`, in a few words (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn texture_slot_state(&self, index: u32, frame: u64) -> String {
        let textures = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Textures, &self.textures);
        match textures.slots.get(&index) {
            None => "not queued".into(),
            Some(w) => format!(
                "queued due={} written@{} looks={} part_mapped={} recheck={} ds_holds={}",
                if w.due <= frame { "now".to_string() } else { format!("@{}", w.due) },
                w.written,
                w.looks,
                w.mapped,
                w.checked.2 != 0,
                w.holds
            ),
        }
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = self.made_alone().key_of(index)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The unsigned integer format with the layout of a 16-bit float or
/// signed integer format (one, two or four channels), if `format` is one.
pub(crate) fn half_format(format: u32) -> Option<u32> {
    match format {
        // R16G16B16A16_FLOAT and _SINT: R16G16B16A16_UINT.
        10 | 14 => Some(12),
        // R16G16_FLOAT and _SINT: R16G16_UINT.
        34 | 38 => Some(36),
        // R16_FLOAT and _SINT: R16_UINT.
        54 | 59 => Some(57),
        _ => None,
    }
}

/// `view` as a shader is shown it where it moves a buffer's 16-bit
/// channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`): a buffer view in a
/// 16-bit float or signed format, in the unsigned format of the same
/// layout, through which every 16-bit pattern is read and stored as it
/// is. `None` for every other view: the title's own serves (an unsigned
/// one already carries the bits; a texture can only be viewed in the
/// formats it was made for).
pub(crate) fn half_view(view: &View) -> Option<View> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    match view {
        View::Shader(resource, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Shader(*resource, Some(desc)))
        }
        View::Unordered(resource, counter, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Unordered(*resource, *counter, Some(desc)))
        }
        View::Placed(unordered, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Placed(*unordered, *resource, desc))
        }
        _ => None,
    }
}

/// The format of `view`, a buffer view through which a shader moves
/// 16-bit channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`), if those integers
/// are not what the console gives: a format of wider channels, a
/// normalised one or a signed 8-bit one, whose values the console converts
/// to and from halves. `None` for the formats the integers serve exactly
/// (16 bits a channel of any kind, through [`half_view`] where needed, and
/// unsigned channels of fewer bits; the title's are 16-bit unsigned) and
/// for any other view.
pub(crate) fn half_inexact(view: &View) -> Option<u32> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    // R16G16B16A16, R16G16 and R16 (float, unsigned, signed), then
    // R10G10B10A2, R8G8B8A8, R8G8 and R8 unsigned.
    const EXACT: [u32; 13] = [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62];
    let format = match view {
        View::Shader(_, Some(desc)) | View::Unordered(_, _, Some(desc)) if desc[1] == BUFFER => {
            desc[0]
        }
        View::Placed(_, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => desc[0],
        _ => return None,
    };
    (!EXACT.contains(&format)).then_some(format)
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The float format of the same width as a 32-bit integer one (other
/// formats as they are). Translated shaders declare every typed buffer load
/// and store as 32-bit float (`victory_shader` `emit/lower.rs`); through a
/// view in a 32-bit integer format a store converts the float to the
/// integer it rounds to, so an integer's bits stored as a float come out as
/// 0 for most of them, where the console moves the bits. The float format
/// moves them unchanged. From GamingRipperDoc's fix in the fork (#684,
/// `bit_exact_format`).
fn bit_exact_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_R32G32B32A32_UINT | DXGI_FORMAT_R32G32B32A32_SINT => DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32_UINT | DXGI_FORMAT_R32G32B32_SINT => DXGI_FORMAT_R32G32B32_FLOAT,
        DXGI_FORMAT_R32G32_UINT | DXGI_FORMAT_R32G32_SINT => DXGI_FORMAT_R32G32_FLOAT,
        DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32_SINT => DXGI_FORMAT_R32_FLOAT,
        other => other,
    }
}

/// The format the host makes a typed buffer view of `format` in:
/// [`bit_exact_format`] by default, the title's with
/// `VICTORY_TYPED_BIT_EXACT=0`. The translator declares typed accesses as
/// float, so through a view of a 32-bit integer format the bits were
/// converted (most became 0). A store of
/// `CS_ShadowRenderTiledCascadesAlphaCBPPPC` (#f133e35e) sets bit 14 of a
/// per-tile word this way (0x2957c): with the title's format the tile's
/// other flags were wiped and the characters' heads and skin vanished once
/// that program ran (RT88a); with the float format of the same width they
/// stay (F1). After fork #684 (GamingRipperDoc, `bit_exact_format`).
fn typed_view_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("VICTORY_TYPED_BIT_EXACT").is_none_or(|v| v != "0")) {
        bit_exact_format(format)
    } else {
        format
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The title memory the host slot of buffer view `view` is made on
/// (address, bytes): the whole buffer the view is of, which is what
/// [`make`] finds the host copy by. [`buffer_range`] is the part of it the
/// view reads.
pub(crate) fn buffer_extent(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Shader(r, Some(_)) | View::Unordered(r, _, Some(_)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )),
        _ => None,
    }
}

/// Whether a buffer view is brought up to date in the host copy its slot
/// shows ([`buffer_copy`]). On by default; `VICTORY_VIEW_EXTENT=0` looks the
/// copy up by the view's range alone, as before.
fn view_extent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_EXTENT").is_none_or(|v| v != "0"))
}

/// The host copy buffer view `view` reads, and the title range (address,
/// bytes) it reads there; `None` for other views and empty ranges.
///
/// The copy is the one the view's slot shows: [`make`] finds it by the
/// whole buffer the view is of ([`buffer_extent`]). Looked up by the range
/// the view reads, as it was, the copy is another one wherever the two
/// lookups differ: for memory outside buffer objects, a view of a part of a
/// buffer that crosses a window line (the buffer's window is two cells, the
/// part's one), and for buffer objects, a part that also lies in a smaller
/// buffer object. The slot then showed a copy nothing brought up to date,
/// and what shaders wrote through it was looked for in the other copy.
pub(crate) fn buffer_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    buffer_copy_with(view, &mut Lookups::each())
}

/// [`buffer_copy`], asking `lookups` for the host copies: with copies
/// kept there by the making of the view's slot ([`HostHeap::mirror_with`]),
/// none is looked up twice.
pub(crate) fn buffer_copy_with(
    view: &View,
    lookups: &mut Lookups,
) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    if len == 0 {
        return None;
    }
    // A view off its buffer's elements is made on a copy of its own bytes
    // (`own_bytes`): the same question `buffer_srv` and `buffer_uav` ask,
    // of the copy they find, so that the copy brought up to date is the
    // one the slot shows.
    if let Some((base, width)) = buffer_extent(view)
        && let Some((first, count, element, stride, unknown)) = buffer_fields(view)
        && let Some(whole) = lookups.find(base, width)
        && let Some((start, bytes)) = own_bytes(whole.offset(base), base, first, count, element, stride, unknown)
        && let Some(own) = lookups.exact(start, bytes)
    {
        return Some((own, start, bytes));
    }
    if view_extent()
        && let Some((base, width)) = buffer_extent(view)
        // (The console is lax: a view may begin past its buffer's end.)
        && address >= base
        && address - base < width.max(1)
        && let Some(shadow) = lookups.find(base, width)
    {
        return Some((shadow, address, len));
    }
    lookups.find(address, len).map(|shadow| (shadow, address, len))
}

/// The host copies of title memory one slot's making and the questions
/// after it ask for ([`crate::shadow::find`], [`crate::shadow::exact`]).
/// [`Lookups::each`] looks every ask up anew, as always; [`Lookups::kept`]
/// (`VICTORY_HEAP_SYNC_CACHE`) looks each distinct ask up once and gives
/// the same copy when it comes again. A slot of the pass over buffer views
/// asked for the same copy two or three times: once to make the slot, and
/// once or twice more for the copy to bring up to date ([`buffer_copy`]).
/// The copies are the same, unless the tables of copies change between
/// two asks; the asks are then answered as if they had come at once.
pub(crate) struct Lookups {
    kept: bool,
    /// The asks kept (address, bytes, the copy): a slot asks for at most
    /// three distinct ones; more are looked up each time.
    finds: [Option<Asked>; 3],
    exacts: [Option<Asked>; 2],
}

type Asked = (u64, u64, Option<std::sync::Arc<crate::shadow::Shadow>>);

impl Lookups {
    /// Every ask looked up when it comes.
    pub(crate) fn each() -> Self {
        Lookups { kept: false, finds: Default::default(), exacts: Default::default() }
    }

    /// Each distinct ask looked up once.
    pub(crate) fn kept() -> Self {
        Lookups { kept: true, finds: Default::default(), exacts: Default::default() }
    }

    /// The lookups kept so far (with [`Lookups::kept`]).
    pub(crate) fn asked(&self) -> u64 {
        self.finds.iter().chain(&self.exacts).filter(|a| a.is_some()).count() as u64
    }

    /// [`crate::shadow::find`].
    fn find(&mut self, address: u64, len: u64) -> Option<std::sync::Arc<crate::shadow::Shadow>> {
        if !self.kept {
            return looked_up(address, len, crate::shadow::find);
        }
        Self::ask(&mut self.finds, address, len, crate::shadow::find)
    }

    /// [`crate::shadow::exact`].
    fn exact(&mut self, address: u64, len: u64) -> Option<std::sync::Arc<crate::shadow::Shadow>> {
        if !self.kept {
            return looked_up(address, len, crate::shadow::exact);
        }
        Self::ask(&mut self.exacts, address, len, crate::shadow::exact)
    }

    /// The copy `look` gives for `(address, len)`, kept in `asked`.
    fn ask(
        asked: &mut [Option<Asked>],
        address: u64,
        len: u64,
        look: fn(u64, u64) -> Option<std::sync::Arc<crate::shadow::Shadow>>,
    ) -> Option<std::sync::Arc<crate::shadow::Shadow>> {
        if let Some((_, _, found)) = asked.iter().flatten().find(|a| (a.0, a.1) == (address, len)) {
            return found.clone();
        }
        let found = looked_up(address, len, look);
        if let Some(room) = asked.iter_mut().find(|a| a.is_none()) {
            *room = Some((address, len, found.clone()));
        }
        found
    }
}

/// VIcky (`VICTORY_HEAP_PASS_THREADS`): the title slots this thread's
/// [`HostHeap::mirror_shared`] found no pool slot for since the last call.
pub(crate) fn take_room_wanted() -> Vec<u32> {
    ROOM_WANTED.with_borrow_mut(std::mem::take)
}

/// `look(address, len)`, timed and counted for the pass over buffer views
/// (`crate::draw::heap_pass::parts`).
fn looked_up(
    address: u64,
    len: u64,
    look: fn(u64, u64) -> Option<std::sync::Arc<crate::shadow::Shadow>>,
) -> Option<std::sync::Arc<crate::shadow::Shadow>> {
    use crate::draw::heap_pass::parts;
    let _lookup = parts::enter(parts::Part::Lookup);
    parts::count(parts::Count::Lookups);
    look(address, len)
}

/// VIcky (`VICTORY_HEAP_SYNC_CACHE`): buffer views made before, by the
/// record that names them, each made once more in a heap of the host's
/// that shaders do not see, from where `CopyDescriptorsSimple` copies it
/// into a slot that is to show it again.
///
/// The title rewrites rings of descriptors every frame: some 100,000 slots
/// a frame got another record (RT33), each then made null by the pass over
/// records copied in as bytes and made again by the pass over buffer views
/// ([`HostHeap::mirror`]): the copy it is made on looked up, checked, and
/// `CreateShaderResourceView` called, some 0.3 us; the copy is a few ns.
///
/// What [`make`] makes of a buffer view is fixed by the view (fixed by its
/// record: an entry's generation changes when its resource goes, and the
/// other records name their view whole) and by the host copies it is made
/// on: the copy of its whole buffer, and the copy it shows (a copy of its
/// own bytes, or the same; [`BufferPlan`]). A copy's number
/// (`crate::shadow::Shadow::id`) is never given to another. So an entry
/// keeps the numbers of both, and a slot is copied from it only when the
/// same copies are looked up now ([`shown_copy`]; lookups the making
/// needs anyway, and which the pass over buffer views asks again,
/// [`Lookups`]); else it is made as before and the entry made again. The
/// copy shown is held from the lookup on (by the lookup, then by the slot
/// as without the cache), so the buffer the entry's view names is alive
/// when it is copied. The cache holds no copy itself: a copy nothing else
/// holds is let go as before.
///
/// Two entries per set, by the record's hash; the older one is replaced.
struct RecordCache {
    /// Held: the entries' views are in it.
    _heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: usize,
    entries: Vec<Cached>,
    /// Counts the uses, for which entry of a set is older.
    tick: u32,
}

#[derive(Clone, Copy, Default)]
struct Cached {
    /// The record (0: an empty entry).
    key: u64,
    /// The numbers of the copy of the whole buffer and of the copy shown.
    whole: u64,
    shown: u64,
    /// [`BufferPlan::print`] of what was made (with `VICTORY_HEAP_SYNC_CHECK`).
    print: u64,
    /// The tick of its last use.
    used: u32,
}

impl RecordCache {
    fn new(size: usize, step: u32) -> Option<Self> {
        let host = host()?;
        let size = size.next_power_of_two().clamp(1 << 10, 1 << 22);
        // SAFETY: plain heap creation.
        let heap: ID3D12DescriptorHeap = match unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: size as u32,
                ..Default::default()
            })
        } {
            Ok(heap) => heap,
            Err(e) => {
                tracing::warn!(error = %e, size, "no heap for the cache of made buffer views: views are made as before");
                return None;
            }
        };
        // SAFETY: a live heap.
        let cpu = unsafe { heap.GetCPUDescriptorHandleForHeapStart() }.ptr;
        tracing::info!(size, "cache of made buffer views (VICTORY_HEAP_SYNC_CACHE)");
        Some(RecordCache { _heap: heap, cpu, step: step as usize, entries: vec![Cached::default(); size], tick: 0 })
    }

    fn at(&self, entry: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.cpu + entry * self.step }
    }

    /// The first of the two entries `key` may be kept in.
    ///
    /// VIcky: the top bits of the record's product with an odd constant,
    /// which every bit of the record moves. It was bits 29 and up, masked:
    /// those move only with the record's low 38 bits, which for a view of
    /// placed memory are its address alone (its shape number lies in bits
    /// 40 and up), so every view at one address, whatever its elements,
    /// fell in one set of two and they put each other out (a ring of views
    /// of one buffer, `a_pass_counts_what_its_slots_came_to`). Which records
    /// are kept changes; what a slot is made of does not.
    fn set(&self, key: u64) -> usize {
        let bits = (self.entries.len() / 2).trailing_zeros();
        ((key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - bits)) as usize) * 2
    }

    /// The entry of `key`, if it is kept.
    fn find(&self, key: u64) -> Option<usize> {
        let set = self.set(key);
        (set..set + 2).find(|&e| self.entries[e].key == key)
    }

    /// The entry to keep `key` in: an empty one of its set, else the older.
    fn room(&self, key: u64) -> usize {
        let set = self.set(key);
        let (a, b) = (self.entries[set], self.entries[set + 1]);
        if a.key == 0 || (b.key != 0 && self.tick.wrapping_sub(a.used) >= self.tick.wrapping_sub(b.used)) {
            set
        } else {
            set + 1
        }
    }

    /// [`make`] of buffer view `view` (record `key`, of the whole buffer
    /// `extent`) at `at`: copied from its entry if the entry was made on
    /// the copies `view` is made on now, else made and kept.
    fn make(
        &mut self,
        view: &View,
        key: u64,
        (base, width): (u64, u64),
        at: D3D12_CPU_DESCRIPTOR_HANDLE,
        lookups: &mut Lookups,
    ) -> Option<Kept> {
        use crate::draw::heap_pass::parts;
        self.tick = self.tick.wrapping_add(1);
        if let Some(e) = self.find(key) {
            let entry = self.entries[e];
            match shown_copy(view, base, width, lookups) {
                Some((whole, shown)) if whole.id() == entry.whole && shown.id() == entry.shown => {
                    if check_now() {
                        // The slow way, with the copies looked up anew.
                        let now = buffer_plan(view, &mut Lookups::each());
                        let same = now.as_ref().is_some_and(|p| p.print() == entry.print);
                        checked(Check::Cache, same, || {
                            format!("record {key:#x}: the cache's view differs from the one made now ({view:?})")
                        });
                        if !same {
                            self.entries[e].key = 0;
                            return now.and_then(|plan| plan.create(at));
                        }
                    }
                    let host = host()?;
                    {
                        let _create = parts::enter(parts::Part::Create);
                        // SAFETY: a slot of a heap shaders do not see, holding a
                        // view of `shown`'s buffer (alive: `shown` holds it), into
                        // a slot of ours of the same kind.
                        unsafe {
                            host.device
                                .CopyDescriptorsSimple(1, at, self.at(e), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                        };
                    }
                    self.entries[e].used = self.tick;
                    FAST.add(Fast::Hits as usize, 1);
                    parts::count(parts::Count::Hit);
                    return Some(Some(shown.buffer.clone()));
                }
                _ => {
                    FAST.add(Fast::Stale as usize, 1);
                    self.entries[e].key = 0;
                }
            }
        }
        // Made as `make` makes it, in the entry first.
        let Some(plan) = buffer_plan(view, lookups) else {
            FAST.add(Fast::Unmade as usize, 1);
            return None;
        };
        let e = self.room(key);
        // (An entry is empty while it is made: a failed making leaves none.)
        self.entries[e].key = 0;
        let kept = {
            let _create = parts::enter(parts::Part::Create);
            let kept = plan.create(self.at(e))?;
            let host = host()?;
            // SAFETY: as above, the view just made.
            unsafe {
                host.device
                    .CopyDescriptorsSimple(1, at, self.at(e), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            };
            kept
        };
        parts::count(parts::Count::Miss);
        self.entries[e] = Cached {
            key,
            whole: plan.whole.id(),
            shown: plan.shadow.id(),
            print: if sync_check() { plan.print() } else { 0 },
            used: self.tick,
        };
        FAST.add(Fast::Misses as usize, 1);
        Some(kept)
    }
}

/// The title memory (address, bytes) of the whole buffer a buffer view is
/// of, if [`RecordCache`] may keep the view: a view of the buffer
/// dimension that [`make`] makes with [`buffer_srv`] or [`buffer_uav`] on
/// that buffer's copy, and of which [`buffer_fields`] reads the fields
/// those two read.
fn cacheable(view: &View) -> Option<(u64, u64)> {
    const BUFFER: u32 = 1;
    match view {
        View::Shader(r, Some(d)) | View::Unordered(r, _, Some(d)) if d[1] == BUFFER => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => {
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            (len != 0).then(|| (u64::from(desc[10]) | u64::from(desc[11]) << 32, len))
        }
        _ => None,
    }
}

/// What [`make`] makes a view [`cacheable`] says it may keep of, as it
/// makes it (the same branches, up to the making itself).
fn buffer_plan(view: &View, lookups: &mut Lookups) -> Option<BufferPlan> {
    const BUFFER: u32 = 1;
    match view {
        View::Shader(r, Some(desc)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            buffer_srv_plan(data as u64, len, srv_desc(desc), lookups)
        }
        View::Unordered(r, _, Some(desc)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            buffer_uav_plan(data as u64, len, uav_desc(desc), lookups)
        }
        View::Placed(unordered, resource, desc) if resource[0] == BUFFER => {
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if len == 0 {
                return None;
            }
            if *unordered {
                buffer_uav_plan(address, len, uav_desc(words), lookups)
            } else {
                buffer_srv_plan(address, len, srv_desc(words), lookups)
            }
        }
        _ => None,
    }
}

/// The copies a view [`cacheable`] says may be kept is made on now: of its
/// whole buffer (`base`, `width`), and the one it shows: a copy of its own
/// bytes where [`own_bytes`] says so and there is one, else the same. As
/// [`buffer_srv`] and [`buffer_uav`] choose them, from the fields
/// [`buffer_fields`] reads as they do.
fn shown_copy(
    view: &View,
    base: u64,
    width: u64,
    lookups: &mut Lookups,
) -> Option<(std::sync::Arc<crate::shadow::Shadow>, std::sync::Arc<crate::shadow::Shadow>)> {
    let whole = lookups.find(base, width)?;
    let (first, count, element, stride, unknown) = buffer_fields(view)?;
    let shown = own_bytes(whole.offset(base), base, first, count, element, stride, unknown)
        .and_then(|(start, bytes)| lookups.exact(start, bytes))
        .unwrap_or_else(|| whole.clone());
    Some((whole, shown))
}

/// Whether `view` is a shader view of a texture the title placed in memory
/// (the only kind [`same_placed_texture`] can be true of).
fn placed_texture_view(view: &View) -> bool {
    matches!(view, View::Placed(false, resource, _) if resource[0] != 1)
}

/// Whether `view` is a buffer view, with a stride, that [`HostHeap::note_buffer`]
/// does not take for a texture's: what it notes of such a view is the same
/// for all of them but the stride. (Tests: the note tells it by
/// [`note_kind::of`], [`note_kind::PLAIN`].)
#[cfg(test)]
fn plain_buffer(view: &View) -> bool {
    view_stride(view).is_some()
        && !matches!(view, View::Shader(r, _) if crate::resource::buffer_bytes(*r as crate::object::This).is_none())
}

/// VIcky: `VICTORY_HEAP_SYNC_CACHE=1` keeps made buffer views by record
/// ([`RecordCache`], 2^18 entries; a number sets them) and lets the making
/// of a slot and the pass over buffer views after it share their lookups
/// ([`Lookups::kept`]). Off (0) by default: nothing changes.
pub(crate) fn cache_size() -> usize {
    #[cfg(test)]
    if let Some((size, _, _)) = TEST_FAST.get() {
        return size;
    }
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| match std::env::var("VICTORY_HEAP_SYNC_CACHE").ok().as_deref() {
        None | Some("" | "0") => 0,
        Some("1") => 1 << 18,
        Some(n) => n.parse().unwrap_or(1 << 18),
    })
}

/// VIcky: `VICTORY_HEAP_BUFFERS_SKIP_SAME=1` skips what a note of a buffer
/// slot would do again ([`HostHeap::note_buffer`]) and the look at the old
/// view of a slot whose new view is no placed texture's
/// ([`HostHeap::mirror`]). Off by default.
pub(crate) fn skip_same() -> bool {
    #[cfg(test)]
    if let Some((_, skip, _)) = TEST_FAST.get() {
        return skip;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_BUFFERS_SKIP_SAME").is_some_and(|v| v == "1"))
}

/// VIcky (diagnostics): `VICTORY_HEAP_SYNC_CHECK=1` does one in
/// [`CHECK_EVERY`] of the work the switches above shortcut the slow way as
/// well, and counts where the two differ (the sync line). A cached view
/// that differs is made the slow way.
pub(crate) fn sync_check() -> bool {
    #[cfg(test)]
    if let Some((_, _, check)) = TEST_FAST.get() {
        return check;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_CHECK").is_some_and(|v| v == "1"))
}

/// One check in this many shortcuts.
pub(crate) const CHECK_EVERY: u64 = 64;

/// Slots [`HostHeap::mirror_due`] mirrors under one lock: some 25 us of
/// them on this machine.
const MIRROR_CHUNK: usize = 64;

/// The slots of the pass over buffer views made under one lock
/// ([`HostHeap::mirror_chunk`]).
pub(crate) const PASS_CHUNK: usize = 64;

/// VIcky (`VICTORY_HEAP_PASS_ONE_LOOK=1`): the marks the pass over buffer
/// views did not set on slots it made ([`HostHeap::note_buffer_marking`]),
/// of slots not marked already: each a look of its next pass spared. Taken
/// by the spread's line (`crate::draw::heap_pass`).
pub(crate) static PASS_MARKS_SPARED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// While [`HostHeap::mirror_due`] mirrors a chunk on this thread: the
    /// host resources its slots let go of, retired together after it.
    static RETIRE_LATER: std::cell::RefCell<Option<Vec<ID3D12Resource>>> = const { std::cell::RefCell::new(None) };
}

thread_local! {
    /// VIcky (`VICTORY_HEAP_KEPT_SPLIT`): while [`HostHeap::mirror_due`]
    /// mirrors a chunk on this thread ([`KeptLater`]), the heap's keeps of
    /// mirrored slots and its marks, made together at the chunk's end.
    static KEPT_LATER: std::cell::RefCell<Option<LaterOf>> = const { std::cell::RefCell::new(None) };
}

/// A change [`HostHeap::later`] may note for the chunk's end.
enum Later {
    /// [`HostHeap::keep`]'s resource for a slot.
    Keep(Option<ID3D12Resource>),
    /// [`HostHeap::mark_changed`] of a slot.
    Mark,
}

/// [`KEPT_LATER`]: the heap (its address), and its changes in order.
struct LaterOf {
    heap: usize,
    keeps: Vec<(u32, Option<ID3D12Resource>)>,
    marks: Vec<u32>,
}

/// VIcky (`VICTORY_HEAP_KEPT_SPLIT`, [`super::kept`]): while it lives, the
/// keeps of `heap`'s mirrored slots and its marks on this thread wait in
/// [`KEPT_LATER`]; when it goes they are made, in their order, the keeps
/// under one lock of the mirrored part and the marks under one lock of
/// `changed`. Made with `made` held alone, and let go before it.
struct KeptLater<'a> {
    heap: &'a HostHeap,
}

impl<'a> KeptLater<'a> {
    /// `None` for a heap whose slots are held in one part.
    fn begin(heap: &'a HostHeap) -> Option<Self> {
        if !heap.kept.split() {
            return None;
        }
        KEPT_LATER.set(Some(LaterOf {
            heap: std::ptr::from_ref(heap) as usize,
            keeps: Vec::new(),
            marks: Vec::new(),
        }));
        Some(KeptLater { heap })
    }
}

impl Drop for KeptLater<'_> {
    fn drop(&mut self) {
        let Some(later) = KEPT_LATER.take() else {
            return;
        };
        self.heap.kept.replace_together(later.keeps, retire_held);
        if !later.marks.is_empty() {
            let mut changed = crate::lot4::locks::mutex(crate::lot4::locks::Lock::Changed, &self.heap.changed);
            changed.extend(later.marks);
        }
    }
}

/// A host resource a slot held and lets go of: retired (`crate::host::retire`)
/// now, or with the others of the chunk [`HostHeap::mirror_due`] is at.
/// A later retirement is no earlier a drop: retired after the chunk, it
/// waits for the submissions started by then, which are as many or more.
fn retire_held(old: ID3D12Resource) {
    let old = RETIRE_LATER.with_borrow_mut(|later| match later {
        Some(held) => {
            held.push(old);
            None
        }
        None => Some(old),
    });
    if let Some(old) = old {
        crate::host::retire(old);
    }
}

/// Whether this shortcut is one to check.
pub(crate) fn check_now() -> bool {
    static TURN: AtomicU64 = AtomicU64::new(0);
    sync_check() && TURN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % CHECK_EVERY == 0
}

#[cfg(test)]
thread_local! {
    /// Tests: the three switches above for this thread (cache size, skip,
    /// check), whatever the environment says.
    pub(crate) static TEST_FAST: std::cell::Cell<Option<(usize, bool, bool)>> = const { std::cell::Cell::new(None) };
}

/// Counters of the switches above, since the last sync line
/// (`super::sync_line`).
#[derive(Clone, Copy)]
pub(crate) enum Fast {
    /// Slots copied from the cache; made and kept; entries whose copies
    /// had changed; views the slow way made nothing of; views the cache
    /// may not keep (made as before).
    Hits,
    Misses,
    Stale,
    Unmade,
    NotCacheable,
    /// Notes of buffer slots skipped.
    NotesSkipped,
}

/// VIcky: kept per thread ([`crate::perthread`]): the heap sync, the pass's
/// threads and the title's threads add here for every slot.
pub(crate) static FAST: crate::perthread::Counters<6> = crate::perthread::Counters::new();

/// What `VICTORY_HEAP_SYNC_CHECK` compares.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Check {
    /// A view copied from the cache against the one made now.
    Cache,
    /// The copy and range the pass over buffer views brings up to date,
    /// with lookups shared, against the ones looked up anew.
    Lookups,
    /// A skipped note against what a full one would have found to do.
    Notes,
}

/// Per [`Check`]: checked, and of them differing, since the last sync line.
pub(crate) static CHECKS: [[AtomicU64; 2]; 3] = [const { [const { AtomicU64::new(0) }; 2] }; 3];

/// Counts a check of `what`; one that differs is logged (the first few).
pub(crate) fn checked(what: Check, same: bool, describe: impl FnOnce() -> String) {
    use std::sync::atomic::Ordering::Relaxed;
    let counts = &CHECKS[what as usize];
    counts[0].fetch_add(1, Relaxed);
    if !same {
        counts[1].fetch_add(1, Relaxed);
        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if SAID.fetch_add(1, Relaxed) < 32 {
            tracing::warn!(?what, "VICTORY_HEAP_SYNC_CHECK: the fast way differs from the slow way: {}", describe());
        }
    }
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// Whether the pass over records copied in as bytes also mirrors slots
/// that draws read through tables lately (`VICTORY_HEAP_SYNC_TABLES=1`;
/// off by default: [`HostHeap::mirror_copied`]).
fn sync_tables() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_TABLES").is_some_and(|v| v == "1"))
}

/// Since the last frame line: slots on changed heap pages that
/// [`HostHeap::mirror_copied`] left to their draws (tables' slots).
pub(crate) static COPIED_TABLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A 64-bit fingerprint of a descriptor record's bytes (16 or 32 of them):
/// equal records give equal prints, and two different records the same
/// print about once in 2^64. Not a hash anyone can attack: the records
/// are our own.
fn record_print(record: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut h = K ^ record.len() as u64;
    let mut words = record.chunks_exact(8);
    for word in &mut words {
        let w = u64::from_le_bytes(word.try_into().unwrap_or([0; 8]));
        h = (h ^ w).wrapping_mul(K);
        h ^= h >> 29;
    }
    for &b in words.remainder() {
        h = (h ^ u64::from(b)).wrapping_mul(K);
    }
    h ^ (h >> 32)
}

/// The first three words of a descriptor record's bytes (tag, view index,
/// generation); zeros for what is missing.
fn words_of(record: &[u8]) -> [u32; 3] {
    let word = |w: usize| u32::from_le_bytes([0, 1, 2, 3].map(|b| record.get(w * 4 + b).copied().unwrap_or(0)));
    [word(0), word(1), word(2)]
}

/// The key a host slot made from the record `words` holds (generation and
/// index), 0 for a record of no view of ours.
fn key_of_words([tag, i, generation]: [u32; 3]) -> u64 {
    if tag == VIEW_TAG { u64::from(generation) << 32 | u64::from(i) } else { 0 }
}

/// Whether a buffer view off its buffer's elements gets a copy of its own
/// bytes (`VICTORY_VIEW_OWN_COPY=0`: it is refused and reads zeros, as
/// before).
fn own_copy_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_OWN_COPY").is_none_or(|v| v != "0"))
}

/// The title memory (start, bytes) of a buffer view that cannot be made on
/// the host copy of its buffer: `count` elements of `element` bytes from
/// element `first` of the buffer at `base`, which lies `offset` bytes into
/// that copy, where `offset` is not a whole number of elements and the view
/// cannot be a raw view of the same bytes either ([`raw_window`]:
/// `unknown`, the view has no format, and `stride` is its structure's).
/// The host counts a view's first element from its buffer's start, so such
/// a view has no host view there; it is made on a copy of its own bytes
/// (`crate::shadow::exact`), where it starts at element 0. `None`: the
/// view is made on its buffer's copy, as every other.
fn own_bytes(offset: u64, base: u64, first: u64, count: u32, element: u64, stride: u32, unknown: bool) -> Option<(u64, u64)> {
    own_bytes_if(own_copy_on(), offset, base, first, count, element, stride, unknown)
}

#[allow(clippy::too_many_arguments)]
fn own_bytes_if(
    on: bool,
    offset: u64,
    base: u64,
    first: u64,
    count: u32,
    element: u64,
    stride: u32,
    unknown: bool,
) -> Option<(u64, u64)> {
    if !on || element == 0 || count == 0 || offset.is_multiple_of(element) {
        return None;
    }
    if unknown && raw_window(offset, first, count, stride).is_some() {
        return None;
    }
    let start = base.checked_add(first.checked_mul(element)?)?;
    Some((start, u64::from(count) * element))
}

/// Of a buffer view: its first element, its elements, the bytes of one (a
/// raw view's: 4), its structure's stride, and whether it has no format.
/// As [`buffer_srv`] and [`buffer_uav`] read them from the same words.
fn buffer_fields(view: &View) -> Option<(u64, u32, u64, u32, bool)> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => (d[..10].try_into().ok()?, *unordered),
        _ => return None,
    };
    if unordered {
        let d = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(d.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        Some((b.FirstElement, b.NumElements, element, b.StructureByteStride, d.Format == DXGI_FORMAT_UNKNOWN))
    } else {
        let d = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        Some((b.FirstElement, b.NumElements, element_size(&d), b.StructureByteStride, d.Format == DXGI_FORMAT_UNKNOWN))
    }
}

/// A structured buffer view (`first`, `count` elements of `stride` bytes,
/// `offset` bytes into its host copy) as a raw view of the same bytes:
/// its first dword and its dwords. Translated shaders read untyped buffers
/// as raw bytes at `element * stride + offset` (the stride from the
/// used-resource entry, or from the table by heap slot for a buffer picked
/// by index), and a raw read through a structured view gives zeros on
/// NVIDIA (reported upstream with PR #193; structured reads through raw
/// views were right there). So the host's view is raw whenever it can be:
/// the bytes must start on 16 (the host's raw view alignment) and be whole
/// dwords. A view that cannot stays structured.
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 {
        return None;
    }
    let start = first.checked_mul(u64::from(stride))?.checked_add(offset)?;
    let bytes = u64::from(count) * u64::from(stride);
    (start % 16 == 0 && bytes % 4 == 0)
        .then(|| u32::try_from(bytes / 4).ok().map(|dwords| (start / 4, dwords)))
        .flatten()
}

/// Whether the host accepts a buffer view with these fields of a buffer of
/// `width` bytes. It removes the device for one it does not (view creation
/// returns nothing), and the console is laxer: the title's heap holds
/// views past their memory's end, of no elements, of typeless formats
/// without the raw flag, with strides the host has no structured buffers
/// for. Since every buffer view of the heap is made for shaders that pick
/// them by index, one such view ended the run (2026-11-24, run 10).
fn buffer_view_ok(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> bool {
    use crate::resource::family;
    let f = format.0 as u32;
    let element = if raw {
        // raw: R32 typeless, counted in dwords, starting on 16 bytes
        if format != DXGI_FORMAT_R32_TYPELESS || stride != 0 || !first.is_multiple_of(4) {
            return false;
        }
        4
    } else if stride > 0 {
        // structured: no format, a stride the host takes
        if format != DXGI_FORMAT_UNKNOWN || stride > 2048 || !stride.is_multiple_of(4) {
            return false;
        }
        u64::from(stride)
    } else {
        // typed: a format with a size that is not typeless
        let typeless = family(f) == f && family(f + 1) == f;
        match crate::resource::element(f) {
            Some((bytes, 1)) if f != 0 && !typeless && bytes > 0 => bytes,
            _ => return false,
        }
    };
    count > 0
        && first
            .checked_add(u64::from(count))
            .and_then(|end| end.checked_mul(element))
            .is_some_and(|end| end <= width)
}

/// What [`buffer_srv`] and [`buffer_uav`] make a buffer view's host slot
/// of: the host copy it shows, the copy of the whole buffer that one was
/// chosen by (the same, unless the view is made on a copy of its own
/// bytes), and the view's description on it.
struct BufferPlan {
    whole: std::sync::Arc<crate::shadow::Shadow>,
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    desc: PlanDesc,
}

enum PlanDesc {
    Shader(D3D12_SHADER_RESOURCE_VIEW_DESC),
    Unordered(D3D12_UNORDERED_ACCESS_VIEW_DESC),
}

impl BufferPlan {
    /// Makes the view at `at`; what the slot then holds.
    fn create(&self, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
        let host = host()?;
        // SAFETY: a live buffer and a slot of a heap of ours.
        unsafe {
            match &self.desc {
                PlanDesc::Shader(desc) => host.device.CreateShaderResourceView(&self.shadow.buffer, Some(desc), at),
                PlanDesc::Unordered(desc) => host.device.CreateUnorderedAccessView(
                    &self.shadow.buffer,
                    None::<&ID3D12Resource>,
                    Some(desc),
                    at,
                ),
            }
        };
        Some(Some(self.shadow.buffer.clone()))
    }

    /// A print of what [`BufferPlan::create`] makes: the buffer, and the
    /// description's fields as a buffer view's (for
    /// `VICTORY_HEAP_SYNC_CHECK`; the cache keeps views of the buffer
    /// dimension alone, [`cacheable`]). Field by field: the descriptions
    /// have padding, which holds whatever was there.
    fn print(&self) -> u64 {
        let buffer = windows::core::Interface::as_raw(&self.shadow.buffer) as u64;
        // SAFETY: the buffer member of the union, which these views use.
        let words: [u64; 9] = unsafe {
            match &self.desc {
                PlanDesc::Shader(d) => {
                    let b = d.Anonymous.Buffer;
                    let fields = [d.Format.0 as u32, d.ViewDimension.0 as u32, d.Shader4ComponentMapping];
                    [0, buffer, u64::from(fields[0]), u64::from(fields[1]), u64::from(fields[2]), b.FirstElement, u64::from(b.NumElements), u64::from(b.StructureByteStride), u64::from(b.Flags.0 as u32)]
                }
                PlanDesc::Unordered(d) => {
                    let b = d.Anonymous.Buffer;
                    let fields = [d.Format.0 as u32, d.ViewDimension.0 as u32, b.CounterOffsetInBytes as u32];
                    [1, buffer, u64::from(fields[0]), u64::from(fields[1]), u64::from(fields[2]), b.FirstElement, u64::from(b.NumElements), u64::from(b.StructureByteStride), u64::from(b.Flags.0 as u32)]
                }
            }
        };
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        record_print(&bytes)
    }
}

fn buffer_srv(
    address: u64,
    len: u64,
    desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    lookups: &mut Lookups,
) -> Option<Kept> {
    buffer_srv_plan(address, len, desc, lookups)?.create(at)
}

/// [`buffer_srv`] up to the making of the view.
fn buffer_srv_plan(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    lookups: &mut Lookups,
) -> Option<BufferPlan> {
    let shadow = lookups.find(address, len)?;
    let whole = shadow.clone();
    let offset = shadow.offset(address);
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, offset) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        match own_bytes(offset, address, b.FirstElement, b.NumElements, element_size(&desc), b.StructureByteStride, unknown)
            .and_then(|(start, bytes)| lookups.exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, 0)
            }
            None => (shadow, offset),
        }
    };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    // SAFETY: a buffer view's union member.
    let structured = unsafe { desc.Anonymous.Buffer };
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, structured.FirstElement, structured.NumElements, structured.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
    }
    // (the same element size: the offsets below hold)
    desc.Format = typed_view_format(desc.Format);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if window.is_none() {
        if element == 0 || offset % element != 0 {
            once!(
                offset,
                element,
                "buffer view not on an element of its host copy"
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "buffer view the host would refuse left null"
        );
        return None;
    }
    // The console has no component mapping; one the host does not know is the default's.
    if desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING {
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    }
    Some(BufferPlan { whole, shadow, desc: PlanDesc::Shader(desc) })
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    lookups: &mut Lookups,
) -> Option<Kept> {
    buffer_uav_plan(address, len, desc, lookups)?.create(at)
}

/// [`buffer_uav`] up to the making of the view.
fn buffer_uav_plan(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    lookups: &mut Lookups,
) -> Option<BufferPlan> {
    let shadow = lookups.find(address, len)?;
    let whole = shadow.clone();
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, own) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        match own_bytes(shadow.offset(address), address, b.FirstElement, b.NumElements, element, b.StructureByteStride, unknown)
            .and_then(|(start, bytes)| lookups.exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, true)
            }
            None => (shadow, false),
        }
    };
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = if own { 0 } else { shadow.offset(address) };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        if element == 0 || offset % element != 0 {
            once!(offset, element, "unordered buffer view not on an element of its host copy");
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    buffer.CounterOffsetInBytes = 0;
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    // (the same element size: the first element above holds)
    desc.Format = typed_view_format(desc.Format);
    if window.is_some() {
        // (Left out at first: every structured unordered access view was then
        // refused by `buffer_view_ok` and left null, runs 15 and 16.)
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    }
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "unordered buffer view the host would refuse left null"
        );
        return None;
    }
    Some(BufferPlan { whole, shadow, desc: PlanDesc::Unordered(desc) })
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        if unordered {
            crate::resource::DEPTH_UNORDERED[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    // A view of a target whose format the host refuses, in the title's
    // format: in the stand-in the host copy has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// The title's unordered access view of a depth texture, as a view of the
/// texture's colour twin `twin` ([`crate::resource::Twin`]): the same mip
/// and slices, in the colour format of the twin's family for a depth or
/// typeless format (R16_UNORM for 16-bit depth, R32_FLOAT for 32-bit; the
/// bits are the same). Without a description: mip 0 of every slice.
/// The twin of a stencil plane (R8) stands for the views of that plane
/// only, which become R8_UINT views of its one plane; `None` for a view of
/// the depth plane, which the twin does not stand for. The depth plane has
/// a twin of its own, viewed as a depth texture's without stencil: a view
/// in the format of the depth plane of a depth-stencil format (21, 46)
/// takes the twin's colour format too.
fn twin_uav(
    twin: &ID3D12Resource,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { twin.GetDesc() };
    let mut desc = desc.cloned().unwrap_or_else(|| {
        let mut all = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        if d.DepthOrArraySize > 1 {
            all.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            all.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: u32::from(d.DepthOrArraySize),
                ..Default::default()
            };
        }
        all
    });
    // A stencil plane's twin (R8 typeless, 60): the title's view is of the
    // stencil plane if it names plane 1 or a stencil view format (22, 47).
    if d.Format.0 == 60 {
        let mut plane = 0;
        match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2D };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2D = view;
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2DArray };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2DArray = view;
            }
            _ => {}
        }
        if plane != 1 && !matches!(desc.Format.0, 22 | 47) {
            return None;
        }
        desc.Format.0 = 62;
        return Some(desc);
    }
    // None, typeless and depth formats (R16: 0, 53, 55; R32: 0, 39, 40),
    // and those of a depth-stencil format's depth plane (19 to 21, 44 to
    // 46), for the twin of such a plane.
    desc.Format.0 = match (desc.Format.0, d.Format.0) {
        (0 | 53 | 55 | 19..=21 | 44..=46, 53) => 56,
        (0 | 39 | 40 | 19..=21 | 44..=46, 39) => 41,
        (other, _) => other,
    };
    Some(desc)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    make_with(view, at, &mut Lookups::each())
}

/// [`make_with`] for a slot [`HostHeap::make_slot`] makes without the cache
/// of made views: timed and counted for the pass over buffer views
/// (`crate::draw::heap_pass::parts`).
fn made_otherwise(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE, lookups: &mut Lookups) -> Option<Kept> {
    use crate::draw::heap_pass::parts;
    let kept = {
        let _create = parts::enter(parts::Part::Create);
        make_with(view, at, lookups)
    };
    if kept.is_some() {
        parts::count(parts::Count::Other);
    }
    kept
}

/// [`make`], asking `lookups` for the host copies of buffers.
fn make_with(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE, lookups: &mut Lookups) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        View::NullConstant => {
            null_for(view, at);
            Some(None)
        }
        &View::Constant(address, size) => {
            let shadow = lookups.find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // The title's views of textures with a stencil plane, as it
                // describes them (format, dimension, component mapping, a
                // pad, then first level, levels, plane): each distinct one
                // once. They say how its shaders read the stencil plane
                // (the lighting takes each pixel's material class and its
                // "inside a room" bit from it).
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() }.Format.0;
                    if matches!(own, 19..=22 | 44..=47) {
                        static SEEN: Mutex<Vec<(i32, Option<[u32; 10]>)>> = Mutex::new(Vec::new());
                        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                        if seen.len() < 64 && !seen.contains(&(own, *desc)) {
                            seen.push((own, *desc));
                            tracing::info!(
                                ?resource,
                                texture_format = own,
                                view = ?desc,
                                "shader view of a depth-stencil texture"
                            );
                        }
                    }
                }
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // Levels on demand (`crate::tiles`): a texture made level by
                // level is shown from the first level that is the title's
                // and has memory, if the title's view starts before it (a
                // level without memory reads zeros: a black texture). The
                // title's own view mostly starts there already.
                if let Some(first) = crate::resource::first_shown(resource)
                    && first > 0
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() };
                    // The texture's own view (the title gave none): all of it.
                    let whole = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: own.Format,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: 5768,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MostDetailedMip: 0,
                                MipLevels: u32::MAX,
                                PlaneSlice: 0,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    };
                    crate::tiles::show_from(desc.get_or_insert(whole), first, u32::from(own.MipLevels));
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at, lookups)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at, lookups)
                    }
                    _ => None,
                };
            };
            // The first such views of depth textures are logged as the
            // title describes them (format, dimension, then mip, slices and
            // plane): they say which planes shaders write, and as what.
            if crate::resource::depth_and_unordered(resource) {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 512 {
                    tracing::info!(
                        ?resource,
                        view = ?desc,
                        "unordered access view of a depth texture"
                    );
                }
            }
            let mut desc = desc.as_ref().map(uav_desc);
            // A depth texture's unordered access is its colour twin's: the
            // host gives a depth texture none.
            let twin = crate::resource::twin_resource(resource)
                .and_then(|twin| Some((twin, twin_uav(twin, desc.as_ref())?)))
                // A view of the depth plane of a texture with a stencil
                // plane: of that plane's own twin.
                .or_else(|| {
                    let twin = crate::resource::twin_plane_resource(resource)?;
                    Some((twin, twin_uav(twin, desc.as_ref())?))
                });
            let texture = match twin {
                Some((twin, view)) => {
                    desc = Some(view);
                    twin
                }
                None => texture,
            };
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make_with(&view, at, lookups);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            // An empty title buffer: nothing to view, and no host copy to
            // look for (#70: such views were in flight at device removals).
            if len == 0 {
                return None;
            }
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at, lookups)
            } else {
                buffer_srv(address, len, srv_desc(words), at, lookups)
            }
        }
    }
}

/// Writes of a slot with another view of the placed texture it showed,
/// since the start: the slot kept showing the old view.
pub(crate) static KEPT_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a slot written with another view of the placed texture it
/// shows keeps showing the old one until the new is made;
/// `VICTORY_KEEP_VIEWS=0` makes it null at once, as before.
fn keep_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_VIEWS").is_none_or(|v| v != "0"))
}

/// Whether both are shader views of one texture the title placed in
/// memory: the same address and the same resource description, whatever
/// part of it each shows.
fn same_placed_texture(a: &View, b: &View) -> bool {
    match (a, b) {
        (View::Placed(false, resource, desc), View::Placed(false, other, other_desc)) => {
            resource[0] != 1 && resource == other && desc[10..12] == other_desc[10..12]
        }
        _ => false,
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::NullConstant | View::Constant(..) => {
            // SAFETY: a live host heap slot; a null description creates an unbound CBV.
            return unsafe { host.device.CreateConstantBufferView(None::<*const D3D12_CONSTANT_BUFFER_VIEW_DESC>, at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// Where a raw buffer view starts and how many dwords it covers, both inside
/// `size` bytes of buffer: `None` when the request has nothing in the buffer.
///
/// `FirstElement` is 64-bit and `NumElements` 32-bit, so only the length is cut
/// at `u32::MAX`; the start is used as asked and never moved. Moving it is what
/// a clamp would do, and clamping the start of a request that lies entirely past
/// the buffer turns it into a view of the buffer's tail: 100 bytes of buffer
/// with an offset of 112 used to view bytes 96 to 99 as if they had been asked
/// for. A start at or past the last dword is a view of nothing instead.
///
/// Callers pass `offset` on 16 bytes: `draw.rs` biases by `offset & 15` on the
/// UAV path and makes a null view for a root SRV that is not on 16, so a raw
/// view's start stays on that boundary. An offset off it is refused here too.
#[derive(Clone, Copy)]
struct RawRange {
    first: u64,
    elements: u32,
}

/// Why a request lost part of its range, for the counters below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cut {
    /// The start is at or past the buffer's last dword: nothing to view.
    Past,
    /// The length was cut to what the buffer holds.
    Length,
    /// The title named less than a dword: one element, as before this guard.
    Short,
    /// The start is not on the 16 bytes a raw view needs.
    Unaligned,
}

fn raw_range(size: u64, offset: u64, len: u64) -> (Option<RawRange>, Option<Cut>) {
    let words = size / 4;
    if !offset.is_multiple_of(16) {
        return (None, Some(Cut::Unaligned));
    }
    let asked = offset / 4;
    if asked >= words {
        return (None, Some(Cut::Past));
    }
    let avail = words - asked;
    // A length under a dword made one element before this guard. The floor is
    // kept because nothing was gathered about why the title would ask for less
    // than a dword; the buffer's end still caps it.
    let want = (len / 4).max(1);
    let elements = want.min(avail).min(u64::from(u32::MAX));
    let cut = if elements < want {
        Some(Cut::Length)
    } else if len < 4 {
        Some(Cut::Short)
    } else {
        None
    };
    (
        Some(RawRange {
            first: asked,
            elements: elements as u32,
        }),
        cut,
    )
}

/// Whether a raw buffer view's range is kept inside its buffer
/// ([`raw_range`]; on unless `VICTORY_RAW_VIEW_CLAMP=0`, which views the
/// range as asked, as before). From fxceless/victory-fork pull request
/// 1266 by lichenlang89 (dd5e8c9e), with the switch as foxhound 9f8838b1
/// carries it.
fn raw_view_clamp() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RAW_VIEW_CLAMP").is_none_or(|v| v != "0"))
}

/// The range a raw view of `buffer` from `offset`, `len` bytes, gets
/// ([`raw_range`], what it cut counted as `kind`); `None`: a null view.
fn raw_view_range(buffer: &ID3D12Resource, offset: u64, len: u64, kind: &'static str) -> Option<RawRange> {
    if !raw_view_clamp() {
        return Some(RawRange { first: offset / 4, elements: (len / 4).clamp(1, u64::from(u32::MAX)) as u32 });
    }
    // SAFETY: a live buffer.
    let size = unsafe { buffer.GetDesc() }.Width;
    let (range, cut) = raw_range(size, offset, len);
    if let Some(cut) = cut {
        raw_range_count(kind, cut, offset, len, size);
    }
    range
}

/// Counts what [`raw_range`] refused or cut, by kind (SRV, UAV) and cause, so
/// that a run can say whether the guard applied at all. One atomic per cell,
/// said on the first and every 4096th: no lock, nothing per view beyond that.
fn raw_range_count(kind: &'static str, cut: Cut, offset: u64, len: u64, size: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SRV: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    static UAV: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    let cell = match (kind, cut) {
        ("SRV", Cut::Past) => &SRV[0],
        ("SRV", Cut::Length) => &SRV[1],
        ("SRV", Cut::Short) => &SRV[2],
        ("SRV", Cut::Unaligned) => &SRV[3],
        (_, Cut::Past) => &UAV[0],
        (_, Cut::Length) => &UAV[1],
        (_, Cut::Short) => &UAV[2],
        (_, Cut::Unaligned) => &UAV[3],
    };
    let n = cell.fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n % 4096 == 0 {
        tracing::warn!(
            kind,
            cut = ?cut,
            offset,
            len,
            size,
            seen = n,
            "a raw buffer view the title's range did not cover"
        );
    }
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `VICTORY_TYPED_BIT_EXACT`: typed buffer views of 32-bit integer
    /// formats in the float format of the same width (the same element
    /// size); every other format is the title's.
    #[test]
    fn typed_buffer_views_of_32_bit_integers_move_bits() {
        for (title, host) in [
            (DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_FLOAT),
            (DXGI_FORMAT_R32G32B32_SINT, DXGI_FORMAT_R32G32B32_FLOAT),
            (DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_FLOAT),
        ] {
            assert_eq!(bit_exact_format(title), host);
            assert_eq!(crate::resource::element(title.0 as u32), crate::resource::element(host.0 as u32));
        }
        for kept in [
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32_TYPELESS,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_R16G16_UINT,
            DXGI_FORMAT_R8G8B8A8_UNORM,
        ] {
            assert_eq!(bit_exact_format(kept), kept);
        }
    }

    /// The unsigned twin of a 16-bit float or signed format has its layout,
    /// and only buffer views in such a format are shown as another view.
    #[test]
    fn only_16_bit_float_and_signed_buffer_views_get_an_unsigned_twin() {
        for (format, unsigned) in [(10, 12), (14, 12), (34, 36), (38, 36), (54, 57), (59, 57)] {
            assert_eq!(half_format(format), Some(unsigned));
            assert_eq!(
                crate::resource::element(format),
                crate::resource::element(unsigned),
                "format {format}"
            );
        }
        // The unsigned formats themselves, normalised 16-bit formats, and
        // wider and narrower ones.
        for other in [0, 12, 36, 57, 11, 13, 35, 37, 56, 58, 42, 17, 30, 61] {
            assert_eq!(half_format(other), None, "format {other}");
        }
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        // R16G16B16A16_FLOAT: a buffer of halves a program moves as they are.
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(10, BUFFER)))),
            Some(View::Unordered(7, 0, Some(words(12, BUFFER))))
        );
        assert_eq!(
            half_view(&View::Shader(7, Some(words(54, BUFFER)))),
            Some(View::Shader(7, Some(words(57, BUFFER))))
        );
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(38, BUFFER));
        placed[10] = 0x1234_0000;
        let mut twin = placed;
        twin[0] = 36;
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(
            half_view(&View::Placed(true, memory, placed)),
            Some(View::Placed(true, memory, twin))
        );
        // The title's kind (R16G16B16A16_UINT and R16_UINT buffers), a
        // texture view (dimension 4: 2D), a view without a description, a
        // constant buffer: the title's own.
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(12, BUFFER)))),
            None
        );
        assert_eq!(half_view(&View::Shader(7, Some(words(57, BUFFER)))), None);
        assert_eq!(half_view(&View::Unordered(7, 0, Some(words(10, 4)))), None);
        assert_eq!(half_view(&View::Shader(7, None)), None);
        assert_eq!(half_view(&View::Constant(0x1000, 256)), None);
    }

    /// The buffer views through which the integers a shader moves halves
    /// as are not what the console gives are named; the title's are not.
    #[test]
    fn views_whose_values_the_console_would_convert_are_named() {
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        for exact in [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62] {
            let view = View::Unordered(7, 0, Some(words(exact, BUFFER)));
            assert_eq!(half_inexact(&view), None, "format {exact}");
        }
        // R32_FLOAT, R32_UINT, R16G16B16A16_UNORM, R8G8B8A8_UNORM, R8_SINT,
        // and a structured or raw view (no format).
        for inexact in [41, 42, 11, 28, 64, 0] {
            let view = View::Shader(7, Some(words(inexact, BUFFER)));
            assert_eq!(half_inexact(&view), Some(inexact));
        }
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(41, BUFFER));
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(half_inexact(&View::Placed(true, memory, placed)), Some(41));
        // Not a buffer view.
        assert_eq!(half_inexact(&View::Shader(7, Some(words(41, 4)))), None);
        assert_eq!(half_inexact(&View::Constant(0x1000, 256)), None);
    }

    /// The two ways a translated `buffer_load_format_d16_xyzw` and
    /// `buffer_store_format_d16_xyzw` have moved a typed buffer's 16-bit
    /// channels, on the host device, over a buffer that holds every 16-bit
    /// pattern. As unsigned integers through an unsigned view (the title's
    /// own R16G16B16A16_UINT, or the twin `half_view` gives a float one)
    /// every pattern comes back as it was. By way of floats, as before
    /// (load as floats, each to its half's bits; each half widened and
    /// stored as a float): through the title's integer view that is
    /// outside what D3D12 defines, and what the device gives is printed,
    /// not checked (1 pattern of 65,536 on the cards tried: the network
    /// that moves the title's faces computed zeros).
    #[test]
    fn halves_cross_an_unsigned_buffer_view_unchanged() {
        const MOVES: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
#ifdef BY_FLOATS
RWBuffer<float4> from : register(u0);
RWBuffer<float4> to : register(u1);
#else
RWBuffer<uint4> from : register(u0);
RWBuffer<uint4> to : register(u1);
#endif
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void BY_FLOATS_OR_NOT(uint3 id : SV_DispatchThreadID) {
#ifdef BY_FLOATS
    uint4 h = f32tof16(from[id.x]);
#else
    uint4 h = from[id.x] & 0xffff;
#endif
    uint low = h.x | h.y << 16;
    uint high = h.z | h.w << 16;
#ifdef BY_FLOATS
    to[id.x] = float4(f16tof32(low), f16tof32(low >> 16), f16tof32(high), f16tof32(high >> 16));
#else
    to[id.x] = uint4(low & 0xffff, low >> 16, high & 0xffff, high >> 16);
#endif
}
"#;
        const ELEMENTS: usize = 1 << 14;
        const BYTES: u64 = ELEMENTS as u64 * 8;
        const TITLES: u32 = 12; // R16G16B16A16_UINT
        let Some(host) = host() else {
            return; // no host GPU
        };
        let heap: ID3D12DescriptorHeap = {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain heap creation.
            unsafe { host.device.CreateDescriptorHeap(&desc) }.expect("a view heap")
        };
        // How many of the 65,536 patterns come back as they were, moved by
        // the program `by_floats` or not, through views in `format`.
        let moved = |by_floats: bool, format: u32| -> usize {
            let source = if by_floats {
                MOVES
                    .replace("BY_FLOATS_OR_NOT", "main")
                    .replace("#ifdef BY_FLOATS", "#if 1")
            } else {
                MOVES
                    .replace("BY_FLOATS_OR_NOT", "main")
                    .replace("#ifdef BY_FLOATS", "#if 0")
            };
            let code = crate::host::compile(
                &source,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
            )
            .expect("the test's shader");
            let (from, filled) = host.shared_buffer(BYTES).expect("a buffer");
            let (to, result) = host.shared_buffer(BYTES).expect("a buffer");
            // SAFETY: the shader's own root signature and plain creation
            // calls; both mappings hold `BYTES` bytes; the views are of
            // live buffers, in slots of the heap made above; an open list.
            unsafe {
                let root: ID3D12RootSignature = host
                    .device
                    .CreateRootSignature(0, &code)
                    .expect("a root signature");
                let pso: ID3D12PipelineState = host
                    .device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .expect("a pipeline");
                let filled = std::slice::from_raw_parts_mut(filled.cast::<u16>(), ELEMENTS * 4);
                for (pattern, half) in filled.iter_mut().enumerate() {
                    *half = pattern as u16;
                }
                std::ptr::write_bytes(result, 0xee, BYTES as usize);
                let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT(format as i32),
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            NumElements: ELEMENTS as u32,
                            ..Default::default()
                        },
                    },
                };
                let start = heap.GetCPUDescriptorHandleForHeapStart();
                let step = host
                    .device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                    as usize;
                host.device.CreateUnorderedAccessView(
                    &from,
                    None::<&ID3D12Resource>,
                    Some(&view),
                    start,
                );
                let second = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start.ptr + step,
                };
                host.device.CreateUnorderedAccessView(
                    &to,
                    None::<&ID3D12Resource>,
                    Some(&view),
                    second,
                );
                let list = host.new_list().expect("a host list");
                list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list
                    .SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
                list.list.Dispatch(ELEMENTS as u32 / 64, 1, 1);
                list.close();
                host.execute(&[&list]);
                assert!(
                    host.device.GetDeviceRemovedReason().is_ok(),
                    "the device, after views in format {format}"
                );
                let result = std::slice::from_raw_parts(result.cast::<u16>(), ELEMENTS * 4);
                (0..=u16::MAX)
                    .filter(|&p| result[usize::from(p)] == p)
                    .count()
            }
        };
        // A float view of halves is shown as the title's kind of view.
        assert_eq!(half_format(10), Some(TITLES));
        let unsigned = moved(false, TITLES);
        let by_floats = moved(true, TITLES);
        eprintln!(
            "halves through an R16G16B16A16_UINT buffer view: {unsigned} of 65536 patterns came back unchanged moved as unsigned integers, {by_floats} moved by way of floats"
        );
        assert_eq!(
            unsigned, 65_536,
            "patterns unchanged, moved as unsigned integers"
        );
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// What the texture code asks of the spans (levels on demand): where
    /// the next texture starts inside a span, the lowest level a view of a
    /// texture shows, and a texture's slots.
    #[test]
    fn spans_say_the_next_start_and_the_lowest_level_viewed() {
        let mut spans = Spans::default();
        // A 512x512 texture of 0x56000 bytes viewed from levels 3 and 1,
        // and the next one 0x6000 after it (run 188).
        spans.note(1, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(1, 3);
        spans.note(2, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(2, 1);
        spans.note(3, 0x1_6000, 0x1_6000 + 0x5_6000, true);
        assert_eq!(spans.next_start(0x1_0000, 0x1_0000 + 0x5_6000), Some(0x1_6000));
        assert_eq!(spans.next_start(0x1_6000, 0x1_6000 + 0x5_6000), None);
        assert_eq!(spans.first_viewed(0x1_0000), Some(1));
        assert_eq!(spans.first_viewed(0x1_6000), None);
        assert_eq!(spans.slots_at(0x1_0000), vec![1, 2]);
        // The view from level 1 goes: the texture is viewed from level 3.
        spans.forget(2);
        assert_eq!(spans.first_viewed(0x1_0000), Some(3));
    }

    /// Views of one placed texture are told from views of others by the
    /// address and the resource description, not by what they show.
    #[test]
    fn views_of_one_placed_texture_are_told_by_address_and_description() {
        let texture = [3, 0, 0, 0, 512, 0, 512, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |resource: [u32; 14], first_mip: u32, address: u32| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[4], desc[5], desc[10], desc[11]) = (98, 4, first_mip, 8 - first_mip, address, 0xb);
            View::Placed(false, resource, desc)
        };
        let low = view(texture, 3, 0x1000_0000);
        // more mips of the same texture
        assert!(same_placed_texture(&low, &view(texture, 0, 0x1000_0000)));
        // another address, another description, a buffer, an unordered view
        assert!(!same_placed_texture(&low, &view(texture, 3, 0x2000_0000)));
        let mut wider = texture;
        wider[4] = 1024;
        assert!(!same_placed_texture(&low, &view(wider, 3, 0x1000_0000)));
        let mut buffer = texture;
        buffer[0] = 1;
        assert!(!same_placed_texture(&view(buffer, 3, 0x1000_0000), &view(buffer, 3, 0x1000_0000)));
        let View::Placed(_, resource, desc) = low.clone() else { unreachable!() };
        assert!(!same_placed_texture(&low, &View::Placed(true, resource, desc)));
        assert!(!same_placed_texture(&low, &View::Shader(0, None)));
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn a_structured_view_made_raw_passes_the_host_check() {
        // 100 elements of 20 bytes from element 8, in a copy that starts 32
        // bytes before the view's memory: bytes 192..2192.
        let (first, count) = raw_window(32, 8, 100, 20).unwrap();
        assert_eq!((first, count), (48, 500));
        assert!(buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 4096));
        // the format a raw view must name; the structured one's is refused
        assert!(!buffer_view_ok(DXGI_FORMAT_UNKNOWN, first, count, 0, true, 4096));
        // past the buffer, not on 16 bytes, no elements
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 2000));
        assert!(raw_window(4, 0, 100, 20).is_none());
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, true, 4096));
        // a structured view the host takes as it is
        assert!(buffer_view_ok(DXGI_FORMAT_UNKNOWN, 1, 10, 12, false, 4096));
    }

    /// The queue of textures picked by index serves the slots looked at
    /// before first, then the newest write and the oldest in turns; a
    /// slot whose memory is not there is looked at again after
    /// 1, 2, 4 frames up to the limit; and what the pass reports about a
    /// slot written again meanwhile is dropped.
    #[test]
    fn the_texture_queue_serves_the_newest_writes_first() {
        let mut q = TextureQueue::default();
        // Slots 10..18 written in frames 1..9, slot 5 last.
        for i in 0..8u32 {
            q.written(10 + i, u64::from(i) + 1, 0, false);
        }
        q.written(5, 9, 0, false);
        let order: Vec<u32> = q.due(9, 100).into_iter().map(|d| d.0).collect();
        // newest, oldest, newest, oldest, and so on
        assert_eq!(order, [5, 10, 17, 11, 16, 12, 15, 13, 14]);
        assert_eq!(q.due(9, 2).len(), 2);
        assert_eq!(q.census(9), [9, 9, 0, 0, 8]);
        // Slot 17's memory is not mapped: again next frame, then in 2, 4, 4.
        let write = |q: &TextureQueue, index: u32| q.due(u64::MAX, 100).iter().find(|d| d.0 == index).unwrap().1;
        let w17 = write(&q, 17);
        let mut frame = 9;
        for wait in [1, 2, 4, 4] {
            q.later(17, w17, frame, 4);
            assert!(!q.due(frame + wait - 1, 100).iter().any(|d| d.0 == 17));
            frame += wait;
            // Due again, and ahead of every slot never looked at.
            assert_eq!(q.due(frame, 100)[0].0, 17);
        }
        assert_eq!(q.census(9)[2], 1);
        q.done(17, w17);
        assert!(!q.due(u64::MAX, 100).iter().any(|d| d.0 == 17));
        // Uploaded with part of its memory there: kept, with what was there.
        let w16 = write(&q, 16);
        q.partly(16, w16, 12, 4096);
        assert_eq!(q.mapped(16), 4096);
        assert_eq!(q.due(12, 100)[0].0, 16);
        // The title writes slot 16 again while the pass works on it: what
        // the pass then says about the old write changes nothing.
        q.written(16, 13, 0, false);
        q.done(16, w16);
        q.later(16, w16, 13, 4);
        assert_eq!(q.mapped(16), 0);
        assert_eq!(q.due(13, 100)[0].0, 16);
        // A draw that makes the host slot (`seen`) keeps what the queue knows,
        let w15 = write(&q, 15);
        q.partly(15, w15, 20, 8192);
        q.written(15, 14, 0, true);
        assert_eq!(q.mapped(15), 8192);
        // and queues a slot the queue did not hold.
        q.written(99, 14, 0, true);
        assert!(q.due(14, 100).iter().any(|d| d.0 == 99));
        // A write due later (a table's slot) is not handed out before.
        q.written(98, 14, 16, false);
        assert!(!q.due(15, 100).iter().any(|d| d.0 == 98));
        assert!(q.due(16, 100).iter().any(|d| d.0 == 98));
    }

    /// VIcky: the looks after a whole upload carry the ranges the
    /// fingerprint read (262625e33); with `VICTORY_RECHECK_PARTLY` an upload
    /// in part keeps its part and has its fingerprint noted too, which a
    /// look that finds no more mapped keeps and a new write forgets; with
    /// `VICTORY_DS_HOLD` a slot is held at most so often for one write of
    /// it (then uploaded), an old write is passed over, and memory mapped
    /// under it again starts the count anew.
    #[test]
    fn rechecks_carry_their_ranges_parts_keep_theirs_and_holds_are_bounded() {
        let mut q = TextureQueue::default();
        q.written(1, 1, 0, false);
        let w1 = q.due(1, 10)[0].1;
        let mips: Mips = vec![(0x1000, 1024), (0x8000, 1024)].into();
        q.recheck(1, w1, 5, (0x10_0000, 0x2_0000), Some(mips.clone()), 77, 0, 1);
        let (checked, got, mapped) = q.checked(1).unwrap();
        assert_eq!(checked, (0x10_0000, 0x2_0000, 77, 0, 1));
        assert_eq!(got.as_deref(), Some(&*mips));
        assert_eq!(mapped, 0);
        assert!(q.due(4, 10).is_empty());
        // In part: due as `partly` makes it, its part kept, the print noted.
        q.written(2, 2, 0, false);
        let w2 = q.due(2, 10).iter().find(|d| d.0 == 2).unwrap().1;
        q.recheck_partly(2, w2, 3, (0x20_0000, 0x4_0000), Some(mips.clone()), 99, 2, 0x1_0000);
        assert_eq!(q.mapped(2), 0x1_0000);
        let (checked, _, mapped) = q.checked(2).unwrap();
        assert_eq!((checked.2, mapped), (99, 0x1_0000));
        assert_eq!(q.census(3)[3], 1, "counted as uploaded in part");
        assert_eq!(q.due(3, 10)[0].0, 2, "looked at before the slots never looked at");
        q.later(2, w2, 3, 16);
        assert_eq!(q.checked(2).map(|c| c.2), Some(0x1_0000));
        // Uploaded in part again without one (a table's): it is forgotten.
        q.partly(2, w2, 4, 0x2_0000);
        assert!(q.checked(2).is_none());
        assert_eq!(q.mapped(2), 0x2_0000);
        q.written(2, 4, 0, false);
        assert!(q.checked(2).is_none());
        // Held while a write is pending: three times for this write, then not.
        let w2 = q.due(4, 10).iter().find(|d| d.0 == 2).unwrap().1;
        for k in 0..3 {
            assert!(q.ds_hold(2, w2, 5 + k, 3));
            assert!(!q.due(4 + k, 10).iter().any(|d| d.0 == 2));
        }
        assert!(!q.ds_hold(2, w2, 9, 3));
        assert!(q.ds_hold(2, w2 + 1000, 9, 3), "a write the slot no longer has");
        q.changed(2, 10, 11);
        let w2 = q.due(11, 10).iter().find(|d| d.0 == 2).unwrap().1;
        assert!(q.ds_hold(2, w2, 12, 3));
    }

    /// Slots held back for lack of room on the card are not handed out
    /// until their frame, keep their place, are not counted as waiting
    /// for their memory, and all come back at once when the card has room;
    /// a slot written again meanwhile is the title's new write.
    #[test]
    fn slots_held_for_lack_of_room_come_back_when_there_is_room() {
        let mut q = TextureQueue::default();
        for i in 0..4u32 {
            q.written(20 + i, 1, 0, false);
        }
        let due = q.due(1, 100);
        assert_eq!(due.len(), 4);
        // The card is full: slots 20 and 21 are held back until frame 101.
        for &(index, write) in due.iter().filter(|d| d.0 < 22) {
            q.hold(index, write, 101);
        }
        let now: Vec<u32> = q.due(2, 100).into_iter().map(|d| d.0).collect();
        assert!(!now.contains(&20) && !now.contains(&21));
        assert!(now.contains(&22) && now.contains(&23));
        assert_eq!(q.census(2)[2], 0, "held slots do not wait for their memory");
        // Due again at their frame, in the order they had.
        assert_eq!(q.due(101, 100), due);
        // Slot 21 is written again while held: due as written.
        q.written(21, 3, 0, false);
        assert!(q.due(3, 100).iter().any(|d| d.0 == 21));
        // Room again at frame 5: slot 20 is due at once, and only it was
        // still held as it was.
        assert_eq!(q.room(5), 1);
        assert!(q.due(5, 100).iter().any(|d| d.0 == 20));
        assert_eq!(q.room(6), 0);
    }

    /// Textures of which only the smallest levels are held, packed into
    /// one another's address range (the three of run 188: 0x2c000 bytes
    /// each at full size, their starts 0x2000 and 0x4000 apart): writing
    /// any of them, in any order and again, leaves the others.
    #[test]
    fn packed_neighbours_stay() {
        let mut spans = Spans::default();
        assert_eq!(spans.rule, Rule::Start);
        let at = 0xb_86bf_8000u64;
        let (a, b, c) = (at, at + 0x2000, at + 0x6000);
        assert!(spans.note(105_998, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        assert!(spans.note(108_521, c, c + 0x2c000, true).is_empty());
        // The first written again (as a title does when levels arrive),
        // from another slot too: its span reaches over the other two.
        assert!(spans.note(7, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        let mut all = spans.slots_in(0, u64::MAX);
        all.sort_unstable();
        assert_eq!(all, [7, 105_998, 106_004, 108_521]);
        // A draw makes a slot of the second texture: it is one of them.
        assert!(spans.note(9, b, b + 0x2c000, false).is_empty());
        assert_eq!(spans.by_slot.get(&9), Some(&b));
    }

    /// A texture written where another starts replaces it: at the same
    /// address with another length, or inside the block of its smallest
    /// levels. The packed neighbours of the one replaced stay.
    #[test]
    fn a_texture_written_at_the_start_of_another_replaces_it() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        // Another length from the same start: slot 1 is stale, slot 2 not.
        assert_eq!(spans.note(3, at, at + 0xb000, true), [1]);
        // A start inside the first 4 KiB of slot 3's texture: stale too.
        assert_eq!(spans.note(4, at + 0x800, at + 0x1800, true), [3]);
        // And the other way: a texture that starts just before one.
        assert_eq!(spans.note(5, at + 0x1800, at + 0x4800, true), [2]);
        let mut left = spans.slots_in(0, u64::MAX);
        left.sort_unstable();
        assert_eq!(left, [4, 5]);
        // Exactly 4 KiB apart: both there.
        assert!(spans.note(6, at + 0x2800, at + 0x3800, true).is_empty());
        // A draw's view that clashes is left out, and nothing is stale.
        assert!(spans.note(8, at + 0x900, at + 0x2900, false).is_empty());
        assert_eq!(spans.by_slot.get(&8), None);
        assert_eq!(spans.by_slot.len(), 3);
    }

    /// A slot written with another texture leaves its old span; the same
    /// view written again changes nothing.
    #[test]
    fn a_slot_rewritten_moves_to_its_new_span() {
        let mut spans = Spans::default();
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert!(spans.note(2, 0x12000, 0x3e000, true).is_empty());
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [1]);
        // Slot 1 now views a texture far away.
        assert!(spans.note(1, 0x90000, 0x91000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x12000, 0x12001), [2]);
        assert_eq!(spans.slots_in(0x90000, 0x90001), [1]);
        spans.forget(1);
        spans.forget(2);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps, unmaps or writes names the textures that
    /// are there: among packed neighbours the one whose held levels the
    /// range lies in (it ends where the next one starts), and the ones
    /// that start inside the range.
    #[test]
    fn memory_changed_under_packed_neighbours_names_the_ones_there() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // Inside the first one's held levels.
        assert_eq!(spans.slots_in(at + 0x1000, at + 0x2000), [1]);
        // Inside the second's: not the first, which ends where it starts.
        assert_eq!(spans.slots_in(at + 0x3000, at + 0x4000), [2]);
        // Past the last start: the last one.
        assert_eq!(spans.slots_in(at + 0x20000, at + 0x21000), [3]);
        // A range over all of them.
        assert_eq!(spans.slots_in(at, at + 0x8000), [1, 2, 3]);
        // Past the end of all: none.
        assert_eq!(spans.slots_in(at + 0x32000, at + 0x33000), [] as [u32; 0]);
    }

    /// A write names the textures that hold its bytes: the one that
    /// starts where the write starts, not the neighbour before it, whose
    /// held levels end there.
    #[test]
    fn a_write_at_the_start_of_a_packed_texture_names_it_alone() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // A request that fills the second texture's held levels.
        assert_eq!(spans.slots_in(at + 0x2000, at + 0x6000), [1, 2]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x6000), [2]);
        // From inside the first one's held levels on: both.
        assert_eq!(spans.slots_holding(at + 0x1000, at + 0x3000), [1, 2]);
        // A whole 64 KiB page: every texture that starts in it.
        assert_eq!(spans.slots_holding(at, at + 0x10000), [1, 2, 3]);
        // Past the last start, the last one; past all, or no bytes, none.
        assert_eq!(spans.slots_holding(at + 0x20000, at + 0x21000), [3]);
        assert_eq!(spans.slots_holding(at + 0x32000, at + 0x33000), [] as [u32; 0]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x2000), [] as [u32; 0]);
    }

    /// A write behind the guards' back under a placed texture: a slot
    /// waiting for its first upload is left alone, one uploaded before is
    /// counted, and made due two frames on when that is asked for. (The
    /// ranges are handed over here, not through the list all heaps share.)
    #[test]
    fn a_page_write_makes_an_uploaded_texture_slot_due_when_asked() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fed_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc);
        assert!(heap.note_buffer(1, Some(&view), false).is_empty());
        let written = [(at, at + 0x1000)];
        // Waiting for its first upload: held, not uploaded, nothing done.
        assert_eq!(heap.slots_written_under(&written, 50, true), [1, 0, 0]);
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Uploaded and out of the queue. Memory elsewhere: nothing.
        assert_eq!(heap.slots_written_under(&[(at + 0x10_0000, at + 0x10_1000)], 60, true), [0, 0, 0]);
        // Under it, counted only: it stays out of the queue.
        assert_eq!(heap.slots_written_under(&written, 60, false), [1, 1, 0]);
        assert_eq!(slots(u64::MAX), []);
        // Under it, asked for: due two frames on, not before.
        assert_eq!(heap.slots_written_under(&written, 60, true), [1, 1, 1]);
        assert_eq!(slots(61), []);
        assert_eq!(slots(62), [1]);
    }

    /// `VICTORY_STALE_TEXTURES=0`: nothing is ever stale, and a view that
    /// overlaps others is left out of the spans, as before.
    #[test]
    fn never_stale_leaves_overlapping_views_out() {
        let mut spans = Spans { rule: Rule::Never, ..Default::default() };
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x2000, 0x4000, true).is_empty());
        assert!(spans.note(3, 0x1000, 0x2000, true).is_empty());
        assert_eq!(spans.slots_in(0, u64::MAX), [1]);
    }

    /// The rule until run 188 (`VICTORY_STALE_TEXTURES=overlap`): a view
    /// written over memory that older views describe otherwise makes
    /// those stale; views of the same span are one texture's.
    #[test]
    fn a_newer_view_of_other_memory_makes_the_older_stale() {
        let mut spans = Spans { rule: Rule::Overlap, ..Default::default() };
        // Three textures side by side, the first in two slots.
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(3, 0x3000, 0x4000, true).is_empty());
        assert!(spans.note(4, 0x8000, 0x9000, true).is_empty());
        // The same slot written with the same span again: nothing changes.
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        assert_eq!(spans.slots_in(0x4000, 0x8000), [] as [u32; 0]);
        // A draw makes a slot whose view overlaps them: left out, nothing stale.
        assert!(spans.note(9, 0x2000, 0x3800, false).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        // The title writes a texture over the end of the first and all of
        // the second: their three slots are stale.
        let mut stale = spans.note(5, 0x2000, 0x4000, true);
        stale.sort_unstable();
        assert_eq!(stale, [1, 2, 3]);
        assert_eq!(spans.slots_in(0, u64::MAX), [5, 4]);
        // Another start in the same memory: stale in turn. Another length
        // from the same start too.
        assert_eq!(spans.note(6, 0x2800, 0x3000, true), [5]);
        assert_eq!(spans.note(7, 0x2800, 0x2c00, true), [6]);
        // A slot that moves to another texture leaves its span behind.
        assert!(spans.note(7, 0x8000, 0x9000, true).is_empty());
        assert_eq!(spans.slots_in(0x2800, 0x2c00), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x8000, 0x8001), [4, 7]);
        spans.forget(4);
        spans.forget(7);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps under a slot's texture makes the slot due
    /// again from the next frame; a stale slot leaves the queue.
    #[test]
    fn remapped_memory_makes_its_texture_slots_due() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fee_0000_0000u64;
        let view = |address: u64, width: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let len = crate::resource::placed_bytes(&[3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0]).unwrap();
        assert!(heap.note_buffer(1, Some(&view(at, 64)), false).is_empty());
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Memory elsewhere: nothing. Under the texture: due from the next frame.
        remapped((at + 2 * len) as usize, 0x1000);
        remapped(at as usize, 0x1000);
        assert_eq!(heap.texture_slots_remapped(50), 1);
        assert_eq!(slots(50), []);
        assert_eq!(slots(51), [1]);
        // A wider texture written at the same address: slot 1 is stale.
        assert_eq!(heap.note_buffer(2, Some(&view(at, 128)), false), [1]);
        assert_eq!(slots(u64::MAX), [2]);
    }

    /// A slot a draw read through a table leaves the queue; the title's
    /// next write of it waits for its draw, and is due after the grace
    /// frames if none reads it.
    #[test]
    fn slots_read_through_tables_leave_the_texture_queue() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 100;
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        let texture = View::Placed(false, [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 0x105, 0, 0], [0; 16]);
        assert!(is_texture_view(&texture));
        heap.queue_texture(2, true, false, frame);
        heap.queue_texture(3, true, false, frame);
        assert_eq!(slots(frame), [3, 2]);
        heap.tabled_in(3, true, frame);
        assert_eq!(slots(frame), [2]);
        // Made in place by the draw that read it: still out.
        heap.queue_texture(3, true, true, frame);
        assert_eq!(slots(frame), [2]);
        // Written again in the next frame: due once the grace frames have
        // passed without a draw that reads it.
        heap.queue_texture(3, true, false, frame + 1);
        assert_eq!(slots(frame + TABLED_GRACE), [2]);
        assert_eq!(slots(frame + 1 + TABLED_GRACE), [3, 2]);
        // The draw comes after all: out again.
        heap.tabled_in(3, true, frame + 1);
        assert_eq!(slots(u64::MAX), [2]);
        // A slot of a table no draw read stays queued; its next write waits.
        heap.tabled_in(2, false, frame + 1);
        assert_eq!(slots(frame + 1), [2]);
        heap.queue_texture(2, true, false, frame + 2);
        assert_eq!(slots(frame + 2), []);
        // Long after the last table, a write is due at once.
        heap.queue_texture(2, true, false, frame + 3 + TABLED_FRAMES);
        assert_eq!(slots(frame + 3 + TABLED_FRAMES), [2]);
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(heap.id(), left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(heap.id(), left, 2);
        heap.release(heap.id(), right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(heap.id(), first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(heap.id(), last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    #[test]
    fn a_null_cbv_drops_the_slots_previous_resource() {
        let Some(host) = host() else { return };
        let heap = HostHeap::new(false, 4).expect("view heap");
        let (buffer, _) = host.shared_buffer(256).expect("buffer");
        heap.keep(0, Some(buffer));
        let mut slot = [0u64; 4];
        let handle = slot.as_mut_ptr() as u64;
        crate::descriptor::create_cbv(std::ptr::null(), std::ptr::null(), handle);
        assert_eq!(heap.mirror(0, handle, false), Some(View::NullConstant));
        assert!(heap.kept.with(0, |slot| slot.expect("slot 0").is_none()));
        // SAFETY: the device is live.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// VIcky (`VICTORY_HEAP_KEPT_SPLIT=1`): in a chunk of the sync's
    /// mirroring, a mirrored slot's keeps and the marks wait for the
    /// chunk's end and are made there in their order; a dynamic slot's keep
    /// is made at once. Without the split, nothing waits.
    #[test]
    fn a_chunks_keeps_and_marks_are_made_at_its_end() {
        let Some(host) = host() else { return };
        let raw = |r: &ID3D12Resource| windows::core::Interface::as_raw(r) as usize;
        for split in [true, false] {
            super::super::kept::TEST_SPLIT.set(Some(split));
            let heap = HostHeap::new(false, 4).expect("view heap");
            super::super::kept::TEST_SPLIT.set(None);
            assert_eq!(heap.kept.split(), split);
            let (a, _) = host.shared_buffer(256).expect("buffer");
            let (b, _) = host.shared_buffer(256).expect("buffer");
            let shown = |slot: u32| heap.kept.with(slot, |s| s?.as_ref().map(raw));
            let later = KeptLater::begin(&heap);
            assert_eq!(later.is_some(), split);
            heap.keep(1, Some(a.clone()));
            heap.keep(1, Some(b.clone()));
            heap.mark_changed(3, true);
            heap.keep(heap.dynamic_first, Some(a.clone()));
            assert_eq!(shown(heap.dynamic_first), Some(raw(&a)), "a dynamic slot's keep waited");
            let marked = |heap: &HostHeap| heap.changed.lock().unwrap().contains(&3);
            if split {
                assert_eq!(shown(1), None, "made before the chunk's end");
                assert!(!marked(&heap), "marked before the chunk's end");
            }
            drop(later);
            assert_eq!(shown(1), Some(raw(&b)), "split {split}");
            assert!(marked(&heap), "split {split}");
        }
    }

    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(heap.id(), a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// trustno1 47747d62 (fxceless #1324): a release naming another heap's
    /// identity frees nothing.
    #[test]
    fn dynamic_slots_are_given_back_only_to_the_heap_that_took_them() {
        let (Some(a), Some(b)) = (HostHeap::new(false, 4), HostHeap::new(false, 4)) else {
            return; // no host GPU
        };
        assert_ne!(a.id(), b.id());
        let first = a.dynamic(2).unwrap();
        let free = || a.free.load(std::sync::atomic::Ordering::Relaxed);
        let taken = free();
        a.release(b.id(), first, 2);
        assert_eq!(free(), taken, "released with another heap's identity: nothing is freed");
        a.release(a.id(), first, 2);
        assert_eq!(free(), taken + 2);
    }

    /// The title's view heap on NVIDIA's 1,000,000 views with a
    /// heap map: title slots below 131,072 at their own index, 606,782 pool
    /// slots, the map at 737,854, and the fork's dynamic slots (from
    /// 737,855) and guard slot (999,999, the translator's limit) where they
    /// were. A window past the room is cut to it; the pool holds no more
    /// slots than the title has past the window.
    #[test]
    fn the_titles_view_heap_is_laid_out_in_a_million_slots() {
        let l = MapLayout::new(1_245_184, WINDOW, MAP_SLOT);
        assert_eq!((l.window, l.pool, l.slot), (131_072, 606_782, 737_854));
        assert_eq!(l.dynamic_first(), PORTABLE_VIEWS - DYNAMIC - 1, "the fork's first dynamic slot");
        assert_eq!(l.guard(), PORTABLE_VIEWS - 1, "the fork's guard slot");
        assert_eq!(l.size(), PORTABLE_VIEWS);
        let wide = MapLayout::new(1_245_184, 900_000, MAP_SLOT);
        assert_eq!((wide.window, wide.pool), (MAP_SLOT, 0));
        let small = MapLayout::new(20, 16, 24);
        assert_eq!((small.window, small.pool, small.dynamic_first()), (16, 4, 25));
        assert_eq!(small.size(), 25 + DYNAMIC + 1);
    }

    /// A view heap made while the translator has a heap map (as a capped
    /// host's `probe_view_limit` sets it) is laid out as [`MapLayout`] says,
    /// its map's raw view at the map slot; one no larger than the window
    /// keeps the fork's layout.
    #[test]
    fn a_view_heap_with_a_heap_map_keeps_the_dynamic_slots_after_the_map() {
        if host().is_none() {
            return;
        }
        TEST_MAP.set(Some((16, 24)));
        let mapped = HostHeap::new(false, 64);
        let fits = HostHeap::new(false, 8);
        TEST_MAP.set(None);
        let (mapped, fits) = (mapped.unwrap(), fits.unwrap());
        assert_eq!((mapped.count(), mapped.title_count, mapped.dynamic_first), (16, 64, 25));
        assert!(mapped.map.is_some());
        {
            let made = mapped.made.read().unwrap();
            assert_eq!(made.keys.len(), 16 + 8, "the window's and the pool's");
            assert_eq!(made.map.as_ref().map(SlotMap::len), Some(48));
        }
        assert_eq!(mapped.tabled.len(), 64, "by title slot");
        assert_eq!(mapped.kept.len(), (25 + DYNAMIC) as usize);
        assert!(mapped.kept.with(24, |slot| slot.expect("slot 24").is_some()), "the map's raw view holds it");
        assert_eq!(mapped.dynamic(DYNAMIC), Some(25));
        assert_eq!(mapped.dynamic(1), None);
        assert_eq!((fits.count(), fits.dynamic_first), (8, 8));
        assert!(fits.map.is_none() && fits.made.read().unwrap().map.is_none());
        assert_eq!(fits.dynamic(1), Some(8));
    }

    /// A title view heap of `count` slots and its first descriptor, made
    /// as the title makes them (its host heap is made at first use).
    fn title_heap(count: u32) -> (crate::object::This, u64) {
        #[repr(C)]
        struct Extended(super::super::HeapDesc, super::super::Placement);
        let desc = super::super::HeapDesc {
            kind: 0,
            count,
            flags: 1,
            nodes: 0,
        };
        let none = super::super::Placement { address: 0, size: 0 };
        let extended = Extended(desc, none);
        let mut out = std::ptr::null();
        let made = super::super::create_heap(
            std::ptr::null(),
            &extended.0,
            &super::super::HEAP_IIDS[0],
            &mut out,
        );
        assert_eq!(made, crate::object::S_OK);
        (out, super::super::heap(out).start)
    }

    /// Writes a constant buffer view of `address` into title slot `slot`.
    fn title_cbv(start: u64, slot: u32, address: u64) {
        super::super::create_cbv(
            std::ptr::null(),
            &[address as u32, (address >> 32) as u32, 256, 0],
            start + u64::from(slot) * TITLE_SLOT_BYTES,
        );
    }

    fn release(object: crate::object::This) -> u32 {
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(object, 0x10)(object)
    }

    /// The heap map entry of title slot `title`, and its pool slot.
    fn entry(heap: &HostHeap, title: u32) -> (u32, Option<u32>) {
        let window = heap.count();
        let at = heap.map.as_ref().unwrap().at;
        // SAFETY: an entry of the mapped heap map.
        let entry = unsafe { (at as *const u32).add((title - window) as usize).read_volatile() };
        let pool = heap.made.read().unwrap().map.as_ref().unwrap().host(title);
        (entry, pool)
    }

    /// Title slots past the identity window that the title writes take
    /// pool slots, published in the heap map (entries hold the slot's
    /// bitwise not) once made; slots below it keep their index. A slot the
    /// title writes again keeps its pool slot; one it empties gives it back
    /// (entry zeroed: the guard), quarantined while submissions run.
    #[test]
    fn title_slots_past_the_window_take_pool_slots() {
        if host().is_none() {
            return;
        }
        let (title, start) = title_heap(64);
        title_cbv(start, 3, 0x1_0000);
        title_cbv(start, 40, 0x2_0000);
        TEST_MAP.set(Some((16, 24)));
        let heap = super::super::heap(title).host().expect("a host heap");
        TEST_MAP.set(None);
        assert_eq!(heap.count(), 16);
        let (e40, pooled) = entry(heap, 40);
        let pooled = pooled.expect("a pool slot");
        assert!((16..24).contains(&pooled), "{pooled}");
        assert_eq!(e40, !pooled);
        assert_eq!(entry(heap, 41), (0, None));
        assert_eq!(heap.made.read().unwrap().host(3), Some(3), "below the window: its own");
        assert_eq!(heap.title_start.load(std::sync::atomic::Ordering::Relaxed), start);

        // Written again: the same pool slot. Another slot: another one.
        title_cbv(start, 40, 0x3_0000);
        assert_eq!(entry(heap, 40), (!pooled, Some(pooled)));
        title_cbv(start, 63, 0x3_0000);
        let other = entry(heap, 63).1.expect("a pool slot");
        assert_ne!(other, pooled);

        // Emptied (as a write of no view finds it): the entry reads the
        // guard and the pool slot waits.
        let handle = start + 40 * TITLE_SLOT_BYTES;
        // SAFETY: a descriptor slot of the title heap.
        unsafe { (handle as *mut u8).write_bytes(0, 32) };
        assert!(heap.mirror(40, handle, false).is_none());
        assert_eq!(entry(heap, 40), (0, None));
        let stats = heap.made.read().unwrap().map.as_ref().unwrap().stats();
        assert_eq!((stats.mapped, stats.quarantined), (1, 1));
        assert_eq!(release(title), 0);
    }

    /// With the pool used up, the pool slots of title slots that hold no
    /// live view any more (overwritten as bytes: no descriptor write says
    /// so) are given back, and the slot that asked takes one.
    #[test]
    fn a_used_up_pool_gives_back_slots_without_views() {
        if host().is_none() {
            return;
        }
        let (title, start) = title_heap(64);
        TEST_MAP.set(Some((16, 20)));
        let heap = super::super::heap(title).host().expect("a host heap");
        TEST_MAP.set(None);
        for slot in 16..20 {
            title_cbv(start, slot, 0x1_0000 + u64::from(slot) * 0x100);
        }
        let freed = entry(heap, 17).1.expect("a pool slot");
        // SAFETY: a descriptor slot of the title heap.
        unsafe { ((start + 17 * TITLE_SLOT_BYTES) as *mut u8).write_bytes(0, 32) };
        title_cbv(start, 30, 0x9_0000);
        assert_eq!(entry(heap, 17), (0, None), "given back");
        assert!(entry(heap, 16).1.is_some() && entry(heap, 19).1.is_some());
        // Handed out once the submissions running when it was given back
        // have finished (other tests' may be).
        let (running, _) = crate::host::submissions();
        let waited = std::time::Instant::now();
        while entry(heap, 30).1.is_none() && waited.elapsed() < std::time::Duration::from_secs(20) {
            if crate::host::submissions().1 >= running {
                let _ = heap.mirror(30, start + 30 * TITLE_SLOT_BYTES, false);
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        assert_eq!(entry(heap, 30), (!freed, Some(freed)));
        assert_eq!(release(title), 0);
    }
}

#[cfg(test)]
mod record_print_tests {
    use super::record_print;

    /// Equal records print alike; a record that differs in any one byte,
    /// or in its length, prints otherwise.
    #[test]
    fn a_record_that_changed_prints_otherwise() {
        let record: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(37).wrapping_add(5)).collect();
        let print = record_print(&record);
        assert_eq!(print, record_print(&record.clone()));
        for at in 0..32 {
            for bit in 0..8 {
                let mut other = record.clone();
                other[at] ^= 1 << bit;
                assert_ne!(record_print(&other), print, "byte {at} bit {bit}");
            }
        }
        assert_ne!(record_print(&record[..16]), print);
        assert_ne!(record_print(&[0u8; 32]), record_print(&[0u8; 16]));
        // A record of zeros is told from "never looked at" by the caller's `| 1`.
        assert_ne!(record_print(&[0u8; 32]) | 1, 0);
    }
}

#[cfg(test)]
mod own_copy_tests {
    use super::own_bytes_if;

    /// A buffer view gets a copy of its own bytes only when it has no host
    /// view on its buffer's copy: off that copy's elements, and not a
    /// structured view that can be a raw view of the same bytes.
    #[test]
    fn a_view_off_its_copys_elements_gets_its_own_bytes() {
        let base = 0xd_9000_0000u64;
        // On an element: the buffer's copy (the two cases of every run's
        // log are the ones below).
        assert_eq!(own_bytes_if(true, 24, base, 3, 10, 12, 0, false), None);
        assert_eq!(own_bytes_if(true, 0, base, 0, 10, 12, 0, false), None);
        // A typed view of 12-byte elements, 5,222,240 bytes into its copy
        // (8 past an element): its own bytes, from its first element.
        assert_eq!(own_bytes_if(true, 5_222_240, base, 3, 10, 12, 0, false), Some((base + 36, 120)));
        // A structured view of 32-byte elements 8 bytes in: not on 16, so
        // no raw view of the copy either.
        assert_eq!(own_bytes_if(true, 8, base, 2, 4, 32, 32, true), Some((base + 64, 128)));
        // The same 16 bytes in: off its elements, but a raw view of the
        // copy's bytes does (as before).
        assert_eq!(own_bytes_if(true, 16, base, 2, 4, 32, 32, true), None);
        // No elements, no size, or the switch off: as before.
        assert_eq!(own_bytes_if(true, 8, base, 2, 0, 32, 32, true), None);
        assert_eq!(own_bytes_if(true, 8, base, 2, 4, 0, 0, false), None);
        assert_eq!(own_bytes_if(false, 8, base, 2, 4, 32, 32, true), None);
    }
}

/// The switches of the passes over the heap's records
/// (`VICTORY_HEAP_SYNC_CACHE`, `VICTORY_HEAP_BUFFERS_SKIP_SAME`,
/// `VICTORY_HEAP_SYNC_THREADS`): each does what the slow way does. The
/// tests that need the host device return early without one.
#[cfg(test)]
mod fast_tests {
    use super::*;

    /// A title view heap of `count` slots and its first descriptor.
    fn title_heap(count: u32) -> (crate::object::This, u64) {
        #[repr(C)]
        struct Extended(super::super::HeapDesc, super::super::Placement);
        let desc = super::super::HeapDesc {
            kind: 0,
            count,
            flags: 1,
            nodes: 0,
        };
        let extended = Extended(desc, super::super::Placement { address: 0, size: 0 });
        let mut out = std::ptr::null();
        let made = super::super::create_heap(std::ptr::null(), &extended.0, &super::super::HEAP_IIDS[0], &mut out);
        assert_eq!(made, crate::object::S_OK);
        (out, super::super::heap(out).start)
    }

    fn release(object: crate::object::This) -> u32 {
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(object, 0x10)(object)
    }

    /// VIcky (`VICTORY_SPANS_QUICK`, after Belobug a284ca74's test): every
    /// slot in the spans of placed textures has its bit (what makes the
    /// quick way safe: a slot without one is not there); a slot forgotten
    /// there loses it; one another's view made stale keeps it ("may").
    #[test]
    fn the_spanned_bits_hold_every_slot_in_the_spans() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let holds = |heap: &HostHeap| heap.placed.lock().unwrap().by_slot.keys().all(|&slot| heap.in_spans(slot));
        let at = 0x7fec_0000_0000u64;
        assert!(heap.note_buffer(1, Some(&texture_view(at)), false).is_empty());
        assert!(heap.in_spans(1) && !heap.in_spans(2) && holds(&heap));
        // A texture starting in the first one's first 4 KiB: slot 1 is stale,
        // out of the spans, and keeps its bit.
        assert_eq!(heap.note_buffer(2, Some(&texture_view(at + 0x100)), false), [1]);
        assert!(heap.in_spans(1) && heap.in_spans(2) && holds(&heap));
        assert!(!heap.placed.lock().unwrap().by_slot.contains_key(&1));
        // Written with no view, or with a buffer view: forgotten, bit cleared.
        assert!(heap.note_buffer(2, None, false).is_empty());
        assert!(heap.note_buffer(1, Some(&buffer_view(at + 0x10_0000, 0x1_0000, 0, 4)), false).is_empty());
        assert!(!heap.in_spans(1) && !heap.in_spans(2) && holds(&heap));
        assert!(heap.placed.lock().unwrap().by_slot.is_empty());
        // A slot never in the spans stays without its bit.
        assert!(heap.note_buffer(3, None, false).is_empty());
        assert!(!heap.in_spans(3));
    }

    /// A placed structured buffer view of `width` bytes at `address`:
    /// `count` elements of 16 bytes from element `first`.
    fn buffer_view(address: u64, width: u32, first: u32, count: u32) -> View {
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, width);
        let mut desc = [0u32; 16];
        // unknown format, buffer, default mapping
        (desc[1], desc[2], desc[4], desc[6], desc[7]) = (1, 5768, first, count, 16);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, resource, desc)
    }

    /// A placed 2D texture view at `address` (64x64 BC7, one level).
    fn texture_view(address: u64) -> View {
        let resource = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
        let mut desc = [0u32; 16];
        (desc[0], desc[1], desc[2], desc[5]) = (98, 4, 5768, 1);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, resource, desc)
    }

    /// Writes `view`'s record into title slot `slot` as bytes, as a title
    /// that copies records does; its key.
    fn write_record(start: u64, slot: u32, view: &View) -> u64 {
        let [i, generation] = super::super::words_for(view.clone());
        let handle = start + u64::from(slot) * TITLE_SLOT_BYTES;
        // SAFETY: a descriptor slot of the title heap.
        unsafe { (handle as *mut [u32; 3]).write_unaligned([super::super::VIEW_TAG, i, generation]) };
        u64::from(generation) << 32 | u64::from(i)
    }

    /// A heap made with the switches `(cache, skip, check)` for this
    /// thread; they stay set until `TEST_FAST` is set again.
    fn heap_with(count: u32, fast: (usize, bool, bool)) -> Option<(crate::object::This, u64, &'static HostHeap)> {
        host()?;
        TEST_FAST.set(Some(fast));
        let (title, start) = title_heap(count);
        let heap = super::super::heap(title).host()?;
        Some((title, start, heap))
    }

    /// The buffer title slot `index`'s host slot holds, as a raw pointer.
    fn shown(heap: &HostHeap, index: u32) -> Option<usize> {
        let host = heap.made.read().unwrap().host(index)?;
        heap.kept
            .with(host, |slot| slot?.as_ref().map(|b| windows::core::Interface::as_raw(b) as usize))
    }

    /// The same record in another slot is copied from the cache; the entry
    /// holds the copies it was made on, and what the slow way makes of the
    /// view now is what the entry was made of.
    #[test]
    fn a_record_made_before_is_copied_from_the_cache() {
        let Some((title, start, heap)) = heap_with(64, (1 << 10, false, true)) else {
            return;
        };
        let base = 0x41_0000_0000u64;
        let view = buffer_view(base, 0x1_0000, 4, 64);
        let key = write_record(start, 1, &view);
        write_record(start, 2, &view);
        assert!(heap.mirror(1, start + TITLE_SLOT_BYTES, true).is_some());
        let entry = {
            let made = heap.made.read().unwrap();
            let cache = made.cache.lock().unwrap();
            let cache = cache.table.as_ref().expect("a cache");
            cache.entries[cache.find(key).expect("kept")]
        };
        let whole = crate::shadow::find(base, 0x1_0000).unwrap();
        assert_eq!((entry.whole, entry.shown), (whole.id(), whole.id()));
        let plan = buffer_plan(&view, &mut Lookups::each()).unwrap();
        assert_eq!(entry.print, plan.print(), "kept as the slow way makes it");
        let used = entry.used;
        assert!(heap.mirror(2, start + 2 * TITLE_SLOT_BYTES, true).is_some());
        {
            let made = heap.made.read().unwrap();
            let cache = made.cache.lock().unwrap();
            let cache = cache.table.as_ref().unwrap();
            let e = cache.find(key).unwrap();
            assert_ne!(cache.entries[e].used, used, "used again");
        }
        let buffer = windows::core::Interface::as_raw(&whole.buffer) as usize;
        assert_eq!(shown(heap, 1), Some(buffer));
        assert_eq!(shown(heap, 2), Some(buffer), "the copied slot holds the same buffer");
        TEST_FAST.set(None);
        assert_eq!(release(title), 0);
    }

    /// When the copy behind the view's memory is another one (the title
    /// released its buffer and made one there again), the entry is not
    /// used: the slot is made on the new copy, and the entry made again.
    #[test]
    fn an_entry_whose_copy_changed_is_made_again() {
        let Some((title, start, heap)) = heap_with(64, (1 << 10, false, false)) else {
            return;
        };
        let base = 0x42_0000_0000u64;
        crate::shadow::register(0x7e57_0001, base, 0x1_0000);
        let view = buffer_view(base, 0x1_0000, 0, 64);
        let key = write_record(start, 5, &view);
        heap.mirror(5, start + 5 * TITLE_SLOT_BYTES, true);
        let old = crate::shadow::find(base, 0x1_0000).unwrap();
        crate::shadow::unregister(0x7e57_0001, base);
        crate::shadow::register(0x7e57_0002, base, 0x1_0000);
        let new = crate::shadow::find(base, 0x1_0000).unwrap();
        assert_ne!(old.id(), new.id());
        // Written again elsewhere (the same record): not copied.
        write_record(start, 6, &view);
        heap.mirror(6, start + 6 * TITLE_SLOT_BYTES, true);
        let entry = {
            let made = heap.made.read().unwrap();
            let cache = made.cache.lock().unwrap();
            let cache = cache.table.as_ref().unwrap();
            cache.entries[cache.find(key).unwrap()]
        };
        assert_eq!(entry.whole, new.id(), "made again on the new copy");
        assert_eq!(shown(heap, 6), Some(windows::core::Interface::as_raw(&new.buffer) as usize));
        assert_eq!(shown(heap, 5), Some(windows::core::Interface::as_raw(&old.buffer) as usize), "untouched");
        crate::shadow::unregister(0x7e57_0002, base);
        TEST_FAST.set(None);
        assert_eq!(release(title), 0);
    }

    /// Views a cache may keep: buffer views made on their buffer's copy;
    /// not textures, constant buffers or empty buffers.
    #[test]
    fn only_buffer_views_on_a_copy_are_kept() {
        let base = 0x43_0000_0000u64;
        assert_eq!(cacheable(&buffer_view(base, 0x1_0000, 0, 4)), Some((base, 0x1_0000)));
        assert_eq!(cacheable(&buffer_view(base, 0, 0, 4)), None, "an empty buffer");
        assert_eq!(cacheable(&texture_view(base)), None);
        assert_eq!(cacheable(&View::Constant(base, 256)), None);
        assert_eq!(cacheable(&View::NullConstant), None);
        // A view of a buffer object of another dimension than a buffer's.
        assert_eq!(cacheable(&View::Shader(0, Some([28, 4, 5768, 0, 0, 1, 0, 0, 0, 0]))), None);
        assert_eq!(cacheable(&View::Shader(0, None)), None);
    }

    /// Records are told apart by their whole key; a set keeps two, and a
    /// third replaces the one used longer ago.
    #[test]
    fn a_full_set_gives_up_its_older_entry() {
        if host().is_none() {
            return;
        }
        let mut cache = RecordCache::new(1 << 10, 32).unwrap();
        let set = cache.set(7);
        // Three keys of one set.
        let mut same: Vec<u64> = (8..1 << 20).filter(|&k| cache.set(k) == set).take(2).collect();
        same.insert(0, 7);
        let room = cache.room(same[0]);
        cache.entries[room] = Cached { key: same[0], used: 1, ..Cached::default() };
        let room = cache.room(same[1]);
        assert_ne!(cache.entries[room].key, same[0], "an empty one first");
        cache.entries[room] = Cached { key: same[1], used: 2, ..Cached::default() };
        cache.tick = 3;
        assert_eq!(cache.find(same[0]).map(|e| cache.entries[e].key), Some(same[0]));
        assert_eq!(cache.find(same[2]), None);
        // The first used again: the second is the older now.
        let first = cache.find(same[0]).unwrap();
        cache.entries[first].used = 3;
        cache.tick = 4;
        let room = cache.room(same[2]);
        assert_eq!(cache.entries[room].key, same[1]);
    }

    /// Records that differ in their high bits alone (views of placed memory
    /// at one address with other shapes) are kept in sets of their own.
    #[test]
    fn records_that_differ_in_their_high_bits_take_other_sets() {
        if host().is_none() {
            return;
        }
        let cache = RecordCache::new(1 << 18, 32).unwrap();
        // A placed view's record: its address in the low 40 bits, its shape
        // above.
        let at = 0x8000_0061u64 << 32;
        let sets: std::collections::HashSet<usize> = (0..64u64).map(|shape| cache.set(at | shape << 40)).collect();
        assert!(sets.len() >= 60, "{} sets for 64 shapes", sets.len());
        assert!(sets.iter().all(|&s| s % 2 == 0 && s < cache.entries.len()));
    }

    /// Lookups kept look each ask up once and give the same copy again.
    #[test]
    fn kept_lookups_ask_once() {
        if host().is_none() {
            return;
        }
        let base = 0x44_0000_0000u64;
        let mut kept = Lookups::kept();
        let a = kept.find(base, 0x1000).unwrap();
        let b = kept.find(base, 0x1000).unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b));
        assert_eq!(kept.asked(), 1);
        kept.find(base + 0x1000, 0x1000);
        assert_eq!(kept.asked(), 2);
        let mut each = Lookups::each();
        each.find(base, 0x1000);
        assert_eq!(each.asked(), 0, "nothing kept");
        // The pass's question gets the copy the making found.
        let view = buffer_view(base, 0x1_0000, 0, 16);
        let mut lookups = Lookups::kept();
        let made = buffer_plan(&view, &mut lookups).unwrap();
        let asked = lookups.asked();
        let (copy, address, len) = buffer_copy_with(&view, &mut lookups).unwrap();
        assert_eq!(lookups.asked(), asked, "no lookup more");
        assert_eq!(copy.id(), made.shadow.id());
        let (anew, a2, l2) = buffer_copy(&view).unwrap();
        assert_eq!((anew.id(), a2, l2), (copy.id(), address, len));
    }

    /// The plain buffer views whose notes may be skipped.
    #[test]
    fn plain_buffer_views_are_told() {
        let base = 0x45_0000_0000u64;
        assert!(plain_buffer(&buffer_view(base, 0x1_0000, 0, 4)));
        assert!(!plain_buffer(&texture_view(base)));
        assert!(!plain_buffer(&View::Constant(base, 256)));
        assert!(placed_texture_view(&texture_view(base)));
        assert!(!placed_texture_view(&buffer_view(base, 0x1_0000, 0, 4)));
    }

    /// A buffer slot written with another buffer view skips the note's
    /// work but the stride and the mark; a texture view or no view in
    /// between makes the next note a full one, which takes the slot out of
    /// the texture queue and the spans again.
    #[test]
    fn a_second_buffer_note_does_only_what_differs() {
        let Some((title, _, heap)) = heap_with(64, (0, true, false)) else {
            return;
        };
        let base = 0x46_0000_0000u64;
        let in_queue = |i: u32| heap.textures.lock().unwrap().slots.contains_key(&i);
        let in_spans = |i: u32| heap.placed.lock().unwrap().by_slot.contains_key(&i);
        let listed = |i: u32| heap.buffers.lock().unwrap().contains(&i);
        let stride = |i: u32| {
            let (_, _, data) = heap.strides.as_ref().unwrap();
            // SAFETY: an entry of the mapped table.
            unsafe { (*data as *const u32).add(i as usize).read_volatile() }
        };
        assert!(heap.note_buffer(3, Some(&buffer_view(base, 0x1_0000, 0, 4)), false).is_empty());
        assert!(heap.noted_plain(3));
        assert!(listed(3) && !in_queue(3) && !in_spans(3));
        heap.buffer_changes();
        // Another buffer view, of 32-byte elements: the stride and the mark.
        let mut wide = buffer_view(base + 0x1_0000, 0x1_0000, 0, 4);
        if let View::Placed(_, _, desc) = &mut wide {
            desc[7] = 32;
        }
        assert!(heap.note_buffer(3, Some(&wide), false).is_empty());
        assert_eq!(stride(3), 32);
        assert_eq!(heap.buffer_changes(), vec![3]);
        // A placed texture: a full note (in the queue and the spans, no
        // buffer slot), then a buffer view again: a full note as well.
        heap.note_buffer(3, Some(&texture_view(base + 0x10_0000)), false);
        assert!(!heap.noted_plain(3));
        assert!(in_queue(3) && in_spans(3) && !listed(3));
        heap.note_buffer(3, Some(&buffer_view(base, 0x1_0000, 0, 4)), false);
        assert!(heap.noted_plain(3) && listed(3) && !in_queue(3) && !in_spans(3));
        assert_eq!(stride(3), 16);
        // No view: full, out of the buffer slots.
        heap.note_buffer(3, None, false);
        assert!(!heap.noted_plain(3) && !listed(3));
        TEST_FAST.set(None);
        assert_eq!(release(title), 0);
    }

    /// Without the switch nothing is noted and nothing is skipped.
    #[test]
    fn without_the_switch_every_note_is_full() {
        let Some((title, _, heap)) = heap_with(64, (0, false, false)) else {
            return;
        };
        assert!(heap.noted.is_empty());
        let view = buffer_view(0x47_0000_0000, 0x1_0000, 0, 4);
        heap.note_buffer(2, Some(&view), false);
        assert!(!heap.noted_plain(2));
        TEST_FAST.set(None);
        assert_eq!(release(title), 0);
    }

    /// The state of every title slot of a heap that a pass leaves: its
    /// host slot's record, whether it holds a buffer, the slots written
    /// since the pass over buffer views looked, and the buffer slots.
    fn state(heap: &HostHeap, slots: u32) -> (Vec<u64>, Vec<bool>, Vec<u32>, Vec<u32>) {
        let keys = (0..slots)
            .map(|i| {
                let made = heap.made.read().unwrap();
                made.key(made.host(i).unwrap())
            })
            .collect();
        let kept = (0..slots).map(|i| shown(heap, i).is_some()).collect();
        let mut changed = heap.buffer_changes();
        changed.sort_unstable();
        (keys, kept, changed, heap.buffer_slots())
    }

    /// The pass over buffer views over `slots` as `heap_buffers` makes it:
    /// one by one, or with `VICTORY_HEAP_SYNC_CACHE` in slot order, a chunk
    /// under one lock, with the lookups shared. The copy and range each
    /// slot's view is brought up to date in.
    #[allow(clippy::type_complexity)]
    fn pass(
        heap: &HostHeap,
        start: u64,
        slots: &[u32],
        cached: bool,
    ) -> Vec<(u32, Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>)> {
        let handle = |i: u32| start + u64::from(i) * TITLE_SLOT_BYTES;
        if !cached {
            return slots
                .iter()
                .map(|&i| {
                    let mut lookups = Lookups::each();
                    let copy = heap.mirror_with(i, handle(i), &mut lookups, true).and_then(|v| buffer_copy_with(&v, &mut lookups));
                    (i, copy)
                })
                .collect();
        }
        let mut sorted = slots.to_vec();
        sorted.sort_unstable();
        let mut out = Vec::new();
        for chunk in sorted.chunks(PASS_CHUNK) {
            let ahead: Vec<(u32, u64)> = chunk.iter().map(|&i| (i, handle(i))).collect();
            let mut lookups: Vec<Lookups> = ahead.iter().map(|_| Lookups::kept()).collect();
            let views = heap.mirror_chunk(&ahead, &mut lookups, true);
            for ((&i, view), lookups) in chunk.iter().zip(views).zip(&mut lookups) {
                out.push((i, view.and_then(|v| buffer_copy_with(&v, lookups))));
            }
        }
        out
    }

    /// Records written as bytes, mirrored page by page, and the same
    /// records mirrored in two parts and under one lock a chunk
    /// (`VICTORY_HEAP_SYNC_THREADS`), with the cache and the skipped
    /// notes: every slot ends in the same state, and the pass over buffer
    /// views after it makes the same slots on the same copies.
    #[test]
    fn the_fast_ways_leave_every_slot_as_the_slow_way() {
        if host().is_none() {
            return;
        }
        const SLOTS: u32 = 4096;
        let base = 0x48_0000_0000u64;
        let views: Vec<View> = (0..40u64)
            .map(|k| {
                if k % 7 == 3 {
                    texture_view(base + 0x100_0000 + k * 0x1_0000)
                } else {
                    buffer_view(base + k * 0x1_0000, 0x1_0000, (k % 3) as u32, 64)
                }
            })
            .collect();
        let run = |fast: (usize, bool, bool), chunked: bool| {
            let (title, start, heap) = heap_with(SLOTS, fast).unwrap();
            let mut states = Vec::new();
            for frame in 0..3u32 {
                for s in 0..SLOTS {
                    if (s + frame) % 5 != 0 {
                        write_record(start, s, &views[((s * 7 + frame * 13) % views.len() as u32) as usize]);
                    }
                }
                let bytes: Vec<u8> = {
                    // SAFETY: the title heap's own memory.
                    unsafe { std::slice::from_raw_parts(start as *const u8, (SLOTS as u64 * TITLE_SLOT_BYTES) as usize) }.to_vec()
                };
                let page = crate::watch::PAGE as u64 / TITLE_SLOT_BYTES;
                for first in (0..u64::from(SLOTS)).step_by(page as usize) {
                    let from = (start + first * TITLE_SLOT_BYTES) as usize;
                    let page_bytes = &bytes[(first * TITLE_SLOT_BYTES) as usize..][..(page * TITLE_SLOT_BYTES) as usize];
                    if chunked {
                        let (_, due) = heap.copied_due(first..first + page, (start, TITLE_SLOT_BYTES), from, page_bytes);
                        heap.mirror_due(&due);
                    } else {
                        heap.mirror_copied_page(first..first + page, (start, TITLE_SLOT_BYTES), from, page_bytes);
                    }
                }
                let after_sync = state(heap, SLOTS);
                // The pass over buffer views, as `heap_buffers` makes it.
                let copies = pass(heap, start, &after_sync.2, fast.0 > 0)
                    .into_iter()
                    .map(|(index, copy)| (index, copy.map(|(s, a, l)| (s.id(), a, l)), shown(heap, index)))
                    .collect::<Vec<_>>();
                states.push((after_sync, copies, state(heap, SLOTS)));
            }
            TEST_FAST.set(None);
            assert_eq!(release(title), 0);
            states
        };
        let slow = run((0, false, false), false);
        let fast = run((1 << 12, true, true), true);
        assert_eq!(slow.len(), fast.len());
        for (frame, (s, f)) in slow.iter().zip(&fast).enumerate() {
            assert_eq!(s.0, f.0, "frame {frame}: after the records' pass");
            assert_eq!(s.1, f.1, "frame {frame}: the pass over buffer views");
            assert_eq!(s.2, f.2, "frame {frame}: after it");
        }
    }

    /// Where the time of the pass over records copied in as bytes goes,
    /// and of the pass over buffer views that follows it, on the host
    /// device, with and without the switches: a ring of 100,000 records
    /// rewritten every frame, each a buffer view seen before in another
    /// slot. Not run with the suite (`--ignored`); it prints.
    #[test]
    #[ignore]
    fn where_the_time_of_a_mirrored_record_goes() {
        use std::time::Instant;
        if host().is_none() {
            return;
        }
        const SLOTS: u32 = 1 << 17;
        const RING: u32 = 100_000;
        let base = 0x49_0000_0000u64;
        for distinct in [4_000u64, 150_000] {
            let records: Vec<View> = (0..distinct).map(|k| buffer_view(base + (k % 4000) * 0x1_0000, 0x1_0000, (k / 4000) as u32, 64)).collect();
            for (label, fast, chunked, busy) in [
                ("as before", (0, false, false), false, false),
                ("skip same", (0, true, false), false, false),
                ("cache", (1 << 18, false, false), false, false),
                ("cache + skip", (1 << 18, true, false), false, false),
                ("cache + skip + chunks", (1 << 18, true, false), true, false),
                ("as before, a busy reader", (0, false, false), false, true),
                ("cache + skip + chunks, a busy reader", (1 << 18, true, false), true, true),
            ] {
                let (title, start, heap) = heap_with(SLOTS, fast).unwrap();
                // A thread that reads table slots all the while, as draws
                // do: the same records, made once, then only looked at.
                let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let reader = busy.then(|| {
                    for s in RING..RING + 256 {
                        write_record(start, s, &records[0]);
                    }
                    let (stop, at) = (stop.clone(), std::ptr::from_ref(heap) as usize);
                    std::thread::spawn(move || {
                        // SAFETY: the heap lives until the thread is joined.
                        let heap = unsafe { &*(at as *const HostHeap) };
                        let mut s = 0;
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            heap.mirror(RING + s % 256, start + u64::from(RING + s % 256) * TITLE_SLOT_BYTES, true);
                            s += 1;
                        }
                    })
                });
                let mut times = Vec::new();
                let mut parts: Vec<[f64; 3]> = Vec::new();
                for frame in 0..5u32 {
                    for s in 0..RING {
                        write_record(start, s, &records[((u64::from(s) + u64::from(frame) * 7919) % distinct) as usize]);
                    }
                    let t = Instant::now();
                    let end = start + u64::from(RING) * TITLE_SLOT_BYTES;
                    let mut from = start;
                    let mut due_time = std::time::Duration::ZERO;
                    while from < end {
                        let to = (from + crate::watch::PAGE as u64).min(end);
                        // SAFETY: the title heap's own memory.
                        let bytes = unsafe { std::slice::from_raw_parts(from as *const u8, (to - from) as usize) }.to_vec();
                        let slots = (from - start) / TITLE_SLOT_BYTES..(to - start) / TITLE_SLOT_BYTES;
                        let d = Instant::now();
                        let (_, due) = heap.copied_due(slots, (start, TITLE_SLOT_BYTES), from as usize, &bytes);
                        due_time += d.elapsed();
                        if chunked {
                            heap.mirror_due(&due);
                        } else {
                            for &(index, handle) in &due {
                                heap.mirror(index, handle, false);
                            }
                        }
                        from = to;
                    }
                    let sync = t.elapsed();
                    let t = Instant::now();
                    let changed = heap.buffer_changes();
                    std::hint::black_box(pass(heap, start, &changed, fast.0 > 0));
                    let (made_time, copy_time) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
                    times.push((sync.as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3, changed.len()));
                    parts.push([due_time, made_time, copy_time].map(|d| d.as_secs_f64() * 1e3));
                }
                if label == "cache + skip" {
                    // The parts of a slot's making, over the ring's slots in
                    // slot order, each made from the view it holds.
                    let slots: Vec<u32> = (0..RING).collect();
                    let views: Vec<View> = slots
                        .iter()
                        .map(|&s| {
                            // SAFETY: a descriptor slot of the title heap.
                            let [_, i, g] = unsafe { ((start + u64::from(s) * TITLE_SLOT_BYTES) as *const [u32; 3]).read_unaligned() };
                            super::super::view_at(i, g).unwrap()
                        })
                        .collect();
                    let each = |t: Instant| t.elapsed().as_nanos() as f64 / f64::from(RING);
                    let t = Instant::now();
                    for &s in &slots {
                        let [_, i, g] = unsafe { ((start + u64::from(s) * TITLE_SLOT_BYTES) as *const [u32; 3]).read_unaligned() };
                        let _made = heap.made.write().unwrap();
                        std::hint::black_box(super::super::view_at(i, g));
                    }
                    eprintln!("    record read, lock, view_at: {:.0} ns", each(t));
                    let t = Instant::now();
                    for &s in &slots {
                        std::hint::black_box(heap.note_buffer(s, Some(&views[s as usize]), true));
                    }
                    eprintln!("    note (skipped): {:.0} ns", each(t));
                    let t = Instant::now();
                    for &s in &slots {
                        let made = heap.made.write().unwrap();
                        let key = unsafe { ((start + u64::from(s) * TITLE_SLOT_BYTES + 4) as *const u64).read_unaligned() };
                        std::hint::black_box(heap.make_slot(&made, true, &views[s as usize], key, s, &mut Lookups::kept()));
                    }
                    eprintln!("    make_slot (cache hit): {:.0} ns", each(t));
                    let t = Instant::now();
                    for &s in &slots {
                        let mut l = Lookups::kept();
                        std::hint::black_box(shown_copy(&views[s as usize], base + (u64::from(s) % 4000) * 0x1_0000, 0x1_0000, &mut l));
                    }
                    eprintln!("    shown_copy: {:.0} ns", each(t));
                    let keys: Vec<u64> = slots
                        .iter()
                        .map(|&s| unsafe { ((start + u64::from(s) * TITLE_SLOT_BYTES + 4) as *const u64).read_unaligned() })
                        .collect();
                    {
                        let made = heap.made.read().unwrap();
                        let cache = made.cache.lock().unwrap();
                        let cache = cache.table.as_ref().unwrap();
                        let t = Instant::now();
                        for &k in &keys {
                            std::hint::black_box(cache.find(k));
                        }
                        eprintln!("    cache find: {:.0} ns", each(t));
                        let host_device = host().unwrap();
                        let t = Instant::now();
                        for (&s, &k) in slots.iter().zip(&keys) {
                            let e = cache.find(k).unwrap_or(0);
                            unsafe { host_device.device.CopyDescriptorsSimple(1, heap.cpu(s), cache.at(e), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
                        }
                        eprintln!("    cache find + CopyDescriptorsSimple: {:.0} ns", each(t));
                    }
                    let t = Instant::now();
                    for &s in &slots {
                        let mut l = Lookups::kept();
                        std::hint::black_box(l.find(base + (u64::from(s) % 4000) * 0x1_0000, 0x1_0000));
                    }
                    eprintln!("    Lookups::kept + one find: {:.0} ns", each(t));
                    let t = Instant::now();
                    for &s in &slots {
                        std::hint::black_box(crate::shadow::find(base + (u64::from(s) % 4000) * 0x1_0000, 0x1_0000));
                    }
                    eprintln!("    find: {:.0} ns", each(t));
                    let t = Instant::now();
                    for &s in &slots {
                        heap.changed.lock().unwrap().insert(s);
                    }
                    eprintln!("    changed insert: {:.0} ns", each(t));
                    let t = Instant::now();
                    for &s in &slots {
                        heap.note_stride(s, Some(16));
                    }
                    eprintln!("    note_stride: {:.0} ns", each(t));
                    let t = Instant::now();
                    for &s in &slots {
                        std::hint::black_box(plain_buffer(&views[s as usize]));
                    }
                    eprintln!("    plain_buffer: {:.0} ns", each(t));
                    let buffer = crate::shadow::find(base, 0x1_0000).unwrap().buffer.clone();
                    let t = Instant::now();
                    for &s in &slots {
                        heap.keep(s, Some(buffer.clone()));
                    }
                    eprintln!("    keep: {:.0} ns", each(t));
                    let t = Instant::now();
                    for &s in &slots {
                        null_for(&views[s as usize], heap.cpu(s));
                    }
                    eprintln!("    null_for: {:.0} ns", each(t));
                }
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                if let Some(reader) = reader {
                    reader.join().unwrap();
                }
                // The frames after the first (the cache is filled then).
                let later = &times[1..];
                let n = later.len() as f64;
                let part = |i: usize| parts[1..].iter().map(|p| p[i]).sum::<f64>() / n;
                eprintln!(
                    "{distinct:>6} records, {label:<38}: sync {:6.1} ms (prints {:.1}), pass {:6.1} ms over {} slots (first frame {:.1} + {:.1})",
                    later.iter().map(|t| t.0).sum::<f64>() / n,
                    part(0),
                    later.iter().map(|t| t.1).sum::<f64>() / n,
                    later[0].2,
                    times[0].0,
                    times[0].1,
                );
                TEST_FAST.set(None);
                assert_eq!(release(title), 0);
            }
        }
    }
    /// VIcky (`VICTORY_MADE_TURNS`, roadmap A2): views for the turns' tests,
    /// a buffer view, a texture view or a null constant view by `k`.
    fn turn_views(base: u64) -> Vec<View> {
        (0..40u64)
            .map(|k| {
                if k % 7 == 3 {
                    texture_view(base + 0x100_0000 + k * 0x1_0000)
                } else if k % 11 == 5 {
                    View::NullConstant
                } else {
                    buffer_view(base + k * 0x1_0000, 0x1_0000, (k % 3) as u32, 64)
                }
            })
            .collect()
    }

    /// Every title slot's host slot record and whether it holds a resource.
    fn keys_and_kept(heap: &HostHeap, slots: std::ops::Range<u32>) -> Vec<(u64, bool)> {
        slots
            .map(|i| {
                let key = {
                    let made = heap.made.read().unwrap();
                    made.host(i).map_or(0, |host| made.key(host))
                };
                (key, shown(heap, i).is_some())
            })
            .collect()
    }

    /// VIcky (`VICTORY_MADE_TURNS`): the heap sync's chunks and the pass's,
    /// made while four threads wait for `made` all along (each let in
    /// between two slots), leave every slot as they do without the turns,
    /// frame after frame; and `made` was handed over.
    #[test]
    fn turns_leave_every_slot_as_without_them() {
        if host().is_none() {
            return;
        }
        const SLOTS: u32 = 4096;
        let base = 0x49_0000_0000u64;
        let views = turn_views(base);
        let run = |turns: bool| {
            let (title, start, heap) = heap_with(SLOTS, (1 << 12, true, false)).unwrap();
            TEST_TURNS.set(Some(turns));
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let waiters: Vec<_> = (0..if turns { 4 } else { 0 })
                .map(|_| {
                    let stop = stop.clone();
                    std::thread::spawn(move || {
                        TEST_TURNS.set(Some(true));
                        let mut n = 0u64;
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            let made = heap.made_alone();
                            std::hint::black_box(made.key_of(0));
                            drop(made);
                            n += 1;
                            std::thread::yield_now();
                        }
                        n
                    })
                })
                .collect();
            let mut states = Vec::new();
            for frame in 0..3u32 {
                for s in 0..SLOTS {
                    if (s + frame) % 5 != 0 {
                        write_record(start, s, &views[((s * 7 + frame * 13) % views.len() as u32) as usize]);
                    }
                }
                let bytes: Vec<u8> = {
                    // SAFETY: the title heap's own memory.
                    unsafe { std::slice::from_raw_parts(start as *const u8, (SLOTS as u64 * TITLE_SLOT_BYTES) as usize) }.to_vec()
                };
                let page = crate::watch::PAGE as u64 / TITLE_SLOT_BYTES;
                for first in (0..u64::from(SLOTS)).step_by(page as usize) {
                    let from = (start + first * TITLE_SLOT_BYTES) as usize;
                    let page_bytes = &bytes[(first * TITLE_SLOT_BYTES) as usize..][..(page * TITLE_SLOT_BYTES) as usize];
                    let (_, due) = heap.copied_due(first..first + page, (start, TITLE_SLOT_BYTES), from, page_bytes);
                    heap.mirror_due(&due);
                }
                let after_sync = state(heap, SLOTS);
                let copies = pass(heap, start, &after_sync.2, true)
                    .into_iter()
                    .map(|(index, copy)| (index, copy.map(|(s, a, l)| (s.id(), a, l)), shown(heap, index)))
                    .collect::<Vec<_>>();
                states.push((after_sync, copies, state(heap, SLOTS)));
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            for waiter in waiters {
                assert!(waiter.join().unwrap() > 0);
            }
            TEST_TURNS.set(None);
            TEST_FAST.set(None);
            assert_eq!(release(title), 0);
            states
        };
        let before = TURNS_TAKEN.load(std::sync::atomic::Ordering::Relaxed);
        let without = run(false);
        let with = run(true);
        assert!(TURNS_TAKEN.load(std::sync::atomic::Ordering::Relaxed) > before, "made was handed over");
        assert_eq!(without.len(), with.len());
        for (frame, (a, b)) in without.iter().zip(&with).enumerate() {
            assert_eq!(a.0, b.0, "frame {frame}: after the records' pass");
            assert_eq!(a.1, b.1, "frame {frame}: the pass over buffer views");
            assert_eq!(a.2, b.2, "frame {frame}: after it");
        }
    }

    /// VIcky (`VICTORY_MADE_TURNS`): twelve threads on one heap, rounds of
    /// records each over slots of its own: the heap sync's chunks (one
    /// thread), the pass's chunks (one), and ten that make one slot at a
    /// time as the title's recording threads do, all with the turns on.
    /// No deadlock, and every slot ends as the same rounds made one after
    /// the other on one thread leave it.
    #[test]
    fn twelve_threads_take_turns_on_one_heap() {
        if host().is_none() {
            return;
        }
        const SLOTS: u32 = 4096;
        const ROUNDS: u32 = 12;
        const SYNC: std::ops::Range<u32> = 0..2048;
        const PASS: std::ops::Range<u32> = 2048..2560;
        const SHORT: u32 = 96;
        let base = 0x4a_0000_0000u64;
        let views = std::sync::Arc::new(turn_views(base));
        let fast = (1 << 12, true, false);
        let view_of = |views: &[View], slot: u32, round: u32| views[((slot * 7 + round * 13) % views.len() as u32) as usize].clone();
        // One thread's rounds over its slots: `who` 0 the sync, 1 the pass,
        // 2.. the one-slot makers.
        let rounds = move |heap: &HostHeap, start: u64, views: &[View], who: u32, rounds: std::ops::Range<u32>| {
            let handle = |i: u32| start + u64::from(i) * TITLE_SLOT_BYTES;
            let slots: std::ops::Range<u32> = match who {
                0 => SYNC,
                1 => PASS,
                t => PASS.end + (t - 2) * SHORT..PASS.end + (t - 1) * SHORT,
            };
            for round in rounds {
                for s in slots.clone() {
                    write_record(start, s, &view_of(views, s, round));
                }
                match who {
                    0 => heap.mirror_due(&slots.clone().map(|s| (s, handle(s))).collect::<Vec<_>>()),
                    1 => {
                        let all: Vec<u32> = slots.clone().collect();
                        for chunk in all.chunks(PASS_CHUNK) {
                            let ahead: Vec<(u32, u64)> = chunk.iter().map(|&i| (i, handle(i))).collect();
                            let mut lookups: Vec<Lookups> = ahead.iter().map(|_| Lookups::kept()).collect();
                            let _ = heap.mirror_chunk(&ahead, &mut lookups, true);
                        }
                    }
                    _ => {
                        for s in slots.clone() {
                            let _ = heap.mirror(s, handle(s), false);
                        }
                    }
                }
            }
        };
        let end = PASS.end + 10 * SHORT;
        // Twelve threads at once, turns on.
        let (title, start, heap) = heap_with(SLOTS, fast).unwrap();
        let threads: Vec<_> = (0..12u32)
            .map(|who| {
                let views = views.clone();
                std::thread::spawn(move || {
                    TEST_FAST.set(Some(fast));
                    TEST_TURNS.set(Some(true));
                    rounds(heap, start, &views, who, 0..ROUNDS);
                })
            })
            .collect();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        for (t, thread) in threads.into_iter().enumerate() {
            while !thread.is_finished() {
                assert!(std::time::Instant::now() < deadline, "thread {t} still running: a deadlock");
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            thread.join().unwrap();
        }
        let together = keys_and_kept(heap, 0..end);
        // The same rounds on one thread, turns off.
        let (one_title, one_start, one) = heap_with(SLOTS, fast).unwrap();
        TEST_TURNS.set(Some(false));
        for round in 0..ROUNDS {
            for who in 0..12 {
                rounds(one, one_start, &views, who, round..round + 1);
            }
        }
        TEST_TURNS.set(None);
        let alone = keys_and_kept(one, 0..end);
        TEST_FAST.set(None);
        for (slot, (a, b)) in alone.iter().zip(&together).enumerate() {
            assert_eq!(a, b, "slot {slot}");
        }
        assert_eq!(release(title), 0);
        assert_eq!(release(one_title), 0);
    }

    /// VIcky (`VICTORY_MADE_PASS_FIRST`, [`made_first`]): one thread's rounds
    /// over slots of its own: the heap sync's chunks (`who` 0), or the pass
    /// over buffer views' chunks made alone, on a thread counted as a
    /// pass's (`who` 1), with the pass-first switch as `pass_first` (on,
    /// hand-over) for this thread.
    fn pass_first_rounds(
        heap: &HostHeap,
        start: u64,
        views: &[View],
        who: u32,
        slots: std::ops::Range<u32>,
        rounds: std::ops::Range<u32>,
        pass_first: Option<(bool, bool)>,
    ) {
        made_first::TEST_PASS_FIRST.set(pass_first);
        let handle = |i: u32| start + u64::from(i) * TITLE_SLOT_BYTES;
        for round in rounds {
            for s in slots.clone() {
                write_record(start, s, &views[((s * 7 + round * 13) % views.len() as u32) as usize]);
            }
            if who == 0 {
                heap.mirror_due(&slots.clone().map(|s| (s, handle(s))).collect::<Vec<_>>());
                continue;
            }
            let taking = crate::draw::heap_pass::parts::Taking::begin();
            let all: Vec<u32> = slots.clone().collect();
            for chunk in all.chunks(PASS_CHUNK) {
                let ahead: Vec<(u32, u64)> = chunk.iter().map(|&i| (i, handle(i))).collect();
                let mut lookups: Vec<Lookups> = ahead.iter().map(|_| Lookups::kept()).collect();
                let _ = heap.mirror_chunk(&ahead, &mut lookups, true);
            }
            let _ = taking.end();
        }
        made_first::TEST_PASS_FIRST.set(None);
    }

    /// VIcky (`VICTORY_MADE_PASS_FIRST`): the heap sync's chunks on one
    /// thread and the pass's on another, at once, with the pass going first
    /// (it spins, and the sync's chunks hand `made` to it): every slot ends
    /// as the same rounds made one after the other with the switch off leave
    /// it (records, held resources); the pass said it waited, and the sync
    /// handed `made` over. No deadlock.
    #[test]
    fn the_pass_first_leaves_every_slot_as_without_it() {
        if host().is_none() {
            return;
        }
        const SLOTS: u32 = 4096;
        const ROUNDS: u32 = 24;
        const SYNC: std::ops::Range<u32> = 0..3072;
        const PASS: std::ops::Range<u32> = 3072..4096;
        let base = 0x4c_0000_0000u64;
        let views = std::sync::Arc::new(turn_views(base));
        let fast = (1 << 12, true, false);
        let (waits, turns) = (
            made_first::PASS_WAITS.load(std::sync::atomic::Ordering::Relaxed),
            made_first::PASS_TURNS.load(std::sync::atomic::Ordering::Relaxed),
        );
        // Both at once, the pass first.
        let (title, start, heap) = heap_with(SLOTS, fast).unwrap();
        let threads: Vec<_> = [(0u32, SYNC), (1, PASS)]
            .into_iter()
            .map(|(who, slots)| {
                let views = views.clone();
                std::thread::spawn(move || {
                    TEST_FAST.set(Some(fast));
                    pass_first_rounds(heap, start, &views, who, slots, 0..ROUNDS, Some((true, true)));
                })
            })
            .collect();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        for (t, thread) in threads.into_iter().enumerate() {
            while !thread.is_finished() {
                assert!(std::time::Instant::now() < deadline, "thread {t} still running: a deadlock");
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            thread.join().unwrap();
        }
        let together = keys_and_kept(heap, 0..SLOTS);
        assert_eq!(heap.turns.pass_waiting.load(std::sync::atomic::Ordering::SeqCst), 0, "a wait left said");
        // (A host view heap is about a million slots: one at a time.)
        assert_eq!(release(title), 0);
        // The same rounds one after the other, switch off.
        let (one_title, one_start, one) = heap_with(SLOTS, fast).unwrap();
        for round in 0..ROUNDS {
            pass_first_rounds(one, one_start, &views, 0, SYNC, round..round + 1, Some((false, false)));
            pass_first_rounds(one, one_start, &views, 1, PASS, round..round + 1, Some((false, false)));
        }
        let alone = keys_and_kept(one, 0..SLOTS);
        TEST_FAST.set(None);
        for (slot, (a, b)) in alone.iter().zip(&together).enumerate() {
            assert_eq!(a, b, "slot {slot}");
        }
        assert!(made_first::PASS_WAITS.load(std::sync::atomic::Ordering::Relaxed) > waits, "the pass never said it waited");
        assert!(made_first::PASS_TURNS.load(std::sync::atomic::Ordering::Relaxed) > turns, "the sync never handed made over");
        assert_eq!(release(one_title), 0);
    }

    /// Micro-benchmark (`VICTORY_MADE_PASS_FIRST`): one thread makes 8,192
    /// slots in the heap sync's chunks over and over (records rewritten each
    /// round) while another, counted as the pass's, makes 2,048 slots of its
    /// own in the pass's chunks over and over; the pass's wait for `made` a
    /// chunk and the sync's slots a second, off, spinning only, and
    /// spinning with the sync's hand-over:
    /// `cargo test -p victory-gpu --profile diag --lib descriptor::host::fast_tests::bench_pass_first -- --ignored --nocapture`.
    #[test]
    #[ignore = "a benchmark: run alone, with --nocapture"]
    fn bench_pass_first() {
        if host().is_none() {
            return;
        }
        const SLOTS: u32 = 1 << 14;
        const SYNC: u32 = 8192;
        const PASS: u32 = 2048;
        let base = 0x4d_0000_0000u64;
        let views = std::sync::Arc::new(turn_views(base));
        let fast = (1 << 12, true, false);
        let (title, start, heap) = heap_with(SLOTS, fast).unwrap();
        for (name, mode) in [("off", (false, false)), ("spin", (true, false)), ("spin + hand", (true, true)), ("off", (false, false)), ("spin", (true, false)), ("spin + hand", (true, true))] {
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let sync = {
                let (stop, views) = (stop.clone(), views.clone());
                std::thread::spawn(move || {
                    TEST_FAST.set(Some(fast));
                    let (mut slots, mut round) = (0u64, 0u32);
                    let began = std::time::Instant::now();
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        pass_first_rounds(heap, start, &views, 0, 0..SYNC, round..round + 1, Some(mode));
                        slots += u64::from(SYNC);
                        round += 1;
                    }
                    slots as f64 / began.elapsed().as_secs_f64()
                })
            };
            let pass = {
                let (stop, views) = (stop.clone(), views.clone());
                std::thread::spawn(move || {
                    TEST_FAST.set(Some(fast));
                    made_first::TEST_PASS_FIRST.set(Some(mode));
                    let handle = |i: u32| start + u64::from(i) * TITLE_SLOT_BYTES;
                    let (mut chunks, mut waited_ns, mut round) = (0u64, 0u64, 0u32);
                    let began = std::time::Instant::now();
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        for s in SYNC..SYNC + PASS {
                            write_record(start, s, &views[((s * 7 + round * 13) % views.len() as u32) as usize]);
                        }
                        let taking = crate::draw::heap_pass::parts::Taking::begin();
                        let all: Vec<u32> = (SYNC..SYNC + PASS).collect();
                        for chunk in all.chunks(PASS_CHUNK) {
                            let ahead: Vec<(u32, u64)> = chunk.iter().map(|&i| (i, handle(i))).collect();
                            let mut lookups: Vec<Lookups> = ahead.iter().map(|_| Lookups::kept()).collect();
                            let _ = heap.mirror_chunk(&ahead, &mut lookups, true);
                            chunks += 1;
                        }
                        waited_ns += taking.end().waited_ns;
                        round += 1;
                    }
                    made_first::TEST_PASS_FIRST.set(None);
                    (chunks, waited_ns, began.elapsed())
                })
            };
            std::thread::sleep(std::time::Duration::from_secs(2));
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            let rate = sync.join().unwrap();
            let (chunks, waited_ns, took) = pass.join().unwrap();
            println!(
                "pass-first bench, {name:<11}: sync {rate:.0} slots/s; the pass {chunks} chunks in {:.2} s, {:.1} us waited for made a chunk, {:.1} % of its time",
                took.as_secs_f64(),
                waited_ns as f64 / 1e3 / chunks.max(1) as f64,
                waited_ns as f64 / 1e9 / took.as_secs_f64() * 100.0,
            );
        }
        TEST_FAST.set(None);
        assert_eq!(release(title), 0);
    }

    /// Micro-benchmark (`VICTORY_MADE_TURNS`): one thread makes 8,192
    /// slots in the heap sync's chunks over and over while eleven threads
    /// each make one slot of their own every 50 us, as the title's
    /// recording threads do; the one-slot makers' time in `mirror` (the
    /// wait for `made` and the slot) and the sync's slots a second, without
    /// and with the turns:
    /// `cargo test -p victory-gpu --profile diag --lib descriptor::host::fast_tests::bench_turns -- --ignored --nocapture`.
    #[test]
    #[ignore = "a benchmark: run alone, with --nocapture"]
    fn bench_turns() {
        if host().is_none() {
            return;
        }
        const SLOTS: u32 = 1 << 14;
        const SYNC: u32 = 8192;
        let base = 0x4b_0000_0000u64;
        let views = std::sync::Arc::new(turn_views(base));
        let fast = (1 << 12, true, false);
        let (title, start, heap) = heap_with(SLOTS, fast).unwrap();
        let handle = move |i: u32| start + u64::from(i) * TITLE_SLOT_BYTES;
        for turns in [false, true, false, true] {
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let sync = {
                let (stop, views) = (stop.clone(), views.clone());
                std::thread::spawn(move || {
                    TEST_FAST.set(Some(fast));
                    TEST_TURNS.set(Some(turns));
                    let due: Vec<(u32, u64)> = (0..SYNC).map(|s| (s, handle(s))).collect();
                    let (mut slots, mut round) = (0u64, 0u32);
                    let began = std::time::Instant::now();
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        for s in 0..SYNC {
                            write_record(start, s, &views[((s * 7 + round * 13) % views.len() as u32) as usize]);
                        }
                        heap.mirror_due(&due);
                        slots += u64::from(SYNC);
                        round += 1;
                    }
                    slots as f64 / began.elapsed().as_secs_f64()
                })
            };
            let shorts: Vec<_> = (0..11u32)
                .map(|t| {
                    let (stop, views) = (stop.clone(), views.clone());
                    std::thread::spawn(move || {
                        TEST_FAST.set(Some(fast));
                        TEST_TURNS.set(Some(turns));
                        let mut times: Vec<u64> = Vec::new();
                        let mut n = 0u32;
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            let pause = std::time::Instant::now();
                            while pause.elapsed() < std::time::Duration::from_micros(50) {
                                std::hint::spin_loop();
                            }
                            let s = SYNC + t * 64 + n % 64;
                            write_record(start, s, &views[((s + n) % views.len() as u32) as usize]);
                            let at = std::time::Instant::now();
                            let _ = heap.mirror(s, handle(s), false);
                            times.push(at.elapsed().as_nanos() as u64);
                            n += 1;
                        }
                        times
                    })
                })
                .collect();
            std::thread::sleep(std::time::Duration::from_secs(2));
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            let rate = sync.join().unwrap();
            let mut times: Vec<u64> = shorts.into_iter().flat_map(|s| s.join().unwrap()).collect();
            times.sort_unstable();
            let mean = times.iter().sum::<u64>() as f64 / times.len().max(1) as f64 / 1e3;
            let at = |q: f64| times.get(((times.len() as f64 * q) as usize).min(times.len().saturating_sub(1))).copied().unwrap_or(0) as f64 / 1e3;
            println!(
                "turns bench, turns {}: sync {:.0} slots/s; eleven one-slot makers: {} calls, mean {mean:.1} us, median {:.1}, p99 {:.1}, max {:.0}",
                if turns { "on " } else { "off" },
                rate,
                times.len(),
                at(0.5),
                at(0.99),
                at(1.0)
            );
        }
        TEST_FAST.set(None);
        assert_eq!(release(title), 0);
    }
}

#[cfg(test)]
mod raw_range_tests {
    use super::*;

    fn rr(size: u64, offset: u64, len: u64) -> (Option<(u64, u32)>, Option<Cut>) {
        let (range, cut) = raw_range(size, offset, len);
        (range.map(|v| (v.first, v.elements)), cut)
    }

    fn range(size: u64, offset: u64, len: u64) -> Option<(u64, u32)> {
        rr(size, offset, len).0
    }

    /// The table of the first review, plus the counterexample it found: 100
    /// bytes of buffer, an offset of 112 — the clamp this replaced viewed bytes
    /// 96 to 99, which the request had asked nothing about.
    #[test]
    fn the_range_stays_inside_the_buffer() {
        assert_eq!(range(1024, 0, 256), Some((0, 64)));
        assert_eq!(range(1024, 768, 512), Some((192, 64)));
        assert_eq!(range(1024, 1024, 16), None, "a start at the end");
        assert_eq!(range(1024, 1040, 16), None, "a start past the end");
        assert_eq!(range(100, 112, 16), None, "no intersection: never the tail");
        assert_eq!(range(100, 96, 16), Some((24, 1)), "the last dword");
        assert_eq!(range(100, 96, 4), Some((24, 1)));
        assert_eq!(range(0, 0, 16), None, "an empty buffer");
        assert_eq!(range(3, 0, 16), None, "no whole dword");
        assert_eq!(range(1024, 0, 0), Some((0, 1)), "the floor of one element stands");
        assert_eq!(range(1024, 0, 1), Some((0, 1)), "as does a length under a dword");
        assert_eq!(range(1024, 3, 16), None, "not on 16 bytes");
        assert_eq!(range(1024, u64::MAX, 16), None, "no wrap, no clamp back");
        assert_eq!(range(1024, 0, u64::MAX), Some((0, 256)), "the length is cut");
    }

    /// Why a request was refused or cut, which is what the counters report.
    #[test]
    fn the_cause_is_the_one_the_counters_would_report() {
        assert_eq!(rr(100, 112, 16), (None, Some(Cut::Past)));
        assert_eq!(rr(1024, 3, 16), (None, Some(Cut::Unaligned)));
        assert_eq!(rr(1024, 0, 0), (Some((0, 1)), Some(Cut::Short)));
        assert_eq!(rr(1024, 768, 512), (Some((192, 64)), Some(Cut::Length)));
        // Nothing cut, nothing refused: no counter is touched at all.
        assert_eq!(rr(1024, 0, 256), (Some((0, 64)), None));
        assert_eq!(rr(1024, 0, 1020), (Some((0, 255)), None), "255 of 256 dwords");
    }

    /// `FirstElement` is 64-bit: a start past the 32-bit element index is kept
    /// as it is, not narrowed to `u32::MAX`.
    #[test]
    fn a_start_past_32_bits_is_kept() {
        let size = 1u64 << 40;
        let offset = (u64::from(u32::MAX) + 10) * 16;
        let range = range(size, offset, 64).expect("inside the buffer");
        assert_eq!(range.0, offset / 4);
        assert!(range.0 > u64::from(u32::MAX));
        assert_eq!(range.1, 16);
    }

    /// `NumElements` is 32-bit: the length alone is cut at `u32::MAX`.
    #[test]
    fn the_length_is_cut_at_32_bits() {
        let size = 1u64 << 40;
        assert_eq!(range(size, 0, u64::MAX), Some((0, u32::MAX)));
    }

    /// An aligned start far past a buffer is still refused: alignment is not a
    /// reason to accept a request, and the check is not reached by the
    /// unaligned case first.
    #[test]
    fn an_aligned_start_past_the_buffer_is_refused() {
        let size = 1u64 << 40;
        assert_eq!(rr(size, 1u64 << 60, 64), (None, Some(Cut::Past)));
        assert_eq!(rr(size, size, 64), (None, Some(Cut::Past)));
        assert_eq!(rr(size, size - 16, 64), (Some((size / 4 - 4, 4)), Some(Cut::Length)));
    }

    /// Every accepted range holds the four things the patch promises, and no
    /// request that the input rules allow is turned away for no reason.
    #[test]
    fn accepted_ranges_hold_their_invariant() {
        let sizes = [0u64, 1, 3, 4, 100, 1024, 4096, 1 << 30];
        let offsets = [0u64, 4, 16, 48, 96, 112, 1024, 1040, 4096, u64::MAX];
        let lens = [0u64, 1, 3, 4, 16, 512, 1 << 20, u64::MAX];
        for size in sizes {
            for offset in offsets {
                for len in lens {
                    let (range, _) = raw_range(size, offset, len);
                    let aligned = offset.is_multiple_of(16);
                    let inside = offset / 4 < size / 4;
                    match range {
                        Some(range) => {
                            assert!(range.elements > 0, "{size} {offset} {len}");
                            assert_eq!(
                                range.first,
                                offset / 4,
                                "{size} {offset} {len}: the start moved"
                            );
                            assert_eq!(range.first % 4, 0, "a raw view starts on 16 bytes");
                            assert!(
                                range.first + u64::from(range.elements) <= size / 4,
                                "{size} {offset} {len}: {} + {} past the buffer",
                                range.first,
                                range.elements
                            );
                        }
                        None => assert!(
                            !(aligned && inside),
                            "{size} {offset} {len}: refused though the rules allow it"
                        ),
                    }
                }
            }
        }
    }
}

/// VIcky (`VICTORY_TABLE_BOUND`, stage 1 of the structural fps design,
/// [`left`]): the marks of tables made afresh past the identity window, the
/// phases of `alt`, the flip's sweep and the counters.
#[cfg(test)]
mod table_bound_tests {
    use super::*;

    /// A title view heap of `count` slots and its first descriptor.
    pub(super) fn title_heap(count: u32) -> (crate::object::This, u64) {
        #[repr(C)]
        struct Extended(super::super::HeapDesc, super::super::Placement);
        let desc = super::super::HeapDesc {
            kind: 0,
            count,
            flags: 1,
            nodes: 0,
        };
        let extended = Extended(desc, super::super::Placement { address: 0, size: 0 });
        let mut out = std::ptr::null();
        let made = super::super::create_heap(std::ptr::null(), &extended.0, &super::super::HEAP_IIDS[0], &mut out);
        assert_eq!(made, crate::object::S_OK);
        (out, super::super::heap(out).start)
    }

    pub(super) fn release(object: crate::object::This) -> u32 {
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(object, 0x10)(object)
    }

    /// A title heap of 64 slots whose host heap has a heap map (an identity
    /// window of 16, the pool up to 24), made with the switch as `bound`
    /// (set, live) on this thread, which stays so until it is set again.
    pub(super) fn mapped_heap(bound: (bool, bool)) -> Option<(crate::object::This, u64, &'static HostHeap)> {
        host()?;
        left::TEST_BOUND.set(Some(bound));
        let (title, start) = title_heap(64);
        TEST_MAP.set(Some((16, 24)));
        let heap = super::super::heap(title).host();
        TEST_MAP.set(None);
        Some((title, start, heap?))
    }

    /// A placed buffer view of 64 KiB at `address`.
    pub(super) fn buffer_view(address: u64) -> View {
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, 0x1_0000);
        let mut desc = [0u32; 16];
        (desc[1], desc[2], desc[4], desc[6], desc[7]) = (1, 5768, 0, 64, 16);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, resource, desc)
    }

    /// A placed 2D texture view at `address` (64x64 BC7, one level).
    pub(super) fn texture_view(address: u64) -> View {
        let resource = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
        let mut desc = [0u32; 16];
        (desc[0], desc[1], desc[2], desc[5]) = (98, 4, 5768, 1);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, resource, desc)
    }

    /// Writes `view`'s record (or none: zeros) into title slot `slot` as
    /// bytes, as a title that copies records does; its key.
    pub(super) fn write_record(start: u64, slot: u32, view: Option<&View>) -> u64 {
        let handle = start + u64::from(slot) * TITLE_SLOT_BYTES;
        let Some(view) = view else {
            // SAFETY: a descriptor slot of the title heap.
            unsafe { (handle as *mut u8).write_bytes(0, TITLE_SLOT_BYTES as usize) };
            return 0;
        };
        let [i, generation] = super::super::words_for(view.clone());
        // SAFETY: a descriptor slot of the title heap.
        unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
        u64::from(generation) << 32 | u64::from(i)
    }

    /// The heap sync's look at title slot `slot` ([`HostHeap::mirror_copied`]):
    /// whether it mirrored the record there.
    pub(super) fn sync(heap: &HostHeap, start: u64, slot: u32) -> bool {
        let handle = start + u64::from(slot) * TITLE_SLOT_BYTES;
        // SAFETY: a descriptor slot of the title heap.
        let record = unsafe { std::slice::from_raw_parts(handle as *const u8, TITLE_SLOT_BYTES as usize) }.to_vec();
        heap.mirror_copied(slot, handle, &record)
    }

    /// The slots of the queue of textures picked by index due at `frame`.
    fn due(heap: &HostHeap, frame: u64) -> Vec<u32> {
        let mut due: Vec<u32> = heap.texture_slots_due(frame, 64).0.into_iter().map(|d| d.0).collect();
        due.sort_unstable();
        due
    }

    /// As a draw that made title slot `slot` from the record it holds: its
    /// host slot's key.
    pub(super) fn made_by_draw(heap: &HostHeap, slot: u32, key: u64) {
        let made = heap.made_alone();
        let host = made.host(slot).expect("a pool slot");
        made.set_key(host, key);
    }

    /// A table made afresh past the identity window marks its title slots
    /// (texture marks, and buffer marks) only with the switch; one past the
    /// title slots the host heap mirrors never does. Texture marks last 240
    /// frames, buffer marks 8.
    #[test]
    fn tables_past_the_window_mark_their_slots_with_the_switch() {
        for set in [false, true] {
            let Some((title, _, heap)) = mapped_heap((set, true)) else {
                left::TEST_BOUND.set(None);
                return;
            };
            assert_eq!((heap.count(), heap.mirrored()), (16, 64));
            assert_eq!(heap.buffer_marks.len(), if set { 64 } else { 0 });
            let frame = 1_000;
            let first = left::title_first(Some(40), 4, heap.count(), heap.mirrored());
            assert_eq!(first, set.then_some(40), "past the window");
            assert_eq!(left::title_first(Some(62), 4, heap.count(), heap.mirrored()), None, "past the mirrored slots");
            assert_eq!(left::title_first(Some(8), 4, heap.count(), heap.mirrored()), Some(8), "in the window");
            if let Some(first) = first {
                heap.tabled_in(first + 1, true, frame);
                heap.slot_tabled_in(first + 1, frame);
            }
            assert_eq!(heap.tabled_lately(41, frame), set);
            assert_eq!(heap.buffer_marked_lately(41, frame), set);
            assert_eq!(heap.tabled_lately(41, frame + TABLED_FRAMES), set);
            assert!(!heap.tabled_lately(41, frame + TABLED_FRAMES + 1), "expired");
            assert_eq!(heap.buffer_marked_lately(41, frame + u64::from(left::FRAMES)), set);
            assert!(!heap.buffer_marked_lately(41, frame + u64::from(left::FRAMES) + 1), "expired");
            assert_eq!(release(title), 0);
        }
        left::TEST_BOUND.set(None);
    }

    /// In the off phase of `alt` the marks are set but nothing acts on
    /// them: a read takes no slot out of the queue, a write waits for no
    /// draw, the heap sync leaves nothing. In the ON phase, as `1`.
    #[test]
    fn marks_act_only_in_the_on_phase() {
        let Some((title, start, heap)) = mapped_heap((true, true)) else {
            left::TEST_BOUND.set(None);
            return;
        };
        let frame = 2_000;
        heap.queue_texture(40, true, false, frame);
        heap.tabled_in(40, true, frame);
        assert!(due(heap, frame).is_empty(), "read through a table: out of the queue");
        heap.queue_texture(40, true, false, frame + 1);
        assert!(due(heap, frame + 1).is_empty(), "a write waits for its draw");
        left::TEST_BOUND.set(Some((true, false)));
        heap.queue_texture(41, true, false, frame);
        heap.tabled_in(41, true, frame);
        assert_eq!(due(heap, frame), [41], "left in the queue in the off phase");
        assert!(heap.tabled_marked(41, frame) && !heap.tabled_lately(41, frame), "marked, not acted on");
        heap.queue_texture(40, true, false, frame + 2);
        assert_eq!(due(heap, frame + 2), [40, 41], "a write due at once in the off phase");
        // The heap sync mirrors a new record in a marked slot in the off
        // phase, and leaves it in the ON phase.
        let now = crate::frame::current();
        heap.tabled_in(42, true, now);
        write_record(start, 42, Some(&texture_view(0x7fed_0000_0000)));
        assert!(sync(heap, start, 42), "off: mirrored");
        left::TEST_BOUND.set(Some((true, true)));
        write_record(start, 42, Some(&texture_view(0x7fed_0010_0000)));
        assert!(!sync(heap, start, 42), "ON: left to the draws");
        left::TEST_BOUND.set(None);
        assert_eq!(release(title), 0);
    }

    /// At the flip to off, a slot with a live mark that the heap sync left
    /// is mirrored as the sync would have mirrored it, and one its draw made
    /// already has its texture put back in the queue; the records are
    /// noted as seen. A flip sweeps once, and only from ON to off.
    #[test]
    fn the_flip_to_off_sweeps_the_marked_slots() {
        let Some((title, start, heap)) = mapped_heap((true, true)) else {
            left::TEST_BOUND.set(None);
            return;
        };
        let frame = crate::frame::current();
        // Slot 40: mirrored, then read through a table made afresh (out of
        // the queue), then written again: the sync leaves it.
        write_record(start, 40, Some(&texture_view(0x7fee_0000_0000)));
        assert!(sync(heap, start, 40));
        assert_eq!(due(heap, frame), [40]);
        heap.tabled_in(40, true, frame);
        write_record(start, 40, Some(&texture_view(0x7fee_0010_0000)));
        assert!(!sync(heap, start, 40), "left to the draws");
        // Slot 41: mirrored, made by the draw that read it, out of the queue.
        let key = write_record(start, 41, Some(&texture_view(0x7fee_0020_0000)));
        assert!(sync(heap, start, 41));
        made_by_draw(heap, 41, key);
        heap.tabled_in(41, true, frame);
        assert!(due(heap, frame).is_empty());
        // No flip while ON.
        assert_eq!(heap.table_bound_flip((start, TITLE_SLOT_BYTES)), None);
        left::TEST_BOUND.set(Some((true, false)));
        let swept = heap.table_bound_flip((start, TITLE_SLOT_BYTES)).expect("the flip to off sweeps");
        assert_eq!(swept, [2, 1, 1], "two marked, 40 mirrored, 41's texture back in the queue");
        assert_eq!(due(heap, frame), [40, 41]);
        assert!(!sync(heap, start, 40), "its record noted as seen");
        assert_eq!(heap.table_bound_flip((start, TITLE_SLOT_BYTES)), None, "once a flip");
        left::TEST_BOUND.set(Some((true, true)));
        assert_eq!(heap.table_bound_flip((start, TITLE_SLOT_BYTES)), None, "off to ON: nothing");
        left::TEST_BOUND.set(None);
        assert_eq!(release(title), 0);
    }

    /// The heap sync's new records are counted by the first rule of stages
    /// 2 and 3 that would take them; those a texture mark leaves are not
    /// (the sync line counts them). The pass's slots with a buffer mark too.
    #[test]
    fn the_sync_counts_its_records_by_rule() {
        let Some((title, start, heap)) = mapped_heap((true, true)) else {
            left::TEST_BOUND.set(None);
            return;
        };
        left::TEST_COUNTS.set(Some(Default::default()));
        let frame = crate::frame::current();
        // (Below 1 TiB: placed views named whole in their records.)
        let at = |n: u64| 0x6b_0000_0000 + n * 0x10_0000;
        // A buffer mark would leave 40; the grace would hold 41 and 43.
        heap.slot_tabled_in(40, frame);
        write_record(start, 40, Some(&buffer_view(at(0))));
        write_record(start, 41, Some(&buffer_view(at(1))));
        write_record(start, 43, Some(&texture_view(at(3))));
        for slot in [40, 41, 43] {
            assert!(sync(heap, start, slot));
        }
        // No view: into a slot that shows none (44), one that shows one (45).
        write_record(start, 44, Some(&buffer_view(at(4))));
        let key = write_record(start, 45, Some(&buffer_view(at(5))));
        assert!(sync(heap, start, 44) && sync(heap, start, 45));
        made_by_draw(heap, 45, key);
        write_record(start, 44, None);
        write_record(start, 45, None);
        assert!(sync(heap, start, 44) && sync(heap, start, 45));
        // A texture mark's slot: left, not counted by rule.
        heap.tabled_in(46, true, frame);
        write_record(start, 46, Some(&texture_view(at(6))));
        assert!(!sync(heap, start, 46));
        heap.count_pass_marks(&[3, 40, 41], frame);
        let counts = left::TEST_COUNTS.take().expect("counted");
        let get = |what: left::C| counts[what as usize];
        assert_eq!(get(left::C::WouldLeave), 1);
        assert_eq!(get(left::C::WouldHold), 4, "41, 43, and 44 and 45 first: {counts:?}");
        assert_eq!(get(left::C::NoViewNone), 1);
        assert_eq!(get(left::C::NoViewShown), 1);
        assert_eq!(get(left::C::TexturesMade) + get(left::C::BuffersMade) + get(left::C::OthersMade), 0);
        assert_eq!((get(left::C::PassSlots), get(left::C::PassMarked)), (3, 1));
        left::TEST_BOUND.set(None);
        assert_eq!(release(title), 0);
    }

    /// The slots of the queue due at `frame`, with their write numbers.
    fn due_writes(heap: &HostHeap, frame: u64) -> Vec<(u32, u64)> {
        let mut due = heap.texture_slots_due(frame, 64).0;
        due.sort_unstable();
        due
    }

    /// As the pass that brought every slot due at `frame` up to date: out
    /// of the queue.
    fn all_done(heap: &HostHeap, frame: u64) {
        for (slot, write) in due_writes(heap, frame) {
            heap.texture_slot_done(slot, write, None);
        }
    }

    /// `VICTORY_TABLE_BOUND_FRESH`: a slot a table only spans (no draw reads
    /// it through the table) gets no mark with it, so its write is due at
    /// once and the heap sync mirrors its new record, as for any slot picked
    /// by index; without it (stage 1 as first measured) the mark leaves it
    /// to the draws. A read leaves a slot to the draws either way.
    #[test]
    fn with_fresh_only_a_read_leaves_a_slot_to_the_draws() {
        for fresh in [false, true] {
            let Some((title, start, heap)) = mapped_heap((true, true)) else {
                left::TEST_BOUND.set(None);
                return;
            };
            left::TEST_FRESH.set(Some(fresh));
            let frame = crate::frame::current();
            let at = |n: u64| 0x7fec_0000_0000 + n * 0x10_0000 + if fresh { 0x1_0000_0000 } else { 0 };
            // 41: only spanned by a table made afresh.
            assert!(!heap.tabled_in(41, false, frame), "never a first read");
            assert_eq!(heap.tabled_marked(41, frame), !fresh);
            heap.queue_texture(41, true, false, frame);
            assert_eq!(due(heap, frame), if fresh { vec![41] } else { vec![] }, "a write due at once with FRESH");
            write_record(start, 41, Some(&texture_view(at(1))));
            assert_eq!(sync(heap, start, 41), fresh, "the sync mirrors it with FRESH, leaves it without");
            // 42: read through a table: the draws' with or without FRESH.
            assert!(heap.tabled_in(42, true, frame), "the first read of the frame");
            assert!(!heap.tabled_in(42, true, frame), "once a frame");
            write_record(start, 42, Some(&texture_view(at(2))));
            assert!(!sync(heap, start, 42), "left to the draws");
            left::TEST_FRESH.set(None);
            left::TEST_BOUND.set(None);
            assert_eq!(release(title), 0);
        }
    }

    /// `VICTORY_TABLE_BOUND_FRESH`: a texture slot a draw reads through a
    /// table (out of the queue) whose memory changes after its upload is
    /// made again: a write behind the guards (`VICTORY_PAGE_WRITES` bit 2)
    /// and a remap make it due, and so does the re-due that follows a
    /// refresh that missed a page (`VICTORY_NO_WAIT_FALLBACK`, through
    /// [`remapped`]), and the DirectStorage hold of the pass's look applies
    /// to it once it is due. Its placed memory stays in the spans when the
    /// draw makes it in place from a new record. Without FRESH, as stage 1
    /// was first measured: left to the draws, nothing makes it due.
    #[test]
    fn with_fresh_a_table_read_slot_is_made_again_when_its_memory_changes() {
        for fresh in [false, true] {
            let Some((title, start, heap)) = mapped_heap((true, true)) else {
                left::TEST_BOUND.set(None);
                return;
            };
            left::TEST_FRESH.set(Some(fresh));
            let frame = crate::frame::current();
            let base = 0x7feb_0000_0000 + if fresh { 0x1_0000_0000 } else { 0 };
            let (a, b) = (base, base + 0x10_0000);
            // Mirrored by the sync (its memory in the spans), then read
            // through a table: out of the queue.
            write_record(start, 40, Some(&texture_view(a)));
            assert!(sync(heap, start, 40));
            assert_eq!(due(heap, frame), [40]);
            heap.tabled_in(40, true, frame);
            assert!(due(heap, frame).is_empty(), "read through a table: out of the queue");
            // DirectStorage wrote its bytes behind the guards.
            let held = heap.slots_written_under(&[(a, a + 0x1000)], frame, true);
            assert_eq!(held, if fresh { [1, 1, 1] } else { [0, 0, 0] });
            assert_eq!(due(heap, frame + PAGE_WRITE_AFTER), if fresh { vec![40] } else { vec![] });
            if fresh {
                // Due: the pass's look holds it under a write under way.
                let [(slot, write)] = due_writes(heap, frame + PAGE_WRITE_AFTER)[..] else {
                    panic!("one due slot");
                };
                assert_eq!(slot, 40);
                assert!(heap.texture_slot_ds_hold(40, write, frame + PAGE_WRITE_AFTER + 1, crate::texstale::DS_HOLD_MOST));
                assert!(due(heap, frame + PAGE_WRITE_AFTER).is_empty(), "held a frame");
                assert_eq!(due(heap, frame + PAGE_WRITE_AFTER + 1), [40]);
            }
            // Made again (the pass found it staged, or uploaded it); then a
            // remap, or a refresh that found a page missing, under its
            // memory.
            all_done(heap, frame + PAGE_WRITE_AFTER + 1);
            let n = heap.slots_remapped_in(&[(a + 0x100, a + 0x200)], frame);
            assert_eq!(n, usize::from(fresh));
            assert_eq!(due(heap, frame + 1), if fresh { vec![40] } else { vec![] });
            all_done(heap, frame + 1);
            // Its draw makes the slot in place from a new record: with FRESH
            // its new memory is in the spans, so a write there makes it due.
            let key = write_record(start, 40, Some(&texture_view(b)));
            assert!(!sync(heap, start, 40), "left to the draws");
            let handle = start + 40 * TITLE_SLOT_BYTES;
            heap.mirror(40, handle, true);
            let made = heap.made_alone();
            let shown = made.host(40).and_then(|host| made.key_of(host));
            drop(made);
            assert_eq!(shown, Some(key), "made in place by its draw");
            let in_spans = !crate::lot4::locks::mutex(crate::lot4::locks::Lock::Placed, &*heap.placed).slots_in(b, b + 1).is_empty();
            assert_eq!(in_spans, fresh, "kept in the spans with FRESH");
            all_done(heap, frame);
            let held = heap.slots_written_under(&[(b, b + 0x1000)], frame, true);
            assert_eq!(held[2], usize::from(fresh));
            left::TEST_FRESH.set(None);
            left::TEST_BOUND.set(None);
            assert_eq!(release(title), 0);
        }
    }

    /// `VICTORY_TABLE_BOUND_FRESH` with `=alt`: at the flip to off the slots
    /// draws read are swept as before (40 left by the sync and mirrored, 41
    /// made by its draw and put back in the queue), a slot a table only
    /// spanned has no mark to sweep (its record was mirrored when written),
    /// and the slots stay found by memory in the off phase: a write behind
    /// the guards there makes 41 due again. Nothing is left stale.
    #[test]
    fn with_fresh_the_flip_to_off_leaves_nothing_stale() {
        let Some((title, start, heap)) = mapped_heap((true, true)) else {
            left::TEST_BOUND.set(None);
            return;
        };
        left::TEST_FRESH.set(Some(true));
        let frame = crate::frame::current();
        let at = |n: u64| 0x7fea_0000_0000 + n * 0x10_0000;
        write_record(start, 40, Some(&texture_view(at(0))));
        assert!(sync(heap, start, 40));
        heap.tabled_in(40, true, frame);
        write_record(start, 40, Some(&texture_view(at(10))));
        assert!(!sync(heap, start, 40), "left to the draws");
        let key = write_record(start, 41, Some(&texture_view(at(1))));
        assert!(sync(heap, start, 41));
        made_by_draw(heap, 41, key);
        heap.tabled_in(41, true, frame);
        heap.tabled_in(42, false, frame);
        write_record(start, 42, Some(&texture_view(at(2))));
        assert!(sync(heap, start, 42), "only spanned: mirrored");
        assert_eq!(due(heap, frame), [42]);
        left::TEST_BOUND.set(Some((true, false)));
        let swept = heap.table_bound_flip((start, TITLE_SLOT_BYTES)).expect("the flip to off sweeps");
        assert_eq!(swept, [2, 1, 1], "40 and 41 marked (not 42): 40 mirrored, 41 back in the queue");
        assert_eq!(due(heap, frame), [40, 41, 42]);
        all_done(heap, frame);
        let n = heap.slots_written_under(&[(at(1), at(1) + 0x1000), (at(10), at(10) + 0x1000)], frame, true);
        assert_eq!(n[2], 2, "41 and the new memory of 40 found in the off phase");
        left::TEST_FRESH.set(None);
        left::TEST_BOUND.set(None);
        assert_eq!(release(title), 0);
    }
}

#[cfg(test)]
mod tables_left_tests {
    use super::table_bound_tests::{buffer_view, mapped_heap, release, sync, texture_view, write_record};
    use super::*;
    use left::{Ck, Mode2, TEST_BOUND, TEST_CHECKS, TEST_LEFT};
    use std::sync::Arc;

    /// A title heap of 64 slots whose host heap has a heap map (window 16),
    /// made with stage 2 as `mode` (`live`, `grace`, the hold on) and stage
    /// 1's bound on this thread, which stays so until [`done`].
    fn heap2(mode: Mode2, live: bool, grace: bool) -> Option<(crate::object::This, u64, &'static HostHeap)> {
        TEST_LEFT.set(Some((mode, live, grace, true)));
        let made = mapped_heap((true, true));
        if made.is_none() {
            done();
        }
        made
    }

    /// Stage 2 now as `mode` (`live`, `grace`, `hold`) on this thread.
    fn now(mode: Mode2, live: bool, grace: bool, hold: bool) {
        TEST_LEFT.set(Some((mode, live, grace, hold)));
    }

    fn done() {
        TEST_LEFT.set(None);
        TEST_BOUND.set(None);
        crate::shadow::TEST_IDLE.set(None);
    }

    fn handle(start: u64, slot: u32) -> u64 {
        start + u64::from(slot) * TITLE_SLOT_BYTES
    }

    /// The record a host slot shows (0: none, or not made).
    fn shows(heap: &HostHeap, slot: u32) -> u64 {
        let made = heap.made_alone();
        made.host(slot).and_then(|host| made.key_of(host)).unwrap_or(0)
    }

    /// Stage 2's rule at the heap sync: a buffer or constant view in a slot
    /// with a buffer mark of the last 8 frames is left to the draws (its
    /// print not noted, its left bit set, looked at again with its page); a
    /// texture view, a record of no view, a slot without a live mark, and
    /// the off phase of `alt` are mirrored as before. The pass over buffer
    /// views takes the marked slots out of its list, in ON phases only.
    #[test]
    fn marked_buffer_and_constant_records_are_left_and_the_rest_mirrored() {
        let Some((title, start, heap)) = heap2(Mode2::Alt, true, false) else {
            return;
        };
        let frame = crate::frame::current();
        let at = |n: u64| 0x6c_0000_0000 + n * 0x10_0000;
        // 45 mirrored before any mark, then marked.
        write_record(start, 45, Some(&buffer_view(at(5))));
        assert!(sync(heap, start, 45));
        for slot in [40, 41, 42, 45] {
            heap.slot_tabled_in(slot, frame);
        }
        // 46's mark is too old.
        heap.slot_tabled_in(46, frame.wrapping_sub(20));
        write_record(start, 40, Some(&buffer_view(at(0))));
        write_record(start, 41, Some(&View::Constant(at(1), 256)));
        write_record(start, 42, Some(&texture_view(at(2))));
        write_record(start, 44, Some(&buffer_view(at(4))));
        write_record(start, 46, Some(&buffer_view(at(6))));
        assert!(!sync(heap, start, 40), "a buffer view in a marked slot: left");
        assert!(!sync(heap, start, 41), "a constant view in a marked slot: left");
        assert!(sync(heap, start, 42), "a texture view: stage 1's rules, mirrored here");
        assert!(sync(heap, start, 44), "no mark: mirrored");
        assert!(sync(heap, start, 46), "an expired mark: mirrored");
        assert!(!sync(heap, start, 40), "looked at again with its page, left again");
        // A record of no view in a marked slot empties it, as before.
        write_record(start, 45, None);
        assert!(sync(heap, start, 45));
        assert_eq!(heap.left_state().bits_take(|_| true), [40, 41], "the left bits");
        // The pass: the marked slots out of its list.
        let mut slots = vec![3, 40, 44, 45];
        assert_eq!(heap.leave_to_draws(&mut slots, frame), [40, 45]);
        assert_eq!(slots, [3, 44]);
        // The off phase: mirrored, and the pass keeps its slots.
        now(Mode2::Alt, false, false, true);
        write_record(start, 41, Some(&View::Constant(at(7), 256)));
        assert!(sync(heap, start, 41), "off: mirrored");
        let mut slots = vec![3, 40, 44];
        assert!(heap.leave_to_draws(&mut slots, frame).is_empty());
        assert_eq!(slots, [3, 40, 44]);
        done();
        assert_eq!(release(title), 0);
    }

    /// `VICTORY_TABLES_LEFT_GRACE=1`: a new constant or placed buffer record
    /// in a slot without a mark waits four frames (its print noted); then it
    /// is mirrored, unless a draw's table claimed its slot meanwhile, which
    /// leaves it to the draws as if it had been left when found. A placed
    /// texture is not held. Without the grace, mirrored at once.
    #[test]
    fn the_grace_holds_new_buffer_records_until_a_draw_claims_them() {
        let Some((title, start, heap)) = heap2(Mode2::On, true, true) else {
            return;
        };
        let frame = crate::frame::current();
        let at = |n: u64| 0x6d_0000_0000 + n * 0x10_0000;
        let span = (start, TITLE_SLOT_BYTES);
        write_record(start, 41, Some(&buffer_view(at(1))));
        write_record(start, 42, Some(&View::Constant(at(2), 256)));
        write_record(start, 43, Some(&texture_view(at(3))));
        assert!(!sync(heap, start, 41), "waits");
        assert!(!sync(heap, start, 42), "waits");
        assert!(sync(heap, start, 43), "a placed texture is not held");
        assert!(!sync(heap, start, 41), "its print noted: nothing new");
        assert!(heap.buffer_slots().is_empty(), "nothing noted yet");
        // A draw's table claims 42 during its grace.
        heap.slot_tabled_in(42, frame + 1);
        assert_eq!(heap.mirror_waited_in(frame + left::GRACE - 1, span, false), 0, "not yet");
        assert_eq!(heap.mirror_waited_in(frame + left::GRACE, span, false), 1, "41 mirrored, 42 claimed");
        assert_eq!(heap.buffer_slots(), [41]);
        assert!(!sync(heap, start, 42), "claimed: left to the draws, looked at again");
        assert_eq!(heap.left_state().bits_take(|_| true), [42]);
        // Without the grace: mirrored at once.
        now(Mode2::On, true, false, true);
        write_record(start, 44, Some(&buffer_view(at(4))));
        assert!(sync(heap, start, 44));
        done();
        assert_eq!(release(title), 0);
    }

    /// The hold keeps copies: a window copy a draw read through a slot left
    /// to the draws is held by that slot, so `let_go_idle` does not let it go
    /// once idle (as the slot's host slot kept it without the switch). Once
    /// the title writes another record there, the hold goes, and the idle
    /// copy is let go (and made again at the next read). With
    /// `VICTORY_TABLES_LEFT_HOLD=0`, or in the off phase, no hold is taken.
    #[test]
    fn the_hold_keeps_copies() {
        let Some((title, start, heap)) = heap2(Mode2::Alt, true, false) else {
            return;
        };
        let frame = crate::frame::current();
        let window = 0x7ff3_4000_0000u64;
        let copy = |n: u64| crate::shadow::find(window + n * 0x40_0000, 0x100).expect("a window copy");
        let let_go_idle = |id: u64| {
            crate::shadow::TEST_IDLE.set(Some(id));
            crate::shadow::let_go_idle();
            crate::shadow::TEST_IDLE.set(None);
        };
        // Slot 40: a draw reads it through a table, its record left to it.
        heap.slot_tabled_in(40, frame);
        write_record(start, 40, Some(&buffer_view(window)));
        assert!(!sync(heap, start, 40), "left to the draws");
        let first = copy(0);
        let id = first.id();
        heap.hold_copy(40, handle(start, 40), &first);
        assert_eq!(heap.left_state().holds.held(40), Some(Arc::as_ptr(&first) as usize));
        drop(first);
        let_go_idle(id);
        assert_eq!(copy(0).id(), id, "held by the slot: kept");
        // Another record there: the sync lets the hold go; idle, let go.
        write_record(start, 40, Some(&buffer_view(window + 0x10_0000)));
        assert!(!sync(heap, start, 40), "left to the draws");
        assert_eq!(heap.left_state().holds.held(40), None, "the hold went with the record");
        let_go_idle(id);
        assert_ne!(copy(0).id(), id, "let go, then made again");
        // No hold in the off phase, nor with the hold off.
        let other = copy(1);
        for (live, hold) in [(false, true), (true, false)] {
            now(Mode2::Alt, live, false, hold);
            heap.hold_copy(40, handle(start, 40), &other);
            assert_eq!(heap.left_state().holds.held(40), None, "live {live}, hold {hold}");
        }
        let id = other.id();
        drop(other);
        let_go_idle(id);
        assert_ne!(copy(1).id(), id, "nothing held it");
        done();
        assert_eq!(release(title), 0);
    }

    /// A hold goes once its slot's mark is 240 frames old (the heal), and
    /// every hold goes at the flip to off; a draw that reads the slot for the
    /// same record again takes nothing new.
    #[test]
    fn holds_expire_with_their_mark_and_go_at_the_flip() {
        let Some((title, start, heap)) = heap2(Mode2::Alt, true, false) else {
            return;
        };
        let frame = crate::frame::current();
        let window = 0x7ff3_8000_0000u64;
        let copy = crate::shadow::find(window, 0x100).expect("a window copy");
        for slot in [40, 41] {
            heap.slot_tabled_in(slot, frame);
            write_record(start, slot, Some(&buffer_view(window + u64::from(slot) * 0x1000)));
            heap.hold_copy(slot, handle(start, slot), &copy);
        }
        let holds = &heap.left_state().holds;
        assert!(holds.held(40).is_some() && holds.held(41).is_some());
        let key = key_of_words(super::super::record_words(handle(start, 40)));
        let (new, old) = holds.take(40, key, &copy);
        assert!(!new && old.is_none(), "the same copy for the same record: nothing new");
        // 41 read again much later: its mark is fresh, 40's is not.
        heap.slot_tabled_in(41, frame + 200);
        let late = frame + u64::from(left::HOLD_FRAMES) + 1;
        assert_eq!(heap.left_heal((start, TITLE_SLOT_BYTES), late), [0, 1]);
        assert!(holds.held(40).is_none(), "expired with its mark");
        assert!(holds.held(41).is_some());
        now(Mode2::Alt, false, false, true);
        let swept = heap.tables_left_walk((start, TITLE_SLOT_BYTES)).expect("the flip to off sweeps");
        assert_eq!(swept[3], 1, "41's hold let go at the flip");
        assert!(holds.held(41).is_none());
        done();
        assert_eq!(release(title), 0);
    }

    /// `alt`: at the flip to off, every slot the sync left to the draws and
    /// every record waiting its grace is mirrored as the sync would have
    /// mirrored it, so the off phase starts from what it would have had: the
    /// host slots, the buffer slots noted and the prints are those of a heap
    /// that never had the switch. One sweep a flip, none from off to ON.
    #[test]
    fn the_flip_to_off_leaves_nothing_stale() {
        let at = |n: u64| 0x6e_0000_0000 + n * 0x10_0000;
        let records = [
            (40u32, Some(buffer_view(at(0)))),
            (41, Some(View::Constant(at(1), 256))),
            (42, Some(buffer_view(at(2)))),
            (43, Some(texture_view(at(3)))),
        ];
        // What a heap without the switch makes of the same records.
        let Some((plain, plain_start, plain_heap)) = heap2(Mode2::Off, false, false) else {
            return;
        };
        for (slot, view) in &records {
            write_record(plain_start, *slot, view.as_ref());
            assert!(sync(plain_heap, plain_start, *slot));
        }
        let Some((title, start, heap)) = heap2(Mode2::Alt, true, true) else {
            return;
        };
        let frame = crate::frame::current();
        heap.slot_tabled_in(40, frame);
        heap.slot_tabled_in(41, frame);
        for (slot, view) in &records {
            write_record(start, *slot, view.as_ref());
        }
        assert!(!sync(heap, start, 40), "left");
        assert!(!sync(heap, start, 41), "left");
        assert!(!sync(heap, start, 42), "waits its grace");
        assert!(sync(heap, start, 43), "a texture: mirrored");
        assert_eq!(heap.tables_left_walk((start, TITLE_SLOT_BYTES)), None, "no flip while ON");
        now(Mode2::Alt, false, true, true);
        let swept = heap.tables_left_walk((start, TITLE_SLOT_BYTES)).expect("the flip to off sweeps");
        assert_eq!(swept, [2, 1, 3, 0], "two left, one waiting, three mirrored, no holds");
        assert_eq!(heap.tables_left_walk((start, TITLE_SLOT_BYTES)), None, "once a flip");
        for slot in [40, 41, 42, 43] {
            assert!(!sync(heap, start, slot), "slot {slot}: its record noted as seen");
            assert_eq!(shows(heap, slot), shows(plain_heap, slot), "slot {slot}: shown as without the switch");
        }
        assert_eq!(heap.buffer_slots(), plain_heap.buffer_slots(), "the buffer slots noted");
        assert!(heap.left_state().bits_take(|_| true).is_empty(), "no left bit");
        assert_eq!(heap.left_state().waiting.lock().unwrap().len(), 0, "nothing waiting");
        now(Mode2::Alt, true, true, true);
        assert_eq!(heap.tables_left_walk((start, TITLE_SLOT_BYTES)), None, "off to ON: nothing");
        done();
        assert_eq!(release(title), 0);
        assert_eq!(release(plain), 0);
    }

    /// `check`: everything is mirrored as without the switch, and the model
    /// notes the slots whose host slot the switch would have left showing
    /// another view than the title's record: a read by index of one at a
    /// submission is a violation; a draw's making in place, which the switch
    /// does too, ends it; the pass's making of a slot the switch leaves does
    /// not. A list's notes are checked at its submission and taken.
    #[test]
    fn check_counts_reads_the_switch_would_have_left_stale() {
        let Some(device) = crate::host::host() else {
            return;
        };
        let Some((title, start, heap)) = heap2(Mode2::Check, false, false) else {
            return;
        };
        TEST_CHECKS.set(Some(Default::default()));
        let frame = crate::frame::current();
        let at = |n: u64| 0x6f_0000_0000 + n * 0x10_0000;
        let since = |slot: u32| heap.left_state().diverged_since(slot);
        heap.slot_tabled_in(40, frame);
        write_record(start, 40, Some(&buffer_view(at(0))));
        write_record(start, 41, Some(&buffer_view(at(1))));
        assert!(sync(heap, start, 40), "mirrored as without the switch");
        assert!(sync(heap, start, 41));
        assert!(since(40) != 0 && since(41) == 0, "40 left by the switch: its host slot would show nothing");
        heap.check_read(40);
        heap.check_read(41);
        // A draw reads 40 through a table read in place: made, as with the
        // switch.
        heap.mirror(40, handle(start, 40), true);
        assert_eq!(since(40), 0);
        heap.check_read(40);
        // A new record: the pass makes it, a making the switch would not do.
        write_record(start, 40, Some(&buffer_view(at(2))));
        assert!(sync(heap, start, 40));
        left::left_making(|| heap.mirror(40, handle(start, 40), true));
        assert!(since(40) != 0, "still what the switch would leave");
        heap.check_pass(&[40], frame);
        // A list's read by index, checked at its submission.
        let list = device.new_list().expect("a list");
        let at_heap = super::super::heap(title);
        list.note_left_read(std::ptr::from_ref(at_heap) as usize, at_heap.start, 40);
        list.note_left_picks();
        left::check_submission(&[&list]);
        assert_eq!(list.take_left_reads(), (Vec::new(), false), "taken");
        let counts = TEST_CHECKS.take().expect("counted");
        let get = |what: Ck| counts[what as usize];
        assert_eq!(get(Ck::Reads), 4, "{counts:?}");
        assert_eq!(get(Ck::Violations), 2, "40 twice: before its draw, and at the submission");
        assert_eq!(get(Ck::SyncLeft), 2);
        assert_eq!((get(Ck::PassLeft), get(Ck::PassStale)), (1, 1));
        assert_eq!((get(Ck::Lists), get(Ck::Picking)), (1, 1));
        assert!(get(Ck::Exposed) >= 1);
        list.close();
        done();
        assert_eq!(release(title), 0);
    }
}
