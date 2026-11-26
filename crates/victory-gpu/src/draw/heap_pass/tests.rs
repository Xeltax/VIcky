//! Tests of the pass over buffer views beside (`super`): the steps put in
//! place of its step, its end, the check's comparison, and, with a host
//! device, the pass beside against the pass in place on heaps written alike.

use super::*;
use crate::descriptor::host::TEST_FAST;
use crate::object::This;
use std::sync::mpsc;

/// A step of a list as [`splice`] sees it, for the tests that need no
/// list.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Step {
    Copy,
    Refresh(u32),
    Other,
    Here,
}

fn seen(step: &Step) -> Seen<u32> {
    match step {
        Step::Copy => Seen::Barrier,
        Step::Refresh(key) => Seen::Refresh(*key),
        Step::Other | Step::Here => Seen::Other,
    }
}

/// The refreshes take the step's place, as `defer_refresh` would have
/// recorded them there: one the list refreshed since its last copy is left
/// out, one given twice is put in once, and a later refresh of one put in
/// goes, up to the next copy only.
#[test]
fn the_refreshes_take_the_place_of_the_step_as_they_would_have_been_recorded() {
    use Step::*;
    let mut steps = vec![
        Refresh(1),
        Copy,
        Refresh(2),
        Other,
        Refresh(3),
        Here,
        Refresh(4),
        Refresh(5),
        Other,
        Copy,
        Refresh(4),
    ];
    let given = [2, 4, 6, 6, 1].map(|k| (k, Refresh(k))).to_vec();
    assert_eq!(splice(&mut steps, 5, given, seen), 3);
    assert_eq!(
        steps,
        [
            Refresh(1),
            Copy,
            Refresh(2),
            Other,
            Refresh(3),
            // 2 was refreshed since the copy; 1 before it.
            Refresh(4),
            Refresh(6),
            Refresh(1),
            // The later 4 would have found itself refreshed.
            Refresh(5),
            Other,
            Copy,
            // After a copy, refreshed again.
            Refresh(4),
        ]
    );
    // At the start of a list, and with nothing to put in: the step goes.
    let mut steps = vec![Here, Refresh(7), Copy];
    assert_eq!(splice(&mut steps, 0, vec![(7, Refresh(7))], seen), 1);
    assert_eq!(steps, [Refresh(7), Copy]);
    let mut steps = vec![Other, Here, Other];
    assert_eq!(splice(&mut steps, 1, Vec::new(), seen), 0);
    assert_eq!(steps, [Other, Other]);
}

/// A pass ends once: by its filler, or by the filler's drop (a pass that
/// never ran ends with nothing); its refreshes are taken once; its panic
/// comes back where they are taken.
#[test]
fn a_pass_ends_once_and_its_panic_comes_back_where_its_refreshes_are_taken() {
    let run = Run::new();
    let pass = run.pass.clone();
    drop(run);
    pass.wait();
    assert!(pass.take_refreshes().is_empty());
    let run = Run::new();
    let pass = run.pass.clone();
    let waiting = {
        let pass = pass.clone();
        std::thread::spawn(move || pass.wait())
    };
    run.filler.end(Vec::new(), Some(Box::new("a pass that fails")), None);
    waiting.join().expect("the waiting thread");
    let taken = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pass.take_refreshes()));
    assert!(taken.is_err(), "the panic was lost");
    // Given once: the next one takes nothing.
    assert!(pass.take_refreshes().is_empty());
}

/// The check tells a slot whose view changed between the call and the
/// pass from one with the same view and another copy or range.
#[test]
fn the_check_tells_changed_slots_from_differing_ones() {
    let at = |index, address: u64, range: Option<(u64, u64, u64)>| Found {
        index,
        view: Some(View::Constant(address, 256)),
        range,
    };
    let call = [
        at(1, 0x1000, Some((7, 0x1000, 0x1100))),
        at(2, 0x2000, Some((7, 0x2000, 0x2100))),
        at(3, 0x3000, Some((7, 0x3000, 0x3100))),
        at(4, 0x4000, None),
        Found { index: 5, view: None, range: None },
        at(6, 0x6000, None),
    ];
    let beside = [
        // As at the call.
        at(1, 0x1000, Some((7, 0x1000, 0x1100))),
        Found { index: 5, view: None, range: None },
        // The title wrote another view.
        at(2, 0x2800, Some((7, 0x2800, 0x2900))),
        // The same view, on another copy.
        at(3, 0x3000, Some((8, 0x3000, 0x3100))),
        // The same view, a copy found where none was.
        at(4, 0x4000, Some((9, 0x4000, 0x4100))),
        // (Slot 6 not looked at.)
    ];
    assert_eq!(compare(&call, &beside), Checked { slots: 6, differed: 3, changed: 1 });
    assert_eq!(compare(&call, &call), Checked { slots: 6, differed: 0, changed: 0 });
}

/// One pass a frame: the frame's first ask starts it, the others wait for
/// it; and a list that waits for it is held at its submission until it
/// has ended.
#[test]
fn the_frames_other_lists_wait_for_its_pass() {
    // Frames no run reaches, so that nothing else asks for them.
    let frame = u64::MAX - 0x5e;
    let Turn::Run(run) = turn(frame) else {
        panic!("the frame's first ask starts its pass");
    };
    let Turn::Wait(pass) = turn(frame) else {
        panic!("the frame's other asks wait for it");
    };
    assert!(Arc::ptr_eq(&pass, &run.pass));
    let Turn::Run(next) = turn(frame + 1) else {
        panic!("the next frame has a pass of its own");
    };
    assert!(!Arc::ptr_eq(&next.pass, &run.pass));
    drop(next);
    let Some(host) = crate::host::host() else {
        return;
    };
    let list = host.new_list().expect("a host list");
    list.after_heap_pass(&pass);
    list.after_heap_pass(&pass);
    let ended = Arc::new(AtomicBool::new(false));
    let ender = {
        let ended = ended.clone();
        let Run { filler, .. } = run;
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            ended.store(true, Relaxed);
            filler.end(Vec::new(), None, None);
        })
    };
    list.join_heap_passes();
    assert!(ended.load(Relaxed), "the list went on before the pass ended");
    ender.join().expect("the ending thread");
    // Nothing is left to wait for.
    list.join_heap_passes();
    list.close();
}

/// A title view heap of `count` slots with its host heap; nothing the
/// passes kept for an earlier heap at the same address.
fn title_heap(count: u32) -> Option<(This, &'static Heap, &'static HostHeap)> {
    use windows::core::Interface;
    crate::host::host()?;
    let null: This = std::ptr::null();
    let mut object: This = null;
    // Views, `count` slots, shader-visible; the heap's own memory.
    let desc = [0u32, count, 1, 0, 0, 0, 0, 0];
    let iid = windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap::IID;
    let made = crate::descriptor::create_heap(null, desc.as_ptr().cast(), std::ptr::from_ref(&iid).cast(), &mut object);
    assert_eq!(made, crate::object::S_OK);
    let heap = crate::descriptor::heap(object);
    let host = heap.host()?;
    forget(host);
    Some((object, heap, host))
}

fn release(object: This) -> u32 {
    crate::object::slot::<extern "system" fn(This) -> u32>(object, 0x10)(object)
}

/// A placed structured buffer view of `width` bytes at `address`: `count`
/// elements of 16 bytes from element `first`.
fn buffer_view(address: u64, width: u32, first: u32, count: u32) -> View {
    let mut resource = [0u32; 14];
    (resource[0], resource[4]) = (1, width);
    let mut desc = [0u32; 16];
    // unknown format, buffer, default mapping
    (desc[1], desc[2], desc[4], desc[6], desc[7]) = (1, 5768, first, count, 16);
    (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
    View::Placed(false, resource, desc)
}

/// Title slot `slot` of `heap` written with `view` by a call, as the title
/// writes it.
fn write(heap: &Heap, slot: u32, view: View) {
    crate::descriptor::write_view(heap.handle(slot).expect("a slot"), view);
}

/// The ranges the passes keep for `host`, by slot: copy (by number), from,
/// to.
fn kept(host: &HostHeap) -> std::collections::BTreeMap<u32, (u64, u64, u64)> {
    let key = std::ptr::from_ref(host) as usize;
    RANGES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|kept| kept.get(&key))
        .map(|(_, ranges)| ranges.iter().map(|(&i, (s, from, to))| (i, (s.id(), *from, *to))).collect())
        .unwrap_or_default()
}

/// Holds the lane's thread until the sender is dropped or sends: jobs
/// given meanwhile wait behind it.
fn hold_lane() -> mpsc::Sender<()> {
    let (go, wait) = mpsc::channel::<()>();
    let _ = lane().expect("a lane").give(Box::new(move |_: &Ticket| {
        let _ = wait.recv();
    }));
    go
}

/// Two heaps written alike, one passed over in place and the other beside,
/// frame after frame (the first pass over every buffer slot, the next ones
/// over the slots written since; buffer views moved, emptied and turned
/// into constant views), with the cache of made views and without: the
/// lists get the same refreshes, the passes keep the same ranges, every
/// slot shows the same buffer, and the check finds every slot as it was at
/// the call.
#[test]
fn the_pass_beside_finds_what_the_pass_in_place_finds() {
    let Some(device) = crate::host::host() else {
        return;
    };
    for cache in [0usize, 1 << 10] {
        TEST_FAST.set(Some((cache, false, false)));
        let (Some(a), Some(b)) = (title_heap(64), title_heap(64)) else {
            TEST_FAST.set(None);
            return;
        };
        let base = 0x5b_0000_0000u64 + ((cache as u64) << 28);
        let (one, two) = (0x7e57_5b01 + cache, 0x7e57_5b02 + cache);
        crate::shadow::register(one, base, 0x1_0000);
        crate::shadow::register(two, base + 0x10_0000, 0x4_0000);
        let both = |slot: u32, view: View| {
            write(a.1, slot, view.clone());
            write(b.1, slot, view);
        };
        for slot in 1..=6 {
            both(slot, buffer_view(base, 0x1_0000, 4 * slot, 8));
        }
        for slot in 7..=9 {
            both(slot, buffer_view(base + 0x10_0000, 0x4_0000, 16 * slot, 32));
        }
        // Memory outside the title's buffers, and a constant view.
        both(11, buffer_view(base + 0x80_0000, 0x2000, 0, 16));
        both(12, View::Constant(base + 0x200, 256));
        let rounds: [&dyn Fn(); 3] = [
            &|| {},
            &|| {
                both(2, buffer_view(base, 0x1_0000, 64, 8));
                both(3, View::NullConstant);
                both(13, buffer_view(base + 0x10_0000, 0x4_0000, 0, 4));
            },
            &|| {
                both(5, buffer_view(base + 0x10_0000, 0x4_0000, 8, 8));
                both(7, View::Constant(base + 0x10_0400, 512));
                both(3, buffer_view(base, 0x1_0000, 0, 2));
            },
        ];
        for (round, change) in rounds.iter().enumerate() {
            change();
            let frame = 0x5b00 + round as u64;
            let (in_place_list, beside_list) = (device.new_list().expect("a list"), device.new_list().expect("a list"));
            in_place(&in_place_list, a.1, a.2, frame);
            let run = Run::new();
            let pass = run.pass.clone();
            start(&beside_list, b.1, b.2, frame, run, true);
            assert_eq!(beside_list.heap_pass_steps(), 1, "its step is in the list");
            beside_list.join_heap_passes();
            assert_eq!(beside_list.heap_pass_steps(), 0, "put in place");
            let checked = pass.checked().expect("a checked pass");
            assert!(checked.slots > 0, "round {round}");
            assert_eq!((checked.differed, checked.changed), (0, 0), "round {round}, cache {cache}");
            let (mut there, mut here) = (in_place_list.refresh_keys(), beside_list.refresh_keys());
            assert!(!there.is_empty(), "round {round}: something to refresh");
            there.sort_unstable();
            here.sort_unstable();
            assert_eq!(there, here, "round {round}, cache {cache}: the refreshes");
            assert_eq!(kept(a.2), kept(b.2), "round {round}, cache {cache}: the ranges kept");
            for slot in 0..64 {
                assert_eq!(a.2.made(slot), b.2.made(slot), "round {round}, cache {cache}, slot {slot}: made");
                assert_eq!(
                    a.2.shown_raw(slot),
                    b.2.shown_raw(slot),
                    "round {round}, cache {cache}, slot {slot}: the buffer shown"
                );
            }
            in_place_list.close();
            beside_list.close();
        }
        crate::shadow::unregister(one, base);
        crate::shadow::unregister(two, base + 0x10_0000);
        forget(a.2);
        forget(b.2);
        assert_eq!(release(a.0), 0);
        assert_eq!(release(b.0), 0);
    }
    TEST_FAST.set(None);
}

/// A slot the title writes after the call, before the pass beside gets to
/// it, is made from its new record, as the pass in place made one another
/// thread wrote before it got there; its new range is brought up to date,
/// and the check counts it as changed between the call and the pass.
#[test]
fn a_slot_written_after_the_call_is_made_from_its_new_record() {
    let Some(device) = crate::host::host() else {
        return;
    };
    TEST_FAST.set(Some((0, false, false)));
    let Some((object, heap, host)) = title_heap(32) else {
        TEST_FAST.set(None);
        return;
    };
    let base = 0x5c_0000_0000u64;
    crate::shadow::register(0x7e57_5c01, base, 0x1_0000);
    crate::shadow::register(0x7e57_5c02, base + 0x10_0000, 0x1_0000);
    write(heap, 3, buffer_view(base, 0x1_0000, 0, 8));
    write(heap, 4, buffer_view(base, 0x1_0000, 8, 8));
    // The first pass, over every buffer slot.
    let list = device.new_list().expect("a list");
    let run = Run::new();
    start(&list, heap, host, 0x5c00, run, false);
    list.join_heap_passes();
    list.close();
    let old = crate::shadow::find(base, 0x1_0000).expect("a copy");
    assert_eq!(host.shown_raw(3), Some(windows::core::Interface::as_raw(&old.buffer) as usize));
    // The next one is handed over while the lane is busy; the title writes
    // slot 3 before it runs.
    let list = device.new_list().expect("a list");
    let hold = hold_lane();
    let run = Run::new();
    let pass = run.pass.clone();
    start(&list, heap, host, 0x5c01, run, true);
    write(heap, 3, buffer_view(base + 0x10_0000, 0x1_0000, 0, 8));
    drop(hold);
    list.join_heap_passes();
    let new = crate::shadow::find(base + 0x10_0000, 0x1_0000).expect("a copy");
    assert_eq!(
        host.shown_raw(3),
        Some(windows::core::Interface::as_raw(&new.buffer) as usize),
        "made from the new record"
    );
    let ranges = kept(host);
    assert_eq!(ranges.get(&3), Some(&(new.id(), base + 0x10_0000, base + 0x10_0000 + 8 * 16)));
    assert_eq!(ranges.get(&4), Some(&(old.id(), base + 8 * 16, base + 16 * 16)));
    let new_key = std::sync::Arc::as_ptr(&new) as usize;
    assert!(
        list.refresh_keys().iter().any(|&(copy, address, _)| copy == new_key && address == (base + 0x10_0000) as usize),
        "the new range is brought up to date"
    );
    let checked = pass.checked().expect("a checked pass");
    assert_eq!((checked.changed, checked.differed), (1, 0), "{checked:?}");
    list.close();
    crate::shadow::unregister(0x7e57_5c01, base);
    crate::shadow::unregister(0x7e57_5c02, base + 0x10_0000);
    forget(host);
    assert_eq!(release(object), 0);
    TEST_FAST.set(None);
}

/// A pass whose heap the title released before the pass's turn is not run:
/// it ends with nothing to bring up to date, and the list goes on.
#[test]
fn a_pass_whose_heap_went_ends_with_nothing() {
    let Some(device) = crate::host::host() else {
        return;
    };
    let Some((object, heap, host)) = title_heap(16) else {
        return;
    };
    let base = 0x5d_0000_0000u64;
    write(heap, 2, buffer_view(base, 0x1000, 0, 8));
    let list = device.new_list().expect("a list");
    let hold = hold_lane();
    let run = Run::new();
    start(&list, heap, host, 0x5d00, run, false);
    forget(host);
    assert_eq!(release(object), 0);
    drop(hold);
    list.join_heap_passes();
    assert!(list.refresh_keys().is_empty());
    assert_eq!(list.heap_pass_steps(), 0);
    list.close();
}

/// A small generator for the models below (xorshift).
fn draws(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// With the spread, each buffer slot is in the window of one turn of 16,
/// fixed by its index: in any 16 passes in a row, a slot that stayed a
/// buffer slot throughout is in exactly one window, and no slot is in two,
/// however many slots came and went meanwhile; a window holds the buffer
/// slots of its turn, in order.
#[test]
fn the_window_visits_every_slot_once_in_16_passes_however_the_slots_change() {
    let mut next = draws(0x5eed_5b1e_ad00_0001);
    let mut slots: BTreeSet<u32> = (0..3_000).chain(40_000..41_000).collect();
    // Per slot, the passes it was a buffer slot at and the passes whose
    // window it was in (one bit a pass).
    let (mut present, mut came) = (HashMap::<u32, u128>::new(), HashMap::<u32, u128>::new());
    let passes = 96u32;
    for n in 0..passes {
        match n % 4 {
            // A stretch comes, or goes.
            0 => {
                let from = (next() % 120_000) as u32;
                slots.extend(from..from + (next() % 3_000) as u32);
            }
            1 => {
                let from = (next() % 120_000) as u32;
                let to = from + (next() % 2_000) as u32;
                slots.retain(|&i| !(from..to).contains(&i));
            }
            // Single slots come and go.
            2 => {
                for _ in 0..400 {
                    let i = (next() % 120_000) as u32;
                    if !slots.remove(&i) {
                        slots.insert(i);
                    }
                }
            }
            _ => {}
        }
        let turn = n % 16;
        let window = window(&slots, turn);
        assert!(window.windows(2).all(|w| w[0] < w[1]), "pass {n}: in order, each once");
        let of_turn: Vec<u32> = slots.iter().copied().filter(|&i| turn_of(i) == turn).collect();
        assert_eq!(window, of_turn, "pass {n}: the buffer slots of turn {turn}");
        for &i in &slots {
            *present.entry(i).or_default() |= 1u128 << n;
        }
        for &i in &window {
            *came.entry(i).or_default() |= 1u128 << n;
        }
    }
    for start in 0..=passes - 16 {
        let these = ((1u128 << 16) - 1) << start;
        for (&i, &came) in &came {
            let times = (came & these).count_ones();
            assert!(times <= 1, "slot {i} came {times} times in passes {start} to {}", start + 15);
            if present[&i] & these == these {
                assert_eq!(times, 1, "slot {i}, a buffer slot throughout passes {start} to {}", start + 15);
            }
        }
        // (A slot never in a window was never a buffer slot 16 passes in a
        // row.)
        for (&i, &present) in &present {
            if present & these == these {
                assert!(came.contains_key(&i), "slot {i}");
            }
        }
    }
}

/// The windows cut the buffer slots about evenly, whatever their layout:
/// a stretch, slots that come every 8th or in fours of each 16 (tables),
/// and stretches strewn over a large heap.
#[test]
fn the_windows_are_about_even_whatever_the_layout() {
    let mut next = draws(0x5eed_5b1e_ad00_0002);
    let mut strewn = BTreeSet::new();
    while strewn.len() < 150_000 {
        let from = (next() % 1_200_000) as u32;
        strewn.extend(from..from + 64 + (next() % 4_000) as u32);
    }
    let layouts: [(&str, BTreeSet<u32>); 4] = [
        ("a stretch", (0..154_265).collect()),
        ("every 8th", (0..1_245_184).step_by(8).collect()),
        ("fours of each 16", (0..600_000).filter(|i| i % 16 < 4).collect()),
        ("strewn stretches", strewn),
    ];
    for (layout, slots) in layouts {
        let sizes: Vec<usize> = (0..16).map(|turn| window(&slots, turn).len()).collect();
        assert_eq!(sizes.iter().sum::<usize>(), slots.len(), "{layout}: each slot in one window");
        let even = slots.len() as f64 / 16.0;
        let (least, most) = (*sizes.iter().min().unwrap() as f64, *sizes.iter().max().unwrap() as f64);
        assert!(least > 0.8 * even && most < 1.2 * even, "{layout}: {sizes:?}");
    }
}

/// A model of a heap for the passes ([`choose`], [`Afresh::apply`]): its
/// buffer slots, what each one's view reads now (none: no copy), and the
/// slots written since the last pass.
#[derive(Clone, Default)]
struct Model {
    slots: BTreeSet<u32>,
    reads: HashMap<u32, u64>,
    changed: HashSet<u32>,
}

impl Model {
    /// Pass `n` over the model, as [`in_place`] runs it: the ranges of its
    /// slots dropped and kept again if their views read one. Returns how
    /// many slots it looked at.
    fn pass(&mut self, kept: &mut HashMap<u32, u64>, n: u64, spread: bool) -> usize {
        let changed: Vec<u32> = self.changed.drain().collect();
        let (mut slots, afresh) =
            choose(n, spread, changed, || self.slots.iter().copied().collect(), |turn| window(&self.slots, turn));
        afresh.apply(kept, &mut slots);
        for &i in &slots {
            kept.remove(&i);
            if self.slots.contains(&i)
                && let Some(&read) = self.reads.get(&i)
            {
                kept.insert(i, read);
            }
        }
        slots.len()
    }

    /// What a pass that looked at every slot now would keep.
    fn truth(&self) -> HashMap<u32, u64> {
        self.slots.iter().filter_map(|&i| Some((i, *self.reads.get(&i)?))).collect()
    }
}

/// On a model of a heap (slots written, emptied, and some whose views read
/// another copy, or one for the first time, with no write: a released
/// resource, memory that got a host copy late), the spread passes keep what
/// the passes over every slot each 16th pass keep: the same after 16
/// passes with nothing written, and with both, a change without a write is
/// seen within 16 passes. No spread pass looks at all the slots after the
/// first.
#[test]
fn the_spread_passes_keep_what_the_full_passes_keep_on_a_model() {
    let mut next = draws(0x5eed_5b1e_ad00_0003);
    let mut model = Model::default();
    for i in (0..20_000).chain(70_000..90_000).filter(|i| i % 5 != 3) {
        model.slots.insert(i);
        model.reads.insert(i, u64::from(i));
        model.changed.insert(i);
    }
    let (mut full, mut spread) = (model.clone(), model);
    let (mut kept_full, mut kept_spread) = (HashMap::new(), HashMap::new());
    // Per slot, the pass after its last change without a write.
    let mut unwritten: HashMap<u32, u64> = HashMap::new();
    let mut version = 1u64 << 32;
    for n in 0..120u64 {
        // Passes 40 to 55 and 104 to 119: nothing written.
        let quiet = (40..56).contains(&n) || (104..120).contains(&n);
        if !quiet {
            for _ in 0..300 {
                let i = (next() % 100_000) as u32;
                version += 1;
                let both = |m: &mut Model| match version % 4 {
                    // Written with a view, or emptied.
                    0 | 1 => {
                        m.slots.insert(i);
                        m.reads.insert(i, version);
                        m.changed.insert(i);
                    }
                    2 => {
                        m.slots.remove(&i);
                        m.changed.insert(i);
                    }
                    // Its view reads another copy, with no write.
                    _ => {
                        m.reads.insert(i, version);
                    }
                };
                both(&mut full);
                both(&mut spread);
                if version % 4 == 3 {
                    unwritten.insert(i, n);
                }
            }
        }
        let looked = (full.pass(&mut kept_full, n, false), spread.pass(&mut kept_spread, n, true));
        if n > 0 {
            assert!(looked.1 < spread.slots.len() / 4, "pass {n}: {looked:?} of {}", spread.slots.len());
        }
        let truth = full.truth();
        assert_eq!(truth, spread.truth());
        for (&i, &since) in &unwritten {
            if n >= since + 15 {
                assert_eq!(kept_full.get(&i), truth.get(&i), "pass {n}, slot {i}, changed before pass {since}: over all");
                assert_eq!(kept_spread.get(&i), truth.get(&i), "pass {n}, slot {i}, changed before pass {since}: spread");
            }
        }
        if n == 55 || n == 119 {
            assert_eq!(kept_full, truth, "pass {n}: over all");
            assert_eq!(kept_spread, truth, "pass {n}: spread");
        }
    }
}

/// Three heaps written alike: passed over in place with a pass over every
/// buffer slot each 16th pass, and with the spread in place and beside
/// (checked every pass); with the cache of made views and without. Buffer
/// slots in runs of every turn and on two buffers; each pass, one written
/// in the pass's own window (looked at once). At every pass the spread in
/// place and beside keep the same ranges and show the same buffers, the
/// check finds every slot as at the call, and the spread keeps what the
/// pass without it keeps. (A view that reads another copy with no write is
/// the model's: `the_spread_passes_keep_what_the_full_passes_keep_on_a_model`.)
#[test]
fn the_spread_pass_keeps_what_the_full_pass_keeps() {
    let Some(device) = crate::host::host() else {
        return;
    };
    let runs = 64u32;
    let run_of = |turn: u32| (0..runs).find(|&r| turn_of(r * RUN) == turn).expect("a run of each turn");
    for cache in [0usize, 1 << 10] {
        TEST_FAST.set(Some((cache, false, false)));
        let (Some(full), Some(here), Some(beside)) = (title_heap(runs * RUN), title_heap(runs * RUN), title_heap(runs * RUN))
        else {
            TEST_FAST.set(None);
            return;
        };
        let base = 0x5e_0000_0000u64 + ((cache as u64) << 28);
        let (one, two) = (0x7e57_5e01 + cache, 0x7e57_5e02 + cache);
        crate::shadow::register(one, base, 0x10_0000);
        crate::shadow::register(two, base + 0x100_0000, 0x1000);
        let all = |slot: u32, view: View| {
            for heap in [full.1, here.1, beside.1] {
                write(heap, slot, view.clone());
            }
        };
        for run in 0..runs {
            all(run * RUN + 5, buffer_view(base, 0x10_0000, 4 * run, 8));
            all(run * RUN + 40, buffer_view(base, 0x10_0000, 4 * run + 2, 8));
            all(run * RUN + 20, buffer_view(base + 0x100_0000, 0x1000, run % 32, 4));
        }
        for n in 0..=16u64 {
            if n > 0 {
                let turn = (n % 16) as u32;
                all(run_of(turn) * RUN + 5, buffer_view(base, 0x10_0000, 300 + n as u32, 8));
            }
            let frame = 0x5e00 + n;
            let lists = [(); 3].map(|()| device.new_list().expect("a list"));
            TEST_SPREAD.set(Some(false));
            in_place(&lists[0], full.1, full.2, frame);
            TEST_SPREAD.set(Some(true));
            let made = in_place(&lists[1], here.1, here.2, frame);
            // (A pass's making marks the slots it made, with the spread and
            // without: pass 1 looks at every slot again, and each pass at the
            // slot written for the last.)
            if n > 1 {
                // The window's 3 slots of each of its runs (the slot written
                // in it once), and the one written for the last pass.
                let window = 3 * (0..runs).filter(|&r| turn_of(r * RUN) == (n % 16) as u32).count();
                assert_eq!(made.slots, window + 1, "pass {n}, cache {cache}: looked at once");
            }
            let run = Run::new();
            let pass = run.pass.clone();
            start(&lists[2], beside.1, beside.2, frame, run, true);
            lists[2].join_heap_passes();
            TEST_SPREAD.set(None);
            let checked = pass.checked().expect("a checked pass");
            assert_eq!((checked.differed, checked.changed), (0, 0), "pass {n}, cache {cache}");
            let spread = kept(here.2);
            assert_eq!(spread, kept(beside.2), "pass {n}, cache {cache}: in place and beside");
            for slot in 0..runs * RUN {
                assert_eq!(here.2.made(slot), beside.2.made(slot), "pass {n}, cache {cache}, slot {slot}: made");
                assert_eq!(here.2.shown_raw(slot), beside.2.shown_raw(slot), "pass {n}, cache {cache}, slot {slot}");
            }
            assert_eq!(spread, kept(full.2), "pass {n}, cache {cache}: with the spread and without");
            for list in lists {
                list.close();
            }
        }
        crate::shadow::unregister(one, base);
        crate::shadow::unregister(two, base + 0x100_0000);
        for heap in [&full, &here, &beside] {
            forget(heap.2);
        }
        for heap in [full, here, beside] {
            assert_eq!(release(heap.0), 0);
        }
    }
    TEST_FAST.set(None);
}

/// With `VICTORY_HEAP_PASS_DUE`, on a model of a heap whose buffer slots
/// come and go and are written (every note of a slot marks it, as
/// `HostHeap::note_buffer` does): in any 16 passes in a row, a slot that was
/// a buffer slot at each of their calls is looked at at least once; a slot
/// is in a window only if no pass looked at it in the 15 before; and in 16
/// passes over a heap nothing writes, every buffer slot is looked at once.
#[test]
fn the_due_windows_look_at_every_buffer_slot_once_in_any_16_passes() {
    let mut next = draws(0x5eed_5b1e_ad00_0004);
    let mut slots: BTreeSet<u32> = (0..3_000).chain(40_000..41_000).collect();
    let mut changed: HashSet<u32> = slots.iter().copied().collect();
    let mut looks = Looks::default();
    // Per slot, the passes it was a buffer slot at and the passes that
    // looked at it (one bit a pass).
    let (mut present, mut came) = (HashMap::<u32, u128>::new(), HashMap::<u32, u128>::new());
    let passes = 112u32;
    for n in 0..passes {
        // Passes 64 to 95: nothing written.
        if !(64..96).contains(&n) {
            match n % 4 {
                0 => {
                    let from = (next() % 120_000) as u32;
                    for i in from..from + (next() % 3_000) as u32 {
                        slots.insert(i);
                        changed.insert(i);
                    }
                }
                1 => {
                    let from = (next() % 120_000) as u32;
                    let to = from + (next() % 2_000) as u32;
                    let gone: Vec<u32> = slots.range(from..to).copied().collect();
                    for i in gone {
                        slots.remove(&i);
                        changed.insert(i);
                    }
                }
                2 => {
                    for _ in 0..400 {
                        let i = (next() % 120_000) as u32;
                        if !slots.remove(&i) {
                            slots.insert(i);
                        }
                        changed.insert(i);
                    }
                }
                // Buffer slots written again.
                _ => {
                    let all: Vec<u32> = slots.iter().copied().collect();
                    for _ in 0..2_000 {
                        changed.insert(all[(next() % all.len() as u64) as usize]);
                    }
                }
            }
        }
        let written: Vec<u32> = changed.drain().collect();
        let taken = written.len();
        let (list, _) = choose_due(
            u64::from(n),
            &mut looks,
            written,
            || slots.iter().copied().collect(),
            |unseen| unseen.into_iter().filter(|i| slots.contains(i)).collect(),
        );
        let before = if n == 0 { 0 } else { ((1u128 << n.min(15)) - 1) << (n - n.min(15)) };
        if n > 0 {
            for &i in &list[taken..] {
                assert!(slots.contains(&i), "pass {n}: slot {i} in the window is a buffer slot");
                let came = came.get(&i).copied().unwrap_or(0);
                assert_eq!(came & before, 0, "pass {n}: slot {i} in the window was looked at in the 15 passes before");
            }
        }
        for &i in &slots {
            *present.entry(i).or_default() |= 1u128 << n;
        }
        for &i in &list {
            let came = came.entry(i).or_default();
            assert_eq!(*came & (1u128 << n), 0, "pass {n}: slot {i} twice");
            *came |= 1u128 << n;
        }
    }
    for start in 0..=passes - 16 {
        let these = ((1u128 << 16) - 1) << start;
        for (&i, &present) in &present {
            if present & these == these {
                let times = (came.get(&i).copied().unwrap_or(0) & these).count_ones();
                assert!(times >= 1, "slot {i}, a buffer slot throughout passes {start} to {}", start + 15);
                // Nothing written in them: once.
                if (64..=80).contains(&start) {
                    assert_eq!(times, 1, "slot {i}, passes {start} to {}", start + 15);
                }
            }
        }
    }
}

/// A model of a title heap for the slots the passes look at: slots
/// `0..fixed` written once before the first pass, the rest a ring the title
/// writes `per_pass` slots of before each pass, on from where it stopped (a
/// ring two passes long is two areas written in turns); a written slot holds
/// a buffer view if a hash of its index says so (`share` in 100).
#[derive(Clone, Copy, Debug)]
struct Layout {
    heap: u32,
    fixed: u32,
    per_pass: u32,
    share: u64,
}

/// Over `passes` passes on `layout` (the first over every buffer slot), the
/// mean per pass from the 32nd on of: the slots looked at, those taken as
/// written, those in the window, and the buffer slots. `due`:
/// `VICTORY_HEAP_PASS_DUE` (else the windows of turns); `one_look`:
/// `VICTORY_HEAP_PASS_ONE_LOOK` (else a pass's making of a slot the title
/// wrote marks it again).
fn looked_on(layout: Layout, due: bool, one_look: bool, passes: u64) -> [f64; 4] {
    /// The title's write of slot `i` (a buffer view or not): noted, and
    /// left for the pass to make.
    #[derive(Default)]
    struct Title {
        slots: BTreeSet<u32>,
        changed: HashSet<u32>,
        unmade: HashSet<u32>,
    }
    impl Title {
        fn write(&mut self, i: u32, buffer: bool) {
            if buffer {
                self.slots.insert(i);
                self.changed.insert(i);
                self.unmade.insert(i);
            } else {
                self.unmade.remove(&i);
                if self.slots.remove(&i) {
                    self.changed.insert(i);
                }
            }
        }
    }
    let buffer = |i: u32| (u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) % 100 < layout.share;
    let mut title = Title::default();
    for i in 0..layout.fixed {
        title.write(i, buffer(i));
    }
    let ring = layout.heap - layout.fixed;
    let (mut at, mut looks, mut sums, mut counted) = (0, Looks::default(), [0f64; 4], 0f64);
    for n in 0..passes {
        for _ in 0..layout.per_pass {
            let i = layout.fixed + at;
            title.write(i, buffer(i));
            at = (at + 1) % ring;
        }
        let written: Vec<u32> = title.changed.drain().collect();
        let taken = written.len();
        let mut in_window = 0;
        let slots = &title.slots;
        let (mut list, afresh) = if due {
            choose_due(n, &mut looks, written, || slots.iter().copied().collect(), |unseen| {
                let window: Vec<u32> = unseen.into_iter().filter(|i| slots.contains(i)).collect();
                in_window = window.len();
                window
            })
        } else {
            choose(n, true, written, || slots.iter().copied().collect(), |turn| {
                let window = window(slots, turn);
                in_window = window.len();
                window
            })
        };
        afresh.apply(&mut HashMap::<u32, ()>::new(), &mut list);
        // The pass makes the slots the title wrote; without the switch
        // that marks them again.
        for &i in &list {
            if title.unmade.remove(&i) && !one_look {
                title.changed.insert(i);
            }
        }
        if n >= 32 {
            for (sum, x) in sums.iter_mut().zip([list.len(), taken, in_window, title.slots.len()]) {
                *sum += x as f64;
            }
            counted += 1.0;
        }
    }
    sums.map(|sum| sum / counted)
}

/// On models of the title's heap at a sixteenth of RT81b's (1,245,184 slots,
/// some 460,000 of them buffer slots, some 45,000 buffer slots written a
/// pass): the title's writes as a ring over the whole heap, and as two areas
/// written in turns beside slots written once. The pass's own making marked
/// every slot the title wrote a second time, and `VICTORY_HEAP_PASS_ONE_LOOK`
/// halves the slots taken as written; the turn's window holds a sixteenth
/// of the buffer slots, and `VICTORY_HEAP_PASS_DUE`'s only those no pass
/// looked at in the last 16: next to none with the ring, the slots written
/// once with the two areas. (`--nocapture` prints the numbers.)
#[test]
fn the_passes_look_at_fewer_slots_on_models_of_the_heap() {
    let layouts = [
        ("a ring over the heap", Layout { heap: 77_824, fixed: 0, per_pass: 7_600, share: 37 }),
        ("two areas in turns", Layout { heap: 77_824, fixed: 62_626, per_pass: 7_599, share: 37 }),
    ];
    for (name, layout) in layouts {
        let [before, one, due, both] =
            [(false, false), (false, true), (true, false), (true, true)].map(|(due, one)| looked_on(layout, due, one, 96));
        for (how, [looked, taken, window, slots]) in
            [("before", before), ("one look", one), ("due", due), ("one look and due", both)]
        {
            println!(
                "{name}, {how}: {looked:.0} slots a pass, {taken:.0} taken as written, {window:.0} in the window, of \
                 {slots:.0} buffer slots"
            );
        }
        assert_eq!(before[3], both[3], "{name}: the same buffer slots");
        assert!(before[3] > 25_000.0 && before[3] < 32_000.0, "{name}: {before:?}");
        // Marked again by the making: twice the slots written.
        assert!(one[1] < 0.6 * before[1], "{name}: {one:?} against {before:?}");
        assert!(due[2] < before[2], "{name}: {due:?} against {before:?}");
        assert!(both[0] < 0.7 * before[0], "{name}: {both:?} against {before:?}");
    }
    let ring = looked_on(layouts[0].1, true, true, 96);
    let turns = looked_on(layouts[0].1, false, true, 96);
    assert!(ring[2] < 0.1 * turns[2], "{ring:?} against {turns:?}");
}

/// With `VICTORY_HEAP_PASS_ONE_LOOK`, a slot the title writes once is
/// looked at once: by the pass after the write, which makes it, and not by
/// the next one; one the title writes again after that making is looked at
/// again, its new range kept; one written again between the call and the
/// making (beside) is looked at by the next pass too. Without the switch
/// the making marks the slot and the next pass looks at it a second time.
/// In place and beside, alike (the check counts the slots of the pass
/// beside, and finds each as at the call).
#[test]
fn a_slot_written_once_is_looked_at_once() {
    let Some(device) = crate::host::host() else {
        return;
    };
    for one_look in [false, true] {
        TEST_SPREAD.set(Some(false));
        TEST_ONE_LOOK.set(Some(one_look));
        let (Some(here), Some(beside)) = (title_heap(32), title_heap(32)) else {
            TEST_SPREAD.set(None);
            TEST_ONE_LOOK.set(None);
            return;
        };
        let base = 0x5f_0000_0000u64 + (u64::from(one_look) << 28);
        let id = 0x7e57_5f01 + usize::from(one_look);
        crate::shadow::register(id, base, 0x1_0000);
        let both = |slot: u32, first: u32| {
            write(here.1, slot, buffer_view(base, 0x1_0000, first, 8));
            write(beside.1, slot, buffer_view(base, 0x1_0000, first, 8));
        };
        both(3, 0);
        both(4, 8);
        // Per pass: the first element of slot 3's view written before it
        // (none: nothing written), and the slots it looks at with the
        // switch and without.
        let rounds: [(Option<u32>, usize, usize); 5] = [
            // Over every buffer slot.
            (None, 2, 2),
            // Slot 3 written: looked at once; without, slots 3 and 4 again.
            (Some(16), 1, 2),
            // Nothing written: without, slot 3 again.
            (None, 0, 1),
            // Written again after the making.
            (Some(24), 1, 1),
            (None, 0, 1),
        ];
        for (n, (first, with, without)) in rounds.into_iter().enumerate() {
            if let Some(first) = first {
                both(3, first);
            }
            let frame = 0x5f00 + n as u64;
            let lists = [(); 2].map(|()| device.new_list().expect("a list"));
            let made = in_place(&lists[0], here.1, here.2, frame);
            let run = Run::new();
            let pass = run.pass.clone();
            start(&lists[1], beside.1, beside.2, frame, run, true);
            lists[1].join_heap_passes();
            let checked = pass.checked().expect("a checked pass");
            let want = if one_look { with } else { without };
            assert_eq!(made.slots, want, "pass {n}, one look {one_look}: in place");
            assert_eq!(checked.slots, want as u64, "pass {n}, one look {one_look}: beside");
            assert_eq!((checked.differed, checked.changed), (0, 0), "pass {n}, one look {one_look}");
            assert_eq!(kept(here.2), kept(beside.2), "pass {n}, one look {one_look}: the ranges kept");
            for list in lists {
                list.close();
            }
        }
        let copy = crate::shadow::find(base, 0x1_0000).expect("a copy");
        assert_eq!(kept(beside.2).get(&3), Some(&(copy.id(), base + 24 * 16, base + 32 * 16)), "the new range");
        // Beside: slot 4 written before the call and again before the
        // making; the pass makes it from the second record, and the next
        // one looks at it again (the second write's mark), once.
        write(beside.1, 4, buffer_view(base, 0x1_0000, 40, 8));
        for (n, want) in [(5u64, 1u64), (6, 1), (7, 0)] {
            let list = device.new_list().expect("a list");
            let run = Run::new();
            let pass = run.pass.clone();
            let hold = (n == 5).then(hold_lane);
            start(&list, beside.1, beside.2, 0x5f00 + n, run, true);
            if hold.is_some() {
                write(beside.1, 4, buffer_view(base, 0x1_0000, 48, 8));
            }
            drop(hold);
            list.join_heap_passes();
            let checked = pass.checked().expect("a checked pass");
            assert_eq!(checked.slots, want, "pass {n}, one look {one_look}");
            assert_eq!(checked.differed, 0, "pass {n}, one look {one_look}");
            list.close();
        }
        assert_eq!(kept(beside.2).get(&4), Some(&(copy.id(), base + 48 * 16, base + 56 * 16)), "made from the second record");
        crate::shadow::unregister(id, base);
        for heap in [&here, &beside] {
            forget(heap.2);
        }
        for heap in [here, beside] {
            assert_eq!(release(heap.0), 0);
        }
    }
    TEST_SPREAD.set(None);
    TEST_ONE_LOOK.set(None);
}

/// Sets [`spread`], [`due_window`] and [`one_look`] on this thread (`None`:
/// as the environment says).
fn switches(on: Option<bool>) {
    TEST_SPREAD.set(on);
    TEST_DUE.set(on);
    TEST_ONE_LOOK.set(on);
}

/// Three heaps written alike: passed over in place with a pass over every
/// buffer slot each 16th pass, and with the spread, `VICTORY_HEAP_PASS_DUE`
/// and `VICTORY_HEAP_PASS_ONE_LOOK` in place and beside (checked every
/// pass); with the cache of made views and without. 24 buffer slots in runs
/// of several turns and on two buffers; before each pass after the first,
/// one of them written (each again 24 passes later), and one turned into a
/// constant view and back. At every pass the three keep the same ranges, in
/// place and beside show the same buffers, and the check finds every slot as
/// at the call; a pass looks at the slot written for it and at the ones no
/// pass looked at in the 15 before: none up to pass 15, then the slots of
/// the first pass not written since, then each slot written 16 passes
/// before.
#[test]
fn the_due_window_keeps_what_the_full_pass_keeps() {
    let Some(device) = crate::host::host() else {
        return;
    };
    let runs = 8u32;
    let slots: Vec<u32> = (0..runs).flat_map(|r| [r * RUN + 5, r * RUN + 20, r * RUN + 40]).collect();
    for cache in [0usize, 1 << 10] {
        TEST_FAST.set(Some((cache, false, false)));
        let (Some(full), Some(here), Some(beside)) = (title_heap(runs * RUN), title_heap(runs * RUN), title_heap(runs * RUN))
        else {
            TEST_FAST.set(None);
            return;
        };
        let base = 0x60_0000_0000u64 + ((cache as u64) << 28);
        let (one, two) = (0x7e57_6001 + cache, 0x7e57_6002 + cache);
        crate::shadow::register(one, base, 0x10_0000);
        crate::shadow::register(two, base + 0x100_0000, 0x1000);
        let all = |slot: u32, view: View| {
            for heap in [full.1, here.1, beside.1] {
                write(heap, slot, view.clone());
            }
        };
        for (k, &slot) in slots.iter().enumerate() {
            let k = k as u32;
            all(slot, match k % 3 {
                2 => buffer_view(base + 0x100_0000, 0x1000, k % 32, 4),
                _ => buffer_view(base, 0x10_0000, 4 * k, 8),
            });
        }
        for n in 0..=40u64 {
            if n > 0 {
                all(slots[((n - 1) % 24) as usize], buffer_view(base, 0x10_0000, 300 + n as u32, 8));
            }
            if n == 20 {
                all(slots[3], View::Constant(base + 0x400, 256));
            }
            if n == 22 {
                all(slots[3], buffer_view(base, 0x10_0000, 900, 8));
            }
            let frame = 0x6000 + n;
            let lists = [(); 3].map(|()| device.new_list().expect("a list"));
            switches(Some(false));
            in_place(&lists[0], full.1, full.2, frame);
            switches(Some(true));
            let made = in_place(&lists[1], here.1, here.2, frame);
            let run = Run::new();
            let pass = run.pass.clone();
            start(&lists[2], beside.1, beside.2, frame, run, true);
            switches(None);
            lists[2].join_heap_passes();
            let want = match n {
                0 => Some(24),
                1..=15 => Some(1),
                // The 8 slots of the first pass not written since.
                16 => Some(9),
                17..=19 => Some(2),
                _ => None,
            };
            if let Some(want) = want {
                assert_eq!(made.slots, want, "pass {n}, cache {cache}: in place");
            }
            assert!(made.slots <= 3 || n == 0 || n == 16, "pass {n}, cache {cache}: {} slots", made.slots);
            let checked = pass.checked().expect("a checked pass");
            assert_eq!(checked.slots, made.slots as u64, "pass {n}, cache {cache}: beside");
            assert_eq!((checked.differed, checked.changed), (0, 0), "pass {n}, cache {cache}");
            let due = kept(here.2);
            assert_eq!(due, kept(beside.2), "pass {n}, cache {cache}: in place and beside");
            for slot in 0..runs * RUN {
                assert_eq!(here.2.made(slot), beside.2.made(slot), "pass {n}, cache {cache}, slot {slot}: made");
                assert_eq!(here.2.shown_raw(slot), beside.2.shown_raw(slot), "pass {n}, cache {cache}, slot {slot}");
            }
            assert_eq!(due, kept(full.2), "pass {n}, cache {cache}: with the window of slots due and without the spread");
            for list in lists {
                list.close();
            }
        }
        crate::shadow::unregister(one, base);
        crate::shadow::unregister(two, base + 0x100_0000);
        for heap in [&full, &here, &beside] {
            forget(heap.2);
        }
        for heap in [full, here, beside] {
            assert_eq!(release(heap.0), 0);
        }
    }
    TEST_FAST.set(None);
}

/// What a pass counts of its slots (`parts`): with the cache of made views,
/// buffer views made and kept in it, one of the same view copied from it,
/// a record of no view of ours; at the next pass, the views made from their
/// record already; every view with a copy and range.
#[test]
fn a_pass_counts_what_its_slots_came_to() {
    use parts::Count;
    if crate::host::host().is_none() {
        return;
    }
    TEST_FAST.set(Some((1 << 10, false, false)));
    let Some((object, heap, host)) = title_heap(64) else {
        TEST_FAST.set(None);
        return;
    };
    let base = 0x61_0000_0000u64;
    crate::shadow::register(0x7e57_6101, base, 0x1_0000);
    // At different addresses: the cache keeps two records a set, and the
    // set is chosen by the record's low bits (the address of a placed view).
    for slot in 1..=4 {
        write(heap, slot, buffer_view(base + 0x1000 * u64::from(slot), 0x1000, 0, 8));
    }
    // The view of slot 1 again; slot 6 never written.
    write(heap, 5, buffer_view(base + 0x1000, 0x1000, 0, 8));
    let mut ranges = Ranges::new();
    let mut slots: Vec<u32> = (1..=6).collect();
    let took = buffers_taken(heap, host, &mut slots, &mut ranges, None, true);
    let of = |took: &parts::Took| {
        [Count::Miss, Count::Hit, Count::NotOurs, Count::Same, Count::Range, Count::NoRange].map(|c| took.of(c))
    };
    assert_eq!(of(&took), [4, 1, 1, 0, 5, 0], "{took:?}");
    assert_eq!(took.of(Count::Chunks), 1, "one chunk under one lock");
    assert!(took.of(Count::Lookups) > 0);
    assert_eq!(ranges.len(), 5);
    let took = buffers_taken(heap, host, &mut slots, &mut ranges, None, true);
    assert_eq!(of(&took), [0, 0, 1, 5, 5, 0], "{took:?}");
    assert!(!parts::counting(), "the pass's counting ended with it");
    crate::shadow::unregister(0x7e57_6101, base);
    forget(host);
    assert_eq!(release(object), 0);
    TEST_FAST.set(None);
}

/// What a pass on one thread over `heap`'s slot `slot` finds of it (the
/// heap's slots made already: nothing is made).
fn f1_for(heap: &(This, &'static Heap, &'static HostHeap), slot: u32) -> Found {
    let mut found = Vec::new();
    buffers_taken(heap.1, heap.2, &mut [slot], &mut Ranges::new(), Some(&mut found), true);
    found.pop().expect("the slot")
}

/// What the passes kept for `host`'s slots and left in them: per slot,
/// whether its host slot is made and the buffer it shows.
fn left(host: &HostHeap, slots: u32) -> Vec<(bool, Option<usize>)> {
    (0..slots).map(|slot| (host.made(slot), host.shown_raw(slot))).collect()
}

/// Two heaps written alike, one passed over on one thread and the other on
/// four (`VICTORY_HEAP_PASS_THREADS`), round after round (every slot looked
/// at; buffer views moved, emptied and turned into constant views, memory
/// outside the title's buffers), with the cache of made views and without,
/// and with a heap map whose pool the threads take slots from at once: the
/// passes keep the same ranges and find the same views, copies and ranges,
/// every slot is made alike and shows the same buffer, no two title slots
/// share a pool slot, and the check finds nothing that differs.
#[test]
fn the_pass_on_threads_makes_and_keeps_what_the_pass_on_one_thread_does() {
    const SLOTS: u32 = 1024;
    if crate::host::host().is_none() {
        return;
    }
    for (cache, map) in [(0usize, None), (1 << 10, None), (1 << 10, Some((64u32, 64 + 960))), (0, Some((64, 64 + 960)))] {
        TEST_FAST.set(Some((cache, false, false)));
        crate::descriptor::host::TEST_MAP.set(map);
        let (Some(one), Some(four)) = (title_heap(SLOTS), title_heap(SLOTS)) else {
            crate::descriptor::host::TEST_MAP.set(None);
            TEST_FAST.set(None);
            return;
        };
        crate::descriptor::host::TEST_MAP.set(None);
        let tag = cache as u64 + u64::from(map.is_some()) * 7;
        let base = 0x62_0000_0000u64 + (tag << 28);
        let (b1, b2) = (0x7e57_6201 + tag as usize, 0x7e57_6202 + tag as usize);
        crate::shadow::register(b1, base, 0x10_0000);
        crate::shadow::register(b2, base + 0x100_0000, 0x4_0000);
        let both = |slot: u32, view: View| {
            write(one.1, slot, view.clone());
            write(four.1, slot, view);
        };
        let view_of = |k: u32| match k % 9 {
            0 => View::Constant(base + 0x200 + u64::from(k % 16) * 0x100, 256),
            1 => buffer_view(base + 0x800_0000 + u64::from(k % 5) * 0x1_0000, 0x2000, 0, 16),
            2 | 3 => buffer_view(base + 0x100_0000, 0x4_0000, 16 * (k % 64), 32),
            _ => buffer_view(base + u64::from(k % 13) * 0x1_0000, 0x1_0000, 4 * (k % 32), 8),
        };
        for slot in 1..SLOTS {
            if slot % 11 != 0 {
                both(slot, view_of(slot));
            }
        }
        let (mut r1, mut r4) = (Ranges::new(), Ranges::new());
        for round in 0..4u32 {
            if round > 0 {
                for slot in (round..SLOTS).step_by(7) {
                    match (slot + round) % 5 {
                        0 => both(slot, View::NullConstant),
                        _ => both(slot, view_of(slot * 3 + round)),
                    }
                }
            }
            let all: Vec<u32> = (0..SLOTS).collect();
            let (mut f1, mut f4) = (Vec::new(), Vec::new());
            threads::TEST_THREADS.set(Some(1));
            buffers(one.1, one.2, &mut all.clone(), &mut r1, Some(&mut f1), true);
            threads::TEST_THREADS.set(Some(4));
            assert_eq!(threads::wanted(all.len()), Some(4));
            buffers(four.1, four.2, &mut all.clone(), &mut r4, Some(&mut f4), true);
            threads::TEST_THREADS.set(None);
            let what = format!("round {round}, cache {cache}, map {map:?}");
            assert_eq!(f1, f4, "{what}: what each slot came to");
            let keys = |r: &Ranges| -> std::collections::BTreeMap<u32, (u64, u64, u64)> {
                r.iter().map(|(&i, (s, from, to))| (i, (s.id(), *from, *to))).collect()
            };
            assert!(!r1.is_empty(), "{what}");
            assert_eq!(keys(&r1), keys(&r4), "{what}: the ranges kept");
            assert_eq!(left(one.2, SLOTS), left(four.2, SLOTS), "{what}: the slots made");
            let mut pooled: Vec<u32> = (0..SLOTS).filter_map(|slot| four.2.host_of(slot)).collect();
            let n = pooled.len();
            pooled.sort_unstable();
            pooled.dedup();
            assert_eq!(pooled.len(), n, "{what}: a host slot for each title slot");
            let checked = threads::check(four.1, four.2, &all, &f4, true);
            assert_eq!(checked.slots, f4.len() as u64, "{what}");
            assert_eq!((checked.differed, checked.changed, checked.made), (0, 0, 0), "{what}: {checked:?}");
        }
        // A slot left unmade (its host view dropped as if its memory went):
        // the check finds it, and the pass on one thread makes it again.
        let slot = (1..SLOTS).find(|&slot| four.2.made(slot)).expect("a made slot");
        assert!(four.2.texture_slot_gone(slot));
        let all: Vec<u32> = (0..SLOTS).collect();
        let mut f4 = Vec::new();
        threads::TEST_THREADS.set(Some(4));
        let mut slots = all.clone();
        // Not this slot: the threads would make it again.
        slots.retain(|&s| s != slot);
        buffers(four.1, four.2, &mut slots, &mut r4, Some(&mut f4), true);
        threads::TEST_THREADS.set(None);
        f4.push(f1_for(&one, slot));
        f4.sort_by_key(|f| f.index);
        let checked = threads::check(four.1, four.2, &all, &f4, true);
        assert_eq!((checked.differed, checked.changed, checked.made), (1, 0, 1), "cache {cache}, map {map:?}: {checked:?}");
        assert!(four.2.made(slot), "made again");
        crate::shadow::unregister(b1, base);
        crate::shadow::unregister(b2, base + 0x100_0000);
        forget(one.2);
        forget(four.2);
        assert_eq!(release(one.0), 0);
        assert_eq!(release(four.0), 0);
    }
    TEST_FAST.set(None);
}

/// With a heap map whose pool is used up, a thread of the pass gives back
/// no other slot's pool slot: the slots that wanted one are made once the
/// threads are done, after the pool slots of title slots without a live
/// view (overwritten as bytes, not in the pass) were given back.
#[test]
fn the_threads_leave_a_used_up_pool_to_the_pass_after_them() {
    const SLOTS: u32 = 512;
    if crate::host::host().is_none() {
        return;
    }
    TEST_FAST.set(Some((1 << 10, false, false)));
    // The identity window up to 16, a pool of 8.
    crate::descriptor::host::TEST_MAP.set(Some((16, 24)));
    let Some((object, heap, host)) = title_heap(SLOTS) else {
        crate::descriptor::host::TEST_MAP.set(None);
        TEST_FAST.set(None);
        return;
    };
    crate::descriptor::host::TEST_MAP.set(None);
    let base = 0x63_0000_0000u64;
    crate::shadow::register(0x7e57_6301, base, 0x10_0000);
    let view = |k: u32| buffer_view(base + u64::from(k) * 0x1000, 0x1000, 0, 8);
    let held: Vec<u32> = (100..108).collect();
    for &slot in &held {
        write(heap, slot, view(slot));
    }
    let mut ranges = Ranges::new();
    threads::TEST_THREADS.set(Some(4));
    buffers(heap, host, &mut held.clone(), &mut ranges, None, true);
    assert!(held.iter().all(|&slot| host.host_of(slot).is_some()), "the pool's 8 slots held");
    // Four of them overwritten as bytes; four new slots past the window,
    // their records copied in as bytes too (from slots below the window: a
    // call there takes no pool slot, which past it would give the dead ones
    // back at once).
    for &slot in &held[..4] {
        // SAFETY: a descriptor slot of the title heap.
        unsafe { (heap.handle(slot).expect("a slot") as *mut u8).write_bytes(0, 32) };
    }
    let new: Vec<u32> = (300..304).collect();
    for (k, &slot) in new.iter().enumerate() {
        let below = 1 + k as u32;
        write(heap, below, view(slot));
        // SAFETY: two descriptor slots of the title heap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                heap.handle(below).expect("a slot") as *const u8,
                heap.handle(slot).expect("a slot") as *mut u8,
                32,
            )
        };
    }
    // A pass over the new slots and others with nothing written, on four
    // threads (more than two chunks).
    let mut slots: Vec<u32> = new.iter().copied().chain(376..SLOTS).collect();
    buffers(heap, host, &mut slots, &mut ranges, None, true);
    threads::TEST_THREADS.set(None);
    assert!(held[..4].iter().all(|&slot| host.host_of(slot).is_none()), "given back after the threads");
    assert!(held[4..].iter().all(|&slot| host.host_of(slot).is_some()), "still held");
    // The new slots take the ones given back once no submission that may
    // read them runs (other tests' may); made or not, each is the slot's own.
    let mut pooled: Vec<u32> = (0..SLOTS).filter_map(|slot| host.host_of(slot)).collect();
    let n = pooled.len();
    pooled.sort_unstable();
    pooled.dedup();
    assert_eq!(pooled.len(), n, "a host slot for each title slot");
    for &slot in &new {
        assert_eq!(host.made(slot), host.host_of(slot).is_some(), "slot {slot}");
        assert!(ranges.contains_key(&slot), "slot {slot}: its range kept all the same");
    }
    crate::shadow::unregister(0x7e57_6301, base);
    forget(host);
    assert_eq!(release(object), 0);
    TEST_FAST.set(None);
}

/// Passes on four threads while another thread writes slots all the while
/// (each call takes the heap's slot state alone, between the threads'
/// chunks): nothing waits for ever, no two title slots share a pool slot,
/// and once the writes stop, a pass on the threads leaves every slot as a
/// pass on one thread leaves it (the check finds nothing that differs).
#[test]
fn the_threads_and_the_title_s_writes_take_turns() {
    const SLOTS: u32 = 2048;
    if crate::host::host().is_none() {
        return;
    }
    TEST_FAST.set(Some((1 << 10, true, false)));
    crate::descriptor::host::TEST_MAP.set(Some((64, 64 + 1984)));
    let Some((object, heap, host)) = title_heap(SLOTS) else {
        crate::descriptor::host::TEST_MAP.set(None);
        TEST_FAST.set(None);
        return;
    };
    crate::descriptor::host::TEST_MAP.set(None);
    let base = 0x64_0000_0000u64;
    crate::shadow::register(0x7e57_6401, base, 0x10_0000);
    let view = move |k: u64| buffer_view(base + (k % 61) * 0x4000, 0x4000, (k % 7) as u32, 8);
    for slot in 1..SLOTS {
        write(heap, slot, view(u64::from(slot)));
    }
    let stop = Arc::new(AtomicBool::new(false));
    let heap_at = std::ptr::from_ref(heap) as usize;
    let writer = {
        let stop = stop.clone();
        std::thread::spawn(move || {
            // SAFETY: the heap lives until the thread is joined.
            let heap = unsafe { &*(heap_at as *const Heap) };
            let mut next = draws(0x6401);
            let mut writes = 0u64;
            while !stop.load(Relaxed) {
                let k = next();
                let slot = 1 + (k % u64::from(SLOTS - 1)) as u32;
                write(heap, slot, if k % 13 == 0 { View::NullConstant } else { view(k >> 8) });
                writes += 1;
            }
            writes
        })
    };
    let mut ranges = Ranges::new();
    threads::TEST_THREADS.set(Some(4));
    for _ in 0..12 {
        let mut slots: Vec<u32> = (0..SLOTS).collect();
        buffers(heap, host, &mut slots, &mut ranges, None, true);
    }
    stop.store(true, Relaxed);
    let writes = writer.join().expect("the writing thread");
    assert!(writes > 0);
    let all: Vec<u32> = (0..SLOTS).collect();
    let mut found = Vec::new();
    buffers(heap, host, &mut all.clone(), &mut ranges, Some(&mut found), true);
    threads::TEST_THREADS.set(None);
    let mut pooled: Vec<u32> = (0..SLOTS).filter_map(|slot| host.host_of(slot)).collect();
    let n = pooled.len();
    pooled.sort_unstable();
    pooled.dedup();
    assert_eq!(pooled.len(), n, "a host slot for each title slot");
    let checked = threads::check(heap, host, &all, &found, true);
    assert_eq!((checked.differed, checked.changed, checked.made), (0, 0, 0), "{checked:?} after {writes} writes");
    crate::shadow::unregister(0x7e57_6401, base);
    forget(host);
    assert_eq!(release(object), 0);
    TEST_FAST.set(None);
}
