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
