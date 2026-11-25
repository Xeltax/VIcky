//! Adapts EatCodeSleep2's synthetic reservation scenario from VIctory PR #1126
//! (bb65bc32a98eaac94d5e79f43d550b94cc809396) to the fork's mapped view heap.

use std::sync::Arc;

use victory_shader::emit::HostShader;
use victory_shader::{HwStage, ResourceType, RootType, UsedResource};
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap;
use windows::core::{Interface, s};

use super::{Shader, Translated, host::HostPipeline};
use crate::descriptor::host::{DYNAMIC, TEST_MAP};
use crate::draw::State;
use crate::object::{S_OK, This, slot};

struct TitleHeap(This);

impl TitleHeap {
    fn new(kind: u32, count: u32) -> Self {
        // Shader-visible title heap description with an empty placement extension.
        let desc = [kind, count, 1, 0, 0, 0, 0, 0];
        let iid = ID3D12DescriptorHeap::IID;
        let mut heap = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                desc.as_ptr().cast(),
                std::ptr::from_ref(&iid).cast(),
                &mut heap,
            ),
            S_OK
        );
        Self(heap)
    }
}

impl Drop for TitleHeap {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

fn compute_pipeline(used: &[UsedResource]) -> HostPipeline {
    let source = b"[numthreads(1,1,1)] void main() {}";
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and out pointer are valid for the call.
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("rejected_bind"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            None,
        )
    }
    .expect("compile the no-op compute shader");
    let code = code.expect("compiled shader blob");
    // SAFETY: the blob owns the byte range.
    let dxil = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast(), code.GetBufferSize()).to_vec()
    };

    let shader = Shader {
        stage: "compute",
        hash: "rejected_bind".to_owned(),
        container: Vec::new(),
        translated: Some(Translated {
            host: HostShader {
                stage: HwStage::Cs,
                dxil,
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                root_indexed_views: Vec::new(),
            },
            used: used.to_vec(),
            stores: false,
        })
        .into(),
    };
    HostPipeline::new(&[Arc::new(shader)], None).expect("create the test pipeline")
}

#[test]
fn rejected_bind_releases_only_its_new_tables() {
    let Some(host) = crate::host::host() else {
        eprintln!("skipping rejected bind regression: no D3D12 device");
        return;
    };
    let table = |slot, width: u32| UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: (width - 1) * 8,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot,
    };
    let used = [table(0, 2), table(1, 4), table(2, 3)];
    let pipeline = compute_pipeline(&used);
    let views = TitleHeap::new(0, 32);
    let samplers = TitleHeap::new(1, 1);
    let title = crate::descriptor::heap(views.0);
    let previous_map = TEST_MAP.replace(Some((16, 24)));
    let mapped = title.host();
    TEST_MAP.set(previous_map);
    let heap = mapped.expect("create the mapped host view heap");
    assert_eq!(heap.count(), 16);

    let list = host.new_list().expect("host command list");
    let mut state = State::default();
    state.set_heaps(&[views.0, samplers.0]);
    let blocker = heap
        .dynamic(DYNAMIC - 8)
        .expect("leave eight dynamic slots free");
    let four_slot_table = title.handle(18).unwrap();
    let three_slot_table = title.handle(22).unwrap();
    state.set_table(true, 0, title.handle(16).unwrap());
    assert!(
        state.bind(&list, &pipeline, true),
        "accept the two-slot table"
    );

    let free = heap
        .dynamic(6)
        .expect("six slots remain after the accepted bind");
    assert!(
        heap.dynamic(1).is_none(),
        "the first bind must reserve two slots"
    );
    heap.release(free, 6);

    // Six free slots allow the first new table (four), then refuse the next (three).
    state.set_table(true, 1, four_slot_table);
    state.set_table(true, 2, three_slot_table);
    assert!(
        !state.bind(&list, &pipeline, true),
        "reject the two new tables"
    );

    let probe = heap
        .dynamic(6)
        .expect("rejected bind must restore all six free slots");
    assert!(
        heap.dynamic(1).is_none(),
        "the accepted range must still be reserved"
    );

    // With a full heap, successful binds can only reuse cached tables.
    state.set_table(true, 1, 0);
    state.set_table(true, 2, 0);
    assert!(
        state.bind(&list, &pipeline, true),
        "reuse the accepted table"
    );
    state.set_table(true, 1, four_slot_table);
    assert!(
        !state.bind(&list, &pipeline, true),
        "the rolled-back four-slot table must be absent from the cache"
    );

    heap.release(probe, 6);
    heap.release(blocker, DYNAMIC - 8);
    state.set_table(true, 2, three_slot_table);
    assert!(
        state.bind(&list, &pipeline, true),
        "retry after pressure clears"
    );

    list.close();
    drop(list);
    drop(state);
    let all = heap
        .dynamic(DYNAMIC)
        .expect("dropping the recording must return every slot");
    heap.release(all, DYNAMIC);
}
