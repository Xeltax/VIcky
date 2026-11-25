//! GPU readback for the two Series S display passes.

use super::*;

#[test]
fn stretch_and_overlay_preserve_the_expected_pixels() {
    check_display(None, [[0, 0, 255, 255], [0, 0, 255, 255], [0, 128, 127, 255],
                         [255, 0, 0, 255], [255, 0, 0, 255]]);
}

#[test]
fn dim_scene_colour_is_visible_beneath_the_interface() {
    check_display(Some([0.001, 0.001, 0.001, 0.0]),
        [[109, 109, 109, 255], [109, 109, 109, 255], [54, 182, 54, 255],
         [255, 0, 0, 255], [255, 0, 0, 255]]);
}

fn check_display(dim_scene: Option<[f32; 4]>, expected: [[u8; 4]; 5]) {
    use crate::host::{host, transition};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT as RGBA16F;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM as RGBA8;
    let Some(host) = host() else {
        eprintln!("no D3D12 device; skipping display readback");
        return;
    };
    let texture = |width: u32, height, format, flags, state| {
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a texture with one subresource.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: u64::from(width),
                        Height: height,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: format,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Flags: flags,
                        ..Default::default()
                    },
                    state,
                    None,
                    &mut resource,
                )
                .unwrap();
        }
        resource.unwrap()
    };
    let sub = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: the copy location borrows this resource.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let rows = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: the copy location borrows this resource.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: RGBA8,
                    Width: 5,
                    Height: 3,
                    Depth: 1,
                    RowPitch: 256,
                },
            },
        },
    };
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let rtv = D3D12_RESOURCE_STATE_RENDER_TARGET;
    let scene_format = if dim_scene.is_some() { RGBA16F } else { RGBA8 };
    let scene = texture(3, 2, scene_format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, rtv);
    let ui = texture(5, 3, RGBA8, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, rtv);
    let target = texture(
        5,
        3,
        RGBA8,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        uav,
    );
    let readback = host.read_back_buffer(768).unwrap();
    let commands = host.new_list().unwrap();
    let list = &commands.list;
    // SAFETY: both views and all resources live through command completion.
    unsafe {
        let scene_view = host.rtv(Some(&scene), None);
        let ui_view = host.rtv(Some(&ui), None);
        list.ClearRenderTargetView(scene_view, &dim_scene.unwrap_or([0.0, 0.0, 1.0, 1.0]), None);
        list.ClearRenderTargetView(ui_view, &[0.0; 4], None);
        let strip = |left, right| RECT {
            left,
            right,
            top: 0,
            bottom: 3,
        };
        list.ClearRenderTargetView(
            ui_view,
            &[0.0, 128.0 / 255.0, 0.0, 128.0 / 255.0],
            Some(&[strip(2, 3)]),
        );
        list.ClearRenderTargetView(ui_view, &[1.0, 0.0, 0.0, 1.0], Some(&[strip(3, 5)]));
        list.ResourceBarrier(&[
            transition(&scene, rtv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(&ui, rtv, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
        ]);
        assert!(record_stretch(&host.device, list, &target, &scene, [3, 2]));
        list.ResourceBarrier(&[transition(&target, uav, rtv)]);
        if dim_scene.is_some() {
            list.ResourceBarrier(&[transition(&scene, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)]);
            assert!(record(&host.device, list, &target, RGBA8, (5, 3), (&scene, scene_format), Layer::Scene));
        }
        assert!(record(
            &host.device,
            list,
            &target,
            RGBA8,
            (5, 3),
            (&ui, RGBA8),
            Layer::Overlay,
        ));
        list.ResourceBarrier(&[transition(&target, rtv, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyTextureRegion(&rows(&readback), 0, 0, 0, &sub(&target), None);
    }
    commands.close();
    host.execute(&[&commands]);
    // SAFETY: execute waited for the copy. Each row contains five texels.
    let mut ptr = std::ptr::null_mut();
    let mut pixels = [[0u8; 20]; 3];
    unsafe {
        readback.Map(0, None, Some(&mut ptr)).unwrap();
        for (y, row) in pixels.iter_mut().enumerate() {
            row.copy_from_slice(std::slice::from_raw_parts(
                ptr.cast::<u8>().add(y * 256),
                20,
            ));
        }
        readback.Unmap(0, None);
    }
    for row in pixels {
        assert_eq!(&row[..], expected.as_flattened());
    }
}
