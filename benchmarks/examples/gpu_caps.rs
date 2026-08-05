//! Report which modern wgpu features this adapter actually exposes, so the
//! roadmap is grounded in hardware rather than in the spec.
use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::render::settings::WgpuFeatures;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, report)
        .run();
}

fn report(device: Res<RenderDevice>, mut exit: MessageWriter<AppExit>) {
    let f = device.features();
    let limits = device.limits();
    for (name, flag) in [
        ("MULTI_DRAW_INDIRECT_COUNT", WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT),
        ("INDIRECT_FIRST_INSTANCE", WgpuFeatures::INDIRECT_FIRST_INSTANCE),
        ("SUBGROUP", WgpuFeatures::SUBGROUP),
        ("SUBGROUP_VERTEX", WgpuFeatures::SUBGROUP_VERTEX),
        ("DUAL_SOURCE_BLENDING", WgpuFeatures::DUAL_SOURCE_BLENDING),
        ("TEXTURE_BINDING_ARRAY", WgpuFeatures::TEXTURE_BINDING_ARRAY),
        ("PARTIALLY_BOUND_BINDING_ARRAY", WgpuFeatures::PARTIALLY_BOUND_BINDING_ARRAY),
        ("STORAGE_RESOURCE_BINDING_ARRAY", WgpuFeatures::STORAGE_RESOURCE_BINDING_ARRAY),
        ("SHADER_INT64_ATOMIC_ALL_OPS", WgpuFeatures::SHADER_INT64_ATOMIC_ALL_OPS),
        ("TEXTURE_ATOMIC", WgpuFeatures::TEXTURE_ATOMIC),
        ("SHADER_FLOAT32_ATOMIC", WgpuFeatures::SHADER_FLOAT32_ATOMIC),
        ("EXPERIMENTAL_MESH_SHADER", WgpuFeatures::EXPERIMENTAL_MESH_SHADER),
    ] {
        info!("{:<32} {}", name, if f.contains(flag) { "YES" } else { "no" });
    }
    info!("max_compute_workgroup_size_x = {}", limits.max_compute_workgroup_size_x);
    info!("max_storage_buffer_binding_size = {} MB", limits.max_storage_buffer_binding_size / (1024 * 1024));
    exit.write(AppExit::Success);
}
