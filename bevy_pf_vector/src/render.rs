//! Render-world half of the engine: extract, tessellate-on-first-sight,
//! persistent buffers, pipelines, and the vector pass itself.
//!
//! Performance model (portable across NVIDIA/AMD/Intel/Apple via wgpu — no
//! vendor extensions):
//! - Geometry is tessellated once per unique content hash and lives in
//!   persistent vertex/index buffers. Steady-state per-frame CPU work is
//!   hashing + one 36-byte instance write per shape.
//! - Exact-coverage triangles mean near-zero overdraw, unlike SDF quad
//!   renderers that shade full bounding rects.
//! - Opaque instances (alpha == 1, the HUD common case) draw with depth
//!   write + test and no blending: early-z rejects hidden fragments on both
//!   immediate-mode and tile-based GPUs, and draw order becomes irrelevant,
//!   so opaque batches group purely by geometry for maximal instancing.
//! - Only translucent instances pay blend cost, back-to-front, depth
//!   read-only.
//! - Instances are 36 bytes (2x2 affine + translation + z + RGBA8), not a
//!   mat4 — half the vertex-fetch bandwidth of the naive layout.

use std::collections::HashMap;
use std::ops::Range;

use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::core_pipeline::core_2d::CORE_2D_DEPTH_FORMAT;
use bevy::core_pipeline::schedule::{Core2d, Core2dSystems};
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_resource::binding_types::uniform_buffer;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    BlendState, Buffer, BufferInitDescriptor, BufferUsages, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState,
    FragmentState, IndexFormat, MultisampleState, PipelineCache, PrimitiveState,
    RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, StencilState, StoreOp,
    TextureFormat, UniformBuffer, VertexFormat, VertexState, VertexStepMode,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy::shader::Shader;

use crate::tess;
use crate::VectorShape;

pub const VECTOR_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("7a3f1c2e-9b4d-4b8a-a2f0-5e1d3c6b9f01");

pub struct VectorRenderPlugin;

impl Plugin for VectorRenderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, VECTOR_SHADER_HANDLE, "vector.wgsl", Shader::from_wgsl);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<GeometryCache>()
            .init_resource::<ExtractedShapes>()
            .init_resource::<VectorBuffers>()
            .init_resource::<VectorViewBindGroups>()
            .init_resource::<VectorPipeline>()
            .add_systems(ExtractSchedule, extract_shapes)
            .add_systems(
                Render,
                (
                    queue_vector_pipelines.in_set(RenderSystems::Queue),
                    prepare_vector_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_view_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(Core2d, vector_pass.in_set(Core2dSystems::EarlyPostProcess));
    }
}

// ---------------------------------------------------------------- gpu data

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVertex {
    position: [f32; 2],
}

/// 36 bytes. Columns of the 2x2 linear part, translation + z, RGBA8 color.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstance {
    linear: [f32; 4],
    translation_z: [f32; 4],
    color: [u8; 4],
}

fn vertex_buffer_layout() -> VertexBufferLayout {
    VertexBufferLayout::from_vertex_formats(VertexStepMode::Vertex, [VertexFormat::Float32x2])
}

fn instance_buffer_layout() -> VertexBufferLayout {
    let mut layout = VertexBufferLayout::from_vertex_formats(
        VertexStepMode::Instance,
        [
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Unorm8x4,
        ],
    );
    // from_vertex_formats numbers locations from 0; instance attributes
    // follow the vertex buffer's location 0.
    for (i, attribute) in layout.attributes.iter_mut().enumerate() {
        attribute.shader_location = 1 + i as u32;
    }
    layout
}

// ---------------------------------------------------------------- resources

#[derive(Clone, Copy)]
struct GeometryRange {
    base_vertex: i32,
    first_index: u32,
    index_count: u32,
}

/// Tessellated geometry, appended once per unique content hash. Persistent
/// across frames; `dirty` triggers a GPU re-upload when new geometry lands.
#[derive(Resource, Default)]
pub struct GeometryCache {
    ranges: HashMap<u64, Option<GeometryRange>>,
    vertices: Vec<GpuVertex>,
    indices: Vec<u32>,
    dirty: bool,
}

impl GeometryCache {
    fn ensure(&mut self, key: u64, tessellate: impl FnOnce() -> Option<tess::TessellatedGeometry>) {
        if self.ranges.contains_key(&key) {
            return;
        }
        let range = tessellate().map(|geometry| {
            let base_vertex = self.vertices.len() as i32;
            let first_index = self.indices.len() as u32;
            self.vertices
                .extend(geometry.vertices.iter().map(|&position| GpuVertex { position }));
            self.indices.extend_from_slice(&geometry.indices);
            self.dirty = true;
            GeometryRange {
                base_vertex,
                first_index,
                index_count: geometry.indices.len() as u32,
            }
        });
        self.ranges.insert(key, range);
    }
}

struct ExtractedInstance {
    geometry: u64,
    z: f32,
    linear: [f32; 4],
    translation: [f32; 2],
    color: [u8; 4],
    opaque: bool,
}

#[derive(Resource, Default)]
pub struct ExtractedShapes(Vec<ExtractedInstance>);

struct VectorBatch {
    indices: Range<u32>,
    base_vertex: i32,
    instances: Range<u32>,
}

#[derive(Resource, Default)]
pub struct VectorBuffers {
    vertex: Option<Buffer>,
    index: Option<Buffer>,
    instance: Option<Buffer>,
    instance_capacity: usize,
    opaque_batches: Vec<VectorBatch>,
    blend_batches: Vec<VectorBatch>,
}

#[derive(Default)]
struct ViewEntry {
    uniform: UniformBuffer<Mat4>,
    bind_group: Option<BindGroup>,
}

#[derive(Resource, Default)]
pub struct VectorViewBindGroups {
    per_view: HashMap<Entity, ViewEntry>,
}

#[derive(Resource)]
pub struct VectorPipeline {
    view_layout: BindGroupLayoutDescriptor,
    variants: HashMap<(TextureFormat, u32, bool), CachedRenderPipelineId>,
}

impl Default for VectorPipeline {
    fn default() -> Self {
        Self {
            view_layout: BindGroupLayoutDescriptor::new(
                "pf_vector_view_layout",
                &BindGroupLayoutEntries::single(
                    ShaderStages::VERTEX,
                    uniform_buffer::<Mat4>(false),
                ),
            ),
            variants: HashMap::new(),
        }
    }
}

impl VectorPipeline {
    fn ensure(
        &mut self,
        cache: &PipelineCache,
        format: TextureFormat,
        samples: u32,
        opaque: bool,
    ) -> CachedRenderPipelineId {
        *self
            .variants
            .entry((format, samples, opaque))
            .or_insert_with(|| {
                cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some(
                        if opaque { "pf_vector_opaque_pipeline" } else { "pf_vector_blend_pipeline" }
                            .into(),
                    ),
                    layout: vec![self.view_layout.clone()],
                    vertex: VertexState {
                        shader: VECTOR_SHADER_HANDLE,
                        entry_point: Some("vertex".into()),
                        shader_defs: Vec::new(),
                        buffers: vec![vertex_buffer_layout(), instance_buffer_layout()],
                    },
                    fragment: Some(FragmentState {
                        shader: VECTOR_SHADER_HANDLE,
                        entry_point: Some("fragment".into()),
                        shader_defs: Vec::new(),
                        targets: vec![Some(ColorTargetState {
                            format,
                            blend: (!opaque).then_some(BlendState::ALPHA_BLENDING),
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    primitive: PrimitiveState::default(),
                    // Matches bevy 2D convention: higher z = closer.
                    depth_stencil: Some(DepthStencilState {
                        format: CORE_2D_DEPTH_FORMAT,
                        depth_write_enabled: Some(opaque),
                        depth_compare: Some(CompareFunction::GreaterEqual),
                        stencil: StencilState::default(),
                        bias: DepthBiasState::default(),
                    }),
                    multisample: MultisampleState {
                        count: samples,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    immediate_size: 0,
                    zero_initialize_workgroup_memory: false,
                })
            })
    }

    fn get(
        &self,
        format: TextureFormat,
        samples: u32,
        opaque: bool,
    ) -> Option<CachedRenderPipelineId> {
        self.variants.get(&(format, samples, opaque)).copied()
    }
}

// ---------------------------------------------------------------- systems

fn pack_color(color: LinearRgba) -> [u8; 4] {
    let quantize = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    [
        quantize(color.red),
        quantize(color.green),
        quantize(color.blue),
        quantize(color.alpha),
    ]
}

/// Copies shape instances into the render world, tessellating any geometry
/// seen for the first time. Steady-state cost is hashing plus a 36-byte push
/// per instance — no per-frame path processing.
fn extract_shapes(
    shapes: Extract<Query<(&VectorShape, &GlobalTransform)>>,
    mut cache: ResMut<GeometryCache>,
    mut extracted: ResMut<ExtractedShapes>,
) {
    extracted.0.clear();
    for (shape, transform) in &shapes {
        let model = transform.to_matrix();
        let linear = [
            model.x_axis.x,
            model.x_axis.y,
            model.y_axis.x,
            model.y_axis.y,
        ];
        let translation = [model.w_axis.x, model.w_axis.y];
        let z = model.w_axis.z;
        if let Some(color) = shape.style.fill {
            let key = tess::fill_key(&shape.commands);
            cache.ensure(key, || tess::tessellate_fill(&shape.commands));
            extracted.0.push(ExtractedInstance {
                geometry: key,
                z,
                linear,
                translation,
                color: pack_color(color),
                opaque: color.alpha >= 1.0,
            });
        }
        if let Some(stroke) = shape.style.stroke {
            let key = tess::stroke_key(&shape.commands, &stroke);
            cache.ensure(key, || tess::tessellate_stroke(&shape.commands, &stroke));
            extracted.0.push(ExtractedInstance {
                geometry: key,
                // Strokes draw over their own fill at equal z.
                z: z + 1.0e-4,
                linear,
                translation,
                color: pack_color(stroke.color),
                opaque: stroke.color.alpha >= 1.0,
            });
        }
    }
}

fn queue_vector_pipelines(
    mut pipeline: ResMut<VectorPipeline>,
    cache: Res<PipelineCache>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    for (view, msaa) in &views {
        pipeline.ensure(&cache, view.target_format, msaa.samples(), true);
        pipeline.ensure(&cache, view.target_format, msaa.samples(), false);
    }
}

fn push_batch(
    batches: &mut Vec<VectorBatch>,
    range: GeometryRange,
    instance_index: u32,
    contiguous: bool,
) {
    if contiguous {
        batches.last_mut().unwrap().instances.end = instance_index + 1;
    } else {
        batches.push(VectorBatch {
            indices: range.first_index..range.first_index + range.index_count,
            base_vertex: range.base_vertex,
            instances: instance_index..instance_index + 1,
        });
    }
}

/// Uploads geometry when it changed and rebuilds the per-frame instance
/// buffer. Opaque instances sort by geometry (depth testing makes order
/// irrelevant) for maximal instancing; translucent ones sort back-to-front
/// and batch only adjacent runs.
fn prepare_vector_buffers(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut cache: ResMut<GeometryCache>,
    mut extracted: ResMut<ExtractedShapes>,
    mut buffers: ResMut<VectorBuffers>,
) {
    if cache.dirty && !cache.vertices.is_empty() {
        buffers.vertex = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pf_vector_vertices"),
            contents: bytemuck::cast_slice(&cache.vertices),
            usage: BufferUsages::VERTEX,
        }));
        buffers.index = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pf_vector_indices"),
            contents: bytemuck::cast_slice(&cache.indices),
            usage: BufferUsages::INDEX,
        }));
        cache.dirty = false;
    }

    // Opaque first (grouped by geometry, front-to-back within a group),
    // then translucent back-to-front.
    extracted.0.sort_unstable_by(|a, b| {
        b.opaque
            .cmp(&a.opaque)
            .then_with(|| {
                if a.opaque {
                    a.geometry.cmp(&b.geometry).then(b.z.total_cmp(&a.z))
                } else {
                    a.z.total_cmp(&b.z)
                }
            })
    });

    buffers.opaque_batches.clear();
    buffers.blend_batches.clear();
    let mut instances: Vec<GpuInstance> = Vec::with_capacity(extracted.0.len());
    let mut last_key: Option<(u64, bool)> = None;
    for instance in &extracted.0 {
        let Some(Some(range)) = cache.ranges.get(&instance.geometry) else {
            continue;
        };
        let index = instances.len() as u32;
        instances.push(GpuInstance {
            linear: instance.linear,
            translation_z: [
                instance.translation[0],
                instance.translation[1],
                instance.z,
                0.0,
            ],
            color: instance.color,
        });
        let contiguous = last_key == Some((instance.geometry, instance.opaque));
        let batches = if instance.opaque {
            &mut buffers.opaque_batches
        } else {
            &mut buffers.blend_batches
        };
        push_batch(batches, *range, index, contiguous);
        last_key = Some((instance.geometry, instance.opaque));
    }

    if instances.is_empty() {
        buffers.opaque_batches.clear();
        buffers.blend_batches.clear();
        return;
    }
    let bytes: &[u8] = bytemuck::cast_slice(&instances);
    if buffers.instance.is_none() || buffers.instance_capacity < bytes.len() {
        buffers.instance = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pf_vector_instances"),
            contents: bytes,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        }));
        buffers.instance_capacity = bytes.len();
    } else if let Some(buffer) = &buffers.instance {
        queue.write_buffer(buffer, 0, bytes);
    }
}

fn prepare_view_bind_groups(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline: Res<VectorPipeline>,
    views: Query<(Entity, &ExtractedView)>,
    mut bind_groups: ResMut<VectorViewBindGroups>,
) {
    for (entity, view) in &views {
        let clip_from_world = view.clip_from_world.unwrap_or_else(|| {
            view.clip_from_view * view.world_from_view.to_matrix().inverse()
        });
        let entry = bind_groups.per_view.entry(entity).or_default();
        entry.uniform.set(clip_from_world);
        entry.uniform.write_buffer(&device, &queue);
        if entry.bind_group.is_none() {
            let layout = device
                .create_bind_group_layout("pf_vector_view_layout", &pipeline.view_layout.entries);
            entry.bind_group = Some(device.create_bind_group(
                "pf_vector_view_bind_group",
                &layout,
                &BindGroupEntries::single(entry.uniform.binding().unwrap()),
            ));
        }
    }
}

/// The vector pass. Runs in the `Core2d` schedule after the main pass:
/// opaque batches first (depth write, no blend), then translucent
/// back-to-front (depth read-only, alpha blend).
fn vector_pass(
    pipeline: Res<VectorPipeline>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Res<VectorBuffers>,
    bind_groups: Res<VectorViewBindGroups>,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewTarget,
        &ViewDepthTexture,
        &Msaa,
    )>,
    mut ctx: RenderContext,
) {
    if buffers.opaque_batches.is_empty() && buffers.blend_batches.is_empty() {
        return;
    }
    let view_entity = view.entity();
    let (camera, extracted_view, target, depth, msaa) = view.into_inner();

    let get_pipeline = |opaque| {
        pipeline
            .get(extracted_view.target_format, msaa.samples(), opaque)
            .and_then(|id| pipeline_cache.get_render_pipeline(id))
    };
    let (Some(opaque_pipeline), Some(blend_pipeline)) = (get_pipeline(true), get_pipeline(false))
    else {
        // Still compiling; skip the frame rather than stall.
        return;
    };
    let (Some(vertex), Some(index), Some(instance)) =
        (&buffers.vertex, &buffers.index, &buffers.instance)
    else {
        return;
    };
    let Some(bind_group) = bind_groups
        .per_view
        .get(&view_entity)
        .and_then(|entry| entry.bind_group.as_ref())
    else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("pf_vector_pass"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut pass, "vector_pass");

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, vertex.slice(..));
    pass.set_vertex_buffer(1, instance.slice(..));
    pass.set_index_buffer(index.slice(..), IndexFormat::Uint32);

    if !buffers.opaque_batches.is_empty() {
        pass.set_render_pipeline(opaque_pipeline);
        for batch in &buffers.opaque_batches {
            pass.draw_indexed(batch.indices.clone(), batch.base_vertex, batch.instances.clone());
        }
    }
    if !buffers.blend_batches.is_empty() {
        pass.set_render_pipeline(blend_pipeline);
        for batch in &buffers.blend_batches {
            pass.draw_indexed(batch.indices.clone(), batch.base_vertex, batch.instances.clone());
        }
    }

    pass_span.end(&mut pass);
}
