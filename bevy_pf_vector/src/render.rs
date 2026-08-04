//! Render-world half of the engine: extract, tessellate-on-first-sight,
//! persistent buffers, pipelines, and the vector pass itself.
//!
//! Performance model (portable across NVIDIA/AMD/Intel/Apple via wgpu — no
//! vendor extensions, optional features detected at runtime):
//! - Geometry is tessellated once per unique content hash and lives in
//!   persistent vertex/index buffers. Steady-state per-frame CPU work is
//!   hashing + one 36-byte instance write per shape.
//! - Exact-coverage triangles mean near-zero overdraw, unlike SDF quad
//!   renderers that shade full bounding rects.
//! - Edges antialias analytically: a one-screen-pixel fringe extruded in
//!   the vertex shader (see vector.wgsl), so the engine renders
//!   single-sample — no 4x MSAA bandwidth tax, which especially matters on
//!   tile-based GPUs.
//! - Opaque interiors (alpha == 1, the HUD common case) draw with depth
//!   write + test and no blending: early-z rejects hidden fragments, and
//!   draw order becomes irrelevant, so opaque batches group purely by
//!   geometry for maximal instancing. Translucent interiors and all fringes
//!   blend back-to-front, depth read-only.
//! - Where the device offers MULTI_DRAW_INDIRECT + INDIRECT_FIRST_INSTANCE
//!   (desktop Vulkan/DX12), each phase submits as ONE
//!   multi_draw_indexed_indirect; elsewhere a plain draw loop.

use std::ops::Range;

use bevy::platform::collections::HashMap;

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
    BlendState, Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages,
    CachedRenderPipelineId,
    ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState,
    FragmentState, IndexFormat, MultisampleState, PipelineCache, PrimitiveState,
    RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, ShaderType, StencilState,
    StoreOp, TextureFormat, UniformBuffer, VertexFormat, VertexState, VertexStepMode,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::settings::WgpuFeatures;
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

/// 20 bytes: position, outward silhouette normal (zero for interior
/// vertices), coverage.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVertex {
    position: [f32; 2],
    normal: [f32; 2],
    coverage: f32,
}

/// 36 bytes. Columns of the 2x2 linear part, translation + z, RGBA8 color.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstance {
    linear: [f32; 4],
    translation_z: [f32; 4],
    color: [u8; 4],
}

/// Layout-compatible with wgpu's DrawIndexedIndirectArgs.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[derive(ShaderType, Clone, Copy)]
struct VectorViewUniform {
    clip_from_world: Mat4,
    viewport: Vec4,
}

fn vertex_buffer_layout() -> VertexBufferLayout {
    VertexBufferLayout::from_vertex_formats(
        VertexStepMode::Vertex,
        [
            VertexFormat::Float32x2,
            VertexFormat::Float32x2,
            VertexFormat::Float32,
        ],
    )
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
    // follow the three vertex-buffer locations.
    for (i, attribute) in layout.attributes.iter_mut().enumerate() {
        attribute.shader_location = 3 + i as u32;
    }
    layout
}

// ---------------------------------------------------------------- resources

#[derive(Clone, Copy)]
struct GeometryRange {
    base_vertex: i32,
    interior_first: u32,
    interior_count: u32,
    fringe_first: u32,
    fringe_count: u32,
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
            self.vertices.extend(geometry.vertices.iter().map(|v| GpuVertex {
                position: v.position,
                normal: v.normal,
                coverage: v.coverage,
            }));
            let interior_first = self.indices.len() as u32;
            self.indices.extend_from_slice(&geometry.interior_indices);
            let fringe_first = self.indices.len() as u32;
            self.indices.extend_from_slice(&geometry.fringe_indices);
            self.dirty = true;
            GeometryRange {
                base_vertex,
                interior_first,
                interior_count: geometry.interior_indices.len() as u32,
                fringe_first,
                fringe_count: geometry.fringe_indices.len() as u32,
            }
        });
        self.ranges.insert(key, range);
    }
}

/// Where an instance's triangles live: the persistent tessellate-once cache,
/// or this frame's transient region (shapes whose path changed this frame).
#[derive(Clone, Copy)]
enum GeometryRef {
    Cached(u64),
    Dynamic(GeometryRange),
}

struct ExtractedInstance {
    geometry: GeometryRef,
    z: f32,
    linear: [f32; 4],
    translation: [f32; 2],
    color: [u8; 4],
    opaque: bool,
}

#[derive(Resource, Default)]
pub struct ExtractedShapes {
    items: Vec<ExtractedInstance>,
    /// Transient mesh for this frame's changed shapes, appended after the
    /// static cache in the shared vertex/index buffers.
    dynamic_vertices: Vec<GpuVertex>,
    dynamic_indices: Vec<u32>,
}

impl ExtractedShapes {
    fn append_dynamic(&mut self, geometry: tess::TessellatedGeometry) -> GeometryRange {
        let base_vertex = self.dynamic_vertices.len() as i32;
        self.dynamic_vertices.extend(geometry.vertices.iter().map(|v| GpuVertex {
            position: v.position,
            normal: v.normal,
            coverage: v.coverage,
        }));
        let interior_first = self.dynamic_indices.len() as u32;
        self.dynamic_indices.extend_from_slice(&geometry.interior_indices);
        let fringe_first = self.dynamic_indices.len() as u32;
        self.dynamic_indices.extend_from_slice(&geometry.fringe_indices);
        GeometryRange {
            base_vertex,
            interior_first,
            interior_count: geometry.interior_indices.len() as u32,
            fringe_first,
            fringe_count: geometry.fringe_indices.len() as u32,
        }
    }
}

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
    indirect: Option<Buffer>,
    indirect_capacity: usize,
    /// Element counts of the static (cached) region at last upload; the
    /// dynamic region starts here and is rewritten per frame.
    static_vertex_count: usize,
    static_index_count: usize,
    /// Allocated element capacities of the shared vertex/index buffers.
    vertex_capacity: usize,
    index_capacity: usize,
    /// Batch counts per phase when multi-draw is active; batch lists always.
    opaque_batches: Vec<VectorBatch>,
    blend_batches: Vec<VectorBatch>,
    use_multi_draw: bool,
    /// Hash of the layout-affecting fields (count, geometry, z, opacity) of
    /// the extracted set. While it holds steady — the common HUD case, where
    /// only transforms/colors animate — sorting, batching, and indirect-args
    /// building are all skipped and instances upload through `permutation`.
    /// Dynamic geometry refs hash their per-frame ranges, so any dynamic
    /// content naturally forces the rebuild path.
    layout_fingerprint: u64,
    /// Extracted-item index for each instance slot, in draw order.
    permutation: Vec<u32>,
}

#[derive(Default)]
struct ViewEntry {
    uniform: UniformBuffer<VectorViewUniform>,
    bind_group: Option<BindGroup>,
}

impl Default for VectorViewUniform {
    fn default() -> Self {
        Self { clip_from_world: Mat4::IDENTITY, viewport: Vec4::ONE }
    }
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
                    uniform_buffer::<VectorViewUniform>(false),
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

/// Copies shape instances into the render world. Unchanged shapes hit the
/// tessellate-once cache (steady-state cost: hashing plus a 36-byte push).
/// Shapes whose `VectorShape` mutated this frame tessellate into the
/// transient region instead — dynamic topology works, priced per changed
/// shape, without poisoning the cache.
fn extract_shapes(
    shapes: Extract<Query<(Ref<VectorShape>, &GlobalTransform)>>,
    mut cache: ResMut<GeometryCache>,
    mut extracted: ResMut<ExtractedShapes>,
) {
    extracted.items.clear();
    extracted.dynamic_vertices.clear();
    extracted.dynamic_indices.clear();

    // Epoch flush: mutation churn appends new cache entries; when the cache
    // gets absurd, drop it wholesale and let live shapes re-populate.
    if cache.ranges.len() > 8192 {
        *cache = GeometryCache::default();
    }

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
        // `is_changed` is true on the frame a path/style mutates (and on
        // spawn); those shapes skip the cache entirely this frame.
        let dynamic = shape.is_changed();
        if let Some(color) = shape.style.fill {
            let geometry = if dynamic {
                tess::tessellate_fill(&shape.commands)
                    .map(|g| GeometryRef::Dynamic(extracted.append_dynamic(g)))
            } else {
                let key = tess::fill_key(&shape.commands);
                cache.ensure(key, || tess::tessellate_fill(&shape.commands));
                Some(GeometryRef::Cached(key))
            };
            if let Some(geometry) = geometry {
                extracted.items.push(ExtractedInstance {
                    geometry,
                    z,
                    linear,
                    translation,
                    color: pack_color(color),
                    opaque: color.alpha >= 1.0,
                });
            }
        }
        if let Some(stroke) = shape.style.stroke {
            let geometry = if dynamic {
                tess::tessellate_stroke(&shape.commands, &stroke)
                    .map(|g| GeometryRef::Dynamic(extracted.append_dynamic(g)))
            } else {
                let key = tess::stroke_key(&shape.commands, &stroke);
                cache.ensure(key, || tess::tessellate_stroke(&shape.commands, &stroke));
                Some(GeometryRef::Cached(key))
            };
            if let Some(geometry) = geometry {
                extracted.items.push(ExtractedInstance {
                    geometry,
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

/// Uploads geometry when it changed, then builds the per-frame instance and
/// batch lists:
/// - opaque interiors, grouped by geometry (depth testing makes order
///   irrelevant) for maximal instancing;
/// - blend items back-to-front: translucent interiors and every silhouette
///   fringe, batching adjacent runs of the same (geometry, part).
/// When the device supports it, batches are also encoded into an indirect
/// args buffer so the pass is two multi-draw calls.
fn prepare_vector_buffers(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut cache: ResMut<GeometryCache>,
    extracted: Res<ExtractedShapes>,
    mut buffers: ResMut<VectorBuffers>,
) {
    // Shared vertex/index buffers: static (cached) region plus a per-frame
    // dynamic tail. Recreate when the static region changes or capacity is
    // exceeded; otherwise only the tail is rewritten each frame.
    let dyn_vertex_count = extracted.dynamic_vertices.len();
    let dyn_index_count = extracted.dynamic_indices.len();
    let need_vertices = cache.vertices.len() + dyn_vertex_count;
    let need_indices = cache.indices.len() + dyn_index_count;
    let recreate = need_vertices > 0
        && (cache.dirty
            || buffers.vertex.is_none()
            || buffers.index.is_none()
            || buffers.vertex_capacity < need_vertices
            || buffers.index_capacity < need_indices
            || buffers.static_vertex_count != cache.vertices.len()
            || buffers.static_index_count != cache.indices.len());
    if recreate {
        let vertex_capacity = need_vertices + dyn_vertex_count.max(256);
        let index_capacity = need_indices + dyn_index_count.max(1024);
        let vertex = device.create_buffer(&BufferDescriptor {
            label: Some("pf_vector_vertices"),
            size: (vertex_capacity * size_of::<GpuVertex>()) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index = device.create_buffer(&BufferDescriptor {
            label: Some("pf_vector_indices"),
            size: (index_capacity * size_of::<u32>()) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !cache.vertices.is_empty() {
            queue.write_buffer(&vertex, 0, bytemuck::cast_slice(&cache.vertices));
            queue.write_buffer(&index, 0, bytemuck::cast_slice(&cache.indices));
        }
        buffers.vertex = Some(vertex);
        buffers.index = Some(index);
        buffers.vertex_capacity = vertex_capacity;
        buffers.index_capacity = index_capacity;
        buffers.static_vertex_count = cache.vertices.len();
        buffers.static_index_count = cache.indices.len();
        cache.dirty = false;
    }
    if dyn_vertex_count > 0 {
        if let (Some(vertex), Some(index)) = (&buffers.vertex, &buffers.index) {
            queue.write_buffer(
                vertex,
                (buffers.static_vertex_count * size_of::<GpuVertex>()) as u64,
                bytemuck::cast_slice(&extracted.dynamic_vertices),
            );
            queue.write_buffer(
                index,
                (buffers.static_index_count * size_of::<u32>()) as u64,
                bytemuck::cast_slice(&extracted.dynamic_indices),
            );
        }
    }

    // Resolve a geometry ref to buffer-space ranges (dynamic ranges offset
    // past the static region) and derive a grouping key: cached geometry
    // batches across instances, dynamic geometry is unique per item.
    let static_vertex_count = buffers.static_vertex_count as i32;
    let static_index_count = buffers.static_index_count as u32;
    let resolve = |geometry: &GeometryRef| -> Option<GeometryRange> {
        match geometry {
            GeometryRef::Cached(key) => cache.ranges.get(key).copied().flatten(),
            GeometryRef::Dynamic(range) => Some(GeometryRange {
                base_vertex: range.base_vertex + static_vertex_count,
                interior_first: range.interior_first + static_index_count,
                interior_count: range.interior_count,
                fringe_first: range.fringe_first + static_index_count,
                fringe_count: range.fringe_count,
            }),
        }
    };
    let group_key = |geometry: &GeometryRef| -> (u8, u64) {
        match geometry {
            GeometryRef::Cached(key) => (0, *key),
            GeometryRef::Dynamic(range) => (
                1,
                (u64::from(range.interior_first) << 32) | u64::from(range.base_vertex as u32),
            ),
        }
    };

    // Layout fingerprint: when the structure is unchanged, the cached batch
    // lists, indirect args, and draw-order permutation all remain valid, and
    // per-frame CPU collapses to a gather + one buffer write. Dynamic shapes
    // with frame-stable tessellation sizes keep identical ranges, so a purely
    // parameter-animated HUD stays on this path too.
    let fingerprint = {
        use std::hash::{Hash, Hasher};
        let mut hasher = tess::fast_hasher();
        extracted.items.len().hash(&mut hasher);
        for item in &extracted.items {
            group_key(&item.geometry).hash(&mut hasher);
            item.z.to_bits().hash(&mut hasher);
            item.opaque.hash(&mut hasher);
        }
        hasher.finish()
    };

    let gpu_instance = |item: &ExtractedInstance| GpuInstance {
        linear: item.linear,
        translation_z: [item.translation[0], item.translation[1], item.z, 0.0],
        color: item.color,
    };

    if fingerprint == buffers.layout_fingerprint && !buffers.permutation.is_empty() {
        // Fast path: transforms/colors (and stable-size dynamic geometry)
        // changed at most — gather in cached draw order and upload.
        let instances: Vec<GpuInstance> = buffers
            .permutation
            .iter()
            .map(|&i| gpu_instance(&extracted.items[i as usize]))
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        if let Some(buffer) = &buffers.instance {
            if buffers.instance_capacity >= bytes.len() {
                queue.write_buffer(buffer, 0, bytes);
                return;
            }
        }
        // Capacity lost somehow — fall through to a full rebuild.
    }
    buffers.layout_fingerprint = fingerprint;
    buffers.permutation.clear();
    buffers.opaque_batches.clear();
    buffers.blend_batches.clear();

    let mut instances: Vec<GpuInstance> = Vec::with_capacity(extracted.items.len() * 2);
    let mut push_instance = |instances: &mut Vec<GpuInstance>,
                             permutation: &mut Vec<u32>,
                             item_index: usize,
                             item: &ExtractedInstance|
     -> u32 {
        let index = instances.len() as u32;
        instances.push(gpu_instance(item));
        permutation.push(item_index as u32);
        index
    };

    // Section 1: opaque interiors, geometry-grouped, front-to-back in group.
    let mut opaque_order: Vec<usize> = (0..extracted.items.len())
        .filter(|&i| extracted.items[i].opaque)
        .collect();
    opaque_order.sort_unstable_by(|&a, &b| {
        let (ia, ib) = (&extracted.items[a], &extracted.items[b]);
        group_key(&ia.geometry)
            .cmp(&group_key(&ib.geometry))
            .then(ib.z.total_cmp(&ia.z))
    });
    let mut last_geometry: Option<(u8, u64)> = None;
    for &item_index in &opaque_order {
        let item = &extracted.items[item_index];
        let Some(range) = resolve(&item.geometry) else {
            continue;
        };
        if range.interior_count == 0 {
            continue;
        }
        let index = push_instance(&mut instances, &mut buffers.permutation, item_index, item);
        let key = group_key(&item.geometry);
        if last_geometry == Some(key) {
            buffers.opaque_batches.last_mut().unwrap().instances.end = index + 1;
        } else {
            buffers.opaque_batches.push(VectorBatch {
                indices: range.interior_first..range.interior_first + range.interior_count,
                base_vertex: range.base_vertex,
                instances: index..index + 1,
            });
            last_geometry = Some(key);
        }
    }

    // Section 2: blend items, back-to-front. Each item is (source instance,
    // part); a translucent shape contributes interior + fringe, an opaque
    // one only its fringe.
    let mut blend_order: Vec<(usize, bool)> = Vec::new();
    let mut z_sorted: Vec<usize> = (0..extracted.items.len()).collect();
    z_sorted.sort_by(|&a, &b| extracted.items[a].z.total_cmp(&extracted.items[b].z));
    for &item_index in &z_sorted {
        if !extracted.items[item_index].opaque {
            blend_order.push((item_index, false));
        }
        blend_order.push((item_index, true));
    }
    let mut last_key: Option<((u8, u64), bool)> = None;
    for &(item_index, is_fringe) in &blend_order {
        let item = &extracted.items[item_index];
        let Some(range) = resolve(&item.geometry) else {
            continue;
        };
        let (first, count) = if is_fringe {
            (range.fringe_first, range.fringe_count)
        } else {
            (range.interior_first, range.interior_count)
        };
        if count == 0 {
            continue;
        }
        let index = push_instance(&mut instances, &mut buffers.permutation, item_index, item);
        let key = (group_key(&item.geometry), is_fringe);
        if last_key == Some(key) {
            buffers.blend_batches.last_mut().unwrap().instances.end = index + 1;
        } else {
            buffers.blend_batches.push(VectorBatch {
                indices: first..first + count,
                base_vertex: range.base_vertex,
                instances: index..index + 1,
            });
            last_key = Some(key);
        }
    }

    if instances.is_empty() {
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

    // Indirect args: one multi-draw per phase where the device allows it.
    // wgpu 29 exposes multi_draw_indexed_indirect unconditionally (native
    // single-call on Vulkan/DX12, emulated per-arg elsewhere); only
    // first_instance-in-indirect-args needs a feature gate.
    buffers.use_multi_draw = device
        .features()
        .contains(WgpuFeatures::INDIRECT_FIRST_INSTANCE);
    if buffers.use_multi_draw {
        let args: Vec<GpuDrawIndexedIndirect> = buffers
            .opaque_batches
            .iter()
            .chain(buffers.blend_batches.iter())
            .map(|batch| GpuDrawIndexedIndirect {
                index_count: batch.indices.end - batch.indices.start,
                instance_count: batch.instances.end - batch.instances.start,
                first_index: batch.indices.start,
                base_vertex: batch.base_vertex,
                first_instance: batch.instances.start,
            })
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&args);
        if buffers.indirect.is_none() || buffers.indirect_capacity < bytes.len() {
            buffers.indirect = Some(device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("pf_vector_indirect"),
                contents: bytes,
                usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            }));
            buffers.indirect_capacity = bytes.len();
        } else if let Some(buffer) = &buffers.indirect {
            queue.write_buffer(buffer, 0, bytes);
        }
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
        let viewport = view.viewport.as_vec4();
        let entry = bind_groups.per_view.entry(entity).or_default();
        entry.uniform.set(VectorViewUniform { clip_from_world, viewport });
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
/// opaque interiors first (depth write, no blend, early-z), then blend items
/// back-to-front (translucent interiors + AA fringes, depth read-only).
/// Two multi-draw calls total when the device supports indirect.
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

    let indirect = buffers.use_multi_draw.then(|| buffers.indirect.as_ref()).flatten();
    let args_size = size_of::<GpuDrawIndexedIndirect>() as u64;

    if !buffers.opaque_batches.is_empty() {
        pass.set_render_pipeline(opaque_pipeline);
        match indirect {
            Some(indirect) => pass.multi_draw_indexed_indirect(
                indirect,
                0,
                buffers.opaque_batches.len() as u32,
            ),
            None => {
                for batch in &buffers.opaque_batches {
                    pass.draw_indexed(
                        batch.indices.clone(),
                        batch.base_vertex,
                        batch.instances.clone(),
                    );
                }
            }
        }
    }
    if !buffers.blend_batches.is_empty() {
        pass.set_render_pipeline(blend_pipeline);
        match indirect {
            Some(indirect) => pass.multi_draw_indexed_indirect(
                indirect,
                buffers.opaque_batches.len() as u64 * args_size,
                buffers.blend_batches.len() as u32,
            ),
            None => {
                for batch in &buffers.blend_batches {
                    pass.draw_indexed(
                        batch.indices.clone(),
                        batch.base_vertex,
                        batch.instances.clone(),
                    );
                }
            }
        }
    }

    pass_span.end(&mut pass);
}
