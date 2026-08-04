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
use crate::{ClippedBy, HudTransform, VectorClipShape, VectorPrimitive, VectorShape};

pub const VECTOR_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("7a3f1c2e-9b4d-4b8a-a2f0-5e1d3c6b9f01");
pub const VECTOR_PARAM_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("2c8e5b1a-4f7d-4c3e-9a06-8b21d75c4e90");

pub struct VectorRenderPlugin;

impl Plugin for VectorRenderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, VECTOR_SHADER_HANDLE, "vector.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            VECTOR_PARAM_SHADER_HANDLE,
            "vector_param.wgsl",
            Shader::from_wgsl
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<GeometryCache>()
            .init_resource::<ExtractedShapes>()
            .init_resource::<ExtractedParametrics>()
            .init_resource::<ExtractedClips>()
            .init_resource::<GradientAtlas>()
            .init_resource::<VectorBuffers>()
            .init_resource::<VectorViewBindGroups>()
            .init_resource::<VectorPipeline>()
            .add_systems(
                ExtractSchedule,
                (extract_clips, extract_shapes, extract_primitives).chain(),
            )
            .add_systems(
                Render,
                (
                    queue_vector_pipelines.in_set(RenderSystems::Queue),
                    prepare_vector_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_parametrics.in_set(RenderSystems::PrepareResources),
                    prepare_clips.in_set(RenderSystems::PrepareResources),
                    prepare_gradients.in_set(RenderSystems::PrepareResources),
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

/// 56 bytes: 2x2 linear part, translation + z + clip pack, RGBA8 tint,
/// brush params (gradient geometry in local space), packed brush meta
/// (atlas_row * 4 + kind; kind 0 solid / 1 linear / 2 radial).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstance {
    linear: [f32; 4],
    translation_z: [f32; 4],
    color: [u8; 4],
    brush_params: [f32; 4],
    brush_meta: f32,
}

/// One analytic clip entry: inverse world transform of the clip entity plus
/// SDF parameters. A circle is a rounded rect with zero half-extents.
/// 48 bytes, 16-aligned for storage-buffer array stride.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuClip {
    inv_linear: [f32; 4],
    inv_translation: [f32; 2],
    half_extents: [f32; 2],
    radius: f32,
    _pad: [f32; 3],
}

/// Fixed clip-buffer capacity: created once so view bind groups stay stable.
const MAX_CLIPS: usize = 1024;

/// Packs a clip chain reference into one f32 (rides in the instance's spare
/// lane): index * 8 + count, exact in f32 for index < 2^20.
fn pack_clip(start: u32, count: u32) -> f32 {
    (start * 8 + count.min(7)) as f32
}

/// Per-frame clip entries and resolved chains, keyed by chain head entity.
#[derive(Resource, Default)]
pub struct ExtractedClips {
    entries: Vec<GpuClip>,
    chains: HashMap<Entity, (u32, u32)>,
}

/// Gradient lookup atlas: each unique stop list bakes once into a 256-texel
/// sRGB row; instances reference rows by index. 256 rows; epoch-flushed if
/// content churn ever fills it.
pub const GRADIENT_ATLAS_SIZE: u32 = 256;
/// Number of gradient rows in the atlas.
pub const GRADIENT_ATLAS_ROWS: u32 = 1024;

#[derive(Resource, Default)]
pub struct GradientAtlas {
    rows: HashMap<u64, u32>,
    next_row: u32,
    /// Baked rows not yet uploaded: (row, 256 RGBA8 texels).
    pending: Vec<(u32, Vec<u8>)>,
}

impl GradientAtlas {
    /// Returns the atlas row for a stop list, baking it on first sight.
    fn ensure_row(&mut self, stops: &[crate::path::GradientStop]) -> Option<u32> {
        use std::hash::{Hash, Hasher};
        let mut hasher = tess::fast_hasher();
        for stop in stops {
            stop.offset.to_bits().hash(&mut hasher);
            stop.color.red.to_bits().hash(&mut hasher);
            stop.color.green.to_bits().hash(&mut hasher);
            stop.color.blue.to_bits().hash(&mut hasher);
            stop.color.alpha.to_bits().hash(&mut hasher);
        }
        let key = hasher.finish();
        if let Some(&row) = self.rows.get(&key) {
            return Some(row);
        }
        if self.next_row >= GRADIENT_ATLAS_ROWS {
            // Full: degrade to solid rather than thrash. Real content reuses
            // gradients; 1024 unique simultaneous gradients is the budget.
            return None;
        }
        let row = self.next_row;
        self.next_row += 1;
        self.rows.insert(key, row);

        let mut sorted: Vec<_> = stops.to_vec();
        sorted.sort_by(|a, b| a.offset.total_cmp(&b.offset));
        let mut texels = Vec::with_capacity(GRADIENT_ATLAS_SIZE as usize * 4);
        for i in 0..GRADIENT_ATLAS_SIZE {
            let t = i as f32 / (GRADIENT_ATLAS_SIZE - 1) as f32;
            let color = sample_stops(&sorted, t);
            let srgba: Srgba = color.into();
            texels.push((srgba.red.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            texels.push((srgba.green.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            texels.push((srgba.blue.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            texels.push((srgba.alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        self.pending.push((row, texels));
        Some(row)
    }
}

fn sample_stops(sorted: &[crate::path::GradientStop], t: f32) -> LinearRgba {
    match sorted {
        [] => LinearRgba::WHITE,
        [only] => only.color,
        _ => {
            if t <= sorted[0].offset {
                return sorted[0].color;
            }
            for pair in sorted.windows(2) {
                if t <= pair[1].offset {
                    let span = (pair[1].offset - pair[0].offset).max(1.0e-6);
                    let k = (t - pair[0].offset) / span;
                    return LinearRgba {
                        red: pair[0].color.red + (pair[1].color.red - pair[0].color.red) * k,
                        green: pair[0].color.green
                            + (pair[1].color.green - pair[0].color.green) * k,
                        blue: pair[0].color.blue + (pair[1].color.blue - pair[0].color.blue) * k,
                        alpha: pair[0].color.alpha
                            + (pair[1].color.alpha - pair[0].color.alpha) * k,
                    };
                }
            }
            sorted.last().unwrap().color
        }
    }
}

/// Resolves a brush into (tint, gradient params, packed meta) instance data.
fn resolve_brush(brush: &crate::path::Brush, atlas: &mut GradientAtlas) -> ([u8; 4], [f32; 4], f32) {
    use crate::path::Brush;
    match brush {
        Brush::Solid(color) => (pack_color(*color), [0.0; 4], 0.0),
        Brush::Linear { start, end, stops } => match atlas.ensure_row(stops) {
            Some(row) => (
                [255; 4],
                [start.x, start.y, end.x, end.y],
                (row * 4 + 1) as f32,
            ),
            None => (
                pack_color(stops.first().map(|s| s.color).unwrap_or(LinearRgba::WHITE)),
                [0.0; 4],
                0.0,
            ),
        },
        Brush::Radial { center, radius, stops } => match atlas.ensure_row(stops) {
            Some(row) => (
                [255; 4],
                [center.x, center.y, *radius, 0.0],
                (row * 4 + 2) as f32,
            ),
            None => (
                pack_color(stops.first().map(|s| s.color).unwrap_or(LinearRgba::WHITE)),
                [0.0; 4],
                0.0,
            ),
        },
    }
}

/// 52 bytes: the tessellated-instance fields plus the primitive parameters
/// ([start, sweep, inner, outer] for arcs).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParamInstance {
    linear: [f32; 4],
    translation_z: [f32; 4],
    color: [u8; 4],
    params: [f32; 4],
}

fn param_instance_buffer_layout() -> VertexBufferLayout {
    let mut layout = VertexBufferLayout::from_vertex_formats(
        VertexStepMode::Instance,
        [
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Unorm8x4,
            VertexFormat::Float32x4,
        ],
    );
    for (i, attribute) in layout.attributes.iter_mut().enumerate() {
        attribute.shader_location = 3 + i as u32;
    }
    layout
}

/// Canonical arc strip: vertices encode (t, side) + fringe directions; the
/// vertex shader turns them into ring-segment geometry per instance.
/// Returns (vertices, indices, interior_index_count) with fringe indices
/// following the interior ones.
fn canonical_arc_mesh() -> (Vec<GpuVertex>, Vec<u32>) {
    const SEGMENTS: u32 = 64;
    let n = SEGMENTS;
    let mut vertices = Vec::with_capacity((4 * (n + 1) + 4) as usize);
    // Interior: inner/outer pair per step, full coverage, no fringe.
    for i in 0..=n {
        let t = i as f32 / n as f32;
        for side in [0.0f32, 1.0] {
            vertices.push(GpuVertex { position: [t, side], normal: [0.0, 0.0], coverage: 1.0 });
        }
    }
    // Fringe rings: outer edge displaces +radial, inner edge -radial.
    let outer_ring = vertices.len() as u32;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        vertices.push(GpuVertex { position: [t, 1.0], normal: [1.0, 0.0], coverage: 0.0 });
    }
    let inner_ring = vertices.len() as u32;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        vertices.push(GpuVertex { position: [t, 0.0], normal: [-1.0, 0.0], coverage: 0.0 });
    }
    // End caps displace tangentially (backward at t=0, forward at t=1).
    let caps = vertices.len() as u32;
    vertices.push(GpuVertex { position: [0.0, 0.0], normal: [0.0, -1.0], coverage: 0.0 });
    vertices.push(GpuVertex { position: [0.0, 1.0], normal: [0.0, -1.0], coverage: 0.0 });
    vertices.push(GpuVertex { position: [1.0, 0.0], normal: [0.0, 1.0], coverage: 0.0 });
    vertices.push(GpuVertex { position: [1.0, 1.0], normal: [0.0, 1.0], coverage: 0.0 });

    let (inner_at, outer_at) = (|i: u32| i * 2, |i: u32| i * 2 + 1);
    let mut indices = Vec::new();
    for i in 0..n {
        indices.extend_from_slice(&[
            inner_at(i),
            outer_at(i),
            inner_at(i + 1),
            inner_at(i + 1),
            outer_at(i),
            outer_at(i + 1),
        ]);
    }
    // Fringe indices follow the interior block.
    for i in 0..n {
        indices.extend_from_slice(&[
            outer_at(i),
            outer_ring + i,
            outer_at(i + 1),
            outer_at(i + 1),
            outer_ring + i,
            outer_ring + i + 1,
        ]);
        indices.extend_from_slice(&[
            inner_ring + i,
            inner_at(i),
            inner_ring + i + 1,
            inner_ring + i + 1,
            inner_at(i),
            inner_at(i + 1),
        ]);
    }
    indices.extend_from_slice(&[
        inner_at(0), caps, outer_at(0), outer_at(0), caps, caps + 1,
    ]);
    indices.extend_from_slice(&[
        inner_at(n), caps + 2, outer_at(n), outer_at(n), caps + 2, caps + 3,
    ]);
    (vertices, indices)
}

/// Interior index count of the canonical arc mesh (fringe indices follow).
const ARC_INTERIOR_INDEX_COUNT: u32 = 64 * 6;

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
            VertexFormat::Float32x4,
            VertexFormat::Float32,
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
    brush_params: [f32; 4],
    brush_meta: f32,
    opaque: bool,
    clip: f32,
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

struct ParamItem {
    z: f32,
    linear: [f32; 4],
    translation: [f32; 2],
    color: [u8; 4],
    params: [f32; 4],
    opaque: bool,
    clip: f32,
}

#[derive(Resource, Default)]
pub struct ExtractedParametrics(Vec<ParamItem>);

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
    /// Parametric primitives: canonical mesh (built once) + per-frame
    /// instances, opaque section first then translucent back-to-front.
    param_vertex: Option<Buffer>,
    param_index: Option<Buffer>,
    param_instance: Option<Buffer>,
    param_instance_capacity: usize,
    param_opaque_count: u32,
    param_total_count: u32,
    /// Fixed-capacity analytic clip storage (created once; bind groups stay
    /// stable), rewritten per frame.
    clip: Option<Buffer>,
    /// Gradient LUT atlas (created once), plus its view and sampler.
    gradient_texture: Option<bevy::render::render_resource::Texture>,
    gradient_view: Option<bevy::render::render_resource::TextureView>,
    gradient_sampler: Option<bevy::render::render_resource::Sampler>,
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
    param_variants: HashMap<(TextureFormat, u32, bool), CachedRenderPipelineId>,
}

impl Default for VectorPipeline {
    fn default() -> Self {
        Self {
            view_layout: BindGroupLayoutDescriptor::new(
                "pf_vector_view_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::VERTEX_FRAGMENT,
                    (
                        (0, uniform_buffer::<VectorViewUniform>(false)),
                        (1, bevy::render::render_resource::binding_types::storage_buffer_read_only_sized(false, None)),
                        (2, bevy::render::render_resource::binding_types::texture_2d(bevy::render::render_resource::TextureSampleType::Float { filterable: true })),
                        (3, bevy::render::render_resource::binding_types::sampler(bevy::render::render_resource::SamplerBindingType::Filtering)),
                    ),
                ),
            ),
            variants: HashMap::new(),
            param_variants: HashMap::new(),
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

    fn ensure_param(
        &mut self,
        cache: &PipelineCache,
        format: TextureFormat,
        samples: u32,
        opaque: bool,
    ) -> CachedRenderPipelineId {
        let view_layout = self.view_layout.clone();
        *self
            .param_variants
            .entry((format, samples, opaque))
            .or_insert_with(|| {
                cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some(
                        if opaque {
                            "pf_vector_param_opaque_pipeline"
                        } else {
                            "pf_vector_param_blend_pipeline"
                        }
                        .into(),
                    ),
                    layout: vec![view_layout],
                    vertex: VertexState {
                        shader: VECTOR_PARAM_SHADER_HANDLE,
                        entry_point: Some("vertex".into()),
                        shader_defs: Vec::new(),
                        buffers: vec![vertex_buffer_layout(), param_instance_buffer_layout()],
                    },
                    fragment: Some(FragmentState {
                        shader: VECTOR_PARAM_SHADER_HANDLE,
                        entry_point: Some("fragment".into()),
                        shader_defs: Vec::new(),
                        targets: vec![Some(ColorTargetState {
                            format,
                            blend: (!opaque).then_some(BlendState::ALPHA_BLENDING),
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    primitive: PrimitiveState::default(),
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

    fn get_param(
        &self,
        format: TextureFormat,
        samples: u32,
        opaque: bool,
    ) -> Option<CachedRenderPipelineId> {
        self.param_variants.get(&(format, samples, opaque)).copied()
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
    shapes: Extract<
        Query<(Ref<VectorShape>, &GlobalTransform, Option<&ClippedBy>), Without<HudTransform>>,
    >,
    flat_shapes: Extract<Query<(Ref<VectorShape>, &HudTransform, Option<&ClippedBy>)>>,
    clips: Res<ExtractedClips>,
    mut cache: ResMut<GeometryCache>,
    mut extracted: ResMut<ExtractedShapes>,
    mut atlas: ResMut<GradientAtlas>,
) {
    extracted.items.clear();
    extracted.dynamic_vertices.clear();
    extracted.dynamic_indices.clear();

    // Epoch flush: mutation churn appends new cache entries; when the cache
    // gets absurd, drop it wholesale and let live shapes re-populate.
    if cache.ranges.len() > 8192 {
        *cache = GeometryCache::default();
    }

    for (shape, transform, clipped) in &shapes {
        let model = transform.to_matrix();
        let linear = [
            model.x_axis.x,
            model.x_axis.y,
            model.y_axis.x,
            model.y_axis.y,
        ];
        let translation = [model.w_axis.x, model.w_axis.y];
        push_shape(
            &mut cache,
            &mut extracted,
            &mut atlas,
            &clips,
            &shape,
            linear,
            translation,
            model.w_axis.z,
            clipped,
        );
    }
    for (shape, hud, clipped) in &flat_shapes {
        let (linear, translation, z) = hud.decompose();
        push_shape(&mut cache, &mut extracted, &mut atlas, &clips, &shape, linear, translation, z, clipped);
    }
}

#[allow(clippy::too_many_arguments)]
fn push_shape(
    cache: &mut GeometryCache,
    extracted: &mut ExtractedShapes,
    atlas: &mut GradientAtlas,
    clips: &ExtractedClips,
    shape: &Ref<VectorShape>,
    linear: [f32; 4],
    translation: [f32; 2],
    z: f32,
    clipped: Option<&ClippedBy>,
) {
    let clip = clipped
        .and_then(|c| clips.chains.get(&c.0))
        .map(|&(start, count)| pack_clip(start, count))
        .unwrap_or(0.0);
    {
        // `is_changed` is true on the frame a path/style mutates (and on
        // spawn); those shapes skip the cache entirely this frame.
        let dynamic = shape.is_changed();
        if let Some(brush) = &shape.style.fill {
            let rule = shape.style.fill_rule;
            let geometry = if dynamic {
                tess::tessellate_fill(&shape.commands, rule)
                    .map(|g| GeometryRef::Dynamic(extracted.append_dynamic(g)))
            } else {
                let key = tess::fill_key(&shape.commands, rule);
                cache.ensure(key, || tess::tessellate_fill(&shape.commands, rule));
                Some(GeometryRef::Cached(key))
            };
            if let Some(geometry) = geometry {
                let (color, brush_params, brush_meta) = resolve_brush(brush, atlas);
                extracted.items.push(ExtractedInstance {
                    geometry,
                    z,
                    linear,
                    translation,
                    color,
                    brush_params,
                    brush_meta,
                    // Clipped instances need blending (both for the alpha
                    // knock-out and for AA clip edges).
                    opaque: brush.is_opaque() && clip == 0.0,
                    clip,
                });
            }
        }
        if let Some(stroke) = &shape.style.stroke {
            let geometry = if dynamic {
                tess::tessellate_stroke(&shape.commands, stroke)
                    .map(|g| GeometryRef::Dynamic(extracted.append_dynamic(g)))
            } else {
                let key = tess::stroke_key(&shape.commands, stroke);
                cache.ensure(key, || tess::tessellate_stroke(&shape.commands, stroke));
                Some(GeometryRef::Cached(key))
            };
            if let Some(geometry) = geometry {
                let (color, brush_params, brush_meta) = resolve_brush(&stroke.brush, atlas);
                extracted.items.push(ExtractedInstance {
                    geometry,
                    // Strokes draw over their own fill at equal z.
                    z: z + 1.0e-4,
                    linear,
                    translation,
                    color,
                    brush_params,
                    brush_meta,
                    opaque: stroke.brush.is_opaque() && clip == 0.0,
                    clip,
                });
            }
        }
    }
}

/// Shared per-item processing for both transform sources.
#[allow(clippy::too_many_arguments)]
fn push_primitive(
    extracted: &mut ExtractedParametrics,
    clips: &ExtractedClips,
    primitive: &VectorPrimitive,
    linear: [f32; 4],
    translation: [f32; 2],
    z: f32,
    clipped: Option<&ClippedBy>,
) {
    let clip = clipped
        .and_then(|c| clips.chains.get(&c.0))
        .map(|&(start, count)| pack_clip(start, count))
        .unwrap_or(0.0);
    let VectorPrimitive::Arc { inner, outer, start, sweep, color } = *primitive;
    extracted.0.push(ParamItem {
        z,
        linear,
        translation,
        color: pack_color(color),
        params: [start, sweep, inner, outer],
        opaque: color.alpha >= 1.0 && clip == 0.0,
        clip,
    });
}

fn queue_vector_pipelines(
    mut pipeline: ResMut<VectorPipeline>,
    cache: Res<PipelineCache>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    for (view, msaa) in &views {
        for opaque in [true, false] {
            pipeline.ensure(&cache, view.target_format, msaa.samples(), opaque);
            pipeline.ensure_param(&cache, view.target_format, msaa.samples(), opaque);
        }
    }
}

/// Resolves clip entities into per-frame SDF entries and chains. Runs before
/// shape/primitive extraction so instances can reference chains by index.
fn extract_clips(
    clips: Extract<
        Query<
            (Entity, &VectorClipShape, &GlobalTransform, Option<&ClippedBy>),
            Without<HudTransform>,
        >,
    >,
    flat_clips: Extract<Query<(Entity, &VectorClipShape, &HudTransform, Option<&ClippedBy>)>>,
    mut extracted: ResMut<ExtractedClips>,
) {
    extracted.entries.clear();
    extracted.chains.clear();

    struct ClipNode {
        entry: GpuClip,
        parent: Option<Entity>,
    }
    fn clip_node(
        shape: &VectorClipShape,
        linear: Mat2,
        translation: Vec2,
        parent: Option<&ClippedBy>,
    ) -> ClipNode {
        let inverse = linear.inverse();
        let inv_translation = -(inverse * translation);
        let (half_extents, radius) = match *shape {
            VectorClipShape::RoundedRect { half_extents, radius } => {
                ([half_extents.x - radius, half_extents.y - radius], radius)
            }
            VectorClipShape::Circle { radius } => ([0.0, 0.0], radius),
        };
        ClipNode {
            entry: GpuClip {
                inv_linear: inverse.to_cols_array(),
                inv_translation: inv_translation.to_array(),
                half_extents,
                radius,
                _pad: [0.0; 3],
            },
            parent: parent.map(|p| p.0),
        }
    }

    let mut nodes: HashMap<Entity, ClipNode> = HashMap::default();
    for (entity, shape, transform, parent) in &clips {
        let model = transform.to_matrix();
        let linear = Mat2::from_cols_array(&[
            model.x_axis.x,
            model.x_axis.y,
            model.y_axis.x,
            model.y_axis.y,
        ]);
        let translation = Vec2::new(model.w_axis.x, model.w_axis.y);
        nodes.insert(entity, clip_node(shape, linear, translation, parent));
    }
    for (entity, shape, hud, parent) in &flat_clips {
        let (linear, translation, _z) = hud.decompose();
        nodes.insert(
            entity,
            clip_node(
                shape,
                Mat2::from_cols_array(&linear),
                Vec2::new(translation[0], translation[1]),
                parent,
            ),
        );
    }

    // Materialize each head's chain (walking ancestors, capped at 4) once.
    let heads: Vec<Entity> = nodes.keys().copied().collect();
    for head in heads {
        let start = extracted.entries.len() as u32;
        if start as usize >= MAX_CLIPS {
            break;
        }
        let mut count = 0u32;
        let mut cursor = Some(head);
        while let (Some(entity), true) = (cursor, count < 4) {
            let Some(node) = nodes.get(&entity) else { break };
            if extracted.entries.len() >= MAX_CLIPS {
                break;
            }
            extracted.entries.push(node.entry);
            count += 1;
            cursor = node.parent;
        }
        extracted.chains.insert(head, (start, count));
    }
}

/// Copies parametric primitives into the render world — pure instance data,
/// no geometry work of any kind.
fn extract_primitives(
    primitives: Extract<
        Query<(&VectorPrimitive, &GlobalTransform, Option<&ClippedBy>), Without<HudTransform>>,
    >,
    flat_primitives: Extract<Query<(&VectorPrimitive, &HudTransform, Option<&ClippedBy>)>>,
    clips: Res<ExtractedClips>,
    mut extracted: ResMut<ExtractedParametrics>,
) {
    extracted.0.clear();
    for (primitive, transform, clipped) in &primitives {
        let model = transform.to_matrix();
        push_primitive(
            &mut extracted,
            &clips,
            primitive,
            [model.x_axis.x, model.x_axis.y, model.y_axis.x, model.y_axis.y],
            [model.w_axis.x, model.w_axis.y],
            model.w_axis.z,
            clipped,
        );
    }
    for (primitive, hud, clipped) in &flat_primitives {
        let (linear, translation, z) = hud.decompose();
        push_primitive(&mut extracted, &clips, primitive, linear, translation, z, clipped);
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
            item.clip.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    };

    let gpu_instance = |item: &ExtractedInstance| GpuInstance {
        linear: item.linear,
        translation_z: [item.translation[0], item.translation[1], item.z, item.clip],
        color: item.color,
        brush_params: item.brush_params,
        brush_meta: item.brush_meta,
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
    let push_instance = |instances: &mut Vec<GpuInstance>,
                             permutation: &mut Vec<u32>,
                             item_index: usize,
                             item: &ExtractedInstance|
     -> u32 {
        let index = instances.len() as u32;
        instances.push(gpu_instance(item));
        permutation.push(item_index as u32);
        index
    };

    // Section 1: opaque interiors, geometry-grouped for instancing, with
    // groups ordered front-to-back by their nearest member (and items
    // front-to-back within each group) so early-z rejects the most hidden
    // fragments under heavy overlap.
    let mut opaque_order: Vec<usize> = (0..extracted.items.len())
        .filter(|&i| extracted.items[i].opaque)
        .collect();
    let mut group_front: HashMap<(u8, u64), f32> = HashMap::default();
    for &i in &opaque_order {
        let item = &extracted.items[i];
        let entry = group_front.entry(group_key(&item.geometry)).or_insert(item.z);
        *entry = entry.max(item.z);
    }
    opaque_order.sort_unstable_by(|&a, &b| {
        let (ia, ib) = (&extracted.items[a], &extracted.items[b]);
        let (ka, kb) = (group_key(&ia.geometry), group_key(&ib.geometry));
        group_front[&kb]
            .total_cmp(&group_front[&ka])
            .then(ka.cmp(&kb))
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

/// Parametric primitives: build the canonical mesh once, then rewrite the
/// (tiny) instance buffer each frame — opaque section first, translucent
/// back-to-front behind it.
fn prepare_parametrics(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut extracted: ResMut<ExtractedParametrics>,
    mut buffers: ResMut<VectorBuffers>,
) {
    if extracted.0.is_empty() {
        buffers.param_total_count = 0;
        buffers.param_opaque_count = 0;
        return;
    }
    if buffers.param_vertex.is_none() {
        let (vertices, indices) = canonical_arc_mesh();
        buffers.param_vertex = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pf_vector_param_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        }));
        buffers.param_index = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pf_vector_param_indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        }));
    }

    extracted.0.sort_unstable_by(|a, b| {
        b.opaque.cmp(&a.opaque).then(a.z.total_cmp(&b.z))
    });
    let instances: Vec<GpuParamInstance> = extracted
        .0
        .iter()
        .map(|item| GpuParamInstance {
            linear: item.linear,
            translation_z: [item.translation[0], item.translation[1], item.z, item.clip],
            color: item.color,
            params: item.params,
        })
        .collect();
    buffers.param_total_count = instances.len() as u32;
    buffers.param_opaque_count = extracted.0.iter().filter(|i| i.opaque).count() as u32;

    let bytes: &[u8] = bytemuck::cast_slice(&instances);
    if buffers.param_instance.is_none() || buffers.param_instance_capacity < bytes.len() {
        buffers.param_instance = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pf_vector_param_instances"),
            contents: bytes,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        }));
        buffers.param_instance_capacity = bytes.len();
    } else if let Some(buffer) = &buffers.param_instance {
        queue.write_buffer(buffer, 0, bytes);
    }
}

/// Uploads this frame's analytic clip entries into the fixed-capacity
/// storage buffer (created once so view bind groups never churn).
fn prepare_clips(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    extracted: Res<ExtractedClips>,
    mut buffers: ResMut<VectorBuffers>,
) {
    let buffer = buffers.clip.get_or_insert_with(|| {
        device.create_buffer(&BufferDescriptor {
            label: Some("pf_vector_clips"),
            size: (MAX_CLIPS * size_of::<GpuClip>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    if !extracted.entries.is_empty() {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&extracted.entries));
    }
    if std::env::var("PF_DEBUG_CLIPS").is_ok() {
        eprintln!(
            "clips: {} entries, {} chains",
            extracted.entries.len(),
            extracted.chains.len()
        );
        if let Some(e) = extracted.entries.first() {
            eprintln!(
                "first entry: inv={:?} t={:?} half={:?} r={}",
                e.inv_linear, e.inv_translation, e.half_extents, e.radius
            );
        }
    }
}

/// Creates the gradient atlas once and uploads any newly-baked rows.
fn prepare_gradients(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut atlas: ResMut<GradientAtlas>,
    mut buffers: ResMut<VectorBuffers>,
) {
    use bevy::render::render_resource::{
        AddressMode, Extent3d, FilterMode, Origin3d, SamplerDescriptor, TexelCopyBufferLayout,
        TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureDimension, TextureUsages,
    };
    if buffers.gradient_texture.is_none() {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("pf_vector_gradients"),
            size: Extent3d {
                width: GRADIENT_ATLAS_SIZE,
                height: GRADIENT_ATLAS_ROWS,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        buffers.gradient_view = Some(texture.create_view(&Default::default()));
        buffers.gradient_texture = Some(texture);
        buffers.gradient_sampler = Some(device.create_sampler(&SamplerDescriptor {
            label: Some("pf_vector_gradient_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        }));
    }
    if let Some(texture) = &buffers.gradient_texture {
        for (row, texels) in atlas.pending.drain(..) {
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: row, z: 0 },
                    aspect: TextureAspect::All,
                },
                &texels,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(GRADIENT_ATLAS_SIZE * 4),
                    rows_per_image: None,
                },
                Extent3d { width: GRADIENT_ATLAS_SIZE, height: 1, depth_or_array_layers: 1 },
            );
        }
    }
}

fn prepare_view_bind_groups(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline: Res<VectorPipeline>,
    buffers: Res<VectorBuffers>,
    views: Query<(Entity, &ExtractedView)>,
    mut bind_groups: ResMut<VectorViewBindGroups>,
) {
    let (Some(clip_buffer), Some(gradient_view), Some(gradient_sampler)) =
        (&buffers.clip, &buffers.gradient_view, &buffers.gradient_sampler)
    else {
        return;
    };
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
                &BindGroupEntries::with_indices((
                    (0, entry.uniform.binding().unwrap()),
                    (1, clip_buffer.as_entire_binding()),
                    (2, gradient_view),
                    (3, gradient_sampler),
                )),
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
    let has_tessellated = (!buffers.opaque_batches.is_empty()
        || !buffers.blend_batches.is_empty())
        && buffers.vertex.is_some()
        && buffers.index.is_some()
        && buffers.instance.is_some();
    let has_params = buffers.param_total_count > 0
        && buffers.param_vertex.is_some()
        && buffers.param_index.is_some()
        && buffers.param_instance.is_some();
    if !has_tessellated && !has_params {
        return;
    }
    let view_entity = view.entity();
    let (camera, extracted_view, target, depth, msaa) = view.into_inner();

    let get = |id: Option<CachedRenderPipelineId>| {
        id.and_then(|id| pipeline_cache.get_render_pipeline(id))
    };
    let (format, samples) = (extracted_view.target_format, msaa.samples());
    let opaque_pipeline = get(pipeline.get(format, samples, true));
    let blend_pipeline = get(pipeline.get(format, samples, false));
    let param_opaque_pipeline = get(pipeline.get_param(format, samples, true));
    let param_blend_pipeline = get(pipeline.get_param(format, samples, false));
    let tess_ready = has_tessellated && opaque_pipeline.is_some() && blend_pipeline.is_some();
    let params_ready =
        has_params && param_opaque_pipeline.is_some() && param_blend_pipeline.is_some();
    if !tess_ready && !params_ready {
        // Still compiling; skip the frame rather than stall.
        return;
    }
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

    let indirect = buffers.use_multi_draw.then(|| buffers.indirect.as_ref()).flatten();
    let args_size = size_of::<GpuDrawIndexedIndirect>() as u64;

    macro_rules! bind_tessellated {
        () => {{
            pass.set_vertex_buffer(0, buffers.vertex.as_ref().unwrap().slice(..));
            pass.set_vertex_buffer(1, buffers.instance.as_ref().unwrap().slice(..));
            pass.set_index_buffer(
                buffers.index.as_ref().unwrap().slice(..),
                IndexFormat::Uint32,
            );
        }};
    }
    macro_rules! bind_params {
        () => {{
            pass.set_vertex_buffer(0, buffers.param_vertex.as_ref().unwrap().slice(..));
            pass.set_vertex_buffer(1, buffers.param_instance.as_ref().unwrap().slice(..));
            pass.set_index_buffer(
                buffers.param_index.as_ref().unwrap().slice(..),
                IndexFormat::Uint32,
            );
        }};
    }

    // Opaque phase: depth-tested, order-free across both sources.
    if tess_ready && !buffers.opaque_batches.is_empty() {
        bind_tessellated!();
        pass.set_render_pipeline(opaque_pipeline.unwrap());
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
    if params_ready && buffers.param_opaque_count > 0 {
        bind_params!();
        pass.set_render_pipeline(param_opaque_pipeline.unwrap());
        pass.draw_indexed(0..ARC_INTERIOR_INDEX_COUNT, 0, 0..buffers.param_opaque_count);
    }

    // Blend phase. Tessellated blend items are strictly back-to-front among
    // themselves; parametric translucent interiors and all parametric
    // fringes draw after them (1px fringes — cross-source ordering error is
    // visually negligible for HUD content).
    if tess_ready && !buffers.blend_batches.is_empty() {
        bind_tessellated!();
        pass.set_render_pipeline(blend_pipeline.unwrap());
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
    if params_ready {
        let total = buffers.param_total_count;
        let translucent = buffers.param_opaque_count..total;
        bind_params!();
        pass.set_render_pipeline(param_blend_pipeline.unwrap());
        if !translucent.is_empty() {
            pass.draw_indexed(0..ARC_INTERIOR_INDEX_COUNT, 0, translucent);
        }
        // Fringes for every parametric instance.
        let index_count: u32 = (64 * 6) + (64 * 12) + 12;
        pass.draw_indexed(ARC_INTERIOR_INDEX_COUNT..index_count, 0, 0..total);
    }

    pass_span.end(&mut pass);
}
