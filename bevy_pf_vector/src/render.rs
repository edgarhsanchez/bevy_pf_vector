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
use bevy::camera::visibility::RenderLayers;
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

use crate::painter::VectorPainterQueue;
use crate::tess;
use crate::{ClippedBy, HudTransform, VectorClipShape, VectorPrimitive, VectorShape};

pub const VECTOR_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("7a3f1c2e-9b4d-4b8a-a2f0-5e1d3c6b9f01");
pub const VECTOR_PARAM_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("2c8e5b1a-4f7d-4c3e-9a06-8b21d75c4e90");
pub const VECTOR_SDF_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("5d1b7f34-0a62-4e19-8c7d-3f9e2a4b6c81");

pub struct VectorRenderPlugin;

impl Plugin for VectorRenderPlugin {
    fn build(&self, app: &mut App) {
        // Degrade instead of panicking when there is no renderer. A headless
        // App (MinimalPlugins, or any integration test that exercises UI logic
        // without a GPU) has no `Assets<Shader>`, and `load_internal_asset!`
        // unwraps it -- so merely ADDING this plugin took the process down.
        // That made bevy_pf's `vector_gpu` feature untestable headlessly: it
        // pulls in this plugin, so `cargo test --features vector_gpu` failed
        // in a test that never drew anything.
        //
        // Nothing below can function without a render app anyway, so skipping
        // is the honest behaviour; drawing is simply absent, as it already is
        // for every other renderer in a headless app.
        if !app.world().contains_resource::<Assets<Shader>>() {
            return;
        }
        load_internal_asset!(app, VECTOR_SHADER_HANDLE, "vector.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            VECTOR_PARAM_SHADER_HANDLE,
            "vector_param.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            VECTOR_SDF_SHADER_HANDLE,
            "vector_sdf.wgsl",
            Shader::from_wgsl
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<GeometryCache>()
            .init_resource::<ExtractedShapes>()
            .init_resource::<ExtractedParametrics>()
            .init_resource::<ExtractedSdf>()
            .init_resource::<ExtractedClips>()
            .init_resource::<LayerTable>()
            .init_resource::<GeometryKeys>()
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
                    prepare_sdf.in_set(RenderSystems::PrepareResources),
                    prepare_clips.in_set(RenderSystems::PrepareResources),
                    prepare_gradients.in_set(RenderSystems::PrepareResources),
                    prepare_view_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            // Prepass slot: vector content draws BEFORE the main 2D pass.
            // Opaque vector interiors write depth, and Bevy's sprite/text/
            // mesh2d pipelines depth-test (GreaterEqual, no write), so
            // main-pass content interleaves with vector content by z: lower-z
            // sprites are occluded by higher-z opaque vector chrome, higher-z
            // widgets/text draw over it. Bonus: pixels behind opaque HUD are
            // early-z culled out of the (expensive) main pass instead of the
            // other way around. Known limit: TRANSLUCENT vector content
            // cannot occlude main-pass content above it — keep translucent
            // vector z above legacy content or use an overlay camera.
            .add_systems(Core2d, vector_pass.in_set(Core2dSystems::Prepass));
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

/// Content-hash keys per shape entity, so an unchanged shape never re-hashes
/// its path. Hashing every path every frame was the steady-state cost of the
/// extract loop — for a mostly-static HUD, which is the workload this engine
/// exists for, it dominated everything else it does per frame.
#[derive(Resource, Default)]
pub struct GeometryKeys {
    keys: HashMap<Entity, (Option<u64>, Option<u64>)>,
}

/// Per-frame interned `RenderLayers` masks. Instances and batches carry a
/// small id into this table instead of the mask itself; the pass resolves
/// ids against the current frame's table, so layer *content* changes are
/// picked up even on the layout-fingerprint fast path (the id assignment
/// pattern is part of the fingerprint, the masks themselves need not be).
/// Id 0 is always the default mask (layer 0).
#[derive(Resource, Default)]
pub struct LayerTable {
    masks: Vec<RenderLayers>,
}

impl LayerTable {
    fn clear(&mut self) {
        self.masks.clear();
    }

    fn intern(&mut self, layers: Option<&RenderLayers>) -> u16 {
        if self.masks.is_empty() {
            self.masks.push(RenderLayers::default());
        }
        match layers {
            None => 0,
            Some(layers) => match self.masks.iter().position(|m| m == layers) {
                Some(index) => index as u16,
                None => {
                    self.masks.push(layers.clone());
                    (self.masks.len() - 1) as u16
                }
            },
        }
    }

    fn visible(&self, id: u16, view: &RenderLayers) -> bool {
        self.masks
            .get(id as usize)
            .is_none_or(|mask| mask.intersects(view))
    }
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
    // Interior: inner/outer pair per step at full coverage. These carry
    // their outward directions (radial for the ring edges, tangential at
    // the caps) so the vertex shader can inset them half a pixel — the AA
    // band straddles the authored edge instead of hanging outside it.
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let tangential = if i == 0 {
            -1.0
        } else if i == n {
            1.0
        } else {
            0.0
        };
        for side in [0.0f32, 1.0] {
            let radial = if side == 0.0 { -1.0 } else { 1.0 };
            vertices.push(GpuVertex {
                position: [t, side],
                normal: [radial, tangential],
                coverage: 1.0,
            });
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
    // End-cap corners displace tangentially (backward at t=0, forward at
    // t=1) AND radially, so each is the outward twin of the interior corner
    // vertex it pairs with.
    let caps = vertices.len() as u32;
    vertices.push(GpuVertex { position: [0.0, 0.0], normal: [-1.0, -1.0], coverage: 0.0 });
    vertices.push(GpuVertex { position: [0.0, 1.0], normal: [1.0, -1.0], coverage: 0.0 });
    vertices.push(GpuVertex { position: [1.0, 0.0], normal: [-1.0, 1.0], coverage: 0.0 });
    vertices.push(GpuVertex { position: [1.0, 1.0], normal: [1.0, 1.0], coverage: 0.0 });

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
#[allow(dead_code)] // `Dynamic` is retained: see the cache note in `push_shape`.
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
    layer: u16,
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
    /// Retained alongside `GeometryRef::Dynamic`: with the cache now consulted
    /// for changed shapes, nothing routes here, but a path that morphs every
    /// frame still wants a transient region rather than a cache entry per
    /// frame. Re-wiring that needs a "has this key churned?" signal.
    #[allow(dead_code)]
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
    layer: u16,
}

struct ParamItem {
    z: f32,
    linear: [f32; 4],
    translation: [f32; 2],
    color: [u8; 4],
    params: [f32; 4],
    opaque: bool,
    clip: f32,
    layer: u16,
}

#[derive(Resource, Default)]
pub struct ExtractedParametrics(Vec<ParamItem>);

/// SDF primitives (rect / rounded rect / circle / line). Same instance
/// layout as the arcs, different mesh and shader: one quad, distance
/// evaluated per fragment.
#[derive(Resource, Default)]
pub struct ExtractedSdf(Vec<ParamItem>);

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
    /// Contiguous layer-homogeneous runs over the sorted parametric
    /// instances, so per-view filtering can skip whole runs.
    param_layer_runs: Vec<(Range<u32>, u16)>,
    /// SDF primitives: a unit quad (built once) plus per-frame instances.
    sdf_vertex: Option<Buffer>,
    sdf_index: Option<Buffer>,
    sdf_instance: Option<Buffer>,
    sdf_instance_capacity: usize,
    sdf_opaque_count: u32,
    sdf_total_count: u32,
    sdf_layer_runs: Vec<(Range<u32>, u16)>,
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
    sdf_variants: HashMap<(TextureFormat, u32, bool), CachedRenderPipelineId>,
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
            sdf_variants: HashMap::new(),
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

    fn ensure_sdf(
        &mut self,
        cache: &PipelineCache,
        format: TextureFormat,
        samples: u32,
        opaque: bool,
    ) -> CachedRenderPipelineId {
        let view_layout = self.view_layout.clone();
        *self
            .sdf_variants
            .entry((format, samples, opaque))
            .or_insert_with(|| {
                cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some(
                        if opaque { "pf_vector_sdf_opaque_pipeline" } else { "pf_vector_sdf_blend_pipeline" }
                            .into(),
                    ),
                    layout: vec![view_layout],
                    vertex: VertexState {
                        shader: VECTOR_SDF_SHADER_HANDLE,
                        entry_point: Some("vertex".into()),
                        shader_defs: Vec::new(),
                        buffers: vec![vertex_buffer_layout(), param_instance_buffer_layout()],
                    },
                    fragment: Some(FragmentState {
                        shader: VECTOR_SDF_SHADER_HANDLE,
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

    fn get_sdf(
        &self,
        format: TextureFormat,
        samples: u32,
        opaque: bool,
    ) -> Option<CachedRenderPipelineId> {
        self.sdf_variants.get(&(format, samples, opaque)).copied()
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
#[allow(clippy::too_many_arguments)]
fn extract_shapes(
    shapes: Extract<
        Query<
            (
                Entity,
                Ref<VectorShape>,
                &GlobalTransform,
                Option<&ClippedBy>,
                Option<&RenderLayers>,
            ),
            Without<HudTransform>,
        >,
    >,
    flat_shapes: Extract<
        Query<(
            Entity,
            Ref<VectorShape>,
            &HudTransform,
            Option<&ClippedBy>,
            Option<&RenderLayers>,
        )>,
    >,
    painted: Extract<Res<VectorPainterQueue>>,
    clips: Res<ExtractedClips>,
    mut cache: ResMut<GeometryCache>,
    mut extracted: ResMut<ExtractedShapes>,
    mut atlas: ResMut<GradientAtlas>,
    mut layers: ResMut<LayerTable>,
    mut keys: ResMut<GeometryKeys>,
) {
    // Live shape count from last frame, read before the clear: the epoch
    // flush below needs to know the working-set size.
    let live_last = extracted.items.len();

    extracted.items.clear();
    extracted.dynamic_vertices.clear();
    extracted.dynamic_indices.clear();
    layers.clear();

    // Epoch flush: mutation churn appends cache entries that nothing draws
    // any more, so the cache is dropped wholesale once it is mostly garbage.
    //
    // The threshold MUST scale with the working set. A fixed cap (this was
    // 8192) is not "the cache got absurd", it is "the scene got big": a scene
    // with more distinct geometries than the cap wipes the cache EVERY FRAME
    // and re-tessellates everything, turning the engine's central bet — one
    // tessellation, then instancing forever — into its opposite. That is
    // invisible at 200-5000 elements and catastrophic at 200k (measured:
    // 972 ms/frame against 18 ms of GPU).
    //
    // Flushing only when the cache dwarfs what is actually on screen keeps
    // the original intent (bound garbage from paths that morph every frame)
    // without punishing scenes that legitimately hold many distinct shapes.
    let flush_at = 8192.max(live_last.saturating_mul(4));
    if cache.ranges.len() > flush_at {
        *cache = GeometryCache::default();
        keys.keys.clear();
    }
    // Entities despawn without telling us; keep the key map from growing
    // without bound in a long session of churn.
    if keys.keys.len() > 65536 {
        keys.keys.clear();
    }

    for (entity, shape, transform, clipped, shape_layers) in &shapes {
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
            &shape.commands,
            &shape.style,
            entity,
            &mut keys,
            shape.is_changed(),
            linear,
            translation,
            model.w_axis.z,
            clipped,
            layers.intern(shape_layers),
        );
    }
    for (entity, shape, hud, clipped, shape_layers) in &flat_shapes {
        let (linear, translation, z) = hud.decompose();
        push_shape(
            &mut cache,
            &mut extracted,
            &mut atlas,
            &clips,
            &shape.commands,
            &shape.style,
            entity,
            &mut keys,
            shape.is_changed(),
            linear,
            translation,
            z,
            clipped,
            layers.intern(shape_layers),
        );
    }
    // Immediate-mode painted shapes: always through the geometry cache —
    // frame-repeated painting of the same local-space geometry costs one
    // instance write, exactly like a retained shape.
    for item in &painted.shapes {
        let layer = layers.intern(item.layers.as_ref());
        push_shape(
            &mut cache,
            &mut extracted,
            &mut atlas,
            &clips,
            &item.commands,
            &item.style,
            Entity::PLACEHOLDER,
            &mut keys,
            true,
            item.linear,
            item.translation,
            item.z,
            None,
            layer,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_shape(
    cache: &mut GeometryCache,
    extracted: &mut ExtractedShapes,
    atlas: &mut GradientAtlas,
    clips: &ExtractedClips,
    commands: &[crate::path::PathCommand],
    style: &crate::path::PathStyle,
    entity: Entity,
    keys: &mut GeometryKeys,
    // True on the frame a path/style mutates (and on spawn). Does not select
    // a tessellation path any more (see the cache note below); it selects
    // whether the content hash has to be recomputed at all.
    changed: bool,
    linear: [f32; 4],
    translation: [f32; 2],
    z: f32,
    clipped: Option<&ClippedBy>,
    layer: u16,
) {
    let clip = clipped
        .and_then(|c| clips.chains.get(&c.0))
        .map(|&(start, count)| pack_clip(start, count))
        .unwrap_or(0.0);
    {
        if let Some(brush) = &style.fill {
            let rule = style.fill_rule;
            // Reuse the previous content hash when nothing changed. Hashing
            // is the whole steady-state cost of this loop for a static HUD,
            // and re-deriving an identical key every frame for thousands of
            // unchanged shapes is pure waste.
            //
            // Content-hash and consult the cache even when the component
            // changed. Bevy's change detection is per-COMPONENT, so editing a
            // colour marks the whole `VectorShape` changed — and treating
            // "changed" as "new topology" re-tessellates an identical path
            // every frame. That is the dominant UI case (a `Fill` bound to
            // data) and it measured ~2x SLOWER than CPU rasterization in
            // bevy_pf's shape backend before this fix.
            //
            // A genuinely new path still tessellates, and because the cache is
            // content-addressed it stores one entry per distinct geometry; the
            // epoch flush in `extract_shapes` caps growth for paths that morph
            // every frame.
            let cached = (!changed)
                .then(|| keys.keys.get(&entity).and_then(|k| k.0))
                .flatten();
            let key = match cached {
                // A key in the per-entity map was `ensure`d when it was
                // computed, and the two maps are cleared together, so the
                // geometry is already resident: skip the second hash lookup.
                // That halves the per-shape map traffic in steady state, and
                // map traffic is what `extract_shapes` spends its time on.
                Some(key) => key,
                None => {
                    let key = tess::fill_key(commands, rule);
                    keys.keys.entry(entity).or_default().0 = Some(key);
                    cache.ensure(key, || tess::tessellate_fill(commands, rule));
                    key
                }
            };
            let geometry = Some(GeometryRef::Cached(key));
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
                    layer,
                });
            }
        }
        if let Some(stroke) = &style.stroke {
            // Cached by content, and by entity, for the same reasons as the
            // fill above.
            let cached = (!changed)
                .then(|| keys.keys.get(&entity).and_then(|k| k.1))
                .flatten();
            let key = match cached {
                // Already resident — see the fill above.
                Some(key) => key,
                None => {
                    let key = tess::stroke_key(commands, stroke);
                    keys.keys.entry(entity).or_default().1 = Some(key);
                    cache.ensure(key, || tess::tessellate_stroke(commands, stroke));
                    key
                }
            };
            let geometry = Some(GeometryRef::Cached(key));
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
                    layer,
                });
            }
        }
    }
}

/// Shared per-item processing for both transform sources.
#[allow(clippy::too_many_arguments)]
fn push_primitive(
    extracted: &mut ExtractedParametrics,
    sdf: &mut ExtractedSdf,
    clips: &ExtractedClips,
    primitive: &VectorPrimitive,
    linear: [f32; 4],
    translation: [f32; 2],
    z: f32,
    clipped: Option<&ClippedBy>,
    layer: u16,
) {
    let clip = clipped
        .and_then(|c| clips.chains.get(&c.0))
        .map(|&(start, count)| pack_clip(start, count))
        .unwrap_or(0.0);
    let (target, params, color) = match *primitive {
        VectorPrimitive::Arc { inner, outer, start, sweep, color } => {
            (&mut extracted.0, [start, sweep, inner, outer], color)
        }
        VectorPrimitive::Rect { size, radius, thickness, color } => (
            &mut sdf.0,
            [size.x, size.y, radius, thickness],
            color,
        ),
    };
    target.push(ParamItem {
        z,
        linear,
        translation,
        color: pack_color(color),
        params,
        // A stroked SDF primitive is mostly hole, so it always blends; a
        // filled one can take the opaque path like anything else.
        opaque: color.alpha >= 1.0
            && clip == 0.0
            && !matches!(*primitive, VectorPrimitive::Rect { thickness, .. } if thickness > 0.0),
        clip,
        layer,
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
            pipeline.ensure_sdf(&cache, view.target_format, msaa.samples(), opaque);
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
        Query<
            (&VectorPrimitive, &GlobalTransform, Option<&ClippedBy>, Option<&RenderLayers>),
            Without<HudTransform>,
        >,
    >,
    flat_primitives: Extract<
        Query<(&VectorPrimitive, &HudTransform, Option<&ClippedBy>, Option<&RenderLayers>)>,
    >,
    painted: Extract<Res<VectorPainterQueue>>,
    clips: Res<ExtractedClips>,
    mut extracted: ResMut<ExtractedParametrics>,
    mut sdf: ResMut<ExtractedSdf>,
    mut layers: ResMut<LayerTable>,
) {
    extracted.0.clear();
    sdf.0.clear();
    for (primitive, transform, clipped, item_layers) in &primitives {
        let model = transform.to_matrix();
        push_primitive(
            &mut extracted,
            &mut sdf,
            &clips,
            primitive,
            [model.x_axis.x, model.x_axis.y, model.y_axis.x, model.y_axis.y],
            [model.w_axis.x, model.w_axis.y],
            model.w_axis.z,
            clipped,
            layers.intern(item_layers),
        );
    }
    for (primitive, hud, clipped, item_layers) in &flat_primitives {
        let (linear, translation, z) = hud.decompose();
        push_primitive(
            &mut extracted,
            &mut sdf,
            &clips,
            primitive,
            linear,
            translation,
            z,
            clipped,
            layers.intern(item_layers),
        );
    }
    // Immediate-mode painted arcs: pure instance data, like everything else
    // on the parametric path.
    for arc in &painted.arcs {
        let layer = layers.intern(arc.layers.as_ref());
        extracted.0.push(ParamItem {
            z: arc.z,
            linear: arc.linear,
            translation: arc.translation,
            color: pack_color(arc.color),
            params: [arc.start, arc.sweep, arc.inner, arc.outer],
            opaque: arc.color.alpha >= 1.0,
            clip: 0.0,
            layer,
        });
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
    // Batches must be homogeneous in geometry AND layer mask (per-view
    // filtering skips whole batches), so the layer id is part of the key.
    let group_key = |item: &ExtractedInstance| -> (u8, u64, u16) {
        match &item.geometry {
            GeometryRef::Cached(key) => (0, *key, item.layer),
            GeometryRef::Dynamic(range) => (
                1,
                (u64::from(range.interior_first) << 32) | u64::from(range.base_vertex as u32),
                item.layer,
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
            group_key(item).hash(&mut hasher);
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
    let mut group_front: HashMap<(u8, u64, u16), f32> = HashMap::default();
    for &i in &opaque_order {
        let item = &extracted.items[i];
        let entry = group_front.entry(group_key(item)).or_insert(item.z);
        *entry = entry.max(item.z);
    }
    opaque_order.sort_unstable_by(|&a, &b| {
        let (ia, ib) = (&extracted.items[a], &extracted.items[b]);
        let (ka, kb) = (group_key(ia), group_key(ib));
        group_front[&kb]
            .total_cmp(&group_front[&ka])
            .then(ka.cmp(&kb))
            .then(ib.z.total_cmp(&ia.z))
    });
    let mut last_geometry: Option<(u8, u64, u16)> = None;
    for &item_index in &opaque_order {
        let item = &extracted.items[item_index];
        let Some(range) = resolve(&item.geometry) else {
            continue;
        };
        if range.interior_count == 0 {
            continue;
        }
        let index = push_instance(&mut instances, &mut buffers.permutation, item_index, item);
        let key = group_key(item);
        if last_geometry == Some(key) {
            buffers.opaque_batches.last_mut().unwrap().instances.end = index + 1;
        } else {
            buffers.opaque_batches.push(VectorBatch {
                indices: range.interior_first..range.interior_first + range.interior_count,
                base_vertex: range.base_vertex,
                instances: index..index + 1,
                layer: item.layer,
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
    let mut last_key: Option<((u8, u64, u16), bool)> = None;
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
        let key = (group_key(item), is_fringe);
        if last_key == Some(key) {
            buffers.blend_batches.last_mut().unwrap().instances.end = index + 1;
        } else {
            buffers.blend_batches.push(VectorBatch {
                indices: first..first + count,
                base_vertex: range.base_vertex,
                instances: index..index + 1,
                layer: item.layer,
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
        buffers.param_layer_runs.clear();
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

    // Opaque section groups by layer (order-free under depth testing) to
    // minimize layer runs; the translucent section stays strictly z-ordered
    // and splits into runs wherever the layer changes.
    extracted.0.sort_unstable_by(|a, b| {
        b.opaque
            .cmp(&a.opaque)
            .then_with(|| if a.opaque { a.layer.cmp(&b.layer) } else { std::cmp::Ordering::Equal })
            .then(a.z.total_cmp(&b.z))
    });
    buffers.param_layer_runs.clear();
    for (i, item) in extracted.0.iter().enumerate() {
        match buffers.param_layer_runs.last_mut() {
            Some((range, layer)) if *layer == item.layer => range.end = i as u32 + 1,
            _ => buffers.param_layer_runs.push((i as u32..i as u32 + 1, item.layer)),
        }
    }
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

/// SDF primitives: a unit quad built once, then a per-frame instance
/// rewrite. Nothing here scales with shape SIZE — that is the whole point of
/// the primitive, and why a resizing bar costs one instance write.
fn prepare_sdf(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut extracted: ResMut<ExtractedSdf>,
    mut buffers: ResMut<VectorBuffers>,
) {
    if extracted.0.is_empty() {
        buffers.sdf_total_count = 0;
        buffers.sdf_opaque_count = 0;
        buffers.sdf_layer_runs.clear();
        return;
    }
    if buffers.sdf_vertex.is_none() {
        // Unit quad in [-1, 1]; the vertex shader scales it per instance.
        let corners = [[-1.0f32, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let vertices: Vec<GpuVertex> = corners
            .iter()
            .map(|&position| GpuVertex { position, normal: [0.0, 0.0], coverage: 1.0 })
            .collect();
        buffers.sdf_vertex = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pf_vector_sdf_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        }));
        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        buffers.sdf_index = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pf_vector_sdf_indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        }));
    }

    extracted.0.sort_unstable_by(|a, b| {
        b.opaque
            .cmp(&a.opaque)
            .then_with(|| if a.opaque { a.layer.cmp(&b.layer) } else { std::cmp::Ordering::Equal })
            .then(a.z.total_cmp(&b.z))
    });
    buffers.sdf_layer_runs.clear();
    for (i, item) in extracted.0.iter().enumerate() {
        match buffers.sdf_layer_runs.last_mut() {
            Some((range, layer)) if *layer == item.layer => range.end = i as u32 + 1,
            _ => buffers.sdf_layer_runs.push((i as u32..i as u32 + 1, item.layer)),
        }
    }
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
    buffers.sdf_total_count = instances.len() as u32;
    buffers.sdf_opaque_count = extracted.0.iter().filter(|i| i.opaque).count() as u32;

    let bytes: &[u8] = bytemuck::cast_slice(&instances);
    if buffers.sdf_instance.is_none() || buffers.sdf_instance_capacity < bytes.len() {
        buffers.sdf_instance = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pf_vector_sdf_instances"),
            contents: bytes,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        }));
        buffers.sdf_instance_capacity = bytes.len();
    } else if let Some(buffer) = &buffers.sdf_instance {
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

/// The vector pass. Runs in the `Core2d` schedule before the main 2D pass
/// (see plugin registration for the depth-interleaving rationale): opaque
/// interiors first (depth write, no blend, early-z), then blend items
/// back-to-front (translucent interiors + AA fringes, depth read-only).
/// Two multi-draw calls total when the device supports indirect.
fn vector_pass(
    pipeline: Res<VectorPipeline>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Res<VectorBuffers>,
    bind_groups: Res<VectorViewBindGroups>,
    layer_table: Res<LayerTable>,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewTarget,
        &ViewDepthTexture,
        &Msaa,
        Option<&RenderLayers>,
    )>,
    mut ctx: RenderContext,
) {
    let has_tessellated = (!buffers.opaque_batches.is_empty()
        || !buffers.blend_batches.is_empty())
        && buffers.vertex.is_some()
        && buffers.index.is_some()
        && buffers.instance.is_some();
    let has_sdf = buffers.sdf_total_count > 0
        && buffers.sdf_vertex.is_some()
        && buffers.sdf_index.is_some()
        && buffers.sdf_instance.is_some();
    let has_params = buffers.param_total_count > 0
        && buffers.param_vertex.is_some()
        && buffers.param_index.is_some()
        && buffers.param_instance.is_some();
    if !has_tessellated && !has_params && !has_sdf {
        return;
    }
    let view_entity = view.entity();
    let (camera, extracted_view, target, depth, msaa, view_layers) = view.into_inner();
    let view_mask = view_layers.cloned().unwrap_or_default();

    let get = |id: Option<CachedRenderPipelineId>| {
        id.and_then(|id| pipeline_cache.get_render_pipeline(id))
    };
    let (format, samples) = (extracted_view.target_format, msaa.samples());
    let opaque_pipeline = get(pipeline.get(format, samples, true));
    let blend_pipeline = get(pipeline.get(format, samples, false));
    let param_opaque_pipeline = get(pipeline.get_param(format, samples, true));
    let param_blend_pipeline = get(pipeline.get_param(format, samples, false));
    let sdf_opaque_pipeline = get(pipeline.get_sdf(format, samples, true));
    let sdf_blend_pipeline = get(pipeline.get_sdf(format, samples, false));
    let sdf_ready = has_sdf && sdf_opaque_pipeline.is_some() && sdf_blend_pipeline.is_some();
    let tess_ready = has_tessellated && opaque_pipeline.is_some() && blend_pipeline.is_some();
    let params_ready =
        has_params && param_opaque_pipeline.is_some() && param_blend_pipeline.is_some();
    if !tess_ready && !params_ready && !sdf_ready {
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
    macro_rules! bind_sdf {
        () => {{
            pass.set_vertex_buffer(0, buffers.sdf_vertex.as_ref().unwrap().slice(..));
            pass.set_vertex_buffer(1, buffers.sdf_instance.as_ref().unwrap().slice(..));
            pass.set_index_buffer(
                buffers.sdf_index.as_ref().unwrap().slice(..),
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

    // Batches whose layer mask misses this view are skipped; contiguous
    // visible runs still submit as single multi-draws, so the single-camera
    // common case (everything visible) remains exactly two multi-draw calls.
    macro_rules! draw_filtered {
        ($batches:expr, $args_base:expr) => {{
            let batches = $batches;
            match indirect {
                Some(indirect) => {
                    let mut i = 0usize;
                    while i < batches.len() {
                        if !layer_table.visible(batches[i].layer, &view_mask) {
                            i += 1;
                            continue;
                        }
                        let start = i;
                        while i < batches.len()
                            && layer_table.visible(batches[i].layer, &view_mask)
                        {
                            i += 1;
                        }
                        pass.multi_draw_indexed_indirect(
                            indirect,
                            ($args_base + start) as u64 * args_size,
                            (i - start) as u32,
                        );
                    }
                }
                None => {
                    for batch in batches {
                        if !layer_table.visible(batch.layer, &view_mask) {
                            continue;
                        }
                        pass.draw_indexed(
                            batch.indices.clone(),
                            batch.base_vertex,
                            batch.instances.clone(),
                        );
                    }
                }
            }
        }};
    }

    // Opaque phase: depth-tested, order-free across both sources.
    if tess_ready && !buffers.opaque_batches.is_empty() {
        bind_tessellated!();
        pass.set_render_pipeline(opaque_pipeline.unwrap());
        draw_filtered!(&buffers.opaque_batches, 0);
    }
    if sdf_ready && buffers.sdf_opaque_count > 0 {
        bind_sdf!();
        pass.set_render_pipeline(sdf_opaque_pipeline.unwrap());
        for (range, layer) in &buffers.sdf_layer_runs {
            let run = range.start..range.end.min(buffers.sdf_opaque_count);
            if run.start < run.end && layer_table.visible(*layer, &view_mask) {
                pass.draw_indexed(0..6, 0, run);
            }
        }
    }
    if params_ready && buffers.param_opaque_count > 0 {
        bind_params!();
        pass.set_render_pipeline(param_opaque_pipeline.unwrap());
        for (range, layer) in &buffers.param_layer_runs {
            let run = range.start..range.end.min(buffers.param_opaque_count);
            if run.start < run.end && layer_table.visible(*layer, &view_mask) {
                pass.draw_indexed(0..ARC_INTERIOR_INDEX_COUNT, 0, run);
            }
        }
    }

    // Blend phase. Tessellated blend items are strictly back-to-front among
    // themselves; parametric translucent interiors and all parametric
    // fringes draw after them (1px fringes — cross-source ordering error is
    // visually negligible for HUD content).
    if tess_ready && !buffers.blend_batches.is_empty() {
        bind_tessellated!();
        pass.set_render_pipeline(blend_pipeline.unwrap());
        draw_filtered!(&buffers.blend_batches, buffers.opaque_batches.len());
    }
    if params_ready {
        let total = buffers.param_total_count;
        bind_params!();
        pass.set_render_pipeline(param_blend_pipeline.unwrap());
        for (range, layer) in &buffers.param_layer_runs {
            let run = range.start.max(buffers.param_opaque_count)..range.end.min(total);
            if run.start < run.end && layer_table.visible(*layer, &view_mask) {
                pass.draw_indexed(0..ARC_INTERIOR_INDEX_COUNT, 0, run);
            }
        }
        // Fringes for every parametric instance.
        let index_count: u32 = (64 * 6) + (64 * 12) + 12;
        for (range, layer) in &buffers.param_layer_runs {
            if layer_table.visible(*layer, &view_mask) {
                pass.draw_indexed(ARC_INTERIOR_INDEX_COUNT..index_count, 0, range.clone());
            }
        }
    }

    if sdf_ready {
        // Translucent SDF primitives, back-to-front within each layer run.
        // No separate fringe pass: antialiasing is per-fragment here, so the
        // edge needs no extra geometry.
        bind_sdf!();
        pass.set_render_pipeline(sdf_blend_pipeline.unwrap());
        for (range, layer) in &buffers.sdf_layer_runs {
            let run = range.start.max(buffers.sdf_opaque_count)..range.end;
            if run.start < run.end && layer_table.visible(*layer, &view_mask) {
                pass.draw_indexed(0..6, 0, run);
            }
        }
    }

    pass_span.end(&mut pass);
}
