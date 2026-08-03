//! The seam. Engine iterations implement this so alternative implementations
//! can be swapped at runtime and benchmarked head-to-head in the same loop.

use bevy::prelude::*;
use bevy::render::render_resource::{CommandEncoder, TextureView};

/// Stable handle to geometry uploaded via [`VectorBackend::upload_geometry`].
/// Allocated by the plugin when a vector asset loads; meaningless across runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeometryId(pub u32);

/// Path outline in the usual verb form, matching what lyon/kurbo consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathCommand {
    MoveTo(Vec2),
    LineTo(Vec2),
    QuadTo { ctrl: Vec2, to: Vec2 },
    CubicTo { ctrl1: Vec2, ctrl2: Vec2, to: Vec2 },
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathStyle {
    pub fill: Option<LinearRgba>,
    pub stroke: Option<StrokeStyle>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeStyle {
    pub color: LinearRgba,
    pub width: f32,
    pub join: LineJoin,
    pub cap: LineCap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineJoin {
    Miter,
    Round,
    Bevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineCap {
    Butt,
    Round,
    Square,
}

/// What a backend records into: Bevy's command encoder for the frame. The
/// finished buffer is handed to `PendingCommandBuffers` before the render
/// graph's Submit set runs.
pub struct BackendEncoder<'a> {
    pub encoder: &'a mut CommandEncoder,
}

/// A resolved, GPU-ready vector scene for one frame.
pub struct FrameScene<'a> {
    pub target: &'a TextureView,
    pub target_size: UVec2,
    /// Instances whose topology is unchanged since upload; transform-only updates.
    pub static_instances: &'a [VectorInstance],
    /// Instances whose geometry changed this frame and must be re-processed.
    pub dynamic: &'a [DynamicPath],
}

#[derive(Clone, Copy)]
pub struct VectorInstance {
    pub geometry: GeometryId,
    pub transform: Mat4,
    pub color: LinearRgba,
}

pub struct DynamicPath {
    pub commands: Vec<PathCommand>,
    pub transform: Mat4,
    pub style: PathStyle,
}

pub trait VectorBackend: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    /// Called once per geometry asset. Backends that tessellate ahead of time
    /// do the work here; backends that rasterize per-frame may no-op.
    fn upload_geometry(&mut self, id: GeometryId, path: &[PathCommand]);

    /// Record draw work for one frame. Must not block on GPU.
    fn record(&mut self, scene: &FrameScene, encoder: &mut BackendEncoder<'_>);
}
