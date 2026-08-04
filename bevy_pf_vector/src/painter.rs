//! Immediate-mode painting: a per-frame command queue over the same
//! tessellate-once machinery as retained [`VectorShape`](crate::VectorShape)
//! entities.
//!
//! Why this is still fast: painted geometry is authored in *local* space and
//! keyed by content hash, so a HUD that paints the same rects/circles/paths
//! every frame (the immediate-mode norm) hits the persistent geometry cache
//! and costs exactly what a retained shape costs — one instance write. Only
//! geometry whose *dimensions* change tessellates again; transforms, colors,
//! and z never do. Ring wedges route to the parametric arc path and never
//! tessellate at all.

use bevy::camera::visibility::RenderLayers;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use crate::path::{Brush, LineCap, LineJoin, PathCommand, PathStyle, StrokeStyle};

/// One shape painted this frame.
pub(crate) struct PaintedShape {
    pub commands: Vec<PathCommand>,
    pub style: PathStyle,
    pub linear: [f32; 4],
    pub translation: [f32; 2],
    pub z: f32,
    pub layers: Option<RenderLayers>,
}

/// One parametric arc painted this frame (never tessellates).
pub(crate) struct PaintedArc {
    pub inner: f32,
    pub outer: f32,
    pub start: f32,
    pub sweep: f32,
    pub color: LinearRgba,
    pub linear: [f32; 4],
    pub translation: [f32; 2],
    pub z: f32,
    pub layers: Option<RenderLayers>,
}

/// Main-world queue the render world extracts from; cleared in `First` so
/// `Update`-schedule painting is visible to the following extract.
#[derive(Resource, Default)]
pub struct VectorPainterQueue {
    pub(crate) shapes: Vec<PaintedShape>,
    pub(crate) arcs: Vec<PaintedArc>,
}

pub(crate) fn clear_painter_queue(mut queue: ResMut<VectorPainterQueue>) {
    queue.shapes.clear();
    queue.arcs.clear();
}

/// Painter state that persists across calls (and, like any `Local`, across
/// runs of the same system — set what you rely on).
#[derive(Clone, Debug)]
pub struct PainterConfig {
    /// Full transform; the 2D projection (xy linear part + translation) is
    /// what instances use, `translation.z` is the depth/sort key.
    pub transform: Transform,
    /// Limit painting to cameras whose `RenderLayers` intersect these;
    /// `None` = default layer 0.
    pub render_layers: Option<RenderLayers>,
    pub cap: LineCap,
    pub join: LineJoin,
}

impl Default for PainterConfig {
    fn default() -> Self {
        Self {
            transform: Transform::IDENTITY,
            render_layers: None,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
        }
    }
}

/// Immediate-mode painter. Take it as a system parameter and draw; painted
/// content lives for the current frame.
///
/// ```ignore
/// fn hud(mut painter: VectorPainter) {
///     painter.render_layers = Some(RenderLayers::layer(1));
///     painter.transform.translation.z = -50.0;
///     painter.fill_rect(Vec2::ZERO, Vec2::new(300.0, 200.0), Color::WHITE);
/// }
/// ```
#[derive(SystemParam)]
pub struct VectorPainter<'w, 's> {
    queue: ResMut<'w, VectorPainterQueue>,
    config: Local<'s, PainterConfig>,
}

impl std::ops::Deref for VectorPainter<'_, '_> {
    type Target = PainterConfig;
    fn deref(&self) -> &PainterConfig {
        &self.config
    }
}

impl std::ops::DerefMut for VectorPainter<'_, '_> {
    fn deref_mut(&mut self) -> &mut PainterConfig {
        &mut self.config
    }
}

impl VectorPainter<'_, '_> {
    /// Reset painter state to defaults (identity transform, layer 0).
    pub fn reset(&mut self) {
        *self.config = PainterConfig::default();
    }

    /// (2x2 column-major linear, translation.xy, z) of the current transform,
    /// with `offset` applied in painter-local space.
    fn place(&self, offset: Vec2) -> ([f32; 4], [f32; 2], f32) {
        let m = self.config.transform.to_matrix();
        let linear = [m.x_axis.x, m.x_axis.y, m.y_axis.x, m.y_axis.y];
        let t = Vec2::new(
            m.w_axis.x + linear[0] * offset.x + linear[2] * offset.y,
            m.w_axis.y + linear[1] * offset.x + linear[3] * offset.y,
        );
        (linear, [t.x, t.y], m.w_axis.z)
    }

    fn push(&mut self, offset: Vec2, commands: Vec<PathCommand>, style: PathStyle) {
        let (linear, translation, z) = self.place(offset);
        self.queue.shapes.push(PaintedShape {
            commands,
            style,
            linear,
            translation,
            z,
            layers: self.config.render_layers.clone(),
        });
    }

    fn stroke_style(&self, brush: impl Into<Brush>, width: f32) -> StrokeStyle {
        StrokeStyle {
            join: self.config.join,
            cap: self.config.cap,
            ..StrokeStyle::new(brush, width)
        }
    }

    /// Paint arbitrary path commands with an arbitrary style — the full
    /// engine feature set (gradients, dashes, fill rules) in immediate mode.
    /// Author the path in local space around the origin and position with
    /// `offset` so repeated frames share cached geometry.
    pub fn shape(&mut self, offset: Vec2, commands: Vec<PathCommand>, style: PathStyle) {
        self.push(offset, commands, style);
    }

    pub fn fill_path(&mut self, offset: Vec2, commands: Vec<PathCommand>, color: Color) {
        self.push(offset, commands, PathStyle::fill(color.to_linear()));
    }

    pub fn stroke_path(
        &mut self,
        offset: Vec2,
        commands: Vec<PathCommand>,
        width: f32,
        color: Color,
    ) {
        let stroke = self.stroke_style(color.to_linear(), width);
        self.push(offset, commands, PathStyle::stroke(stroke));
    }

    /// Filled rectangle centered on `center` (painter-local).
    pub fn fill_rect(&mut self, center: Vec2, size: Vec2, color: Color) {
        self.push(center, rect_commands(size), PathStyle::fill(color.to_linear()));
    }

    /// Rectangle outline centered on `center`, stroked at `width`.
    pub fn stroke_rect(&mut self, center: Vec2, size: Vec2, width: f32, color: Color) {
        let stroke = self.stroke_style(color.to_linear(), width);
        self.push(center, rect_commands(size), PathStyle::stroke(stroke));
    }

    pub fn fill_circle(&mut self, center: Vec2, radius: f32, color: Color) {
        self.push(center, circle_commands(radius), PathStyle::fill(color.to_linear()));
    }

    pub fn stroke_circle(&mut self, center: Vec2, radius: f32, width: f32, color: Color) {
        let stroke = self.stroke_style(color.to_linear(), width);
        self.push(center, circle_commands(radius), PathStyle::stroke(stroke));
    }

    /// Filled triangle with painter-local vertices.
    pub fn fill_triangle(&mut self, a: Vec2, b: Vec2, c: Vec2, color: Color) {
        // Author around the centroid so congruent triangles share geometry.
        let centroid = (a + b + c) / 3.0;
        let commands = vec![
            PathCommand::MoveTo(a - centroid),
            PathCommand::LineTo(b - centroid),
            PathCommand::LineTo(c - centroid),
            PathCommand::Close,
        ];
        self.push(centroid, commands, PathStyle::fill(color.to_linear()));
    }

    /// Straight segment from `start` to `end` stroked at `width` with the
    /// painter's current cap.
    pub fn line(&mut self, start: Vec2, end: Vec2, width: f32, color: Color) {
        // Author around the midpoint: all equal-length parallel segments
        // (tick marks, scanlines) collapse to shared cached geometry.
        let mid = (start + end) * 0.5;
        let commands = vec![
            PathCommand::MoveTo(start - mid),
            PathCommand::LineTo(end - mid),
        ];
        let stroke = self.stroke_style(color.to_linear(), width);
        self.push(mid, commands, PathStyle::stroke(stroke));
    }

    /// Ring segment (the parametric fast path — no tessellation, one
    /// instance write). Angles in radians, counter-clockwise from +X;
    /// negative `sweep` draws clockwise.
    pub fn arc(
        &mut self,
        center: Vec2,
        inner: f32,
        outer: f32,
        start: f32,
        sweep: f32,
        color: Color,
    ) {
        let (linear, translation, z) = self.place(center);
        self.queue.arcs.push(PaintedArc {
            inner,
            outer,
            start,
            sweep,
            color: color.to_linear(),
            linear,
            translation,
            z,
            layers: self.config.render_layers.clone(),
        });
    }
}

fn rect_commands(size: Vec2) -> Vec<PathCommand> {
    let h = size * 0.5;
    vec![
        PathCommand::MoveTo(Vec2::new(-h.x, -h.y)),
        PathCommand::LineTo(Vec2::new(h.x, -h.y)),
        PathCommand::LineTo(Vec2::new(h.x, h.y)),
        PathCommand::LineTo(Vec2::new(-h.x, h.y)),
        PathCommand::Close,
    ]
}

fn circle_commands(radius: f32) -> Vec<PathCommand> {
    // Standard 4-cubic circle approximation; tessellation tolerance takes it
    // the rest of the way.
    let k = 0.552_285 * radius;
    let r = radius;
    vec![
        PathCommand::MoveTo(Vec2::new(r, 0.0)),
        PathCommand::CubicTo { ctrl1: Vec2::new(r, k), ctrl2: Vec2::new(k, r), to: Vec2::new(0.0, r) },
        PathCommand::CubicTo { ctrl1: Vec2::new(-k, r), ctrl2: Vec2::new(-r, k), to: Vec2::new(-r, 0.0) },
        PathCommand::CubicTo { ctrl1: Vec2::new(-r, -k), ctrl2: Vec2::new(-k, -r), to: Vec2::new(0.0, -r) },
        PathCommand::CubicTo { ctrl1: Vec2::new(k, -r), ctrl2: Vec2::new(r, -k), to: Vec2::new(r, 0.0) },
        PathCommand::Close,
    ]
}
