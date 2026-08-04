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
use bevy::window::PrimaryWindow;

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

    // --- stateful mode -----------------------------------------------------
    // The retained methods below (`fill_rect`, `stroke_circle`, ...) take
    // their paint explicitly. These fields drive the ALTERNATIVE stateful
    // primitives (`rect`, `circle`, `ngon`, ...), which exist so a codebase
    // written against an immediate-mode painter of the
    // set-state-then-draw kind ports by swapping the parameter type rather
    // than rewriting every call site.
    /// Paint for the stateful primitives.
    pub color: Color,
    /// `true` strokes the outline, `false` fills.
    pub hollow: bool,
    /// Stroke width for the stateful primitives, in physical screen pixels
    /// (matching the convention HUD code written against `ThicknessType::Pixels`
    /// already assumes).
    pub thickness: f32,
}

impl Default for PainterConfig {
    fn default() -> Self {
        Self {
            transform: Transform::IDENTITY,
            render_layers: None,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            color: Color::WHITE,
            hollow: false,
            thickness: 1.0,
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
    windows: Query<'w, 's, &'static Window, With<PrimaryWindow>>,
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

    /// World units spanning `pixels` physical screen pixels, for the
    /// standard 2D HUD setup (one world unit per logical pixel — Bevy's
    /// default `ScalingMode::WindowSize`).
    ///
    /// Stroke widths and everything else the painter takes are in world
    /// units, which is what keeps a HUD's proportions stable across
    /// resolutions. Use this only where a feature must be a fixed number of
    /// device pixels regardless of DPI — hairline rules, 1px separators —
    /// the equivalent of `bevy_vector_shapes`' `ThicknessType::Pixels`.
    pub fn screen_px(&self, pixels: f32) -> f32 {
        let scale = self
            .windows
            .iter()
            .next()
            .map(|window| window.scale_factor())
            .unwrap_or(1.0);
        pixels / scale.max(1.0e-6)
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

    // ---------------------------------------------------------------- state
    // Stateful primitives: position and paint come from the config, so a
    // caller does `set_translation(..); color = ..; rect(size)`.

    /// Stateful arc: a ring of the current `thickness` centred on `radius`,
    /// swept from `start` to `end` (radians, clockwise from +Y — the
    /// convention the HUD code this mirrors uses).
    pub fn arc(&mut self, radius: f32, start: f32, end: f32) {
        let (color, _, thickness) = self.paint();
        let inner = (radius - thickness * 0.5).max(0.0);
        let outer = radius + thickness * 0.5;
        let engine_start = std::f32::consts::FRAC_PI_2 - start;
        self.ring(Vec2::ZERO, inner, outer, engine_start, start - end, color);
    }

    /// No-op: this painter is always 2D. Present so code written against a
    /// painter that supports both modes ports without edits.
    pub fn set_2d(&mut self) {}

    /// Move the painter's origin.
    pub fn set_translation(&mut self, translation: Vec3) {
        self.config.transform.translation = translation;
    }

    /// Set the painter's rotation. Only the Z component affects 2D output.
    pub fn set_rotation(&mut self, rotation: Quat) {
        self.config.transform.rotation = rotation;
    }

    /// Rotate around Z, in radians.
    pub fn set_rotation_z(&mut self, radians: f32) {
        self.config.transform.rotation = Quat::from_rotation_z(radians);
    }

    pub fn set_color(&mut self, color: Color) {
        self.config.color = color;
    }

    /// Rectangle centred on the painter origin, filled or stroked per
    /// `hollow`.
    pub fn rect(&mut self, size: Vec2) {
        let (color, hollow, thickness) = self.paint();
        if hollow {
            self.stroke_rect(Vec2::ZERO, size, thickness, color);
        } else {
            self.fill_rect(Vec2::ZERO, size, color);
        }
    }

    /// Circle centred on the painter origin.
    pub fn circle(&mut self, radius: f32) {
        let (color, hollow, thickness) = self.paint();
        if hollow {
            self.stroke_circle(Vec2::ZERO, radius, thickness, color);
        } else {
            self.fill_circle(Vec2::ZERO, radius, color);
        }
    }

    /// Regular polygon with `sides` sides, centred on the painter origin.
    /// First vertex points +Y, matching the usual convention.
    pub fn ngon(&mut self, sides: f32, radius: f32) {
        let n = (sides.round() as usize).max(3);
        let commands = ngon_commands(n, radius);
        let (color, hollow, thickness) = self.paint();
        if hollow {
            let stroke = StrokeStyle { width: thickness, ..self.stroke_style(color.to_linear(), thickness) };
            self.push(Vec2::ZERO, commands, PathStyle::stroke(stroke));
        } else {
            self.push(Vec2::ZERO, commands, PathStyle::fill(color.to_linear()));
        }
    }

    /// Triangle with painter-local vertices, filled or stroked per `hollow`.
    pub fn triangle(&mut self, a: Vec2, b: Vec2, c: Vec2) {
        let (color, hollow, thickness) = self.paint();
        if hollow {
            let centroid = (a + b + c) / 3.0;
            let commands = vec![
                PathCommand::MoveTo(a - centroid),
                PathCommand::LineTo(b - centroid),
                PathCommand::LineTo(c - centroid),
                PathCommand::Close,
            ];
            let stroke = self.stroke_style(color.to_linear(), thickness);
            self.push(centroid, commands, PathStyle::stroke(stroke));
        } else {
            self.fill_triangle(a, b, c, color);
        }
    }

    /// Stateful line between two painter-local points. Takes `Vec3` because
    /// the callers this exists for work in 3D HUD space; z is ignored beyond
    /// the painter's own depth.
    pub fn line_3d(&mut self, start: Vec3, end: Vec3) {
        let (color, _, thickness) = self.paint();
        self.line(start.truncate(), end.truncate(), thickness, color);
    }

    /// Current paint, with thickness converted from screen pixels to the
    /// world units the geometry is authored in.
    fn paint(&self) -> (Color, bool, f32) {
        (
            self.config.color,
            self.config.hollow,
            self.screen_px(self.config.thickness),
        )
    }

    /// Ring segment (the parametric fast path — no tessellation, one
    /// instance write). Angles in radians, counter-clockwise from +X;
    /// negative `sweep` draws clockwise.
    pub fn ring(
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

fn ngon_commands(sides: usize, radius: f32) -> Vec<PathCommand> {
    let mut commands = Vec::with_capacity(sides + 2);
    for i in 0..sides {
        // Start at +Y and go clockwise, which is what regular-polygon HUD
        // markers (hexes, triangles) are drawn assuming.
        let a = std::f32::consts::FRAC_PI_2 - (i as f32 / sides as f32) * std::f32::consts::TAU;
        let p = Vec2::new(a.cos() * radius, a.sin() * radius);
        if i == 0 {
            commands.push(PathCommand::MoveTo(p));
        } else {
            commands.push(PathCommand::LineTo(p));
        }
    }
    commands.push(PathCommand::Close);
    commands
}
