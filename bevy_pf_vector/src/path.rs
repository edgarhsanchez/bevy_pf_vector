//! Path authoring types: outlines in verb form (what lyon/kurbo consume)
//! plus fill/stroke styling with full brush support. These are the data half
//! of the public API; rendering lives in `render.rs`.

use bevy::prelude::*;

/// Path outline in the usual verb form, matching what lyon/kurbo consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathCommand {
    MoveTo(Vec2),
    LineTo(Vec2),
    QuadTo { ctrl: Vec2, to: Vec2 },
    CubicTo { ctrl1: Vec2, ctrl2: Vec2, to: Vec2 },
    Close,
}

/// A gradient stop: offset in 0..1 plus color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientStop {
    pub offset: f32,
    pub color: LinearRgba,
}

/// Paint for fills and strokes. Gradients are multi-stop, defined in the
/// shape's local space, baked once into a GPU lookup atlas (keyed by stop
/// content), and evaluated per fragment — a gradient costs the same instance
/// write as a solid color once its row exists.
#[derive(Clone, Debug, PartialEq)]
pub enum Brush {
    Solid(LinearRgba),
    Linear {
        start: Vec2,
        end: Vec2,
        stops: Vec<GradientStop>,
    },
    Radial {
        center: Vec2,
        radius: f32,
        stops: Vec<GradientStop>,
    },
}

impl From<LinearRgba> for Brush {
    fn from(color: LinearRgba) -> Self {
        Brush::Solid(color)
    }
}

impl Brush {
    /// Whether every texel this brush produces is fully opaque.
    pub fn is_opaque(&self) -> bool {
        match self {
            Brush::Solid(c) => c.alpha >= 1.0,
            Brush::Linear { stops, .. } | Brush::Radial { stops, .. } => {
                stops.iter().all(|s| s.color.alpha >= 1.0)
            }
        }
    }
}

/// How self-intersecting fill regions resolve. `NonZero` is the common
/// vector-graphics default; WPF/XAML defaults to `EvenOdd`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillRule {
    #[default]
    NonZero,
    EvenOdd,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathStyle {
    pub fill: Option<Brush>,
    pub stroke: Option<StrokeStyle>,
    pub fill_rule: FillRule,
}

impl PathStyle {
    pub fn fill(brush: impl Into<Brush>) -> Self {
        Self { fill: Some(brush.into()), stroke: None, fill_rule: FillRule::default() }
    }

    pub fn stroke(stroke: StrokeStyle) -> Self {
        Self { fill: None, stroke: Some(stroke), fill_rule: FillRule::default() }
    }
}

/// Arbitrary-length dash pattern (WPF-style) plus phase offset, in local
/// units. Expanded at tessellation time via kurbo — still tessellate-once.
#[derive(Clone, Debug, PartialEq)]
pub struct DashPattern {
    pub pattern: Vec<f32>,
    pub offset: f32,
}

impl DashPattern {
    pub fn new(pattern: impl Into<Vec<f32>>) -> Self {
        Self { pattern: pattern.into(), offset: 0.0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrokeStyle {
    pub brush: Brush,
    pub width: f32,
    pub join: LineJoin,
    pub cap: LineCap,
    /// Sharp-corner cutoff for miter joins, as a ratio of stroke width.
    pub miter_limit: f32,
    pub dash: Option<DashPattern>,
}

impl StrokeStyle {
    pub fn new(brush: impl Into<Brush>, width: f32) -> Self {
        Self {
            brush: brush.into(),
            width,
            join: LineJoin::Miter,
            cap: LineCap::Butt,
            miter_limit: 4.0,
            dash: None,
        }
    }
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
