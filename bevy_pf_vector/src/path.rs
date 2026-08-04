//! Path authoring types: outlines in verb form (what lyon/kurbo consume)
//! plus fill/stroke styling. These are the data half of the public API;
//! rendering lives in `render.rs`.

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
    /// Dash pattern as (on, off) lengths in local units. Dashed strokes are
    /// expanded via kurbo and fill-tessellated — still tessellate-once.
    pub dash: Option<[f32; 2]>,
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
