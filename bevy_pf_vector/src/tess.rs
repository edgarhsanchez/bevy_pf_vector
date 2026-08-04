//! Path → triangle mesh, via lyon. Runs once per unique geometry (content
//! hash), never per frame — the engine's entire premise.

use std::hash::{Hash, Hasher};

use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, LineCap as LyonCap,
    LineJoin as LyonJoin, StrokeOptions, StrokeTessellator, StrokeVertex, VertexBuffers,
};

use crate::backend::{LineCap, LineJoin, PathCommand, StrokeStyle};

/// Max distance in local units between a curve and its flattened form.
/// Shapes are authored in pixel-scale units, so this is ~1/4 px.
pub const TOLERANCE: f32 = 0.25;

pub struct TessellatedGeometry {
    pub vertices: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

fn build_path(commands: &[PathCommand]) -> Path {
    let mut builder = Path::builder();
    let mut open = false;
    for command in commands {
        match *command {
            PathCommand::MoveTo(p) => {
                if open {
                    builder.end(false);
                }
                builder.begin(point(p.x, p.y));
                open = true;
            }
            PathCommand::LineTo(p) => {
                if open {
                    builder.line_to(point(p.x, p.y));
                }
            }
            PathCommand::QuadTo { ctrl, to } => {
                if open {
                    builder.quadratic_bezier_to(point(ctrl.x, ctrl.y), point(to.x, to.y));
                }
            }
            PathCommand::CubicTo { ctrl1, ctrl2, to } => {
                if open {
                    builder.cubic_bezier_to(
                        point(ctrl1.x, ctrl1.y),
                        point(ctrl2.x, ctrl2.y),
                        point(to.x, to.y),
                    );
                }
            }
            PathCommand::Close => {
                if open {
                    builder.end(true);
                    open = false;
                }
            }
        }
    }
    if open {
        builder.end(false);
    }
    builder.build()
}

pub fn tessellate_fill(commands: &[PathCommand]) -> Option<TessellatedGeometry> {
    let path = build_path(commands);
    let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    FillTessellator::new()
        .tessellate_path(
            &path,
            &FillOptions::tolerance(TOLERANCE),
            &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| v.position().to_array()),
        )
        .ok()?;
    (!buffers.indices.is_empty()).then_some(TessellatedGeometry {
        vertices: buffers.vertices,
        indices: buffers.indices,
    })
}

pub fn tessellate_stroke(
    commands: &[PathCommand],
    stroke: &StrokeStyle,
) -> Option<TessellatedGeometry> {
    let path = build_path(commands);
    let cap = match stroke.cap {
        LineCap::Butt => LyonCap::Butt,
        LineCap::Round => LyonCap::Round,
        LineCap::Square => LyonCap::Square,
    };
    let join = match stroke.join {
        LineJoin::Miter => LyonJoin::Miter,
        LineJoin::Round => LyonJoin::Round,
        LineJoin::Bevel => LyonJoin::Bevel,
    };
    let options = StrokeOptions::tolerance(TOLERANCE)
        .with_line_width(stroke.width)
        .with_start_cap(cap)
        .with_end_cap(cap)
        .with_line_join(join);
    let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    StrokeTessellator::new()
        .tessellate_path(
            &path,
            &options,
            &mut BuffersBuilder::new(&mut buffers, |v: StrokeVertex| v.position().to_array()),
        )
        .ok()?;
    (!buffers.indices.is_empty()).then_some(TessellatedGeometry {
        vertices: buffers.vertices,
        indices: buffers.indices,
    })
}

/// Content hash identifying a fill geometry. Instances of identical paths
/// share GPU geometry and draw instanced.
pub fn fill_key(commands: &[PathCommand]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    0u8.hash(&mut hasher);
    hash_commands(commands, &mut hasher);
    hasher.finish()
}

/// Content hash for a stroke geometry — width/join/cap change the mesh, so
/// they are part of the key.
pub fn stroke_key(commands: &[PathCommand], stroke: &StrokeStyle) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    1u8.hash(&mut hasher);
    stroke.width.to_bits().hash(&mut hasher);
    (stroke.join as u8).hash(&mut hasher);
    (stroke.cap as u8).hash(&mut hasher);
    hash_commands(commands, &mut hasher);
    hasher.finish()
}

fn hash_point(p: bevy::math::Vec2, hasher: &mut impl Hasher) {
    p.x.to_bits().hash(hasher);
    p.y.to_bits().hash(hasher);
}

fn hash_commands(commands: &[PathCommand], hasher: &mut impl Hasher) {
    for command in commands {
        match *command {
            PathCommand::MoveTo(p) => {
                hash_point(p, hasher);
                0u8.hash(hasher);
            }
            PathCommand::LineTo(p) => {
                hash_point(p, hasher);
                1u8.hash(hasher);
            }
            PathCommand::QuadTo { ctrl, to } => {
                hash_point(ctrl, hasher);
                hash_point(to, hasher);
                2u8.hash(hasher);
            }
            PathCommand::CubicTo { ctrl1, ctrl2, to } => {
                hash_point(ctrl1, hasher);
                hash_point(ctrl2, hasher);
                hash_point(to, hasher);
                3u8.hash(hasher);
            }
            PathCommand::Close => 4u8.hash(hasher),
        }
    }
}
