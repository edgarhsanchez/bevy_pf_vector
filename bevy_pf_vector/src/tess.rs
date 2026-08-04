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

/// A tessellated vertex: position, outward silhouette normal, and coverage.
/// Interior vertices have zero normal and coverage 1 — the vertex shader
/// displaces by `normal` in *screen space*, so interior vertices don't move.
/// Fringe ring vertices carry the outward normal and coverage 0.
#[derive(Clone, Copy)]
pub struct TessVertex {
    pub position: [f32; 2],
    pub normal: [f32; 2],
    pub coverage: f32,
}

/// Interior triangles plus a one-screen-pixel antialiasing fringe along the
/// mesh silhouette. The fringe is generated from the mesh itself (boundary
/// edges), so fills, strokes, and holes are handled uniformly.
pub struct TessellatedGeometry {
    pub vertices: Vec<TessVertex>,
    /// Full-coverage interior triangles.
    pub interior_indices: Vec<u32>,
    /// Alpha-ramp fringe triangles (always blended).
    pub fringe_indices: Vec<u32>,
}

/// Builds the AA fringe: finds boundary edges (owned by exactly one
/// triangle, orientation-corrected so they wind CCW around filled area),
/// accumulates per-vertex outward normals, then emits a ring of coverage-0
/// vertices and two triangles per boundary edge.
fn build_geometry(positions: Vec<[f32; 2]>, interior_indices: Vec<u32>) -> TessellatedGeometry {
    // foldhash, not SipHash — this runs for every tessellation.
    use bevy::platform::collections::HashMap;

    let mut boundary: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    for triangle in interior_indices.chunks_exact(3) {
        let (a, b, c) = (triangle[0], triangle[1], triangle[2]);
        let (pa, pb, pc) = (
            positions[a as usize],
            positions[b as usize],
            positions[c as usize],
        );
        // Signed area decides this triangle's winding; record edges so they
        // consistently run CCW around filled area regardless.
        let area = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]);
        let edges = if area >= 0.0 {
            [(a, b), (b, c), (c, a)]
        } else {
            [(b, a), (c, b), (a, c)]
        };
        for (from, to) in edges {
            let key = (from.min(to), from.max(to));
            // An edge shared by two triangles is interior — drop it.
            if boundary.remove(&key).is_none() {
                boundary.insert(key, (from, to));
            }
        }
    }

    // Outward normal per boundary vertex: average of adjacent edge normals.
    // For CCW winding the filled side is left of travel, so outward is the
    // right-hand perpendicular.
    let mut normals: HashMap<u32, [f32; 2]> = HashMap::new();
    for &(from, to) in boundary.values() {
        let (pf, pt) = (positions[from as usize], positions[to as usize]);
        let (dx, dy) = (pt[0] - pf[0], pt[1] - pf[1]);
        let length = (dx * dx + dy * dy).sqrt();
        if length <= f32::EPSILON {
            continue;
        }
        let outward = [dy / length, -dx / length];
        for index in [from, to] {
            let n = normals.entry(index).or_insert([0.0, 0.0]);
            n[0] += outward[0];
            n[1] += outward[1];
        }
    }

    let mut vertices: Vec<TessVertex> = positions
        .iter()
        .map(|&position| TessVertex { position, normal: [0.0, 0.0], coverage: 1.0 })
        .collect();

    // One displaced coverage-0 twin per boundary vertex.
    let mut ring: HashMap<u32, u32> = HashMap::new();
    let mut fringe_indices = Vec::with_capacity(boundary.len() * 6);
    for (&index, normal) in &normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        let normal = if length > 1.0e-3 {
            [normal[0] / length, normal[1] / length]
        } else {
            // Opposing edges cancelled (degenerate sliver) — no displacement.
            [0.0, 0.0]
        };
        let outer = vertices.len() as u32;
        vertices.push(TessVertex {
            position: positions[index as usize],
            normal,
            coverage: 0.0,
        });
        ring.insert(index, outer);
    }
    for &(from, to) in boundary.values() {
        let (Some(&outer_from), Some(&outer_to)) = (ring.get(&from), ring.get(&to)) else {
            continue;
        };
        fringe_indices.extend_from_slice(&[from, outer_from, to, to, outer_from, outer_to]);
    }

    TessellatedGeometry { vertices, interior_indices, fringe_indices }
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
    (!buffers.indices.is_empty()).then(|| build_geometry(buffers.vertices, buffers.indices))
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
    (!buffers.indices.is_empty()).then(|| build_geometry(buffers.vertices, buffers.indices))
}

/// foldhash — these keys are computed per shape per frame; SipHash is
/// measurably wasteful here.
pub(crate) fn fast_hasher() -> impl Hasher {
    use std::hash::BuildHasher;
    bevy::platform::hash::FixedState::default().build_hasher()
}

/// Content hash identifying a fill geometry. Instances of identical paths
/// share GPU geometry and draw instanced.
pub fn fill_key(commands: &[PathCommand]) -> u64 {
    let mut hasher = fast_hasher();
    0u8.hash(&mut hasher);
    hash_commands(commands, &mut hasher);
    hasher.finish()
}

/// Content hash for a stroke geometry — width/join/cap change the mesh, so
/// they are part of the key.
pub fn stroke_key(commands: &[PathCommand], stroke: &StrokeStyle) -> u64 {
    let mut hasher = fast_hasher();
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
