//! bevy_pf_vector — a vector/UI rendering engine for bevy_pf.
//!
//! Original engine, one narrow bet: HUD content has mostly-static topology,
//! so paths are tessellated once (lyon) and redrawn as GPU-instanced geometry.
//! Read ARCHITECTURE.md before extending this.

pub mod path;
pub mod render;
pub mod tess;

use bevy::prelude::*;

pub use path::{LineCap, LineJoin, PathCommand, PathStyle, StrokeStyle};

/// Flat 2D transform for HUD elements. Use INSTEAD of animating `Transform`:
/// entities animated through this component never dirty the transform
/// hierarchy, so Bevy's propagation systems have nothing to do — the last
/// engine-controllable frame cost for large animated HUDs. Entities without
/// a parent/children relationship should prefer it; `Transform`-driven
/// entities (hierarchies like a compass with tick children) keep working
/// unchanged, and when both are present this one wins.
#[derive(Component, Clone, Copy, Debug)]
pub struct HudTransform {
    pub translation: Vec3,
    /// Rotation around Z in radians.
    pub rotation: f32,
    pub scale: Vec2,
}

impl Default for HudTransform {
    fn default() -> Self {
        Self { translation: Vec3::ZERO, rotation: 0.0, scale: Vec2::ONE }
    }
}

impl HudTransform {
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self { translation: Vec3::new(x, y, z), ..Default::default() }
    }

    /// (2x2 column-major linear part, translation.xy, z) for extraction.
    pub(crate) fn decompose(&self) -> ([f32; 4], [f32; 2], f32) {
        let (sin, cos) = bevy::math::ops::sin_cos(self.rotation);
        (
            [
                cos * self.scale.x,
                sin * self.scale.x,
                -sin * self.scale.y,
                cos * self.scale.y,
            ],
            [self.translation.x, self.translation.y],
            self.translation.z,
        )
    }
}

/// A vector shape authored as a path outline plus style. Topology is treated
/// as static: the path is tessellated once when the component is added; if
/// the component is mutated, the shape re-tessellates that frame (priced per
/// changed shape). For continuously parameter-animated primitives, prefer
/// [`VectorPrimitive`] — those never tessellate at all.
#[derive(Component, Clone, Debug)]
#[require(Transform)]
pub struct VectorShape {
    pub commands: Vec<PathCommand>,
    pub style: PathStyle,
}

/// Parametric primitives evaluated entirely in the vertex shader from a
/// canonical mesh — animating their parameters costs one instance write per
/// frame, no tessellation, and all instances of a primitive kind draw in a
/// single instanced call. The gauge/meter fast path.
#[derive(Component, Clone, Copy, Debug)]
#[require(Transform)]
pub enum VectorPrimitive {
    /// Ring segment. Angles in radians, y-up, counter-clockwise.
    Arc {
        inner: f32,
        outer: f32,
        start: f32,
        sweep: f32,
        color: LinearRgba,
    },
}

/// A clip region. Entities with this component don't render; content
/// references them via [`ClippedBy`]. Clips nest by putting `ClippedBy` on a
/// clip entity itself (up to 4 levels). Evaluated analytically in the
/// fragment shader — clip edges are antialiased and clipping costs no extra
/// draw calls, state changes, or stencil passes.
#[derive(Component, Clone, Copy, Debug)]
#[require(Transform)]
pub enum VectorClipShape {
    RoundedRect { half_extents: Vec2, radius: f32 },
    Circle { radius: f32 },
}

/// Clips the entity's rendering to the referenced [`VectorClipShape`] entity
/// (and that clip's own ancestors, if it is itself clipped).
#[derive(Component, Clone, Copy, Debug)]
pub struct ClippedBy(pub Entity);

pub struct PfVectorPlugin;

impl Plugin for PfVectorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(render::VectorRenderPlugin);
    }
}
