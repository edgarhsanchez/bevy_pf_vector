//! bevy_pf_vector — a vector/UI rendering engine for bevy_pf.
//!
//! Original engine, one narrow bet: HUD content has mostly-static topology,
//! so paths are tessellated once (lyon) and redrawn as GPU-instanced geometry.
//! Read ARCHITECTURE.md before extending this.

pub mod backend;
pub mod node;
pub mod render;
pub mod tess;

use bevy::prelude::*;

pub use backend::{
    GeometryId, LineCap, LineJoin, PathCommand, PathStyle, StrokeStyle, VectorBackend,
};

/// A vector shape authored as a path outline plus style. Topology is treated
/// as static: the path is tessellated once when the component is added; only
/// `Transform` and style parameters are expected to change per frame.
#[derive(Component, Clone, Debug)]
#[require(Transform)]
pub struct VectorShape {
    pub commands: Vec<PathCommand>,
    pub style: PathStyle,
}

pub struct PfVectorPlugin;

impl Plugin for PfVectorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(render::VectorRenderPlugin);
    }
}
