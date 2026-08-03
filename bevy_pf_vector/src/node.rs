//! Render-graph entry point.
//!
//! Bevy 0.19's render graph is the [`RenderGraph`] ECS *schedule*
//! (`bevy_render::renderer::RenderGraph`); what used to be a graph node is a
//! plain system ordered via [`RenderGraphSystems`]. Camera passes run inside
//! `camera_driver` (bevy_core_pipeline), which executes each camera's own
//! schedule (`Core2d` etc.); one-off work like ours is ordered before/after it
//! in the root schedule's `Render` set, pushing finished command buffers into
//! [`PendingCommandBuffers`] before the `Submit` set runs.

use bevy::ecs::world::World;

// Re-exported so the plugin can `add_systems(RenderGraph, vector_pass.in_set(...))`
// without reaching into bevy_render paths itself.
pub use bevy::render::renderer::{PendingCommandBuffers, RenderGraph, RenderGraphSystems};

/// Records the vector pass for this frame. Added to the [`RenderGraph`]
/// schedule in [`RenderGraphSystems::Render`], after `camera_driver`.
pub fn vector_pass(_world: &mut World) {
    // TODO(engine): fetch the installed VectorBackend resource, build a
    // FrameScene from extracted instances, record onto a fresh encoder,
    // push it into PendingCommandBuffers.
}
