//! Render-graph entry point.
//!
//! Bevy 0.19 removed the `Node` trait: the render graph is now the
//! [`RenderGraph`] ECS *schedule* (`bevy_render::renderer::RenderGraph`), and
//! what used to be a graph node is now a plain system ordered via
//! [`RenderGraphSystems`]. Camera passes run inside `camera_driver`
//! (bevy_core_pipeline), which executes each camera's own schedule (`Core2d`
//! etc.); one-off work like ours is ordered before/after it in the root
//! schedule's `Render` set.
//!
//! The barrier discipline is still the part that will bite on the
//! native-interop path: rive submits on the shared queue, and Bevy must not
//! sample the target texture before that submission completes. In 0.19 terms:
//! the vector system must run in `RenderGraphSystems::Render` *after*
//! `camera_driver`, push its encoder into `PendingCommandBuffers` (tessellated
//! path) or submit directly on the raw queue with an image barrier (native
//! path, `owns_submission() == true`) before `RenderGraphSystems::Submit`.

use bevy::ecs::world::World;

// Re-exported so the plugin can `add_systems(RenderGraph, vector_pass.in_set(...))`
// without reaching into bevy_render paths itself.
pub use bevy::render::renderer::{PendingCommandBuffers, RenderGraph, RenderGraphSystems};

/// Records the vector pass for this frame. Added to the [`RenderGraph`]
/// schedule in [`RenderGraphSystems::Render`], after `camera_driver`.
pub fn vector_pass(_world: &mut World) {
    // TODO(path C): fetch the installed VectorBackend resource, build a
    //   FrameScene from extracted instances, record onto a fresh encoder,
    //   push it into PendingCommandBuffers.
    // TODO(path A): flush pending buffers first, call rive
    //   RenderContext::flush() on the shared VkQueue, insert an image
    //   barrier, then let RenderGraphSystems::Submit proceed.
    //   See ARCHITECTURE.md §2.
}
