//! bevy_pf_vector — vector/UI rendering for bevy_pf.
//!
//! Read ARCHITECTURE.md before extending this. In particular §1 (what this can
//! and cannot beat) and §2 (why there is no pure-wgpu path to Rive's fast path).

pub mod backend;
pub mod node;
#[cfg(feature = "rive-native")]
pub mod ffi;

use bevy::prelude::*;

pub struct PfVectorPlugin {
    pub backend: BackendChoice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    /// Path C — tessellate once, instance. Portable. Start here.
    Tessellated,
    /// Path A — share the native device with rive-runtime. Desktop only.
    RiveNative,
}

impl Default for PfVectorPlugin {
    fn default() -> Self {
        Self { backend: BackendChoice::Tessellated }
    }
}

impl Plugin for PfVectorPlugin {
    fn build(&self, _app: &mut App) {
        // TODO: register assets, add node::vector_pass to the RenderGraph
        // schedule (RenderGraphSystems::Render, after camera_driver), install
        // the selected backend as a render-world resource.
    }
}
