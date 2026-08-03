# Context for Claude Code

Carried over from a claude.ai planning session (2026-08-03). This file is the
handoff; there was no transcript import, so what mattered is written down here.

## What this project is

A vector/UI renderer for `bevy_pf`, targeting Bevy 0.19. It exists because
`rive-bevy` is not viable to depend on (see below), and because the HUD workload
has a property general-purpose renderers can't exploit.

## Decisions already made — do not re-litigate without new evidence

1. **Not writing a general-purpose renderer.** Beating Skia/Vello/Rive broadly is
   not the goal and was explicitly rejected as unachievable. The target is one
   narrow workload: game HUD with mostly-static topology and high instance counts.
2. **Tessellate-once-and-instance (path C) is the primary approach.** Lyon/kurbo
   at asset load, GPU instancing per frame. Portable, no exotic GPU features.
3. **Native rive interop (path A) is the secondary approach**, for dynamic
   strokes, feathering, and heavy clipping that path C can't cover.
4. **Compute rasterization (path B) is rejected** — that's Vello's territory and
   a multi-year effort.
5. **No CUDA.** Vendor-locked, buys nothing for vector rasterization over compute
   through wgpu. DLSS is a separate, unrelated optional 3D upscaling path.
6. **The benchmark harness gets built before the renderer.** See ARCHITECTURE.md §4-5.

## The technical constraint that drives everything

The Rive Renderer's fast path requires fragment shader interlock
(`VK_EXT_fragment_shader_interlock`) or D3D rasterizer-ordered views. WebGPU has
had an open request for this since 2019 and wgpu does not expose it. Bevy is wgpu.
Therefore path A requires `wgpu-hal` `Device::as_hal::<Vulkan>()` to hand the raw
`VkDevice`/queue to rive's `RenderContext` — there is no pure-wgpu route.

## Why not just use rive-bevy

It renders through Vello, not the Rive Renderer, so Rive's performance argument
doesn't transfer. Known issues: gaps/overdraw at image-mesh triangle borders,
incorrect rendering at high clip counts, all strokes forced to round joins/caps.
Repo state as of 2026-08-03: 33 commits total, no published releases, 3
contributors. Demo-grade.

## Verified 2026-08-03 (Claude Code session)

- Dependencies resolve: bevy 0.19.0, bevy_vector_shapes 0.13.1 (targets bevy
  0.19 / wgpu 29). `wgpu-hal` pin corrected 27 → 29 to match.
- `bevy_pf_vector` compiles (`cargo check`, both default and `rive-native`
  feature sets). Missing seam types (GeometryId, PathCommand, PathStyle,
  BackendEncoder) were defined in backend.rs.
- **Bevy 0.19 removed the Node-based render graph.** `RenderGraph` is now a
  `ScheduleLabel` (`bevy::render::renderer::RenderGraph`) with
  `RenderGraphSystems::{Begin, Render, Submit, Finish}`; passes are plain ECS
  systems. Camera rendering runs per-camera schedules (`Core2d`/`Core3d` in
  bevy_core_pipeline) via `camera_driver`. Custom passes push encoders into
  `PendingCommandBuffers` before `Submit`. node.rs was rewritten accordingly;
  path A's barrier discipline now means "submit on the raw queue between
  Render and Submit".
- First task (harness, ARCHITECTURE.md §5) is DONE and validated: GPU
  timestamp queries + pipeline statistics report on desktop Vulkan (RTX
  A6000), and the harness separates both load (200 vs 5000 elements: 38x GPU
  time) and backend (shapes vs sprites at 200 elements: non-overlapping
  distributions, identical primitive counts). See README and `results/`.

## Unverified claims — check before relying on them

- rive-runtime's non-interlock fallback modes (atomic, MSAA) were described from
  architectural knowledge, NOT confirmed against the current tree. Verify in
  `vendor/rive-runtime` before planning the portable/WASM story around them.

## Next task

Workload 2 from ARCHITECTURE.md §4 (animated HUD, state-machine-driven
parameters) needs a definition that doesn't depend on rive; then begin path C:
lyon tessellation at asset load + GPU instancing behind `VectorBackend`,
measured against the `shapes` control in this harness. Vendoring
(`vendor-upstream.sh`) requires `git init` first — the directory is not yet a
git repository.
