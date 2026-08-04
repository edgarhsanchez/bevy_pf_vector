# Context for Claude Code

Carried over from a claude.ai planning session (2026-08-03), then revised in
Claude Code sessions the same day. This file is the handoff; keep it current.

## What this project is

An original vector/UI rendering engine for `bevy_pf`, targeting Bevy 0.19.
Not a port or binding of any existing renderer. It exists because the HUD
workload has a property general-purpose renderers can't exploit:
mostly-static topology, so tessellate once and GPU-instance every frame.

## Decisions already made — do not re-litigate without new evidence

1. **"Better" is a benchmark claim, scoped to bevy_pf's workloads.** The
   engine must beat bevy_vector_shapes, rive-bevy/Vello, and other opponents
   on the suite in ARCHITECTURE.md §4 (frame-time and GPU-time percentiles).
   Beating Skia/Vello/Rive on *their* general workloads (arbitrary dynamic
   vector art, documents) was rejected as unachievable — if bevy_pf needs a
   workload the suite doesn't cover, add it to the suite rather than widening
   the engine's mission.
2. **Tessellate-once-and-instance is the engine.** Lyon/kurbo at asset load,
   GPU instancing per frame. Portable, no exotic GPU features.
3. **The engine stays pure (user decision, 2026-08-03).** No third-party
   renderer bindings in the engine crate. The former "path A" (rive native
   interop via wgpu-hal/ash + C++ shim) was REMOVED — ffi.rs deleted, the
   `rive-native` feature and wgpu-hal/ash deps dropped. Competing renderers
   appear only as vendored benchmark opponents under `benchmarks/vendor/`.
4. **Compute rasterization is rejected** — that's Vello's territory and a
   multi-year effort.
5. **No CUDA.** Vendor-locked, buys nothing for vector rasterization over
   compute through wgpu.
6. **Benchmarks before renderer claims.** The harness was built and validated
   first; every performance claim must come from it. Benchmark code, results,
   and vendor pins live in `benchmarks/`, never in the engine crate.

## Layout

- `bevy_pf_vector/` — the engine crate (deps: bevy, lyon, kurbo — nothing else)
- `benchmarks/` — harness crate, `results/`, `vendor-upstream.sh`
- `../bevy_pf_vector_testbed` — sibling repo: integration proving ground
  consuming this crate as a path dependency. bevy_pf itself has no local
  checkout yet; when it lands, the testbed is where integration happens.

## Verified 2026-08-03 (Claude Code session)

- Dependencies resolve: bevy 0.19.0, bevy_vector_shapes 0.13.1 (targets bevy
  0.19 / wgpu 29). Workspace compiles clean.
- **Bevy 0.19 removed the Node-based render graph.** `RenderGraph` is now a
  `ScheduleLabel` (`bevy::render::renderer::RenderGraph`) with
  `RenderGraphSystems::{Begin, Render, Submit, Finish}`; passes are plain ECS
  systems. Camera rendering runs per-camera schedules (`Core2d`/`Core3d` in
  bevy_core_pipeline) via `camera_driver`. Custom passes push encoders into
  `PendingCommandBuffers` before `Submit`. node.rs reflects this.
- Harness (ARCHITECTURE.md §5) is DONE and validated: GPU timestamp queries +
  pipeline statistics report on desktop Vulkan (RTX A6000); separates load
  (200 vs 5000 elements: 38x GPU time) and backend (shapes vs sprites at 200
  elements: non-overlapping distributions, identical primitive counts).
  Bar to beat for workload 1: shapes control ~0.023 ms GPU / 200 elements.

## Engine status (2026-08-03)

Implemented, proven in the testbed (engine output matches gizmo ground
truth; single-sample AA verified by screenshot), and measured. Workload 1,
RTX A6000/Vulkan, p50: 200 elements 0.0082 ms GPU (control 0.0225 = 2.7x);
5000 elements 0.137 ms GPU (control 0.932 = 6.8x), pass-record CPU
0.0019 ms (control 0.322). Design points, all portable wgpu, no code taken
from any reference renderer:

- tessellate-once keyed by content hash; exact-coverage triangles (~4x
  fewer fragment invocations than SDF quads)
- analytic AA: mesh-boundary fringe (boundary edges extracted from the
  tessellation with signed-area orientation correction, averaged outward
  vertex normals) extruded one screen pixel in the vertex shader —
  single-sample rendering, no MSAA; engine cameras use Msaa::Off
- opaque/blend split — opaque interiors draw with early-z depth write
  grouped by geometry (order-free under depth), translucent interiors +
  all fringes blend back-to-front
- 36-byte instances (2x2 affine + translation + z + RGBA8); one
  multi_draw_indexed_indirect per phase when INDIRECT_FIRST_INSTANCE is
  available (runtime-detected), draw loop otherwise

## Opponent comparisons (workload 1, RTX A6000/Vulkan, p50)

- vello 0.9 measured IN-PROCESS (shared wgpu 29 device, per-frame scene
  encode from retained BezPaths, GPU bracketed with timestamp spans around
  its internal submit, composited via fullscreen sprite): 0.816 ms @200 el,
  1.493 ms @5000 el, plus 0.4-1.1 ms/frame CPU scene encode. Engine wins
  ~100x @200, ~11x @5000. Vello's ~0.8 ms floor is canvas-sized compute —
  the per-frame cost thesis, measured. Screenshot-verified identical
  workload (`--backend vello --screenshot`).
- Native Rive Renderer: NOT measurable in-process (rive-bevy pins old Bevy
  and renders via vello anyway; native renderer would need a standalone
  harness — the in-engine interop path was removed by the purity decision).
- Skia / Pathfinder: unmeasured, would need standalone harnesses.

## Next tasks

- Workloads 2-4 (animated params, clip stress, stroke stress) in the
  benchmark suite — vello backend already wired as the opponent.
- Style-change handling (VectorShape color changes without re-tessellation
  — currently color is per-instance so it works, but path edits leak old
  geometry in the cache; add eviction).
- AMD / Intel / Apple-Metal measurement passes when hardware is available;
  "fastest" claims are NVIDIA/Vulkan-only until then.
- bevy_pf integration in the sibling testbed once bevy_pf lands locally.
