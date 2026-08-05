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
  vertex normals) forming a HALF-pixel band on each side of the authored
  edge — the full-coverage vertex insets 0.5px, its coverage-0 twin
  extrudes 0.5px, via `(0.5 - coverage)` in the vertex shader; the ramp
  centers on the edge and covered area is preserved. Single-sample
  rendering, no MSAA; engine cameras use Msaa::Off
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
- Native Rive Renderer (C++): MEASURED via a standalone harness — built
  rive-runtime at the pinned SHA on Windows (clang/lld + ninja via their
  build_rive.sh under Git Bash + vcvars + Vulkan SDK; premake needs vswhere
  on PATH and GLFW needs user32/gdi32/shell32 added for the lld link line),
  with `benchmarks/rive-native-bench.patch` adding a --bench N mode to
  path_fiddle that ports the harness workload generator exactly (same
  SplitMix64 stream). Vulkan interlock fast path active, 2560x1440,
  IMMEDIATE present. Results: 0.204 ms/frame @200 el, 1.716 ms @5000 el
  (render-only loop). Engine renderer cost ~0.015/~0.19 ms → ~10x faster
  at both scales. Fallback modes verified to exist in source
  (InterlockMode{rasterOrdering, atomics, clockwise, clockwiseAtomic,
  msaa}, renderer/include/rive/renderer/gpu.hpp:743) — closes the old
  "unverified claims" item. Vendor tree is gitignored; the patch file is
  the reproducibility artifact.
- Skia / Pathfinder: unmeasured, would need standalone harnesses.

## Workload 4: WON (2026-08-03) — suite complete

StrokeStyle.dash + kurbo stroke expansion (dash/join/cap correct) ->
fill-tessellated once; undashed strokes stay on lyon. 300 strokes p50:
engine 0.81 ms frame / 0.0236 GPU; Vello 1.33 / 0.833; bevy_vector_shapes cannot
run it. All four ARCHITECTURE workloads + overlap stress now measured
and won on every backend able to compete. Remaining frontier: rive
native on W2-4, Skia/Pathfinder harnesses, AMD/Intel/Metal.

## Workload 3: WON (2026-08-03)

Analytic clip chains: VectorClipShape (rounded-rect/circle) entities +
ClippedBy(entity) references, nested up to 4; extract resolves chains into
a fixed 1024-entry storage buffer of inverse transforms + SDF params; each
instance packs (chain_start*8+count) into its spare instance-lane float;
fragment multiplies AA'd SDF coverage per entry (straight-line unrolled, no
loop). Clipped instances route to the blend phase (opaque REPLACE would
ignore the alpha knock-out — that was bug #1). Results (12 panels + outer,
240 shapes, p50): engine 0.78 ms frame / 0.0266 ms GPU; Vello (native clip
layers, chain-diff push/pop encoding in the backend) 1.11 / 0.953;
bevy_vector_shapes CANNOT run it (no clipping). ~36x GPU vs vello.

Bug worth remembering: view.viewport is (origin.xy, size.zw) —
`viewport.x` is ORIGIN (0!), width is `.z`. Using .x made px_world
infinite: every clip SDF flattened to coverage 0.5 AND fringe AA had been
silently degenerate since the AA milestone (fringes displaced to infinity
= no AA). Found by screenshot-driven shader bisection (dump cov, dump raw
buffer words, dump varyings). Post-fix honest numbers: W1 200 el
0.0143 ms GPU (was 0.0092 pre-fix/broken-AA), 5000 el 0.200 (was 0.137),
W2 0.0155 (was 0.0102) — all wins hold. clips_raw is read as
array<vec4<f32>> reconstructed per entry (typed-struct storage reads also
misbehaved during debugging; raw reads are unambiguous).

## Workload 2: WON (2026-08-03)

VectorPrimitive::Arc (parametric instanced primitives): canonical (t, side)
strip mesh, vertex shader computes ring-segment geometry + analytic AA
fringes from per-instance [start, sweep, inner, outer]. Animating = one
52-byte instance write; all arcs draw in one instanced call per phase.
Workload 2 (200 el / 50 dynamic, p50): engine 0.73 ms frame / 0.0102 GPU
vs control 0.99 / 0.0215 — wins both. Draw-order caveat: parametric blend
items draw after tessellated blend items (1px fringes, negligible).
Next primitive candidates when bevy_pf needs them: rounded-rect/bar, pie.

## Workload 2 history (2026-08-03)

Dynamic topology is implemented: shapes whose VectorShape mutates
tessellate into a transient tail region of the shared buffers (unchanged
shapes stay cached; stable-size dynamic shapes keep the prepare fast path;
epoch flush caps cache growth). Results (200 el / 50 animated arcs, p50):
shapes control 0.99 ms frame (SDF params are its native model), vello 1.19,
engine 1.52 frame but 0.0092 ms GPU (2.3x better than control). Profiling
(bevy/trace_chrome feature; parse the trace-*.json B/E spans) attributed
the frame gap to extract-side re-tessellation; switching all hot-path
hashing to foldhash (tess::fast_hasher) recovered 0.3 ms here and 0.29 ms
on the 5000-static frame (now 1.16 ms). The structural fix for workload 2
is PARAMETRIC INSTANCED PRIMITIVES: canonical (t, side) strip meshes whose
vertex shader computes arc/bar geometry from per-instance params — makes
parameter animation zero-CPU and should win workload-2 frame time outright.

## friginrain2 integration status (2026-08-03)

PfVectorPlugin is live in the game (D:\github\friginrain2, runtime.rs
beside Shape2dPlugin; commit c9abaa7 there). Full game type-checks and
boots with the engine active (18s smoke run, real gameplay systems
ticking, no engine-related errors). Migration surface mapped from their
code: dialogs/HUD draw via IMMEDIATE-MODE ShapePainter with RenderLayers
(frizbi_dialog/mod.rs draw_chrome etc.). Both migration prerequisites landed (engine commit bf93403):
RenderLayers-aware per-view filtering, and the immediate-mode
VectorPainter. The vector pass moved to Core2dSystems::Prepass so
opaque vector content depth-interleaves with main-pass 2D content
(sprites/text/bevy_vector_shapes draw over it by z) — the enabler for
piecemeal migration. First surface ported (game commit c76c815):
frizbi_dialog draw_chrome/draw_account_tray via
hud_component_lab::pf_shapes, the signature-compatible VectorPainter
port of their shapes helpers. Migration template: swap the painter
param + shapes:: -> pf_shapes:: per system. Still legacy: all widgets
(widgets.rs), tuner, other HUD painters. Still open: wire bevy_pf''s
shapes.rs (its docs reserve a GPU-backend seam; currently tiny-skia CPU
rasterization into UI images — correct integration is UI-render-phase
injection or per-shape render-to-texture).

## AA band fix + fidelity comparison (2026-08-04)

Found by building a side-by-side harness against the incumbent
(`hud_component_lab/examples/pf_shapes_compare.rs` in friginrain2: same
specimens, same frame, legacy left / engine right).

Engine bug it exposed: the fringe extruded a FULL pixel OUTWARD with the
interior still at full coverage to the boundary — every shape rendered
~1px larger than authored and thin strokes carried ~2x their weight
(loud on HUD chrome, which is stacked translucent hairlines). Fix: the
AA band straddles the edge (see design points above). Applies to both
the tessellated path (tess.rs stores the outward normal on the interior
boundary vertex too) and the parametric arc path (canonical mesh interior
ring/cap vertices carry radial+tangential directions; radial and
tangential are perpendicular so corner vertices inset correctly).
Verified against the testbed gizmo ground truth, and it is slightly
FASTER (fewer covered fragments): W1 200 el 0.0133 ms GPU p50 (was
0.0143), 5000 el 0.1894 (was 0.1987).

Two things that are NOT engine bugs, both confirmed by pixel
measurement, worth knowing before porting more game surfaces:
1. `bevy_vector_shapes`' `ThicknessType::Pixels` is PHYSICAL pixels;
   engine stroke widths are world units (= logical px under
   ScalingMode::WindowSize). On the 2x-DPI dev display that made ported
   strokes 2x heavy until `pf_shapes` routed thicknesses through the new
   `VectorPainter::screen_px()`.
2. bevy_vector_shapes SILENTLY DROPS content drawn at the same z as a
   fill it draws first: across the whole legacy panel the sampled G
   channel is flat [28..28] — exactly `surface_container`, no overlay
   contribution — while the engine shows [26..61] (black + cyan
   scanlines) and the accent underline. So friginrain2's scanlines,
   accent underlines, and tray avatar circles are authored but have
   never been visible. Where no overlay exists both render exactly
   (19,28,34): no color-space discrepancy. PORTING A SURFACE THEREFORE
   MAKES THAT HIDDEN ART DIRECTION APPEAR — intended by the code, but a
   visible change to the game.

## bevy_pf GPU backend — analysis + design (2026-08-04)

friginrain2's UI is now essentially all bevy_pf/XAML (its immediate-mode
dialog was deleted and the frizbi tuner ported; only tool_workspace is
left). So the engine reaches that game ONLY through bevy_pf's shape
backend. Current path, `bevy_pf/crates/bevy_pf/src/shapes.rs`:

`rasterize_shapes` (plugin.rs:140) walks entities with `PfShape` +
`ComputedNode`; when the laid-out pixel size changed OR the `PfShape`
changed, it tiny-skia rasterizes at that size, **allocates a brand-new
`Image` asset**, and inserts an `ImageNode`; bevy_ui composites it.

What that means for where a GPU backend actually pays — measure before
building:
- STATIC shapes already cost nothing per frame (cached; bevy_ui just
  composites the texture). Porting them to the engine wins little.
- The pathological case is a DATA-BOUND shape: a `Fill`/`Stroke` bound
  to a VM re-rasterizes on the CPU **and allocates a new Image asset and
  re-uploads the whole texture on every change**. Colour is per-instance
  in our engine, so this is where the engine is 100x, not 2x.
- So: target dynamic/bound shapes first; a cheap independent win is to
  reuse the existing texture instead of allocating a new asset per
  repaint (and skip re-tessellation entirely for colour-only changes).

Design for the real integration (NOT per-shape render-to-texture — a
render pass per shape would be worse than what is there now, and would
throw away the instancing the whole engine is built on):
- One shared atlas texture. All `PfShape`s draw into it in a SINGLE
  instanced engine pass, positioned from `ComputedNode` layout.
- Each shape keeps an `ImageNode`, but via
  `ImageNode::from_atlas_image(atlas, TextureAtlas { rect })` — verified
  present in bevy_ui 0.19 (widget/image.rs:99) — so bevy_ui keeps
  owning compositing, clipping, `Overflow::Clip`, and z-order. That is
  the part not worth reimplementing.
- Geometry maps cleanly: `ShapeGeometry::{Rectangle,Ellipse,Line,
  Polyline,Path}` -> `PathCommand`, `PfBrush` -> `Brush` (gradients
  already supported), dash array/caps/joins/fill rule all already exist
  in `StrokeStyle`/`FillRule`.
- bevy_pf gains a `bevy_pf_vector` dependency (it has no renderer seam
  or feature flags today — `crates/bevy_pf/Cargo.toml` [features] is
  empty). The engine stays pure: the dependency points bevy_pf -> engine,
  never the reverse.

## bevy_pf GPU backend — SHIPPED + measured (2026-08-04)

Implemented per the design above and working end to end: bevy_pf feature
`vector_gpu`, module `crates/bevy_pf/src/shapes_gpu.rs`, enabled by
friginrain2. Correctness verified by `examples/shapes_gpu_check.rs`
(rect, rounded+stroke, ellipse, gradient, line, dashed, polyline,
cubic+arc path — all through the GPU path only, screenshot-checked).
Slots are reserved on a 16px grain and mutated in place on resize;
overflow rebuilds the atlas; anything unslotted falls back to CPU.

Two real bugs it exposed, both fixed and both worth remembering:
1. ENGINE: `extract_shapes` treated "component changed" as "topology
   changed". Bevy change detection is per-component, so a colour edit
   re-tessellated the path EVERY FRAME. Now content-hash + cache lookup
   regardless of the flag (engine commit aa2e043). This alone was
   1.675 -> 1.005 ms p50 on 200 animated shapes.
2. bevy_pf: the atlas camera re-rendered + cleared 2048² every frame.
   Now gated on a dirty counter; static UI costs nothing.

HONEST RESULT — the GPU backend does NOT currently beat CPU tiny-skia on
UI workloads (frame-time p50, RTX A6000, 1280x720):

| workload (200 shapes unless noted)   | CPU    | GPU    |
|--------------------------------------|--------|--------|
| 56x40 static                         | 0.847  | 0.857  |
| 56x40 animated fill                  | 0.858  | 0.983  |
| 24 x 300x220 animated                | 0.794  | 0.959  |
| 24 x 300x220 animated, 256-pt path   | 0.789  | 0.849  |
| 120x96 animated, 256-pt path         | 0.856  | 0.889  |

Read the trend, not the rows: the gap closes as geometry gets complex
(0.165 -> 0.033 ms) because tessellate-once amortises while tiny-skia
re-flattens. But it never crosses in the range a UI plausibly occupies.
tiny-skia is simply fast at solid fills, and the atlas round-trip (extra
view + pass + per-frame sync) costs more than it saves. The earlier claim
in this file that bound shapes would be "100x" was WRONG and is retracted
— it assumed re-rasterization dominated; measurement says it does not.

To make it win, remove fixed overhead, not rasterization: the atlas clear
is the whole 2048² for a few small slots (clear only live slots, or size
the atlas to content), and `shape_to_vector` rebuilds the whole command
Vec per changed shape per frame (split path from paint so a colour edit
touches neither). Until then `vector_gpu` is opt-in and honest: it buys
GPU headroom and the engine's analytic clipping/gradients, not frame time.

## Scaling: the 1M standard tier (2026-08-04)

`benchmarks/run-suite.ps1` is the standard suite; tiers are
200 / 1k / 5k / 50k / 1M. One element count can be made to prove either
conclusion about this design, so the CURVE is the measurement.

**1,000,000 elements, RTX A6000** — vector pass GPU **33.55 ms p50**
(65.5M clipper invocations, 8.5M fragments), pass-record CPU 2.31 ms,
but total frame >=250 ms (`cpu_frame_ms` reads exactly 250.000 at every
percentile: that is Bevy's virtual-time max-delta clamp saturating, not
a real number).

So at 1M the GPU does its share in 33 ms while ~215 ms goes somewhere
else on the CPU — extract, sort, batch, and Bevy's own propagation over
1M entities. **The wall is per-element CPU work, not the renderer.**
GPU 0.034 us/element vs CPU ~0.215 us/element: 6.4x.

That reframes the optimisation list. SIMD is NOT the lever (measured
below); parallelising the serial extract + sort is.

## GPU shape backend: the scaling curve (2026-08-04)

Earlier rows in this file concluded the GPU shape backend "does not beat
CPU". That was measured only at <=200 shapes and was WRONG as a general
claim. Frame-time p50, animated fills, bevy_pf `shapes_backend_bench`:

| shapes | CPU (tiny-skia) | GPU (engine) | |
|--------|-----------------|--------------|--|
| 200    | 0.858           | 0.983        | CPU 1.15x |
| 1000   | 1.268           | 1.250        | parity |
| 2000   | 1.759           | 1.474        | GPU 1.19x |
| 5000   | 4.795           | 2.871        | GPU 1.67x |

Crossover ~1000 shapes: below it the atlas round-trip (extra view, pass,
per-frame sync) dominates; above it tiny-skia's per-shape rasterization
does. Both are true; quoting either alone is not.

## SIMD and build options — measured, not assumed (2026-08-04)

- `-C target-cpu=native`: NO improvement. 5000 animated shapes, GPU
  2.937 vs 2.871 baseline; CPU 5.366 vs 4.795. Both within noise or
  slightly worse. glam is already SSE2 by default on x86_64, and these
  loops are not float-math bound — they are allocation, hashing, ECS
  iteration and memory traffic. Wider vectors do not help that.
- Release profile is already right (opt-level 3, fat LTO,
  codegen-units 1, panic=abort in friginrain2).
- Per-entity content-hash reuse (engine `GeometryKeys`) landed: correct,
  but measured within run-to-run noise at 5000 static elements, so
  hashing was not the bottleneck either. Kept because it is strictly
  less work, not because it showed up.

What is actually left on the table, in order of expected payoff:
1. Parallelise `extract_shapes` and the draw-order sort (the ~215 us/1k
   elements of serial CPU work the 1M tier exposes).
2. Split path from paint in bevy_pf's `shape_to_vector`: a colour
   animation currently rebuilds and re-hashes the whole command Vec.
3. Fewer triangles per element (65 per element at 1M; the AA fringe is
   a large part of that).
4. Clear only live atlas slots instead of the whole 2048^2.

## PROFILED: where the CPU time actually goes (2026-08-04)

Measured, not inferred. `--features chrome` (bevy/trace + trace_chrome)
writes trace-*.json; parse B/E pairs into SELF time per span. Tracy is
also wired (`--features tracy`, binaries in E:/github/tracy) but its
per-system zones were not landing; chrome tracing is the working path,
as it was for the earlier workload-2 investigation.

200,000 elements, self time share:

| span                                    | share |
|-----------------------------------------|-------|
| `render::extract_shapes`                | 40.6% |
| `render::prepare_vector_buffers`        |  5.2% |
| transform propagation (par_for_each)    |  0.4% |
| `render::vector_pass`                   |  0.0% |

CORRECTION: the 1M note above attributed the missing ~215 ms/frame to
"extract, sort, batch, and Bevy's own propagation". Propagation is 0.4%
— NEGLIGIBLE. It is almost entirely `extract_shapes`, which is OUR code.
Forking or patching Bevy would have aimed at the wrong target; that idea
is parked until something actually points at Bevy.

Absolute numbers under chrome tracing are inflated (~16x versus the
uninstrumented frame time), so read the SHARES, not the milliseconds.

So the optimisation list from the 1M tier resolves to one item:
`extract_shapes` is the wall. It runs serially over every shape doing a
transform decompose, two hash-map lookups (content key + geometry cache)
and an instance push. Parallelising it — per-thread instance buffers
concatenated after — is the single highest-value change left in the
engine.

## THE EPOCH-FLUSH BUG (2026-08-04) — found by the 1M tier

The single worst bug in the engine so far, invisible for the whole
project because every benchmark tier was too small to trigger it.

`extract_shapes` flushed the whole geometry cache when it exceeded a
FIXED 8192 entries. That threshold is not "the cache got absurd", it is
"the scene got big". Any scene with more than 8192 DISTINCT geometries
wiped the cache every frame and re-tessellated everything, inverting the
engine's central bet — tessellate once, instance forever — into
tessellate-everything-every-frame.

The harness randomises each element's size, so every element is a
distinct geometry. At 200/5000 elements the cache stays under the cap
and all previous numbers were honest. At 200k it flushed every frame.

Fixed: the threshold now scales with the working set,
`8192.max(live_last * 4)`, so a flush means "mostly garbage", not
"large scene".

| tier | before | after | GPU after |
|------|--------|-------|-----------|
| 200,000 | 972 ms/frame | **47.8 ms** | 13.95 ms |
| 1,000,000 | ~4.9 s (est) | **289 ms** | 52.9 ms |

20x at 200k, ~17x at 1M. Also fixed: the harness let Bevy clamp virtual
time to 250 ms, so every tier slower than 4 FPS reported exactly
250.000 at every percentile — a saturated reading that looks like data.
`Time::<Virtual>::max_delta` is now 60 s.

After the fix, 1M is 289 ms frame against 52.9 ms GPU, i.e. ~236 ms of
CPU for 1M elements = 0.24 us/element. THAT is the real per-element
extract cost, and parallelising `extract_shapes` is what attacks it.

Lesson worth keeping: this bug was only reachable by testing far beyond
the intended workload. The suite tiers to 1M for exactly this reason.

## friginrain2 UI migration — state (2026-08-04)

Goal: one UI stack (bevy_pf/XAML), everything else deleted.

DONE:
- Immediate-mode settings dialog deleted (it was unreachable dead code).
- Frizbi tuner ported to XAML (`frizbi_tuner.xaml` + TunerVm).
- `hud_component_lab` cut from 36 components to 3: only button, card and
  progress are referenced from the game. higher_order, scroll_area and 30
  examples deleted with them — 11,921 lines.
- `tool_workspace` no longer references bevy_vector_shapes: its node graph
  paints through the engine's VectorPainter, and bezier wires are now one
  cached cubic path each instead of a 24-segment polyline per frame.

DONE since:
- `friginrain_hud` (separate crate) ported: minimap, skill tree, hud_edit
  and its shared theme drawing layer now paint through the engine.
- hud_component_lab's last three widgets and its demo runner ported;
  legacy `shapes.rs` stripped to painter-free helpers.
- **`bevy_vector_shapes` is removed from every manifest.** The client has
  ONE 2D vector renderer. Builds and boots clean.

The port was a type swap, not a rewrite, because the engine grew a
STATEFUL painter mode (set_translation/set_rotation/color/hollow/thickness
then rect/circle/ngon/arc) mirroring the API that code was written
against. Worth keeping for any future migration off an immediate-mode
painter.

REMAINING:
- `tool_workspace` still builds cards/buttons/progress from
  hud_component_lab — they render on the engine now, but they are not
  XAML. Converting them is what lets the lab be deleted outright;
  `theme` and `interaction` then need rehoming (tokens already exist as
  `assets/ui/obsidian/tokens.xaml`).
- The in-world HUD is engine-drawn, not XAML, and that is the right
  answer: a minimap and skill-tree canvas are per-frame vector drawing,
  not stock controls.

## THE LIMIT OF TESSELLATE-ONCE (2026-08-04) — learned the hard way

Porting friginrain_hud (in-world minimap/skill tree/hud_edit) to the
immediate-mode painter took the game from 100+ FPS to ~20 with heavy
blinking. Reverted.

This is a DESIGN mismatch, not a bug, and it bounds where this engine
should be used:

- The engine tessellates once, keyed by geometry CONTENT. Steady state is
  one instance write per shape — a huge win for shapes whose SIZE is
  stable while transform/colour animate.
- An in-world HUD is the opposite: bars, arcs and blips whose SIZE
  changes every frame. Every frame mints new geometry keys, the cache
  grows, hits the flush threshold, is wiped, and everything
  re-tessellates. A tessellation storm; the blinking is the churn.
- `bevy_vector_shapes` draws SDF primitives, so arbitrary sizes cost it
  no tessellation. For that content it is the better tool.

The engine's own answer already exists and is the pattern to extend:
`VectorPrimitive::Arc` computes geometry in the VERTEX SHADER from
per-instance params, so animating it is one instance write and zero
tessellation. Rect, rounded-rect, circle and line need the same treatment
before any immediate-mode HUD belongs on this painter.

Rule of thumb until then: static or transform-animated content -> this
engine. Per-frame size-varying primitives -> SDF, or a parametric
primitive if one exists.

Also off for now: the bevy_pf `vector_gpu` atlas backend in friginrain2
(missing borders, blinking, layout shift). Feature-gated, still builds.
Before it returns, `shapes_gpu_check` must exercise shapes ACROSS FRAMES
with resize and remount — screenshotting one settled frame is what let all
of this through.

## SDF PRIMITIVES — the design, read out of a working implementation

Researched rather than invented, after the friginrain_hud port failed. The
reference is `C:/github/bevy_vector_shapes/src/render/shaders/` (on disk),
cross-checked against current sources: SDF is still the best practice for
GPU UI in 2026, precisely because it needs no tessellation, no intersection
maths and no CPU processing per size change.

How bevy_vector_shapes actually does a rect (shapes/rect.wgsl):

- ONE quad per instance. Never tessellates, for any shape, ever.
- Instance data: 4x vec4 matrix, color, `thickness`, `flags`, `size: vec2`,
  `corner_radii: vec4`. SIZE IS PER-INSTANCE — that is the whole trick, and
  the thing our tessellate-once cache cannot do.
- Vertex: scales the unit quad by `size`, plus `AA_PADDING = 2.0` so the
  antialiased edge has room. Outputs uv scaled so the SHORTEST side is 1,
  and caps corner radii at half the shortest side.
- Fragment:
      dist = rectSDF(uv, size - radii) - radii
      in_shape *= step_aa(-thickness, dist) * step_aa(dist, 0.0)
  where `rectSDF` is the standard
      length(max(abs(p) - size, 0)) + min(0, max(to_corner.x, to_corner.y))
  FILL AND STROKE ARE THE SAME SHADER: a band test on the distance. Filled
  is just thickness = 1.0 in uv space.
- Antialiasing is `step_aa`, which uses the SCREEN-SPACE DERIVATIVE of the
  distance (`dpdx`/`dpdy` -> `length`) as the pixel footprint. That is why
  it stays crisp at any scale with no geometry work — versus our fringe
  approach, which bakes AA into extruded geometry and therefore has to
  re-tessellate when size changes.
- Fragments outside the shape `discard` before any texture sample.

Mapping onto this engine: `VectorPrimitive::Arc` is ALREADY this pattern
(vertex-shader geometry from per-instance params, zero tessellation), so
the work is to add siblings, not a new subsystem:

    VectorPrimitive::Rect   { size, corner_radii, thickness, color }
    VectorPrimitive::Circle { radius, thickness, color }
    VectorPrimitive::Line   { start, end, thickness, cap, color }

with a quad mesh instead of the arc's (t, side) strip, and an SDF fragment
shader. `thickness == 0` means filled. Once those exist, an immediate-mode
HUD can animate size every frame at one instance write per shape, and
`friginrain_hud` can move over for real.

Do NOT re-attempt that port before these land — the failure was structural,
not incidental.

## CAPABILITY AUDIT vs RIVE (2026-08-05) — we are NOT feature-complete

Read from the vendored rive-runtime (benchmarks/vendor/rive-runtime,
gitignored): `include/rive/renderer.hpp` for the API surface,
`renderer/include/rive/renderer/gpu.hpp` for ShaderFeatures and
InterlockMode. This is the honest gap list, not a benchmark.

| capability                    | Rive | ours |
|-------------------------------|------|------|
| fill rules (nonzero/evenodd)  | yes  | yes  |
| stroke join/cap/miter         | yes  | yes  |
| dash patterns                 | yes  | yes  |
| linear + radial gradients     | yes  | yes  |
| ARBITRARY PATH CLIPPING       | yes  | NO — rounded-rect/circle, 4 deep |
| nested clipping               | yes  | partial (same 4 deep) |
| ADVANCED BLEND MODES (16)     | yes  | NO — srcOver only |
| FEATHER / blur                | yes  | NO   |
| IMAGE FILLS + image meshes    | yes  | NO   |
| save/restore + layer opacity  | yes  | NO   |
| dither                        | yes  | NO   |

For bevy_pf specifically, four of those are not exotic — XAML uses them:
`Clip` takes arbitrary geometry, `Opacity` on a container is a layer
(save/restore) group, `ImageBrush` is an image fill, and effects want blend
modes. Our engine cannot express any of them today.

WHY, architecturally — this is the important part. Advanced blend and
correct arbitrary clipping need to READ the destination while drawing.
Rive offers five strategies for that (`InterlockMode`):

- `rasterOrdering` — fragment shader interlock. Fastest. Not exposed by
  wgpu, which is the constraint ARCHITECTURE.md 2 was built around.
- `atomics` — portable via atomics, no barriers.
- `clockwise` / `clockwiseAtomic` — override every fill rule with a
  clockwise rule; experimental.
- `msaa` — MSAA + stencil. **This one IS expressible in wgpu.**

So the path to feature parity is not an optimisation, it is adopting a
strategy: either an atomics-based or an MSAA+stencil pass structure. Our
current design (single pass, alpha blend, analytic SDF clip) is a
deliberate subset and cannot be extended to these features by tuning.

Rive also uses an UBER SHADER with `ShaderFeatures` bits
(ENABLE_CLIPPING, ENABLE_CLIP_RECT, ENABLE_ADVANCED_BLEND, ENABLE_FEATHER,
ENABLE_EVEN_ODD, ENABLE_NESTED_CLIPPING, ENABLE_HSL_BLEND_MODES,
ENABLE_DITHER) compiled per batch, so simple content pays nothing for
features it does not use. We compile pipeline variants per (format, msaa,
opaque) only — worth copying if the feature set grows.

RECOMMENDATION: do not bolt these on one at a time. Decide first whether
bevy_pf actually needs XAML Clip/Opacity/ImageBrush; if it does, the
honest options are (a) adopt Rive's msaa+stencil structure for correctness,
or (b) keep this engine as the fast subset renderer and let bevy_ui/another
renderer own the general case — which is what the last three attempts
concluded empirically anyway.

## OTHER RENDERERS WORTH STUDYING (2026-08-05)

Rive is not the only reference, and two of these matter more to our
constraints than Rive does.

- **ThorVG 1.0** (C++, released 2026, SVG + Lottie) ships a **WebGPU
  backend**. That is the single most relevant codebase to this project:
  WebGPU is the same capability envelope wgpu gives us, so whatever ThorVG
  does for arbitrary clipping, blend modes and layer opacity is a
  DEMONSTRATION that those are achievable without fragment shader
  interlock. Rive's fast path needs an extension wgpu does not expose;
  ThorVG's does not. Read this before choosing a strategy from the Rive
  audit above. https://github.com/thorvg/thorvg
- **Blend2D** (C++, JIT-compiled CPU rasterizer) is reported to still beat
  GPU renderers including Skia and Vello on many benchmarks. That is a
  direct, independent confirmation of what we measured the hard way: a good
  CPU rasterizer is extremely competitive for UI-sized content, and
  tiny-skia beating our atlas backend 2.4x in-game was not an anomaly.
  https://blend2d.com/about.html
- **Vello** (Rust, wgpu) — GPU compute-centric, uses prefix-scan to
  parallelise sorting/clipping that normally needs CPU or intermediate
  textures. Already vendored as a benchmark opponent. Note `vello-cpu`
  beats Skia and Cairo on CPU, and Vello's GPU advantage is much larger on
  Apple Silicon than on a desktop iGPU — worth remembering before quoting
  any single-machine number.
  https://github.com/linebender/vello
- **Skia** — the general-purpose baseline; still unmeasured here.
- Reference list: https://github.com/zhanba/awesome-2d-graphics-rendering

Order of study for the next person: ThorVG's WebGPU backend first (same
constraints as us, has the features we lack), Blend2D second (why CPU
rasterization keeps winning at UI sizes), Vello third (compute approach we
rejected, but its clipping strategy is instructive).

## Next tasks

- DONE: HudTransform (flat, hierarchy-free; --flat in benchmarks; ~3%
  frame at 5000 el — Bevy propagation is well parallelized so the honest
  win is small). Dead scaffolding removed: node.rs (superseded stub) and
  the never-implemented VectorBackend seam trait (its purpose left with
  path A; backend.rs renamed path.rs holding the authoring types). Still
  future: change-detection extraction + persistent instance slots for
  mostly-static HUDs. Prepare already skips sort/batch/indirect rebuilds
  via a layout fingerprint (rebuilds only when count/geometry/z/opacity
  change); upstreaming a static-transform fast path to Bevy is the
  long-term "fix Bevy" option.
- Workloads 2-4 (animated params, clip stress, stroke stress) in the
  benchmark suite — vello backend already wired as the opponent, rive
  native harness patch reusable for both.
- Style-change handling (VectorShape color changes without re-tessellation
  — currently color is per-instance so it works, but path edits leak old
  geometry in the cache; add eviction).
- AMD / Intel / Apple-Metal measurement passes when hardware is available;
  "fastest" claims are NVIDIA/Vulkan-only until then.
- bevy_pf integration in the sibling testbed once bevy_pf lands locally.
