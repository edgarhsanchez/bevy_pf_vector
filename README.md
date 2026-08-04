# bevy_pf_vector

An original vector/UI rendering engine for `bevy_pf`, targeting Bevy 0.19.

One narrow bet: game HUD content has mostly-static topology, so paths are
tessellated once at asset load (lyon) and redrawn as GPU-instanced geometry —
skipping the per-frame path processing that general-purpose engines
(Skia, Vello, Rive) are structurally unable to skip. "Better" is defined as
winning the benchmark suite in `benchmarks/` on frame-time and GPU-time
percentiles, not asserted.

The engine crate is pure: no third-party renderer bindings. Competing
renderers exist in this repo only as benchmark opponents, quarantined under
`benchmarks/`.

Read `ARCHITECTURE.md` first — especially §1 (what "better" can honestly
mean) and §5 (why the benchmark harness was built before the renderer).

## Layout

    bevy_pf_vector/     the engine crate (pure — bevy + lyon/kurbo only)
    benchmarks/         harness, results, and vendor pins for opponents
    ARCHITECTURE.md     scope, design, benchmark plan

A sibling project, `../bevy_pf_vector_testbed`, is the integration proving
ground: a Bevy app consuming this crate as a path dependency.

## Running the benchmarks

    cargo run --release -p benchmarks -- --backend shapes --elements 200
    cargo run --release -p benchmarks -- --backend sprites --elements 200

Flags: `--backend shapes|sprites`, `--elements N` (default 200), `--frames N`
(default 600), `--warmup N` (default 120), `--out DIR`, `--label NAME`.
Prints p50/p95/p99 for CPU frame time and every `render/*` diagnostic
(per-pass GPU time via timestamp queries, pipeline statistics), and writes
JSON with raw samples to `benchmarks/results/`. Requires an adapter with
`TIMESTAMP_QUERY` + `PIPELINE_STATISTICS_QUERY` (desktop Vulkan/DX12).

Harness validity was established before any engine work: at 200 elements the
`shapes` and `sprites` backends produce non-overlapping GPU-time
distributions (~0.023 ms vs ~0.014 ms on an RTX A6000) with identical
vertex/primitive counts, and a 200 → 5000 element sweep scales GPU time 38x.

## Workload 1 results (RTX A6000, Vulkan — p50 over 600 frames)

Each backend measured at its shipped configuration: the control under its
default MSAA 4x, vello with its default Area AA and full compute pipeline
(scene re-encoded per frame from retained BezPaths, GPU time bracketed by
timestamps around its submission), the engine single-sample (its AA is
analytic). Same 2560x1440 target, same rng-identical workload — verified
visually via `--screenshot` for engine and vello.

| backend | 200 el GPU | 5000 el GPU | per-frame path/scene CPU (5000 el) |
|---|---|---|---|
| vello 0.9 (in-process, shared device) | 0.816 ms | 1.493 ms | 0.385 ms encode (p95 1.14) |
| shapes (bevy_vector_shapes control) | 0.0225 ms | 0.932 ms | — |
| engine, first slice (MSAA 4x) | 0.0174 ms | 0.324 ms | — |
| **engine, current** | **0.0082 ms** | **0.137 ms** | **none (tessellate-once)** |

**Native Rive Renderer (C++), measured.** Built from rive-runtime at the
pinned SHA (clang/lld, Vulkan backend, SPIR-V via glslang) with a local
`--bench N` patch (`benchmarks/rive-native-bench.patch`) that ports this
harness's exact workload generator (same SplitMix64 stream, layout, kinds,
retained RenderPaths) into `path_fiddle`. Runs on its **interlock fast
path** (`fragmentShaderPixelInterlock` active) at 2560x1440, uncapped
present. Its minimal GLFW loop does nothing but render, so frame time ~=
renderer cost:

| | 200 el | 5000 el |
|---|---|---|
| rive native frame (p50) | 0.204 ms | 1.716 ms |
| engine renderer cost (GPU pass + record + prepare) | ~0.015 ms | ~0.19 ms |

Renderer-for-renderer the engine is roughly an order of magnitude faster at
both scales, and our entire Bevy frame at 5000 elements (1.16 ms, ECS and
all — after layout-fingerprint caching, foldhash keys made prepare a pure gather and the
benchmark client's animation went parallel; control measured with the same
client at 3.31 ms) clearly beats rive's render-only loop (1.72 ms). The dominant
remaining frame cost is Bevy's transform propagation, not the renderer —
the next lever is an opt-in flat HUD transform path that bypasses the
hierarchy, plus change-detection extraction for mostly-static HUDs. Rive pays per-frame path
processing and flush work by design — the cost this engine's
tessellate-once model eliminates. Caveats: rive numbers are frame-time (its
loop is render-only, but CPU/GPU overlap means GPU-only could be lower);
comparison is Vulkan-on-NVIDIA only; rive's atomic/MSAA fallback modes and
feather/clip-heavy content (workloads 3-4) not yet measured.

Engine vs vello: ~100x at 200 elements, ~11x at 5000. Vello's ~0.8 ms floor
at low element counts is its canvas-sized compute pipeline — the per-frame
cost a general-purpose renderer pays regardless of content, and exactly what
the tessellate-once design avoids. This is not a knock on vello: it handles
arbitrary dynamic scenes this engine deliberately does not.

## Workload 2 results (animated parameters: 200 elements, 50 arcs whose
## sweep rewrites their path every frame — p50 over 600 frames)

| backend | frame | render GPU |
|---|---|---|
| shapes (SDF control) | **0.99 ms** | 0.0215 ms |
| vello 0.9 | 1.19 ms | 0.854 ms |
| engine (`--dynamic 50`) | 1.52 ms | **0.0092 ms** |

Honest reading: the engine now *supports* per-frame topology change (shapes
whose `VectorShape` mutates tessellate into a transient buffer region;
unchanged shapes stay on the tessellate-once path; stable-size dynamic
shapes even keep the prepare fast path) — and its GPU time stays 2.3x ahead
of the control. But the control wins workload-2 *frame* time: parametric
SDF arcs are bevy_vector_shapes' native primitive, costing it zero geometry
work, while we pay ~1 ms CPU re-tessellating 50 arcs per frame. The
recorded fix (next milestone): parametric instanced primitives — a
canonical (t, side) strip whose vertex shader computes ring-segment
positions from per-instance start/sweep/radii, making parameter animation
free for the common gauge/bar/arc cases while arbitrary paths keep the
tessellation path.

Not yet measured: Skia, Pathfinder (different stacks, standalone
harnesses), rive native on workload 2, and workloads 3-4.

The engine wins by structure, not tuning:

- exact-coverage tessellated triangles — ~4x fewer fragment invocations
  than SDF bounding quads at 5000 elements;
- analytic edge AA: a one-screen-pixel silhouette fringe extruded in the
  vertex shader (geometry stays static, width is resolution- and
  zoom-independent), so the engine renders single-sample — no 4x MSAA
  bandwidth tax;
- opaque HUD content draws with early-z depth testing instead of blending,
  grouped by geometry for maximal instancing;
- 36-byte instances; each phase submits as one `multi_draw_indexed_indirect`
  where the device offers indirect-first-instance (fallback: plain loop).

Measured on NVIDIA/Vulkan only so far — AMD, Intel, and Apple/Metal numbers
are pending hardware access, and every technique used is portable wgpu (no
vendor extensions; optional features detected at runtime).

Vendoring benchmark opponents (optional, from Git Bash):

    ./benchmarks/vendor-upstream.sh

## License

Intended as MIT OR Apache-2.0 per `Cargo.toml`. LICENSE files not yet added.
