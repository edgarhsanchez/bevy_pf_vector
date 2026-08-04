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

Backend names map to products: `engine` = bevy_pf_vector (this repo's
renderer, the thing being proven); `shapes` = bevy_vector_shapes 0.13
(third-party Bevy crate, the original baseline); `vello` = Vello 0.9
(Linebender, in-process); `sprites` = plain Bevy sprites (harness
validation only). The native C++ Rive Renderer runs via its own patched
harness (see below), not a `--backend`.

Flags: `--backend engine|shapes|vello|sprites`, `--elements N` (default 200), `--frames N`
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

Each backend measured at its shipped configuration: bevy_vector_shapes
under its default MSAA 4x, Vello with its default Area AA and full compute pipeline
(scene re-encoded per frame from retained BezPaths, GPU time bracketed by
timestamps around its submission), the engine single-sample (its AA is
analytic). Same 2560x1440 target, same rng-identical workload — verified
visually via `--screenshot` for engine and vello.

| backend | 200 el GPU | 5000 el GPU | per-frame path/scene CPU (5000 el) |
|---|---|---|---|
| Vello 0.9 (Linebender, in-process) | 0.816 ms | 1.493 ms | 0.385 ms encode (p95 1.14) |
| bevy_vector_shapes 0.13 (third-party crate) | 0.0225 ms | 0.932 ms | — |
| bevy_pf_vector — first slice (MSAA 4x) | 0.0174 ms | 0.324 ms | — |
| **bevy_pf_vector (ours)** | **0.0143 ms** | **0.200 ms** | **none (tessellate-once)** |

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
| Rive Renderer (native C++) frame (p50) | 0.204 ms | 1.716 ms |
| bevy_pf_vector (ours) renderer cost (GPU pass + record + prepare) | ~0.015 ms | ~0.19 ms |

Renderer-for-renderer the engine is roughly an order of magnitude faster at
both scales, and our entire Bevy frame at 5000 elements (1.16 ms, ECS and
all — after layout-fingerprint caching, foldhash keys made prepare a pure gather and the
benchmark client's animation went parallel; bevy_vector_shapes measured
with the same client at 3.31 ms) clearly beats the Rive Renderer's render-only loop (1.72 ms). An opt-in flat
`HudTransform` component now exists for hierarchy-free HUD elements
(animating it never dirties the transform graph; measured ~3% frame gain at
5000 animated elements — Bevy 0.19's propagation is already well
parallelized, so the win is modest and honest). Rive pays per-frame path
processing and flush work by design — the cost this engine's
tessellate-once model eliminates. Caveats: rive numbers are frame-time (its
loop is render-only, but CPU/GPU overlap means GPU-only could be lower);
comparison is Vulkan-on-NVIDIA only; rive's atomic/MSAA fallback modes and
feather/clip-heavy content (workloads 3-4) not yet measured.

bevy_pf_vector vs Vello: ~100x at 200 elements, ~11x at 5000. Vello's ~0.8 ms floor
at low element counts is its canvas-sized compute pipeline — the per-frame
cost a general-purpose renderer pays regardless of content, and exactly what
the tessellate-once design avoids. This is not a knock on vello: it handles
arbitrary dynamic scenes this engine deliberately does not.

## Workload 2 results (animated parameters: 200 elements, 50 arcs whose
## sweep rewrites their path every frame — p50 over 600 frames)

| backend | frame | render GPU |
|---|---|---|
| bevy_vector_shapes 0.13 (third-party crate) | 0.99 ms | 0.0215 ms |
| Vello 0.9 (Linebender) | 1.19 ms | 0.854 ms |
| bevy_pf_vector — VectorShape mutation (re-tessellating) | 1.52 ms | 0.0092 ms |
| **bevy_pf_vector (ours) — parametric arcs** | **0.78 ms** | **0.0155 ms** |

The engine wins workload 2 on both metrics via `VectorPrimitive` — a
canonical (t, side) strip mesh whose vertex shader computes ring-segment
geometry (and its analytic AA fringes) from per-instance
[start, sweep, inner, outer]. Animating a gauge is one 52-byte instance
write: zero tessellation, all arcs in one instanced draw. Arbitrary-path
mutation is also supported (rows above): changed `VectorShape`s tessellate
into a transient buffer region, priced per changed shape, cache untouched.

## Workload 3 results (nested clips: 12 rounded-rect panels inside an outer
## region, 240 overflowing shapes, 2-level chains — p50 over 600 frames)

| backend | frame | render GPU |
|---|---|---|
| bevy_vector_shapes 0.13 | not supported (no clipping) | — |
| Vello 0.9 (Linebender, native clip layers) | 1.11 ms | 0.953 ms |
| **bevy_pf_vector (ours)** (`--clips 12`) | **0.78 ms** | **0.0266 ms** |

Clipping is analytic: clip shapes (`VectorClipShape` rounded-rects/circles,
nested via `ClippedBy`, up to 4 deep) live in a storage buffer as inverse
transforms + SDF parameters; each instance carries a packed chain reference
in its spare instance lane and the fragment shader multiplies coverage per
entry. Clip edges are antialiased (stencil clipping can't do that), no
extra draw calls or state changes, batching fully preserved. Clipped
instances route through the blend phase for the alpha knock-out.

Note on current numbers: a viewport-component bug (origin.x where width
belonged) had been making fringe AA degenerate and was found via
screenshot-driven shader bisection while building workload 3; with it fixed
the engine pays its real AA cost — earlier GPU figures were ~30-45% lower
but with broken edge AA. The tables above are the honest, corrected values.

## Overlap stress (2000 shapes stacked ~15 deep in a central disc — p50)

| backend | render GPU | fragment invocations |
|---|---|---|
| bevy_vector_shapes 0.13 | 0.285 ms | 12.9 M |
| Vello 0.9 (Linebender) | 1.083 ms | — |
| **bevy_pf_vector (ours)** (`--overlap`) | **0.0686 ms** | **1.07 M** |

Dense overlap is where the depth-based opaque path pays off hardest: opaque
groups draw front-to-back (nearest group first, preserving instancing), so
early-z rejects ~12x the fragment work bevy_vector_shapes shades. Honest limits:
the win applies to opaque content — translucent stacks blend per layer like
every rasterizer — and crossing AA fringes of different shapes can
double-blend (conflation), invisible in HUDs, occasionally visible in
dense art; rive's interlock avoids it, vello's area AA has its own
conflation artifacts.

## Workload 4 results (stroke stress: 300 stroked paths, cycled
## miter/round/bevel joins and butt/round/square caps, half dashed — p50)

| backend | frame | render GPU |
|---|---|---|
| bevy_vector_shapes 0.13 | not supported (no polyline strokes/joins/dashes) | — |
| Vello 0.9 (Linebender, native dashed strokes) | 1.33 ms | 0.833 ms |
| **bevy_pf_vector (ours)** (`--strokes 300`) | **0.81 ms** | **0.0236 ms** |

`StrokeStyle` now supports `dash: Some([on, off])`: dashed strokes expand to
fill outlines via kurbo stroke expansion (real joins, caps, dashes) and
tessellate once like everything else; undashed strokes keep the direct lyon
path. Joins and caps are honored exactly — the correctness gap
ARCHITECTURE.md called out in rive-bevy (which forces round joins/caps).

With this, all four workloads of the ARCHITECTURE.md suite plus overlap
stress are measured. The engine wins every workload against every opponent
able to run it. Not yet measured: Skia, Pathfinder (standalone harnesses),
rive native beyond workload 1, and non-NVIDIA GPUs.

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
