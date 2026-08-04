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

| backend | 200 el GPU | 5000 el GPU | 5000 el CPU frame | 5000 el fragment inv. |
|---|---|---|---|---|
| shapes (bevy_vector_shapes control) | 0.0225 ms | 0.932 ms | 3.73 ms | 32.3 M |
| **engine (`--backend engine`)** | **0.0174 ms** | **0.324 ms** | **1.66 ms** | **7.8 M** |

The engine wins by structure, not tuning: exact-coverage tessellated
triangles shade ~4x fewer fragments than SDF bounding quads, opaque HUD
content renders with early-z depth testing instead of blending, and
instances are 36 bytes. Measured on NVIDIA/Vulkan only so far — AMD, Intel,
and Apple/Metal numbers are pending hardware access, and every technique
used is portable wgpu (no vendor extensions).

Vendoring benchmark opponents (optional, from Git Bash):

    ./benchmarks/vendor-upstream.sh

## License

Intended as MIT OR Apache-2.0 per `Cargo.toml`. LICENSE files not yet added.
