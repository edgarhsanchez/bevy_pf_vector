# bevy_pf_vector

Vector/UI rendering for `bevy_pf`. The crate scaffold compiles against Bevy
0.19 and the benchmark harness (ARCHITECTURE.md §5) is built and validated;
the renderer itself is not started.

Read `ARCHITECTURE.md` first, especially:

- **§1** what this can and cannot realistically outperform
- **§2** why there is no pure-wgpu path to the Rive Renderer's fast path
- **§5** why the benchmark harness gets built before the renderer

## Layout

    bevy_pf_vector/     the crate
    harness/            benchmark harness (ARCHITECTURE.md §4-5)
    vendor-upstream.sh  pins upstream renderers as submodules under vendor/
    ARCHITECTURE.md     scope, design, benchmark plan

## Running the harness

    cargo run --release -p harness -- --backend shapes --elements 200
    cargo run --release -p harness -- --backend sprites --elements 200

Flags: `--backend shapes|sprites`, `--elements N` (default 200), `--frames N`
(default 600), `--warmup N` (default 120), `--out DIR`, `--label NAME`.
Prints p50/p95/p99 for CPU frame time and every `render/*` diagnostic
(per-pass GPU time via timestamp queries, pipeline statistics), and writes
JSON with raw samples to `results/`. Requires an adapter with
`TIMESTAMP_QUERY` + `PIPELINE_STATISTICS_QUERY` (desktop Vulkan/DX12).

The `sprites` backend exists to prove the harness can separate two backends
known to differ: at 200 elements the distributions are non-overlapping
(shapes ~0.023 ms vs sprites ~0.014 ms GPU in the 2D main pass on an RTX
A6000), with identical vertex/primitive counts confirming equal draw load.

## Vendoring upstream

Run from the repo root in Git Bash (not PowerShell — it's a bash script):

    ./vendor-upstream.sh

Skia is commented out; it's ~1.5 GB and only needed as reference.

## License

Intended as MIT OR Apache-2.0 per `Cargo.toml`. LICENSE files not yet added.
