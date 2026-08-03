# bevy_pf_vector — scope, architecture, and what is actually achievable

Status: design scaffold. Nothing here has been compiled or benchmarked.

## 1. Scope correction, stated plainly

Three parts of the request need pushback before any code is worth writing.

**"Outperform them" is not a deliverable I can hand you.** Skia is ~20 years of
work; Vello is a multi-year Linebender effort with a compute-based architecture;
the Rive Renderer is a narrow, well-tuned GPU vector rasterizer. A new engine
that beats all three at their own targets does not come out of a design session.
What *is* achievable is beating them **on one narrow workload** — game HUD/UI
with mostly-static vector geometry, high instance counts, and a known-ahead-of-
time asset set. That is a real gap, and it is winnable, because the general-
purpose engines pay per-frame costs for content they cannot assume is static.
Any claim beyond that narrow target is unsupported until benchmarked.

**CUDA is the wrong dependency.** A game client renderer must run on AMD, Intel,
Apple, and Android GPUs. CUDA locks you to NVIDIA and buys nothing for vector
rasterization that compute shaders through wgpu do not already provide. Drop it.
DLSS (already in your workspace) is a separate, legitimate NVIDIA-only *optional*
path for 3D upscaling — do not confuse the two.

**NDK matters only if Android ships.** If it does, it constrains the design
hard: no fragment shader interlock on most Android drivers, so the fallback path
below becomes the primary path, not the fallback.

I also cannot fetch the CUDA, ROCm, or NDK SDKs here — they are multi-GB,
license-gated, and off the allowed network list. They are toolchain installs on
your machine, not repo contents.

## 2. The constraint that determines the whole design

The Rive Renderer's fast path needs fragment shader interlock
(`VK_EXT_fragment_shader_interlock`) or D3D rasterizer-ordered views. WebGPU has
had an open request for this since 2019; wgpu does not expose it. Bevy is wgpu.

Therefore there are exactly three viable paths, and they are not equivalent:

| Path | Mechanism | Portability | Effort |
|---|---|---|---|
| A. Native interop | `wgpu-hal` `as_hal()` → hand `VkDevice`/`ID3D12Device` to Rive's `RenderContext`, render into a Bevy-owned texture | Desktop only, per-backend work | Medium |
| B. Compute rasterization | Vello-style: no interlock needed, sorting/binning in compute | Everywhere incl. WASM/Android | High |
| C. Cached tessellation | Tessellate once (lyon/kurbo), instance and re-transform on GPU | Everywhere, trivially | Low |

**C is the narrow win.** For HUD content the geometry rarely changes topology —
only transforms, colors, and state-machine-driven parameters do. Skia, Vello, and
Rive all re-path-process per frame because they cannot assume otherwise. A
tessellate-once-and-instance pipeline sidesteps that entirely. This is also what
`bevy_vector_shapes` does for SDF primitives, but it does not handle arbitrary
paths or `.riv` state machines.

Recommended sequencing: **C first** (ships, portable, measurable), **A second**
(for the cases C cannot cover — dynamic strokes, feathering, heavy clipping),
**B never** unless you decide to own a rasterizer as a product in itself.

## 3. Crate shape

    bevy_pf_vector/
      src/
        lib.rs        plugin registration, asset types
        backend.rs    VectorBackend trait — the seam between C and A
        node.rs       render graph node + explicit barriers
        ffi.rs        rive-runtime C ABI surface (path A only)

The `VectorBackend` trait is the important artifact. If it is drawn correctly,
path C and path A are swappable at runtime and benchmarkable head-to-head in the
same frame loop. If it is drawn wrong, you will rewrite the plugin when A lands.

## 4. Benchmark plan — write this before the renderer

Comparison is only meaningful within a workload. Define these four and measure
all candidates (bevy_vector_shapes, rive-bevy/Vello, path C, path A) against them:

1. **Static HUD** — 200 vector elements, transforms only, no topology change.
2. **Animated HUD** — 50 elements driven by state machines, per-frame parameters.
3. **Clip stress** — nested clips (Vello's documented weak point).
4. **Stroke stress** — dashed/tapered strokes with real joins and caps
   (rive-bevy currently forces round joins/caps on Vello — a correctness gap,
   not just a speed one).

Metrics: frame time p50/p99, GPU time via timestamp queries, draw call count,
buffer upload bytes/frame. Not FPS averages.

## 5. First milestone

Do *not* start with the renderer. Start with the harness: workload 1 above,
rendering through `bevy_vector_shapes` as the control, with timestamp queries
wired up. If the harness cannot distinguish two known-different backends, no
subsequent performance claim from this project means anything.
