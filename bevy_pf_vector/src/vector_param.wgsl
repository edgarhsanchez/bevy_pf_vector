// bevy_pf_vector — parametric instanced primitives (arcs / ring segments).
//
// The mesh is a canonical strip: vertices carry (t, side) instead of
// positions, and the vertex shader computes ring-segment geometry from
// per-instance parameters [start, sweep, inner, outer]. Animating a gauge
// is a 52-byte instance write — zero CPU tessellation — and every arc in
// the scene draws in one instanced call.
//
// AA matches the tessellated path: fringe vertices (coverage 0) displace
// one screen pixel outward, here computed analytically — radially on the
// curved edges, tangentially on the end caps.

struct VectorView {
    clip_from_world: mat4x4<f32>,
    // (origin.xy, size.zw) in physical pixels.
    viewport: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: VectorView;
struct ClipEntry {
    inv_linear: vec4<f32>,
    inv_translation: vec2<f32>,
    half_extents: vec2<f32>,
    radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// Raw vec4 view — see vector.wgsl for why.
@group(0) @binding(1) var<storage, read> clips_raw: array<vec4<f32>>;

// Analytic nested clipping: multiply coverage by an antialiased
// rounded-rect/circle SDF per chain entry. AA width is derived from the
// view scale (no derivatives — safe in any control flow).
fn clip_one(world: vec2<f32>, e: u32) -> f32 {
    let v0 = clips_raw[e * 3u];
    let v1 = clips_raw[e * 3u + 1u];
    let v2 = clips_raw[e * 3u + 2u];
    let p = vec2<f32>(
        v0.x * world.x + v0.z * world.y + v1.x,
        v0.y * world.x + v0.w * world.y + v1.y,
    );
    let q = abs(p) - v1.zw;
    let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - v2.x;
    let px_world = 2.0 / (view.clip_from_world[0][0] * view.viewport.z);
    let aa = max(px_world * length(v0.xy), 1.0e-5);
    return clamp(0.5 - d / aa, 0.0, 1.0);
}

// Straight-line nested-clip evaluation (a dynamic loop miscompiles on some
// naga/driver combinations); up to 4 chain entries.
fn clip_coverage(world: vec2<f32>, pack: u32) -> f32 {
    let count = pack & 7u;
    let index = pack >> 3u;
    var cov = 1.0;
    if (count > 0u) { cov = cov * clip_one(world, index); }
    if (count > 1u) { cov = cov * clip_one(world, index + 1u); }
    if (count > 2u) { cov = cov * clip_one(world, index + 2u); }
    if (count > 3u) { cov = cov * clip_one(world, index + 3u); }
    return cov;
}

struct VertexIn {
    // x: t along the arc (0..1); y: 0 = inner edge, 1 = outer edge.
    @location(0) t_side: vec2<f32>,
    // Fringe displacement directions: x radial (-1/0/+1), y tangential.
    @location(1) fringe: vec2<f32>,
    @location(2) coverage: f32,
    @location(3) i_linear: vec4<f32>,
    @location(4) i_translation_z: vec4<f32>,
    @location(5) i_color: vec4<f32>,
    // [start_angle, sweep, inner_radius, outer_radius]
    @location(6) i_params: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) coverage: f32,
    @location(2) world_xy: vec2<f32>,
    @location(3) clip_pack_f: f32,
}

fn apply_linear(linear: vec4<f32>, v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        linear.x * v.x + linear.z * v.y,
        linear.y * v.x + linear.w * v.y,
    );
}

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    let angle = in.i_params.x + in.t_side.x * in.i_params.y;
    let dir = vec2<f32>(cos(angle), sin(angle));
    let radius = mix(in.i_params.z, in.i_params.w, in.t_side.y);
    var world_xy =
        apply_linear(in.i_linear, dir * radius)
        + vec2<f32>(in.i_translation_z.x, in.i_translation_z.y);

    // One-pixel analytic fringe, resolution- and scale-independent.
    if (in.fringe.x != 0.0 || in.fringe.y != 0.0) {
        let px_world = 2.0 / (view.clip_from_world[0][0] * view.viewport.z);
        var offset = vec2<f32>(0.0, 0.0);
        if (in.fringe.x != 0.0) {
            let radial = apply_linear(in.i_linear, dir);
            offset += normalize(radial) * in.fringe.x;
        }
        if (in.fringe.y != 0.0) {
            let tangent = apply_linear(in.i_linear, vec2<f32>(-dir.y, dir.x));
            offset += normalize(tangent) * in.fringe.y;
        }
        world_xy += offset * px_world;
    }

    var out: VertexOut;
    out.clip_position =
        view.clip_from_world * vec4<f32>(world_xy, in.i_translation_z.z, 1.0);
    out.color = in.i_color;
    out.coverage = in.coverage;
    out.world_xy = world_xy;
    out.clip_pack_f = in.i_translation_z.w;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb, in.color.a * in.coverage * clip_coverage(in.world_xy, u32(in.clip_pack_f + 0.5)));
}
