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
    // xy = viewport size in physical pixels.
    viewport: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: VectorView;

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
        let px_world = 2.0 / (view.clip_from_world[0][0] * view.viewport.x);
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
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb, in.color.a * in.coverage);
}
