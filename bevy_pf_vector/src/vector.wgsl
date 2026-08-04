// bevy_pf_vector — instanced vector geometry.
//
// Vertices are pre-tessellated path meshes in local space. Instances are
// 36 bytes: a 2x2 affine + translation + depth + packed RGBA8 color —
// 2D needs no mat4, and slim instances halve vertex-fetch bandwidth on
// every GPU architecture.
//
// Two pipeline variants share these entry points: opaque (depth write,
// no blend — early-z rejects hidden HUD fragments on both immediate-mode
// and tile-based GPUs) and blended (back-to-front, depth read-only).

struct VectorView {
    clip_from_world: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> view: VectorView;

struct VertexIn {
    @location(0) position: vec2<f32>,
    // [m00, m01, m10, m11] — columns of the 2x2 linear part.
    @location(1) i_linear: vec4<f32>,
    // [tx, ty, z, unused]
    @location(2) i_translation_z: vec4<f32>,
    // Unorm8x4 — hardware-expanded to 0..1 floats for free.
    @location(3) i_color: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    let world_xy = vec2<f32>(
        in.i_linear.x * in.position.x + in.i_linear.z * in.position.y + in.i_translation_z.x,
        in.i_linear.y * in.position.x + in.i_linear.w * in.position.y + in.i_translation_z.y,
    );
    var out: VertexOut;
    out.clip_position =
        view.clip_from_world * vec4<f32>(world_xy, in.i_translation_z.z, 1.0);
    out.color = in.i_color;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
