// Signed-distance primitives: rect, rounded rect, circle and line, all from
// ONE quad per instance and one shader.
//
// Why this exists alongside the tessellated path: tessellation is keyed by
// geometry CONTENT, so a shape whose SIZE changes every frame mints new
// geometry every frame and re-tessellates. That is fine for HUD chrome whose
// dimensions are fixed and whose transform/colour animate; it is pathological
// for bars, meters and blips that resize continuously. Here `size` is an
// INSTANCE parameter, so resizing costs one instance write and zero
// tessellation.
//
// One primitive covers four shapes because a rounded rect degenerates:
//   radius == 0            -> rect
//   radius == min(size)/2  -> circle/capsule
//   thin size.y + rotation -> line with round caps
// Fill and stroke are the same code path: a band test on the distance, with
// `thickness <= 0` meaning filled. Technique follows bevy_vector_shapes'
// shapes/rect.wgsl (read, not guessed) — see CLAUDE.md.

struct VectorView {
    clip_from_world: mat4x4<f32>,
    // (origin.xy, size.zw) — width is .z, NOT .x. See CLAUDE.md.
    viewport: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: VectorView;
@group(0) @binding(1) var<storage, read> clips_raw: array<vec4<f32>>;

fn clip_one(world: vec2<f32>, index: u32) -> f32 {
    let base = index * 3u;
    let v0 = clips_raw[base];
    let v1 = clips_raw[base + 1u];
    let v2 = clips_raw[base + 2u];
    let local = vec2<f32>(
        v0.x * world.x + v0.z * world.y + v1.x,
        v0.y * world.x + v0.w * world.y + v1.y,
    );
    let half_extents = vec2<f32>(v1.z, v1.w);
    let radius = v2.x;
    let q = abs(local) - half_extents;
    let dist = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    let px_world = 2.0 / (view.clip_from_world[0][0] * view.viewport.z);
    let aa = max(px_world * length(v0.xy), 1.0e-5);
    return clamp(0.5 - dist / aa, 0.0, 1.0);
}

fn clip_coverage(world: vec2<f32>, pack: u32) -> f32 {
    let count = pack % 8u;
    let index = pack / 8u;
    var cov = 1.0;
    if (count > 0u) { cov = cov * clip_one(world, index); }
    if (count > 1u) { cov = cov * clip_one(world, index + 1u); }
    if (count > 2u) { cov = cov * clip_one(world, index + 2u); }
    if (count > 3u) { cov = cov * clip_one(world, index + 3u); }
    return cov;
}

struct VertexIn {
    // Unit quad corner in [-1, 1].
    @location(0) position: vec2<f32>,
    @location(1) unused_normal: vec2<f32>,
    @location(2) unused_coverage: f32,
    @location(3) i_linear: vec4<f32>,
    @location(4) i_translation_z: vec4<f32>,
    @location(5) i_color: vec4<f32>,
    // [size.x, size.y, corner_radius, thickness]
    @location(6) i_params: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Position within the shape, in local units, for the distance field.
    @location(1) local_xy: vec2<f32>,
    @location(2) world_xy: vec2<f32>,
    @location(3) clip_pack_f: f32,
    @location(4) half_size: vec2<f32>,
    @location(5) radius: f32,
    @location(6) thickness: f32,
}

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    let size = in.i_params.xy;
    let half_size = size * 0.5;
    // Corner radius cannot exceed half the shortest side, or the SDF inverts.
    let radius = clamp(in.i_params.z, 0.0, min(half_size.x, half_size.y));
    let thickness = in.i_params.w;

    // Pad the quad so the antialiased edge — and, for a stroke, the outer
    // half of the band — has somewhere to live. Two pixels matches the
    // reference and covers the derivative footprint at any scale.
    let px_world = 2.0 / (view.clip_from_world[0][0] * view.viewport.z);
    let pad = px_world * 2.0;

    let local = in.position * (half_size + vec2<f32>(pad));
    let world_xy = vec2<f32>(
        in.i_linear.x * local.x + in.i_linear.z * local.y + in.i_translation_z.x,
        in.i_linear.y * local.x + in.i_linear.w * local.y + in.i_translation_z.y,
    );

    var out: VertexOut;
    out.clip_position =
        view.clip_from_world * vec4<f32>(world_xy, in.i_translation_z.z, 1.0);
    out.color = in.i_color;
    out.local_xy = local;
    out.world_xy = world_xy;
    out.clip_pack_f = in.i_translation_z.w;
    out.half_size = half_size;
    out.radius = radius;
    out.thickness = thickness;
    return out;
}

/// Distance from `p` to a box of half-extents `b`. Negative inside.
fn box_sdf(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    // Rounded rect: shrink the box by the radius, then subtract it back.
    let dist = box_sdf(in.local_xy, in.half_size - vec2<f32>(in.radius)) - in.radius;

    // Antialias from the SCREEN-SPACE derivative of the distance. This is why
    // the primitive is resolution- and scale-independent without any geometry
    // work: the pixel footprint is measured per fragment rather than baked
    // into an extruded fringe.
    let aa = max(fwidth(dist), 1.0e-5);

    // Outer edge for both fill and stroke.
    var coverage = 1.0 - smoothstep(-aa, aa, dist);
    if (in.thickness > 0.0) {
        // Stroke: knock out everything further inside than `thickness`, so
        // the surviving band straddles the authored edge.
        coverage = coverage * smoothstep(-aa, aa, dist + in.thickness);
    }

    let alpha = in.color.a * coverage
        * clip_coverage(in.world_xy, u32(in.clip_pack_f + 0.5));
    if (alpha < 0.0001) {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
