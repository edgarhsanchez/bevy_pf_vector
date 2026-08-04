// bevy_pf_vector — instanced vector geometry with analytic edge AA.
//
// Vertices are pre-tessellated path meshes in local space, carrying an
// outward silhouette normal and a coverage value. Fringe vertices
// (coverage 0) are displaced exactly one screen pixel outward in the vertex
// shader, so antialiasing is resolution- and zoom-independent while the
// geometry stays static. Interior vertices have zero normal — no movement.
//
// This removes the need for MSAA: interiors render on the opaque early-z
// pipeline, fringes alpha-blend, edges resolve analytically.
//
// Instances are 36 bytes: 2x2 affine + translation + depth + RGBA8 color.

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

// Raw vec4 view: reading through the typed struct misreads on this naga
// version, so entries are reconstructed from 3 vec4s per 48-byte entry.
@group(0) @binding(1) var<storage, read> clips_raw: array<vec4<f32>>;
@group(0) @binding(2) var gradient_tex: texture_2d<f32>;
@group(0) @binding(3) var gradient_sampler: sampler;

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
    @location(0) position: vec2<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) coverage: f32,
    // [m00, m01, m10, m11] — columns of the 2x2 linear part.
    @location(3) i_linear: vec4<f32>,
    // [tx, ty, z, unused]
    @location(4) i_translation_z: vec4<f32>,
    // Unorm8x4 — hardware-expanded to 0..1 floats for free.
    @location(5) i_color: vec4<f32>,
    // Gradient geometry in local space: linear = (start, end); radial =
    // (center, radius, _).
    @location(6) i_brush_params: vec4<f32>,
    // atlas_row * 4 + kind (0 solid / 1 linear / 2 radial).
    @location(7) i_brush_meta: f32,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) coverage: f32,
    @location(2) world_xy: vec2<f32>,
    @location(3) clip_pack_f: f32,
    @location(4) local_xy: vec2<f32>,
    @location(5) brush_params: vec4<f32>,
    @location(6) brush_meta: f32,
}

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    var world_xy = vec2<f32>(
        in.i_linear.x * in.position.x + in.i_linear.z * in.position.y + in.i_translation_z.x,
        in.i_linear.y * in.position.x + in.i_linear.w * in.position.y + in.i_translation_z.y,
    );

    // Antialiasing band, straddling the true edge. Boundary vertices carry
    // the outward silhouette normal; the displacement is HALF a screen pixel
    // scaled by (0.5 - coverage), so the full-coverage vertex moves half a
    // pixel INWARD and its coverage-0 twin half a pixel OUTWARD. The ramp is
    // therefore centered on the authored edge: coverage 0.5 exactly on it,
    // and the shape's covered area is preserved.
    //
    // Extruding a full pixel outward without insetting (the naive version)
    // inflates every shape by a pixel and gives thin strokes about twice
    // their authored weight — visible immediately when stacking translucent
    // 1px strokes, which is exactly what HUD chrome does.
    //
    // The normal is rotated/scaled by the instance's linear part then
    // renormalized, so instance scale never changes band width.
    // clip_from_world[0][0] == 2 / world_width for an unrotated 2D camera,
    // so one pixel is 2 / (clip00 * viewport_width) world units.
    let world_normal = vec2<f32>(
        in.i_linear.x * in.normal.x + in.i_linear.z * in.normal.y,
        in.i_linear.y * in.normal.x + in.i_linear.w * in.normal.y,
    );
    let len = length(world_normal);
    if (len > 1.0e-6) {
        let px_world = 2.0 / (view.clip_from_world[0][0] * view.viewport.z);
        world_xy += (world_normal / len) * px_world * (0.5 - in.coverage);
    }

    var out: VertexOut;
    out.clip_position =
        view.clip_from_world * vec4<f32>(world_xy, in.i_translation_z.z, 1.0);
    out.color = in.i_color;
    out.coverage = in.coverage;
    out.world_xy = world_xy;
    out.clip_pack_f = in.i_translation_z.w;
    out.local_xy = in.position;
    out.brush_params = in.i_brush_params;
    out.brush_meta = in.i_brush_meta;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    var base = in.color;
    let bmeta = u32(in.brush_meta + 0.5);
    let kind = bmeta & 3u;
    if (kind != 0u) {
        var t = 0.0;
        if (kind == 1u) {
            let d = in.brush_params.zw - in.brush_params.xy;
            t = dot(in.local_xy - in.brush_params.xy, d) / max(dot(d, d), 1.0e-6);
        } else {
            t = length(in.local_xy - in.brush_params.xy) / max(in.brush_params.z, 1.0e-6);
        }
        let row = f32(bmeta >> 2u);
        let uv = vec2<f32>(
            (clamp(t, 0.0, 1.0) * 255.0 + 0.5) / 256.0,
            (row + 0.5) / 1024.0,
        );
        base = textureSampleLevel(gradient_tex, gradient_sampler, uv, 0.0) * in.color;
    }
    return vec4<f32>(base.rgb, base.a * in.coverage * clip_coverage(in.world_xy, u32(in.clip_pack_f + 0.5)));
}
