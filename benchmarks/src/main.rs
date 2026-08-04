//! Benchmark harness — ARCHITECTURE.md §4-5.
//!
//! Workload 1 (static HUD): N vector elements with fixed topology, transforms
//! animated every frame. Control backend is `bevy_vector_shapes`; a plain
//! `Sprite` backend is included as a known-cheaper reference so the harness can
//! prove it distinguishes two backends that must differ.
//!
//! Metrics: per-frame CPU time plus everything `RenderDiagnosticsPlugin`
//! records — per-pass GPU time via timestamp queries (`.../elapsed_gpu`, ms)
//! and pipeline statistics (shader invocation counts). Not FPS averages;
//! p50/p95/p99 over a fixed sample window, raw samples kept in the JSON.
//!
//! Usage:
//!   cargo run --release -p benchmarks -- [--backend shapes|sprites]
//!       [--elements N] [--frames N] [--warmup N] [--out DIR] [--label NAME]

use std::collections::BTreeMap;
use std::path::PathBuf;

use bevy::diagnostic::DiagnosticsStore;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::renderer::RenderAdapterInfo;
use bevy::render::settings::{WgpuFeatures, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy::window::{PresentMode, WindowResolution};
use bevy_vector_shapes::prelude::*;

// ---------------------------------------------------------------- config

mod vello_backend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Shapes,
    Sprites,
    Engine,
    Vello,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Backend::Shapes => "shapes",
            Backend::Sprites => "sprites",
            Backend::Engine => "engine",
            Backend::Vello => "vello",
        }
    }
}

#[derive(Resource, Clone, Debug)]
struct BenchConfig {
    backend: Backend,
    elements: u32,
    /// Workload 2: the first N elements are gauge arcs whose sweep angle
    /// rewrites their path every frame (topology churn).
    dynamic: u32,
    /// Workload 3: number of clip panels (nested inside one outer clip).
    clips: u32,
    /// Overlap stress: cluster all elements in a small central disc.
    overlap: bool,
    /// Workload 4: N stroked paths (mixed joins/caps, half dashed).
    strokes: u32,
    /// Engine only: animate via flat HudTransform (no hierarchy propagation).
    flat: bool,
    /// Fill shapes with multi-stop linear/radial gradients.
    gradients: bool,
    warmup: u32,
    frames: u32,
    out_dir: PathBuf,
    label: String,
    screenshot: bool,
}

fn parse_args() -> BenchConfig {
    let mut backend = Backend::Shapes;
    let mut elements = 200u32;
    let mut dynamic = 0u32;
    let mut clips = 0u32;
    let mut warmup = 120u32;
    let mut frames = 600u32;
    let mut out_dir = PathBuf::from("benchmarks/results");
    let mut label = None;
    let mut screenshot = false;
    let mut overlap = false;
    let mut strokes = 0u32;
    let mut flat = false;
    let mut gradients = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("missing value after {arg}"))
        };
        match arg.as_str() {
            "--backend" => {
                backend = match value().as_str() {
                    "shapes" => Backend::Shapes,
                    "sprites" => Backend::Sprites,
                    "engine" => Backend::Engine,
                    "vello" => Backend::Vello,
                    other => panic!("unknown backend '{other}' (shapes|sprites|engine|vello)"),
                }
            }
            "--elements" => elements = value().parse().expect("--elements"),
            "--dynamic" => dynamic = value().parse().expect("--dynamic"),
            "--clips" => clips = value().parse().expect("--clips"),
            "--warmup" => warmup = value().parse().expect("--warmup"),
            "--frames" => frames = value().parse().expect("--frames"),
            "--out" => out_dir = PathBuf::from(value()),
            "--label" => label = Some(value()),
            "--screenshot" => screenshot = true,
            "--overlap" => overlap = true,
            "--flat" => flat = true,
            "--gradients" => gradients = true,
            "--strokes" => strokes = value().parse().expect("--strokes"),
            other => panic!("unknown argument '{other}'"),
        }
    }

    let label = label.unwrap_or_else(|| {
        if strokes > 0 {
            format!("{}_{}strk_{}f", backend.name(), strokes, frames)
        } else if flat {
            format!("{}_{}el_flat_{}f", backend.name(), elements, frames)
        } else if gradients {
            format!("{}_{}el_grad_{}f", backend.name(), elements, frames)
        } else if overlap {
            format!("{}_{}el_ovl_{}f", backend.name(), elements, frames)
        } else if clips > 0 {
            format!("{}_{}el_{}clips_{}f", backend.name(), elements, clips, frames)
        } else if dynamic > 0 {
            format!("{}_{}el_{}dyn_{}f", backend.name(), elements, dynamic, frames)
        } else {
            format!("{}_{}el_{}f", backend.name(), elements, frames)
        }
    });
    BenchConfig { backend, elements, dynamic, clips, overlap, strokes, flat, gradients, warmup, frames, out_dir, label, screenshot }
}

// ---------------------------------------------------------------- rng

/// SplitMix64 — deterministic layout, no rand dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    fn pick(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

// ---------------------------------------------------------------- workload

/// Transform-only animation state. Topology never changes — this is the
/// property path C exists to exploit.
#[derive(Component)]
struct Animated {
    translation: Vec3,
    base_rotation: f32,
    base_scale: f32,
    phase: f32,
    speed: f32,
}

/// Frame index drives animation instead of wall time so runs are reproducible.
#[derive(Resource, Default)]
struct FrameCount(u32);

/// Workload-2 marker: a gauge arc whose sweep rewrites its geometry each
/// frame. Phase/speed come from the sibling `Animated` component.
#[derive(Component)]
struct DynamicArc {
    outer: f32,
    inner: f32,
}

/// Closed ring segment (outer arc, then inner arc reversed), y-up local.
fn ring_segment_path(
    outer: f32,
    inner: f32,
    start: f32,
    end: f32,
    segments: u32,
) -> Vec<bevy_pf_vector::PathCommand> {
    use bevy_pf_vector::PathCommand;
    let mut commands = Vec::with_capacity(segments as usize * 2 + 3);
    let angle = |i: u32| start + (end - start) * i as f32 / segments as f32;
    let at = |radius: f32, a: f32| Vec2::new(ops::cos(a), ops::sin(a)) * radius;
    commands.push(PathCommand::MoveTo(at(outer, angle(0))));
    for i in 1..=segments {
        commands.push(PathCommand::LineTo(at(outer, angle(i))));
    }
    for i in (0..=segments).rev() {
        commands.push(PathCommand::LineTo(at(inner, angle(i))));
    }
    commands.push(PathCommand::Close);
    commands
}

fn arc_sweep(t: f32, speed: f32, phase: f32) -> f32 {
    0.75 * std::f32::consts::TAU * (0.5 + 0.5 * ops::sin(t * speed + phase))
}

/// Workload 2, vello: rewrite the arc paths every frame (its model).
fn animate_arcs(
    frame: Res<FrameCount>,
    mut query: Query<(&Animated, &DynamicArc, &mut bevy_pf_vector::VectorShape)>,
) {
    let t = frame.0 as f32 / 120.0;
    for (anim, arc, mut shape) in &mut query {
        let sweep = arc_sweep(t, anim.speed, anim.phase);
        shape.commands = ring_segment_path(arc.outer, arc.inner, -0.75, -0.75 + sweep, 40);
    }
}

/// Workload 2, engine: sweep is a parameter — one instance write, no
/// geometry work of any kind.
fn animate_arcs_param(
    frame: Res<FrameCount>,
    mut query: Query<(&Animated, &mut bevy_pf_vector::VectorPrimitive)>,
) {
    let t = frame.0 as f32 / 120.0;
    for (anim, mut primitive) in &mut query {
        let bevy_pf_vector::VectorPrimitive::Arc { sweep, .. } = &mut *primitive;
        *sweep = arc_sweep(t, anim.speed, anim.phase);
    }
}

/// Workload 2, shapes control: mutate the SDF disc's arc angles (its native
/// dynamic-parameter path — no re-tessellation in that model).
fn animate_discs(
    frame: Res<FrameCount>,
    mut query: Query<(&Animated, &DynamicArc, &mut DiscComponent)>,
) {
    let t = frame.0 as f32 / 120.0;
    for (anim, _arc, mut disc) in &mut query {
        let sweep = arc_sweep(t, anim.speed, anim.phase);
        disc.start_angle = -0.75;
        disc.end_angle = -0.75 + sweep;
    }
}

const HUD_PALETTE: [Color; 6] = [
    Color::srgb(0.91, 0.30, 0.24),
    Color::srgb(0.18, 0.80, 0.44),
    Color::srgb(0.20, 0.60, 0.86),
    Color::srgb(0.95, 0.77, 0.06),
    Color::srgb(0.61, 0.35, 0.71),
    Color::srgb(0.92, 0.92, 0.92),
];

/// Deterministic HUD-ish layout: jittered grid filling ~1280x720 world
/// units — or, in overlap-stress mode, a dense central disc (art-like deep
/// stacking).
fn layout(i: u32, total: u32, rng: &mut Rng, overlap: bool) -> (Vec3, f32) {
    if overlap {
        let radius = 260.0 * rng.f32().sqrt();
        let angle = rng.range(0.0, std::f32::consts::TAU);
        let size = rng.range(9.0, 26.0);
        return (
            Vec3::new(radius * ops::cos(angle), radius * ops::sin(angle), i as f32 * 0.001),
            size,
        );
    }
    let cols = ((total as f32 * 16.0 / 9.0).sqrt().ceil() as u32).max(1);
    let rows = total.div_ceil(cols);
    let cell_w = 1180.0 / cols as f32;
    let cell_h = 620.0 / rows as f32;
    let col = i % cols;
    let row = i / cols;
    let x = -590.0 + cell_w * (col as f32 + 0.5) + rng.range(-0.3, 0.3) * cell_w;
    let y = -310.0 + cell_h * (row as f32 + 0.5) + rng.range(-0.3, 0.3) * cell_h;
    // Unique z per element keeps transparent sort order stable across frames.
    let z = i as f32 * 0.001;
    let size = rng.range(9.0, 26.0);
    (Vec3::new(x, y, z), size)
}

fn animated(pos: Vec3, size_scale: f32, rng: &mut Rng) -> (Transform, Animated) {
    let anim = Animated {
        translation: pos,
        base_rotation: rng.range(0.0, std::f32::consts::TAU),
        base_scale: size_scale,
        phase: rng.range(0.0, std::f32::consts::TAU),
        speed: rng.range(0.5, 2.0),
    };
    let transform = Transform::from_translation(pos)
        .with_rotation(Quat::from_rotation_z(anim.base_rotation))
        .with_scale(Vec3::splat(anim.base_scale));
    (transform, anim)
}

fn setup_shapes(mut commands: Commands, cfg: Res<BenchConfig>) {
    commands.spawn(Camera2d);
    let mut rng = Rng(0xB3_59_1D);

    for i in 0..cfg.elements {
        let (pos, size) = layout(i, cfg.elements, &mut rng, cfg.overlap);
        let color = HUD_PALETTE[rng.pick(HUD_PALETTE.len() as u32) as usize];

        let mut config = ShapeConfig::default_2d();
        config.color = color;
        // Shapes are authored at unit-ish size and scaled by the transform so
        // the animation path is identical across shape kinds.
        let (transform, anim) = animated(pos, size, &mut rng);
        config.transform = transform;

        // Workload 2 mirror: SDF arc discs with animated sweep — the
        // control's native dynamic-parameter path.
        if i < cfg.dynamic {
            config.hollow = true;
            config.thickness = 0.38;
            config.cap = Cap::None;
            commands
                .spawn(ShapeBundle::new(
                    &config,
                    DiscComponent::arc(&config, 1.0, -0.75, 1.6),
                ))
                .insert((anim, DynamicArc { outer: size, inner: size * 0.62 }));
            continue;
        }

        let kind = rng.pick(100);
        match kind {
            // Filled discs and hollow rings — gauge/indicator staples.
            0..40 => {
                if kind >= 20 {
                    config.hollow = true;
                    config.thickness = rng.range(0.1, 0.3);
                }
                commands
                    .spawn(ShapeBundle::new(&config, DiscComponent::circle(&config, 1.0)))
                    .insert(anim);
            }
            // Rects: filled, rounded, or hollow frames.
            40..65 => {
                if kind >= 55 {
                    config.hollow = true;
                    config.thickness = rng.range(0.1, 0.25);
                }
                if kind % 2 == 0 {
                    config.corner_radii = Vec4::splat(rng.range(0.1, 0.4));
                }
                let aspect = rng.range(0.6, 2.4);
                commands
                    .spawn(ShapeBundle::new(
                        &config,
                        RectangleComponent::new(&config, Vec2::new(2.0 * aspect, 2.0)),
                    ))
                    .insert(anim);
            }
            // Regular polygons, 3-8 sides.
            65..80 => {
                if kind >= 74 {
                    config.hollow = true;
                    config.thickness = rng.range(0.1, 0.3);
                }
                let sides = 3 + rng.pick(6);
                commands
                    .spawn(ShapeBundle::new(
                        &config,
                        RegularPolygonComponent::new(&config, sides as f32, 1.0),
                    ))
                    .insert(anim);
            }
            // Lines with round caps — tick marks, separators.
            _ => {
                config.thickness = rng.range(0.15, 0.4);
                config.cap = Cap::Round;
                let dir = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 0.0)
                    .normalize_or(Vec3::X);
                commands
                    .spawn(ShapeBundle::new(
                        &config,
                        LineComponent::new(&config, -dir, dir),
                    ))
                    .insert(anim);
            }
        }
    }
}

/// Workload 4: stroke stress. N stroked paths — wavy polylines and circles
/// with cycled joins (miter/round/bevel) and caps (butt/round/square), half
/// of them dashed. Engine tessellates each stroke once (dashed via kurbo
/// stroke expansion); vello strokes per frame with its native dash support.
fn setup_stroke_workload(mut commands: Commands, cfg: Res<BenchConfig>) {
    use bevy_pf_vector::{LineCap, LineJoin, PathCommand, PathStyle, StrokeStyle, VectorShape};
    use paths::circle_path;
    if cfg.strokes == 0 {
        return;
    }
    let mut rng = Rng(0x57_20_4B);
    for i in 0..cfg.strokes {
        let (pos, size) = layout(i, cfg.strokes, &mut rng, cfg.overlap);
        let color = HUD_PALETTE[rng.pick(HUD_PALETTE.len() as u32) as usize].to_linear();
        let (transform, anim) = animated(pos, 1.0, &mut rng);
        let width = rng.range(0.12, 0.3) * size;
        let join = match i % 3 {
            0 => LineJoin::Miter,
            1 => LineJoin::Round,
            _ => LineJoin::Bevel,
        };
        let cap = match (i / 3) % 3 {
            0 => LineCap::Butt,
            1 => LineCap::Round,
            _ => LineCap::Square,
        };
        let dash = (i % 2 == 0)
            .then(|| bevy_pf_vector::path::DashPattern::new(vec![width * 2.4, width * 1.6]));
        let commands_path = if i % 5 == 4 {
            // Stroked (often dashed) circle — tick-ring look.
            circle_path(size)
        } else {
            // Wavy polyline with sharp direction changes to stress joins.
            let mut path = Vec::new();
            let span = 3.2 * size;
            let segments = 8;
            for s in 0..=segments {
                let t = s as f32 / segments as f32;
                let x = -span / 2.0 + span * t;
                let y = size * 0.8 * ops::sin(t * 9.4 + i as f32);
                let command = if s == 0 {
                    PathCommand::MoveTo(Vec2::new(x, y))
                } else {
                    PathCommand::LineTo(Vec2::new(x, y))
                };
                path.push(command);
            }
            path
        };
        commands.spawn((
            VectorShape {
                commands: commands_path,
                style: PathStyle::stroke(StrokeStyle {
                    brush: color.into(),
                    width,
                    join,
                    cap,
                    miter_limit: 4.0,
                    dash,
                }),
            },
            transform,
            anim,
        ));
    }
}

/// Workload 3: nested clips. One outer rounded-rect "safe area", `clips`
/// rounded-rect panels inside it, and `elements` shapes distributed across
/// the panels — every shape clipped by its panel, every panel clipped by the
/// outer region (2-level chains). Content deliberately overflows panel
/// bounds so clipping visibly cuts. Engine and vello share this ECS content;
/// vello encodes it as nested clip layers (its native model).
fn setup_clip_workload(mut commands: Commands, cfg: Res<BenchConfig>) {
    use bevy_pf_vector::{ClippedBy, VectorClipShape, VectorShape};
    use paths::*;
    if cfg.clips == 0 {
        return;
    }

    let outer = commands
        .spawn((
            VectorClipShape::RoundedRect {
                half_extents: Vec2::new(600.0, 320.0),
                radius: 48.0,
            },
            Transform::IDENTITY,
        ))
        .id();

    let mut rng = Rng(0xC11B5);
    let p = cfg.clips;
    let cols = ((p as f32 * 16.0 / 9.0).sqrt().ceil() as u32).max(1);
    let rows = p.div_ceil(cols);
    let (cell_w, cell_h) = (1180.0 / cols as f32, 620.0 / rows as f32);
    let mut panels = Vec::new();
    for i in 0..p {
        let x = -590.0 + cell_w * ((i % cols) as f32 + 0.5);
        let y = -310.0 + cell_h * ((i / cols) as f32 + 0.5);
        let half = Vec2::new(cell_w * 0.42, cell_h * 0.40);
        let panel = commands
            .spawn((
                VectorClipShape::RoundedRect { half_extents: half, radius: 14.0 },
                Transform::from_xyz(x, y, 0.0),
                ClippedBy(outer),
            ))
            .id();
        // Visible panel background, clipped by the outer region only.
        commands.spawn((
            VectorShape {
                commands: rounded_rect_path(half.x * 2.0, half.y * 2.0, 14.0),
                style: fill(LinearRgba::new(1.0, 1.0, 1.0, 0.08)),
            },
            Transform::from_xyz(x, y, -0.5 + i as f32 * 0.001),
            ClippedBy(outer),
        ));
        panels.push((panel, x, y, half));
    }

    for j in 0..cfg.elements {
        let (panel, px, py, half) = panels[(j % p) as usize];
        // Overflowing positions — the clip must do real work.
        let pos = Vec3::new(
            px + rng.range(-1.3, 1.3) * half.x,
            py + rng.range(-1.3, 1.3) * half.y,
            j as f32 * 0.001,
        );
        let size = rng.range(8.0, 22.0);
        let color = HUD_PALETTE[rng.pick(HUD_PALETTE.len() as u32) as usize].to_linear();
        let (transform, anim) = animated(pos, 1.0, &mut rng);
        let path = match rng.pick(3) {
            0 => circle_path(size),
            1 => rounded_rect_path(2.6 * size, 1.8 * size, size * 0.3),
            _ => ngon_path(3 + rng.pick(6), size),
        };
        commands.spawn((
            VectorShape { commands: path, style: fill(color) },
            transform,
            anim,
            ClippedBy(panel),
        ));
    }
}

fn setup_sprites(mut commands: Commands, cfg: Res<BenchConfig>) {
    commands.spawn(Camera2d);
    let mut rng = Rng(0xB3_59_1D);

    for i in 0..cfg.elements {
        let (pos, size) = layout(i, cfg.elements, &mut rng, cfg.overlap);
        let color = HUD_PALETTE[rng.pick(HUD_PALETTE.len() as u32) as usize];
        let (transform, anim) = animated(pos, size, &mut rng);
        commands.spawn((
            Sprite {
                color,
                custom_size: Some(Vec2::splat(2.0)),
                ..Default::default()
            },
            transform,
            anim,
        ));
    }
}

/// Workload 1 authored through the engine's own API. Consumes the rng in
/// exactly the same order as `setup_shapes`, so layout, sizes, colors, and
/// kinds are identical to the control — sizes are baked into path
/// coordinates instead of transform scale so tessellation density is right.
fn fill(brush: impl Into<bevy_pf_vector::path::Brush>) -> bevy_pf_vector::PathStyle {
    bevy_pf_vector::PathStyle::fill(brush)
}
fn stroke(color: LinearRgba, width: f32) -> bevy_pf_vector::PathStyle {
    use bevy_pf_vector::{LineCap, LineJoin, StrokeStyle};
    let mut style = StrokeStyle::new(color, width);
    style.join = LineJoin::Round;
    style.cap = LineCap::Round;
    bevy_pf_vector::PathStyle::stroke(style)
}
mod paths {
    use super::*;
    use bevy_pf_vector::PathCommand;

    /// Circle as four cubic Béziers (k = 0.5522847).
    pub fn circle_path(r: f32) -> Vec<PathCommand> {
        let k = 0.552_284_75 * r;
        vec![
            PathCommand::MoveTo(Vec2::new(r, 0.0)),
            PathCommand::CubicTo {
                ctrl1: Vec2::new(r, k),
                ctrl2: Vec2::new(k, r),
                to: Vec2::new(0.0, r),
            },
            PathCommand::CubicTo {
                ctrl1: Vec2::new(-k, r),
                ctrl2: Vec2::new(-r, k),
                to: Vec2::new(-r, 0.0),
            },
            PathCommand::CubicTo {
                ctrl1: Vec2::new(-r, -k),
                ctrl2: Vec2::new(-k, -r),
                to: Vec2::new(0.0, -r),
            },
            PathCommand::CubicTo {
                ctrl1: Vec2::new(k, -r),
                ctrl2: Vec2::new(r, -k),
                to: Vec2::new(r, 0.0),
            },
            PathCommand::Close,
        ]
    }
    pub fn rounded_rect_path(w: f32, h: f32, r: f32) -> Vec<PathCommand> {
        let (hw, hh) = (w / 2.0, h / 2.0);
        let r = r.min(hw).min(hh);
        if r <= 0.0 {
            return vec![
                PathCommand::MoveTo(Vec2::new(-hw, hh)),
                PathCommand::LineTo(Vec2::new(hw, hh)),
                PathCommand::LineTo(Vec2::new(hw, -hh)),
                PathCommand::LineTo(Vec2::new(-hw, -hh)),
                PathCommand::Close,
            ];
        }
        vec![
            PathCommand::MoveTo(Vec2::new(-hw + r, hh)),
            PathCommand::LineTo(Vec2::new(hw - r, hh)),
            PathCommand::QuadTo { ctrl: Vec2::new(hw, hh), to: Vec2::new(hw, hh - r) },
            PathCommand::LineTo(Vec2::new(hw, -hh + r)),
            PathCommand::QuadTo { ctrl: Vec2::new(hw, -hh), to: Vec2::new(hw - r, -hh) },
            PathCommand::LineTo(Vec2::new(-hw + r, -hh)),
            PathCommand::QuadTo { ctrl: Vec2::new(-hw, -hh), to: Vec2::new(-hw, -hh + r) },
            PathCommand::LineTo(Vec2::new(-hw, hh - r)),
            PathCommand::QuadTo { ctrl: Vec2::new(-hw, hh), to: Vec2::new(-hw + r, hh) },
            PathCommand::Close,
        ]
    }
    pub fn ngon_path(sides: u32, r: f32) -> Vec<PathCommand> {
        let mut commands = Vec::new();
        for i in 0..sides {
            let a = i as f32 / sides as f32 * std::f32::consts::TAU + std::f32::consts::FRAC_PI_2;
            let p = Vec2::new(bevy::math::ops::cos(a), bevy::math::ops::sin(a)) * r;
            commands.push(if i == 0 { PathCommand::MoveTo(p) } else { PathCommand::LineTo(p) });
        }
        commands.push(PathCommand::Close);
        commands
    }
}

fn setup_engine(mut commands: Commands, cfg: Res<BenchConfig>) {
    use bevy_pf_vector::{PathCommand, VectorShape};
    use paths::*;

    // Analytic fringe AA makes MSAA redundant for the engine — single-sample
    // rendering is part of its shipped configuration, as MSAA 4x is part of
    // the control's.
    commands.spawn((Camera2d, bevy::render::view::Msaa::Off));
    if cfg.clips > 0 || cfg.strokes > 0 {
        // Workloads 3/4 replace the standard content.
        return;
    }
    let mut rng = Rng(0xB3_59_1D);

    for i in 0..cfg.elements {
        let (pos, size) = layout(i, cfg.elements, &mut rng, cfg.overlap);
        let color = HUD_PALETTE[rng.pick(HUD_PALETTE.len() as u32) as usize].to_linear();
        // Gradient paint when requested: pick from a reusable 24-entry
        // gradient palette (real UIs share gradients; per-shape-unique
        // gradients at scale would be an atlas-budget pathology, and the
        // engine degrades those to solid rather than thrash).
        let paint: bevy_pf_vector::path::Brush = if cfg.gradients {
            use bevy_pf_vector::path::{Brush, GradientStop};
            let g = rng.pick(24);
            let mut grng = Rng(0x6AD_0000 + u64::from(g));
            let c0 = HUD_PALETTE[grng.pick(6) as usize].to_linear();
            let mid = HUD_PALETTE[grng.pick(6) as usize].to_linear();
            let last = HUD_PALETTE[grng.pick(6) as usize].to_linear();
            let stops = vec![
                GradientStop { offset: 0.0, color: c0 },
                GradientStop { offset: grng.range(0.3, 0.7), color: mid },
                GradientStop { offset: 1.0, color: last },
            ];
            if g % 2 == 0 {
                Brush::Linear { start: Vec2::splat(-size), end: Vec2::splat(size), stops }
            } else {
                Brush::Radial { center: Vec2::ZERO, radius: size * 1.2, stops }
            }
        } else {
            color.into()
        };
        // Same rng draws as the control; scale stays 1.0 — size is in the path.
        let (transform, anim) = animated(pos, 1.0, &mut rng);

        // Workload 2: the first `dynamic` elements are gauge arcs. The
        // engine backend uses its parametric fast path (sweep animation is
        // an instance write — zero tessellation); the vello backend renders
        // the same arcs as mutating paths (its model). No extra rng draws,
        // so the stream stays aligned with the other backends.
        if i < cfg.dynamic {
            let (outer, inner) = (size, size * 0.62);
            let mut entity = commands.spawn((transform, anim, DynamicArc { outer, inner }));
            if cfg.backend == Backend::Engine {
                entity.insert(bevy_pf_vector::VectorPrimitive::Arc {
                    inner,
                    outer,
                    start: -0.75,
                    sweep: 2.35,
                    color: color,
                });
            } else {
                entity.insert(VectorShape {
                    commands: ring_segment_path(outer, inner, -0.75, 1.6, 40),
                    style: fill(color),
                });
            }
            continue;
        }

        let kind = rng.pick(100);
        let shape = match kind {
            0..40 => {
                if kind >= 20 {
                    let thickness = rng.range(0.1, 0.3) * size;
                    VectorShape { commands: circle_path(size), style: stroke(color, thickness) }
                } else {
                    VectorShape { commands: circle_path(size), style: fill(paint.clone()) }
                }
            }
            40..65 => {
                let thickness =
                    (kind >= 55).then(|| rng.range(0.1, 0.25) * size);
                let corner = (kind % 2 == 0).then(|| rng.range(0.1, 0.4) * size).unwrap_or(0.0);
                let aspect = rng.range(0.6, 2.4);
                let path = rounded_rect_path(2.0 * aspect * size, 2.0 * size, corner);
                match thickness {
                    Some(thickness) => VectorShape { commands: path, style: stroke(color, thickness) },
                    None => VectorShape { commands: path, style: fill(paint.clone()) },
                }
            }
            65..80 => {
                let thickness = (kind >= 74).then(|| rng.range(0.1, 0.3) * size);
                let sides = 3 + rng.pick(6);
                let path = ngon_path(sides, size);
                match thickness {
                    Some(thickness) => VectorShape { commands: path, style: stroke(color, thickness) },
                    None => VectorShape { commands: path, style: fill(paint.clone()) },
                }
            }
            _ => {
                let thickness = rng.range(0.15, 0.4) * size;
                let dir = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 0.0)
                    .normalize_or(Vec3::X)
                    * size;
                VectorShape {
                    commands: vec![
                        PathCommand::MoveTo(-dir.truncate()),
                        PathCommand::LineTo(dir.truncate()),
                    ],
                    style: stroke(color, thickness),
                }
            }
        };
        if cfg.flat {
            let hud = bevy_pf_vector::HudTransform {
                translation: anim.translation,
                rotation: anim.base_rotation,
                scale: Vec2::splat(anim.base_scale),
            };
            commands.spawn((shape, hud, anim));
        } else {
            commands.spawn((shape, transform, anim));
        }
    }
}

fn animate(
    mut frame: ResMut<FrameCount>,
    mut query: Query<(&Animated, &mut Transform), Without<bevy_pf_vector::HudTransform>>,
) {
    frame.0 += 1;
    // Virtual time: 1/120 s per frame regardless of real frame rate.
    let t = frame.0 as f32 / 120.0;
    // Parallel across the compute pool — the animation is the benchmark
    // client's cost and applies identically to every backend.
    query.par_iter_mut().for_each(|(anim, mut transform)| {
        let angle = anim.base_rotation + 0.35 * ops::sin(t * anim.speed + anim.phase);
        let scale = anim.base_scale * (1.0 + 0.05 * ops::sin(t * anim.speed * 1.7 + anim.phase));
        *transform = Transform::from_translation(anim.translation)
            .with_rotation(Quat::from_rotation_z(angle))
            .with_scale(Vec3::splat(scale));
    });
}

/// Flat-transform variant: animates HudTransform, leaving Transform (and
/// therefore Bevy's propagation systems) untouched.
fn animate_flat(
    frame: Res<FrameCount>,
    mut query: Query<(&Animated, &mut bevy_pf_vector::HudTransform)>,
) {
    let t = frame.0 as f32 / 120.0;
    query.par_iter_mut().for_each(|(anim, mut hud)| {
        hud.rotation = anim.base_rotation + 0.35 * ops::sin(t * anim.speed + anim.phase);
        hud.scale = Vec2::splat(
            anim.base_scale * (1.0 + 0.05 * ops::sin(t * anim.speed * 1.7 + anim.phase)),
        );
        hud.translation = anim.translation;
    });
}

// ---------------------------------------------------------------- metrics

#[derive(Default)]
struct Series {
    values: Vec<f64>,
    last_time: Option<Instant>,
}

#[derive(Resource, Default)]
struct BenchState {
    cpu_ms: Vec<f64>,
    render: BTreeMap<String, Series>,
    announced: bool,
}

fn sample(
    cfg: Res<BenchConfig>,
    frame: Res<FrameCount>,
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    adapter: Option<Res<RenderAdapterInfo>>,
    mut state: ResMut<BenchState>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if cfg.screenshot && frame.0 == 90 {
        commands
            .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
            .observe(bevy::render::view::screenshot::save_to_disk(
                cfg.out_dir.join(format!("{}.png", cfg.label)),
            ));
    }

    if frame.0 <= cfg.warmup {
        return;
    }

    if !state.announced {
        state.announced = true;
        if let Some(info) = &adapter {
            println!("adapter: {} ({:?})", info.name, info.backend);
        }
        let paths: Vec<&str> = diagnostics
            .iter()
            .map(|d| d.path().as_str())
            .filter(|p| p.starts_with("render/"))
            .collect();
        println!("render diagnostics available: {paths:?}");
        if !paths.iter().any(|p| p.ends_with("elapsed_gpu")) {
            eprintln!(
                "WARNING: no elapsed_gpu diagnostics — timestamp queries are \
                 not active on this adapter; GPU numbers will be missing"
            );
        }
    }

    if (state.cpu_ms.len() as u32) < cfg.frames {
        state.cpu_ms.push(time.delta().as_secs_f64() * 1000.0);
    }

    // GPU readback lags a few frames; dedupe on the measurement timestamp so
    // each GPU sample is counted exactly once.
    for diagnostic in diagnostics.iter() {
        let path = diagnostic.path().as_str();
        if !path.starts_with("render/") {
            continue;
        }
        let Some(measurement) = diagnostic.measurement() else {
            continue;
        };
        let series = state.render.entry(path.to_string()).or_default();
        if series.last_time != Some(measurement.time) && (series.values.len() as u32) < cfg.frames
        {
            series.last_time = Some(measurement.time);
            series.values.push(measurement.value);
        }
    }

    if (state.cpu_ms.len() as u32) >= cfg.frames {
        let adapter_name = adapter
            .map(|a| format!("{} ({:?})", a.name, a.backend))
            .unwrap_or_else(|| "unknown".into());
        finish(&cfg, &state, &adapter_name);
        exit.write(AppExit::Success);
    }
}

struct Stats {
    count: usize,
    mean: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    min: f64,
    max: f64,
}

fn stats(values: &[f64]) -> Option<Stats> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| sorted[((p / 100.0) * (sorted.len() - 1) as f64).round() as usize];
    Some(Stats {
        count: sorted.len(),
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        p50: pct(50.0),
        p95: pct(95.0),
        p99: pct(99.0),
        min: sorted[0],
        max: sorted[sorted.len() - 1],
    })
}

fn finish(cfg: &BenchConfig, state: &BenchState, adapter: &str) {
    println!();
    println!(
        "== {} | backend={} elements={} frames={} (warmup {}) ==",
        cfg.label,
        cfg.backend.name(),
        cfg.elements,
        cfg.frames,
        cfg.warmup
    );
    println!(
        "{:<55} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "metric", "count", "mean", "p50", "p95", "p99"
    );

    let mut json = String::from("{\n");
    json.push_str(&format!("  \"label\": \"{}\",\n", cfg.label));
    json.push_str(&format!("  \"backend\": \"{}\",\n", cfg.backend.name()));
    json.push_str(&format!("  \"elements\": {},\n", cfg.elements));
    json.push_str(&format!("  \"warmup\": {},\n", cfg.warmup));
    json.push_str(&format!("  \"frames\": {},\n", cfg.frames));
    json.push_str(&format!("  \"adapter\": \"{}\",\n", adapter.replace('"', "'")));
    json.push_str("  \"metrics\": {\n");

    let mut entries: Vec<(String, &[f64])> = vec![("cpu_frame_ms".into(), &state.cpu_ms)];
    for (path, series) in &state.render {
        entries.push((path.clone(), &series.values));
    }

    let mut first = true;
    for (name, values) in entries {
        let Some(s) = stats(values) else { continue };
        println!(
            "{:<55} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            name, s.count, s.mean, s.p50, s.p95, s.p99
        );
        if !first {
            json.push_str(",\n");
        }
        first = false;
        json.push_str(&format!(
            "    \"{}\": {{\n      \"count\": {}, \"mean\": {:.6}, \"p50\": {:.6}, \
             \"p95\": {:.6}, \"p99\": {:.6}, \"min\": {:.6}, \"max\": {:.6},\n      \"raw\": [{}]\n    }}",
            name,
            s.count,
            s.mean,
            s.p50,
            s.p95,
            s.p99,
            s.min,
            s.max,
            values
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    json.push_str("\n  }\n}\n");

    std::fs::create_dir_all(&cfg.out_dir).expect("create results dir");
    let path = cfg.out_dir.join(format!("{}.json", cfg.label));
    std::fs::write(&path, json).expect("write results json");
    println!("\nwrote {}", path.display());
}

// ---------------------------------------------------------------- app

fn main() {
    let cfg = parse_args();
    if (cfg.clips > 0 || cfg.strokes > 0 || cfg.gradients) && !matches!(cfg.backend, Backend::Engine | Backend::Vello) {
        panic!(
            "workloads 3/4 (--clips/--strokes) require --backend engine|vello: '{}' lacks the features",
            cfg.backend.name()
        );
    }
    println!(
        "harness: backend={} elements={} warmup={} frames={}",
        cfg.backend.name(),
        cfg.elements,
        cfg.warmup,
        cfg.frames
    );

    // Timestamp + pipeline-statistics queries are what make GPU numbers real.
    // Supported on Vulkan/DX12 desktop; requesting them on an adapter that
    // lacks them fails device creation loudly, which we prefer to silence.
    let mut wgpu_settings = WgpuSettings::default();
    wgpu_settings.features |=
        WgpuFeatures::TIMESTAMP_QUERY | WgpuFeatures::PIPELINE_STATISTICS_QUERY;
    // Lets the engine backend collapse each phase into one multi-draw
    // (indirect args carry first_instance). Desktop Vulkan/DX12 support
    // this; the engine falls back to a draw loop where it's absent.
    wgpu_settings.features |= WgpuFeatures::INDIRECT_FIRST_INSTANCE;

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(RenderPlugin {
                render_creation: wgpu_settings.into(),
                ..Default::default()
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: format!("pf_vector harness — {}", cfg.label),
                    resolution: WindowResolution::new(1280, 720),
                    // Uncapped presentation; vsync would flatten every number.
                    present_mode: PresentMode::AutoNoVsync,
                    ..Default::default()
                }),
                ..Default::default()
            }),
    )
    .add_plugins(RenderDiagnosticsPlugin)
    .add_plugins(Shape2dPlugin::default())
    // Bevy clamps virtual time to 250 ms per frame by default, so any tier
    // slower than 4 FPS reports exactly 250.000 at every percentile — a
    // saturated reading that looks like data. The top-end tiers (200k, 1M)
    // are all slower than that, so raise the ceiling far past anything we
    // will measure.
    .insert_resource({
        let mut time = Time::<Virtual>::default();
        time.set_max_delta(std::time::Duration::from_secs(60));
        time
    })
    .insert_resource(cfg.clone())
    .init_resource::<FrameCount>()
    .init_resource::<BenchState>()
    .add_systems(Update, (animate, sample).chain());

    match cfg.backend {
        Backend::Shapes => app
            .add_systems(Startup, setup_shapes)
            .add_systems(Update, animate_discs.after(animate).before(sample)),
        Backend::Sprites => app.add_systems(Startup, setup_sprites),
        Backend::Engine => app
            .add_plugins(bevy_pf_vector::PfVectorPlugin)
            .add_systems(Startup, (setup_engine, setup_clip_workload, setup_stroke_workload))
            .add_systems(
                Update,
                (animate_arcs_param, animate_flat).after(animate).before(sample),
            ),
        // Same VectorShape entities as the engine backend, rendered by vello.
        Backend::Vello => app
            .add_plugins(vello_backend::VelloBackendPlugin)
            .add_systems(Startup, (setup_engine, setup_clip_workload, setup_stroke_workload))
            .add_systems(Update, animate_arcs.after(animate).before(sample)),
    };

    app.run();
}
