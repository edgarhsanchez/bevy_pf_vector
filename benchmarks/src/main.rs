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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Shapes,
    Sprites,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Backend::Shapes => "shapes",
            Backend::Sprites => "sprites",
        }
    }
}

#[derive(Resource, Clone, Debug)]
struct BenchConfig {
    backend: Backend,
    elements: u32,
    warmup: u32,
    frames: u32,
    out_dir: PathBuf,
    label: String,
}

fn parse_args() -> BenchConfig {
    let mut backend = Backend::Shapes;
    let mut elements = 200u32;
    let mut warmup = 120u32;
    let mut frames = 600u32;
    let mut out_dir = PathBuf::from("benchmarks/results");
    let mut label = None;

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
                    other => panic!("unknown backend '{other}' (shapes|sprites)"),
                }
            }
            "--elements" => elements = value().parse().expect("--elements"),
            "--warmup" => warmup = value().parse().expect("--warmup"),
            "--frames" => frames = value().parse().expect("--frames"),
            "--out" => out_dir = PathBuf::from(value()),
            "--label" => label = Some(value()),
            other => panic!("unknown argument '{other}'"),
        }
    }

    let label =
        label.unwrap_or_else(|| format!("{}_{}el_{}f", backend.name(), elements, frames));
    BenchConfig { backend, elements, warmup, frames, out_dir, label }
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

const HUD_PALETTE: [Color; 6] = [
    Color::srgb(0.91, 0.30, 0.24),
    Color::srgb(0.18, 0.80, 0.44),
    Color::srgb(0.20, 0.60, 0.86),
    Color::srgb(0.95, 0.77, 0.06),
    Color::srgb(0.61, 0.35, 0.71),
    Color::srgb(0.92, 0.92, 0.92),
];

/// Deterministic HUD-ish layout: jittered grid filling ~1280x720 world units.
fn layout(i: u32, total: u32, rng: &mut Rng) -> (Vec3, f32) {
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
        let (pos, size) = layout(i, cfg.elements, &mut rng);
        let color = HUD_PALETTE[rng.pick(HUD_PALETTE.len() as u32) as usize];

        let mut config = ShapeConfig::default_2d();
        config.color = color;
        // Shapes are authored at unit-ish size and scaled by the transform so
        // the animation path is identical across shape kinds.
        let (transform, anim) = animated(pos, size, &mut rng);
        config.transform = transform;

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

fn setup_sprites(mut commands: Commands, cfg: Res<BenchConfig>) {
    commands.spawn(Camera2d);
    let mut rng = Rng(0xB3_59_1D);

    for i in 0..cfg.elements {
        let (pos, size) = layout(i, cfg.elements, &mut rng);
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

fn animate(mut frame: ResMut<FrameCount>, mut query: Query<(&Animated, &mut Transform)>) {
    frame.0 += 1;
    // Virtual time: 1/120 s per frame regardless of real frame rate.
    let t = frame.0 as f32 / 120.0;
    for (anim, mut transform) in &mut query {
        let angle = anim.base_rotation + 0.35 * ops::sin(t * anim.speed + anim.phase);
        let scale = anim.base_scale * (1.0 + 0.05 * ops::sin(t * anim.speed * 1.7 + anim.phase));
        *transform = Transform::from_translation(anim.translation)
            .with_rotation(Quat::from_rotation_z(angle))
            .with_scale(Vec3::splat(scale));
    }
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
    mut exit: MessageWriter<AppExit>,
) {
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
    .insert_resource(cfg.clone())
    .init_resource::<FrameCount>()
    .init_resource::<BenchState>()
    .add_systems(Update, (animate, sample).chain());

    match cfg.backend {
        Backend::Shapes => app.add_systems(Startup, setup_shapes),
        Backend::Sprites => app.add_systems(Startup, setup_sprites),
    };

    app.run();
}
