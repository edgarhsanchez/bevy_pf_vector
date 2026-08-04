//! Vello 0.9 as an in-process benchmark opponent.
//!
//! Vello shares Bevy's wgpu 29 device, renders the same `VectorShape`
//! entities, and is measured by the same GPU timestamp infrastructure as
//! every other backend (`render/vello_render/elapsed_gpu|elapsed_cpu`).
//!
//! Modeled the way a well-written vello app would work: `kurbo::BezPath`s
//! are built once per entity and retained; every frame the scene is
//! re-encoded from them and vello runs its full compute pipeline. That
//! per-frame path processing is vello's architecture — it is exactly the
//! cost the engine's tessellate-once design avoids, so it must be measured,
//! not assumed.
//!
//! GPU timing: vello submits internally, so the span's begin/end timestamps
//! are submitted immediately before and after `render_to_texture` on the
//! same queue — the bracket covers exactly vello's GPU work. The result is
//! then composited into the view target with a texture copy recorded
//! through the normal frame encoder (which submits later, at Submit).

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::schedule::{Core2d, Core2dSystems};
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::diagnostic::{DiagnosticsRecorder, RecordDiagnostics};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    CommandEncoderDescriptor, Extent3d, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue, ViewQuery};
use bevy::render::texture::GpuImage;
use bevy::render::view::ExtractedView;
use bevy::render::{Extract, ExtractSchedule, RenderApp};
use bevy_pf_vector::{PathCommand, VectorShape};
use vello::kurbo::{Affine, BezPath, Cap, Join, Stroke};
use vello::peniko::{Color as VelloColor, Fill};
use vello::{AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene};

pub struct VelloBackendPlugin;

impl Plugin for VelloBackendPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<VelloTarget>::default())
            .add_systems(Startup, setup_vello_target);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<VelloWorkload>()
            .add_systems(ExtractSchedule, extract_vello_shapes)
            .add_systems(Core2d, vello_pass.in_set(Core2dSystems::EarlyPostProcess));
    }
}

/// The image vello renders into; displayed by a fullscreen sprite so the
/// output goes through the normal 2D composite (and stays visually
/// verifiable), with no custom blit pipeline.
#[derive(Resource, Clone, ExtractResource)]
struct VelloTarget(Handle<Image>);

fn setup_vello_target(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    window: Query<&Window>,
) {
    let window = window.single().expect("primary window");
    let (width, height) = (window.physical_width(), window.physical_height());
    let mut image = Image::new_fill(
        Extent3d { width, height, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let handle = images.add(image);
    commands.spawn(Sprite {
        image: handle.clone(),
        custom_size: Some(Vec2::new(window.width(), window.height())),
        ..Default::default()
    });
    commands.insert_resource(VelloTarget(handle));
}

struct RetainedItem {
    fill: Option<(BezPath, VelloColor)>,
    stroke: Option<(Stroke, BezPath, VelloColor)>,
}

#[derive(Default, Resource)]
struct VelloWorkload {
    /// BezPaths built once per entity — retained, like a real vello app.
    retained: HashMap<Entity, RetainedItem>,
    /// Per-frame: (entity, world affine, z), z-sorted at encode time.
    frame: Vec<(Entity, [f64; 6], f32)>,
}

fn to_vello_color(color: LinearRgba) -> VelloColor {
    let srgba: Srgba = color.into();
    VelloColor::from_rgba8(
        (srgba.red.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        (srgba.green.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        (srgba.blue.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        (srgba.alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
    )
}

fn to_bez_path(commands: &[PathCommand]) -> BezPath {
    let mut path = BezPath::new();
    let p = |v: Vec2| (f64::from(v.x), f64::from(v.y));
    for command in commands {
        match *command {
            PathCommand::MoveTo(to) => path.move_to(p(to)),
            PathCommand::LineTo(to) => path.line_to(p(to)),
            PathCommand::QuadTo { ctrl, to } => path.quad_to(p(ctrl), p(to)),
            PathCommand::CubicTo { ctrl1, ctrl2, to } => path.curve_to(p(ctrl1), p(ctrl2), p(to)),
            PathCommand::Close => path.close_path(),
        }
    }
    path
}

fn extract_vello_shapes(
    shapes: Extract<Query<(Entity, &VectorShape, &GlobalTransform)>>,
    mut workload: ResMut<VelloWorkload>,
) {
    workload.frame.clear();
    for (entity, shape, transform) in &shapes {
        workload.retained.entry(entity).or_insert_with(|| RetainedItem {
            fill: shape
                .style
                .fill
                .map(|color| (to_bez_path(&shape.commands), to_vello_color(color))),
            stroke: shape.style.stroke.map(|stroke| {
                let cap = match stroke.cap {
                    bevy_pf_vector::LineCap::Butt => Cap::Butt,
                    bevy_pf_vector::LineCap::Round => Cap::Round,
                    bevy_pf_vector::LineCap::Square => Cap::Square,
                };
                let join = match stroke.join {
                    bevy_pf_vector::LineJoin::Miter => Join::Miter,
                    bevy_pf_vector::LineJoin::Round => Join::Round,
                    bevy_pf_vector::LineJoin::Bevel => Join::Bevel,
                };
                (
                    Stroke::new(f64::from(stroke.width)).with_caps(cap).with_join(join),
                    to_bez_path(&shape.commands),
                    to_vello_color(stroke.color),
                )
            }),
        });
        let model = transform.to_matrix();
        workload.frame.push((
            entity,
            [
                f64::from(model.x_axis.x),
                f64::from(model.x_axis.y),
                f64::from(model.y_axis.x),
                f64::from(model.y_axis.y),
                f64::from(model.w_axis.x),
                f64::from(model.w_axis.y),
            ],
            model.w_axis.z,
        ));
    }
}

struct VelloCtx {
    renderer: Renderer,
    scene: Scene,
}

fn vello_pass(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut workload: ResMut<VelloWorkload>,
    diagnostics: Option<Res<DiagnosticsRecorder>>,
    mut ctx: Local<Option<VelloCtx>>,
    target: Option<Res<VelloTarget>>,
    images: Res<RenderAssets<GpuImage>>,
    view: ViewQuery<&ExtractedView>,
) {
    if workload.frame.is_empty() {
        return;
    }
    let Some(gpu_image) = target.as_ref().and_then(|t| images.get(&t.0)) else {
        return;
    };
    let (width, height) = (gpu_image.texture.width(), gpu_image.texture.height());
    let extracted_view = view.into_inner();

    let ctx = ctx.get_or_insert_with(|| {
        let renderer = Renderer::new(
            device.wgpu_device(),
            RendererOptions {
                antialiasing_support: AaSupport {
                    area: true,
                    msaa8: false,
                    msaa16: false,
                },
                ..Default::default()
            },
        )
        .expect("create vello renderer");
        VelloCtx { renderer, scene: Scene::new() }
    });

    // Re-encode the scene — vello's per-frame cost, deliberately measured.
    ctx.scene.reset();
    workload.frame.sort_by(|a, b| a.2.total_cmp(&b.2));
    // World units are logical pixels, y-up centered; the canvas is physical
    // pixels, y-down. clip00 == 2 / world_width, so physical-per-world is
    // width * clip00 / 2.
    let scale = f64::from(width) * f64::from(extracted_view.clip_from_view.x_axis.x) / 2.0;
    let flip = Affine::new([
        scale,
        0.0,
        0.0,
        -scale,
        f64::from(width) / 2.0,
        f64::from(height) / 2.0,
    ]);
    for (entity, affine, _z) in &workload.frame {
        let Some(item) = workload.retained.get(entity) else {
            continue;
        };
        let transform = flip * Affine::new(*affine);
        if let Some((path, color)) = &item.fill {
            ctx.scene.fill(Fill::NonZero, transform, *color, None, path);
        }
        if let Some((stroke, path, color)) = &item.stroke {
            ctx.scene.stroke(stroke, transform, *color, None, path);
        }
    }

    // GPU bracket: timestamps submitted immediately before and after vello's
    // own internal submission, on the same queue.
    let diagnostics = diagnostics.as_deref();
    let mut begin = device.create_command_encoder(&CommandEncoderDescriptor::default());
    let span = diagnostics.time_span(&mut begin, "vello_render");
    queue.submit([begin.finish()]);

    ctx.renderer
        .render_to_texture(
            device.wgpu_device(),
            &queue,
            &ctx.scene,
            &gpu_image.texture_view,
            &RenderParams {
                base_color: VelloColor::from_rgba8(37, 41, 46, 255),
                width,
                height,
                antialiasing_method: AaConfig::Area,
            },
        )
        .expect("vello render");

    let mut end = device.create_command_encoder(&CommandEncoderDescriptor::default());
    span.end(&mut end);
    queue.submit([end.finish()]);
}
