#include "rt_lab2.hpp"
#include "rt_lab_internal.hpp"

#include "rt_lab2.inl"
#include "rt_lab2_gradient.inl"
#include "rt_lab2_probe.hpp"
#include "rt_lab2_stats.hpp"
#include "rt_lab_trace.inl"
#include "rt_lab_resolve.inl"
#include "rt_lab_convergence.inl"
#include "rt_lab_measure.inl"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <imgui.h>
#include <nlohmann/json.hpp>

auto rt_lab2_default_settings() -> RtLab2Settings
{
    return RtLab2Settings{
        .cell_size_px = 16.0f,
        .min_cell_size = 0.02f,
        .level_blend_width = 0.5f,
        .max_untouched_frames = 64,
        .history_samples_per_pixel = 48.0f,
        .min_history_samples = 16.0f,
        .max_history_samples = 65536.0f,
        .max_frame_samples = 0.0f,
        .cell_capacity_log2 = static_cast<f32>(RT_LAB2_CELL_CAPACITY_LOG2_DEFAULT),
        .exposure_alpha = 0.25f,
        .exposure_strength = 1.0f,
        .exposure_min_scale = 0.25f,
        .redistribution_enabled = 1,
        .max_rays_per_pixel = 12,
        .redistribution_target = 0.5f,
        .lookup_kernel = RT_LAB2_KERNEL_GAUSSIAN,
        .lookup_taps = 4,
        .lookup_distance = 0.7f,
        .lookup_project = 1,
        .lookup_min_samples = 1,
        .integrator_enabled = 1,
        .window_radius = 2,
        .variance_gamma = 2.0f,
        .alpha_min = 0.033f,
        .alpha_max = 0.128f,
        .variance_reference = 0.15f,
        .spatial_strength = 0.5f,
        .max_frames = 64,
        .plane_tolerance = 2.0f,
        .normal_threshold = 0.9f,
        .gradient_enabled = 1,
        .gradient_full_coverage = 64.0f,
        .gradient_blur_taps = 12,
        .gradient_blur_radius = 64.0f,
        .gradient_min_variance = 0.002f,
        .gradient_kalman_q = 0.95f, // evidence decay per frame
        .gradient_kalman_gate = 2.0f, // significance threshold (sigmas)
        .gradient_cap_k = 0.05f,
        .gradient_sample_variance = 5.0f, // quartic cut gain
        .gradient_glide = 0.15f, // fraction of the evidence applied per frame
        .gradient_correct_integrator = 1,
        .reset = 0,
        .animate_noise = 1,
    };
}

///
/// === Frame update ===
///

auto rt_lab2_frame_update(RenderContext & render_context, bool reference_phase) -> bool
{
    auto & lab2 = render_context.rt_lab.lab2;
    auto & settings = render_context.render_data.rt_lab_settings;
    lab2.active = lab2.enabled && !reference_phase && settings.enabled != 0;
    if (lab2.active)
    {
        // The trace and the geometry stay the lab's (hit shading, AO on hit, noise); everything of the old denoiser is off.
        // The ray budget decides which pixels shoot: below 1 the distribute pass takes a rotated Bayer subset per 8x8 tile.
        f32 const budget = std::max(lab2.ray_budget, 1.0f / 64.0f);
        settings.half_res = 0;
        settings.animate_noise = 1;
        settings.redistribution_enabled = 1;
        settings.ray_budget = budget;
        // The old demand pass asks for 1 base ray per pixel only: the convergence target 0 makes calc_desired_extra_rays return 0 extras. The cell driven
        // extras come from entry_rt_lab2_demand. max_rays_per_pixel is the real max, so the old distribute pass draws the rays shot views (DEBUG_DRAW_MODE_RTGI_*_RAYS_SHOT)
        // correctly normalized (heat = rays / max).
        settings.max_rays_per_pixel = std::max(lab2.settings.max_rays_per_pixel, 1);
        settings.fast_convergence_samples = 0.0f;
        settings.min_ray_budget = budget >= 1.0f ? 1.0f : (lab2.settings.redistribution_enabled != 0 ? std::clamp(lab2.min_base_coverage, 0.0f, 1.0f) : 0.0f);
        settings.temporal_enabled = 0;
        settings.temporal_gradient_enabled = 0;
        settings.temporal_gradient_all_base_rays = 0;
        settings.split_history_enabled = 0;
        settings.spatial_enabled = 0;
        settings.coarse_anti_lag_enabled = 0;
        settings.ray_filter_mode = std::clamp(lab2.ray_filter_mode, 0, 2);
        settings.ray_filter_radius = std::clamp(lab2.ray_filter_radius, 0, RT_LAB_RAY_FILTER_MAX_RADIUS);
        settings.ray_filter_tolerance = lab2.ray_filter_tolerance;
        settings.ray_filter_factor = lab2.ray_filter_factor;
        settings.sh_guide_mode = 0;
        settings.distribution_guide_mode = RT_LAB_DISTRIBUTION_OFF;
        settings.luma_moments_enabled = 0;
    }
    bool rebuild = lab2.active != lab2.built_active;
    lab2.built_active = lab2.active;
    // The cell buffer's size is part of the graph (a new size = a rebuild, the grid starts empty).
    i32 const capacity_log2 = rt_lab2_cell_capacity_log2(lab2.settings);
    if (capacity_log2 != lab2.built_capacity_log2)
    {
        rebuild = true;
        lab2.built_capacity_log2 = capacity_log2;
    }
    // The temporal gradient's passes + its ray record buffer are part of the graph.
    if ((lab2.settings.gradient_enabled != 0) != lab2.built_gradient)
    {
        rebuild = true;
        lab2.built_gradient = lab2.settings.gradient_enabled != 0;
    }
    return rebuild;
}

///
/// === Callbacks ===
///

static auto rt_lab2_settings_address(daxa::TaskInterface ti, RenderContext * render_context) -> daxa::DeviceAddress
{
    auto & lab2 = render_context->rt_lab.lab2;
    RtLab2Settings settings = lab2.settings;
    settings.reset = lab2.frame_reset ? 1 : 0;
    settings.animate_noise = 1;
    // The temporal resolve is the shared one (RtLabState::integrator, part of the preset): its settings drive the integrator pass of the cells too.
    auto const & integrator = render_context->rt_lab.integrator;
    settings.integrator_enabled = integrator.enabled ? 1 : 0;
    settings.window_radius = integrator.settings.window_radius;
    settings.variance_gamma = integrator.settings.variance_gamma;
    settings.alpha_min = integrator.settings.alpha_min;
    settings.alpha_max = integrator.settings.alpha_max;
    settings.variance_reference = integrator.settings.variance_reference;
    settings.spatial_strength = integrator.settings.spatial_strength;
    settings.max_frames = integrator.settings.max_frames;
    settings.plane_tolerance = integrator.settings.plane_tolerance;
    settings.normal_threshold = integrator.settings.normal_threshold;
    if (render_context->rt_lab.lab4.active)
    {
        settings.gradient_enabled = 0; // RT Lab 4 feeds the cells from the ReSTIR stage: no ray records for the temporal gradient
    }
    auto alloc = ti.allocator->allocate(sizeof(RtLab2Settings)).value();
    std::memcpy(alloc.host_address, &settings, sizeof(RtLab2Settings));
    return alloc.device_address;
}

template <typename Push>
static void rt_lab2_dispatch_pixels(daxa::TaskInterface ti, RenderContext * render_context, Push push, daxa::TaskImageAttachmentIndex size_image, u32 render_time, std::string const & pipeline_name)
{
    push.settings = rt_lab2_settings_address(ti, render_context);
    rt_lab_dispatch(ti, render_context, push, size_image, RT_LAB2_PASS_X, render_time, pipeline_name);
}

// One thread per cell slot.
static void rt_lab2_dispatch_cells(daxa::TaskInterface ti, RenderContext * render_context, u32 render_time, std::string const & pipeline_name)
{
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, render_time);
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(pipeline_name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLab2CellPassPush push = {};
    push.attach = alloc.device_address;
    push.settings = rt_lab2_settings_address(ti, render_context);
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(rt_lab2_cell_capacity(render_context->rt_lab.lab2.settings), static_cast<u32>(RT_LAB2_CELL_PASS_X)), 1, 1});
}

// First pass of the frame: decides the frame's reset (a test phase start, or a freshly built graph whose persistent
// resources are undefined) and empties the cells then. All later passes read RtLab2State::frame_reset.
static void rt_lab2_reset_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto & lab2 = render_context->rt_lab.lab2;
    lab2.frame_reset = render_context->render_data.rt_lab_settings.reset_history != 0 || lab2.clear_requested;
    lab2.clear_requested = false;
    if (lab2.frame_reset)
    {
        rt_lab2_dispatch_cells(ti, render_context, RenderTimes::index<"RT_LAB", "LAB2_RESET">(), rt_lab2_reset_compile_info().name);
    }
}

static void rt_lab2_query_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab2QueryH::Info::AT;
    rt_lab2_dispatch_pixels(ti, render_context, RtLab2QueryPush{}, AT.pixel_cell, RenderTimes::index<"RT_LAB", "LAB2_QUERY">(), rt_lab2_query_compile_info().name);
}

static void rt_lab2_demand_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    if (render_context->rt_lab.lab2.settings.redistribution_enabled == 0)
    {
        return;
    }
    auto const & AT = RtLab2DemandH::Info::AT;
    rt_lab2_dispatch_pixels(ti, render_context, RtLab2DemandPush{}, AT.ray_demand, RenderTimes::index<"RT_LAB", "LAB2_DEMAND">(), rt_lab2_demand_compile_info().name);
}

static void rt_lab2_rays_debug_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab2RaysDebugH::Info::AT; // the shader exits at once unless one of the rays shot debug modes is active
    rt_lab2_dispatch_pixels(ti, render_context, RtLab2RaysDebugPush{}, AT.lab_depth, RenderTimes::index<"RT_LAB", "LAB2_RAYS_DEBUG">(), rt_lab2_rays_debug_compile_info().name);
}

static void rt_lab2_select_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab2SelectH::Info::AT;
    auto const lab = ti.info(AT.lab_depth).value().size;
    RtLab2SelectPush push = {};
    push.lab_size = {lab.x, lab.y};
    rt_lab2_dispatch_pixels(ti, render_context, push, AT.candidate, RenderTimes::index<"RT_LAB", "LAB2_DEMAND">(), rt_lab2_select_compile_info().name);
}

static void rt_lab2_distribute_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 ray_capacity)
{
    auto const & AT = RtLab2DistributeH::Info::AT;
    auto const lab = ti.info(AT.ray_alloc).value().size;
    RtLab2DistributePush push = {};
    push.lab_size = {lab.x, lab.y};
    push.ray_capacity = ray_capacity;
    rt_lab2_dispatch_pixels(ti, render_context, push, AT.candidate, RenderTimes::index<"RT_LAB", "RAY_DISTRIBUTE">(), rt_lab2_distribute_compile_info().name);
}

static void rt_lab2_gradient_trace_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RT_LAB", "LAB2_GRADIENT_TRACE">());
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab2_gradient_trace_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLab2GradientTracePush push = {};
    push.attach = alloc.device_address;
    push.settings = rt_lab2_settings_address(ti, render_context);
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(rt_lab2_cell_capacity(render_context->rt_lab.lab2.settings), static_cast<u32>(RT_LAB2_CELL_PASS_X)), 1, 1});
}

static void rt_lab2_gradient_spatial_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab2GradientSpatialH::Info::AT;
    auto const lab = ti.info(AT.lab_depth).value().size;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RT_LAB", "LAB2_GRADIENT_SPATIAL">());
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab2_gradient_spatial_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLab2GradientSpatialPush push = {};
    push.attach = alloc.device_address;
    push.settings = rt_lab2_settings_address(ti, render_context);
    push.lab_size = {lab.x, lab.y};
    push.size = {(lab.x + 1) / 2, (lab.y + 1) / 2};
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(push.size.x, static_cast<u32>(RT_LAB2_PASS_X)), round_up_div(push.size.y, static_cast<u32>(RT_LAB2_PASS_Y)), 1});
}

static void rt_lab2_gradient_record_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 ray_capacity)
{
    auto const & AT = RtLab2GradientRecordH::Info::AT;
    auto const lab = ti.info(AT.lab_depth).value().size;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RT_LAB", "LAB2_GRADIENT_RECORD">());
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab2_gradient_record_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLab2GradientRecordPush push = {};
    push.attach = alloc.device_address;
    push.settings = rt_lab2_settings_address(ti, render_context);
    push.size = {lab.x, lab.y};
    push.ray_capacity = ray_capacity;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(ray_capacity, static_cast<u32>(RT_LAB_TRACE_THREADS)), 1, 1});
}

static void rt_lab2_prepare_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    rt_lab2_dispatch_cells(ti, render_context, RenderTimes::index<"RT_LAB", "LAB2_CELL_PREPARE">(), rt_lab2_prepare_compile_info().name);
}

static void rt_lab2_accumulate_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab2AccumulateH::Info::AT;
    rt_lab2_dispatch_pixels(ti, render_context, RtLab2AccumulatePush{}, AT.lab_raw_diffuse, RenderTimes::index<"RT_LAB", "LAB2_ACCUMULATE">(), rt_lab2_accumulate_compile_info().name);
}

static void rt_lab2_cell_integrate_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    rt_lab2_dispatch_cells(ti, render_context, RenderTimes::index<"RT_LAB", "LAB2_CELL_INTEGRATE">(), rt_lab2_cell_integrate_compile_info().name);
}

static void rt_lab2_lookup_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab2LookupH::Info::AT;
    rt_lab2_dispatch_pixels(ti, render_context, RtLab2LookupPush{}, AT.lookup, RenderTimes::index<"RT_LAB", "LAB2_LOOKUP">(), rt_lab2_lookup_compile_info().name);
}

static void rt_lab2_integrator_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab2IntegratorH::Info::AT;
    rt_lab2_dispatch_pixels(ti, render_context, RtLab2IntegratorPush{}, AT.integrated, RenderTimes::index<"RT_LAB", "LAB2_INTEGRATOR">(), rt_lab2_integrator_compile_info().name);
}

///
/// === Task graph ===
///

auto tasks_rt_lab2_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult
{
    auto & render_context = info.render_context;
    auto & lab_state = render_context.rt_lab;
    auto const & settings = render_context.render_data.rt_lab_settings; // the lab settings with the RT Lab 2 overrides applied
    auto const render_target_size = render_context.render_data.settings.render_target_size;
    auto const size = daxa::Extent3D{render_target_size.x, render_target_size.y, 1};
    // The graph's persistent resources are undefined after a rebuild: the first frame resets them.
    lab_state.lab2.clear_requested = true;

    // Lab geometry (full res), written by the ray demand. Double buffered: the integrator reprojects against last frame's.
    auto lab_depth = info.tg.create_task_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "rt_lab2_depth_persistent",
    });
    auto lab_face_normal = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "rt_lab2_face_normal_persistent",
    });
    // Per pixel ray mean (.rgb, RTGI_RADIANCE_SCALE units, can exceed f16) and ray count.
    auto lab_raw_diffuse = info.tg.create_task_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = size,
        .name = "rt_lab2_raw_diffuse",
    });
    auto cells = info.tg.create_task_buffer({
        .size = sizeof(RtLab2Cell) * static_cast<u64>(rt_lab2_cell_capacity(lab_state.lab2.settings)),
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT,
        .name = "rt_lab2_cells_persistent",
    });
    auto pixel_cell = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = size,
        .name = "rt_lab2_pixel_cell",
    });
    auto pixel_cell2 = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = size,
        .name = "rt_lab2_pixel_cell2",
    });
    auto lookup = info.tg.create_task_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = size,
        .name = "rt_lab2_lookup",
    });
    auto integrator_history = info.tg.create_task_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT, // .rgb radiance (RTGI_RADIANCE_SCALE units overflow f16), .a frames
        .size = size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "rt_lab2_integrator_persistent",
    });

    // Ray list (the old lab's passes): a selection of the full res pixels shoots.
    u32 const ray_capacity = size.x * size.y * rt_lab_ray_capacity_per_pixel(settings);
    auto ray_counters = info.tg.create_task_buffer({
        .size = sizeof(RtLabRayCounters),
        .name = "rt_lab2_ray_counters",
    });
    auto ray_list = info.tg.create_task_buffer({
        .size = sizeof(u32) * ray_capacity,
        .name = "rt_lab2_ray_list",
    });
    auto rays = info.tg.create_task_buffer({
        .size = sizeof(RtLabRay) * ray_capacity,
        .name = "rt_lab2_rays",
    });
    auto ray_demand = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = size,
        .name = "rt_lab2_ray_demand",
    });
    auto ray_alloc = info.tg.create_task_image({
        .format = daxa::Format::R32G32_UINT,
        .size = size,
        .name = "rt_lab2_ray_alloc",
    });
    info.tg.clear_buffer({.buffer = ray_counters, .name = "rt_lab2_ray_counters_clear"});

    info.tg.add_task(daxa::HeadTask<RtLab2CellPassH::Info>("RtLab2Reset")
        .head_views(RtLab2CellPassH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .cells = cells,
        })
        .executes(rt_lab2_reset_callback, &render_context));

    info.tg.add_task(daxa::HeadTask<RtLabDemandH::Info>()
        .head_views(RtLabDemandH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .view_cam_depth = info.view_cam_depth,
            .view_cam_face_normals = info.view_cam_face_normals,
            .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
            .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .lab_depth_history = lab_depth.previous(),
            .lab_face_normal_history = lab_face_normal.previous(),
            .history = daxa::NullTaskImage,
            .ray_demand = ray_demand,
            .ray_counters = ray_counters,
            .debug_image = info.debug_image,
            .variance_history = daxa::NullTaskImage,
            .gradient2 = daxa::NullTaskImage,
        })
        .executes(rt_lab_demand_callback, &render_context, false, false));

    // The cells' exposure, where a reprojection would happen: every pixel counts itself into its cell, then the cells
    // smooth the count and scale their history by its change.
    info.tg.add_task(daxa::HeadTask<RtLab2QueryH::Info>()
        .head_views(RtLab2QueryH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .cells = cells,
            .pixel_cell = pixel_cell,
            .pixel_cell2 = pixel_cell2,
        })
        .executes(rt_lab2_query_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2CellPassH::Info>("RtLab2Prepare")
        .head_views(RtLab2CellPassH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .cells = cells,
        })
        .executes(rt_lab2_prepare_callback, &render_context));

    // Temporal gradient (per cell): re-trace last frame's recorded rays (measurement per cell), then the spatial blur over one candidate pixel of every quad
    // integrates the blurred gradient into the cells. The cell integrate pass applies it (Kalman, history correction, sample count cut).
    bool const gradient = lab_state.lab2.settings.gradient_enabled != 0;
    daxa::TaskBufferView gradient_records = daxa::NullTaskBuffer;
    if (gradient)
    {
        gradient_records = info.tg.create_task_buffer({
            .size = sizeof(RtLab2GradientRecord) * RT_LAB2_GRADIENT_RECORDS * static_cast<u64>(rt_lab2_cell_capacity(lab_state.lab2.settings)),
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT,
            .name = "rt_lab2_gradient_records_persistent",
        });
        info.tg.add_task(daxa::HeadTask<RtLab2GradientTraceH::Info>()
            .head_views(RtLab2GradientTraceH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .cells = cells,
                .records = gradient_records,
                .mesh_instances = info.mesh_instances,
                .sky = info.sky,
                .sky_transmittance = info.sky_transmittance,
                .light_mask_volume = info.light_mask_volume,
                .pgi_color = info.pgi_color,
                .pgi_visibility = info.pgi_visibility,
                .pgi_info = info.pgi_info,
                .pgi_requests = info.pgi_requests,
                .tlas = info.tlas,
                .vsm_globals = info.vsm_globals,
                .vsm_point_lights = info.vsm_point_lights,
                .vsm_spot_lights = info.vsm_spot_lights,
                .vsm_memory_block = info.vsm_memory_block,
                .vsm_point_spot_page_table = info.vsm_point_spot_page_table,
            })
            .executes(rt_lab2_gradient_trace_callback, &render_context));
        info.tg.add_task(daxa::HeadTask<RtLab2GradientSpatialH::Info>()
            .head_views(RtLab2GradientSpatialH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = lab_depth.current(),
                .lab_face_normal = lab_face_normal.current(),
                .pixel_cell = pixel_cell,
                .pixel_cell2 = pixel_cell2,
                .cells = cells,
            })
            .executes(rt_lab2_gradient_spatial_callback, &render_context));
    }

    // Ray redistribution on quad candidates (see rt_lab2.inl): one candidate pixel per 2x2 quad, rotating every frame.
    auto const candidate_size = daxa::Extent3D{(size.x + 1) / 2, (size.y + 1) / 2, 1};
    auto candidate = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = candidate_size,
        .name = "rt_lab2_candidate",
    });
    auto quad_counters = info.tg.create_task_buffer({
        .size = sizeof(u32) * 4,
        .name = "rt_lab2_quad_counters",
    });
    info.tg.clear_buffer({.buffer = quad_counters, .name = "rt_lab2_quad_counters_clear"});
    info.tg.add_task(daxa::HeadTask<RtLab2SelectH::Info>()
        .head_views(RtLab2SelectH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_depth = lab_depth.current(),
            .pixel_cell = pixel_cell,
            .cells = cells,
            .candidate = candidate,
            .counters = quad_counters,
        })
        .executes(rt_lab2_select_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2DistributeH::Info>()
        .head_views(RtLab2DistributeH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .candidate = candidate,
            .counters = quad_counters,
            .ray_counters = ray_counters,
            .ray_list = ray_list,
            .ray_alloc = ray_alloc,
        })
        .executes(rt_lab2_distribute_callback, &render_context, ray_capacity));

    info.tg.add_task(daxa::HeadTask<RtLab2RaysDebugH::Info>()
        .head_views(RtLab2RaysDebugH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .lab_depth = lab_depth.current(),
            .ray_alloc = ray_alloc,
        })
        .executes(rt_lab2_rays_debug_callback, &render_context));

    info.tg.add_task(daxa::HeadTask<RtLabTraceH::Info>()
        .head_views(RtLabTraceH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .ray_counters = ray_counters,
            .ray_list = ray_list,
            .ray_alloc = ray_alloc,
            .rays = rays,
            .mesh_instances = info.mesh_instances,
            .sky = info.sky,
            .sky_transmittance = info.sky_transmittance,
            .light_mask_volume = info.light_mask_volume,
            .pgi_color = info.pgi_color,
            .pgi_visibility = info.pgi_visibility,
            .pgi_info = info.pgi_info,
            .pgi_requests = info.pgi_requests,
            .tlas = info.tlas,
            .vsm_globals = info.vsm_globals,
            .vsm_point_lights = info.vsm_point_lights,
            .vsm_spot_lights = info.vsm_spot_lights,
            .vsm_memory_block = info.vsm_memory_block,
            .vsm_point_spot_page_table = info.vsm_point_spot_page_table,
            .ray_lengths = daxa::NullTaskBuffer,
        })
        .executes(rt_lab_trace_callback, &render_context, ray_capacity, false));

    // Ray filter (RtLab2State::ray_filter_*, the old lab's pass): clamped rays for the pixel means that feed the cells.
    bool const ray_filter = settings.ray_filter_mode != RT_LAB_RAY_FILTER_OFF;
    daxa::TaskBufferView filtered_rays = daxa::NullTaskBuffer;
    if (ray_filter)
    {
        filtered_rays = info.tg.create_task_buffer({
            .size = sizeof(RtLabRay) * ray_capacity,
            .name = "rt_lab2_filtered_rays",
        });
        info.tg.add_task(daxa::HeadTask<RtLabRayFilterH::Info>()
            .head_views(RtLabRayFilterH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = lab_depth.current(),
                .lab_face_normal = lab_face_normal.current(),
                .ray_counters = ray_counters,
                .ray_list = ray_list,
                .ray_alloc = ray_alloc,
                .rays = rays,
                .filtered_rays = filtered_rays,
            })
            .executes(rt_lab_ray_filter_callback, &render_context, ray_capacity));
    }

    info.tg.add_task(daxa::HeadTask<RtLabGatherH::Info>()
        .head_views(RtLabGatherH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .ray_alloc = ray_alloc,
            .rays = rays,
            .lab_raw_diffuse = lab_raw_diffuse,
            .lab_depth = daxa::NullTaskImage,
            .lab_face_normal = daxa::NullTaskImage,
            .gradient_record_origin = daxa::NullTaskImage,
            .gradient_record_surface = daxa::NullTaskImage,
            .gradient_record_data = daxa::NullTaskImage,
            .filtered_rays = ray_filter ? filtered_rays : daxa::NullTaskBuffer,
        })
        .executes(rt_lab_gather_callback, &render_context, false, ray_filter, false));

    // The rays that win a record slot of their cell (re-traced next frame). After the re-trace of this frame, before the cell integrate.
    if (gradient)
    {
        info.tg.add_task(daxa::HeadTask<RtLab2GradientRecordH::Info>()
            .head_views(RtLab2GradientRecordH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = lab_depth.current(),
                .lab_face_normal = lab_face_normal.current(),
                .ray_counters = ray_counters,
                .ray_list = ray_list,
                .ray_alloc = ray_alloc,
                .rays = rays,
                .pixel_cell = pixel_cell,
                .cells = cells,
                .records = gradient_records,
            })
            .executes(rt_lab2_gradient_record_callback, &render_context, ray_capacity));
    }

    // The grid is the temporal integrator: pixel ray means -> cell sums -> cell history.
    info.tg.add_task(daxa::HeadTask<RtLab2AccumulateH::Info>()
        .head_views(RtLab2AccumulateH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_raw_diffuse = lab_raw_diffuse,
            .pixel_cell = pixel_cell,
            .pixel_cell2 = pixel_cell2,
            .cells = cells,
        })
        .executes(rt_lab2_accumulate_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2CellPassH::Info>("RtLab2CellIntegrate")
        .head_views(RtLab2CellPassH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .cells = cells,
        })
        .executes(rt_lab2_cell_integrate_callback, &render_context));

    info.tg.add_task(daxa::HeadTask<RtLab2LookupH::Info>()
        .head_views(RtLab2LookupH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .cells = cells,
            .lookup = lookup,
        })
        .executes(rt_lab2_lookup_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2IntegratorH::Info>()
        .head_views(RtLab2IntegratorH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .lab_depth_history = lab_depth.previous(),
            .lab_face_normal_history = lab_face_normal.previous(),
            .lookup = lookup,
            .history = integrator_history.previous(),
            .integrated = integrator_history.current(),
            .pixel_cell = pixel_cell,
            .cells = cells,
        })
        .executes(rt_lab2_integrator_callback, &render_context));

    // Grid statistics: counters + histograms of the live cells, after the cell integrate pass.
    if (lab_state.lab2.stats.enabled && !lab_state.lab2.stats.buffer.id().is_empty())
    {
        info.tg.register_buffer(lab_state.lab2.stats.buffer);
        info.tg.add_task(daxa::HeadTask<RtLab2StatsH::Info>()
            .head_views(RtLab2StatsH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .cells = cells,
                .stats = lab_state.lab2.stats.buffer.view(),
            })
            .executes(rt_lab2_stats_callback, &render_context));
    }

    // Live cell probes: read back the cell under the followed pixels.
    if (lab_state.lab2.probe.enabled && !lab_state.lab2.probe.buffer.id().is_empty())
    {
        info.tg.register_buffer(lab_state.lab2.probe.buffer);
        info.tg.add_task(daxa::HeadTask<RtLab2ProbeH::Info>()
            .head_views(RtLab2ProbeH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = lab_depth.current(),
                .lab_face_normal = lab_face_normal.current(),
                .cells = cells,
                .lookup = lookup,
                .integrated = integrator_history.current(),
                .probe = lab_state.lab2.probe.buffer.view(),
            })
            .executes(rt_lab2_probe_callback, &render_context));
    }

    // The old lab's resolve: a full res copy of the integrator output, RTGI_RADIANCE_SCALE divided out.
    auto resolved = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = size,
        .name = "rt_lab_resolved_diffuse",
    });
    info.tg.add_task(daxa::HeadTask<RtLabResolveH::Info>()
        .head_views(RtLabResolveH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .view_cam_depth = info.view_cam_depth,
            .view_cam_face_normals = info.view_cam_face_normals,
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .lab_diffuse = integrator_history.current(),
            .resolved = resolved,
            .lab_split_difference = daxa::NullTaskImage,
            .split_difference_resolved = daxa::NullTaskImage,
        })
        .executes(rt_lab_resolve_callback, &render_context, false, true));
    rt_lab_add_analysis(info, resolved);

    rt_lab_add_test_passes(info, resolved, size);

    return TasksRtgiMainResult{
        .opaque_diffuse = resolved,
        .opaque_specular = daxa::NullTaskImage,
    };
}

///
/// === Cell stage (RT Lab 4) ===
///

// The cell grid half of RT Lab 2 on a per pixel input (lab_raw_diffuse: .rgb radiance in RTGI_RADIANCE_SCALE units, .a = the pixel's sample weight): exposure
// (query, prepare), accumulate, cell integrate, lookup and the temporal resolve (integrator) at the end. No rays, no temporal gradient. Returns the
// integrator output (the resolve's lab_diffuse). lab_depth / lab_face_normal are the lab's PERSISTENT_DOUBLE_BUFFER geometry images.
auto rt_lab2_add_cell_stage(TasksRtgiInfo const & info, daxa::TaskImageView lab_depth, daxa::TaskImageView lab_face_normal, daxa::TaskImageView lab_raw_diffuse, daxa::Extent3D size) -> daxa::TaskImageView
{
    auto & render_context = info.render_context;
    auto & lab_state = render_context.rt_lab;
    lab_state.lab2.clear_requested = true; // the graph's persistent resources are undefined after a rebuild
    auto cells = info.tg.create_task_buffer({
        .size = sizeof(RtLab2Cell) * static_cast<u64>(rt_lab2_cell_capacity(lab_state.lab2.settings)),
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT,
        .name = "rt_lab2_cells_persistent",
    });
    auto pixel_cell = info.tg.create_task_image({.format = daxa::Format::R32_UINT, .size = size, .name = "rt_lab2_pixel_cell"});
    auto pixel_cell2 = info.tg.create_task_image({.format = daxa::Format::R32_UINT, .size = size, .name = "rt_lab2_pixel_cell2"});
    auto lookup = info.tg.create_task_image({.format = daxa::Format::R32G32B32A32_SFLOAT, .size = size, .name = "rt_lab2_lookup"});
    auto integrator_history = info.tg.create_task_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "rt_lab2_integrator_persistent",
    });

    info.tg.add_task(daxa::HeadTask<RtLab2CellPassH::Info>("RtLab2Reset")
        .head_views(RtLab2CellPassH::Info::Views{.globals = render_context.tgpu_render_data.view(), .cells = cells})
        .executes(rt_lab2_reset_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2QueryH::Info>()
        .head_views(RtLab2QueryH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .cells = cells,
            .pixel_cell = pixel_cell,
            .pixel_cell2 = pixel_cell2,
        })
        .executes(rt_lab2_query_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2CellPassH::Info>("RtLab2Prepare")
        .head_views(RtLab2CellPassH::Info::Views{.globals = render_context.tgpu_render_data.view(), .cells = cells})
        .executes(rt_lab2_prepare_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2AccumulateH::Info>()
        .head_views(RtLab2AccumulateH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_raw_diffuse = lab_raw_diffuse,
            .pixel_cell = pixel_cell,
            .pixel_cell2 = pixel_cell2,
            .cells = cells,
        })
        .executes(rt_lab2_accumulate_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2CellPassH::Info>("RtLab2CellIntegrate")
        .head_views(RtLab2CellPassH::Info::Views{.globals = render_context.tgpu_render_data.view(), .cells = cells})
        .executes(rt_lab2_cell_integrate_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2LookupH::Info>()
        .head_views(RtLab2LookupH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .cells = cells,
            .lookup = lookup,
        })
        .executes(rt_lab2_lookup_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLab2IntegratorH::Info>()
        .head_views(RtLab2IntegratorH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .lab_depth_history = lab_depth.previous(),
            .lab_face_normal_history = lab_face_normal.previous(),
            .lookup = lookup,
            .history = integrator_history.previous(),
            .integrated = integrator_history.current(),
            .pixel_cell = pixel_cell,
            .cells = cells,
        })
        .executes(rt_lab2_integrator_callback, &render_context));

    if (lab_state.lab2.stats.enabled && !lab_state.lab2.stats.buffer.id().is_empty())
    {
        info.tg.register_buffer(lab_state.lab2.stats.buffer);
        info.tg.add_task(daxa::HeadTask<RtLab2StatsH::Info>()
            .head_views(RtLab2StatsH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .cells = cells,
                .stats = lab_state.lab2.stats.buffer.view(),
            })
            .executes(rt_lab2_stats_callback, &render_context));
    }
    if (lab_state.lab2.probe.enabled && !lab_state.lab2.probe.buffer.id().is_empty())
    {
        info.tg.register_buffer(lab_state.lab2.probe.buffer);
        info.tg.add_task(daxa::HeadTask<RtLab2ProbeH::Info>()
            .head_views(RtLab2ProbeH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = lab_depth.current(),
                .lab_face_normal = lab_face_normal.current(),
                .cells = cells,
                .lookup = lookup,
                .integrated = integrator_history.current(),
                .probe = lab_state.lab2.probe.buffer.view(),
            })
            .executes(rt_lab2_probe_callback, &render_context));
    }
    return integrator_history.current();
}

///
/// === UI ===
///


// The grid, exposure, lookup and integrator settings (RT Lab 2 section; RT Lab 4 shows them too).
void rt_lab2_cells_ui(RenderContext & render_context)
{
    auto & lab2 = render_context.rt_lab.lab2;
    auto & s = lab2.settings;
    auto timing = [&](std::initializer_list<u32> indices)
    {
        u64 sum = 0;
        for (u32 i : indices)
        {
            sum += render_context.render_times.get_average(i);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%8.1f us", static_cast<f32>(sum) * 0.001f);
    };
    ImGui::SeparatorText("Cells");
    timing({RenderTimes::index<"RT_LAB", "LAB2_QUERY">(), RenderTimes::index<"RT_LAB", "LAB2_CELL_PREPARE">(), RenderTimes::index<"RT_LAB", "LAB2_ACCUMULATE">(), RenderTimes::index<"RT_LAB", "LAB2_CELL_INTEGRATE">()});
    // The slider only edits the pending value, the (graph rebuilding, grid clearing) change is applied when it is released.
    if (!ImGui::IsMouseDown(0))
    {
        lab2.capacity_ui = rt_lab2_cell_capacity_log2(s);
    }
    ImGui::SliderInt("Cell Capacity (2^n)", &lab2.capacity_ui, 16, 24, "2^%d");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        s.cell_capacity_log2 = static_cast<f32>(lab2.capacity_ui);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%u cells, %.0f MB", 1u << static_cast<u32>(lab2.capacity_ui), static_cast<f64>(sizeof(RtLab2Cell)) * static_cast<f64>(1u << static_cast<u32>(lab2.capacity_ui)) / (1024.0 * 1024.0));
    ImGui::SetItemTooltip("Number of hash grid cell slots = the grid's memory (the number next to the slider; plus the gradient's ray records when it is on). Raise it when the Statistics show a high load\n"
                          "(queries fail when the probe windows overflow). Applied when you release the slider: the graph rebuilds and the grid starts empty.");
    ImGui::SliderFloat("Cell Size (px)", &s.cell_size_px, 2.0f, 64.0f, "%.1f");
    ImGui::SetItemTooltip("Target cell size in pixels at the cell's distance: the cascade level is the smallest power of two multiple of\nMin Cell Size that reaches it (cells are between 1x and 2x this).");
    ImGui::SliderFloat("Min Cell Size (m)", &s.min_cell_size, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("History Samples / Pixel", &s.history_samples_per_pixel, 0.0f, 64.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Temporal adaption speed, normalized by the pixel coverage. A cell's history window (cap of its effective sample count) =\nthis x the cell's smoothed visible pixel count. Fewer pixels see the cell -> proportionally fewer rays AND a proportionally smaller window,\nso the adaption speed in frames is the same for every cell: about this / (rays per pixel) frames. 0 = off (cap = Max History Samples).");
    ImGui::SliderFloat("Min History Samples", &s.min_history_samples, 1.0f, 1024.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Lower limit of the window, so a tiny cell still averages a few rays.");
    ImGui::SliderFloat("Max History Samples", &s.max_history_samples, 1.0f, 65536.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Upper limit of the window (the cap itself when History Samples / Pixel is 0).");
    ImGui::SliderFloat("Max Frame Samples", &s.max_frame_samples, 0.0f, 4096.0f, "%.0f");
    ImGui::SetItemTooltip("Cap of the weight one frame gets in a cell's blend n_new / (n_old + n_new) (the frame's mean is unchanged, only how hard it pulls\n"
                          "the history): a frame can never move a cell by more than Max Frame Samples / (history + Max Frame Samples). 0 = off.");
    ImGui::SliderInt("Free After (frames)", &s.max_untouched_frames, 1, 600);
    ImGui::Checkbox("Draw Cell AABBs", reinterpret_cast<bool *>(&render_context.render_data.rtgi_settings.hash_grid_debug_draw_cells));
    ImGui::SetItemTooltip("Draw the world-space box of every cell queried this frame, colored per cell (same switch as the RTGI hash grid). Cells of different normal buckets share a box. Capped by the debug AABB draw capacity.");
    ImGui::SeparatorText("Exposure");
    ImGui::SetItemTooltip("Exposure = how many screen pixels see a cell, smoothed over frames.");
    ImGui::SliderFloat("Exposure Smoothing", &s.exposure_alpha, 0.01f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("Rate of the visible pixel count average.");
    ImGui::SliderFloat("Coverage Weighting", &s.exposure_strength, 0.0f, 2.0f, "%.2f");
    ImGui::SetItemTooltip("The new frame counts proportionally to its pixel coverage relative to the history's: the history samples are scaled by\n(last frame's coverage / this frame's coverage)^strength, in both directions. 1: coverage 4 -> 16 pixels = the new frame counts 4x more against\nthe history, 16 -> 4 = 4x less. (The frame's rays also scale with the coverage.) 0 = off.");
    ImGui::SliderFloat("Coverage Min Scale", &s.exposure_min_scale, 0.01f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("Limits of one frame's scaling: [this, 1 / this].");

    ImGui::SeparatorText("Lookup");
    timing({RenderTimes::index<"RT_LAB", "LAB2_LOOKUP">()});
    char const * const kernels[] = {"Tent (triangular, +-1 cell)", "Cubic B-Spline (+-2 cells)", "Gaussian (sigma 0.75 cells)"};
    ImGui::Combo("Kernel", &s.lookup_kernel, kernels, 3);
    ImGui::SetItemTooltip("Every tap draws a random offset from this distribution (per axis, in cells of the tap's cascade) and reads the cell at that\n"
                          "point. No weights, no deterministic taps: the mean of the taps is the filtered value. The cubic B-spline and the\n"
                          "gaussian are smoother than the tent (and wider).");
    ImGui::SliderInt("Taps", &s.lookup_taps, 1, RT_LAB2_MAX_LOOKUP_TAPS);
    ImGui::SetItemTooltip("Stochastic taps per pixel and frame (each also picks its cascade). Fewer = noisier lookup, the integrator averages it.");
    ImGui::SliderFloat("Kernel Width", &s.lookup_distance, 0.0f, 4.0f, "%.2f");
    ImGui::SetItemTooltip("Scales the kernel offsets: 1 = the kernel's own width in cells, > 1 wider (smoother, blurrier), < 1 narrower (0 = the pixel's own cell).");
    ImGui::SliderFloat("Cascade Blend Width", &s.level_blend_width, 0.0f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("Stochastic cascade selection. Width of the blend region around every level transition in log2 cell size units (0 = hard\n"
                          "levels, 1 = the whole level range). A tap picks the neighbor cascade with probability 0.5 exactly at the transition,\n"
                          "falling to 0 at the edge of the region. The pixels in the region feed both cascades.");
    ImGui::Checkbox("Project Taps On Surface", reinterpret_cast<bool *>(&s.lookup_project));
    ImGui::SetItemTooltip("Tap offsets are projected onto the pixel's surface plane, so taps land on cells of the same surface\n(3D offsets mostly miss: a wall is one cell thick).");
    ImGui::SliderInt("Min Cell Samples", &s.lookup_min_samples, 0, 64);
    ImGui::SetItemTooltip("Taps on cells with fewer effective samples are skipped.");

    ImGui::TextDisabled("Temporal resolve: see the Temporal Resolve section (shared by all pipelines).");
}

void rt_lab2_ui(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    auto & lab2 = lab.lab2;
    auto & s = lab2.settings;
    auto timing = [&](std::initializer_list<u32> indices)
    {
        u64 sum = 0;
        for (u32 i : indices)
        {
            sum += render_context.render_times.get_average(i);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%8.1f us", static_cast<f32>(sum) * 0.001f);
    };
    ImGui::PushID("rt lab 2");
    if (ImGui::Button("Reset Hash Grid Settings"))
    {
        static_cast<RtLab2Config &>(lab2) = RtLab2Config{};
    }
    ImGui::SliderFloat("Ray Budget (rays / px)", &lab2.ray_budget, 1.0f / 64.0f, 4.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Hard budget, full res pixels. Below 1 only that fraction of the pixels shoots (rotated Bayer selection per 8x8 tile),\n"
                          "above 1 every pixel shoots and the extra rays are spread. Graph rebuild when the rounded up value changes.");
    timing({RenderTimes::index<"RT_LAB", "RAY_DEMAND">(), RenderTimes::index<"RT_LAB", "RAY_DISTRIBUTE">(), RenderTimes::index<"RT_LAB", "TRACE">(), RenderTimes::index<"RT_LAB", "RAY_GATHER">()});

    ImGui::SeparatorText("Ray Redistribution");
    timing({RenderTimes::index<"RT_LAB", "LAB2_DEMAND">(), RenderTimes::index<"RT_LAB", "RAY_DISTRIBUTE">()});
    ImGui::Checkbox("Cell Driven Redistribution", reinterpret_cast<bool *>(&s.redistribution_enabled));
    ImGui::SetItemTooltip("Off: every quad candidate shoots 1 ray. On: a pixel asks for 1 base ray plus extras that fall off exponentially with how converged\n"
                          "its CELL is (RTGI demand curve): fresh, disoccluded or changed cells (their samples were scaled down) get the rays,\n"
                          "converged cells give them up. The frame's rays stay under the hard Ray Budget: extras are drained first, down to Min Base Coverage.\n"
                          "Debug view RTGI_DIFFUSE_RAYS_SHOT shows the rays per pixel (full heat = Max Rays Per Pixel).");
    ImGui::BeginDisabled(s.redistribution_enabled == 0);
    ImGui::SliderInt("Max Rays Per Pixel", &s.max_rays_per_pixel, 1, 32);
    ImGui::SliderFloat("Convergence Target", &s.redistribution_target, 0.05f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("Fraction of the cell's history window at which a cell stops asking for extra rays (max extras at 0 samples, exp(-4 x) falloff).");
    ImGui::SliderFloat("Min Base Coverage", &lab2.min_base_coverage, 0.0f, 1.0f, "%.3f");
    ImGui::SetItemTooltip("Fraction of the geometry pixels that keep their base ray when the extras drain the budget (paid first).");
    ImGui::EndDisabled();

    ImGui::SeparatorText("Ray Filter");
    timing({RenderTimes::index<"RT_LAB", "RAY_FILTER">()});
    char const * const ray_filter_modes[] = {"Off", "Absolute Clamp", "Perceptual Clamp (RTGI style)"};
    ImGui::Combo("Filter Mode", &lab2.ray_filter_mode, ray_filter_modes, 3);
    ImGui::SetItemTooltip("One thread per ray after the trace: statistics of all rays of the neighbor pixels (the ray itself excluded), the ray clamped\n"
                          "against them, hue preserving. The pixel means that feed the cells use the filtered rays.\n"
                          "Absolute: per channel ceiling = neighbors' linear mean x factor, the worst channel decides.\n"
                          "Perceptual: like the RTGI firefly clamp, per channel exp(mean of log rgb + tolerance), the worst channel decides.\n"
                          "Clamping removes energy (fireflies / rare bright paths): it biases darker.");
    ImGui::BeginDisabled(lab2.ray_filter_mode == RT_LAB_RAY_FILTER_OFF);
    ImGui::SliderInt("Filter Radius", &lab2.ray_filter_radius, 0, RT_LAB_RAY_FILTER_MAX_RADIUS);
    ImGui::SetItemTooltip("Neighbor pixels (2r + 1)^2 (2 = 5x5). 0 = only the pixel's own other rays.");
    if (lab2.ray_filter_mode == RT_LAB_RAY_FILTER_PERCEPTUAL)
    {
        ImGui::SliderFloat("Filter Tolerance (log)", &lab2.ray_filter_tolerance, 0.0f, 8.0f);
        ImGui::SetItemTooltip("Ceiling = exp(neighbors' mean log + this) per channel (RTGI firefly_perceptual_tolerance, default 4).");
    }
    else
    {
        ImGui::SliderFloat("Filter Factor", &lab2.ray_filter_factor, 1.0f, 1024.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Ceiling = neighbors' linear mean x this per channel (default 2^6 = 64).");
    }
    ImGui::EndDisabled();

    rt_lab2_cells_ui(render_context);
    ImGui::SeparatorText("Temporal Gradient (per cell)");
    timing({RenderTimes::index<"RT_LAB", "LAB2_GRADIENT_TRACE">(), RenderTimes::index<"RT_LAB", "LAB2_GRADIENT_SPATIAL">(), RenderTimes::index<"RT_LAB", "LAB2_GRADIENT_RECORD">()});
    ImGui::Checkbox("Gradient Enabled", reinterpret_cast<bool *>(&s.gradient_enabled));
    ImGui::SetItemTooltip("Every cell remembers up to 4 of its fresh rays (all 4 at Full Coverage pixels, fewer below) and re-traces them bit identically the next frame:\n"
                          "the luminance change is a measurement of how the lighting changed (ln new / old). A spatial pass over one candidate pixel of every 2x2 quad\n"
                          "blurs the measurements of the cells around it (stochastic, geometry aware, precision weighted) and integrates the result of all pixels of a\n"
                          "cell into one gradient value. The cell filters it (Kalman, innovation gate = instant reaction to shocks), applies the shrunk correction to its\n"
                          "history (x exp(c), glide for moderate changes) and cuts its sample count by the uncertainty that is left. Graph rebuild (the grid starts empty).\n"
                          "Debug views RT_LAB2_TEMPORAL_GRADIENT (applied correction + pending evidence: red = brighter, blue = darker) and _TEMPORAL_GRADIENT_LOSS.");
    ImGui::BeginDisabled(s.gradient_enabled == 0);
    ImGui::SliderFloat("Full Coverage (px)", &s.gradient_full_coverage, 1.0f, 512.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Pixel coverage at which a cell records the maximum of 4 rays per frame; smaller cells record proportionally fewer (at least 1).");
    ImGui::SliderInt("Blur Taps", &s.gradient_blur_taps, 0, 16);
    ImGui::SetItemTooltip("Stochastic taps of the spatial blur per half res pixel (plus the pixel itself).");
    ImGui::SliderFloat("Blur Radius (px)", &s.gradient_blur_radius, 1.0f, 128.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Min Variance", &s.gradient_min_variance, 1e-4f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Floor of the variance (log ratio squared) of one cell's measurement. A cell with 1 record has no spread estimate: this is its variance.");
    ImGui::SliderFloat("Evidence Decay", &s.gradient_kalman_q, 0.0f, 0.99f, "%.2f");
    ImGui::SliderFloat("Significance (sigmas)", &s.gradient_kalman_gate, 0.5f, 10.0f, "%.1f");
    ImGui::SetItemTooltip("A measurement further than this from the tracked change (in standard deviations) restarts the filter at the measurement: shocks (flashes) react at once.");
    ImGui::SliderFloat("Sample Count Cut k", &s.gradient_cap_k, 0.0f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("N' = N / (1 + N k L / v), L = the uncertainty left after the correction. Higher = the history is cut harder where the gradient is uncertain.");
    ImGui::SliderFloat("Large Change Cut Gain", &s.gradient_sample_variance, 0.0f, 100.0f, "%.1f");
    ImGui::SliderFloat("Applied Fraction", &s.gradient_glide, 0.0f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("Fraction of a moderate correction applied per frame (the rest stays pending); big changes (ln 2 .. ln 5 and above) are applied at once. 1 = off.");
    ImGui::Checkbox("Correct Integrator History", reinterpret_cast<bool *>(&s.gradient_correct_integrator));
    ImGui::SetItemTooltip("The integrator's reprojected history is scaled by the applied correction of the pixel's cell too. Without it the integrator's slow blend lags the cells.");
    ImGui::EndDisabled();
    ImGui::TextDisabled("Debug views: RT_LAB2_* in the debug view list.");
    ImGui::PopID();
}

// Live cell probes + grid statistics (runtime tools, not settings).
void rt_lab2_tools_ui(RenderContext & render_context)
{
    ImGui::PushID("rt lab 2 tools");
    ImGui::SeparatorText("Live Cell Probes");
    rt_lab2_probe_ui(render_context);
    ImGui::SeparatorText("Statistics");
    rt_lab2_stats_ui(render_context);
    ImGui::PopID();
}

///
/// === Test plan / automation ===
///

void rt_lab2_apply_json(RtLab2State & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys)
{
    auto number = [](nlohmann::json const & value) -> f64
    {
        return value.is_boolean() ? (value.get<bool>() ? 1.0 : 0.0) : value.get<f64>();
    };
    for (auto const & [key, value] : json.items())
    {
        if (key == "enabled")
        {
            state.enabled = number(value) != 0.0;
        }
        else if (key == "ray_budget")
        {
            state.ray_budget = static_cast<f32>(number(value));
        }
        else if (key == "min_base_coverage")
        {
            state.min_base_coverage = static_cast<f32>(number(value));
        }
        else if (key == "ray_filter_mode")
        {
            state.ray_filter_mode = static_cast<i32>(number(value));
        }
        else if (key == "ray_filter_radius")
        {
            state.ray_filter_radius = static_cast<i32>(number(value));
        }
        else if (key == "ray_filter_tolerance")
        {
            state.ray_filter_tolerance = static_cast<f32>(number(value));
        }
        else if (key == "ray_filter_factor")
        {
            state.ray_filter_factor = static_cast<f32>(number(value));
        }
#define RT_LAB2_JSON_FIELD(FIELD)                                                              \
    else if (key == #FIELD)                                                                    \
    {                                                                                          \
        state.settings.FIELD = static_cast<decltype(state.settings.FIELD)>(number(value));     \
    }
        RT_LAB2_JSON_FIELD(cell_size_px)
        RT_LAB2_JSON_FIELD(min_cell_size)
        RT_LAB2_JSON_FIELD(level_blend_width)
        RT_LAB2_JSON_FIELD(cell_capacity_log2)
        RT_LAB2_JSON_FIELD(gradient_enabled)
        RT_LAB2_JSON_FIELD(gradient_full_coverage)
        RT_LAB2_JSON_FIELD(gradient_blur_taps)
        RT_LAB2_JSON_FIELD(gradient_blur_radius)
        RT_LAB2_JSON_FIELD(gradient_min_variance)
        RT_LAB2_JSON_FIELD(gradient_kalman_q)
        RT_LAB2_JSON_FIELD(gradient_kalman_gate)
        RT_LAB2_JSON_FIELD(gradient_cap_k)
        RT_LAB2_JSON_FIELD(gradient_sample_variance)
        RT_LAB2_JSON_FIELD(gradient_glide)
        RT_LAB2_JSON_FIELD(gradient_correct_integrator)
        RT_LAB2_JSON_FIELD(redistribution_enabled)
        RT_LAB2_JSON_FIELD(max_rays_per_pixel)
        RT_LAB2_JSON_FIELD(redistribution_target)
        RT_LAB2_JSON_FIELD(max_untouched_frames)
        RT_LAB2_JSON_FIELD(history_samples_per_pixel)
        RT_LAB2_JSON_FIELD(min_history_samples)
        RT_LAB2_JSON_FIELD(max_history_samples)
        RT_LAB2_JSON_FIELD(max_frame_samples)
        RT_LAB2_JSON_FIELD(exposure_alpha)
        RT_LAB2_JSON_FIELD(exposure_strength)
        RT_LAB2_JSON_FIELD(exposure_min_scale)
        RT_LAB2_JSON_FIELD(lookup_kernel)
        RT_LAB2_JSON_FIELD(lookup_taps)
        RT_LAB2_JSON_FIELD(lookup_distance)
        RT_LAB2_JSON_FIELD(lookup_project)
        RT_LAB2_JSON_FIELD(lookup_min_samples)
#undef RT_LAB2_JSON_FIELD
        else
        {
            unknown_keys.push_back(key);
        }
    }
}
