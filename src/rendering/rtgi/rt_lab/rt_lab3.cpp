#include "rt_lab3.hpp"
#include "rt_lab_internal.hpp"
#include "rt_lab_integrator.hpp"

#include "rt_lab3.inl"
#include "rt_lab_resolve.inl"
#include "rt_lab_convergence.inl"
#include "rt_lab_measure.inl"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <imgui.h>
#include <nlohmann/json.hpp>

auto rt_lab3_default_settings() -> RtLab3Settings
{
    return RtLab3Settings{
        .temporal_enabled = 1,
        .temporal_max_m = 16.0f,
        .temporal_plane_tolerance = 2.0f,
        .temporal_normal_threshold = 0.9f,
        .temporal_visibility = 0,
        .temporal_max_age = 16,
        .temporal_partial_scale = 1.0f,
        .spatial_passes = 1,
        .spatial_taps = 4,
        .spatial_radius = 30.0f,
        .spatial_max_m = 30.0f,
        .spatial_plane_tolerance = 2.0f,
        .spatial_normal_threshold = 0.25f,
        .spatial_surface_test = RT_LAB3_SURFACE_DISTANCE,
        .spatial_distance_factor = 48.0f,
        .spatial_visibility = 1,
        .jacobian_max = 10.0f,
        .min_reconnection_distance = 0.05f,
        .reset = 0,
        .temporal_max_age_jitter = 0.5f,
    };
}

///
/// === Frame update ===
///

auto rt_lab3_frame_update(RenderContext & render_context, bool reference_phase) -> bool
{
    auto & lab3 = render_context.rt_lab.lab3;
    auto & settings = render_context.render_data.rt_lab_settings;
    // RT Lab 4 runs this ReSTIR stage too: it needs the same overrides of the lab settings.
    lab3.active = (lab3.enabled || lab3.classic_denoiser || render_context.rt_lab.lab4.enabled) && !reference_phase && settings.enabled != 0;
    if (lab3.active)
    {
        render_context.rt_lab.lab2.active = false; // RT Lab 3 wins when both are enabled
        lab3.settings.spatial_passes = std::clamp(lab3.settings.spatial_passes, 0, RT_LAB3_MAX_SPATIAL_PASSES);
        // The trace and the geometry stay the lab's (hit shading, AO on hit, noise); everything of the old denoiser is off.
        // Resolution: the preset's Half Resolution checkbox (settings.half_res stays as it is).
        settings.animate_noise = 1;
        // Ray redistribution = the preset's (Trace Settings). The ray demand reads last frame's output (.a = the pixel's accumulated rays shot) as the history
        // length: pixels below the target (fast_convergence_samples) ask for extra rays. The base coverage is exactly min_ray_budget (min AND max), the rest of
        // the budget only buys extra rays where the ray history is low.
        settings.ray_budget = std::max(settings.ray_budget, 1.0f / 64.0f);
        settings.max_base_coverage = settings.redistribution_enabled != 0 ? std::clamp(settings.min_ray_budget, 1.0f / 64.0f, 1.0f) : 0.0f;
        if (settings.redistribution_enabled != 0)
        {
            settings.min_ray_budget = settings.max_base_coverage;
        }
        // The classic denoiser after the ReSTIR stage keeps its temporal / spatial settings; what needs the lab's own ray buffers is off.
        bool const classic = lab3.classic_denoiser && !lab3.enabled && !render_context.rt_lab.lab4.enabled;
        settings.temporal_gradient_enabled = 0;
        settings.temporal_gradient_all_base_rays = 0;
        // The preset's ray filter (settings.ray_filter_*) stays: RT Lab 3 applies it AFTER the reuse, on the per pixel estimate (rt_lab3_filter).
        settings.sh_guide_mode = 0;
        if (!classic)
        {
            settings.temporal_enabled = 0;
            settings.split_history_enabled = 0;
            settings.spatial_enabled = 0;
            settings.coarse_anti_lag_enabled = 0;
            settings.distribution_guide_mode = RT_LAB_DISTRIBUTION_OFF;
            settings.luma_moments_enabled = 0;
        }
    }
    bool rebuild = lab3.active != lab3.built_active;
    lab3.built_active = lab3.active;
    // The number of spatial passes is part of the graph.
    if (lab3.settings.spatial_passes != lab3.built_spatial_passes)
    {
        rebuild = true;
        lab3.built_spatial_passes = lab3.settings.spatial_passes;
    }
    return rebuild;
}

///
/// === Callbacks ===
///

static auto rt_lab3_settings_address(daxa::TaskInterface ti, RenderContext * render_context) -> daxa::DeviceAddress
{
    auto & lab3 = render_context->rt_lab.lab3;
    RtLab3Settings settings = lab3.settings;
    settings.reset = lab3.frame_reset ? 1 : 0;
    auto alloc = ti.allocator->allocate(sizeof(RtLab3Settings)).value();
    std::memcpy(alloc.host_address, &settings, sizeof(RtLab3Settings));
    return alloc.device_address;
}

template <typename Push>
static void rt_lab3_dispatch_pixels(daxa::TaskInterface ti, RenderContext * render_context, Push push, daxa::TaskImageAttachmentIndex size_image, u32 render_time, std::string const & pipeline_name)
{
    push.settings = rt_lab3_settings_address(ti, render_context);
    rt_lab_dispatch(ti, render_context, push, size_image, RT_LAB3_PASS_X, render_time, pipeline_name);
}

// First pass of the frame (the ray demand): decides the frame's reset (a test phase start, or a freshly built graph whose persistent
// resources are undefined). The demand reads last frame's output only without a reset. All later passes read RtLab3State::frame_reset.
static void rt_lab3_demand_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto & lab3 = render_context->rt_lab.lab3;
    lab3.frame_reset = render_context->render_data.rt_lab_settings.reset_history != 0 || lab3.clear_requested;
    lab3.clear_requested = false;
    rt_lab_demand_callback(ti, render_context, !lab3.frame_reset, false);
}

// One thread per ray list entry: dispatched over the whole capacity, entries past this frame's rays exit.
static void rt_lab3_trace_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 ray_capacity)
{
    auto const & AT = RtLab3TraceH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RT_LAB", "TRACE">());
    auto const size = ti.info(AT.lab_depth).value().size;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab3_trace_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLab3TracePush push = {};
    push.attach = alloc.device_address;
    push.size = {size.x, size.y};
    push.ray_capacity = ray_capacity;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(ray_capacity, static_cast<u32>(RT_LAB_TRACE_THREADS)), 1, 1});
}

static void rt_lab3_temporal_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab3TemporalH::Info::AT;
    rt_lab3_dispatch_pixels(ti, render_context, RtLab3TemporalPush{}, AT.out_a, RenderTimes::index<"RT_LAB", "LAB3_TEMPORAL">(), rt_lab3_temporal_compile_info().name);
}

static void rt_lab3_spatial_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass_index)
{
    auto const & AT = RtLab3SpatialH::Info::AT;
    RtLab3SpatialPush push = {};
    push.pass_index = pass_index;
    rt_lab3_dispatch_pixels(ti, render_context, push, AT.out_a, RenderTimes::index<"RT_LAB", "LAB3_SPATIAL">(), rt_lab3_spatial_compile_info().name);
}

static void rt_lab3_shade_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab3ShadeH::Info::AT;
    RtLab3ShadePush push = {};
    rt_lab3_dispatch_pixels(ti, render_context, push, AT.shaded, RenderTimes::index<"RT_LAB", "LAB3_SHADE">(), rt_lab3_shade_compile_info().name);
}

static void rt_lab3_filter_callback(daxa::TaskInterface ti, RenderContext * render_context, bool cell_input_bound, u32 cell_weight_mode, u32 cell_source)
{
    auto const & AT = RtLab3FilterH::Info::AT;
    RtLab3FilterPush push = {};
    push.cell_input_bound = cell_input_bound ? 1u : 0u;
    push.cell_weight_mode = cell_weight_mode;
    push.cell_source = cell_source;
    rt_lab3_dispatch_pixels(ti, render_context, push, AT.filtered, RenderTimes::index<"RT_LAB", "LAB3_FILTER">(), rt_lab3_filter_compile_info().name);
}

///
/// === Task graph ===
///

auto rt_lab3_add_restir(TasksRtgiInfo const & info, i32 cell_weight_mode, i32 cell_source) -> RtLab3Stage
{
    auto & render_context = info.render_context;
    auto & lab_state = render_context.rt_lab;
    auto const & settings = render_context.render_data.rt_lab_settings; // the lab settings with the RT Lab 3 overrides applied
    auto const render_target_size = render_context.render_data.settings.render_target_size;
    u32 const lab_div = settings.half_res ? 2u : 1u;
    auto const size = daxa::Extent3D{render_target_size.x / lab_div, render_target_size.y / lab_div, 1};
    auto const full_size = daxa::Extent3D{render_target_size.x, render_target_size.y, 1};
    // The graph's persistent resources are undefined after a rebuild: the first frame resets them.
    lab_state.lab3.clear_requested = true;

    // Lab geometry (lab res), written by the ray demand. Double buffered: the temporal reuse reprojects against last frame's.
    auto lab_depth = info.tg.create_task_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "rt_lab3_depth_persistent",
    });
    auto lab_face_normal = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "rt_lab3_face_normal_persistent",
    });
    // The final output (.rgb estimate in RTGI_RADIANCE_SCALE units, .a = accumulated rays shot at the pixel): the resolve reads it, next frame's
    // ray demand reads .a as the history length.
    auto output = info.tg.create_task_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "rt_lab3_output_persistent",
    });

    // Reservoir images, see rt_lab3.inl. The temporal result is the history of the next frame (persistent), the spatial passes ping pong
    // between two transient sets.
    struct Reservoirs
    {
        daxa::TaskImageView a = {};
        daxa::TaskImageView b = {};
        daxa::TaskImageView c = {};
    };
    auto const create_reservoir_image = [&](char const * name, daxa::Format format, bool persistent) -> daxa::TaskImageView
    {
        return info.tg.create_task_image({
            .format = format,
            .size = size,
            .lifetime_type = persistent ? daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER : daxa::TaskResourceLifetimeType::TRANSIENT,
            .name = name,
        });
    };
    auto history_a = create_reservoir_image("rt_lab3_reservoir_a_persistent", daxa::Format::R32G32B32A32_SFLOAT, true);
    auto history_b = create_reservoir_image("rt_lab3_reservoir_b_persistent", daxa::Format::R32G32B32A32_SFLOAT, true);
    auto history_c = create_reservoir_image("rt_lab3_reservoir_c_persistent", daxa::Format::R32G32B32A32_UINT, true);
    Reservoirs const temporal = {history_a.current(), history_b.current(), history_c.current()};
    Reservoirs const previous = {history_a.previous(), history_b.previous(), history_c.previous()};
    i32 const spatial_passes = std::clamp(lab_state.lab3.settings.spatial_passes, 0, RT_LAB3_MAX_SPATIAL_PASSES);
    Reservoirs ping = {};
    Reservoirs pong = {};
    if (spatial_passes >= 1)
    {
        ping = {create_reservoir_image("rt_lab3_spatial_ping_a", daxa::Format::R32G32B32A32_SFLOAT, false),
                create_reservoir_image("rt_lab3_spatial_ping_b", daxa::Format::R32G32B32A32_SFLOAT, false),
                create_reservoir_image("rt_lab3_spatial_ping_c", daxa::Format::R32G32B32A32_UINT, false)};
    }
    if (spatial_passes >= 2)
    {
        pong = {create_reservoir_image("rt_lab3_spatial_pong_a", daxa::Format::R32G32B32A32_SFLOAT, false),
                create_reservoir_image("rt_lab3_spatial_pong_b", daxa::Format::R32G32B32A32_SFLOAT, false),
                create_reservoir_image("rt_lab3_spatial_pong_c", daxa::Format::R32G32B32A32_UINT, false)};
    }

    // Ray list (the old lab's passes): a selection of the lab pixels shoots.
    u32 const ray_capacity = size.x * size.y * rt_lab_ray_capacity_per_pixel(settings);
    auto ray_counters = info.tg.create_task_buffer({
        .size = sizeof(RtLabRayCounters),
        .name = "rt_lab3_ray_counters",
    });
    auto ray_list = info.tg.create_task_buffer({
        .size = sizeof(u32) * ray_capacity,
        .name = "rt_lab3_ray_list",
    });
    auto rays = info.tg.create_task_buffer({
        .size = sizeof(RtLab3Ray) * ray_capacity,
        .name = "rt_lab3_rays",
    });
    auto ray_demand = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = size,
        .name = "rt_lab3_ray_demand",
    });
    auto ray_alloc = info.tg.create_task_image({
        .format = daxa::Format::R32G32_UINT,
        .size = size,
        .name = "rt_lab3_ray_alloc",
    });
    info.tg.clear_buffer({.buffer = ray_counters, .name = "rt_lab3_ray_counters_clear"});

    // Ray redistribution (the old lab): the history is last frame's output, its .a (rays shot over time) the history length.
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
            .history = output.previous(),
            .ray_demand = ray_demand,
            .ray_counters = ray_counters,
            .debug_image = info.debug_image,
            .temporal_gradient = daxa::NullTaskImage,
            .variance_history = daxa::NullTaskImage,
        })
        .executes(rt_lab3_demand_callback, &render_context));
    info.tg.add_task(daxa::HeadTask<RtLabDistributeH::Info>()
        .head_views(RtLabDistributeH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .ray_demand = ray_demand,
            .ray_counters = ray_counters,
            .ray_list = ray_list,
            .ray_alloc = ray_alloc,
        })
        .executes(rt_lab_distribute_callback, &render_context, ray_capacity));

    info.tg.add_task(daxa::HeadTask<RtLab3TraceH::Info>()
        .head_views(RtLab3TraceH::Info::Views{
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
        })
        .executes(rt_lab3_trace_callback, &render_context, ray_capacity));

    // Initial candidates + temporal reuse.
    info.tg.add_task(daxa::HeadTask<RtLab3TemporalH::Info>()
        .head_views(RtLab3TemporalH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .lab_depth_history = lab_depth.previous(),
            .lab_face_normal_history = lab_face_normal.previous(),
            .ray_alloc = ray_alloc,
            .rays = rays,
            .history_a = previous.a,
            .history_b = previous.b,
            .history_c = previous.c,
            .out_a = temporal.a,
            .out_b = temporal.b,
            .out_c = temporal.c,
            .mesh_instances = info.mesh_instances,
            .tlas = info.tlas,
        })
        .executes(rt_lab3_temporal_callback, &render_context));

    // Spatial reuse: ping pong, not fed back into the history.
    Reservoirs final_reservoirs = temporal;
    for (i32 pass = 0; pass < spatial_passes; ++pass)
    {
        Reservoirs const & destination = (pass % 2 == 0) ? ping : pong;
        info.tg.add_task(daxa::HeadTask<RtLab3SpatialH::Info>()
            .head_views(RtLab3SpatialH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = lab_depth.current(),
                .lab_face_normal = lab_face_normal.current(),
                .in_a = final_reservoirs.a,
                .in_b = final_reservoirs.b,
                .in_c = final_reservoirs.c,
                .out_a = destination.a,
                .out_b = destination.b,
                .out_c = destination.c,
                .mesh_instances = info.mesh_instances,
                .tlas = info.tlas,
            })
            .executes(rt_lab3_spatial_callback, &render_context, static_cast<u32>(pass)));
        final_reservoirs = destination;
    }

    // Estimate of the final reservoir, then the ray filter on it.
    auto shaded = info.tg.create_task_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT,
        .size = size,
        .name = "rt_lab3_shaded",
    });
    info.tg.add_task(daxa::HeadTask<RtLab3ShadeH::Info>()
        .head_views(RtLab3ShadeH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .in_a = final_reservoirs.a,
            .in_b = final_reservoirs.b,
            .in_c = final_reservoirs.c,
            .shaded = shaded,
        })
        .executes(rt_lab3_shade_callback, &render_context));
    bool const cell_input_bound = cell_weight_mode >= 0;
    daxa::TaskImageView cell_input = daxa::NullTaskImage;
    if (cell_input_bound)
    {
        cell_input = info.tg.create_task_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = size,
            .name = "rt_lab3_cell_input",
        });
    }
    info.tg.add_task(daxa::HeadTask<RtLab3FilterH::Info>()
        .head_views(RtLab3FilterH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_depth = lab_depth.current(),
            .shaded = shaded,
            .filtered = output.current(),
            .cell_input = cell_input,
            .ray_alloc = ray_alloc,
            .rays = rays,
        })
        .executes(rt_lab3_filter_callback, &render_context, cell_input_bound, static_cast<u32>(std::max(cell_weight_mode, 0)), static_cast<u32>(std::max(cell_source, 0))));

    return RtLab3Stage{
        .lab_depth = lab_depth,
        .lab_face_normal = lab_face_normal,
        .output = output.current(),
        .cell_input = cell_input,
        .ray_alloc = ray_alloc,
        .size = size,
        .full_size = full_size,
    };
}


auto tasks_rt_lab3_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult
{
    auto & render_context = info.render_context;
    auto & lab_state = render_context.rt_lab;
    auto const stage = rt_lab3_add_restir(info, -1, 0);
    auto const & lab_depth = stage.lab_depth;
    auto const & lab_face_normal = stage.lab_face_normal;
    auto const & size = stage.size;
    auto const & full_size = stage.full_size;
    auto const & output = stage.output;

    // Temporal resolve (shared feature, rt_lab_integrator.hpp) on the filtered estimate. Its own history: the demand keeps reading `output` (.a = M).
    daxa::TaskImageView resolve_input = output;
    if (lab_state.integrator.enabled)
    {
        resolve_input = rt_lab_add_integrator(info, lab_state.integrator, lab_depth, lab_face_normal, output, size, "rt_lab3_integrator_persistent");
    }

    // The old lab's resolve: RTGI_RADIANCE_SCALE divided out, at half res the bilateral upsample to the full res output.
    auto resolved = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = full_size,
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
            .lab_diffuse = resolve_input,
            .resolved = resolved,
            .lab_split_difference = daxa::NullTaskImage,
            .split_difference_resolved = daxa::NullTaskImage,
        })
        .executes(rt_lab_resolve_callback, &render_context, false, true));
    rt_lab_add_analysis(info, resolved);

    rt_lab_add_test_passes(info, resolved, full_size);

    return TasksRtgiMainResult{
        .opaque_diffuse = resolved,
        .opaque_specular = daxa::NullTaskImage,
    };
}

///
/// === UI ===
///

// Everything of the RT Lab 3 section except the enable checkbox (RT Lab 4 shows it for its ReSTIR stage).
void rt_lab3_settings_ui(RenderContext & render_context)
{
    auto & lab3 = render_context.rt_lab.lab3;
    auto & s = lab3.settings;
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

    ImGui::TextDisabled("Resolution: the Half Resolution checkbox of the preset.");
    ImGui::TextDisabled("Ray budget and redistribution: the preset's Trace Settings (Ray Redistribution).");
    timing({RenderTimes::index<"RT_LAB", "RAY_DEMAND">(), RenderTimes::index<"RT_LAB", "RAY_DISTRIBUTE">(), RenderTimes::index<"RT_LAB", "TRACE">()});

    ImGui::SeparatorText("Temporal Reuse");
    timing({RenderTimes::index<"RT_LAB", "LAB3_TEMPORAL">()});
    bool temporal = s.temporal_enabled != 0;
    if (ImGui::Checkbox("Temporal", &temporal))
    {
        s.temporal_enabled = temporal ? 1 : 0;
    }
    ImGui::SliderFloat("Temporal Max M", &s.temporal_max_m, 1.0f, 128.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Cap of the confidence of the reprojected reservoir. Lower = the history forgets faster (less lag, more noise).");
    ImGui::SliderFloat("Temporal Plane Tolerance", &s.temporal_plane_tolerance, 0.1f, 8.0f, "%.2f px");
    ImGui::SliderFloat("Temporal Normal Threshold", &s.temporal_normal_threshold, 0.0f, 1.0f, "%.3f");
    ImGui::TextDisabled("Temporal reprojection: coherent stochastic bilinear (one valid tap of the 2x2 footprint, picked by its weight with one random number per frame for the whole image).");
    ImGui::SliderInt("Sample Age Cap (frames)", &s.temporal_max_age, 0, 256, s.temporal_max_age == 0 ? "off" : "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Per reservoir: frames since its sample was traced (0 = fresh ray, +1 per temporal reuse, spatial reuse copies the age of the sample). A reservoir whose sample would get older than the cap is not reused (its M is dropped, the pixel asks for more rays), so no sample lives longer than the cap. 0 = off.");
    ImGui::BeginDisabled(s.temporal_max_age == 0);
    ImGui::SliderFloat("Age Cap Jitter", &s.temporal_max_age_jitter, 0.0f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("Each reuse draws its own age limit: cap x (1 - jitter x random). At 0.5 a sample dies somewhere between 50%% and 100%% of the cap,\n"
                          "so the samples of a ray burst (all the same age) do not expire on the same frame. The cap stays the hard maximum. 0 = fixed cap.");
    ImGui::EndDisabled();
    ImGui::SliderFloat("Partial Match M Scale", &s.temporal_partial_scale, 0.0f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("A candidate whose footprint only partly matches (valid weight below 1, it overhangs an edge) keeps lerp(1, weight, this) of its M:\n"
                          "0 = full confidence, 1 = scaled by the weight. Limits how much a partially wrong reprojection can contaminate the pixel.");
    bool temporal_visibility = s.temporal_visibility != 0;
    if (ImGui::Checkbox("Temporal Visibility Ray", &temporal_visibility))
    {
        s.temporal_visibility = temporal_visibility ? 1 : 0;
    }
    ImGui::SetItemTooltip("Shadow ray from the pixel to the reused sample point. Removes light leaking of the history behind moving geometry, costs a ray per pixel.");

    ImGui::SeparatorText("Spatial Reuse");
    timing({RenderTimes::index<"RT_LAB", "LAB3_SPATIAL">()});
    ImGui::SliderInt("Spatial Passes", &s.spatial_passes, 0, RT_LAB3_MAX_SPATIAL_PASSES);
    ImGui::SetItemTooltip("Graph rebuild when it changes. Every pass merges neighbor reservoirs into the pixel's (ping pong). Not fed back into the history.");
    ImGui::SliderInt("Spatial Taps", &s.spatial_taps, 0, RT_LAB3_MAX_SPATIAL_TAPS);
    ImGui::SliderFloat("Spatial Radius", &s.spatial_radius, 1.0f, 64.0f, "%.1f px");
    ImGui::SliderFloat("Spatial Max M", &s.spatial_max_m, 1.0f, 128.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Cap of the confidence of a neighbor reservoir: keeps one very confident neighbor from drowning the pixel's own sample.");
    char const * surface_tests[] = {"Plane Distance", "World Space Distance"};
    ImGui::Combo("Spatial Surface Test", &s.spatial_surface_test, surface_tests, 2);
    ImGui::SetItemTooltip("How a neighbor is tested to lie on the receiver's surface. Plane: its distance from the receiver's tangent plane (accepts anything on the same plane, however far). World space distance: its plain distance from the receiver in pixel widths of the receiving pixel (keeps the reuse local). The normal threshold applies to both.");
    if (s.spatial_surface_test == RT_LAB3_SURFACE_PLANE)
    {
        ImGui::SliderFloat("Spatial Plane Tolerance", &s.spatial_plane_tolerance, 0.1f, 8.0f, "%.2f px");
    }
    else
    {
        ImGui::SliderFloat("Spatial Max Distance", &s.spatial_distance_factor, 0.1f, 64.0f, "%.1f px widths");
    }
    ImGui::SliderFloat("Spatial Normal Threshold", &s.spatial_normal_threshold, 0.0f, 1.0f, "%.3f");
    bool spatial_visibility = s.spatial_visibility != 0;
    if (ImGui::Checkbox("Spatial Visibility Ray", &spatial_visibility))
    {
        s.spatial_visibility = spatial_visibility ? 1 : 0;
    }
    ImGui::SetItemTooltip("Shadow ray from the pixel to every reused neighbor sample (one ray per tap and pass). Without it the reuse leaks light\n"
                          "through walls and around corners.");

    ImGui::SeparatorText("Reconnection");
    ImGui::SliderFloat("Jacobian Max", &s.jacobian_max, 1.0f, 100.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("A reused sample whose reconnection Jacobian is outside [1 / max, max] is rejected (grazing / very different distances).");
    ImGui::SliderFloat("Min Reconnection Distance", &s.min_reconnection_distance, 0.0f, 1.0f, "%.3f m");
    ImGui::SetItemTooltip("Samples closer than this to the receiving pixel are not reused (their Jacobian explodes).");

    timing({RenderTimes::index<"RT_LAB", "LAB3_SHADE">(), RenderTimes::index<"RT_LAB", "LAB3_FILTER">()});
    ImGui::TextDisabled("Ray filter: the Ray Filter of Trace Settings, applied after the reuse on the per pixel estimate.");
    ImGui::TextDisabled("Debug view: RT_LAB3_SAMPLE_COUNT (rays shot at the pixel over time, saturating at the target).");
}

// The ReSTIR section of a preset's settings (the settings are the live copy of that preset's config, see rt_lab_preset_config_ui).
void rt_lab3_ui(RenderContext & render_context)
{
    auto & lab3 = render_context.rt_lab.lab3;
    ImGui::PushID("rt lab 3");
    if (ImGui::Button("Reset ReSTIR Settings"))
    {
        static_cast<RtLab3Config &>(lab3) = RtLab3Config{};
    }
    rt_lab3_settings_ui(render_context);
    ImGui::PopID();
}

///
/// === Automation ===
///

void rt_lab3_apply_json(RtLab3State & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys)
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
        else if (key == "half_res")
        {
            state.half_res = number(value) != 0.0;
        }
#define RT_LAB3_JSON_FIELD(FIELD)                                                              \
    else if (key == #FIELD)                                                                    \
    {                                                                                          \
        state.settings.FIELD = static_cast<decltype(state.settings.FIELD)>(number(value));     \
    }
        RT_LAB3_JSON_FIELD(temporal_enabled)
        RT_LAB3_JSON_FIELD(temporal_max_m)
        RT_LAB3_JSON_FIELD(temporal_plane_tolerance)
        RT_LAB3_JSON_FIELD(temporal_normal_threshold)
        RT_LAB3_JSON_FIELD(temporal_visibility)
        RT_LAB3_JSON_FIELD(temporal_max_age)
        RT_LAB3_JSON_FIELD(temporal_partial_scale)
        RT_LAB3_JSON_FIELD(temporal_max_age_jitter)
        RT_LAB3_JSON_FIELD(spatial_passes)
        RT_LAB3_JSON_FIELD(spatial_taps)
        RT_LAB3_JSON_FIELD(spatial_radius)
        RT_LAB3_JSON_FIELD(spatial_max_m)
        RT_LAB3_JSON_FIELD(spatial_plane_tolerance)
        RT_LAB3_JSON_FIELD(spatial_normal_threshold)
        RT_LAB3_JSON_FIELD(spatial_surface_test)
        RT_LAB3_JSON_FIELD(spatial_distance_factor)
        RT_LAB3_JSON_FIELD(spatial_visibility)
        RT_LAB3_JSON_FIELD(jacobian_max)
        RT_LAB3_JSON_FIELD(min_reconnection_distance)
#undef RT_LAB3_JSON_FIELD
        else
        {
            unknown_keys.push_back(key);
        }
    }
}
