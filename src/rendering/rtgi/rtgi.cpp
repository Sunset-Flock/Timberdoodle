#include "rtgi.hpp"

#include "rtgi_trace_diffuse.inl"

auto rtgi_default_settings() -> RtgiSettings
{
    return RtgiSettings{
        .enabled                              = 1,
        .ao_range                             = 1.0f,
        .ray_samples                          = 1,
        .firefly_filter_enabled               = 1,
        .firefly_filter_ceiling               = 24.0f,
        .firefly_clamp_mode                   = 0,
        .pre_blur_enabled                     = 1,
        .pre_blur_raylength_guiding           = 1,
        .geometric_luma_guiding      = 1,
        .geometric_luma_guiding_factor = 0.7f,
        .pre_blur_base_width                  = 64.0f,
        .pre_blur_sample_count                = 10,
        .pre_blur_iterations                  = 2,
        .temporal_accumulation_enabled        = 1,
        .temporal_fast_history_enabled        = 1,
        .temporal_firefly_filter_enabled      = 1,
        .temporal_firefly_std_dev_clamp       = 2.0f,
        .temporal_variance_fast_history_blend = 2.0f,
        .history_frames                       = 64,
        .post_blur_enabled                    = 1,
        .post_blur_raylength_guiding          = 1,
        .post_blur_geometric_luma_guiding     = 0,
        .post_blur_geometric_luma_guiding_factor = 0.7f,
        .raylength_guide_floor                = 0.2f,
        .post_blur_mode                       = 0,
        .post_blur_variance_guiding           = 1,
        .post_blur_disocclusion_blur_enabled  = 1,
        .post_blur_stride                     = 2,
        .post_blur_max_width                  = 12,
        .post_blur_atrous_iterations          = 4,
        .upscale_enabled                      = 1,
        .sh_resolve_enabled                   = 1,
        .use_compute_trace                    = 0,
        .firefly_center_blur_enabled          = 1,
        .firefly_energy_compensation_enabled  = 1,
        .animate_noise                        = 1,
    };
}
#include "rtgi_pre_filter.inl"
#include "rtgi_pre_blur.inl"
#include "rtgi_temporal.inl"
#include "rtgi_post_blur.inl"
#include "rtgi_upscale.inl"

///
/// === Callbacks ===
///

struct RtgiTraceDiffuseCallbackInfo
{
    bool debug_primary_trace = false;
};
inline void rtgi_trace_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context, RtgiTraceDiffuseCallbackInfo const & info)
{
    auto const & AT = RtgiTraceDiffuseH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RTGI", "TRACE">());
    RtgiTraceDiffusePush push = {
        .debug_primary_trace = static_cast<daxa::b32>(info.debug_primary_trace),
    };
    push.attach = ti.allocator->allocate_fill(RtgiTraceDiffuseH::AttachmentShaderBlob{ti.attachment_shader_blob}).value().device_address;
    auto const & diffuse_raw = ti.info(AT.diffuse_raw).value();
    if (render_context->render_data.rtgi_settings.use_compute_trace)
    {
        auto const & pipeline = render_context->gpu_context->compute_pipelines.at(rtgi_trace_diffuse_compute_compile_info().name);
        ti.recorder.set_pipeline(*pipeline);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({
            round_up_div(diffuse_raw.size.x, 8u),
            round_up_div(diffuse_raw.size.y, 8u),
            1,
        });
    }
    else
    {
        auto const & rt_pipeline = render_context->gpu_context->ray_tracing_pipelines.at(rtgi_trace_diffuse_compile_info().name);
        ti.recorder.set_pipeline(*rt_pipeline.pipeline);
        ti.recorder.push_constant(push);
        ti.recorder.trace_rays({
            .width = diffuse_raw.size.x,
            .height = diffuse_raw.size.y,
            .depth = 1,
            .shader_binding_table = rt_pipeline.sbt,
        });
    }
}

template <typename TaskPush>
inline void dispatch_image_relative(TaskPush push, daxa::TaskInterface ti, RenderContext * render_context, daxa::TaskImageAttachmentIndex dst_image, u32 block_size, u32 render_time, std::string const & compute_pipeline_name)
{
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, render_time);
    auto const dst_image_size = ti.info(dst_image).value().size;
    ti.recorder.set_pipeline(*(render_context->gpu_context->compute_pipelines.at(compute_pipeline_name)));
    if constexpr (std::is_same_v<decltype(push.attach), daxa::DeviceAddress>)
    {
        auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
        std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
        push.attach = alloc.device_address;
    }
    else
    {
        push.attach = ti.attachment_shader_blob;
    }
    push.size = {dst_image_size.x, dst_image_size.y};
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({
        round_up_div(dst_image_size.x, block_size),
        round_up_div(dst_image_size.y, block_size),
        1,
    });
}

inline void rtgi_temporal_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiTemporalH::Info::AT;
    dispatch_image_relative(RtgiTemporalPush(), ti, render_context, AT.half_res_sample_count, RTGI_TEMPORAL_X, RenderTimes::index<"RTGI", "TEMPORAL_ACCUMULATION">(), rtgi_temporal_compile_info().name);
}

inline void rtgi_pre_filter_prepare_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiPreFilterH::Info::AT;
    dispatch_image_relative(RtgiPreFilterPush(), ti, render_context, AT.diffuse_raw, RTGI_PRE_BLUR_PREPARE_X, RenderTimes::index<"RTGI", "PRE_FILTER">(), rtgi_pre_filter_prepare_compile_info().name);
}

inline void rtgi_pre_blur_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass)
{
    auto const & AT = RtgiPreBlurH::Info::AT;
    u32 const render_time = [&]() -> u32 {
        switch (pass)
        {
            case 1:  return RenderTimes::index<"RTGI", "PRE_BLUR1">();
            case 2:  return RenderTimes::index<"RTGI", "PRE_BLUR2">();
            case 3:  return RenderTimes::index<"RTGI", "PRE_BLUR3">();
            default: return RenderTimes::index<"RTGI", "PRE_BLUR0">();
        }
    }();
    dispatch_image_relative(RtgiPreBlurPush{.iteration = pass}, ti, render_context, AT.view_cam_half_res_depth, RTGI_PRE_BLUR_X, render_time, rtgi_pre_blur_compile_info().name);
}

inline void rtgi_post_blur_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass)
{
    auto const & AT = RtgiPostBlurH::Info::AT;
    u32 const render_time = pass == 0 ? RenderTimes::index<"RTGI", "POST_BLUR_VERTICAL">() : RenderTimes::index<"RTGI", "POST_BLUR_HORIZONTAL">();
    dispatch_image_relative(RtgiPostBlurPush{.pass = pass}, ti, render_context, AT.view_cam_half_res_depth, RTGI_POST_BLUR_X, render_time, rtgi_post_blur_compile_info().name);
}

inline void rtgi_atrous_post_blur_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass)
{
    auto const & AT = RtgiPostBlurH::Info::AT;
    u32 const render_time = [&]() -> u32 {
        switch (pass)
        {
            case 1:  return RenderTimes::index<"RTGI", "POST_BLUR_ATROUS1">();
            case 2:  return RenderTimes::index<"RTGI", "POST_BLUR_ATROUS2">();
            case 3:  return RenderTimes::index<"RTGI", "POST_BLUR_ATROUS3">();
            case 4:  return RenderTimes::index<"RTGI", "POST_BLUR_ATROUS4">();
            case 5:  return RenderTimes::index<"RTGI", "POST_BLUR_ATROUS5">();
            case 6:  return RenderTimes::index<"RTGI", "POST_BLUR_ATROUS6">();
            case 7:  return RenderTimes::index<"RTGI", "POST_BLUR_ATROUS7">();
            default: return RenderTimes::index<"RTGI", "POST_BLUR_ATROUS0">();
        }
    }();
    i32 const step_size = 1 << pass;
    dispatch_image_relative(RtgiAtrousPostBlurPush{.step_size = step_size}, ti, render_context, AT.view_cam_half_res_depth, RTGI_POST_BLUR_X, render_time, rtgi_atrous_post_blur_compile_info().name);
}

inline void rtgi_upscale_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiUpscaleDiffuseH::Info::AT;
    dispatch_image_relative(RtgiUpscaleDiffusePush(), ti, render_context, AT.view_cam_depth, RTGI_UPSCALE_DIFFUSE_X, RenderTimes::index<"RTGI", "UPSCALE">(), rtgi_upscale_diffuse_compile_info().name);
}

///
/// === Transient Images ===
///

auto rtgi_create_common_transient_image_info(RenderContext * render_context, daxa::Format format, daxa::u32 size_div, std::string_view name) -> daxa::TaskImageInfo
{
    return daxa::TaskImageInfo{
        .format = format,
        .size = {
            render_context->render_data.settings.render_target_size.x / size_div,
            render_context->render_data.settings.render_target_size.y / size_div,
            1,
        },
        .name = name,
    };
}

// rgb = diffuse radiance, a = ray travel distance
auto rtgi_create_trace_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_task_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, scale_div, name));
}

auto rtgi_create_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_task_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, scale_div, name));
}

auto rtgi_create_diffuse2_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_task_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16_SFLOAT, scale_div, name));
}

auto rtgi_create_upscaled_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_task_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, 1, name));
}

auto rtgi_create_sample_count_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_task_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16_SFLOAT, scale_div, name));
}

auto tasks_rtgi_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult
{
    auto half_res_image_size = daxa::Extent3D{
        info.render_context.render_data.settings.render_target_size.x / 2,
        info.render_context.render_data.settings.render_target_size.y / 2,
        1,
    };
    auto half_res_depth_history = info.tg.create_task_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT,
        .name = "rtgi_half_res_depth_history_persistent",
    });
    auto half_res_face_normal_history = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT,
        .name = "half_res_face_normal_history_persistent",
    });

    auto half_res_sample_count_history = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "half_res_sample_count_history_persistent",
    });
    auto half_res_diffuse_history = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "half_res_diffuse_history_persistent",
    });
    auto half_res_diffuse2_history = info.tg.create_task_image({
        .format = daxa::Format::R16G16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "half_res_diffuse2_history_persistent",
    });
    auto statistics_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "statistics_image_persistent",
    });
    auto half_res_filter_guide_history = info.tg.create_task_image({
        .format = daxa::Format::R8_UNORM,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "half_res_filter_guide_history_persistent",
    });
    auto temporal_geometric_mean_history = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "temporal_geometric_mean_history_persistent",
    });

    auto trace_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_diffuse_raw_image");
    auto trace_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_diffuse2_raw_image");
    auto ray_length_image = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "ray_length_image",
    });
    info.tg.add_task(daxa::HeadTask<RtgiTraceDiffuseH::Info>()
            .head_views(RtgiTraceDiffuseH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .diffuse_raw = trace_diffuse_image,
                .diffuse2_raw = trace_diffuse2_image,
                .ray_length_image = ray_length_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                .meshlet_instances = info.meshlet_instances,
                .mesh_instances = info.mesh_instances,
                .sky = info.sky,
                .sky_transmittance = info.sky_transmittance,
                .light_mask_volume = info.light_mask_volume,
                .pgi_irradiance = info.pgi_irradiance,
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
            .executes(rtgi_trace_diffuse_callback, &info.render_context, RtgiTraceDiffuseCallbackInfo{.debug_primary_trace = false}));

    // auto half_res_sample_count_image = rtgi_create_sample_count_image(info.tg, &info.render_context, "half_res_sample_count_image");

    auto pre_filtered_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "pre_filtered_diffuse_image");
    auto pre_filtered_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "pre_filtered_diffuse2_image");
    auto firefly_factor_image = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "firefly_factor_image",
    });    
    auto geo_mean_perceptual_image = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "geo_mean_perceptual_image",
    });
    auto filter_guide_image = info.tg.create_task_image({
        .format = daxa::Format::R8_UNORM,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "filter_guide_image",
    });
    info.tg.add_task(daxa::HeadTask<RtgiPreFilterH::Info>()
            .head_views(RtgiPreFilterH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .diffuse_raw = trace_diffuse_image,
                .diffuse2_raw = trace_diffuse2_image,
                .ray_length_image = ray_length_image,
                .view_cam_half_res_normals = info.view_cam_half_res_face_normals,
                .view_cam_half_res_albedo = info.view_cam_half_res_albedo,
                .pre_filtered_diffuse_image = pre_filtered_diffuse_image,
                .pre_filtered_diffuse2_image = pre_filtered_diffuse2_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
                .firefly_factor_image = firefly_factor_image,
                .geo_mean_perceptual_image = geo_mean_perceptual_image,
                .filter_guide_image = filter_guide_image,
            })
            .executes(rtgi_pre_filter_prepare_callback, &info.render_context));

    daxa::TaskImageView post_pre_blur_diffuse_image = daxa::NullTaskImage;
    daxa::TaskImageView post_pre_blur_diffuse2_image = daxa::NullTaskImage;
    if (info.render_context.render_data.rtgi_settings.pre_blur_enabled)
    {
        u32 const iterations = static_cast<u32>(info.render_context.render_data.rtgi_settings.pre_blur_iterations);

        auto pre_blurred_diffuse_ping = rtgi_create_diffuse_image(info.tg, &info.render_context, "pre_blurred_diffuse_ping");
        auto pre_blurred_diffuse2_ping = rtgi_create_diffuse2_image(info.tg, &info.render_context, "pre_blurred_diffuse2_ping");
        daxa::TaskImageView pre_blurred_diffuse_pong = daxa::NullTaskImage;
        daxa::TaskImageView pre_blurred_diffuse2_pong = daxa::NullTaskImage;
        if (iterations >= 2)
        {
            pre_blurred_diffuse_pong = rtgi_create_diffuse_image(info.tg, &info.render_context, "pre_blurred_diffuse_pong");
            pre_blurred_diffuse2_pong = rtgi_create_diffuse2_image(info.tg, &info.render_context, "pre_blurred_diffuse2_pong");
        }

        daxa::TaskImageView src_diffuse = pre_filtered_diffuse_image;
        daxa::TaskImageView src_diffuse2 = pre_filtered_diffuse2_image;
        for (u32 i = 0; i < iterations; ++i)
        {
            daxa::TaskImageView dst_diffuse = (i % 2 == 0) ? pre_blurred_diffuse_ping : pre_blurred_diffuse_pong;
            daxa::TaskImageView dst_diffuse2 = (i % 2 == 0) ? pre_blurred_diffuse2_ping : pre_blurred_diffuse2_pong;
            info.tg.add_task(daxa::HeadTask<RtgiPreBlurH::Info>(std::string("RtgiPreBlur") + std::to_string(i))
                    .head_views(RtgiPreBlurH::Info::Views{
                        .globals = info.render_context.tgpu_render_data.view(),
                        .debug_image = info.debug_image,
                        .clocks_image = info.clocks_image,
                        .rtgi_diffuse_before = src_diffuse,
                        .rtgi_diffuse2_before = src_diffuse2,
                        .view_cam_half_res_depth = info.view_cam_half_res_depth,
                        .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
        
                        .rtgi_diffuse_blurred = dst_diffuse,
                        .rtgi_diffuse2_blurred = dst_diffuse2,
                        .firefly_factor_image = firefly_factor_image,
                        .geo_mean_perceptual_image = geo_mean_perceptual_image,
                        .filter_guide_image = filter_guide_image,
                    })
                    .executes(rtgi_pre_blur_diffuse_callback, &info.render_context, i));
            src_diffuse = dst_diffuse;
            src_diffuse2 = dst_diffuse2;
        }
        post_pre_blur_diffuse_image = src_diffuse;
        post_pre_blur_diffuse2_image = src_diffuse2;
    }

info.tg.add_task(daxa::HeadTask<RtgiTemporalH::Info>()
        .head_views(RtgiTemporalH::Info::Views{
            .globals = info.render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .clocks_image = info.clocks_image,
            .half_res_sample_count = half_res_sample_count_history.current(),
            .half_res_sample_count_history = half_res_sample_count_history.previous(),
            .half_res_diffuse_new = post_pre_blur_diffuse_image,
            .pre_filtered_diffuse_new = pre_filtered_diffuse_image,
            .pre_filtered_diffuse2_new = pre_filtered_diffuse2_image,
            .half_res_diffuse_accumulated = half_res_diffuse_history.current(),
            .half_res_diffuse_history = half_res_diffuse_history.previous(),
            .half_res_diffuse2_new = post_pre_blur_diffuse2_image,
            .half_res_diffuse2_accumulated = half_res_diffuse2_history.current(),
            .half_res_diffuse2_history = half_res_diffuse2_history.previous(),
            .statistics_image_accumulated = statistics_image.current(),
            .statistics_image_history = statistics_image.previous(),
            .half_res_depth = info.view_cam_half_res_depth,
            .half_res_depth_history = half_res_depth_history,
            .half_res_normal = info.view_cam_half_res_face_normals,
            .half_res_normal_history = half_res_face_normal_history,
            .filter_guide_new = filter_guide_image,
            .half_res_filter_guide_accumulated = half_res_filter_guide_history.current(),
            .half_res_filter_guide_history = half_res_filter_guide_history.previous(),
            .geometric_mean_new = geo_mean_perceptual_image,
            .temporal_geometric_mean_accumulated = temporal_geometric_mean_history.current(),
            .temporal_geometric_mean_history = temporal_geometric_mean_history.previous(),
        })
        .executes(rtgi_temporal_callback, &info.render_context));

    auto rtgi_post_blur_diffuse_image = half_res_diffuse_history.current();
    auto rtgi_post_blur_diffuse2_image = half_res_diffuse2_history.current();
    if (info.render_context.render_data.rtgi_settings.post_blur_enabled)
    {
        auto const & rtgi_settings = info.render_context.render_data.rtgi_settings;
        if (rtgi_settings.post_blur_mode == 0)
        {
            // Bilateral separable: vertical then horizontal pass
            auto rtgi_post_blur_pass0_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_post_blur_pass0_diffuse_image");
            auto rtgi_post_blur_pass0_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_post_blur_pass0_diffuse2_image");
            info.tg.add_task(daxa::HeadTask<RtgiPostBlurH::Info>("RtgiPostBlurVertical")
                    .head_views(RtgiPostBlurH::Info::Views{
                        .globals = info.render_context.tgpu_render_data.view(),
                        .debug_image = info.debug_image,
                        .clocks_image = info.clocks_image,
                        .rtgi_sample_count = half_res_sample_count_history,
                        .rtgi_diffuse_before = half_res_diffuse_history,
                        .rtgi_diffuse2_before = half_res_diffuse2_history,
                        .view_cam_half_res_depth = info.view_cam_half_res_depth,
                        .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
        
                        .rtgi_diffuse_blurred = rtgi_post_blur_pass0_diffuse_image,
                        .rtgi_diffuse2_blurred = rtgi_post_blur_pass0_diffuse2_image,
                        .statistics_image = statistics_image,
                        .geo_mean_perceptual_image = geo_mean_perceptual_image,
                        .filter_guide_image = half_res_filter_guide_history.current(),
                        .temporal_geometric_mean = temporal_geometric_mean_history.current(),
                    })
                    .executes(rtgi_post_blur_diffuse_callback, &info.render_context, 0u));

            rtgi_post_blur_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_post_blur_diffuse_image");
            rtgi_post_blur_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_post_blur_diffuse2_image");
            info.tg.add_task(daxa::HeadTask<RtgiPostBlurH::Info>("RtgiPostBlurHorizontal")
                    .head_views(RtgiPostBlurH::Info::Views{
                        .globals = info.render_context.tgpu_render_data.view(),
                        .debug_image = info.debug_image,
                        .clocks_image = info.clocks_image,
                        .rtgi_sample_count = half_res_sample_count_history,
                        .rtgi_diffuse_before = rtgi_post_blur_pass0_diffuse_image,
                        .rtgi_diffuse2_before = rtgi_post_blur_pass0_diffuse2_image,
                        .view_cam_half_res_depth = info.view_cam_half_res_depth,
                        .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
        
                        .rtgi_diffuse_blurred = rtgi_post_blur_diffuse_image,
                        .rtgi_diffuse2_blurred = rtgi_post_blur_diffuse2_image,
                        .statistics_image = statistics_image,
                        .geo_mean_perceptual_image = geo_mean_perceptual_image,
                        .filter_guide_image = half_res_filter_guide_history.current(),
                        .temporal_geometric_mean = temporal_geometric_mean_history.current(),
                    })
                    .executes(rtgi_post_blur_diffuse_callback, &info.render_context, 1u));
        }
        else
        {
            // À-trous: N passes ping-ponging, step_size doubles each pass (1, 2, 4, ...)
            u32 const iterations = static_cast<u32>(rtgi_settings.post_blur_atrous_iterations);

            auto atrous_ping = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_atrous_ping");
            auto atrous_ping2 = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_atrous_ping2");
            daxa::TaskImageView atrous_pong = daxa::NullTaskImage;
            daxa::TaskImageView atrous_pong2 = daxa::NullTaskImage;
            if (iterations >= 2)
            {
                atrous_pong = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_atrous_pong");
                atrous_pong2 = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_atrous_pong2");
            }

            daxa::TaskImageView src = half_res_diffuse_history.current();
            daxa::TaskImageView src2 = half_res_diffuse2_history.current();
            for (u32 i = 0; i < iterations; ++i)
            {
                daxa::TaskImageView dst  = (i % 2 == 0) ? atrous_ping  : atrous_pong;
                daxa::TaskImageView dst2 = (i % 2 == 0) ? atrous_ping2 : atrous_pong2;
                info.tg.add_task(daxa::HeadTask<RtgiPostBlurH::Info>(std::string("RtgiAtrousPostBlur") + std::to_string(i))
                        .head_views(RtgiPostBlurH::Info::Views{
                            .globals = info.render_context.tgpu_render_data.view(),
                            .debug_image = info.debug_image,
                            .clocks_image = info.clocks_image,
                            .rtgi_sample_count = half_res_sample_count_history,
                            .rtgi_diffuse_before = src,
                            .rtgi_diffuse2_before = src2,
                            .view_cam_half_res_depth = info.view_cam_half_res_depth,
                            .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
            
                            .rtgi_diffuse_blurred = dst,
                            .rtgi_diffuse2_blurred = dst2,
                            .statistics_image = statistics_image,
                            .geo_mean_perceptual_image = geo_mean_perceptual_image,
                            .filter_guide_image = half_res_filter_guide_history.current(),
                            .temporal_geometric_mean = temporal_geometric_mean_history.current(),
                        })
                        .executes(rtgi_atrous_post_blur_callback, &info.render_context, i));
                src = dst;
                src2 = dst2;
            }
            rtgi_post_blur_diffuse_image = src;
            rtgi_post_blur_diffuse2_image = src2;
        }
    }

    auto full_diffuse_accumulated_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "full_diffuse_accumulated_image", 1);
    auto full_diffuse2_accumulated_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "full_diffuse2_accumulated_image", 1);
    auto full_samplecount_image = rtgi_create_sample_count_image(info.tg, &info.render_context, "full_samplecount_image", 1);
    auto resolved_per_pixel_diffuse = rtgi_create_upscaled_diffuse_image(info.tg, &info.render_context, "resolved_per_pixel_diffuse");

    auto accumualted_color_image = info.tg.create_task_image({
        .format = daxa::Format::R32G32_UINT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x,
            info.render_context.render_data.settings.render_target_size.y,
            1,
        },
        .name = "accumualted_color_image",
    });
    auto accumulated_full_statistics_image = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x,
            info.render_context.render_data.settings.render_target_size.y,
            1,
        },
        .name = "accumulated_full_statistics_image",
    });

    info.tg.add_task(daxa::HeadTask<RtgiUpscaleDiffuseH::Info>()
            .head_views(RtgiUpscaleDiffuseH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,

                .diffuse_half_res = rtgi_post_blur_diffuse_image,
                .diffuse2_half_res = rtgi_post_blur_diffuse2_image,

                .samplecount_half_res = half_res_sample_count_history,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                .view_cam_depth = info.view_cam_depth,
                .view_cam_face_normals = info.view_cam_face_normals,
                .view_camera_detail_normal_image = info.view_camera_detail_normal_image,
                .diffuse_resolved = resolved_per_pixel_diffuse,
            })
            .executes(rtgi_upscale_diffuse_callback, &info.render_context));

    info.tg.copy_image_to_image({.src_image = info.view_cam_half_res_depth, .dst_image = half_res_depth_history, .name = "save rtgi depth history"});
    info.tg.copy_image_to_image({.src_image = info.view_cam_half_res_face_normals, .dst_image = half_res_face_normal_history, .name = "save rtgi face normal history"});

    return TasksRtgiMainResult{
        .opaque_diffuse = resolved_per_pixel_diffuse,
    };
}