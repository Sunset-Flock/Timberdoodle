#include "rtgi.hpp"

#include "rtgi_trace_diffuse.inl"
#include "rtgi_temporal.inl"
#include "rtgi_pre_filter.inl"
#include "rtgi_adaptive_blur.inl"
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

template <typename TaskPush>
inline void dispatch_image_relative(TaskPush push, daxa::TaskInterface ti, RenderContext * render_context, daxa::TaskImageAttachmentIndex dst_image, u32 block_size, u32 render_time, std::string const & compute_pipeline_name)
{
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, render_time);
    auto const dst_image_size = ti.info(dst_image).value().size;
    ti.recorder.set_pipeline(*(render_context->gpu_context->compute_pipelines.at(compute_pipeline_name)));
    push.attach = ti.attachment_shader_blob;
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
    dispatch_image_relative(RtgiTemporalPush(), ti, render_context, AT.half_res_sample_count, RTGI_TEMPORAL_X, RenderTimes::index<"RTGI", "TEMPORAL">(), rtgi_temporal_compile_info().name);
}

inline void rtgi_pre_filter_prepare_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiPreFilterPrepareH::Info::AT;
    dispatch_image_relative(RtgiPreFilterPreparePush(), ti, render_context, AT.diffuse_raw, RTGI_PRE_BLUR_PREPARE_X, RenderTimes::index<"RTGI", "PRE_BLUR_PREPARE">(), rtgi_pre_filter_prepare_compile_info().name);
}

inline void rtgi_pre_filter_apply_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiPreFilterApply::Info::AT;
    dispatch_image_relative(RtgiPreFilterApplyPush(), ti, render_context, AT.diffuse_filtered, RTGI_PRE_BLUR_APPLY_X, RenderTimes::index<"RTGI", "PRE_BLUR_APPLY">(), rtgi_pre_filter_apply_compile_info().name);
}

inline void rtgi_adaptive_blur_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass)
{
    auto const & AT = RtgiAdaptiveBlurH::Info::AT;
    u32 const render_time = pass == 0 ? RenderTimes::index<"RTGI", "BLUR_DIFFUSE_0">() : RenderTimes::index<"RTGI", "BLUR_DIFFUSE_1">();
    dispatch_image_relative(RtgiAdaptiveBlurPush{.pass = pass}, ti, render_context, AT.view_cam_half_res_depth, RTGI_ADAPTIVE_BLUR_DIFFUSE_X, render_time, rtgi_adaptive_blur_compile_info().name);
}

inline void rtgi_post_blur_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass)
{
    auto const & AT = RtgiPostBlurH::Info::AT;
    u32 const render_time = pass == 0 ? RenderTimes::index<"RTGI", "BLUR_POST_0">() : RenderTimes::index<"RTGI", "BLUR_POST_1">();
    dispatch_image_relative(RtgiPostBlurPush{.pass = pass}, ti, render_context, AT.view_cam_half_res_depth, RTGI_POST_BLUR_DIFFUSE_X, render_time, rtgi_post_blur_compile_info().name);
}

inline void rtgi_upscale_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiUpscaleDiffuseH::Info::AT;
    dispatch_image_relative(RtgiUpscaleDiffusePush(), ti, render_context, AT.view_cam_depth, RTGI_UPSCALE_DIFFUSE_X, RenderTimes::index<"RTGI", "UPSCALE_DIFFUSE">(), rtgi_upscale_diffuse_compile_info().name);
}

///
/// === Transient Images ===
///

auto rtgi_create_common_transient_image_info(RenderContext * render_context, daxa::Format format, daxa::u32 size_div, std::string_view name) -> daxa::TaskTransientImageInfo
{
    return daxa::TaskTransientImageInfo{
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
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, scale_div, name));
}

auto rtgi_create_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, scale_div, name));
}

auto rtgi_create_diffuse2_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16_SFLOAT, scale_div, name));
}

auto rtgi_create_upscaled_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, 1, name));
}

auto rtgi_create_sample_count_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16_SFLOAT, scale_div, name));
}

// 16 -> 8 -> 4 -> 2 -> 1
auto create_mip_blur_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    auto info = rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, RTGI_PIXEL_SCALE_DIV, name);
    // round up to multiple of 16 to make sure that all mip texels align exactly 2x2 -> 1
    info.size.x = round_up_to_multiple(info.size.x, 16),
    info.size.y = round_up_to_multiple(info.size.y, 16),
    info.mip_level_count = 5;
    return tg.create_transient_image(info);
}

auto tasks_rtgi_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult
{
    auto trace_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_diffuse_raw_image");
    auto trace_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_diffuse2_raw_image");
    info.tg.add_task(daxa::HeadTask<RtgiTraceDiffuseH::Info>()
            .head_views(RtgiTraceDiffuseH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .diffuse_raw = trace_diffuse_image,
                .diffuse2_raw = trace_diffuse2_image,
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

    auto half_res_sample_count_image = rtgi_create_sample_count_image(info.tg, &info.render_context, "half_res_sample_count_image");

    auto pre_blur_mips_image = create_mip_blur_image(info.tg, &info.render_context, "pre_blur_mips_image").mips(0, 5);
    auto pre_blur_mips2_image = create_mip_blur_image(info.tg, &info.render_context, "pre_blur_mips2_image").mips(0, 5);
    auto firefly_factor_image = info.tg.create_transient_image({
        .format = daxa::Format::R8_UNORM,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "firefly_factor_image",
    });
    info.tg.add_task(daxa::HeadTask<RtgiPreFilterPrepareH::Info>()
            .head_views(RtgiPreFilterPrepareH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .diffuse_raw = trace_diffuse_image,
                .diffuse2_raw = trace_diffuse2_image,
                .view_cam_half_res_normals = info.view_cam_half_res_face_normals,
                .reconstructed_diffuse_history = pre_blur_mips_image,
                .reconstructed_diffuse2_history = pre_blur_mips2_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
                .firefly_factor_image = firefly_factor_image,
            })
            .executes(rtgi_pre_filter_prepare_callback, &info.render_context));

    auto pre_blurred_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "pre_blurred_diffuse_image");
    auto pre_blurred_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "pre_blurred_diffuse2_image");
    info.tg.add_task(daxa::HeadTask<RtgiPreFilterApply::Info>()
            .head_views(RtgiPreFilterApply::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .reconstructed_diffuse_history = pre_blur_mips_image,
                .reconstructed_diffuse2_history = pre_blur_mips2_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
                .view_cam_half_res_normals = info.view_cam_half_res_face_normals,
                .diffuse_filtered = pre_blurred_diffuse_image,
                .diffuse2_filtered = pre_blurred_diffuse2_image,
            })
            .executes(rtgi_pre_filter_apply_callback, &info.render_context));

    daxa::TaskImageView diffuse_filtered = pre_blurred_diffuse_image.mips(0);
    daxa::TaskImageView diffuse2_filtered = pre_blurred_diffuse2_image;
    if (info.render_context.render_data.rtgi_settings.spatial_filter_enabled)
    {
        auto rtgi_blurred0_diffuse_image = pre_blurred_diffuse_image;
        auto rtgi_blurred0_diffuse2_image = pre_blurred_diffuse2_image;
        if (info.render_context.render_data.rtgi_settings.spatial_filter_two_pass_enabled)
        {
            rtgi_blurred0_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_blurred0_diffuse_image");
            rtgi_blurred0_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_blurred0_diffuse2_image");
            info.tg.add_task(daxa::HeadTask<RtgiAdaptiveBlurH::Info>("RtgiAdaptiveBlur1")
                    .head_views(RtgiAdaptiveBlurH::Info::Views{
                        .globals = info.render_context.tgpu_render_data.view(),
                        .debug_image = info.debug_image,
                        .clocks_image = info.clocks_image,
                        .rtgi_diffuse_before = pre_blurred_diffuse_image,
                        .rtgi_diffuse2_before = pre_blurred_diffuse2_image,
                        .rtgi_samplecnt = half_res_sample_count_image,
                        .view_cam_half_res_depth = info.view_cam_half_res_depth,
                        .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                        .rtgi_diffuse_blurred = rtgi_blurred0_diffuse_image,
                        .rtgi_diffuse2_blurred = rtgi_blurred0_diffuse2_image,
                        .firefly_factor_image = firefly_factor_image,
                    })
                    .executes(rtgi_adaptive_blur_diffuse_callback, &info.render_context, 0u));
        }

        auto blurred1_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "blurred1_diffuse_image");
        auto blurred1_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "blurred1_diffuse2_image");
        info.tg.add_task(daxa::HeadTask<RtgiAdaptiveBlurH::Info>("RtgiAdaptiveBlur2")
                .head_views(RtgiAdaptiveBlurH::Info::Views{
                    .globals = info.render_context.tgpu_render_data.view(),
                    .debug_image = info.debug_image,
                    .clocks_image = info.clocks_image,
                    .rtgi_diffuse_before = rtgi_blurred0_diffuse_image,
                    .rtgi_diffuse2_before = rtgi_blurred0_diffuse2_image,
                    .rtgi_samplecnt = half_res_sample_count_image,
                    .view_cam_half_res_depth = info.view_cam_half_res_depth,
                    .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                    .rtgi_diffuse_blurred = blurred1_diffuse_image,
                    .rtgi_diffuse2_blurred = blurred1_diffuse2_image,
                    .firefly_factor_image = firefly_factor_image,
                })
                .executes(rtgi_adaptive_blur_diffuse_callback, &info.render_context, 1u));
        diffuse_filtered = blurred1_diffuse_image;
        diffuse2_filtered = blurred1_diffuse2_image;
    }

    auto accumulated_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "accumulated_diffuse_image");
    auto accumulated_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "accumulated_diffuse2_image");
    auto accumulated_statistics_image = info.tg.create_transient_image({
        .format = daxa::Format::R32_UINT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "accumulated_statistics_image",
    });
    info.tg.add_task(daxa::HeadTask<RtgiTemporalH::Info>()
            .head_views(RtgiTemporalH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .half_res_sample_count = half_res_sample_count_image,
                .half_res_sample_count_history = info.half_res_samplecnt_history,
                .half_res_diffuse_new = diffuse_filtered,
                .half_res_diffuse_accumulated = accumulated_diffuse_image,
                .half_res_diffuse_history = info.half_res_diffuse_history,
                .half_res_diffuse2_new = diffuse2_filtered,
                .half_res_diffuse2_accumulated = accumulated_diffuse2_image,
                .half_res_diffuse2_history = info.half_res_diffuse2_history,
                .half_res_statistics_accumulated = accumulated_statistics_image,
                .half_res_statistics_history = info.half_res_statistics_history,
                .half_res_depth = info.view_cam_half_res_depth,
                .half_res_depth_history = info.half_res_depth_history,
                .half_res_normal = info.view_cam_half_res_face_normals,
                .half_res_normal_history = info.half_res_face_normal_history,
            })
            .executes(rtgi_temporal_callback, &info.render_context));

    auto rtgi_post_blur_diffuse_image = accumulated_diffuse_image;
    auto rtgi_post_blur_diffuse2_image = accumulated_diffuse2_image;
    if (info.render_context.render_data.rtgi_settings.post_blur_enabled)
    {
        auto rtgi_post_blur_pass0_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_post_blur_pass0_diffuse_image");
        auto rtgi_post_blur_pass0_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_post_blur_pass0_diffuse2_image");
        info.tg.add_task(daxa::HeadTask<RtgiPostBlurH::Info>("RtgiPostBlur0")
                .head_views(RtgiPostBlurH::Info::Views{
                    .globals = info.render_context.tgpu_render_data.view(),
                    .debug_image = info.debug_image,
                    .clocks_image = info.clocks_image,
                    .rtgi_sample_count = half_res_sample_count_image,
                    .rtgi_diffuse_before = accumulated_diffuse_image,
                    .rtgi_diffuse2_before = accumulated_diffuse2_image,
                    .view_cam_half_res_depth = info.view_cam_half_res_depth,
                    .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                    .rtgi_diffuse_blurred = rtgi_post_blur_pass0_diffuse_image,
                    .rtgi_diffuse2_blurred = rtgi_post_blur_pass0_diffuse2_image,
                })
                .executes(rtgi_post_blur_diffuse_callback, &info.render_context, 0u));

        rtgi_post_blur_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_post_blur_diffuse_image");
        rtgi_post_blur_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_post_blur_diffuse2_image");
        info.tg.add_task(daxa::HeadTask<RtgiPostBlurH::Info>("RtgiPostBlur1")
                .head_views(RtgiPostBlurH::Info::Views{
                    .globals = info.render_context.tgpu_render_data.view(),
                    .debug_image = info.debug_image,
                    .clocks_image = info.clocks_image,
                    .rtgi_sample_count = half_res_sample_count_image,
                    .rtgi_diffuse_before = rtgi_post_blur_pass0_diffuse_image,
                    .rtgi_diffuse2_before = rtgi_post_blur_pass0_diffuse2_image,
                    .view_cam_half_res_depth = info.view_cam_half_res_depth,
                    .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                    .rtgi_diffuse_blurred = rtgi_post_blur_diffuse_image,
                    .rtgi_diffuse2_blurred = rtgi_post_blur_diffuse2_image,
                })
                .executes(rtgi_post_blur_diffuse_callback, &info.render_context, 1u));
    }

    auto full_diffuse_accumulated_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "full_diffuse_accumulated_image", 1);
    auto full_diffuse2_accumulated_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "full_diffuse2_accumulated_image", 1);
    auto full_samplecount_image = rtgi_create_sample_count_image(info.tg, &info.render_context, "full_samplecount_image", 1);
    auto resolved_per_pixel_diffuse = rtgi_create_upscaled_diffuse_image(info.tg, &info.render_context, "resolved_per_pixel_diffuse");

    auto accumualted_color_image = info.tg.create_transient_image({
        .format = daxa::Format::R32G32_UINT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x,
            info.render_context.render_data.settings.render_target_size.y,
            1,
        },
        .name = "accumualted_color_image",
    });
    auto accumulated_full_statistics_image = info.tg.create_transient_image({
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

                .color_history_full_res = info.color_history,
                .statistics_history_full_res = info.statistics_history,
                .accumulated_color_full_res = accumualted_color_image,
                .accumulated_statistics_full_res = accumulated_full_statistics_image,

                .depth_history_full_res = info.depth_history,
                .face_normal_history_full_res = info.face_normal_history,
                .samplecount_history_full_res = info.samplecount_history,

                .diffuse_full_res = full_diffuse_accumulated_image,
                .diffuse2_full_res = full_diffuse2_accumulated_image,
                .samplecount_full_res = full_samplecount_image,

                .diffuse_half_res = rtgi_post_blur_diffuse_image,
                .diffuse2_half_res = rtgi_post_blur_diffuse2_image,

                .samplecount_half_res = half_res_sample_count_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                .view_cam_depth = info.view_cam_depth,
                .view_cam_face_normals = info.view_cam_face_normals,
                .view_camera_detail_normal_image = info.view_camera_detail_normal_image,
                .diffuse_resolved = resolved_per_pixel_diffuse,
            })
            .executes(rtgi_upscale_diffuse_callback, &info.render_context));

    info.tg.copy_image_to_image({.src = info.view_cam_half_res_depth, .dst = info.half_res_depth_history, .name = "save rtgi depth history"});
    info.tg.copy_image_to_image({.src = half_res_sample_count_image, .dst = info.half_res_samplecnt_history, .name = "save rtgi samplecnt history"});
    info.tg.copy_image_to_image({.src = info.view_cam_half_res_face_normals, .dst = info.half_res_face_normal_history, .name = "save rtgi face normal history"});
    info.tg.copy_image_to_image({.src = full_samplecount_image, .dst = info.samplecount_history, .name = "save full_samplecount_history"});
    info.tg.copy_image_to_image({.src = info.view_cam_face_normals, .dst = info.face_normal_history, .name = "save full_face_normal_history"});
    info.tg.copy_image_to_image({.src = accumualted_color_image, .dst = info.color_history, .name = "save full_color_history"});
    info.tg.copy_image_to_image({.src = accumulated_full_statistics_image, .dst = info.statistics_history, .name = "save full_statistics_history"});
    info.tg.copy_image_to_image({.src = accumulated_diffuse_image, .dst = info.half_res_diffuse_history, .name = "save half_res_diffuse_history"});
    info.tg.copy_image_to_image({.src = accumulated_diffuse2_image, .dst = info.half_res_diffuse2_history, .name = "save half_res_diffuse2_history"});
    info.tg.copy_image_to_image({.src = accumulated_statistics_image, .dst = info.half_res_statistics_history, .name = "save half_res_statistics_history"});

    return TasksRtgiMainResult{
        .opaque_diffuse = resolved_per_pixel_diffuse,
    };
}