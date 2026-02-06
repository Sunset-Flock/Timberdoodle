#include "rtgi.hpp"

#include "rtgi_trace_diffuse.inl"
#include "rtgi_reproject.inl"
#include "rtgi_pre_blur.inl"
#include "rtgi_adaptive_blur.inl"
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
    auto const & diffuse_raw = ti.info(AT.rtgi_diffuse_raw).value();
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

inline void rtgi_denoise_diffuse_reproject_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiReprojectDiffuseH::Info::AT;
    dispatch_image_relative(RtgiReprojectDiffusePush(), ti, render_context, AT.rtgi_samplecnt, RTGI_DENOISE_DIFFUSE_X, RenderTimes::index<"RTGI", "HALFRES_REPROJECT">(), rtgi_reproject_diffuse_compile_info().name);
}

inline void rtgi_pre_blur_prepare_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiPreblurPrepareH::Info::AT;
    dispatch_image_relative(RtgiPreblurPreparePush(), ti, render_context, AT.rtgi_diffuse_raw, RTGI_PRE_BLUR_PREPARE_X, RenderTimes::index<"RTGI", "PRE_BLUR_PREPARE">(), rtgi_pre_blur_prepare_compile_info().name);
}

inline void rtgi_pre_blur_flatten_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiPreBlurFlattenH::Info::AT;
    dispatch_image_relative(RtgiPreBlurFlattenPush(), ti, render_context, AT.rtgi_diffuse_raw, RTGI_PRE_BLUR_FLATTEN_X, RenderTimes::index<"RTGI", "PRE_BLUR_FLATTEN">(), rtgi_pre_blur_flatten_compile_info().name);
}

inline void rtgi_pre_blur_apply_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiPreBlurApply::Info::AT;
    dispatch_image_relative(RtgiPreBlurApplyPush(), ti, render_context, AT.rtgi_diffuse_filtered, RTGI_PRE_BLUR_APPLY_X, RenderTimes::index<"RTGI", "PRE_BLUR_APPLY">(), rtgi_pre_blur_apply_compile_info().name);
}

inline void rtgi_adaptive_blur_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass)
{
    auto const & AT = RtgiAdaptiveBlurH::Info::AT;
    u32 const render_time = pass == 0 ? RenderTimes::index<"RTGI", "BLUR_DIFFUSE_0">() : RenderTimes::index<"RTGI", "BLUR_DIFFUSE_1">();
    dispatch_image_relative(RtgiAdaptiveBlurPush{.pass = pass}, ti, render_context, AT.view_cam_half_res_depth, RTGI_ADAPTIVE_BLUR_DIFFUSE_X, render_time, rtgi_adaptive_blur_diffuse_compile_info().name);
}

inline void rtgi_upscale_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiUpscaleDiffuseH::Info::AT;
    dispatch_image_relative(RtgiUpscaleDiffusePush(), ti, render_context, AT.view_cam_depth, RTGI_UPSCALE_DIFFUSE_X, RenderTimes::index<"RTGI", "UPSCALE_DIFFUSE">(), rtgi_upscale_diffuse_compile_info().name);
}

inline void rtgi_diffuse_temporal_stabilization_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiDiffuseTemporalStabilizationH::Info::AT;
    dispatch_image_relative(RtgiDiffuseTemporalStabilizationPush(), ti, render_context, AT.view_cam_half_res_depth, RTGI_DIFFUSE_TEMPORAL_STABILIZATION_X, RenderTimes::index<"RTGI", "TEMPORAL_STABILIZATION">(), rtgi_diffuse_temporal_stabilization_compile_info().name);
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
auto rtgi_create_trace_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_DIFFUSE_PIXEL_SCALE_DIV)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, scale_div, name));
}

auto rtgi_create_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_DIFFUSE_PIXEL_SCALE_DIV)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, scale_div, name));
}

auto rtgi_create_diffuse2_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_DIFFUSE_PIXEL_SCALE_DIV)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16_SFLOAT, scale_div, name));
}

auto rtgi_create_upscaled_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, 1, name));
}

auto rtgi_create_samplecnt_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_DIFFUSE_PIXEL_SCALE_DIV)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16_SFLOAT, scale_div, name));
}

// 16 -> 8 -> 4 -> 2 -> 1
auto create_mip_blur_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    auto info = rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, RTGI_DIFFUSE_PIXEL_SCALE_DIV, name);
    // round up to multiple of 16 to make sure that all mip texels align exactly 2x2 -> 1
    info.size.x = round_up_to_multiple(info.size.x, 16),
    info.size.y = round_up_to_multiple(info.size.y, 16),
    info.mip_level_count = 5;
    return tg.create_transient_image(info);
}

auto tasks_rtgi_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult
{
    auto rtgi_trace_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_diffuse_raw_image");
    auto rtgi_trace_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_diffuse2_raw_image");
    info.tg.add_task(daxa::HeadTask<RtgiTraceDiffuseH::Info>()
            .head_views(RtgiTraceDiffuseH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .rtgi_diffuse_raw = rtgi_trace_diffuse_image,
                .rtgi_diffuse2_raw = rtgi_trace_diffuse2_image,
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

    auto rtgi_samplecnt_image = rtgi_create_samplecnt_image(info.tg, &info.render_context, "rtgi_samplecnt_image");
    info.tg.add_task(daxa::HeadTask<RtgiReprojectDiffuseH::Info>()
            .head_views(RtgiReprojectDiffuseH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .rtgi_depth_history = info.rtgi_depth_history,
                .rtgi_samplecnt_history = info.rtgi_samplecnt_history,
                .rtgi_face_normal_history = info.rtgi_face_normal_history,
                .rtgi_samplecnt = rtgi_samplecnt_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
            })
            .executes(rtgi_denoise_diffuse_reproject_callback, &info.render_context));

    auto rtgi_flattened_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_flattened_diffuse_image");
    auto rtgi_flattened_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_flattened_diffuse2_image");
    info.tg.add_task(daxa::HeadTask<RtgiPreBlurFlattenH::Info>()
            .head_views(RtgiPreBlurFlattenH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .rtgi_diffuse_raw = rtgi_trace_diffuse_image,
                .rtgi_diffuse2_raw = rtgi_trace_diffuse2_image,
                .view_cam_half_res_normals = info.view_cam_half_res_face_normals,
                .rtgi_flattened_diffuse = rtgi_flattened_diffuse_image,
                .rtgi_flattened_diffuse2 = rtgi_flattened_diffuse2_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
            })
            .executes(rtgi_pre_blur_flatten_callback, &info.render_context));

    auto rtgi_pre_blur_mips_image = create_mip_blur_image(info.tg, &info.render_context, "rtgi_pre_blur_mips_image").mips(0, 5);
    auto rtgi_pre_blur_mips2_image = create_mip_blur_image(info.tg, &info.render_context, "rtgi_pre_blur_mips2_image").mips(0, 5);
    info.tg.add_task(daxa::HeadTask<RtgiPreblurPrepareH::Info>()
            .head_views(RtgiPreblurPrepareH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .rtgi_diffuse_raw = rtgi_flattened_diffuse_image,
                .rtgi_diffuse2_raw = rtgi_flattened_diffuse2_image,
                .rtgi_samplecnt = rtgi_samplecnt_image,
                .view_cam_half_res_normals = info.view_cam_half_res_face_normals,
                .rtgi_reconstructed_diffuse_history = rtgi_pre_blur_mips_image,
                .rtgi_reconstructed_diffuse2_history = rtgi_pre_blur_mips2_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
            })
            .executes(rtgi_pre_blur_prepare_callback, &info.render_context));

    auto rtgi_pre_blurred_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_pre_blurred_diffuse_image");
    auto rtgi_pre_blurred_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_pre_blurred_diffuse2_image");
    info.tg.add_task(daxa::HeadTask<RtgiPreBlurApply::Info>()
            .head_views(RtgiPreBlurApply::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,
                .rtgi_reconstructed_diffuse_history = rtgi_pre_blur_mips_image,
                .rtgi_reconstructed_diffuse2_history = rtgi_pre_blur_mips2_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
                .view_cam_half_res_normals = info.view_cam_half_res_face_normals,
                .rtgi_samplecnt = rtgi_samplecnt_image,
                .rtgi_diffuse_filtered = rtgi_pre_blurred_diffuse_image,
                .rtgi_diffuse2_filtered = rtgi_pre_blurred_diffuse2_image,
            })
            .executes(rtgi_pre_blur_apply_callback, &info.render_context));

    daxa::TaskImageView rtgi_diffuse_filtered = rtgi_pre_blurred_diffuse_image.mips(0);
    daxa::TaskImageView rtgi_diffuse2_filtered = rtgi_pre_blurred_diffuse2_image;
    if (info.render_context.render_data.rtgi_settings.spatial_filter_enabled)
    {
        auto rtgi_blurred0_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_blurred0_diffuse_image");
        auto rtgi_blurred0_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_blurred0_diffuse2_image");
        info.tg.add_task(daxa::HeadTask<RtgiAdaptiveBlurH::Info>("RtgiAdaptiveBlur1")
                .head_views(RtgiAdaptiveBlurH::Info::Views{
                    .globals = info.render_context.tgpu_render_data.view(),
                    .debug_image = info.debug_image,
                    .clocks_image = info.clocks_image,
                    .rtgi_diffuse_before = rtgi_pre_blurred_diffuse_image,
                    .rtgi_diffuse2_before = rtgi_pre_blurred_diffuse2_image,
                    .rtgi_samplecnt = rtgi_samplecnt_image,
                    .view_cam_half_res_depth = info.view_cam_half_res_depth,
                    .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                    .rtgi_diffuse_blurred = rtgi_blurred0_diffuse_image,
                    .rtgi_diffuse2_blurred = rtgi_blurred0_diffuse2_image,
                })
                .executes(rtgi_adaptive_blur_diffuse_callback, &info.render_context, 0u));

        auto rtgi_blurred1_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_blurred1_diffuse_image");
        auto rtgi_blurred1_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_blurred1_diffuse2_image");
        info.tg.add_task(daxa::HeadTask<RtgiAdaptiveBlurH::Info>("RtgiAdaptiveBlur2")
                .head_views(RtgiAdaptiveBlurH::Info::Views{
                    .globals = info.render_context.tgpu_render_data.view(),
                    .debug_image = info.debug_image,
                    .clocks_image = info.clocks_image,
                    .rtgi_diffuse_before = rtgi_blurred0_diffuse_image,
                    .rtgi_diffuse2_before = rtgi_blurred0_diffuse2_image,
                    .rtgi_samplecnt = rtgi_samplecnt_image,
                    .view_cam_half_res_depth = info.view_cam_half_res_depth,
                    .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                    .rtgi_diffuse_blurred = rtgi_blurred1_diffuse_image,
                    .rtgi_diffuse2_blurred = rtgi_blurred1_diffuse2_image,
                })
                .executes(rtgi_adaptive_blur_diffuse_callback, &info.render_context, 1u));
        rtgi_diffuse_filtered = rtgi_blurred1_diffuse_image;
        rtgi_diffuse2_filtered = rtgi_blurred1_diffuse2_image;
    }

    auto rtgi_full_diffuse_accumulated_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_full_diffuse_accumulated_image", 1);
    auto rtgi_full_diffuse2_accumulated_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_full_diffuse2_accumulated_image", 1);
    auto rtgi_full_samplecount_image = rtgi_create_samplecnt_image(info.tg, &info.render_context, "rtgi_full_samplecount_image", 1);
    auto rtgi_per_pixel_diffuse = rtgi_create_upscaled_diffuse_image(info.tg, &info.render_context, "rtgi_per_pixel_diffuse");

    auto rtgi_accumualted_color_image = info.tg.create_transient_image({
        .format = daxa::Format::R32G32_UINT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x,
            info.render_context.render_data.settings.render_target_size.y,
            1,
        },
        .name = "rtgi_accumualted_color_image",
    });
    auto rtgi_accumulated_statistics_image = info.tg.create_transient_image({
        .format = daxa::Format::R32_UINT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x,
            info.render_context.render_data.settings.render_target_size.y,
            1,
        },
        .name = "rtgi_accumulated_statistics_image",
    });

    info.tg.add_task(daxa::HeadTask<RtgiUpscaleDiffuseH::Info>()
            .head_views(RtgiUpscaleDiffuseH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .clocks_image = info.clocks_image,

                .rtgi_color_history_full_res = info.rtgi_full_color_history,
                .rtgi_statistics_history_full_res = info.rtgi_full_statistics_history,
                .rtgi_accumulated_color_full_res = rtgi_accumualted_color_image,
                .rtgi_accumulated_statistics_full_res = rtgi_accumulated_statistics_image,

                .rtgi_depth_history_full_res = info.depth_history,
                .rtgi_face_normal_history_full_res = info.rtgi_full_face_normal_history,
                .rtgi_samplecount_history_full_res = info.rtgi_full_samplecount_history,

                .rtgi_diffuse_full_res = rtgi_full_diffuse_accumulated_image,
                .rtgi_diffuse2_full_res = rtgi_full_diffuse2_accumulated_image,
                .rtgi_samplecount_full_res = rtgi_full_samplecount_image,

                .rtgi_diffuse_half_res = rtgi_diffuse_filtered,
                .rtgi_diffuse2_half_res = rtgi_diffuse2_filtered,

                .rtgi_samplecount_half_res = rtgi_samplecnt_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth,
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals,
                .view_cam_depth = info.view_cam_depth,
                .view_cam_face_normals = info.view_cam_face_normals,
                .view_camera_detail_normal_image = info.view_camera_detail_normal_image,
                .rtgi_diffuse_resolved = rtgi_per_pixel_diffuse,
            })
            .executes(rtgi_upscale_diffuse_callback, &info.render_context));

    info.tg.copy_image_to_image({.src = info.view_cam_half_res_depth, .dst = info.rtgi_depth_history, .name = "save rtgi depth history"});
    info.tg.copy_image_to_image({.src = rtgi_samplecnt_image, .dst = info.rtgi_samplecnt_history, .name = "save rtgi samplecnt history"});
    info.tg.copy_image_to_image({.src = info.view_cam_half_res_face_normals, .dst = info.rtgi_face_normal_history, .name = "save rtgi face normal history"});
    info.tg.copy_image_to_image({.src = rtgi_full_samplecount_image, .dst = info.rtgi_full_samplecount_history, .name = "save rtgi_full_samplecount_history"});
    info.tg.copy_image_to_image({.src = info.view_cam_face_normals, .dst = info.rtgi_full_face_normal_history, .name = "save rtgi_full_face_normal_history"});
    info.tg.copy_image_to_image({.src = rtgi_accumualted_color_image, .dst = info.rtgi_full_color_history, .name = "save rtgi_full_color_history"});
    info.tg.copy_image_to_image({.src = rtgi_accumulated_statistics_image, .dst = info.rtgi_full_statistics_history, .name = "save rtgi_full_statistics_history"});

    return TasksRtgiMainResult{
        .opaque_diffuse = rtgi_per_pixel_diffuse,
    };
}