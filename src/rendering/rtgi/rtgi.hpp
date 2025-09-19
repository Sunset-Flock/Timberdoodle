#pragma once

#include "rtgi_trace_diffuse.inl"
#include "rtgi_reproject_diffuse.inl"
#include "rtgi_reconstruct_history.inl"
#include "rtgi_adaptive_blur.inl"
#include "rtgi_upscale.inl"

///
/// === Pipeline compile infos ===
///

inline auto rtgi_trace_diffuse_compile_info() -> daxa::RayTracingPipelineCompileInfo2
{
    auto file = daxa::ShaderFile{"./src/rendering/rtgi/rtgi_trace_diffuse.hlsl"};
    return daxa::RayTracingPipelineCompileInfo2{
        .ray_gen_infos = {{.source = file, .entry_point = "ray_gen", .language = daxa::ShaderLanguage::SLANG}},
        .any_hit_infos = {{.source = file, .entry_point = "any_hit", .language = daxa::ShaderLanguage::SLANG}},
        .closest_hit_infos = {{.source = daxa::ShaderFile{"./src/rendering/rtgi/rtgi_trace_diffuse_shading.hlsl"}, .entry_point = "closest_hit", .language = daxa::ShaderLanguage::SLANG}},
        .miss_hit_infos = {{.source = daxa::ShaderFile{"./src/rendering/rtgi/rtgi_trace_diffuse_shading.hlsl"}, .entry_point = "miss", .language = daxa::ShaderLanguage::SLANG}},
        .shader_groups_infos = {
            // Gen Group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::GENERAL, .general_shader_index = 0},
            // Miss group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::GENERAL, .general_shader_index = 3},
            // Hit group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP, .closest_hit_shader_index = 2},
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP, .closest_hit_shader_index = 2, .any_hit_shader_index = 1},
        },
        .max_ray_recursion_depth = 3,
        .name = std::string{RtgiTraceDiffuseH::Info::NAME},
    };
}

MAKE_COMPUTE_COMPILE_INFO(rtgi_reproject_diffuse_compile_info, "./src/rendering/rtgi/rtgi_reproject_diffuse.hlsl", "entry_reproject")
MAKE_COMPUTE_COMPILE_INFO(rtgi_reconstruct_history_gen_mips_diffuse_compile_info, "./src/rendering/rtgi/rtgi_reconstruct_history.hlsl", "entry_gen_mips_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_reconstruct_history_apply_diffuse_compile_info, "./src/rendering/rtgi/rtgi_reconstruct_history.hlsl", "entry_apply_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_adaptive_blur_diffuse_compile_info, "./src/rendering/rtgi/rtgi_adaptive_blur.hlsl", "entry_blur_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_pre_blur_diffuse_compile_info, "./src/rendering/rtgi/rtgi_adaptive_blur.hlsl", "entry_pre_blur_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_upscale_diffuse_compile_info, "./src/rendering/rtgi/rtgi_upscale.hlsl", "entry_upscale_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_diffuse_temporal_stabilization_compile_info, "./src/rendering/rtgi/rtgi_reproject_diffuse.hlsl", "entry_temporal_stabilization")

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
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RTGI", "TRACE_DIFFUSE">());
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
inline void rtgi_common_task_callback(TaskPush push, daxa::TaskInterface ti, RenderContext * render_context, daxa::TaskImageAttachmentIndex dst_image, u32 block_size, u32 render_time, std::string const & compute_pipeline_name)
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
    rtgi_common_task_callback(RtgiReprojectDiffusePush(), ti, render_context, AT.rtgi_diffuse_raw, RTGI_DENOISE_DIFFUSE_X, RenderTimes::index<"RTGI", "REPROJECT_DIFFUSE">(), rtgi_reproject_diffuse_compile_info().name);
}

inline void rtgi_reconstruct_history_gen_mips_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiReconstructHistoryGenMipsDiffuseH::Info::AT;
    rtgi_common_task_callback(RtgiReconstructHistoryGenMipsDiffusePush(), ti, render_context, AT.rtgi_diffuse_accumulated, RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X, RenderTimes::index<"RTGI", "RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE">(), rtgi_reconstruct_history_gen_mips_diffuse_compile_info().name);
}

inline void rtgi_reconstruct_history_apply_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiReconstructHistoryApplyDiffuseH::Info::AT;
    rtgi_common_task_callback(RtgiReconstructHistoryApplyDiffusePush(), ti, render_context, AT.rtgi_diffuse_accumulated, RTGI_RECONSTRUCT_HISTORY_APPLY_DIFFUSE_X, RenderTimes::index<"RTGI", "RECONSTRUCT_HISTORY_APPLY_DIFFUSE">(), rtgi_reconstruct_history_apply_diffuse_compile_info().name);
}

inline void rtgi_adaptive_blur_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass)
{
    auto const & AT = RtgiAdaptiveBlurH::Info::AT;
    u32 const render_time = pass == 0 ? RenderTimes::index<"RTGI", "BLUR_DIFFUSE_0">() : RenderTimes::index<"RTGI", "BLUR_DIFFUSE_1">();
    rtgi_common_task_callback(RtgiAdaptiveBlurPush(), ti, render_context, AT.view_cam_half_res_depth, RTGI_ADAPTIVE_BLUR_DIFFUSE_X, render_time, rtgi_adaptive_blur_diffuse_compile_info().name);
}

inline void rtgi_pre_blur_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiPreBlurH::Info::AT;
    rtgi_common_task_callback(RtgiPreBlurPush(), ti, render_context, AT.view_cam_half_res_depth, RTGI_PRE_BLUR_DIFFUSE_X, RenderTimes::index<"RTGI", "PRE_BLUR_DIFFUSE">(), rtgi_pre_blur_diffuse_compile_info().name);
}

inline void rtgi_upscale_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiUpscaleDiffuseH::Info::AT;
    rtgi_common_task_callback(RtgiUpscaleDiffusePush(), ti, render_context, AT.view_cam_depth, RTGI_UPSCALE_DIFFUSE_X, RenderTimes::index<"RTGI", "UPSCALE_DIFFUSE">(), rtgi_upscale_diffuse_compile_info().name);
}

inline void rtgi_diffuse_temporal_stabilization_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiDiffuseTemporalStabilizationH::Info::AT;
    rtgi_common_task_callback(RtgiDiffuseTemporalStabilizationPush(), ti, render_context, AT.view_cam_half_res_depth, RTGI_DIFFUSE_TEMPORAL_STABILIZATION_X, RenderTimes::index<"RTGI", "TEMPORAL_STABILIZATION">(), rtgi_diffuse_temporal_stabilization_compile_info().name);
}

///
/// === Transient Images ===
///

inline auto rtgi_create_common_transient_image_info(RenderContext * render_context, daxa::Format format, daxa::u32 size_div, std::string_view name) -> daxa::TaskTransientImageInfo
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
inline auto rtgi_create_trace_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, RTGI_DIFFUSE_PIXEL_SCALE_DIV, name));
}

inline auto rtgi_create_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, RTGI_DIFFUSE_PIXEL_SCALE_DIV, name));
}

inline auto rtgi_create_diffuse2_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16_SFLOAT, RTGI_DIFFUSE_PIXEL_SCALE_DIV, name));
}

inline auto rtgi_create_upscaled_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, 1, name));
}

inline auto rtgi_create_samplecnt_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16_SFLOAT, RTGI_DIFFUSE_PIXEL_SCALE_DIV, name));
}

// 16 -> 8 -> 4 -> 2 -> 1
inline auto rtgi_create_reconstructed_history_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    auto info = rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, 1, name);
    // round up to multiple of 8 to make sure that all mip texels align exactly 2x2 -> 1
    info.size.x = round_up_to_multiple(info.size.x / (RTGI_DIFFUSE_PIXEL_SCALE_DIV*2), 8),
    info.size.y = round_up_to_multiple(info.size.y / (RTGI_DIFFUSE_PIXEL_SCALE_DIV*2), 8),
    info.mip_level_count = 4;
    return tg.create_transient_image(info);
}

///
/// === Persistent Images ===
///

inline auto rtgi_create_diffuse_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi diffuse history image",
    };
}

inline auto rtgi_create_diffuse2_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R16G16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi diffuse2 history image",
    };
}

inline auto rtgi_create_depth_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R32_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi depth history image",
    };
}

inline auto rtgi_create_samplecnt_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi samplecnt history image",
    };
}

inline auto rtgi_create_face_normal_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R32_UINT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi face normal history image",
    };
}