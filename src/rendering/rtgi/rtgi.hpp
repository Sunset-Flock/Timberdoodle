#pragma once

#include "rtgi_trace_diffuse.inl"
#include "rtgi_reproject_diffuse.inl"
#include "rtgi_reconstruct_history.inl"
#include "rtgi_adaptive_blur.inl"
#include "rtgi_upscale.inl"

inline auto rtgi_trace_diffuse_compile_info() -> daxa::RayTracingPipelineCompileInfo2
{
    auto file = daxa::ShaderFile{"./src/rendering/rtgi/rtgi_trace_diffuse.hlsl"};
    return daxa::RayTracingPipelineCompileInfo2{
        .ray_gen_infos = {{.source = file, .entry_point = "ray_gen", .language = daxa::ShaderLanguage::SLANG}},
        .any_hit_infos = {{.source = file, .entry_point = "any_hit", .language = daxa::ShaderLanguage::SLANG}},
        .closest_hit_infos = {{.source = file, .entry_point = "closest_hit", .language = daxa::ShaderLanguage::SLANG}},
        .miss_hit_infos = {{.source = file, .entry_point = "miss", .language = daxa::ShaderLanguage::SLANG}},
        .shader_groups_infos = {
            // Gen Group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::GENERAL, .general_shader_index = 0},
            // Miss group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::GENERAL, .general_shader_index = 3},
            // Hit group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP, .closest_hit_shader_index = 2},
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP, .closest_hit_shader_index = 2, .any_hit_shader_index = 1},
        },
        .max_ray_recursion_depth = 2,
        .name = std::string{RtgiTraceDiffuseH::Info::NAME},
    };
}

MAKE_COMPUTE_COMPILE_INFO(rtgi_reproject_diffuse_compile_info, "./src/rendering/rtgi/rtgi_reproject_diffuse.hlsl", "entry_reproject")
MAKE_COMPUTE_COMPILE_INFO(rtgi_reconstruct_history_gen_mips_diffuse_compile_info, "./src/rendering/rtgi/rtgi_reconstruct_history.hlsl", "entry_gen_mips_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_reconstruct_history_apply_diffuse_compile_info, "./src/rendering/rtgi/rtgi_reconstruct_history.hlsl", "entry_apply_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_adaptive_blur_diffuse_compile_info, "./src/rendering/rtgi/rtgi_adaptive_blur.hlsl", "entry_blur_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_upscale_diffuse_compile_info, "./src/rendering/rtgi/rtgi_upscale.hlsl", "entry_upscale_diffuse")


void rtgi_trace_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiTraceDiffuseH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RTGI", "TRACE_DIFFUSE">());
    RtgiTraceDiffusePush push = {};
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

void rtgi_denoise_diffuse_reproject_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiReprojectDiffuseH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RTGI", "REPROJECT_DIFFUSE">());
    auto const & diffuse_raw = ti.info(AT.rtgi_diffuse_raw).value();
    ti.recorder.set_pipeline(*(render_context->gpu_context->compute_pipelines.at(rtgi_reproject_diffuse_compile_info().name)));
    ti.recorder.push_constant(RtgiReprojectDiffusePush{
        .attach = ti.attachment_shader_blob,
        .size = {diffuse_raw.size.x, diffuse_raw.size.y},
    });
    ti.recorder.dispatch({
        round_up_div(diffuse_raw.size.x, RTGI_DENOISE_DIFFUSE_X),
        round_up_div(diffuse_raw.size.y, RTGI_DENOISE_DIFFUSE_Y),
        1,
    });
}

void rtgi_reconstruct_history_gen_mips_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiReconstructHistoryGenMipsDiffuseH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RTGI", "RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE">());
    auto const & diffuse_raw = ti.info(AT.rtgi_diffuse_accumulated).value();
    ti.recorder.set_pipeline(*(render_context->gpu_context->compute_pipelines.at(rtgi_reconstruct_history_gen_mips_diffuse_compile_info().name)));
    ti.recorder.push_constant(RtgiReconstructHistoryGenMipsDiffusePush{
        .attach = ti.attachment_shader_blob, 
        .size = {diffuse_raw.size.x, diffuse_raw.size.y},
    });
    ti.recorder.dispatch({
        round_up_div(diffuse_raw.size.x, RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X),
        round_up_div(diffuse_raw.size.y, RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_Y),
        1,
    });
}

void rtgi_reconstruct_history_apply_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiReconstructHistoryApplyDiffuseH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RTGI", "RECONSTRUCT_HISTORY_APPLY_DIFFUSE">());
    auto const & diffuse_raw = ti.info(AT.rtgi_diffuse_accumulated).value();
    ti.recorder.set_pipeline(*(render_context->gpu_context->compute_pipelines.at(rtgi_reconstruct_history_apply_diffuse_compile_info().name)));
    ti.recorder.push_constant(RtgiReconstructHistoryApplyDiffusePush{
        .attach = ti.attachment_shader_blob, 
        .size = {diffuse_raw.size.x, diffuse_raw.size.y},
    });
    ti.recorder.dispatch({
        round_up_div(diffuse_raw.size.x, RTGI_RECONSTRUCT_HISTORY_APPLY_DIFFUSE_X),
        round_up_div(diffuse_raw.size.y, RTGI_RECONSTRUCT_HISTORY_APPLY_DIFFUSE_Y),
        1,
    });
}

void rtgi_adaptive_blur_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass)
{
    auto const & AT = RtgiAdaptiveBlurH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, pass == 0 ? RenderTimes::index<"RTGI", "BLUR_DIFFUSE_0">() : RenderTimes::index<"RTGI", "BLUR_DIFFUSE_1">());
    auto const size = ti.info(AT.view_cam_half_res_depth).value().size;
    ti.recorder.set_pipeline(*(render_context->gpu_context->compute_pipelines.at(rtgi_adaptive_blur_diffuse_compile_info().name)));
    ti.recorder.push_constant(RtgiAdaptiveBlurPush{
        .attach = ti.attachment_shader_blob, 
        .size = {size.x, size.y},
    });
    ti.recorder.dispatch({
        round_up_div(size.x, RTGI_ADAPTIVE_BLUR_DIFFUSE_X),
        round_up_div(size.y, RTGI_ADAPTIVE_BLUR_DIFFUSE_Y),
        1,
    });
}

void rtgi_upscale_diffuse_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiUpscaleDiffuseH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RTGI", "UPSCALE_DIFFUSE">());
    auto const size = ti.info(AT.view_cam_depth).value().size;
    ti.recorder.set_pipeline(*(render_context->gpu_context->compute_pipelines.at(rtgi_upscale_diffuse_compile_info().name)));
    ti.recorder.push_constant(RtgiUpscaleDiffusePush{
        .attach = ti.attachment_shader_blob, 
        .size = {size.x, size.y},
    });
    ti.recorder.dispatch({
        round_up_div(size.x, RTGI_UPSCALE_DIFFUSE_X),
        round_up_div(size.y, RTGI_UPSCALE_DIFFUSE_Y),
        1,
    });
}



// rgb = diffuse radiance, a = ray travel distance
auto rtgi_create_trace_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .name = name,
    });
}

auto rtgi_create_upscaled_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x,
            render_context->render_data.settings.render_target_size.y,
            1,
        },
        .name = name,
    });
}

auto rtgi_create_samplecnt_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .name = name,
    });
}

auto rtgi_create_reconstructed_history_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name)
{
    return tg.create_transient_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {
            (render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV) / 2,
            (render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV) / 2,
            1,
        },
        .mip_level_count = 4,
        .name = name,
    });
}

// rgb = diffuse irradiance, a = accumulated samples
auto rtgi_create_diffuse_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
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

auto rtgi_create_depth_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
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

auto rtgi_create_samplecnt_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
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

auto rtgi_create_face_normal_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
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