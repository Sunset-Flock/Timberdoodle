#include "rtgi.hpp"

#include "rtgi_trace_diffuse.inl"
#include "rtgi_distribute_rays.inl"
#include "rtgi_blend_rays.inl"
#include "rtgi_guide_resample.inl"

auto rtgi_default_settings() -> RtgiSettings
{
    return RtgiSettings{
        .enabled                              = 1,
        .shading_ao_range                             = 1.0f,
        .firefly_filter_enabled               = 1,
        .firefly_perceptual_tolerance         = 4.0f,
        .firefly_clamp_mode                   = 0,
        .pre_blur_enabled                     = 1,
        .pre_blur_ao_guiding           = 1,
        .pre_blur_perceptual_difference_guiding               = 1,
        .pre_blur_ray_count_sample_weighting                  = 1,
        .pre_blur_perceptual_radiance_guide_tolerance        = 0.8f,
        .pre_blur_base_width                  = 20.0f,
        .pre_blur_sample_count                = 8,
        .pre_blur_iterations                  = 1,
        .temporal_accumulation_enabled        = 1,
        .temporal_fast_history_enabled        = 1,
        .temporal_fast_history_frames         = 4,
        .temporal_firefly_filter_enabled      = 1,
        .temporal_firefly_std_dev_clamp       = 3.5f,
        .temporal_variance_fast_history_blend = 4.0f,
        .temporal_parallax_penalty_strength   = 1.0f,
        .max_temporal_samples                 = 64,
        .fast_convergence_samples             = 32.0f,
        .post_blur_enabled                    = 1,
        .post_blur_ao_guiding          = 1,
        .post_blur_ao_guide_floor             = 0.2f,
        .post_blur_perceptual_difference_guiding     = 1,
        .post_blur_perceptual_radiance_guide_tolerance = 0.5f,
        .ao_guide_floor                = 0.2f,
        .max_visibility_pixel_range           = 20.0f,
        .post_blur_mode                       = 0,
        .post_blur_use_lds                    = 1,
        .post_blur_disocclusion_blur_enabled  = 0,
        .post_blur_stride                     = 2,
        .post_blur_max_width                  = 16,
        .post_blur_atrous_iterations          = 4,
        .upscale_enabled                      = 1,
        .sh_resolve_enabled                   = 1,
        .pre_blur_firefly_energy_compensation_enabled  = 1,
        .animate_noise                        = 1,
        .ray_percentage                       = 0.6f, // diffuse + specular rays share this HARD budget (min_ray_budget base floor is paid from it)
        .min_ray_budget                       = 0.125f,
        .max_rays_per_pixel                   = 12,
        .use_repacked_ray_dispatch            = 1,
        .use_ray_redistribution               = 1,
        .trace_use_stbn                       = 0,
        .pioneer_guiding_enabled               = 1,
        .guide_concentration                   = 0.92f,
        .guide_pioneer_trace_max_distance      = 256.0f,
        .specular_enabled                      = 1,
        .specular_firefly_perceptual_tolerance = 3.0f,
        .specular_max_temporal_frames          = 32.0f,
        .specular_pre_blur_scale               = 0.5f,
        .specular_post_blur_scale              = 0.5f,
        .specular_virtual_reprojection         = 1,
        .specular_catrom_history               = 1,
        .specular_upscale_detail_weighting     = 1,
        .ray_share_slope                       = 2.0f, // squared radiance ratio
        .ray_diffuse_metal_tolerance_stops     = 0.1f,
        .ray_specular_roughness_cutoff_start   = 0.8f,
        .ray_specular_roughness_cutoff_end     = 1.0f,
        .specular_guide_mix                    = 0.5f,
        .specular_upward_gloss                 = 0.0f,
        .specular_total_gloss                  = 0.0f,
        .specular_additive_metalness           = 0.0f,
        .specular_reprojection_disagreement_strength = 1.0f,
        .specular_fast_history_enabled         = 1,
        .specular_temporal_accumulation_enabled = 1,
        .specular_fast_convergence_samples     = 32.0f,
        .specular_fast_history_frames          = 4,
        .specular_temporal_firefly_filter_enabled = 1,
        .specular_temporal_firefly_std_dev_clamp = 3.5f,
        .specular_temporal_variance_fast_history_blend = 4.0f,
        .specular_temporal_parallax_penalty_strength = 1.0f,
        .specular_firefly_filter_enabled       = 1,
        .specular_firefly_clamp_mode           = 0,
        .specular_guide_concentration          = 0.92f,
        .specular_max_gloss                    = 0.9f,
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
    auto const & trace_target = ti.info(AT.perceptual_rgb_shortness).value();
    auto const & rt_pipeline = render_context->gpu_context->ray_tracing_pipelines.at(rtgi_trace_diffuse_compile_info().name);
    ti.recorder.set_pipeline(*rt_pipeline.pipeline);
    ti.recorder.push_constant(push);
    ti.recorder.trace_rays({
        .width = trace_target.size.x,
        .height = trace_target.size.y,
        .depth = 1,
        .shader_binding_table = rt_pipeline.sbt,
    });
}

// Pioneer trace, gated by rtgi_settings.pioneer_guiding_enabled -- see pioneer_ray_gen in rtgi_trace_diffuse.hlsl.
// Dispatched at the sparse pioneer grid size (pioneer_hit_y's extent), sharing the SAME ray tracing
// pipeline as the classic/repacked trace (is_pioneer_pass selects the raygen body).
inline void rtgi_trace_pioneer_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiTraceDiffuseH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RTGI", "GUIDE_PIONEER_TRACE">());
    RtgiTraceDiffusePush push = {
        .is_pioneer_pass = true,
    };
    push.attach = ti.allocator->allocate_fill(RtgiTraceDiffuseH::AttachmentShaderBlob{ti.attachment_shader_blob}).value().device_address;
    auto const & pioneer_target = ti.info(AT.pioneer_hit_y).value();
    auto const & rt_pipeline = render_context->gpu_context->ray_tracing_pipelines.at(rtgi_trace_diffuse_compile_info().name);
    ti.recorder.set_pipeline(*rt_pipeline.pipeline);
    ti.recorder.push_constant(push);
    ti.recorder.trace_rays({
        .width = pioneer_target.size.x,
        .height = pioneer_target.size.y,
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

inline void rtgi_temporal_reproject_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiTemporalReprojectH::Info::AT;
    dispatch_image_relative(RtgiTemporalReprojectPush(), ti, render_context, AT.half_res_sample_count, RTGI_TEMPORAL_X, RenderTimes::index<"RTGI", "TEMPORAL_REPROJECT">(), rtgi_temporal_reproject_compile_info().name);
}

inline void rtgi_temporal_accumulate_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiTemporalAccumulateH::Info::AT;
    dispatch_image_relative(RtgiTemporalAccumulatePush(), ti, render_context, AT.half_res_diffuse_accumulated, RTGI_TEMPORAL_X, RenderTimes::index<"RTGI", "TEMPORAL_ACCUMULATION">(), rtgi_temporal_accumulate_compile_info().name);
}

inline void rtgi_pre_filter_prepare_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiPreFilterH::Info::AT;
    dispatch_image_relative(RtgiPreFilterPush(), ti, render_context, AT.perceptual_rgb_shortness, RTGI_PRE_BLUR_PREPARE_X, RenderTimes::index<"RTGI", "PRE_FILTER">(), rtgi_pre_filter_prepare_compile_info().name);
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
    u32 const render_time = pass == 0 ? RenderTimes::index<"RTGI", "POST_BLUR_HORIZONTAL">() : RenderTimes::index<"RTGI", "POST_BLUR_VERTICAL">();
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, render_time);
    auto const dst_image_size = ti.info(AT.view_cam_half_res_depth).value().size;
    ti.recorder.set_pipeline(*(render_context->gpu_context->compute_pipelines.at(
        render_context->render_data.rtgi_settings.post_blur_use_lds
            ? rtgi_post_blur_lds_compile_info().name
            : rtgi_post_blur_compile_info().name)));
    RtgiPostBlurPush push{.pass = pass};
    push.attach = ti.allocator->allocate_fill(RtgiPostBlurH::AttachmentShaderBlob{ti.attachment_shader_blob}).value().device_address;
    push.size = {dst_image_size.x, dst_image_size.y};
    ti.recorder.push_constant(push);
    if (pass == 0)
    {
        // Horizontal: 16 threads along X, 4 along Y
        ti.recorder.dispatch({round_up_div(dst_image_size.x, RTGI_POST_BLUR_X), round_up_div(dst_image_size.y, RTGI_POST_BLUR_Y), 1});
    }
    else
    {
        // Vertical: swizzled — 16-wide dimension covers Y, so group tiles are 4x16 in image space
        ti.recorder.dispatch({round_up_div(dst_image_size.x, RTGI_POST_BLUR_Y), round_up_div(dst_image_size.y, RTGI_POST_BLUR_X), 1});
    }
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

inline void rtgi_distribute_rays_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiDistributeRaysH::Info::AT;
    dispatch_image_relative(RtgiDistributeRaysPush(), ti, render_context, AT.pixel_ray_alloc, RTGI_DISTRIBUTE_RAYS_X, RenderTimes::index<"RTGI", "DISTRIBUTE_RAYS">(), rtgi_distribute_rays_compile_info().name);
}

inline void rtgi_trace_from_list_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiTraceDiffuseH::Info::AT;
    // Report under the shared "TRACE" slot so the trace render time is populated in both the classic
    // and repacked paths (allocate/blend are timed separately under their own slots).
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RTGI", "TRACE">());
    RtgiTraceDiffusePush push = {};
    push.attach = ti.allocator->allocate_fill(RtgiTraceDiffuseH::AttachmentShaderBlob{ti.attachment_shader_blob}).value().device_address;
    auto const & rt_pipeline = render_context->gpu_context->ray_tracing_pipelines.at(rtgi_trace_diffuse_compile_info().name);
    ti.recorder.set_pipeline(*rt_pipeline.pipeline);
    ti.recorder.push_constant(push);
    auto const & trace_target = ti.info(AT.perceptual_rgb_shortness).value();
    // Must cover the whole oversized ray list capacity (base + extras), not just the pixel count.
    const u32 ray_list_capacity = trace_target.size.x * trace_target.size.y * RTGI_RAY_LIST_CAPACITY_MUL;
    // Dispatch as (128, 1, ceil(capacity/128)): shader computes ray_index = z*128 + x.
    // Over-dispatch is fine; the shader guards on ray_index >= ray_list_count.
    ti.recorder.trace_rays({
        .width  = 128,
        .height = 1,
        .depth  = round_up_div(ray_list_capacity, 128u),
        .shader_binding_table = rt_pipeline.sbt,
    });
}

inline void rtgi_blend_rays_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiBlendRaysH::Info::AT;
    dispatch_image_relative(RtgiBlendRaysPush(), ti, render_context, AT.perceptual_rgb_shortness, RTGI_BLEND_RAYS_X, RenderTimes::index<"RTGI", "BLEND_RAYS">(), rtgi_blend_rays_compile_info().name);
}

inline void rtgi_guide_resample_horizontal_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiGuideResampleHorizontalH::Info::AT;
    dispatch_image_relative(RtgiGuideResampleHorizontalPush(), ti, render_context, AT.h_resample_hit_y, RTGI_GUIDE_RESAMPLE_X, RenderTimes::index<"RTGI", "GUIDE_RESAMPLE_H">(), rtgi_guide_resample_horizontal_compile_info().name);
}

inline void rtgi_guide_resample_vertical_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiGuideResampleVerticalH::Info::AT;
    dispatch_image_relative(RtgiGuideResampleVerticalPush(), ti, render_context, AT.pioneer_guide_hit_y, RTGI_GUIDE_RESAMPLE_X, RenderTimes::index<"RTGI", "GUIDE_RESAMPLE_V">(), rtgi_guide_resample_vertical_compile_info().name);
}

inline void rtgi_guide_resolve_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtgiGuideResolveH::Info::AT;
    dispatch_image_relative(RtgiGuideResolvePush(), ti, render_context, AT.guide_sh_y, RTGI_GUIDE_RESAMPLE_X, RenderTimes::index<"RTGI", "GUIDE_RESOLVE">(), rtgi_guide_resolve_compile_info().name);
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
auto rtgi_create_diffuse_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_task_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, scale_div, name));
}

auto rtgi_create_diffuse2_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_task_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16_SFLOAT, scale_div, name));
}

// rgb = specular radiance, a = hit distance
auto rtgi_create_specular_image(daxa::TaskGraph & tg, RenderContext * render_context, std::string_view name, u32 scale_div = RTGI_PIXEL_SCALE_DIV)
{
    return tg.create_task_image(rtgi_create_common_transient_image_info(render_context, daxa::Format::R16G16B16A16_SFLOAT, scale_div, name));
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
    // Packed temporal counters (R16_UINT): normal sample count (10 bits, max 255, 0.25 steps) + fast
    // history frame count (6 bits, max 15, 0.25 steps). See rtgi_pack_sample_counts in rtgi_shared.hlsl.
    auto half_res_sample_count_history = info.tg.create_task_image({
        .format = daxa::Format::R16_UINT,
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
    auto fast_temporal_history = info.tg.create_task_image({
        .format = daxa::Format::R16G16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "fast_temporal_history_persistent",
    });
    auto half_res_ao_guide_history = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "half_res_ao_guide_history_persistent",
    });
    auto temporal_perceptual_radiance_history = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "temporal_perceptual_radiance_history_persistent",
    });
    // Specular history: .rgb radiance, .a hit distance; plus its own frame count (roughness-driven length).
    auto half_res_specular_history = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "half_res_specular_history_persistent",
    });
    // Specular fast history: .x fast brightness mean, .y fast relative variance, .z fast frame count.
    auto specular_fast_history = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "specular_fast_history_persistent",
    });
    auto specular_frames_history = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = half_res_image_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "specular_frames_history_persistent",
    });
    // Per-pixel specular ray stats: .rgb = mean log specular rgb (firefly ceiling input), .a = mean hit distance.
    auto specular_perceptual_rgb_hit_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = half_res_image_size,
        .name = "rtgi_specular_perceptual_rgb_hit_image",
    });

    // Per-pixel: .rgb = geometric mean of the traced rays in log space (mean log rgb); .a = mean ray
    // shortness [0,1]. Feeds the pre-filter firefly ceiling / geometric mean (perceptual radiance inferred from
    // .rgb) and the ambient-occlusion guide (shortness). Replaces the old separate ray_length + radiance channel.
    auto perceptual_rgb_shortness_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = half_res_image_size,
        .name = "rtgi_perceptual_rgb_shortness_image",
    });
    // Rays shot per pixel this frame: consumed by accumulation (sample count increment) and pre-filter (firefly ceiling).
    auto ray_count_image = info.tg.create_task_image({
        .format = daxa::Format::R8_UINT,
        .size = half_res_image_size,
        .name = "rtgi_ray_count_image",
    });

    // Reprojection metadata produced by the temporal reproject pass. This pass runs BEFORE trace so
    // the trace pass can read the reprojected sample count to drive adaptive ray count, and consumed
    // afterwards by the accumulation pass.
    auto reproject_corner_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16_UINT,
        .size = half_res_image_size,
        .name = "rtgi_reproject_corner_image",
    });
    auto reproject_weights_image = info.tg.create_task_image({
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = half_res_image_size,
        .name = "rtgi_reproject_weights_image",
    });

    // Ray demand: extra ray factor per signal (.x diffuse, .y specular), written by reproject (rtgi_calc_ray_share).
    auto ray_impact_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16_SFLOAT,
        .size = half_res_image_size,
        .name = "rtgi_ray_impact_image",
    });

    auto const pioneer_grid_size = daxa::Extent3D{
        round_up_div(half_res_image_size.x, RTGI_GUIDE_PIONEER_GRID_DIV),
        round_up_div(half_res_image_size.y, RTGI_GUIDE_PIONEER_GRID_DIV),
        1,
    };
    auto pioneer_hit_y_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = pioneer_grid_size,
        .name = "rtgi_pioneer_hit_y_image",
    });
    auto guide_sh_y_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = half_res_image_size,
        .name = "rtgi_guide_sh_y_image",
    });
    auto guide_confidence_image = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = half_res_image_size,
        .name = "rtgi_guide_confidence_image",
    });
    // Intermediate, pioneer-resolution-sized: horizontal pass's per-row pick + (weight, valid_count),
    // consumed by the vertical pass. See rtgi_guide_resample.hlsl for the two-stage reservoir math.
    auto h_resample_hit_y_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = pioneer_grid_size,
        .name = "rtgi_h_resample_hit_y_image",
    });
    auto h_resample_weight_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16_SFLOAT,
        .size = pioneer_grid_size,
        .name = "rtgi_h_resample_weight_image",
    });
    // Intermediate, ALSO pioneer-resolution-sized: the vertical pass's own output (one RIS pick per
    // pioneer cell). entry_guide_resolve upsamples this to the final half-res guide_sh_y_image/
    // guide_confidence_image above.
    auto pioneer_guide_hit_y_image = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = pioneer_grid_size,
        .name = "rtgi_pioneer_guide_hit_y_image",
    });
    auto pioneer_guide_confidence_image = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = pioneer_grid_size,
        .name = "rtgi_pioneer_guide_confidence_image",
    });
    bool const pioneer_guiding_enabled = info.render_context.render_data.rtgi_settings.pioneer_guiding_enabled != 0;

    // == Repacked ray dispatch resources =====================================
    const u32 half_res_pixels = half_res_image_size.x * half_res_image_size.y;
    // Oversized so the list holds one base ray per pixel PLUS the adaptive extra-ray budget.
    const u32 ray_list_capacity = half_res_pixels * RTGI_RAY_LIST_CAPACITY_MUL;

    auto ray_counters_buffer = info.tg.create_task_buffer({
        .size = sizeof(RtgiRayCounters),
        .name = "rtgi_ray_counters",
    });
    auto ray_list_buffer = info.tg.create_task_buffer({
        .size = sizeof(RtgiRayEntry) * ray_list_capacity,
        .name = "rtgi_ray_list",
    });
    auto ray_result_buffer = info.tg.create_task_buffer({
        .size = sizeof(RtgiRayResult) * ray_list_capacity,
        .name = "rtgi_ray_result",
    });
    // Single ray budget: a pixel's ray-list entries are [diffuse 0..n_d) then [specular 0..n_s). n_d lives in
    // ray_count_image, n_s here. The reprojected specular history sample count drives the specular demand.
    auto specular_ray_count_image = info.tg.create_task_image({
        .format = daxa::Format::R8_UINT,
        .size = half_res_image_size,
        .name = "rtgi_specular_ray_count_image",
    });
    auto specular_sample_count_image = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = half_res_image_size,
        .name = "rtgi_specular_sample_count_image",
    });
    auto pixel_ray_alloc_image = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = half_res_image_size,
        .name = "rtgi_pixel_ray_alloc",
    });

    info.tg.clear_buffer({.buffer = ray_counters_buffer, .name = "rtgi_ray_counters_clear"});

    if (pioneer_guiding_enabled)
    {
        info.tg.add_task(daxa::HeadTask<RtgiTraceDiffuseH::Info>("RtgiGuidePioneerTrace")
            .head_views(RtgiTraceDiffuseH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .perceptual_rgb_shortness = perceptual_rgb_shortness_image,
                .ray_count_image = ray_count_image,
                .rtgi_sample_count = half_res_sample_count_history.current(),
                .ray_counters = ray_counters_buffer,
                .ray_list = ray_list_buffer,
                .ray_result = ray_result_buffer,
                .specular_ray_count_image = specular_ray_count_image,
                .specular_sample_count = specular_sample_count_image,
                .specular_perceptual_rgb_hit = specular_perceptual_rgb_hit_image,
                .pixel_ray_alloc = pixel_ray_alloc_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                .view_cam_half_res_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
                .pioneer_hit_y = pioneer_hit_y_image,
                .guide_sh_y = guide_sh_y_image,
                .meshlet_instances = info.meshlet_instances,
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
                .ray_impact = ray_impact_image,
            })
            .executes(rtgi_trace_pioneer_callback, &info.render_context));

        info.tg.add_task(daxa::HeadTask<RtgiGuideResampleHorizontalH::Info>()
            .head_views(RtgiGuideResampleHorizontalH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                .pioneer_hit_y = pioneer_hit_y_image,
                .h_resample_hit_y = h_resample_hit_y_image,
                .h_resample_weight = h_resample_weight_image,
            })
            .executes(rtgi_guide_resample_horizontal_callback, &info.render_context));

        info.tg.add_task(daxa::HeadTask<RtgiGuideResampleVerticalH::Info>()
            .head_views(RtgiGuideResampleVerticalH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                .h_resample_hit_y = h_resample_hit_y_image,
                .h_resample_weight = h_resample_weight_image,
                .pioneer_guide_hit_y = pioneer_guide_hit_y_image,
                .pioneer_guide_confidence = pioneer_guide_confidence_image,
            })
            .executes(rtgi_guide_resample_vertical_callback, &info.render_context));

        info.tg.add_task(daxa::HeadTask<RtgiGuideResolveH::Info>()
            .head_views(RtgiGuideResolveH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                .pioneer_guide_hit_y = pioneer_guide_hit_y_image,
                .pioneer_guide_confidence = pioneer_guide_confidence_image,
                .guide_sh_y = guide_sh_y_image,
                .guide_confidence = guide_confidence_image,
            })
            .executes(rtgi_guide_resolve_callback, &info.render_context));
    }

    info.tg.add_task(daxa::HeadTask<RtgiTemporalReprojectH::Info>()
        .head_views(RtgiTemporalReprojectH::Info::Views{
            .globals = info.render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .half_res_depth = info.view_cam_half_res_depth.current(),
            .half_res_normal = info.view_cam_half_res_face_normals.current(),
            .half_res_depth_history = info.view_cam_half_res_depth.previous(),
            .half_res_normal_history = info.view_cam_half_res_face_normals.previous(),
            .half_res_sample_count_history = half_res_sample_count_history.previous(),
            .half_res_sample_count = half_res_sample_count_history.current(),
            .reproject_corner = reproject_corner_image,
            .reproject_weights = reproject_weights_image,
            .ray_counters = ray_counters_buffer,
            .specular_frames_history = specular_frames_history.previous(),
            .specular_sample_count = specular_sample_count_image,
            .half_res_albedo_metalness = info.view_cam_half_res_albedo_metalness,
            .half_res_diffuse_history = half_res_diffuse_history.previous(),
            .half_res_specular_history = half_res_specular_history.previous(),
            .ray_impact = ray_impact_image,
            .half_res_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
        })
        .executes(rtgi_temporal_reproject_callback, &info.render_context));

    bool const use_repacked_ray_dispatch = info.render_context.render_data.rtgi_settings.use_repacked_ray_dispatch != 0;

    if (!use_repacked_ray_dispatch)
    {
        // Classic path: one trace dispatch per pixel writes all trace outputs directly.
        info.tg.add_task(daxa::HeadTask<RtgiTraceDiffuseH::Info>()
                .head_views(RtgiTraceDiffuseH::Info::Views{
                    .globals = info.render_context.tgpu_render_data.view(),
                    .debug_image = info.debug_image,
                    .perceptual_rgb_shortness = perceptual_rgb_shortness_image,
                    .ray_count_image = ray_count_image,
                    .rtgi_sample_count = half_res_sample_count_history.current(),
                    .ray_counters = ray_counters_buffer,
                    .ray_list = ray_list_buffer,
                    .ray_result = ray_result_buffer,
                    .specular_ray_count_image = specular_ray_count_image,
                    .specular_sample_count = specular_sample_count_image,
                    .specular_perceptual_rgb_hit = specular_perceptual_rgb_hit_image,
                    .pixel_ray_alloc = pixel_ray_alloc_image,
                    .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                    .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                    .view_cam_half_res_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
                    .pioneer_hit_y = pioneer_hit_y_image,
                    .guide_sh_y = guide_sh_y_image,
                    .meshlet_instances = info.meshlet_instances,
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
                    .ray_impact = ray_impact_image,
                })
                .executes(rtgi_trace_diffuse_callback, &info.render_context, RtgiTraceDiffuseCallbackInfo{.debug_primary_trace = false}));
    }
    else
    {
        // Repacked path: allocate a global ray budget across tiles, trace from a flat ray list,
        // then blend the per-ray results back into the per-pixel trace outputs.

        // == Allocate rays ====================================================
        info.tg.add_task(daxa::HeadTask<RtgiDistributeRaysH::Info>()
            .head_views(RtgiDistributeRaysH::Info::Views{
                .globals          = info.render_context.tgpu_render_data.view(),
                .debug_image      = info.debug_image,
                .ray_counters     = ray_counters_buffer,
                .rtgi_sample_count = half_res_sample_count_history.current(),
                .ray_list         = ray_list_buffer,
                .pixel_ray_alloc  = pixel_ray_alloc_image,
                .ray_count_image  = ray_count_image,
                .specular_sample_count = specular_sample_count_image,
                .specular_ray_count_image = specular_ray_count_image,
                .ray_impact = ray_impact_image,
            })
            .executes(rtgi_distribute_rays_callback, &info.render_context));

        // == Trace from ray list ===============================================
        info.tg.add_task(daxa::HeadTask<RtgiTraceDiffuseH::Info>("RtgiTraceFromList")
            .head_views(RtgiTraceDiffuseH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .perceptual_rgb_shortness = perceptual_rgb_shortness_image,
                .ray_count_image = ray_count_image,
                .rtgi_sample_count = half_res_sample_count_history.current(),
                .ray_counters = ray_counters_buffer,
                .ray_list = ray_list_buffer,
                .ray_result = ray_result_buffer,
                .specular_ray_count_image = specular_ray_count_image,
                .specular_sample_count = specular_sample_count_image,
                .specular_perceptual_rgb_hit = specular_perceptual_rgb_hit_image,
                .pixel_ray_alloc = pixel_ray_alloc_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                .view_cam_half_res_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
                .pioneer_hit_y = pioneer_hit_y_image,
                .guide_sh_y = guide_sh_y_image,
                .meshlet_instances = info.meshlet_instances,
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
                .ray_impact = ray_impact_image,
            })
            .executes(rtgi_trace_from_list_callback, &info.render_context));

        // == Blend rays into trace outputs =====================================
        info.tg.add_task(daxa::HeadTask<RtgiBlendRaysH::Info>()
            .head_views(RtgiBlendRaysH::Info::Views{
                .globals              = info.render_context.tgpu_render_data.view(),
                .pixel_ray_alloc      = pixel_ray_alloc_image,
                .ray_result           = ray_result_buffer,
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .perceptual_rgb_shortness = perceptual_rgb_shortness_image,
                .ray_count_image      = ray_count_image,
                .specular_ray_count_image = specular_ray_count_image,
                .specular_perceptual_rgb_hit = specular_perceptual_rgb_hit_image,
            })
            .executes(rtgi_blend_rays_callback, &info.render_context));
    }

    // auto half_res_sample_count_image = rtgi_create_sample_count_image(info.tg, &info.render_context, "half_res_sample_count_image");

    auto pre_filtered_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "pre_filtered_diffuse_image");
    auto pre_filtered_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "pre_filtered_diffuse2_image");
    auto pre_filtered_specular_image = rtgi_create_specular_image(info.tg, &info.render_context, "pre_filtered_specular_image");
    auto firefly_factor_image = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "firefly_factor_image",
    });    
    auto specular_firefly_factor_image = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "specular_firefly_factor_image",
    });
    auto perceptual_radiance_image = info.tg.create_task_image({
        .format = daxa::Format::R16_SFLOAT,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "perceptual_radiance_image",
    });
    auto ao_guide_image = info.tg.create_task_image({
        .format = daxa::Format::R8_UNORM,
        .size = {
            info.render_context.render_data.settings.render_target_size.x / 2,
            info.render_context.render_data.settings.render_target_size.y / 2,
            1,
        },
        .name = "ao_guide_image",
    });
    info.tg.add_task(daxa::HeadTask<RtgiPreFilterH::Info>()
            .head_views(RtgiPreFilterH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .perceptual_rgb_shortness = perceptual_rgb_shortness_image,
                .ray_count_image = ray_count_image,
                .pixel_ray_alloc = pixel_ray_alloc_image,
                .ray_result = ray_result_buffer,
                .view_cam_half_res_normals = info.view_cam_half_res_face_normals.current(),
                .pre_filtered_diffuse_image = pre_filtered_diffuse_image,
                .pre_filtered_diffuse2_image = pre_filtered_diffuse2_image,
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .firefly_factor_image = firefly_factor_image,
                .perceptual_radiance_image = perceptual_radiance_image,
                .ao_guide_image = ao_guide_image,
                .specular_ray_count_image = specular_ray_count_image,
                .specular_perceptual_rgb_hit = specular_perceptual_rgb_hit_image,
                .pre_filtered_specular_image = pre_filtered_specular_image,
                .view_cam_half_res_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
                .ray_counters = ray_counters_buffer,
                .specular_firefly_factor_image = specular_firefly_factor_image,
            })
            .executes(rtgi_pre_filter_prepare_callback, &info.render_context));

    daxa::TaskImageView post_pre_blur_diffuse_image = daxa::NullTaskImage;
    daxa::TaskImageView post_pre_blur_diffuse2_image = daxa::NullTaskImage;
    daxa::TaskImageView post_pre_blur_specular_image = daxa::NullTaskImage;
    if (info.render_context.render_data.rtgi_settings.pre_blur_enabled)
    {
        u32 const iterations = static_cast<u32>(info.render_context.render_data.rtgi_settings.pre_blur_iterations);

        auto pre_blurred_diffuse_ping = rtgi_create_diffuse_image(info.tg, &info.render_context, "pre_blurred_diffuse_ping");
        auto pre_blurred_diffuse2_ping = rtgi_create_diffuse2_image(info.tg, &info.render_context, "pre_blurred_diffuse2_ping");
        auto pre_blurred_specular_ping = rtgi_create_specular_image(info.tg, &info.render_context, "pre_blurred_specular_ping");
        daxa::TaskImageView pre_blurred_diffuse_pong = daxa::NullTaskImage;
        daxa::TaskImageView pre_blurred_diffuse2_pong = daxa::NullTaskImage;
        daxa::TaskImageView pre_blurred_specular_pong = daxa::NullTaskImage;
        if (iterations >= 2)
        {
            pre_blurred_diffuse_pong = rtgi_create_diffuse_image(info.tg, &info.render_context, "pre_blurred_diffuse_pong");
            pre_blurred_diffuse2_pong = rtgi_create_diffuse2_image(info.tg, &info.render_context, "pre_blurred_diffuse2_pong");
            pre_blurred_specular_pong = rtgi_create_specular_image(info.tg, &info.render_context, "pre_blurred_specular_pong");
        }

        daxa::TaskImageView src_diffuse = pre_filtered_diffuse_image;
        daxa::TaskImageView src_diffuse2 = pre_filtered_diffuse2_image;
        daxa::TaskImageView src_specular = pre_filtered_specular_image;
        for (u32 i = 0; i < iterations; ++i)
        {
            daxa::TaskImageView dst_diffuse = (i % 2 == 0) ? pre_blurred_diffuse_ping : pre_blurred_diffuse_pong;
            daxa::TaskImageView dst_diffuse2 = (i % 2 == 0) ? pre_blurred_diffuse2_ping : pre_blurred_diffuse2_pong;
            daxa::TaskImageView dst_specular = (i % 2 == 0) ? pre_blurred_specular_ping : pre_blurred_specular_pong;
            info.tg.add_task(daxa::HeadTask<RtgiPreBlurH::Info>(std::string("RtgiPreBlur") + std::to_string(i))
                    .head_views(RtgiPreBlurH::Info::Views{
                        .globals = info.render_context.tgpu_render_data.view(),
                        .debug_image = info.debug_image,
                                    .rtgi_diffuse_before = src_diffuse,
                        .rtgi_diffuse2_before = src_diffuse2,
                        .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                        .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
        
                        .rtgi_diffuse_blurred = dst_diffuse,
                        .rtgi_diffuse2_blurred = dst_diffuse2,
                        .firefly_factor_image = firefly_factor_image,
                        .perceptual_radiance_image = perceptual_radiance_image,
                        .ao_guide_image = ao_guide_image,
                        .ray_count_image = ray_count_image,
                        .rtgi_specular_before = src_specular,
                        .rtgi_specular_blurred = dst_specular,
                        .specular_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
                        .specular_firefly_factor_image = specular_firefly_factor_image,
                        .specular_ray_count_image = specular_ray_count_image,
                    })
                    .executes(rtgi_pre_blur_diffuse_callback, &info.render_context, i));
            src_diffuse = dst_diffuse;
            src_diffuse2 = dst_diffuse2;
            src_specular = dst_specular;
        }
        post_pre_blur_diffuse_image = src_diffuse;
        post_pre_blur_diffuse2_image = src_diffuse2;
        post_pre_blur_specular_image = src_specular;
    }

    // Temporal Accumulation: reads the reprojection metadata to blend history with the new frame.
    info.tg.add_task(daxa::HeadTask<RtgiTemporalAccumulateH::Info>()
        .head_views(RtgiTemporalAccumulateH::Info::Views{
            .globals = info.render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .half_res_sample_count = half_res_sample_count_history.current(),
            .half_res_sample_count_history = half_res_sample_count_history.previous(),
            .ray_count_image = ray_count_image,
            .reproject_corner = reproject_corner_image,
            .reproject_weights = reproject_weights_image,
            .half_res_diffuse_pre_blurred = post_pre_blur_diffuse_image,
            .pre_filtered_diffuse_new = pre_filtered_diffuse_image,
            .pre_filtered_diffuse2_new = pre_filtered_diffuse2_image,
            .half_res_diffuse_accumulated = half_res_diffuse_history.current(),
            .half_res_diffuse_history = half_res_diffuse_history.previous(),
            .half_res_diffuse2_pre_blurred = post_pre_blur_diffuse2_image,
            .half_res_diffuse2_accumulated = half_res_diffuse2_history.current(),
            .half_res_diffuse2_history = half_res_diffuse2_history.previous(),
            .fast_temporal_history_accumulated = fast_temporal_history.current(),
            .fast_temporal_history_history = fast_temporal_history.previous(),
            .ao_guide_new = ao_guide_image,
            .half_res_ao_guide_accumulated = half_res_ao_guide_history.current(),
            .half_res_ao_guide_history = half_res_ao_guide_history.previous(),
            .perceptual_radiance_new = perceptual_radiance_image,
            .temporal_perceptual_radiance_accumulated = temporal_perceptual_radiance_history.current(),
            .temporal_perceptual_radiance_history = temporal_perceptual_radiance_history.previous(),
            .specular_pre_blurred = post_pre_blur_specular_image,
            .pre_filtered_specular_new = pre_filtered_specular_image,
            .half_res_specular_accumulated = half_res_specular_history.current(),
            .half_res_specular_history = half_res_specular_history.previous(),
            .specular_frames_accumulated = specular_frames_history.current(),
            .specular_frames_history = specular_frames_history.previous(),
            .half_res_depth = info.view_cam_half_res_depth.current(),
            .half_res_face_normals = info.view_cam_half_res_face_normals.current(),
            .half_res_depth_history = info.view_cam_half_res_depth.previous(),
            .half_res_face_normals_history = info.view_cam_half_res_face_normals.previous(),
            .specular_ray_count_image = specular_ray_count_image,
            .half_res_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
            .half_res_normal_roughness_history = info.view_cam_half_res_normal_roughness.previous(),
            .ray_counters = ray_counters_buffer,
            .specular_fast_history_accumulated = specular_fast_history.current(),
            .specular_fast_history_history = specular_fast_history.previous(),
        })
        .executes(rtgi_temporal_accumulate_callback, &info.render_context));

    auto rtgi_post_blur_diffuse_image = half_res_diffuse_history.current();
    auto rtgi_post_blur_diffuse2_image = half_res_diffuse2_history.current();
    daxa::TaskImageView rtgi_post_blur_specular_image = half_res_specular_history.current();
    if (info.render_context.render_data.rtgi_settings.post_blur_enabled)
    {
        auto const & rtgi_settings = info.render_context.render_data.rtgi_settings;
        if (rtgi_settings.post_blur_mode == 0)
        {
            // Bilateral separable: horizontal then vertical pass
            auto rtgi_post_blur_pass0_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_post_blur_pass0_diffuse_image");
            auto rtgi_post_blur_pass0_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_post_blur_pass0_diffuse2_image");
            auto rtgi_post_blur_pass0_specular_image = rtgi_create_specular_image(info.tg, &info.render_context, "rtgi_post_blur_pass0_specular_image");
            info.tg.add_task(daxa::HeadTask<RtgiPostBlurH::Info>("RtgiPostBlurHorizontal")
                    .head_views(RtgiPostBlurH::Info::Views{
                        .globals = info.render_context.tgpu_render_data.view(),
                        .debug_image = info.debug_image,
                                    .rtgi_sample_count = half_res_sample_count_history,
                        .rtgi_diffuse_before = half_res_diffuse_history,
                        .rtgi_diffuse2_before = half_res_diffuse2_history,
                        .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                        .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
        
                        .rtgi_diffuse_blurred = rtgi_post_blur_pass0_diffuse_image,
                        .rtgi_diffuse2_blurred = rtgi_post_blur_pass0_diffuse2_image,
                        .ao_guide_image = half_res_ao_guide_history.current(),
                        .temporal_perceptual_radiance = temporal_perceptual_radiance_history.current(),
                        .rtgi_specular_before = half_res_specular_history.current(),
                        .rtgi_specular_blurred = rtgi_post_blur_pass0_specular_image,
                        .specular_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
                    })
                    .executes(rtgi_post_blur_diffuse_callback, &info.render_context, 0u));

            rtgi_post_blur_diffuse_image = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_post_blur_diffuse_image");
            rtgi_post_blur_diffuse2_image = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_post_blur_diffuse2_image");
            rtgi_post_blur_specular_image = rtgi_create_specular_image(info.tg, &info.render_context, "rtgi_post_blur_specular_image");
            info.tg.add_task(daxa::HeadTask<RtgiPostBlurH::Info>("RtgiPostBlurVertical")
                    .head_views(RtgiPostBlurH::Info::Views{
                        .globals = info.render_context.tgpu_render_data.view(),
                        .debug_image = info.debug_image,
                                    .rtgi_sample_count = half_res_sample_count_history,
                        .rtgi_diffuse_before = rtgi_post_blur_pass0_diffuse_image,
                        .rtgi_diffuse2_before = rtgi_post_blur_pass0_diffuse2_image,
                        .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                        .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
        
                        .rtgi_diffuse_blurred = rtgi_post_blur_diffuse_image,
                        .rtgi_diffuse2_blurred = rtgi_post_blur_diffuse2_image,
                        .ao_guide_image = half_res_ao_guide_history.current(),
                        .temporal_perceptual_radiance = temporal_perceptual_radiance_history.current(),
                        .rtgi_specular_before = rtgi_post_blur_pass0_specular_image,
                        .rtgi_specular_blurred = rtgi_post_blur_specular_image,
                        .specular_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
                    })
                    .executes(rtgi_post_blur_diffuse_callback, &info.render_context, 1u));
        }
        else
        {
            // À-trous: N passes ping-ponging, step_size doubles each pass (1, 2, 4, ...)
            u32 const iterations = static_cast<u32>(rtgi_settings.post_blur_atrous_iterations);

            auto atrous_ping = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_atrous_ping");
            auto atrous_ping2 = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_atrous_ping2");
            auto atrous_ping_spec = rtgi_create_specular_image(info.tg, &info.render_context, "rtgi_atrous_ping_spec");
            daxa::TaskImageView atrous_pong = daxa::NullTaskImage;
            daxa::TaskImageView atrous_pong2 = daxa::NullTaskImage;
            daxa::TaskImageView atrous_pong_spec = daxa::NullTaskImage;
            if (iterations >= 2)
            {
                atrous_pong = rtgi_create_diffuse_image(info.tg, &info.render_context, "rtgi_atrous_pong");
                atrous_pong2 = rtgi_create_diffuse2_image(info.tg, &info.render_context, "rtgi_atrous_pong2");
                atrous_pong_spec = rtgi_create_specular_image(info.tg, &info.render_context, "rtgi_atrous_pong_spec");
            }

            daxa::TaskImageView src = half_res_diffuse_history.current();
            daxa::TaskImageView src2 = half_res_diffuse2_history.current();
            daxa::TaskImageView src_spec = half_res_specular_history.current();
            for (u32 i = 0; i < iterations; ++i)
            {
                daxa::TaskImageView dst  = (i % 2 == 0) ? atrous_ping  : atrous_pong;
                daxa::TaskImageView dst2 = (i % 2 == 0) ? atrous_ping2 : atrous_pong2;
                daxa::TaskImageView dst_spec = (i % 2 == 0) ? atrous_ping_spec : atrous_pong_spec;
                info.tg.add_task(daxa::HeadTask<RtgiPostBlurH::Info>(std::string("RtgiAtrousPostBlur") + std::to_string(i))
                        .head_views(RtgiPostBlurH::Info::Views{
                            .globals = info.render_context.tgpu_render_data.view(),
                            .debug_image = info.debug_image,
                                            .rtgi_sample_count = half_res_sample_count_history,
                            .rtgi_diffuse_before = src,
                            .rtgi_diffuse2_before = src2,
                            .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                            .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
            
                            .rtgi_diffuse_blurred = dst,
                            .rtgi_diffuse2_blurred = dst2,
                            .ao_guide_image = half_res_ao_guide_history.current(),
                            .temporal_perceptual_radiance = temporal_perceptual_radiance_history.current(),
                            .rtgi_specular_before = src_spec,
                            .rtgi_specular_blurred = dst_spec,
                            .specular_normal_roughness = info.view_cam_half_res_normal_roughness.current(),
                        })
                        .executes(rtgi_atrous_post_blur_callback, &info.render_context, i));
                src = dst;
                src2 = dst2;
                src_spec = dst_spec;
            }
            rtgi_post_blur_diffuse_image = src;
            rtgi_post_blur_diffuse2_image = src2;
            rtgi_post_blur_specular_image = src_spec;
        }
    }

    auto resolved_per_pixel_diffuse = rtgi_create_upscaled_diffuse_image(info.tg, &info.render_context, "resolved_per_pixel_diffuse");
    auto resolved_per_pixel_specular = rtgi_create_upscaled_diffuse_image(info.tg, &info.render_context, "resolved_per_pixel_specular");

    info.tg.add_task(daxa::HeadTask<RtgiUpscaleDiffuseH::Info>()
            .head_views(RtgiUpscaleDiffuseH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
    
                .diffuse_half_res = rtgi_post_blur_diffuse_image,
                .diffuse2_half_res = rtgi_post_blur_diffuse2_image,

                .samplecount_half_res = half_res_sample_count_history,
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                .view_cam_depth = info.view_cam_depth,
                .view_cam_face_normals = info.view_cam_face_normals,
                .view_camera_detail_normal_image = info.view_camera_detail_normal_image,
                .diffuse_resolved = resolved_per_pixel_diffuse,
                .specular_half_res = rtgi_post_blur_specular_image,
                .specular_resolved = resolved_per_pixel_specular,
                .specular_normal_roughness_half_res = info.view_cam_half_res_normal_roughness.current(),
                .ray_counters = ray_counters_buffer,
            })
            .executes(rtgi_upscale_diffuse_callback, &info.render_context));

    return TasksRtgiMainResult{
        .opaque_diffuse = resolved_per_pixel_diffuse,
        .opaque_specular = resolved_per_pixel_specular,
    };
}