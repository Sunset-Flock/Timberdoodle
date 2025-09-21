#pragma once

#include "clouds.inl"

MAKE_COMPUTE_COMPILE_INFO(raymarch_clouds_compile_info, "./src/rendering/volumetric/raymarch_clouds.hlsl", "entry_raymarch")
MAKE_COMPUTE_COMPILE_INFO(compose_clouds_compile_info, "./src/rendering/volumetric/compose_clouds.hlsl", "entry_compose")

inline auto raymarch_clouds_debug_compile_info() -> daxa::ComputePipelineCompileInfo2 const &
{
    static auto compile_info = []() -> daxa::ComputePipelineCompileInfo2 {
        auto _compile_info = raymarch_clouds_compile_info();
        _compile_info.defines.push_back(daxa::ShaderDefine{
            .name = "DEBUG_RAYMARCH",
            .value = "1"
        });
        _compile_info.name.append("DEBUG permutation");
        return _compile_info;
    }();

    return compile_info;
}

// Debug raymarch is a separate dispatch which launchs a single thread that will perform the raymarch and output debug information.
inline void raymarch_clouds_callback(daxa::TaskInterface ti, RenderContext * render_context, bool const debug_raymarch)
{
    if (! debug_raymarch)
    {
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"CLOUDS", "RAYMARCH">());
    }

    RaymarchCloudsPush push = {};
    push.clouds_resolution = render_context->render_data.settings.render_target_size;
    push.attach = ti.allocator->allocate_fill(RaymarchCloudsH::AttachmentShaderBlob{ti.attachment_shader_blob}).value().device_address;

    auto const & pipeline = debug_raymarch ? 
        render_context->gpu_context->compute_pipelines.at(raymarch_clouds_debug_compile_info().name) : 
        render_context->gpu_context->compute_pipelines.at(raymarch_clouds_compile_info().name);

    ti.recorder.set_pipeline(*pipeline);
    ti.recorder.push_constant(push);

    auto const dispatch_size = debug_raymarch ? 
        daxa_u32vec2{1u, 1u} :
        round_up_div(push.clouds_resolution, {RAYMARCH_CLOUDS_DISPATCH_X, RAYMARCH_CLOUDS_DISPATCH_Y});
    ti.recorder.dispatch({dispatch_size.x, dispatch_size.y, 1});

    if (! debug_raymarch)
    {
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"CLOUDS", "RAYMARCH">());
    }
}

inline void compose_clouds_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto gpu_time = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"CLOUDS", "COMPOSE">());

    ComposeCloudsPush push = {};
    push.main_screen_resolution = render_context->render_data.settings.render_target_size;
    push.attach = ti.allocator->allocate_fill(ComposeCloudsH::AttachmentShaderBlob{ti.attachment_shader_blob}).value().device_address;

    auto const & pipeline = render_context->gpu_context->compute_pipelines.at(compose_clouds_compile_info().name);
    ti.recorder.set_pipeline(*pipeline);
    ti.recorder.push_constant(push);

    auto const dispatch_size = round_up_div(push.main_screen_resolution, {COMPOSE_CLOUDS_DISPATCH_X, COMPOSE_CLOUDS_DISPATCH_Y});
    ti.recorder.dispatch({dispatch_size.x, dispatch_size.y, 1});
}