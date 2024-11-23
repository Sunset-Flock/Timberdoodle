#pragma once

#include "gen_gbuffer.hlsl"

#include <daxa/utils/pipeline_manager.hpp>
#include "../scene_renderer_context.hpp"

/// WARNING: SHADING NORMAL ROUGHNESS UNIMPLEMENTED
/// ===== GBUFFER FORMAT =====
// - Gbuffer directly constructed from compute decoding visbuffer, perfect pixel efficiency.
// - Can afford unusually fat gbuffer as we have 0 overdraw for the gbuffer.
//
// 1. depth 32 bit
// 2. octa encoded 32 bit geometric normals
// 3. octa encoded 24 bit shading normals
// 4. unorm encoded 8 bit roughness
//
/// ===== GBUFFER FORMAT =====

// Octahedral 32bit uint encoded float3 world normal.
static constexpr inline daxa::Format GBUFFER_GEO_NORMAL_FORMAT = daxa::Format::R32_UINT;
// Octagedral 24 bit uint encoded float3 shading normal + 8 bit uint unorm encoded roughness float.
static constexpr inline daxa::Format GBUFFER_SHADING_NORMAL_ROUGHNESS_FORMAT = daxa::Format::R32_UINT;

inline auto gen_gbuffer_pipeline_compile_info()
{
    return daxa::ComputePipelineCompileInfo2{ .source = daxa::ShaderFile{"./src/rendering/tasks/gen_gbuffer.hlsl"}, .entry_point = "entry_gen_gbuffer", .push_constant_size = sizeof(GenGbufferPush), .name = "GenGbuffer" };
}

struct GenGbufferTask : GenGbufferH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(gen_gbuffer_pipeline_compile_info().name));

        auto const info = ti.info(AT.geo_normal_image).value();
        GenGbufferPush push = {};
        push.attachments = ti.attachment_shader_blob;
        push.size = {static_cast<f32>(info.size.x), static_cast<f32>(info.size.y)};
        push.inv_size = {1.0f / push.size.x, 1.0f / push.size.y};
        ti.recorder.push_constant(push);

        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::SHADE_GBUFFER);
        u32 const dispatch_x = round_up_div(info.size.x, GEN_GBUFFER_X);
        u32 const dispatch_y = round_up_div(info.size.y, GEN_GBUFFER_Y);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::SHADE_GBUFFER);
    }
};