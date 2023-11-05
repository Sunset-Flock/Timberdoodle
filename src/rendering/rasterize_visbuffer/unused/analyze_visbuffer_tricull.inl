#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"

#define ANALYZE_VIS_BUFFER_WORKGROUP_X 8
#define ANALYZE_VIS_BUFFER_WORKGROUP_Y 16

DAXA_DECL_TASK_USES_BEGIN(AnalyzeVisbuffer, 1)
DAXA_TASK_USE_IMAGE(u_visbuffer, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(MeshletInstance), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshlet_visibility_bitfields, daxa_RWBufferPtr(daxa_u32vec4), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(u_debug_buffer, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()

struct AnalyzeVisbufferPush
{
    daxa_u32 width;
    daxa_u32 height;
};

#if __cplusplus

#include "../gpu_context.hpp"

struct AnalyzeVisBufferTask
{
    DAXA_USE_TASK_HEADER(AnalyzeVisbuffer)
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/analyze_visbuffer.glsl"}},
        .push_constant_size = sizeof(AnalyzeVisbufferPush),
        .name = std::string{AnalyzeVisbuffer::NAME},
    };
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(AnalyzeVisbuffer::NAME));
        auto const x = ti.get_device().info_image(uses.u_visbuffer.image()).size.x;
        auto const y = ti.get_device().info_image(uses.u_visbuffer.image()).size.y;
        cmd.push_constant(AnalyzeVisbufferPush{
            .width = x,
            .height = y,
        });
        auto const dispatch_x = round_up_div(x, ANALYZE_VIS_BUFFER_WORKGROUP_X * 2);
        auto const dispatch_y = round_up_div(y, ANALYZE_VIS_BUFFER_WORKGROUP_Y * 2);
        cmd.dispatch(dispatch_x, dispatch_y, 1);
    }
};
#endif