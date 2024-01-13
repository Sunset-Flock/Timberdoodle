#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"

#define ANALYZE_VIS_BUFFER_WORKGROUP_X 8
#define ANALYZE_VIS_BUFFER_WORKGROUP_Y 8

DAXA_DECL_TASK_HEAD_BEGIN(AnalyzeVisbuffer2, 4)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, visbuffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), meshlet_visibility_bitfield)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VisibleMeshletList), visible_meshlets)
// DxDy Image
// UV Image
DAXA_DECL_TASK_HEAD_END

struct AnalyzeVisbufferPush2
{
    daxa_BufferPtr(ShaderGlobals) globals;
    daxa_u32vec2 size;
    DAXA_TH_BLOB(AnalyzeVisbuffer2) uses;
};

#if __cplusplus

#include "../../gpu_context.hpp"

struct AnalyzeVisBufferTask2 : AnalyzeVisbuffer2
{
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/analyze_visbuffer.glsl"}},
        .push_constant_size = sizeof(AnalyzeVisbufferPush2),
        .name = std::string{AnalyzeVisbuffer2{}.name()},
    };
    GPUContext * context = {};
    virtual void callback(daxa::TaskInterface ti) const override
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(AnalyzeVisbuffer2{}.name()));
        auto [x,y,z] = ti.device.info_image(ti.img(visbuffer).ids[0]).value().size;
        AnalyzeVisbufferPush2 push = {
            .globals = context->shader_globals_address,
            .size = {x, y},
            .uses = span_to_array<DAXA_TH_BLOB(AnalyzeVisbuffer2){}.size()>(ti.attachment_shader_data_blob),
        };
        ti.recorder.push_constant(push);
        auto const dispatch_x = round_up_div(x, ANALYZE_VIS_BUFFER_WORKGROUP_X * 2);
        auto const dispatch_y = round_up_div(y, ANALYZE_VIS_BUFFER_WORKGROUP_Y * 2);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z =  1});
    }
};
#endif