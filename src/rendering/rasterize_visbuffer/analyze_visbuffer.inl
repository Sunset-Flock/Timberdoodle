#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry_pipeline.inl"

#define ANALYZE_VIS_BUFFER_WORKGROUP_X 8
#define ANALYZE_VIS_BUFFER_WORKGROUP_Y 8

DAXA_DECL_TASK_HEAD_BEGIN(AnalyzeVisbuffer2H, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, visbuffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), meshlet_visibility_bitfield)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VisibleMeshletList), visible_meshlets)
// DxDy Image
// UV Image
DAXA_DECL_TASK_HEAD_END

struct AnalyzeVisbufferPush2
{
    DAXA_TH_BLOB(AnalyzeVisbuffer2H, uses)
    daxa_u32vec2 size;
};

#if __cplusplus

#include "../../gpu_context.hpp"

inline daxa::ComputePipelineCompileInfo analyze_visbufer_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/analyze_visbuffer.glsl"}},
        .push_constant_size = s_cast<u32>(sizeof(AnalyzeVisbufferPush2)),
        .name = std::string{AnalyzeVisbuffer2H::NAME},
    };
};
struct AnalyzeVisBufferTask2 : AnalyzeVisbuffer2H::Task
{
    AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(analyze_visbufer_pipeline_compile_info().name));
        auto [x, y, z] = ti.device.info_image(ti.get(AnalyzeVisbuffer2H::AT.visbuffer).ids[0]).value().size;
        AnalyzeVisbufferPush2 push{.size = {x, y}};
        assign_blob(push.uses, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        auto const dispatch_x = round_up_div(x, ANALYZE_VIS_BUFFER_WORKGROUP_X * 2);
        auto const dispatch_y = round_up_div(y, ANALYZE_VIS_BUFFER_WORKGROUP_Y * 2);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};
#endif