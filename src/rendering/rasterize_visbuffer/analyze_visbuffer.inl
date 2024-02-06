#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/asset.inl"

#define ANALYZE_VIS_BUFFER_WORKGROUP_X 8
#define ANALYZE_VIS_BUFFER_WORKGROUP_Y 8

DAXA_DECL_TASK_HEAD_BEGIN(AnalyzeVisbuffer2, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, visbuffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), meshlet_visibility_bitfield)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VisibleMeshletList), visible_meshlets)
// DxDy Image
// UV Image
DAXA_DECL_TASK_HEAD_END

struct AnalyzeVisbufferPush2
{
    DAXA_TH_BLOB(AnalyzeVisbuffer2, uses)
    daxa_u32vec2 size;
};

#if __cplusplus

#include "../../gpu_context.hpp"

inline daxa::ComputePipelineCompileInfo analyze_visbufer_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/analyze_visbuffer.glsl"}},
        .push_constant_size = s_cast<u32>(sizeof(AnalyzeVisbufferPush2) + AnalyzeVisbuffer2::attachment_shader_data_size()),
        .name = std::string{AnalyzeVisbuffer2{}.name()}};
};
struct AnalyzeVisBufferTask2 : AnalyzeVisbuffer2
{
    AnalyzeVisbuffer2::AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(AnalyzeVisbuffer2{}.name()));
        auto [x, y, z] = ti.device.info_image(ti.get(AnalyzeVisbuffer2::visbuffer).ids[0]).value().size;
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.push_constant(
            AnalyzeVisbufferPush2{.size = {x, y}},
            AnalyzeVisbuffer2::attachment_shader_data_size());
        auto const dispatch_x = round_up_div(x, ANALYZE_VIS_BUFFER_WORKGROUP_X * 2);
        auto const dispatch_y = round_up_div(y, ANALYZE_VIS_BUFFER_WORKGROUP_Y * 2);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};
#endif