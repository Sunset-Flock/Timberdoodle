#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry_pipeline.inl"

#define ANALYZE_VIS_BUFFER_WORKGROUP_X 8
#define ANALYZE_VIS_BUFFER_WORKGROUP_Y 8

DAXA_DECL_TASK_HEAD_BEGIN(AnalyzeVisbuffer2H)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT,    daxa_BufferPtr(RenderGlobalData),           globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY,          REGULAR_2D,                                 visbuffer)                   // MUST BE STORAGE READ BECAUSE OF 64BIT VISBUFFER. WE NEED GENERIC ATTACHMENT ACCESS
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ,                     daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ,                     daxa_BufferPtr(MeshInstancesBufferHead),    mesh_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE,               daxa_RWBufferPtr(daxa_u32),                 meshlet_visibility_bitfield)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE,               daxa_RWBufferPtr(VisibleMeshletList),       visible_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE,               daxa_RWBufferPtr(daxa_u32),                 mesh_visibility_bitfield)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE,         REGULAR_2D,                                 debug_image)
DAXA_DECL_TASK_HEAD_END

struct AnalyzeVisbufferPush2
{
    AnalyzeVisbuffer2H::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
    daxa_f32vec2 inv_size;
};

#if defined(__cplusplus)

#include "../scene_renderer_context.hpp"

inline daxa::ComputePipelineCompileInfo analyze_visbufer_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/analyze_visbuffer.hlsl"},
            .compile_options = {
                .language = daxa::ShaderLanguage::SLANG,
                .create_flags = daxa::ShaderCreateFlagBits::REQUIRE_FULL_SUBGROUPS,
            },
        },
        .push_constant_size = s_cast<u32>(sizeof(AnalyzeVisbufferPush2)),
        .name = std::string{AnalyzeVisbuffer2H::NAME},
    };
};
struct AnalyzeVisBufferTask2 : AnalyzeVisbuffer2H::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(analyze_visbufer_pipeline_compile_info().name));
        auto [x, y, z] = ti.device.image_info(ti.get(AnalyzeVisbuffer2H::AT.visbuffer).ids[0]).value().size;
        ti.recorder.push_constant(AnalyzeVisbufferPush2{
            .attach = ti.attachment_shader_blob,
            .size = {x, y},
            .inv_size = {1.0f / float(x), 1.0f / float(y)},
        });
        auto const dispatch_x = round_up_div(x, ANALYZE_VIS_BUFFER_WORKGROUP_X * 2);
        auto const dispatch_y = round_up_div(y, ANALYZE_VIS_BUFFER_WORKGROUP_Y * 2);
        
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_ANALYZE);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_ANALYZE);
    }
};
#endif