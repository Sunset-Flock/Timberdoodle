#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_TASK_HEAD_BEGIN(WriteSwapchainH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY, REGULAR_2D, color_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, swapchain)
DAXA_DECL_TASK_HEAD_END

struct WriteSwapchainPush
{
    daxa_u32vec2 size;
    DAXA_TH_BLOB(WriteSwapchainH, attachments)
};

#define WRITE_SWAPCHAIN_WG_X 16
#define WRITE_SWAPCHAIN_WG_Y 8

#if __cplusplus

#include "../../gpu_context.hpp"

inline daxa::ComputePipelineCompileInfo write_swapchain_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/write_swapchain.glsl"}},
        .push_constant_size = s_cast<u32>(sizeof(WriteSwapchainPush)),
        .name = std::string{WriteSwapchainH::NAME},
    };
};
struct WriteSwapchainTask : WriteSwapchainH::Task
{
    AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(write_swapchain_pipeline_compile_info().name));
        u32 const dispatch_x = round_up_div(ti.device.info_image(ti.get(AT.swapchain).ids[0]).value().size.x, WRITE_SWAPCHAIN_WG_X);
        u32 const dispatch_y = round_up_div(ti.device.info_image(ti.get(AT.swapchain).ids[0]).value().size.y, WRITE_SWAPCHAIN_WG_Y);
        auto size = ti.device.info_image(ti.get(AT.swapchain).ids[0]).value().size;
        WriteSwapchainPush push{.size = { size.x, size.y } };
        assign_blob(push.attachments, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};
#endif