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
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_SAMPLED, daxa::Texture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_SAMPLED, daxa::Texture2DId<daxa_f32vec4>, color_image)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_WRITE_ONLY, daxa::RWTexture2DId<daxa_f32vec4>, swapchain)
DAXA_DECL_TASK_HEAD_END

struct WriteSwapchainPush
{
    WriteSwapchainH::AttachmentShaderBlob attachments;
    daxa_u32vec2 size;
};

#define WRITE_SWAPCHAIN_WG_X 16
#define WRITE_SWAPCHAIN_WG_Y 8

#if __cplusplus

#include "../../gpu_context.hpp"

inline MAKE_COMPUTE_COMPILE_INFO(write_swapchain_pipeline_compile_info2, "./src/rendering/tasks/write_swapchain.hlsl", "entry_write_swapchain")

struct WriteSwapchainTask : WriteSwapchainH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(write_swapchain_pipeline_compile_info2().name));
        u32 const dispatch_x = round_up_div(ti.device.image_info(ti.get(AT.swapchain).ids[0]).value().size.x, WRITE_SWAPCHAIN_WG_X);
        u32 const dispatch_y = round_up_div(ti.device.image_info(ti.get(AT.swapchain).ids[0]).value().size.y, WRITE_SWAPCHAIN_WG_Y);
        auto size = ti.device.image_info(ti.get(AT.swapchain).ids[0]).value().size;
        WriteSwapchainPush push{.size = { size.x, size.y } };
        push.attachments = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};
#endif