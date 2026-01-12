#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(WriteSwapchainH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, selected_mark_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, color_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, swapchain)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(WriteSwapchainDebugH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, depth_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, swapchain)
DAXA_DECL_TASK_HEAD_END

struct WriteSwapchainPush
{
    WriteSwapchainH::AttachmentShaderBlob attachments;
    daxa_u32vec2 size;
};

struct WriteSwapchainDebugPush
{
    WriteSwapchainDebugH::AttachmentShaderBlob attachments;
    daxa_u32vec2 size;
};

#define WRITE_SWAPCHAIN_WG_X 16
#define WRITE_SWAPCHAIN_WG_Y 8

#if __cplusplus

#include "../../gpu_context.hpp"

MAKE_COMPUTE_COMPILE_INFO(write_swapchain_pipeline_compile_info2, "./src/rendering/tasks/write_swapchain.hlsl", "entry_write_swapchain")
MAKE_COMPUTE_COMPILE_INFO(write_swapchain_debug_pipeline_compile_info2, "./src/rendering/tasks/write_swapchain.hlsl", "entry_write_swapchain_debug")

struct WriteSwapchainTask : WriteSwapchainH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(write_swapchain_pipeline_compile_info2().name));
        u32 const dispatch_x = round_up_div(ti.info(AT.swapchain).value().size.x, WRITE_SWAPCHAIN_WG_X);
        u32 const dispatch_y = round_up_div(ti.info(AT.swapchain).value().size.y, WRITE_SWAPCHAIN_WG_Y);
        auto size = ti.info(AT.swapchain).value().size;
        WriteSwapchainPush push{.size = { size.x, size.y } };
        push.attachments = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};

inline void write_swapchain_debug_callback(daxa::TaskInterface ti, RenderContext* render_context)
{
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(write_swapchain_debug_pipeline_compile_info2().name));
    auto size = ti.info(WriteSwapchainDebugH::Info::AT.swapchain).value().size;
    ti.recorder.push_constant(WriteSwapchainDebugPush{
        .attachments = ti.attachment_shader_blob,
        .size = { size.x, size.y },
    });
    u32 const dispatch_x = round_up_div(size.x, WRITE_SWAPCHAIN_WG_X);
    u32 const dispatch_y = round_up_div(size.y, WRITE_SWAPCHAIN_WG_Y);
    ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
}
#endif