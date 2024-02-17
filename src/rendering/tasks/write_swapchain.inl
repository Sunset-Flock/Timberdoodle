#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_TASK_HEAD_BEGIN(WriteSwapchain, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY, REGULAR_2D, color_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, swapchain)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32), luminance_average)
DAXA_DECL_TASK_HEAD_END

struct WriteSwapchainPush
{
    daxa_u32vec2 size;
    DAXA_TH_BLOB(WriteSwapchain, attachments)
};

#define WRITE_SWAPCHAIN_WG_X 16
#define WRITE_SWAPCHAIN_WG_Y 8

#if __cplusplus

#include "../../gpu_context.hpp"

inline daxa::ComputePipelineCompileInfo write_swapchain_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/write_swapchain.glsl"}},
        .push_constant_size = s_cast<u32>(sizeof(WriteSwapchainPush) + WriteSwapchain::attachment_shader_data_size()),
        .name = std::string{WriteSwapchain{}.name()},
    };
};
struct WriteSwapchainTask : WriteSwapchain
{
    WriteSwapchain::AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(WriteSwapchain{}.name()));
        u32 const dispatch_x = round_up_div(ti.device.info_image(ti.get(swapchain).ids[0]).value().size.x, WRITE_SWAPCHAIN_WG_X);
        u32 const dispatch_y = round_up_div(ti.device.info_image(ti.get(swapchain).ids[0]).value().size.y, WRITE_SWAPCHAIN_WG_Y);
        auto size = ti.device.info_image(ti.get(swapchain).ids[0]).value().size;
        ti.recorder.push_constant(WriteSwapchainPush{
            .size = { size.x, size.y },
        });
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
            .offset = sizeof(WriteSwapchainPush),
        });
        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};
#endif