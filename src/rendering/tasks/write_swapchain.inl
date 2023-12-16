#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_TASK_HEAD_BEGIN(WriteSwapchain, 5)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, swapchain)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY, REGULAR_2D, vis_image)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY, REGULAR_2D, debug_image)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_DECL_TASK_HEAD_END

struct WriteSwapchainPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DAXA_TH_BLOB(WriteSwapchain) uses;
    daxa_u32 width;
    daxa_u32 height;
};

#define WRITE_SWAPCHAIN_WG_X 16
#define WRITE_SWAPCHAIN_WG_Y 8

#if __cplusplus

#include "../../gpu_context.hpp"

struct WriteSwapchainTask : WriteSwapchain
{
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/write_swapchain.glsl"}},
        .push_constant_size = sizeof(WriteSwapchainPush),
        .name = std::string{WriteSwapchain::NAME},
    };
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_pipeline(*context->compute_pipelines.at(WriteSwapchain::NAME));
        u32 const dispatch_x = round_up_div(ti.get_device().info_image(uses.swapchain.image()).value().size.x, WRITE_SWAPCHAIN_WG_X);
        u32 const dispatch_y = round_up_div(ti.get_device().info_image(uses.swapchain.image()).value().size.y, WRITE_SWAPCHAIN_WG_Y);
        auto push = WriteSwapchainPush{
            .globals = context->shader_globals_address,
            .width = ti.get_device().info_image(uses.swapchain.image()).value().size.x,
            .height = ti.get_device().info_image(uses.swapchain.image()).value().size.y,
        };
        ti.copy_task_head_to(&push.uses);
        cmd.push_constant(push);
        cmd.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};
#endif