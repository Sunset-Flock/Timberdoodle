#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"

#define GEN_HIZ_X 16
#define GEN_HIZ_Y 16
#define GEN_HIZ_LEVELS_PER_DISPATCH 12
#define GEN_HIZ_WINDOW_X 64
#define GEN_HIZ_WINDOW_Y 64

DAXA_DECL_TASK_HEAD_BEGIN(GenHizTH, 2)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src)
DAXA_TH_IMAGE_ID_MIP_ARRAY(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, mips, GEN_HIZ_LEVELS_PER_DISPATCH)
DAXA_DECL_TASK_HEAD_END

struct GenHizPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DAXA_TH_BLOB(GenHizTH) uses;
    daxa_RWBufferPtr(daxa_u32) counter;
    daxa_u32 mip_count;
    daxa_u32 total_workgroup_count;
};

#if __cplusplus

#include <format>
#include "../../gpu_context.hpp"

inline static const daxa::ComputePipelineCompileInfo GEN_HIZ_PIPELINE_COMPILE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/gen_hiz.glsl"}},
    .push_constant_size = sizeof(GenHizPush),
    .name = std::string{"GenHiz"},
};

daxa::TaskImageView
task_gen_hiz_single_pass(GPUContext *context, daxa::TaskGraph &task_graph, daxa::TaskImageView src)
{
    const daxa_u32vec2 hiz_size = daxa_u32vec2(context->settings.render_target_size.x / 2, context->settings.render_target_size.y / 2);
    daxa_u32 mip_count = static_cast<daxa_u32>(std::ceil(std::log2(std::max(hiz_size.x, hiz_size.y))));
    mip_count = std::min(mip_count, u32(GEN_HIZ_LEVELS_PER_DISPATCH)) - 1;
    daxa::TaskImageView hiz = task_graph.create_transient_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = {hiz_size.x, hiz_size.y, 1},
        .mip_level_count = mip_count,
        .array_layer_count = 1,
        .sample_count = 1,
        .name = "hiz",
    });
    auto generic_uses = daxa::generic_uses_cast(GenHizTH::Uses{.src = src, .mips = hiz});
    auto tester = daxa::TaskImageUse<>::from(generic_uses[1]);
    task_graph.add_task(daxa::InlineTaskInfo{
        .uses = daxa::generic_uses_cast(GenHizTH::Uses{.src = src, .mips = hiz}),
        .task = [=](daxa::TaskInterface ti)
        {
            auto &recorder = ti.get_recorder();
            auto &device = ti.get_device();
            auto &cmd = ti.get_recorder();
            cmd.set_pipeline(*context->compute_pipelines.at(GenHizTH::NAME));
            auto const dispatch_x = round_up_div(context->settings.render_target_size.x, GEN_HIZ_WINDOW_X);
            auto const dispatch_y = round_up_div(context->settings.render_target_size.y, GEN_HIZ_WINDOW_Y);
            GenHizPush push{
                .globals = context->shader_globals_address,
                .counter = ti.get_allocator().allocate_fill(0u).value().device_address,
                .mip_count = mip_count,
                .total_workgroup_count = dispatch_x * dispatch_y,
            };
            ti.copy_task_head_to(&push.uses);
            cmd.push_constant(push);
            cmd.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
        },
        .name = std::string(GenHizTH::NAME),
    });
    return hiz.view({.level_count = mip_count});
}

#endif