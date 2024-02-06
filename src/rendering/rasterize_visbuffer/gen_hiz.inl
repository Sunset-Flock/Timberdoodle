#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/asset.inl"

#define GEN_HIZ_X 16
#define GEN_HIZ_Y 16
#define GEN_HIZ_LEVELS_PER_DISPATCH 12
#define GEN_HIZ_WINDOW_X 64
#define GEN_HIZ_WINDOW_Y 64

DAXA_DECL_TASK_HEAD_BEGIN(GenHizTH, 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src)
DAXA_TH_IMAGE_ID_MIP_ARRAY(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, mips, GEN_HIZ_LEVELS_PER_DISPATCH)
DAXA_DECL_TASK_HEAD_END

struct GenHizPush
{
    DAXA_TH_BLOB(GenHizTH, uses)
    daxa_RWBufferPtr(daxa_u32) counter;
    daxa_u32 mip_count;
    daxa_u32 total_workgroup_count;
};

#if __cplusplus

#include <format>
#include "../../gpu_context.hpp"

inline daxa::ComputePipelineCompileInfo gen_hiz_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/gen_hiz.glsl"}},
        .push_constant_size = s_cast<u32>(sizeof(GenHizPush) + GenHizTH::attachment_shader_data_size()),
        .name = std::string{"GenHiz"},
    };
};

struct GenHizTask : GenHizTH
{
    GenHizTH::AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(GenHizTH{}.name()));
        auto const dispatch_x = round_up_div(context->settings.render_target_size.x, GEN_HIZ_WINDOW_X);
        auto const dispatch_y = round_up_div(context->settings.render_target_size.y, GEN_HIZ_WINDOW_Y);
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.push_constant(GenHizPush{
            .counter = ti.allocator->allocate_fill(0u).value().device_address,
            .mip_count = ti.get(GenHizTH::mips).view.slice.level_count,
            .total_workgroup_count = dispatch_x * dispatch_y,
        },
        GenHizTH::attachment_shader_data_size());

        ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y, .z = 1});
    }
};

daxa::TaskImageView task_gen_hiz_single_pass(GPUContext * context, daxa::TaskGraph & task_graph, daxa::TaskImageView src, daxa::TaskBufferView globals)
{
    daxa_u32vec2 const hiz_size =
        daxa_u32vec2(context->settings.render_target_size.x / 2, context->settings.render_target_size.y / 2);
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
    task_graph.add_task(GenHizTask{
        .views = std::array{
            daxa::attachment_view(GenHizTask::globals, globals),
            daxa::attachment_view(GenHizTask::src, src),
            daxa::attachment_view(GenHizTask::mips, hiz),
        },
        .context = context,
    });
    return hiz.view({.level_count = mip_count});
}

#endif