#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/asset.inl"

#define GEN_HIZ_X 16
#define GEN_HIZ_Y 16
#define GEN_HIZ_LEVELS_PER_DISPATCH 12
#define GEN_HIZ_WINDOW_X 64
#define GEN_HIZ_WINDOW_Y 64

struct GenHizPush
{
    daxa_ImageViewId src;
    daxa_ImageViewId mips[GEN_HIZ_LEVELS_PER_DISPATCH];
    daxa_u32 mip_count;
    daxa_u64 counter_address;
    daxa_u32 total_workgroup_count;
};

#if __cplusplus

#include <format>
#include "../gpu_context.hpp"

inline static const daxa::ComputePipelineCompileInfo GEN_HIZ_PIPELINE_COMPILE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/gen_hiz.glsl"}},
    .push_constant_size = sizeof(GenHizPush),
    .name = std::string{"GenHiz"},
};

daxa::TaskImageView task_gen_hiz_single_pass(GPUContext * context, daxa::TaskGraph & task_graph, daxa::TaskImageView src)
{
    const daxa_u32vec2 hiz_size = daxa_u32vec2(context->settings.render_target_size.x / 2, context->settings.render_target_size.y / 2);
    const daxa_u32 mip_count = static_cast<daxa_u32>(std::ceil(std::log2(std::max(hiz_size.x, hiz_size.y))));
    daxa::TaskImageView hiz = task_graph.create_transient_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = { hiz_size.x, hiz_size.y, 1 },
        .mip_level_count = mip_count,
        .array_layer_count = 1,
        .sample_count = 1,
        .name = "hiz",
    });
    using namespace daxa::task_resource_uses;
    const daxa_u32 mips_this_dispatch = std::min(mip_count, u32(GEN_HIZ_LEVELS_PER_DISPATCH)) - 1;
    std::vector<daxa::GenericTaskResourceUse> uses = {};
    daxa::TaskImageView src_view = src.view({.base_mip_level = 0});
    uses.push_back(ImageComputeShaderSampled<>{ src_view });
    daxa::TaskImageView dst_views[GEN_HIZ_LEVELS_PER_DISPATCH] = { };
    for (daxa_u32 i = 0; i < mips_this_dispatch; ++i)
    {
        dst_views[i] = hiz.view({.base_mip_level = i});
        uses.push_back(ImageComputeShaderStorageWriteOnly<>{ dst_views[i] });                                 
    }
    task_graph.add_task({
        .uses = uses,
        .task = [=](daxa::TaskInterface ti)
        {
            auto & cmd = ti.get_recorder();
            auto & device = ti.get_device();
            cmd.set_uniform_buffer(context->shader_globals_set_info);
            cmd.set_pipeline(*context->compute_pipelines.at(GEN_HIZ_PIPELINE_COMPILE_INFO.name));
            auto const dispatch_x = round_up_div(context->settings.render_target_size.x, GEN_HIZ_WINDOW_X);
            auto const dispatch_y = round_up_div(context->settings.render_target_size.y, GEN_HIZ_WINDOW_Y);
            auto counter_alloc = ti.get_allocator().allocate(sizeof(u32), sizeof(u32)).value();
            *reinterpret_cast<u32*>(counter_alloc.host_address) = 0;
            GenHizPush push{ 
                .src = ti.uses[src_view].view(),
                .mips = {},
                .mip_count = mips_this_dispatch,
                .counter_address = counter_alloc.device_address,
                .total_workgroup_count = dispatch_x * dispatch_y,
            };
            for (u32 i = 0; i < mips_this_dispatch; ++i)
            {
                push.mips[i] = ti.uses[dst_views[i]].view();
            }
            cmd.push_constant(push);
            cmd.dispatch(dispatch_x, dispatch_y, 1);
        },
        .name = "gen hiz level single pass",
    });
    return hiz.view({.level_count = mip_count});
}

#endif