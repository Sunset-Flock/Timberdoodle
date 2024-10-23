#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"

#define GEN_HIZ_X 16
#define GEN_HIZ_Y 16
#define GEN_HIZ_LEVELS_PER_DISPATCH 12
#define GEN_HIZ_WINDOW_X 64
#define GEN_HIZ_WINDOW_Y 64

DAXA_DECL_TASK_HEAD_BEGIN(GenHizTH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, src)
DAXA_TH_IMAGE_ID_MIP_ARRAY(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, mips, GEN_HIZ_LEVELS_PER_DISPATCH)
DAXA_DECL_TASK_HEAD_END

struct GenHizPush
{
    DAXA_TH_BLOB(GenHizTH, attach)
    daxa_RWBufferPtr(daxa_u32) counter;
    daxa_u32 mip_count;
    daxa_u32 total_workgroup_count;
};

#if DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL

// The hiz generation only works for power of two images.
// The hiz mip chain is always power of two sized.
// the hiz mip chain is sized to the next smaller power of two relative to the src depth image.
// An example size: 1440x2560p src depth image has an hiz with a size of 1024x2048 at mip 0.

// The shader code will only ever see power of two hiz information, all calculations are done in power of two hiz space.
// In order to properly generate and sample the src image, gather operations are used.
// Rescaling the image implicitly in the calculation for mip0 of the hiz.

// The hiz gen works in a single dispatch in order to avoid costly pipeline barriers.
// In order to achieve this all workgroups incement an atomic counter when they are done.
// InterlockedAdd/atomicAdd's return the previous value before the add.
// The last workgroup to finish can use this counter to know it is the last one `(bool last_to_finish = (atomicAdd(counter, 1) == (total_workgroups -1))`

// Each workgroup downsamples a mip tile. A tile is a 64x64 section of the original image.
// NOTE: The src image will be treated as if it was sized to the next power of two in size. A 1440x2560p image will be treated as a 2048x4098 image!
// In total a 64x64 sample area is tapped in the first level. This may differ for non power of two depth sizes, these will be oversampled.

DAXA_DECL_TASK_HEAD_BEGIN(GenHizH2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_SAMPLED, daxa::Texture2DId<float>, src)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(COMPUTE_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DId<float>, hiz, GEN_HIZ_LEVELS_PER_DISPATCH)
DAXA_DECL_TASK_HEAD_END

struct GenHizData
{
    GenHizH2::AttachmentShaderBlob attach;
    daxa_RWBufferPtr(daxa_u32) workgroup_finish_counter;
    daxa_u32 total_workgroup_count;
    daxa_u32 mip_count;
    daxa_u32vec2 dst_mip0_size;
};

struct GenHizPush2
{
    daxa_BufferPtr(GenHizData) data;
};

#endif // #if DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL

#if defined(__cplusplus)

// HLSL version is buggy:
#define HIZ2 1

#include "../scene_renderer_context.hpp"

inline auto gen_hiz_pipeline_compile_info()
{
    return daxa::ComputePipelineCompileInfo{ .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/gen_hiz.glsl"}}, .push_constant_size = s_cast<u32>(sizeof(GenHizPush)), .name = std::string{"GenHiz"} };
};

inline auto gen_hiz_pipeline_compile_info2()
{
    return daxa::ComputePipelineCompileInfo2{ .source = daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/gen_hiz.hlsl"}, .entry_point = "entry_gen_hiz", .push_constant_size = sizeof(GenHizPush2), .name = "GenHiz2" };
}

struct TaskGenHizSinglePassInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & tg;
    daxa::TaskImageView src = {};
    daxa::TaskBufferView globals = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView * hiz = {};
};
void task_gen_hiz_single_pass(TaskGenHizSinglePassInfo const & info)
{
    daxa_u32vec2 const hiz_size = {
        info.render_context->render_data.settings.next_lower_po2_render_target_size.x,
        info.render_context->render_data.settings.next_lower_po2_render_target_size.y,
    };
    u32 mip_count = static_cast<u32>(std::ceil(std::log2(std::max(hiz_size.x, hiz_size.y))));
    mip_count = std::min(mip_count, u32(GEN_HIZ_LEVELS_PER_DISPATCH));
    *info.hiz = info.tg.create_transient_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = {hiz_size.x, hiz_size.y, 1},
        .mip_level_count = mip_count,
        .name = "hiz",
    });
    #if defined(HIZ2)
    info.tg.add_task(daxa::InlineTaskWithHead<GenHizH2::Task>{
        .views = std::array{
            daxa::attachment_view(GenHizH2::AT.globals, info.globals),
            daxa::attachment_view(GenHizH2::AT.debug_image, info.debug_image),
            daxa::attachment_view(GenHizH2::AT.src, info.src),
            daxa::attachment_view(GenHizH2::AT.hiz, *info.hiz),
        },
        .task = [=, render_context = info.render_context](daxa::TaskInterface ti)
        {
            auto const& AT = GenHizH2::AT;
            ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(gen_hiz_pipeline_compile_info2().name));
            daxa_u32vec2 next_higher_po2_render_target_size = {
                render_context->render_data.settings.next_lower_po2_render_target_size.x,
                render_context->render_data.settings.next_lower_po2_render_target_size.y,
            };
            auto const dispatch_x = round_up_div(next_higher_po2_render_target_size.x * 2, GEN_HIZ_WINDOW_X);
            auto const dispatch_y = round_up_div(next_higher_po2_render_target_size.y * 2, GEN_HIZ_WINDOW_Y);
            GenHizData data = {
                .attach = ti.attachment_shader_blob,
                .workgroup_finish_counter = ti.allocator->allocate_fill(0u).value().device_address,
                .total_workgroup_count = dispatch_x * dispatch_y,
                .mip_count = ti.get(AT.hiz).view.slice.level_count,
                .dst_mip0_size = daxa_u32vec2{ ti.info(AT.hiz).value().size.x, ti.info(AT.hiz).value().size.y },
            };
            auto data_alloc = ti.allocator->allocate_fill(data, 8).value();
            ti.recorder.push_constant(data_alloc.device_address);
            
            render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_GEN_HIZ);
            ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y });
            render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_GEN_HIZ);
        },
    });
    #else
    info.tg.add_task(daxa::InlineTaskWithHead<GenHizTH::Task>{
        .views = std::array{
            daxa::attachment_view(GenHizTH::AT.globals, info.globals),
            daxa::attachment_view(GenHizTH::AT.src, info.src),
            daxa::attachment_view(GenHizTH::AT.mips, *info.hiz),
        },
        .task = [=, render_context = info.render_context](daxa::TaskInterface ti)
        {
            auto const& AT = GenHizTH::AT;
            ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(gen_hiz_pipeline_compile_info().name));
            daxa_u32vec2 next_higher_po2_render_target_size = {
                render_context->render_data.settings.next_lower_po2_render_target_size.x,
                render_context->render_data.settings.next_lower_po2_render_target_size.y,
            };
            auto const dispatch_x = round_up_div(next_higher_po2_render_target_size.x * 2, GEN_HIZ_WINDOW_X);
            auto const dispatch_y = round_up_div(next_higher_po2_render_target_size.y * 2, GEN_HIZ_WINDOW_Y);
            GenHizPush push = {
                .attach = ti.attachment_shader_blob,
                .counter = ti.allocator->allocate_fill(0u).value().device_address,
                .mip_count = ti.get(AT.mips).view.slice.level_count,
                .total_workgroup_count = dispatch_x * dispatch_y,
            };
            ti.recorder.push_constant(push);
            ti.recorder.dispatch({.x = dispatch_x, .y = dispatch_y });
        },
    });
    #endif
}

#endif