#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/raytracing.inl"
#include "../../shader_shared/ao.inl"

#define RT_AO_X 8
#define RT_AO_Y 8
#define RTAO_DENOISER_X 8
#define RTAO_DENOISER_Y 8

DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(RayTraceAmbientOcclusionH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtao_raw_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_detail_normals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_visbuffer)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_TLAS_ID(READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct RayTraceAmbientOcclusionPush
{
    daxa_BufferPtr(RayTraceAmbientOcclusionH::AttachmentShaderBlob) attach;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RTAODenoiserH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, normal_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtao_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, normals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtao_raw)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::RWTexture2DId<daxa_f32vec4>, rtao_image)
DAXA_DECL_TASK_HEAD_END

struct RTAODenoiserPush
{
    RTAODenoiserH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
    daxa_f32vec2 inv_size;
};

// auto constexpr size = sizeof(RTAODenoiserH::AttachmentShaderBlob);

#if defined(__cplusplus)

#include "../scene_renderer_context.hpp"

inline auto ray_trace_ao_rt_pipeline_info() -> daxa::RayTracingPipelineCompileInfo2
{
    return {
        .ray_gen_infos = {{.source = daxa::ShaderFile{"./src/rendering/rtao/rtao.hlsl"}, .entry_point = "ray_gen", .language = daxa::ShaderLanguage::SLANG}},
        .any_hit_infos = {{.source = daxa::ShaderFile{"./src/rendering/rtao/rtao.hlsl"}, .entry_point = "any_hit", .language = daxa::ShaderLanguage::SLANG}},
        .closest_hit_infos = {{.source = daxa::ShaderFile{"./src/rendering/rtao/rtao.hlsl"}, .entry_point = "closest_hit", .language = daxa::ShaderLanguage::SLANG}},
        .miss_hit_infos = {{.source = daxa::ShaderFile{"./src/rendering/rtao/rtao.hlsl"}, .entry_point = "miss", .language = daxa::ShaderLanguage::SLANG}},
        .shader_groups_infos = {
            // Gen Group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::GENERAL, .general_shader_index = 0},
            // Miss group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::GENERAL, .general_shader_index = 3},
            // Hit group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP, .closest_hit_shader_index = 2},
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP, .closest_hit_shader_index = 2, .any_hit_shader_index = 1},
        },
        .max_ray_recursion_depth = 1,
        .push_constant_size = sizeof(RayTraceAmbientOcclusionPush),
        .name = std::string{RayTraceAmbientOcclusionH::Info::NAME},
    };
}

struct RayTraceAmbientOcclusionTask : RayTraceAmbientOcclusionH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"RTAO", "TRACE">());
        RayTraceAmbientOcclusionPush push = {};
        push.attach = ti.allocator->allocate_fill(RayTraceAmbientOcclusionH::AttachmentShaderBlob{ti.attachment_shader_blob}).value().device_address;
        auto const & rtao_raw_image = ti.info(AT.rtao_raw_image).value();
        auto const & rt_pipeline = gpu_context->ray_tracing_pipelines.at(ray_trace_ao_rt_pipeline_info().name);
        ti.recorder.set_pipeline(*rt_pipeline.pipeline);
        ti.recorder.push_constant(push);
        ti.recorder.trace_rays({
            .width = rtao_raw_image.size.x,
            .height = rtao_raw_image.size.y,
            .depth = 1,
            .shader_binding_table = rt_pipeline.sbt,
        });
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"RTAO", "TRACE">());
    }
};

MAKE_COMPUTE_COMPILE_INFO(rtao_denoiser_pipeline_info, "./src/rendering/rtao/rtao.hlsl", "entry_rtao_denoiser")

struct RTAODeoinserTask : RTAODenoiserH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"RTAO", "DENOISE">());
        auto info = ti.info(AT.rtao_raw).value();
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(rtao_denoiser_pipeline_info().name));
        ti.recorder.push_constant(RTAODenoiserPush{
            .attach = ti.attachment_shader_blob,
            .size = {info.size.x, info.size.y},
            .inv_size = {1.0f / float(info.size.x), 1.0f / float(info.size.y)},
        });

        ti.recorder.dispatch({
            round_up_div(info.size.x, RTAO_DENOISER_X),
            round_up_div(info.size.y, RTAO_DENOISER_Y),
        });
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"RTAO", "DENOISE">());
    }
};

#endif