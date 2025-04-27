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
#include "../../shader_shared/geometry_pipeline.inl"

#define REF_PT_X 8
#define REF_PT_Y 8

DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(ReferencePathTraceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(READ_WRITE_CONCURRENT, REGULAR_2D, debug_lens_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, pt_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_f32vec4>, history_image)
DAXA_TH_IMAGE_TYPED(READ, daxa::RWTexture2DId<daxa_u32>, vis_image)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, sky)
DAXA_TH_IMAGE_ID(SAMPLED, CUBE, sky_ibl)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, brdf_lut)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32), exposure)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_TLAS_ID(READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct ReferencePathTraceAttachments
{
    ReferencePathTraceH::AttachmentShaderBlob attachments;
};
DAXA_DECL_BUFFER_PTR(ReferencePathTraceAttachments)
struct ReferencePathTracePush
{
    daxa_BufferPtr(ReferencePathTraceAttachments) attachments;
};

#if defined(__cplusplus)

#include "../scene_renderer_context.hpp"

inline auto reference_path_trace_rt_pipeline_info() -> daxa::RayTracingPipelineCompileInfo
{
    return {
        .ray_gen_infos = {
            {
                .source = daxa::ShaderFile{"./src/rendering/path_trace/path_trace.hlsl"},
                .compile_options = {.entry_point = "ray_gen", .language = daxa::ShaderLanguage::SLANG},
            },
        },
        .any_hit_infos = {
            {
                .source = daxa::ShaderFile{"./src/rendering/path_trace/path_trace.hlsl"},
                .compile_options = {.entry_point = "any_hit", .language = daxa::ShaderLanguage::SLANG},
            },
        },
        .closest_hit_infos = {
            {
                .source = daxa::ShaderFile{"./src/rendering/path_trace/path_trace.hlsl"},
                .compile_options = {.entry_point = "closest_hit", .language = daxa::ShaderLanguage::SLANG},
            },
        },
        .miss_hit_infos = {
            {
                .source = daxa::ShaderFile{"./src/rendering/path_trace/path_trace.hlsl"},
                .compile_options = {.entry_point = "miss", .language = daxa::ShaderLanguage::SLANG},
            },
            {
                .source = daxa::ShaderFile{"./src/rendering/path_trace/path_trace.hlsl"},
                .compile_options = {.entry_point = "shadow_miss", .language = daxa::ShaderLanguage::SLANG},
            },
        },
        .shader_groups_infos = {
            // Gen Group
            daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::GENERAL,
                .general_shader_index = 0,
            },
            // Miss group
            daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::GENERAL,
                .general_shader_index = 3,
            },
            daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::GENERAL,
                .general_shader_index = 4,
            },
            // Hit group
            daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP,
                .closest_hit_shader_index = 2,
            },
            daxa::RayTracingShaderGroupInfo{
                .type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP,
                .closest_hit_shader_index = 2,
                .any_hit_shader_index = 1,
            },
        },
        .max_ray_recursion_depth = 1,
        .push_constant_size = sizeof(ReferencePathTracePush),
        .name = std::string{ReferencePathTraceH::NAME},
    };
}

struct ReferencePathTraceTask : ReferencePathTraceH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        // render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::RAY_TRACED_AMBIENT_OCCLUSION);
        if (ti.id(AT.tlas) != gpu_context->dummy_tlas_id)
        {
            ReferencePathTracePush push = {};
            auto alloc = ti.allocator->allocate(sizeof(ReferencePathTraceAttachments));
            std::memcpy(alloc->host_address, ti.attachment_shader_blob.data(), sizeof(ReferencePathTraceH::AttachmentShaderBlob));
            push.attachments = alloc->device_address;
            auto const & pt_image = ti.info(AT.pt_image).value();
            auto const & rt_pipeline = gpu_context->ray_tracing_pipelines.at(reference_path_trace_rt_pipeline_info().name);
            ti.recorder.set_pipeline(*rt_pipeline.pipeline);
            ti.recorder.push_constant(push);
            ti.recorder.trace_rays({
                .width = pt_image.size.x,
                .height = pt_image.size.y,
                .depth = 1,
                .shader_binding_table = rt_pipeline.sbt,
            });
        }
        // render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::RAY_TRACED_AMBIENT_OCCLUSION);
    }
};

#endif