#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/globals.inl"

#define TRANSMITTANCE_X 8
#define TRANSMITTANCE_Y 8

#define MULTISCATTERING_X 1
#define MULTISCATTERING_Y 1
#define MULTISCATTERING_Z 64

#define SKY_X 8
#define SKY_Y 8

#define IBL_CUBE_RES 32
#define IBL_CUBE_X 2
#define IBL_CUBE_Y 2

#if (DAXA_LANGUAGE != DAXA_LANGUAGE_GLSL)
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ComputeTransmittanceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec3>, transmittance)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ComputeMultiscatteringH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec3>, transmittance)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec3>, multiscattering)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ComputeSkyH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec3>, transmittance)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec3>, multiscattering)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, sky)
DAXA_DECL_TASK_HEAD_END
#endif

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(SkyIntoCubemapH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, sky)
DAXA_TH_IMAGE_ID(READ_WRITE, REGULAR_2D_ARRAY, ibl_cube)
DAXA_DECL_TASK_HEAD_END

#if __cplusplus
#include "../scene_renderer_context.hpp"

inline static constexpr char const SKY_SHADER_PATH[] = "./src/rendering/tasks/sky.hlsl";

MAKE_COMPUTE_COMPILE_INFO(compute_transmittance_pipeline_compile_info, "./src/rendering/tasks/sky.hlsl", "compute_transmittance_lut")
MAKE_COMPUTE_COMPILE_INFO(compute_multiscattering_pipeline_compile_info, "./src/rendering/tasks/sky.hlsl", "compute_multiscattering_lut")
MAKE_COMPUTE_COMPILE_INFO(compute_sky_pipeline_compile_info, "./src/rendering/tasks/sky.hlsl", "compute_sky_lut")

inline daxa::ComputePipelineCompileInfo2 sky_into_cubemap_pipeline_compile_info()
{
    return {
        .source = daxa::ShaderFile{"./src/rendering/tasks/sky.glsl"},
        .defines = {{"CUBEMAP", "1"}},
        .required_subgroup_size = 32,
        .push_constant_size = static_cast<u32>(sizeof(SkyIntoCubemapH::AttachmentShaderBlob)),
        .name = std::string{SkyIntoCubemapH::Info::NAME},
    };
}

struct ComputeTransmittanceTask : ComputeTransmittanceH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const transmittance_size = ti.info(AT.transmittance).value().size;
        auto const dispatch_size = u32vec2{
            (transmittance_size.x + TRANSMITTANCE_X - 1) / TRANSMITTANCE_X,
            (transmittance_size.y + TRANSMITTANCE_Y - 1) / TRANSMITTANCE_Y,
        };
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(compute_transmittance_pipeline_compile_info().name));
        ComputeTransmittanceH::AttachmentShaderBlob push{};
        push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
    }
};

struct ComputeMultiscatteringTask : ComputeMultiscatteringH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const multiscattering_size = ti.info(AT.multiscattering).value().size;
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(compute_multiscattering_pipeline_compile_info().name));
        ComputeMultiscatteringH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = multiscattering_size.x, .y = multiscattering_size.y});
    }
};

void compute_sky_task(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = ComputeSkyH::Info::AT;
    auto const sky_size = ti.info(AT.sky).value().size;
    auto const dispatch_size = u32vec2{
        (sky_size.x + SKY_X - 1) / SKY_X,
        (sky_size.y + SKY_Y - 1) / SKY_Y,
    };
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(compute_sky_pipeline_compile_info().name));
    ComputeSkyH::AttachmentShaderBlob push = ti.attachment_shader_blob;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
}

void sky_into_cubemap_task(daxa::TaskInterface ti, GPUContext * gpu_context)
{
    ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(sky_into_cubemap_pipeline_compile_info().name));
    SkyIntoCubemapH::AttachmentShaderBlob push{};
    push = ti.attachment_shader_blob;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({
        (IBL_CUBE_RES + IBL_CUBE_X - 1) / IBL_CUBE_X,
        (IBL_CUBE_RES + IBL_CUBE_Y - 1) / IBL_CUBE_Y,
        6
    });
}
#endif //_cplusplus