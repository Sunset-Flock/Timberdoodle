#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/globals.inl"

#define TRANSMITTANCE_X_DISPATCH 8
#define TRANSMITTANCE_Y_DISPATCH 8

#define SKY_X_DISPATCH 8
#define SKY_Y_DISPATCH 8

#define IBL_CUBE_RES 32
#define IBL_CUBE_X_WG_SIZE 2
#define IBL_CUBE_Y_WG_SIZE 2

DAXA_DECL_TASK_HEAD_BEGIN(ComputeTransmittanceH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, transmittance)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(ComputeMultiscatteringH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, multiscattering)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(ComputeSkyH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, multiscattering)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, sky)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(SkyIntoCubemapH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, ibl_cube)
DAXA_DECL_TASK_HEAD_END

struct ComputeMultiscatteringPush
{
    DAXA_TH_BLOB(ComputeMultiscatteringH, uses)
    daxa_SamplerId sampler_id;
};

struct ComputeSkyPush
{
    DAXA_TH_BLOB(ComputeSkyH, uses)
    daxa_SamplerId sampler_id;
};

#if __cplusplus
#include "../scene_renderer_context.hpp"

inline static constexpr char const SKY_SHADER_PATH[] = "./src/rendering/tasks/sky.glsl";

inline daxa::ComputePipelineCompileInfo compute_transmittance_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{SKY_SHADER_PATH},
            .compile_options = {.defines = {{"TRANSMITTANCE", "1"}}},
        },
        .push_constant_size = sizeof(ComputeTransmittanceH::AttachmentShaderBlob),
        .name = std::string{ComputeTransmittanceH::NAME}};
}
inline daxa::ComputePipelineCompileInfo compute_multiscattering_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{SKY_SHADER_PATH},
            .compile_options = {.defines = {{"MULTISCATTERING", "1"}}},
        },
        .push_constant_size = static_cast<u32>(sizeof(ComputeMultiscatteringPush)),
        .name = std::string{ComputeMultiscatteringH::NAME},
    };
}
inline daxa::ComputePipelineCompileInfo compute_sky_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{SKY_SHADER_PATH},
            .compile_options = {.defines = {{"SKY", "1"}}},
        },
        .push_constant_size = static_cast<u32>(sizeof(ComputeSkyPush)),
        .name = std::string{ComputeSkyH::NAME},
    };
}
inline daxa::ComputePipelineCompileInfo sky_into_cubemap_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{SKY_SHADER_PATH},
            .compile_options = {.defines = {{"CUBEMAP", "1"}}},
        },
        .push_constant_size = static_cast<u32>(sizeof(SkyIntoCubemapH::AttachmentShaderBlob)),
        .name = std::string{SkyIntoCubemapH::NAME},
    };
}

struct ComputeTransmittanceTask : ComputeTransmittanceH::Task
{
    AttachmentViews views = {};
    GPUContext * context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const transmittance_size = ti.info(AT.transmittance).value().size;
        auto const dispatch_size = u32vec2{
            (transmittance_size.x + TRANSMITTANCE_X_DISPATCH - 1) / TRANSMITTANCE_X_DISPATCH,
            (transmittance_size.y + TRANSMITTANCE_Y_DISPATCH - 1) / TRANSMITTANCE_Y_DISPATCH,
        };
        ti.recorder.set_pipeline(*context->compute_pipelines.at(compute_transmittance_pipeline_compile_info().name));
        ComputeTransmittanceH::AttachmentShaderBlob push{};
        assign_blob(push, ti.attachment_shader_blob);
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
        auto const multiscattering_size = render_context->gpuctx->device.image_info(ti.get(AT.multiscattering).ids[0]).value().size;
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(compute_multiscattering_pipeline_compile_info().name));
        ComputeMultiscatteringPush push{.sampler_id = render_context->render_data.samplers.linear_clamp};
        assign_blob(push.uses, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = multiscattering_size.x, .y = multiscattering_size.y});
    }
};

struct ComputeSkyTask : ComputeSkyH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const sky_size = render_context->gpuctx->device.image_info(ti.get(AT.sky).ids[0]).value().size;
        auto const dispatch_size = u32vec2{
            (sky_size.x + SKY_X_DISPATCH - 1) / SKY_X_DISPATCH,
            (sky_size.y + SKY_Y_DISPATCH - 1) / SKY_Y_DISPATCH,
        };
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(compute_sky_pipeline_compile_info().name));
        ComputeSkyPush push{.sampler_id = render_context->render_data.samplers.linear_clamp};
        assign_blob(push.uses, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
    }
};

struct SkyIntoCubemapTask : SkyIntoCubemapH::Task
{
    AttachmentViews views = {};
    GPUContext * context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(sky_into_cubemap_pipeline_compile_info().name));
        SkyIntoCubemapH::AttachmentShaderBlob push{};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({
            (IBL_CUBE_RES + IBL_CUBE_X_WG_SIZE - 1) / IBL_CUBE_X_WG_SIZE,
            (IBL_CUBE_RES + IBL_CUBE_Y_WG_SIZE - 1) / IBL_CUBE_Y_WG_SIZE,
            6
        });
    }
};
#endif //_cplusplus