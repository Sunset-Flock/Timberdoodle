#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/globals.inl"

#define TRANSMITTANCE_X_DISPATCH 8
#define TRANSMITTANCE_Y_DISPATCH 8

#define SKY_X_DISPATCH 8
#define SKY_Y_DISPATCH 8

#define IBL_CUBE_RES 16


DAXA_DECL_TASK_HEAD_BEGIN(ComputeTransmittance, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, transmittance)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(ComputeMultiscattering, 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, multiscattering)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(ComputeSky, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, multiscattering)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D, sky)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(SkyIntoCubemap, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_WRITE_ONLY, REGULAR_2D_ARRAY, ibl_cube)
DAXA_DECL_TASK_HEAD_END

struct ComputeMultiscatteringPush
{
    DAXA_TH_BLOB(ComputeMultiscattering, uses)
    daxa_SamplerId sampler_id;
};

struct ComputeSkyPush
{
    DAXA_TH_BLOB(ComputeSky, uses)
    daxa_SamplerId sampler_id;
};

#if __cplusplus
#include "../../gpu_context.hpp"

inline static constexpr char const SKY_SHADER_PATH[] = "./src/rendering/tasks/sky.glsl";

inline daxa::ComputePipelineCompileInfo compute_transmittance_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{SKY_SHADER_PATH},
            .compile_options = {.defines = {{"TRANSMITTANCE", "1"}}},
        },
        .push_constant_size = ComputeTransmittance::attachment_shader_data_size(),
        .name = std::string{ComputeTransmittance{}.name()}};
}
inline daxa::ComputePipelineCompileInfo compute_multiscattering_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{SKY_SHADER_PATH},
            .compile_options = {.defines = {{"MULTISCATTERING", "1"}}},
        },
        .push_constant_size = static_cast<u32>(sizeof(ComputeMultiscatteringPush) + ComputeMultiscattering::attachment_shader_data_size()),
        .name = std::string{ComputeMultiscattering{}.name()},
    };
}
inline daxa::ComputePipelineCompileInfo compute_sky_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{SKY_SHADER_PATH},
            .compile_options = {.defines = {{"SKY", "1"}}},
        },
        .push_constant_size = static_cast<u32>(sizeof(ComputeSkyPush) + ComputeSky::attachment_shader_data_size()),
        .name = std::string{ComputeSky{}.name()},
    };
}
inline daxa::ComputePipelineCompileInfo sky_into_cubemap_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{SKY_SHADER_PATH},
            .compile_options = {.defines = {{"CUBEMAP", "1"}}},
        },
        .push_constant_size = static_cast<u32>(SkyIntoCubemap::attachment_shader_data_size()),
        .name = std::string{SkyIntoCubemap{}.name()},
    };
}

struct ComputeTransmittanceTask : ComputeTransmittance
{
    ComputeTransmittance::AttachmentViews views = {};
    GPUContext * context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const transmittance_size = context->device.info_image(ti.get(ComputeTransmittance::transmittance).ids[0]).value().size;
        auto const dispatch_size = u32vec2{
            (transmittance_size.x + TRANSMITTANCE_X_DISPATCH - 1) / TRANSMITTANCE_X_DISPATCH,
            (transmittance_size.y + TRANSMITTANCE_Y_DISPATCH - 1) / TRANSMITTANCE_Y_DISPATCH,
        };
        ti.recorder.set_pipeline(*context->compute_pipelines.at(ComputeTransmittance{}.name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
    }
};

struct ComputeMultiscatteringTask : ComputeMultiscattering
{
    ComputeMultiscattering::AttachmentViews views = {};
    GPUContext * context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const multiscattering_size = context->device.info_image(ti.get(ComputeMultiscattering::multiscattering).ids[0]).value().size;
        ti.recorder.set_pipeline(*context->compute_pipelines.at(ComputeMultiscattering{}.name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.push_constant(
            ComputeMultiscatteringPush{.sampler_id = context->shader_globals.samplers.linear_clamp},
            ComputeMultiscattering::attachment_shader_data_size());
        ti.recorder.dispatch({.x = multiscattering_size.x, .y = multiscattering_size.y});
    }
};

struct ComputeSkyTask : ComputeSky
{
    ComputeSky::AttachmentViews views = {};
    GPUContext * context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const sky_size = context->device.info_image(ti.get(ComputeSky::sky).ids[0]).value().size;
        auto const dispatch_size = u32vec2{
            (sky_size.x + SKY_X_DISPATCH - 1) / SKY_X_DISPATCH,
            (sky_size.y + SKY_Y_DISPATCH - 1) / SKY_Y_DISPATCH,
        };
        ti.recorder.set_pipeline(*context->compute_pipelines.at(ComputeSky{}.name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.push_constant(
            ComputeSkyPush{.sampler_id = context->shader_globals.samplers.linear_clamp},
            ComputeSky::attachment_shader_data_size());
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
    }
};

struct SkyIntoCubemapTask : SkyIntoCubemap
{
    SkyIntoCubemap::AttachmentViews views = {};
    GPUContext * context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(SkyIntoCubemap{}.name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.dispatch({(IBL_CUBE_RES + 7) / 8, (IBL_CUBE_RES + 7) / 8, 6});
    }
};
#endif //_cplusplus