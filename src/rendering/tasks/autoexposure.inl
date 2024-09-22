#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/globals.inl"

DAXA_DECL_TASK_HEAD_BEGIN(GenLuminanceHistogramH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(daxa_u32), histogram)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32), luminance_average)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY, REGULAR_2D, color_image)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(GenLuminanceAverageH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), histogram)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_BufferPtr(daxa_f32), luminance_average)
DAXA_DECL_TASK_HEAD_END

#define LUM_HISTOGRAM_BIN_COUNT 256
#define SUM_HISTOGRAM_INDEX 256
#define COMPUTE_HISTOGRAM_WG_X 32
#define COMPUTE_HISTOGRAM_WG_Y 32

#if __cplusplus
#include "../../gpu_context.hpp"

inline daxa::ComputePipelineCompileInfo gen_luminace_histogram_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/tasks/autoexposure.glsl"},
            .compile_options = {.defines = {{"GEN_HISTOGRAM", "1"}}},
        },
        .push_constant_size = sizeof(GenLuminanceHistogramH::AttachmentShaderBlob),
        .name = std::string{GenLuminanceHistogramH::NAME},
    };
};

inline daxa::ComputePipelineCompileInfo gen_luminace_average_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/tasks/autoexposure.glsl"},
            .compile_options = {.defines = {{"GEN_AVERAGE", "1"}}},
        },
        .push_constant_size = sizeof(GenLuminanceAverageH::AttachmentShaderBlob),
        .name = std::string{GenLuminanceAverageH::NAME},
    };
};

struct GenLuminanceHistogramTask : GenLuminanceHistogramH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const offscreen_resolution = render_context->render_data.settings.render_target_size;
        auto const dispatch_size = u32vec2{
            (offscreen_resolution.x + COMPUTE_HISTOGRAM_WG_X - 1) / COMPUTE_HISTOGRAM_WG_X,
            (offscreen_resolution.y + COMPUTE_HISTOGRAM_WG_Y - 1) / COMPUTE_HISTOGRAM_WG_Y,
        };
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(gen_luminace_histogram_pipeline_compile_info().name));
        GenLuminanceHistogramH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
    }
};

struct GenLuminanceAverageTask : GenLuminanceAverageH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(gen_luminace_average_pipeline_compile_info().name));
        GenLuminanceAverageH::AttachmentShaderBlob push = ti.attachment_shader_blob;
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = 1});
    }
};
#endif