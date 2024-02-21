#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/globals.inl"

DAXA_DECL_TASK_HEAD_BEGIN(GenLuminanceHistogram, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(daxa_u32), histogram)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32), luminance_average)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_ONLY, REGULAR_2D, color_image)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(GenLuminanceAverage, 3)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
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
        .push_constant_size = GenLuminanceHistogram::attachment_shader_data_size(),
        .name = std::string{GenLuminanceHistogram{}.name()},
    };
};

inline daxa::ComputePipelineCompileInfo gen_luminace_average_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/tasks/autoexposure.glsl"},
            .compile_options = {.defines = {{"GEN_AVERAGE", "1"}}},
        },
        .push_constant_size = GenLuminanceAverage::attachment_shader_data_size(),
        .name = std::string{GenLuminanceAverage{}.name()},
    };
};

struct GenLuminanceHistogramTask : GenLuminanceHistogram
{
    GenLuminanceHistogram::AttachmentViews views = {};
    GPUContext * context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const offscreen_resolution = context->shader_globals.settings.render_target_size;
        auto const dispatch_size = u32vec2{
            (offscreen_resolution.x + COMPUTE_HISTOGRAM_WG_X - 1) / COMPUTE_HISTOGRAM_WG_X,
            (offscreen_resolution.y + COMPUTE_HISTOGRAM_WG_Y - 1) / COMPUTE_HISTOGRAM_WG_Y,
        };
        ti.recorder.set_pipeline(*context->compute_pipelines.at(GenLuminanceHistogram{}.name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
    }
};

struct GenLuminanceAverageTask : GenLuminanceAverage
{
    GenLuminanceAverage::AttachmentViews views = {};
    GPUContext * context = {};

    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(GenLuminanceAverage{}.name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.dispatch({.x = 1});
    }
};
#endif