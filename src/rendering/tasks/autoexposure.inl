#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/globals.inl"

#define LUM_HISTOGRAM_BIN_COUNT 256
#define SUM_HISTOGRAM_INDEX 256
#define COMPUTE_HISTOGRAM_WG_X 32
#define COMPUTE_HISTOGRAM_WG_Y 32

struct AutoExposureState
{
    // NOTE(grundlett): other shaders will use this buffer just as the exposure
    // Make sure to keep first element always just as a float for exposure.
    float exposure;

    float ev;
    // Reciprocal of exposure (1/exposure), precomputed here so shaders can multiply instead of divide.
    // Field order (exposure, ev, inv_exposure) must match RenderGlobalData so the copy-into-globals task
    // can copy all three contiguously.
    float inv_exposure;
};

DAXA_DECL_BUFFER_PTR(AutoExposureState)

#define ae_hist_t daxa_u32

struct AutoExposureHistogram
{
    ae_hist_t bins[LUM_HISTOGRAM_BIN_COUNT];
    ae_hist_t max_bin_value;
    ae_hist_t bins_total_count;

    float ev_fast;
    float ev_slow;
    float desired_ev;
};

DAXA_DECL_BUFFER_PTR(AutoExposureHistogram)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(GenLuminanceHistogramH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(AutoExposureHistogram), histogram)
DAXA_TH_IMAGE_ID(READ, REGULAR_2D, color_image)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(GenLuminanceAverageH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(AutoExposureHistogram), histogram)
// Writes this frame's exposure into the double-buffered .current() slot. The previous frame's exposure
// (needed for the EMA) is read from globals.exposure_ev, which the copy task filled from .previous().
DAXA_TH_BUFFER_PTR(WRITE, daxa_BufferPtr(AutoExposureState), exposure_state)
DAXA_DECL_TASK_HEAD_END

#if __cplusplus
#include "../../gpu_context.hpp"

inline daxa::ComputePipelineCompileInfo2 gen_luminace_histogram_pipeline_compile_info()
{
    return {
        .source = daxa::ShaderFile{"./src/rendering/tasks/autoexposure.glsl"},
        .defines = {{"GEN_HISTOGRAM", "1"}},
        .push_constant_size = sizeof(GenLuminanceHistogramH::AttachmentShaderBlob),
        .name = std::string{GenLuminanceHistogramH::Info::NAME},
    };
};

inline daxa::ComputePipelineCompileInfo2 gen_luminace_average_pipeline_compile_info()
{
    return {
        .source = daxa::ShaderFile{"./src/rendering/tasks/autoexposure.glsl"},
        .defines = {{"GEN_AVERAGE", "1"}},
        .push_constant_size = sizeof(GenLuminanceAverageH::AttachmentShaderBlob),
        .name = std::string{GenLuminanceAverageH::Info::NAME},
    };
};

inline void gen_luminance_histogram_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = GenLuminanceHistogramH::Info::AT;
    render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"MISC","AUTO_EXPOSURE_GEN_HIST">());
    auto const offscreen_resolution = render_context->render_data.settings.render_target_size;
    auto const dispatch_size = u32vec2{
        (offscreen_resolution.x + COMPUTE_HISTOGRAM_WG_X - 1) / COMPUTE_HISTOGRAM_WG_X,
        (offscreen_resolution.y + COMPUTE_HISTOGRAM_WG_Y - 1) / COMPUTE_HISTOGRAM_WG_Y,
    };
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(gen_luminace_histogram_pipeline_compile_info().name));
    GenLuminanceHistogramH::AttachmentShaderBlob push = ti.attachment_shader_blob;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
    render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"MISC","AUTO_EXPOSURE_GEN_HIST">());
}

inline void gen_luminance_average_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = GenLuminanceAverageH::Info::AT;
    render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"MISC","AUTO_EXPOSURE_AVERAGE">());
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(gen_luminace_average_pipeline_compile_info().name));
    GenLuminanceAverageH::AttachmentShaderBlob push = ti.attachment_shader_blob;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({.x = 1});
    render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"MISC","AUTO_EXPOSURE_AVERAGE">());
}
#endif
