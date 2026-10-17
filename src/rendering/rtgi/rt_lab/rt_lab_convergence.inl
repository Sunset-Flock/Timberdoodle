#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"
#include "../../../shader_shared/readback.inl"

#define RT_LAB_CONVERGENCE_X 8
#define RT_LAB_CONVERGENCE_Y 8

// Convergence error estimate (RtLabState, rt_lab_state.hpp). Last RT Lab tasks, after the resolve.
// Capture targets of the capture pass (push.target).
#define RT_LAB_CAPTURE_NONE 0
#define RT_LAB_CAPTURE_PROGRESSIVE 1
#define RT_LAB_CAPTURE_CUSTOM 2

// Copies this frame's resolved full res diffuse into one of the two persistent capture images. Both captures also store
// that run's POST-SPATIAL split history error prediction |resolved split difference| (UNEXPOSED luminance, the analysis
// applies its frame's exposure to it like to the captures; -1 = none): predicted_error.x = custom run, .y = progressive
// (reference) run. Each capture writes only its own channel (the move estimate captures custom first). The actual error custom - progressive contains BOTH
// runs' errors, so the analysis compares it against sqrt(x^2 + y^2).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabConvergenceCaptureH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, resolved)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, progressive)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, custom)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, split_difference) // full res, null without split history
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, predicted_error)
DAXA_DECL_TASK_HEAD_END

struct RtLabConvergenceCapturePush
{
    daxa_BufferPtr(RtLabConvergenceCaptureH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 target; // RT_LAB_CAPTURE_*
    daxa_u32 split_enabled; // split_difference is bound
};

// Compares the two captures: error sums + signed luminance difference histogram into the general readback, and the
// RT_LAB_CONVERGENCE debug views.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabConvergenceAnalyzeH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, progressive)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, custom)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, predicted_error)
DAXA_DECL_TASK_HEAD_END

struct RtLabConvergenceAnalyzePush
{
    daxa_BufferPtr(RtLabConvergenceAnalyzeH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_f32 histogram_range; // histogram covers [-range, range] signed luminance difference (exposed irradiance)
    daxa_u32 step;            // move estimate step compared this frame, written to the readback (~0u = static estimate)
};
