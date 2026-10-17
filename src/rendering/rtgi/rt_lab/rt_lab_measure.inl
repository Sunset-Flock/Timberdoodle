#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

#include "rt_lab_measure_layout.inl"

#define RT_LAB_MEASURE_STORE_X 64

// One thread per tile: bins both metrics of this frame into frame_histogram (cleared every frame).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabMeasureAccumulateH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, resolved)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, split_difference_resolved)
// Full res running luma statistics (mean, variance, frames, linear depth), unexposed luma: last frame's (reprojected)
// and this frame's.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, luma_statistics_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, luma_statistics)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), frame_histogram)
DAXA_DECL_TASK_HEAD_END

struct RtLabMeasureAccumulatePush
{
    daxa_BufferPtr(RtLabMeasureAccumulateH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;  // tile grid
    daxa_u32 frame;     // frame of the run, 0 = (re)start the luma statistics of every pixel
    daxa_f32 window;    // luma statistics EMA window in frames
};

// Temporal lag: one thread per RT_LAB_MEASURE_LAG_TILE^2 tile, writes its mean unexposed output luminance (-1 = no
// geometry) to lag_run[row x tiles + tile].
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabMeasureLagH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, resolved)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_f32), lag_run) // host visible
DAXA_DECL_TASK_HEAD_END

struct RtLabMeasureLagPush
{
    daxa_BufferPtr(RtLabMeasureLagH::AttachmentShaderBlob) attach;
    daxa_u32vec2 tiles;
    daxa_u32 row;
};

// Appends this frame's histograms to row `row` of the host visible run buffer.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabMeasureStoreH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), frame_histogram)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), run_histograms)
DAXA_DECL_TASK_HEAD_END

struct RtLabMeasureStorePush
{
    daxa_BufferPtr(RtLabMeasureStoreH::AttachmentShaderBlob) attach;
    daxa_u32 row;
};
