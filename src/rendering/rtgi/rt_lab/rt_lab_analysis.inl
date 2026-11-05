#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

// RT Lab analysis (rt_lab_analysis.hpp): passes on the RESOLVED output (full res, scene radiance units).
//
// STATISTICS (spatial only, no history): luminance histogram + 5x5 spatial luminance variance histogram over all geometry pixels, log10 binned in a
// range the CPU adapts smoothly (the range of a frame is a push constant, the slot remembers nothing else). Slot words (u32):
//   0 stamp (never 0)   1..3 unused   then the luminance histogram (RT_LAB_STATS_HIST_WORDS: bins, under, over) and the variance histogram.
// PIXEL PICKER: the resolved color of up to RT_LAB_PICKER_COUNT followed pixels + their 5x5 window luminance statistics, one slot per frame.
//   header (RT_LAB_PICKER_HEADER_WORDS): 0 stamp   1 active mask
//   picker i at header + i * RT_LAB_PICKER_WORDS: 0 flags (bit 0 geometry)  1,2 screen pixel  3..5 resolved rgb  6 window mean luma
//   7 window variance (n - 1)  8 window pixel count  9 exposure  10 window min luma  11 window max luma
#define RT_LAB_STATS_BINS 128
#define RT_LAB_STATS_SLOTS 8
#define RT_LAB_STATS_HIST_WORDS (RT_LAB_STATS_BINS + 2)
#define RT_LAB_STATS_HEADER_WORDS 4
// Exact moments: every RT_LAB_STATS_GROUP^2 thread group writes one partial (luminance then variance, RT_LAB_STATS_METRIC_WORDS floats each:
// count, sum, sum of squares, min, max, count of positive values, sum of log10, sum of squared log10), the CPU adds them up in double.
#define RT_LAB_STATS_GROUP 16
#define RT_LAB_STATS_METRIC_WORDS 8
#define RT_LAB_STATS_PARTIAL_WORDS (2 * RT_LAB_STATS_METRIC_WORDS)
#define RT_LAB_STATS_MAX_GROUPS 40960 // 4K at 16x16 is 32400 groups; larger targets lose the exact moments (the histograms stay complete)
#define RT_LAB_STATS_PARTIALS_OFFSET (RT_LAB_STATS_HEADER_WORDS + 2 * RT_LAB_STATS_HIST_WORDS)
#define RT_LAB_STATS_SLOT_WORDS (RT_LAB_STATS_PARTIALS_OFFSET + RT_LAB_STATS_MAX_GROUPS * RT_LAB_STATS_PARTIAL_WORDS)
#define RT_LAB_STATS_VARIANCE_RADIUS 2

#define RT_LAB_PICKER_COUNT 2
#define RT_LAB_PICKER_SLOTS 8
#define RT_LAB_PICKER_HEADER_WORDS 4
#define RT_LAB_PICKER_WORDS 12
#define RT_LAB_PICKER_SLOT_WORDS (RT_LAB_PICKER_HEADER_WORDS + RT_LAB_PICKER_COUNT * RT_LAB_PICKER_WORDS)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabStatsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, resolved)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), stats) // host visible
DAXA_DECL_TASK_HEAD_END

struct RtLabStatsPush
{
    daxa_BufferPtr(RtLabStatsH::AttachmentShaderBlob) attach;
    daxa_f32 luma_log_min; // log10 range of the luminance histogram
    daxa_f32 luma_log_max;
    daxa_f32 variance_log_min; // log10 range of the variance histogram
    daxa_f32 variance_log_max;
    daxa_u32 slot;
    daxa_u32 stamp;
    daxa_u32 groups_x;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabPickerH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, resolved)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), picker) // host visible
DAXA_DECL_TASK_HEAD_END

struct RtLabPickerPush
{
    daxa_BufferPtr(RtLabPickerH::AttachmentShaderBlob) attach;
    daxa_u32vec2 screen_pixel[RT_LAB_PICKER_COUNT];
    daxa_u32 slot;
    daxa_u32 stamp;
    daxa_u32 active_mask; // bit i: picker i has a pixel
    daxa_u32 _pad;
};
