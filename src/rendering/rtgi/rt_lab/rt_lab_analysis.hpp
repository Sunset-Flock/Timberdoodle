#pragma once

#include <array>
#include <deque>
#include <daxa/utils/task_graph.hpp>

#include "../../../timberdoodle.hpp"

using namespace tido::types;

struct RenderContext;
struct TasksRtgiInfo;

// RT Lab analysis (UI RT Lab -> Analysis -> Statistics / Pixel Picker, passes rt_lab_analysis.hlsl). Both work on the RESOLVED output only.
//  Statistics: spatial, no history. Log10 histograms of the luminance and of the 5x5 spatial luminance variance over all geometry pixels. The
//  bucket range follows the data smoothly (percentile targets, fast widening, slow hysteresis narrowing) so the plot does not jitter.
//  Pixel Picker: up to two followed pixels, every frame their resolved color + 5x5 window statistics are read back; the UI shows the temporal
//  trend and a log histogram of the pixel's luminance over time.
// Embedded in RtLabState::analysis (free of RenderContext so rt_lab_state.hpp can include it).

#define RT_LAB_PICKER_UI_COUNT 2
#define RT_LAB_PICKER_HISTORY_FRAMES 2048

// A log10 axis range that moves smoothly towards a target.
struct RtLabSmoothRange
{
    f64 low = -4.0;
    f64 high = 2.0;
};

struct RtLabStatsHistogram
{
    bool valid = false;
    f64 low = 0.0; // log10 range the bins were binned in
    f64 high = 0.0;
    std::array<u32, 128> bins = {};
    u32 under = 0;
    u32 over = 0;
    u64 total() const;
};

// Exact moments of a statistic over all counted pixels (summed from the per group partials of the GPU, in double).
struct RtLabStatsSummary
{
    bool valid = false;
    u64 count = 0;
    f64 mean = 0.0;
    f64 std_dev = 0.0;
    f64 min = 0.0;
    f64 max = 0.0;
    u64 positive_count = 0;
    f64 geometric_mean = 0.0; // 10^mean(log10 x) over the positive values
    f64 log10_std = 0.0;      // std of log10 x: a multiplicative spread, a factor of 10^log10_std
};

struct RtLabPickerSample
{
    bool geometry = false;
    f32 rgb[3] = {};
    f32 window_mean = 0.0f;
    f32 window_variance = 0.0f;
    f32 window_count = 0.0f;
    f32 window_min = 0.0f;
    f32 window_max = 0.0f;
    f32 exposure = 1.0f;
    f32 luma = 0.0f;
};

struct RtLabPicker
{
    bool active = false;
    i32 pixel[2] = {};
    bool pause = false;
    u64 frames = 0;
    std::deque<RtLabPickerSample> samples = {};
    RtLabSmoothRange histogram_range = {};
    f64 histogram_peak = 0.0;
};

struct RtLabAnalysisState
{
    // Statistics.
    bool stats_enabled = false;
    bool stats_built = false;
    daxa::ExternalTaskBuffer stats_buffer = daxa::ExternalTaskBuffer{{.name = "rt_lab_stats"}};
    RtLabSmoothRange luma_range = {-4.0, 2.0};
    RtLabSmoothRange variance_range = {-8.0, 2.0};
    RtLabStatsHistogram luma_histogram = {};
    RtLabStatsHistogram variance_histogram = {};
    RtLabStatsSummary luma_summary = {};
    RtLabStatsSummary variance_summary = {};
    f64 luma_peak = 0.0;     // smoothed y axis maxima (share of pixels)
    f64 variance_peak = 0.0;
    bool stats_dispatch_pending = false;
    u32 stats_dispatch_stamp = 0;
    f32 stats_dispatch_range[4] = {};
    u32 stats_dispatch_groups[2] = {};

    // Pixel picker.
    bool picker_built = false;
    daxa::ExternalTaskBuffer picker_buffer = daxa::ExternalTaskBuffer{{.name = "rt_lab_picker"}};
    std::array<RtLabPicker, RT_LAB_PICKER_UI_COUNT> pickers = {};
    i32 picking = -1;           // picker that takes the next viewport click
    i32 temporal_window = 60;   // frames the temporal statistics, trend and histogram look at
    bool exposed_units = false; // luminance x exposure x pi (the INDIRECT_DIFFUSE debug view's units)
    bool picker_dispatch_pending = false;
    u32 picker_dispatch_stamp = 0;
    u32 picker_dispatch_mask = 0;
    u32 picker_dispatch_pixel[RT_LAB_PICKER_UI_COUNT][2] = {};

    // Frame stamps: a slot is read once the frames in flight that wrote it are done.
    u32 frame_counter = 0;
    u32 next_stamp = 1;
    u32 stats_last_ingested = 0;
    u32 picker_last_ingested = 0;
    std::array<u32, 8> stats_issued_stamp = {};
    std::array<u32, 8> stats_issued_frame = {};
    std::array<std::array<f32, 4>, 8> stats_issued_range = {};
    std::array<std::array<u32, 2>, 8> stats_issued_groups = {};
    std::array<u32, 8> picker_issued_stamp = {};
    std::array<u32, 8> picker_issued_frame = {};
};

// Start of every frame in the renderer: reads finished slots, decides this frame's dispatches. True = the task graph must be rebuilt.
auto rt_lab_analysis_frame_update(RenderContext & render_context) -> bool;
// UI: a viewport click while a picker is picking sets its pixel (markers are drawn every frame). True = the click was consumed.
auto rt_lab_picker_input(RenderContext & render_context, bool mouse_down, i32 cursor_x, i32 cursor_y, i32 window_width, i32 window_height) -> bool;
void rt_lab_statistics_ui(RenderContext & render_context);
void rt_lab_picker_ui(RenderContext & render_context);
void rt_lab_analysis_destroy(RenderContext & render_context);
// Adds the analysis tasks (when enabled) after a lab's resolve.
void rt_lab_add_analysis(TasksRtgiInfo const & info, daxa::TaskImageView resolved);

void rt_lab_stats_callback(daxa::TaskInterface ti, RenderContext * render_context);
void rt_lab_picker_callback(daxa::TaskInterface ti, RenderContext * render_context);
