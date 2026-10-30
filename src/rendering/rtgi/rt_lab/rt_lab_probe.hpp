#pragma once

#include <array>
#include <deque>
#include <daxa/utils/task_graph.hpp>

#include "../../../timberdoodle.hpp"

using namespace tido::types;

struct RenderContext;

// Live pixel probes (UI RT Lab -> Live Pixel Probes, pass rt_lab_probe.hlsl). Up to two pixels of the viewport are followed every frame and their
// denoiser state is shown side by side: the average colors at every stage (this frame's rays, the temporal mean, the spatially filtered output, what
// is displayed) and the live body / tail distribution estimate of the distribution guide (RtLabSettings::distribution_guide_mode). Embedded in
// RtLabState::probe (free of RenderContext so rt_lab_state.hpp can include it).

#define RT_LAB_PROBE_UI_COUNT 2
#define RT_LAB_PROBE_HISTORY_FRAMES 240 // frames of the trend plots

// One frame of one probe (layout of rt_lab_probe.inl, all radiance in scene units, i.e. RTGI_RADIANCE_SCALE divided out).
struct RtLabProbeSample
{
    bool geometry = false;
    bool has_variance = false;
    bool has_dist = false;     // distribution A + B
    bool has_dist_low = false; // distribution C (dark tail)
    u32 lab_pixel[2] = {};
    f32 linear_depth = 0.0f;
    f32 exposure = 1.0f;
    f32 temporal[3] = {};
    f32 history_length = 0.0f;
    f32 filtered[3] = {};
    f32 effective_count = 0.0f;
    f32 raw[3] = {};
    f32 rays = 0.0f;
    f32 resolved[3] = {};
    f32 variance[3] = {};
    f32 dist_a[4] = {}; // body mean, variance, 3rd central moment, count
    f32 dist_b[4] = {}; // tail fraction, tail mean, tail variance, tail count
    f32 dist_c[4] = {}; // dark tail fraction, mean, variance, count
    // First spatial pass weights with this probe as the center and the other one as the tap (no kernel, ray length or SH guides).
    bool pair_valid = false;
    bool pair_variance_on = false;
    bool pair_distribution_on = false;
    f32 pair_geometry = 1.0f, pair_variance = 1.0f, pair_distribution = 1.0f, pair_total = 1.0f;
};

// Luminance trend point of the plots.
struct RtLabProbeTrend
{
    f32 raw = 0.0f;       // this frame's rays (0 = none)
    f32 temporal = 0.0f;
    f32 filtered = 0.0f;
    f32 resolved = 0.0f;
    f32 length = 0.0f;
    f32 tail_fraction = 0.0f;
};

struct RtLabProbe
{
    bool active = false;    // follows a pixel
    i32 pixel[2] = {};      // screen pixel
    bool pause = false;     // keep showing the last sample
    u64 frames = 0;         // samples received for this pixel
    bool valid = false;     // `latest` holds a sample
    RtLabProbeSample latest = {};
    std::deque<RtLabProbeTrend> trend = {};
};

struct RtLabProbeState
{
    bool enabled = false;   // UI: the pass is part of the graph
    bool built = false;     // graph built with the pass (rebuild trigger)
    daxa::ExternalTaskBuffer buffer = daxa::ExternalTaskBuffer{{.name = "rt_lab_probe"}}; // host visible ring of slots

    std::array<RtLabProbe, RT_LAB_PROBE_UI_COUNT> probes = {};
    i32 picking = -1;       // probe index that takes the next viewport click, -1 = none
    bool exposed_units = false; // luminance x exposure x pi (the INDIRECT_DIFFUSE debug view's units)
    bool log_axis = true;       // distribution plot over a log10 axis

    // Per frame (rt_lab_probe_frame_update), read by the dispatch callback.
    bool dispatch_pending = false;
    u32 dispatch_stamp = 0;
    u32 dispatch_active_mask = 0;
    u32 dispatch_pixel[RT_LAB_PROBE_UI_COUNT][2] = {};

    // Frame stamps: a slot is read once the frames in flight that wrote it are done.
    u32 frame_counter = 0;
    u32 next_stamp = 1;
    u32 last_ingested = 0;
    std::array<u32, 8> issued_stamp = {};
    std::array<u32, 8> issued_frame = {};
    std::array<std::array<i32, 2>, 8> issued_pixel_a = {};
    std::array<std::array<i32, 2>, 8> issued_pixel_b = {};
};

// UI (every frame while the RT Lab window is open): a viewport click while a probe is picking sets its pixel. Returns true when the click was consumed.
auto rt_lab_probe_input(RenderContext & render_context, bool mouse_down, i32 cursor_x, i32 cursor_y, i32 window_width, i32 window_height) -> bool;
// UI: the two probes side by side.
void rt_lab_probe_ui(RenderContext & render_context);
// Start of every frame in the renderer: reads finished slots, decides this frame's dispatch. True = the task graph must be rebuilt.
auto rt_lab_probe_frame_update(RenderContext & render_context) -> bool;
// Renderer destruction.
void rt_lab_probe_destroy(RenderContext & render_context);
// Task graph dispatch callback (RtLabProbeH).
void rt_lab_probe_callback(daxa::TaskInterface ti, RenderContext * render_context, bool spatial_ran, bool variance_bound, bool dist_ab_bound, bool dist_c_bound);
