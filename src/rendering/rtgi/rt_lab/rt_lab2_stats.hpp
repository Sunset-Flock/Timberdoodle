#pragma once

#include <array>
#include <deque>
#include <daxa/utils/task_graph.hpp>

#include "../../../timberdoodle.hpp"

using namespace tido::types;

struct RenderContext;

// RT Lab 2 grid statistics (UI RT Lab -> RT Lab 2 -> Statistics, pass entry_rt_lab2_stats in rt_lab2.hlsl): how many cells the grid holds, how many were
// queried last frame, their coverage, rays and history fill, distributions per cascade level, and time series of all of it. The GPU pass scans every
// cell slot after the cell integrate pass into a host visible slot (zeroed by the CPU); the UI reads it a few frames later. Embedded in RtLab2State::stats
// (free of RenderContext so rt_lab2_state.hpp can include it).

#define RT_LAB2_STATS_UI_BUCKETS 16
#define RT_LAB2_STATS_HISTORY_FRAMES 300

struct RtLab2Stats
{
    bool valid = false;
    u32 live = 0;            // cells in the grid
    u32 touched = 0;         // cells queried by a pixel last frame
    u32 with_data = 0;       // cells with at least one ray in their history
    u32 stale = 0;           // live cells nobody queried last frame
    u32 cells_with_rays = 0; // cells that got rays last frame
    u64 pixels = 0;          // sum of the coverage of the queried cells (pixels in the cascade blend regions count twice)
    u64 rays = 0;            // rays summed into cells last frame
    std::array<u32, RT_LAB2_STATS_UI_BUCKETS> level_live = {};    // live cells per cascade level
    std::array<u32, RT_LAB2_STATS_UI_BUCKETS> level_touched = {}; // queried cells per level
    std::array<u32, RT_LAB2_STATS_UI_BUCKETS> fill = {};          // queried cells with data by history fill (samples / window) in 16 steps
    std::array<u32, RT_LAB2_STATS_UI_BUCKETS> coverage = {};      // queried cells by log2(1 + pixel coverage)
    std::array<u32, RT_LAB2_STATS_UI_BUCKETS> rays_hist = {};     // queried cells by log2(1 + rays last frame)
    std::array<u32, RT_LAB2_STATS_UI_BUCKETS> untouched = {};     // live cells by log2(1 + frames untouched)
    f32 mean_fill = 0.0f;
};

struct RtLab2StatsPoint
{
    f32 live = 0.0f;
    f32 touched = 0.0f;
    f32 with_data = 0.0f;
    f32 pixels = 0.0f;
    f32 rays = 0.0f;
    f32 mean_fill = 0.0f;
    f32 rays_per_pixel = 0.0f;
    f32 load = 0.0f; // live / capacity
};

struct RtLab2StatsState
{
    bool enabled = false;   // UI: the pass is part of the graph
    bool built = false;     // graph built with the pass (rebuild trigger)
    daxa::ExternalTaskBuffer buffer = daxa::ExternalTaskBuffer{{.name = "rt_lab2_stats"}}; // host visible ring of slots

    RtLab2Stats latest = {};
    std::deque<RtLab2StatsPoint> history = {};
    bool pause = false;

    // Per frame (rt_lab2_stats_frame_update), read by the dispatch callback.
    bool dispatch_pending = false;
    u32 dispatch_slot = 0;

    // Ring bookkeeping: a slot is read once the frames in flight that wrote it are done.
    u32 frame_counter = 0;
    u32 next_slot = 0;
    std::array<u32, 8> issued_frame = {};
    std::array<bool, 8> pending = {};
};

// Start of every frame in the renderer: reads finished slots, zeroes + issues this frame's slot. True = the task graph must be rebuilt.
auto rt_lab2_stats_frame_update(RenderContext & render_context) -> bool;
// Renderer destruction.
void rt_lab2_stats_destroy(RenderContext & render_context);
// Task graph dispatch callback (RtLab2StatsH).
void rt_lab2_stats_callback(daxa::TaskInterface ti, RenderContext * render_context);
// UI: counters, graphs, pass timings.
void rt_lab2_stats_ui(RenderContext & render_context);
