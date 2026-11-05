#pragma once

#include <algorithm>

#include "../../../timberdoodle.hpp"
#include "rt_lab2_settings.inl"
#include "rt_lab2_probe.hpp"
#include "rt_lab2_stats.hpp"

using namespace tido::types;

auto rt_lab2_default_settings() -> RtLab2Settings;

// Number of cell slots: 2^cell_capacity_log2 (0 = the default 2^20). Same rule as rt_lab2_capacity in the shader.
inline auto rt_lab2_cell_capacity_log2(RtLab2Settings const & settings) -> i32
{
    f32 const value = settings.cell_capacity_log2 > 0.0f ? settings.cell_capacity_log2 : static_cast<f32>(RT_LAB2_CELL_CAPACITY_LOG2_DEFAULT);
    return static_cast<i32>(std::clamp(value, 12.0f, 24.0f));
}
inline auto rt_lab2_cell_capacity(RtLab2Settings const & settings) -> u32
{
    return 1u << static_cast<u32>(rt_lab2_cell_capacity_log2(settings));
}

// CPU side state of RT Lab 2 (rt_lab2.hpp), embedded in RtLabState::lab2. RT Lab 2 is a second denoiser for the RT Lab
// trace: with `enabled` set (and the RT Lab master switch on) tasks_rt_lab2_main replaces tasks_rt_lab_main. The original
// lab stays as it is; the test setups' progressive reference phases always run it.
// The settings of RT Lab 2 (hash grid): part of every preset (RtLabPresetConfig). RtLab2State holds the live copy of the active preset's.
struct RtLab2Config
{
    // Rays per full res pixel per frame (hard budget). Below 1 only that fraction of the pixels shoots (rotated Bayer
    // selection per 8x8 tile), above 1 every pixel shoots and the rest is spread. The trace itself (hit shading, AO on hit)
    // takes the active RT Lab preset's trace settings.
    f32 ray_budget = 0.25f;
    // The base ray coverage the extra rays never drain below (0..1, fraction of the geometry pixels), see RtLab2Settings::redistribution_enabled.
    f32 min_base_coverage = 0.125f;
    // Ray filter (the old lab's pass after the trace, rt_lab_rays.hlsl entry_rt_lab_ray_filter; applied through the lab settings copy): clamps every ray
    // against the rays of the neighbor pixels, hue preserving. Mode RT_LAB_RAY_FILTER_*: 0 off, 1 absolute (neighbor mean x factor), 2 perceptual (log mean + tolerance).
    i32 ray_filter_mode = 2;
    i32 ray_filter_radius = 3;
    f32 ray_filter_tolerance = 7.0f;
    f32 ray_filter_factor = 64.0f;
    RtLab2Settings settings = rt_lab2_default_settings();
};

struct RtLab2State : RtLab2Config
{
    bool enabled = false;
    // This frame's decision (rt_lab_frame_update): enabled, and no progressive reference phase is running.
    bool active = false;
    // Graph shape: the value the task graph was last asked to be rebuilt for.
    bool built_active = false;
    i32 built_capacity_log2 = RT_LAB2_CELL_CAPACITY_LOG2_DEFAULT; // graph shape: the cell buffer size
    bool built_gradient = false;                                   // graph shape: the temporal gradient passes + the ray record buffer
    i32 capacity_ui = RT_LAB2_CELL_CAPACITY_LOG2_DEFAULT;        // UI: the slider's value, applied when released
    // This frame's reset (a test phase start, or a freshly built graph whose persistent resources are undefined), decided by
    // the first pass of the graph, read by the others.
    bool frame_reset = false;
    bool clear_requested = true;
    // Live cell probes (rt_lab2_probe.hpp).
    RtLab2ProbeState probe = {};
    // Grid statistics + graphs (rt_lab2_stats.hpp).
    RtLab2StatsState stats = {};
};

