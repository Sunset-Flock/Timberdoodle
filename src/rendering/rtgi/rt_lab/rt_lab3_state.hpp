#pragma once

#include <algorithm>

#include "../../../timberdoodle.hpp"
#include "rt_lab3_settings.inl"
#include "rt_lab_integrator_state.hpp"

using namespace tido::types;

auto rt_lab3_default_settings() -> RtLab3Settings;

// CPU side state of RT Lab 3 (rt_lab3.hpp), embedded in RtLabState::lab3. RT Lab 3 is ReSTIR GI on the RT Lab trace: with `enabled`
// set (and the RT Lab master switch on) tasks_rt_lab3_main replaces the other lab denoisers. The progressive reference phases of the
// tests keep running the original lab.
// The settings of RT Lab 3 (ReSTIR GI): part of every preset (RtLabPresetConfig). RtLab3State holds the live copy of the active preset's.
struct RtLab3Config
{
    // Ray redistribution is NOT configured here: the stage uses the preset's Trace Settings (RtLabSettings::redistribution_enabled, ray_budget,
    // min_ray_budget = the exact base coverage, max_rays_per_pixel, fast_convergence_samples = the target rays).
    RtLab3Settings settings = rt_lab3_default_settings();
};

// RT Lab 3 is ReSTIR GI on the RT Lab trace; the live state of the stage. Which stage runs is decided by the preset's pipeline (rt_lab_pipeline_resolve).
// The resolution is the preset's half_res; the temporal resolve is the shared one (RtLabState::integrator).
struct RtLab3State : RtLab3Config
{
    bool enabled = false;
    // The ReSTIR stage feeds the classic denoiser of the original lab (pipeline: ReSTIR + classic). Derived by rt_lab_pipeline_resolve, like `enabled`.
    bool classic_denoiser = false;
    // This frame's decision (rt_lab_frame_update): enabled, and no progressive reference phase is running.
    bool active = false;
    // Graph shape: the values the task graph was last asked to be rebuilt for.
    bool built_active = false;
    i32 built_spatial_passes = 1;
    // Test plan parsing only ("lab3": {"half_res"}): the resolution is the preset's half_res.
    bool half_res = false;
    // This frame's reset (a test phase start, or a freshly built graph whose persistent resources are undefined).
    bool frame_reset = false;
    bool clear_requested = true;
};
