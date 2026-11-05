#pragma once

#include "../../../timberdoodle.hpp"
#include "rt_lab_integrator.inl"

using namespace tido::types;

inline auto rt_lab_integrator_default_settings() -> RtLabIntegratorSettings
{
    return RtLabIntegratorSettings{
        .window_radius = 2,
        .variance_gamma = 2.0f,
        .alpha_min = 0.033f,
        .alpha_max = 0.128f,
        .variance_reference = 0.15f,
        .spatial_strength = 0.5f,
        .max_frames = 64,
        .plane_tolerance = 2.0f,
        .normal_threshold = 0.9f,
        .reset = 0,
    };
}

// The settings of the temporal resolve: part of every preset (RtLabPresetConfig). Default ON (the hash grid's integrator was always on).
struct RtLabIntegratorConfig
{
    bool enabled = true;
    RtLabIntegratorSettings settings = rt_lab_integrator_default_settings();
};

// CPU side state of the ONE temporal resolve (rt_lab_integrator.hpp), RtLabState::integrator, live copy of the active preset's config. Every pipeline uses it.
struct RtLabIntegratorState : RtLabIntegratorConfig
{
    bool built_enabled = false; // graph shape: the value the task graph was last built for
    // The persistent history is undefined after a graph rebuild: the first frame ignores it (set when the pass is added to a graph).
    bool clear_requested = true;
};
