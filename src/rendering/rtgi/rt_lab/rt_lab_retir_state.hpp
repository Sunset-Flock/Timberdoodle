#pragma once

#include "../../../timberdoodle.hpp"
#include "rt_lab_retir_settings.inl"

using namespace tido::types;

inline auto rt_lab_retir_default_settings() -> RtLabRetirSettings
{
    return RtLabRetirSettings{
        .temporal_enabled = 1,
        .temporal_max_m = 8.0f,
        .temporal_plane_tolerance = 2.0f,
        .temporal_normal_threshold = 0.9f,
        .temporal_visibility = 0,
        .temporal_max_age = 8,
        .temporal_max_age_jitter = 0.5f,
        .temporal_partial_scale = 1.0f,
        .jacobian_max = 10.0f,
        .min_reconnection_distance = 0.05f,
    };
}

// ReTIR settings: part of every preset (RtLabPresetConfig).
struct RtLabRetirConfig
{
    RtLabRetirSettings settings = rt_lab_retir_default_settings();
};

// Live copy of the active preset config (RtLabState::retir).
struct RtLabRetirState : RtLabRetirConfig
{
};
