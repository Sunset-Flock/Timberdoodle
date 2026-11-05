#pragma once

#include "../../../timberdoodle.hpp"

using namespace tido::types;

// RT Lab 4 (rt_lab4.hpp): RT Lab 3 (ReSTIR GI) in front of RT Lab 2 (hash grid cells). Its settings are RT Lab 3's (ReSTIR stage) and RT Lab 2's (cells,
// temporal resolve); only the glue between them lives here.
// The glue settings between the ReSTIR stage and a denoiser behind it: part of every preset (RtLabPresetConfig). RtLab4State holds the live copy.
struct RtLab4Config
{
    // What a pixel counts as when its estimate goes into its cell / the classic temporal pass (RT_LAB3_CELL_WEIGHT_*: 0 one per pixel with a reservoir,
    // 1 the fresh rays it shot, 2 the confidence M of its temporal reservoir).
    i32 cell_weight_mode = 1;
    // What goes into the denoiser behind ReSTIR (RT_LAB3_CELL_SOURCE_*): 0 the filtered ReSTIR estimate, 1 the mean of the pixel's fresh rays (independent
    // samples, ReSTIR only drives the ray redistribution and the geometry then).
    i32 cell_source = 0;
};

struct RtLab4State : RtLab4Config
{
    bool enabled = false;
    // This frame's decision (rt_lab_frame_update): enabled, and no progressive reference phase is running.
    bool active = false;
    // Graph shape: the values the task graph was last built for.
    bool built_active = false;
    i32 built_cell_weight_mode = 0;
    i32 built_cell_source = 0;
};
