#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

#define RTGI_DIFFUSE_PIXEL_SCALE_DIV 2

struct RtgiSettings
{
    daxa_i32 enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 firefly_filter_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 spatial_filter_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 disocclusion_filter_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 temporal_accumulation_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 disocclusion_threshold_scale_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 disocclusion_flood_fill_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 temporal_stabilization_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 temporal_reactivity_heuristic_enabled TIDO_DEFAULT_VALUE(0);
    daxa_i32 upscaling_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 history_frames TIDO_DEFAULT_VALUE(16);
    daxa_f32 spatial_filter_width TIDO_DEFAULT_VALUE(16);
};