#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

#define RTGI_PIXEL_SCALE_DIV 2

struct RtgiSettings
{
    daxa_i32 enabled TIDO_DEFAULT_VALUE(1);
    daxa_f32 ao_range TIDO_DEFAULT_VALUE(1.0f);
    daxa_i32 ray_samples TIDO_DEFAULT_VALUE(1);
    daxa_i32 firefly_filter_enabled TIDO_DEFAULT_VALUE(1);
    daxa_f32 firefly_filter_ceiling TIDO_DEFAULT_VALUE(32.0f);
    daxa_i32 pre_blur_enabled TIDO_DEFAULT_VALUE(1);
    daxa_f32 pre_blur_base_width TIDO_DEFAULT_VALUE(32);
    daxa_i32 temporal_accumulation_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 temporal_fast_history_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 temporal_firefly_filter_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 history_frames TIDO_DEFAULT_VALUE(64);
    daxa_i32 post_blur_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 upscale_enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 sh_resolve_enabled TIDO_DEFAULT_VALUE(1);
};