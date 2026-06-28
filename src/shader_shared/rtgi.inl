#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

#define RTGI_PIXEL_SCALE_DIV 2

struct RtgiSettings
{
    daxa_i32 enabled;
    daxa_f32 ao_range;
    daxa_i32 ray_samples;
    daxa_i32 firefly_filter_enabled;
    daxa_f32 firefly_filter_ceiling;
    daxa_i32 firefly_clamp_mode; // 0=multichromatic, 1=monochromatic
    daxa_i32 pre_blur_enabled;
    daxa_i32 pre_blur_geometric_guiding;
    daxa_i32 pre_blur_geometric_mean_guiding;
    daxa_f32 pre_blur_geometric_mean_guiding_factor;
    daxa_f32 pre_blur_base_width;
    daxa_i32 pre_blur_sample_count_min;
    daxa_i32 pre_blur_sample_count_max;
    daxa_i32 pre_blur_iterations;
    daxa_i32 temporal_accumulation_enabled;
    daxa_i32 temporal_fast_history_enabled;
    daxa_i32 temporal_firefly_filter_enabled;
    daxa_f32 temporal_firefly_std_dev_clamp;
    daxa_f32 temporal_variance_fast_history_blend;
    daxa_i32 history_frames;
    daxa_i32 post_blur_enabled;
    daxa_i32 post_blur_geometric_guiding;
    daxa_i32 post_blur_geometric_mean_guiding;
    daxa_f32 post_blur_geometric_mean_guiding_factor;
    daxa_f32 geometric_guide_floor;
    daxa_i32 post_blur_mode;
    daxa_i32 post_blur_variance_guiding;
    daxa_i32 post_blur_disocclusion_blur_enabled;
    daxa_i32 post_blur_stride;
    daxa_i32 post_blur_max_width;
    daxa_i32 post_blur_atrous_iterations;
    daxa_i32 upscale_enabled;
    daxa_i32 sh_resolve_enabled;
    daxa_i32 use_compute_trace;
    daxa_i32 firefly_star_blur_enabled;
    daxa_i32 firefly_energy_compensation_enabled;
    daxa_i32 animate_noise;
};
