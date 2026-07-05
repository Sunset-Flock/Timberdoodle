#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

#define RTGI_PIXEL_SCALE_DIV 2

// The indirect ray list holds the whole global ray budget for one frame. The budget is ray_percentage
// rays per half-res pixel (slider max = this value), so the list holds up to this many x the half-res
// pixel count. NOTE: this scales the ray_list + ray_result VRAM linearly.
#define RTGI_RAY_LIST_CAPACITY_MUL 4

struct RtgiSettings
{
    daxa_i32 enabled;
    daxa_f32 shading_ao_range;
    daxa_i32 firefly_filter_enabled;
    daxa_f32 firefly_filter_ceiling;
    daxa_i32 firefly_clamp_mode; // 0=multichromatic, 1=monochromatic
    daxa_i32 pre_blur_enabled;
    daxa_i32 pre_blur_ao_guiding;
    daxa_i32 pre_blur_perceptual_difference_guiding;
    daxa_i32 pre_blur_ray_count_sample_weighting;
    daxa_f32 pre_blur_perceptual_radiance_guide_tolerance;
    daxa_f32 pre_blur_base_width;
    daxa_i32 pre_blur_sample_count;
    daxa_i32 pre_blur_iterations;
    daxa_i32 temporal_accumulation_enabled;
    daxa_i32 temporal_fast_history_enabled;
    daxa_i32 temporal_fast_history_frames; // fast-history window length in frames (max 15)
    daxa_i32 temporal_firefly_filter_enabled;
    daxa_f32 temporal_firefly_std_dev_clamp;
    daxa_f32 temporal_variance_fast_history_blend;
    // Parallax stretch penalty: when camera motion makes a grazing surface much less grazing (its thin
    // previous-frame strip gets reprojected/stretched across many current pixels), scale down the
    // reprojected history so those pixels re-converge instead of smearing. 0 = disabled, higher = stronger.
    daxa_f32 temporal_parallax_penalty_strength;
    daxa_i32 max_temporal_samples; // max samples a pixel accumulates before its history saturates
    // Sample count a pixel must accumulate before it stops requesting extra rays (its demand ramps
    // linearly from this many extra rays at 0 samples down to 0 extra rays at this count). Decoupled
    // from max_temporal_samples so ray-boost aggressiveness can be tuned independently of history length.
    daxa_f32 fast_convergence_samples;
    daxa_i32 post_blur_enabled;
    daxa_i32 post_blur_ao_guiding;
    daxa_f32 post_blur_ao_guide_floor;   // radius-scaling floor for post blur (analogous to ao_guide_floor for pre blur)
    daxa_i32 post_blur_perceptual_difference_guiding;
    daxa_f32 post_blur_perceptual_radiance_guide_tolerance;
    daxa_f32 ao_guide_floor;             // radius-scaling floor for pre blur
    // Max ray hit distance (in half-res pixel widths) that still counts as a "near" hit for the ray
    // shortness the denoiser guide uses. Rays at/beyond this contribute 0 shortness. (calc_ray_shortness)
    daxa_f32 max_visibility_pixel_range;
    daxa_i32 post_blur_mode;
    // 1 = use the groupshared (LDS-preloading) variant of the separable horizontal/vertical post blur,
    // 0 = the plain texture-fetch variant. Same output; a perf A/B toggle. (À-trous mode is unaffected.)
    daxa_i32 post_blur_use_lds;
    daxa_i32 post_blur_disocclusion_blur_enabled;
    daxa_i32 post_blur_stride;
    daxa_i32 post_blur_max_width;
    daxa_i32 post_blur_atrous_iterations;
    daxa_i32 upscale_enabled;
    daxa_i32 sh_resolve_enabled;
    daxa_i32 firefly_center_blur_enabled;
    daxa_i32 pre_blur_firefly_energy_compensation_enabled;
    daxa_i32 animate_noise;

    daxa_f32 ray_percentage;

    // Guaranteed minimum fraction of geometry pixels that trace a ray every frame, independent of the
    // demand-scaled ray budget. 0.5 -> a rotating checkerboard (2 of every 4 pixels in a quad) always
    // traces, so converged pixels can never be starved below half coverage by disocclusion bursts.
    daxa_f32 min_ray_budget;

    // 0 = classic per-pixel trace (one dispatch per pixel), 1 = repacked ray-list dispatch
    // (reproject demand -> allocate -> trace-from-list -> blend). Only one path runs per frame.
    daxa_i32 use_repacked_ray_dispatch;

    // 1 = each tile's ray budget is proportional to its demand (disoccluded tiles get more rays).
    // 0 = every tile gets the same fixed budget regardless of demand (uniform ray rate per tile).
    daxa_i32 use_ray_redistribution;
};

struct RtgiRayCounters
{
    daxa_u32 total_extra_rays; // sum of (desired_rays - 1) per geometry pixel, written by reproject
    daxa_u32 ray_list_count;   // atomic write cursor filled by the allocate pass
    daxa_u32 total_geo_rays;   // number of geometry (non-sky) pixels = base ray count, written by reproject
    daxa_u32 pad0;
};

// One entry in the flat ray list built by the allocate pass.
struct RtgiRayEntry
{
    daxa_u32 packed_xy;    // (x & 0xFFFF) | ((y & 0xFFFF) << 16), half-res pixel coords
    daxa_u32 sample_index; // which sample of this pixel this entry represents
};

struct RtgiRayResult
{
    daxa_f32vec3 radiance;
    daxa_f32 t;             // raw ray hit distance; the blend pass converts it to shortness [0,1] per ray
    daxa_u32 packed_dir;    // octahedral-packed sample direction (for directional SH in the blend pass)
};

// Per-pixel ray-list offset written by the allocate pass, read by the blend pass. The ray count is
// stored separately in ray_count_image (not duplicated here).
struct RtgiPixelRayAlloc
{
    daxa_u32 ray_offset; // first index in the ray list for this pixel
};
