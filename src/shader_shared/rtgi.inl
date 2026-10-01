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
    daxa_f32 firefly_perceptual_tolerance; // perceptual-space headroom added above the neighborhood mean before a ray is clamped
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
    daxa_i32 max_temporal_samples; // history length in 1-ray frames; with RTGI_RAY_WEIGHTED_HISTORY the count may exceed it at >1 ray/frame (time window stays this long)
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

    // 1 = draw ray directions from spatiotemporal blue-noise (stbnCosDir) during tracing, 0 = plain
    // per-thread hash cosine sampling. STBN gives lower-variance, better-distributed samples per frame.
    daxa_i32 trace_use_stbn;

    // Pioneer guiding: bends every diffuse ray toward a guide direction built by a same-frame, sparse
    // pioneer trace + spatial RIS resample (rtgi_guide_resample.hlsl), instead of sampling a plain
    // cosine hemisphere. See rtgi_sample_guided_diffuse_dir in rtgi_guided_sampling.hlsl. 0 = off
    // (plain cosine/STBN sampling only), 1 = on.
    daxa_i32 pioneer_guiding_enabled;

    // How hard every ray bends toward its guide direction, only meaningful while pioneer_guiding_enabled --
    // NOT derived from how directional any individual pixel's own history reads. [0,1]: 0 = identity
    // (plain cosine hemisphere), 1 = collapsed onto the guide point. See rtgi_concentration_to_kappa in
    // rtgi_guided_sampling.hlsl for the exact mapping.
    daxa_f32 guide_concentration;

    // 0 = plain Schlick pull (rtgi_schlick_pull) -- its own density at the far side of the disc (away
    // from the guide) still fades toward 0 as concentration climbs, so at high concentration almost all
    // rays land in a narrow cone around the guide. 1 = the same Schlick density with a fixed floor mixed
    // in (rtgi_floor_pull) -- verified to converge its far-side density DOWN TO that floor and never
    // below it, so some rays keep landing across the whole disc no matter how high concentration goes.
    // See rtgi_guided_sampling.hlsl.
    daxa_i32 guide_floor_pull_enabled;

    // Max ray length (in meters/world units) for the spatial-pretrace pioneer rays (pioneer_ray_gen). The
    // pioneer trace only needs to find NEARBY bounce lighting to build a useful direction guide -- unlike
    // the main trace's effectively-unbounded rays, letting a pioneer ray travel arbitrarily far just
    // spends its budget on a direction irrelevant to the guided pixel's local neighborhood. A plain value
    // (not a graph-shape setting), read straight into ray.TMax every frame -- no rebuild trigger needed.
    daxa_f32 guide_pioneer_trace_max_distance;

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
