#pragma once

#include "daxa/daxa.inl"

// RT Lab: experimental diffuse RTGI path (rt_lab.hpp). Fully separate from the RTGI denoiser; it only reuses the RTGI
// ray shading core (rtgi_shade.hlsl) and shared helpers.
// The CPU keeps two presets of these (RtLabState, rt_lab_state.hpp) and copies the active one into
// RenderGlobalData::rt_lab_settings every frame; enabled (master switch) and reset_history are not taken from the preset.
#define RT_LAB_ATROUS_MAX_PASSES 8 // RtLabSettings::atrous_passes limit (one render timer each)
// coarse_anti_lag_enabled: how the anti-lag cap acts on the histories.
#define RT_LAB_ANTI_LAG_OFF 0
#define RT_LAB_ANTI_LAG_REDUCE_SAMPLES 1 // the stored history length is cut -> later frames converge faster on the new result
#define RT_LAB_ANTI_LAG_BLEND_ONLY 2     // RTGI style: only this frame's blend uses the cut length, the stored length stays
#define RT_LAB_ANTI_LAG_MODE_MASK 3      // the bits above select how the cut is applied (OFF / REDUCE_SAMPLES / BLEND_ONLY)
// Flag bit: which heuristic computes the cut. Unset = RTGI fast history anti-lag (variance scaling x (1 / slow-fast ratio)^2).
// Set = MSE optimal length: N* = k x fast rel variance / (debiased slow-fast rel difference)^2, k = coarse_anti_lag_variance_blend.
#define RT_LAB_ANTI_LAG_MSE_OPTIMAL 4
#define RT_LAB_RAY_FILTER_OFF 0
#define RT_LAB_RAY_FILTER_VARIANCE 1
#define RT_LAB_RAY_FILTER_PERCEPTUAL 2
#define RT_LAB_RAY_FILTER_MAX_RADIUS 3 // RtLabSettings::ray_filter_radius limit
#define RT_LAB_GRADIENT_MAX_BLUR_RADIUS 32 // RtLabSettings::temporal_gradient_blur_radius limit (gradient cells)

struct RtLabSettings
{
    // Master switch: 1 = the RT Lab path replaces the RTGI denoiser (rtgi_settings.enabled still turns RTGI on / off).
    // Graph shape.
    daxa_i32 enabled;
    // 0 = trace + denoise at full res, 1 = at half res (bilateral upsample in the resolve). Graph shape.
    daxa_i32 half_res;
    daxa_i32 animate_noise;

    // == Trace ==
    // Short range AO on hit, same as the RTGI coarse hit shading (rtgi_coarse_hit_ao): the PGI radiance of a hit is
    // scaled by (min(t, hit_ao_range) / hit_ao_range)^2, as PGI leaks at short hit distances. 0 = raw PGI radiance.
    daxa_i32 hit_ao_enabled;
    daxa_f32 hit_ao_range; // world space meters
    // Ray redistribution (rt_lab_rays.inl, simplified RTGI ray budget): every pixel wants 1 base ray + extras that fall
    // off exponentially with its reprojected history length (RTGI calc_desired_extra_rays: max_rays_per_pixel - 1 extras
    // at length 0, none at fast_convergence_samples); the frame's rays are capped at ray_budget per pixel, draining the
    // extras first down to min_ray_budget base coverage. Off: 1 ray per pixel. Graph shape: redistribution_enabled and
    // ceil(ray_budget) (ray list capacity, rt_lab_ray_capacity_per_pixel).
    daxa_i32 redistribution_enabled;
    daxa_f32 ray_budget;               // rays per lab pixel per frame (hard)
    daxa_f32 min_ray_budget;           // base ray coverage the extras never drain below (0..1)
    daxa_i32 max_rays_per_pixel;       // base included
    daxa_f32 fast_convergence_samples; // history length (samples) at which a pixel stops asking for extras
    // Temporal gradient (A-SVGF style, rt_lab_rays.inl): one ray per 4x4 cell is re-traced next frame from the exact same
    // launch data; the luminance change, blurred in screen space (distance only gaussian, temporal_gradient_blur_radius
    // cells), gives lambda = |sum delta| / sum max(old, new). Reprojected history lengths are scaled by
    // (1 - saturate(lambda x temporal_gradient_scale)) in the demand (before the ray redistribution) and the temporal pass.
    // Needs temporal. Graph shape: temporal_gradient_enabled.
    daxa_i32 temporal_gradient_enabled;
    daxa_i32 temporal_gradient_blur_radius; // cells (4 px each), gaussian sigma = radius / 2, separable, max RT_LAB_GRADIENT_MAX_BLUR_RADIUS
    daxa_f32 temporal_gradient_scale;
    // Gradient sample selection (gather pass, rt_lab_rays.hlsl). 1 = RIS over all of the cell's rays with target
    // luminance + RT_LAB_GRADIENT_RIS_UNIFORM_WEIGHT x cell mean luminance (defensive uniform part, dark rays stay
    // pickable); the re-traced delta and max are scaled by the RIS weight (sum w / M) / w_selected, so each cell estimates
    // its mean change. 0 = uniform: first ray of a pixel with rays, pixel order rotated per frame and cell.
    daxa_i32 temporal_gradient_ris;

    // == Temporal ==
    daxa_i32 temporal_enabled; // graph shape
    // 0 = "smart": RGBA16F history, short history (max_history_frames).
    // 1 = "progressive": RGBA32F history, long accumulation up to progressive_max_samples. Graph shape (history format).
    daxa_i32 temporal_progressive;
    daxa_i32 max_history_frames;
    daxa_i32 progressive_max_samples;
    // Reprojection taps whose position is further than this many pixel widths from the reprojected surface plane are
    // rejected (disocclusion).
    daxa_f32 disocclusion_plane_tolerance;
    // Reprojection taps whose face normal dot is below this are rejected.
    daxa_f32 disocclusion_normal_threshold;
    // 1 = reproject the history with a 12-tap Catmull-Rom (4x4 footprint without corners) when all 12 taps lie on this
    // surface, else bilinear; 0 = always bilinear. Bilinear blurs the history a little every frame under motion.
    daxa_i32 bicubic_reprojection;
    // History length (sample count) reprojection of partially valid footprints:
    // 1 = occluder aware soft normalize (tido RTGI, rtgi_reproject_count_occluder_aware): normalized mean of the valid taps x
    //     (1 - occluder coverage)^0.34, occluder = failed tap more than 2 pixel widths IN FRONT of the expected point
    //     (it hid the point last frame); failed taps behind it / sky (edge overhang) are not penalized.
    // 0 = old RTGI / ReLAX style soft normalize: sum(w * count) / W^0.66, every partial footprint is penalized.
    // Both keep the max(min(max valid tap count, 4), ...) floor.
    daxa_i32 occluder_aware_soft_normalize;
    // Variance guide (rt_lab_temporal.hlsl estimates, rt_lab_blur.hlsl uses): the temporal pass estimates the per sample
    // rgb variance from the 5x5 raw samples on this surface every frame and accumulates it like the color. The fine
    // spatial passes treat center and tap as estimates of their means, N(mean, variance / effective sample count), and
    // weight the tap by how likely both share a mean: exp(-z^2 / (2 k^2)), z^2 = mean over rgb of
    // (mean_c - mean_t)^2 / (var_c / n_c + var_t / n_t), k = variance_guide_sigmas. Few samples = loose test.
    daxa_i32 variance_guide_enabled;
    daxa_f32 variance_guide_sigmas;
    // Split (A/B) history for a per-pixel error estimate of the accumulated mean, |A - B| / 2 (rt_lab_temporal.hlsl).
    // Costs two more history images. Graph shape.
    daxa_i32 split_history_enabled;

    // == Spatial ==
    // Separable bilateral gaussian (horizontal + vertical), geometric edge stopping only. Graph shape.
    daxa_i32 spatial_enabled;
    // 0 = separable gaussian (horizontal + vertical, blur_radius), 1 = a-trous: atrous_passes passes of a 5x5 B3 spline
    // kernel with step 1, 2, 4, 8, ... Same geometric edge stopping. Graph shape.
    daxa_i32 spatial_mode;
    daxa_i32 atrous_passes; // 1..RT_LAB_ATROUS_MAX_PASSES, graph shape
    daxa_i32 blur_radius;          // taps per side per pass, gaussian sigma = radius / 2
    daxa_f32 blur_plane_tolerance; // plane distance in pixel widths at which a tap's weight reaches 0
    daxa_f32 blur_normal_power;    // tap weight *= saturate(dot(n, n_tap))^power
    // Fine spatial filter sample count budget (0 = off): every pass accepts its taps ring by ring from the center
    // outwards; the ring that would push the effective sample count over this is scaled so the pass reaches it exactly,
    // the rest is dropped. A pass whose center already holds this many samples returns the center. Fine filter only.
    daxa_f32 fine_target_samples;

    // == Coarse spatial fallback ==
    // A second, sloppy spatial filter (a-trous, coarse_atrous_passes) on the same temporal output, blended in at the
    // resolve where the fine spatial filter gathered few effective samples (edges, thin geometry, disocclusions).
    // Edge stopping: no plane test, a smooth relative distance test (world distance vs what a camera facing surface at
    // that screen offset would give, exp(-(max(d_rel - 1, 0) / coarse_distance_tolerance)^2)) and a smooth normal test
    // ((1 + dot) / 2)^coarse_normal_power (0 at 180 degrees). Graph shape: coarse_enabled, coarse_atrous_passes.
    daxa_i32 coarse_enabled;
    daxa_i32 coarse_atrous_passes;
    daxa_f32 coarse_distance_tolerance;
    daxa_f32 coarse_normal_power;
    // Variance guide of the coarse filter: same test as variance_guide_* on the mean rgb of the SH-Y + CoCg, own
    // (wider) tolerance in sigmas. Independent of variance_guide_enabled.
    daxa_i32 coarse_variance_guide_enabled;
    daxa_f32 coarse_variance_guide_sigmas;
    // Top up (rt_lab_coarse_top_up): blend t = the smallest t with which lerp(fine, coarse, t) reaches this effective
    // sample count, treating fine / coarse as estimates with variance 1 / N: N_eff = 1 / ((1 - t)^2 / N_fine + t^2 / N_coarse).
    // Unreachable: the minimum variance mix. Capped by coarse_max_blend.
    daxa_f32 coarse_target_samples;
    daxa_f32 coarse_max_blend;

    // == Coarse temporal anti-lag (rt_lab_anti_lag.inl) ==
    // The rays' luminance per 4x4 cell, blurred a lot (edge aware gaussian, coarse_anti_lag_blur_radius cells), drives an
    // RTGI style fast history per cell (fast mean + fast relative variance over coarse_anti_lag_history_frames frames) next
    // to a slow coarse mean (EMA capped at the denoiser's own history length, max_history_frames / progressive_max_samples).
    // RTGI anti-lag: when the slow mean diverges from the fast mean (lighting changed) the history length is cut to
    // N x (1 + sqrt(fast var) x coarse_anti_lag_variance_blend)^2 x (1 / max(1, slow / fast))^2 (fast variance = noise earns
    // trust back) -> caps the coarse slow history and the pixel histories (never the color). Needs temporal.
    // Graph shape: coarse_anti_lag_enabled.
    daxa_i32 coarse_anti_lag_enabled;     // RT_LAB_ANTI_LAG_*
    daxa_i32 coarse_anti_lag_blur_radius;    // cells (4 px), gaussian sigma = radius / 2, separable, max RT_LAB_GRADIENT_MAX_BLUR_RADIUS
    daxa_i32 coarse_anti_lag_history_frames; // fast history window (frames), RTGI temporal_fast_history_frames
    daxa_f32 coarse_anti_lag_variance_blend; // RTGI temporal_variance_fast_history_blend

    // == Ray filter (rt_lab_rays.hlsl entry_rt_lab_ray_filter) ==
    // After the trace, one thread per ray: statistics of all rays of the (2 radius + 1)^2 neighbor pixels on the same
    // surface (the ray itself excluded), the ray clamped against them hue preserving. Mode RT_LAB_RAY_FILTER_*:
    // variance (luminance mean + sigmas x std, optionally also mean - sigmas x std) or perceptual (RTGI firefly clamp
    // style: per channel exp(mean log + tolerance)). The pixel means, temporal SH and coarse clamp use the filtered rays,
    // the temporal gradient the raw ones. Graph shape: mode != off.
    daxa_i32 ray_filter_mode;
    daxa_i32 ray_filter_radius;     // 2 = 5x5, max RT_LAB_RAY_FILTER_MAX_RADIUS
    daxa_f32 ray_filter_sigmas;     // variance mode
    daxa_i32 ray_filter_clamp_low;  // variance mode: also raise rays below mean - sigmas x std
    daxa_f32 ray_filter_tolerance;  // perceptual mode: log units above the neighbors' log mean

    // Set by the CPU for single frames (convergence error estimate phase starts): the temporal pass treats every pixel
    // as disoccluded. Not part of the presets.
    daxa_i32 reset_history;
    // Effective sample count metric (not part of the presets, RtLabState::sample_count_*): the temporal history length
    // clamped to this before the spatial passes accumulate it (0 = no clamp, 1 = spatial samples only), and the
    // RT_LAB_SPATIAL_SAMPLE_COUNT debug view heatmap maximum. Metric only, the filtering is unchanged.
    daxa_f32 sample_count_input_clamp;
    daxa_f32 sample_count_debug_max;
};
