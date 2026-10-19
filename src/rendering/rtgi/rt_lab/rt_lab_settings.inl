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
// Flag bits: noise estimate of the fast history (.y), used by both heuristics. Neither set = EMA of the squared relative
// deviation (relative variance; a light change inflates it quadratically).
// NOISE_MAD: EMA of the relative ABSOLUTE deviation, sigma = MAD x sqrt(pi / 2) (inflates linearly, recovers faster).
// NOISE_SPATIAL: predicted from this frame's spatial ray variance over the blur footprint / effective ray count of the
// blur weights ((sum w n)^2 / sum w^2 n), EMA'd; no temporal deviations at all, so a light change cannot inflate it.
#define RT_LAB_ANTI_LAG_NOISE_MAD 8
#define RT_LAB_ANTI_LAG_NOISE_SPATIAL 16
// KALMAN: replaces fast history / MSE / RTGI heuristic / TAA clip with a per cell Kalman filter of the log coarse mean,
// predicted with the temporal gradient's correction, corrects the pixel histories (ratio), cuts only the uncorrectable
// rest (rt_lab_anti_lag.hlsl, parameters RT_LAB_ANTI_LAG_KALMAN_* there).
#define RT_LAB_ANTI_LAG_KALMAN 32
// temporal_gradient_cap_mode bits. 0 = old behaviour: reprojected lengths x (1 - saturate(lambda x temporal_gradient_scale)).
// Any bit set: the lengths are CAPPED (same frame, no compounding) at the smallest enabled cap, floored at
// temporal_gradient_cap_min; lambda unscaled.
//   MSE:    N* = temporal_gradient_cap_k x v / (lambda^2 - RT_LAB_GRADIENT_CAP_LAMBDA_FLOOR^2), v = per sample perceptual
//           variance (variance laziness estimate, floor variance_laziness_perceptual_floor). k folds in the spatial filter's
//           noise reduction (the pixel's own samples are not the output's noise), so k << 1.
//   SETTLE: the remaining lag lambda x (N / (N + 1))^K falls below temporal_gradient_settle_tolerance within
//           K = temporal_gradient_settle_frames frames: N = 1 / ((lambda / tolerance)^(1 / K) - 1), only when lambda > tolerance.
#define RT_LAB_GRADIENT_CAP_MSE 1
#define RT_LAB_GRADIENT_CAP_SETTLE 2
#define RT_LAB_GRADIENT_CAP_LAMBDA_FLOOR 0.01f
//   KALMAN: replaces the others (rt_lab_rays.inl "Kalman temporal gradient"): the gradient MEASURES the change (log ratio
//           of the re-traced sums + its variance), a per cell Kalman filter tracks it (innovation gate = instant reset on
//           shocks), the history is CORRECTED by the Wiener shrunk change and only the remaining uncertainty L cuts it:
//           N' = N / (1 + N x temporal_gradient_kalman_k x L / v), v = per sample perceptual variance.
#define RT_LAB_GRADIENT_CAP_KALMAN 4
#define RT_LAB_RAY_FILTER_OFF 0
#define RT_LAB_RAY_FILTER_VARIANCE 1
#define RT_LAB_RAY_FILTER_PERCEPTUAL 2
#define RT_LAB_RAY_FILTER_MAX_RADIUS 3 // RtLabSettings::ray_filter_radius limit
// RtLabSettings::ray_length_guide_mode (UI tristate: 0 off, 1 radius scale, 2 radius weighting; tested as bits).
#define RT_LAB_RAY_LENGTH_GUIDE_DISTANCE 1   // radius: donors farther than the accumulated ray length radius lose weight
#define RT_LAB_RAY_LENGTH_GUIDE_SIMILARITY 2 // similarity: donors with a different accumulated radius lose weight
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
    // Variance laziness (demand pass, needs temporal): a pixel whose per sample PERCEPTUAL variance v is low needs fewer
    // samples to reach variance_laziness_target_error (perceptual std of the mean): N = v / target^2. Perceptual = log
    // space with an exposure relative floor: std of log(L + floor) ~ sigma / (mean + floor), floor =
    // variance_laziness_perceptual_floor x display white (exposed). Bright pixels: relative error as before; pixels darker
    // than the floor: their error is barely visible, they get lazy. Luma variance = last frame's variance guide estimate. The convergence target becomes clamp(N, variance_laziness_min_samples,
    // fast_convergence_samples). Applied to the reprojected history length BEFORE the temporal gradient cut (length x
    // fast_convergence_samples / N), so a gradient cut still asks for the full extras. Not graph shape.
    daxa_i32 variance_laziness_enabled;
    daxa_f32 variance_laziness_target_error;
    daxa_f32 variance_laziness_min_samples;
    daxa_f32 variance_laziness_perceptual_floor;
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
    // 1 = the base rays (ray 0 of the pixels in the guaranteed base share, rate min_ray_budget) are recorded and re-traced
    // next frame (lab res records, the samples are summed per cell with atomics) instead of one ray per
    // RT_LAB_GRADIENT_CELL^2 cell. Graph shape.
    daxa_i32 temporal_gradient_all_base_rays;
    daxa_i32 temporal_gradient_dilation_radius; // cells, 0 = off: weighted max dilation of lambda after the blur (reach without dilution), graph shape
    daxa_f32 temporal_gradient_scale;
    daxa_i32 temporal_gradient_cap_mode;          // RT_LAB_GRADIENT_CAP_* bits, 0 = scale mode
    daxa_f32 temporal_gradient_cap_k;             // MSE cap scale
    daxa_f32 temporal_gradient_cap_min;           // cap floor (samples)
    daxa_f32 temporal_gradient_settle_frames;     // SETTLE: K
    daxa_f32 temporal_gradient_settle_tolerance;  // SETTLE: relative lag tolerance
    // History rescale: the reprojected histories (color, split A/B, SH / CoCg, variance) are multiplied by the gradient's
    // luminance ratio sum new / sum old of the re-traced rays (blurred, NOT dilated). With one change direction inside
    // the footprint it follows from the signed lambda: 1 / (1 - lambda) brightening, 1 - lambda darkening (clamped at
    // RT_LAB_GRADIENT_RESCALE_MAX_LAMBDA); ratio^temporal_gradient_rescale_strength. Mixed directions: lambda shrinks,
    // the rescale with it. The history jumps to the new lighting without losing samples.
    daxa_i32 temporal_gradient_rescale;
    daxa_f32 temporal_gradient_rescale_strength;
    // RT_LAB_GRADIENT_RELATIVE_*. SAMPLES (0) = A-SVGF: lambda = |sum w delta| / sum w max relative to the re-traced
    // samples themselves (a dark change can cut a bright pixel's history). PIXEL (1) = the blur + dilation also carry the
    // ABSOLUTE exposed luminance change (weighted mean delta of the re-traced samples), made relative only per pixel
    // against the pixel's own previous history luminance L (nearest previous pixel): lambda = |delta| /
    // max(L + max(delta, 0), floor), signed the same for the rescale (= (L + delta) / L exactly). MIN (2) = min of both
    // (smaller |signed| for the rescale). Not graph shape.
    daxa_i32 temporal_gradient_pixel_relative;
    // Temporal smoothing of the blurred gradient sums (sum w delta, sum w max, sum w samples) per cell, reprojected to the
    // nearest previous cell (plane test): sums = lerp(previous, current, alpha) BEFORE lambda. 1 = off. lambda then is
    // the sustained per frame change over ~1 / alpha frames: the sparse re-trace's frame to frame noise averages out
    // (signed, before the |.|), a real step still shows alpha x lambda in its first frame. Not graph shape.
    daxa_f32 temporal_gradient_history_alpha;
    // Shock bypass of the smoothing: alpha ramps to 1 (= reset to this frame's sums) as THIS frame's lambda goes from
    // shock / 2 to shock, so single frame flashes (black -> bright -> black) still cut at once in both directions.
    daxa_f32 temporal_gradient_history_shock;
    // lambda deadzone d: lambda' = saturate((lambda - d) / (1 - d)) before the scale / cap (not the rescale): removes
    // small, noise level lambdas, keeps big ones almost unchanged (unlike a looser settle tolerance). 0 = off.
    daxa_f32 temporal_gradient_deadzone;
    // KALMAN cap mode: process noise of the per frame log change (how fast the change rate itself may vary), innovation
    // gate in sigmas (beyond = reset to this frame's measurement), information loss scale k (folds in the spatial filter's
    // noise reduction: the output's noise is the history's / spatial gain, so a bias matters k x more than vs v / N).
    daxa_f32 temporal_gradient_kalman_q;
    daxa_f32 temporal_gradient_kalman_gate;
    daxa_f32 temporal_gradient_kalman_k;

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
    // Variance guide variants (fine AND coarse guide, rt_lab_variance_guide_likelihood / _count, rt_lab_shared.hlsl):
    // heavy_tail: Student-t weight (1 + z^2 / (nu k^2))^(-(nu + 1) / 2) instead of the gaussian (polynomial tail: a noisy z no
    //             longer flips a tap between fully in and gone -> less splotchy); nu -> infinity = gaussian, 1 = Cauchy.
    // count_cap:  the effective sample count in the standard error var / n is capped at variance_guide_count_cap (the
    //             spatially compounded count overstates the independent samples -> z inflates in later passes).
    daxa_i32 variance_guide_heavy_tail;
    daxa_f32 variance_guide_tail_nu;
    daxa_i32 variance_guide_count_cap_enabled;
    daxa_f32 variance_guide_count_cap;
    // Gradient aware fine variance guide (rt_lab_blur.hlsl): each fine pass fits the local rgb gradient per screen axis
    // (weighted least squares over its own taps, geometric weights, significance shrinkage) and compares every tap with
    // the center's mean + gradient x offset instead of the plain center mean ("plane distance test in color space").
    daxa_i32 variance_guide_gradient;
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
    // Firefly energy weighting (RTGI pre-blur iteration 0 style, but post temporal): with the ray filter on, the temporal
    // pass accumulates the RAW (unclamped) mean luminance next to the filtered color with the same blend; factor =
    // accumulated raw / filtered luminance (>= 1, energy_pre / energy_post). The first fine and the first coarse spatial
    // pass multiply each tap's (and the center's) color weight by its factor: the clipped energy is spread over the
    // footprint as a bounded blob instead of discarded. Sample counts are not scaled. Not graph shape.
    daxa_i32 firefly_weight_enabled;

    // == Ray length guide (fine spatial filter, rt_lab_blur.hlsl) ==
    // The trace stores every ray's hit distance t (miss = RT_LAB_RAY_T_MAX). RTGI AO guide calculation: per ray
    // calc_ray_shortness = 1 - min(1, (t / R)^2), R = ray_length_guide_radius_px RTGI half res pixel widths
    // (= RTGI max_visibility_pixel_range, default 20), per pixel mean, mean over the 5x5 neighborhood on the same surface
    // (plane + normal test, like the variance estimate), ao guide = 1 - sqrt(mean shortness), accumulated like the color
    // (.x ao guide, .y mean shortness). Guide radius r = ao guide x R. Modes (bits
    // RT_LAB_RAY_LENGTH_GUIDE_*): DISTANCE: tap weight x exp(-(max(d / r - 1, 0) / distance_falloff)^2), d = world
    // distance center -> tap, r floored at one pixel width; SIMILARITY: x exp(-(log2(r_center / r_tap) /
    // similarity_tolerance)^2). Rejected taps are not counted (like the variance guide). Debug RT_LAB_RAY_LENGTH_GUIDE.
    // Graph shape: mode != 0.
    daxa_i32 ray_length_guide_mode;
    daxa_f32 ray_length_guide_radius_px;
    daxa_f32 ray_length_guide_distance_falloff;
    daxa_f32 ray_length_guide_similarity_tolerance;

    // Set by the CPU for single frames (convergence error estimate phase starts): the temporal pass treats every pixel
    // as disoccluded. Not part of the presets.
    daxa_i32 reset_history;
    // Effective sample count metric (not part of the presets, RtLabState::sample_count_*): the temporal history length
    // clamped to this before the spatial passes accumulate it (0 = no clamp, 1 = spatial samples only), and the
    // RT_LAB_SPATIAL_SAMPLE_COUNT debug view heatmap maximum. Metric only, the filtering is unchanged.
    daxa_f32 sample_count_input_clamp;
    daxa_f32 sample_count_debug_max;
};
