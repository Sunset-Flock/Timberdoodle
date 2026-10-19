#pragma once

#include "rt_lab_temporal.inl"

#include "rt_lab_shared.hlsl"
#include "rt_lab_reproject.hlsl"
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtLabTemporalPush rt_lab_temporal_push;

func rt_lab_temporal_debug_mode_active(int mode) -> bool
{
    // The split error views are drawn post-spatial by the resolve.
    return mode == DEBUG_DRAW_MODE_RT_LAB_SAMPLE_COUNT || mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_SAMPLE_COUNT ||
           mode == DEBUG_DRAW_MODE_RT_LAB_VARIANCE || mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG ||
           mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_MEAN || mode == DEBUG_DRAW_MODE_RT_LAB_HISTORY_CUTS ||
           mode == DEBUG_DRAW_MODE_RT_LAB_RAY_LENGTH_GUIDE;
}

func rt_lab_temporal_write_debug(uint2 dtid, float3 color)
{
    let push = rt_lab_temporal_push;
    // The lab may run at half res: the debug image is full res.
    const uint resolution_scale = max(push.attach.globals.settings.render_target_size.x / push.size.x, 1u);
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend), resolution_scale);
}

// Surface test of one previous frame lab pixel (rt_lab_reproject_tap).
func rt_lab_temporal_tap(int2 tap, float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size, out bool occluder) -> float
{
    let push = rt_lab_temporal_push;
    return rt_lab_reproject_tap(push.attach.lab_depth_history.get(), push.attach.lab_face_normal_history.get(),
        &push.attach.globals.view_camera_prev_frame, &push.attach.globals.rt_lab_settings, int2(push.size), tap, position, normal,
        rcp_pixel_width_ws, inv_size, occluder);
}

func rt_lab_temporal_tap_surface_weight(int2 tap, float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size) -> float
{
    bool occluder;
    return rt_lab_temporal_tap(tap, position, normal, rcp_pixel_width_ws, inv_size, occluder);
}

// 1D Catmull-Rom weights of the taps at -1, 0, 1, 2 for fractional position f.
func rt_lab_catrom_weights(float f) -> float4
{
    const float f2 = f * f;
    const float f3 = f2 * f;
    return float4(
        -0.5f * f3 + f2 - 0.5f * f,
        1.5f * f3 - 2.5f * f2 + 1.0f,
        -1.5f * f3 + 2.0f * f2 + 0.5f * f,
        0.5f * f3 - 0.5f * f2);
}

// 5x5 (RT_LAB_VARIANCE_RADIUS 2) neighborhood of the per frame variance estimate.
#define RT_LAB_VARIANCE_RADIUS 2

// Per sample rgb variance of this frame's raw samples in the 5x5 neighborhood on this surface (same plane + normal test as
// the reprojection). Each pixel holds the mean of its n rays: grouped estimate sum n_j (x_j - mean)^2 / (pixels - 1),
// mean = sum n_j x_j / sum n_j (unbiased for equal per sample variance; = the plain (n - 1) estimate at 1 ray per pixel).
// Fewer than 2 pixels with rays: no estimate, the sample itself squared stands in (relative std 1, loose rather than
// over confident).
func rt_lab_temporal_spatial_variance(uint2 dtid, float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size, float3 center_sample) -> float3
{
    let push = rt_lab_temporal_push;
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float3 sum_sq = float3(0.0f, 0.0f, 0.0f);
    float samples = 0.0f; // rays
    float n = 0.0f;       // pixels
    [unroll] for (int y = -RT_LAB_VARIANCE_RADIUS; y <= RT_LAB_VARIANCE_RADIUS; ++y)
    {
        [unroll] for (int x = -RT_LAB_VARIANCE_RADIUS; x <= RT_LAB_VARIANCE_RADIUS; ++x)
        {
            const int2 tap = int2(dtid) + int2(x, y);
            if (any(tap < 0) || any(tap >= int2(push.size)))
            {
                continue;
            }
            const float tap_depth = push.attach.lab_depth.get()[tap];
            if (tap_depth == 0.0f)
            {
                continue;
            }
            const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[tap]);
            const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
            if (dot(normal, tap_normal) < settings.disocclusion_normal_threshold ||
                rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, settings.disocclusion_plane_tolerance) <= 0.0f)
            {
                continue;
            }
            const float4 value = push.attach.lab_raw_diffuse.get()[tap];
            if (value.a <= 0.0f)
            {
                continue; // no ray this frame
            }
            sum += value.rgb * value.a;
            sum_sq += value.rgb * value.rgb * value.a;
            samples += value.a;
            n += 1.0f;
        }
    }
    if (n < 2.0f)
    {
        return center_sample * center_sample;
    }
    const float3 mean = sum / samples;
    return max(sum_sq - samples * mean * mean, 0.0f) / (n - 1.0f);
}

// 5x5 (RT_LAB_RAY_LENGTH_RADIUS 2) neighborhood of the ray length guide estimate.
#define RT_LAB_RAY_LENGTH_RADIUS 2

// Ray length guide (RtLabSettings::ray_length_guide_*), RTGI AO guide style: every ray's hit distance scaled and clamped
// to the RTGI relative max distance of its pixel (calc_ray_shortness: 1 - min(1, (t / R)^2), R =
// rt_lab_ray_length_max_distance), the pixel's mean shortness, the mean over this frame's pixels with rays in the 5x5
// neighborhood on this surface (same plane + normal test as the reprojection; = the RTGI pre-filter's neighborhood mean).
// Returns (ao guide = 1 - sqrt(mean shortness), mean shortness); pixels = 0: no estimate.
func rt_lab_temporal_ray_length(uint2 dtid, float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size, out float pixels) -> float2
{
    let push = rt_lab_temporal_push;
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    float shortness_sum = 0.0f;
    pixels = 0.0f;
    [unroll] for (int y = -RT_LAB_RAY_LENGTH_RADIUS; y <= RT_LAB_RAY_LENGTH_RADIUS; ++y)
    {
        [unroll] for (int x = -RT_LAB_RAY_LENGTH_RADIUS; x <= RT_LAB_RAY_LENGTH_RADIUS; ++x)
        {
            const int2 tap = int2(dtid) + int2(x, y);
            if (any(tap < 0) || any(tap >= int2(push.size)))
            {
                continue;
            }
            const float tap_depth = push.attach.lab_depth.get()[tap];
            if (tap_depth == 0.0f)
            {
                continue;
            }
            const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[tap]);
            const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
            if (dot(normal, tap_normal) < settings.disocclusion_normal_threshold ||
                rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, settings.disocclusion_plane_tolerance) <= 0.0f)
            {
                continue;
            }
            const uint2 alloc = push.attach.ray_alloc.get()[tap];
            if (alloc.y == 0u)
            {
                continue;
            }
            // The tap pixel's own RTGI half res pixel width (calc_ray_shortness multiplies it by the pixel range).
            const float tap_pixel_width = rtgi_half_res_pixel_width_ws(push.attach.globals.settings.render_target_size_inv * 2.0f, camera.near_plane, tap_depth);
            float pixel_shortness = 0.0f;
            for (uint k = 0u; k < alloc.y; ++k)
            {
                pixel_shortness += calc_ray_shortness(push.attach.ray_lengths[alloc.x + k], tap_pixel_width, settings.ray_length_guide_radius_px);
            }
            shortness_sum += pixel_shortness / float(alloc.y);
            pixels += 1.0f;
        }
    }
    const float shortness = pixels > 0.0f ? shortness_sum / pixels : 0.0f;
    return float2(1.0f - sqrt(shortness), shortness);
}

// Blend factor of new_count new samples into a history of reprojected_length, the new length capped at max_length:
// new_count / new length (1 sample: 1 / length as before). 0 without new samples.
func rt_lab_temporal_blend_factor(float reprojected_length, float new_count, float max_length, out float length) -> float
{
    length = min(reprojected_length + new_count, max_length);
    return new_count > 0.0f ? min(new_count / max(length, 1e-3f), 1.0f) : 0.0f;
}

// Like rt_lab_temporal_blend, but the blend uses the length reprojected.a x blend_keep (anti-lag, blend only mode) while
// the stored length grows from the full reprojected.a.
func rt_lab_temporal_blend_kept(float4 reprojected, float3 new_sample, float new_count, float max_length, float blend_keep) -> float4
{
    float length;
    const float blend = rt_lab_temporal_blend_factor(reprojected.a * blend_keep, new_count, max_length, length);
    return float4(lerp(reprojected.rgb, new_sample, blend), min(reprojected.a + new_count, max_length));
}

// Blends the mean of new_count new samples into one reprojected history (.rgb value, .a length) capped at max_length.
func rt_lab_temporal_blend(float4 reprojected, float3 new_sample, float new_count, float max_length) -> float4
{
    float length;
    const float blend = rt_lab_temporal_blend_factor(reprojected.a, new_count, max_length, length);
    return float4(lerp(reprojected.rgb, new_sample, blend), length);
}

// Temporal accumulation, no tricks: reproject the lab history with the previous camera (bilinear footprint), reject
// taps that are not on the same surface (plane distance + face normal), blend the frame's n rays in by n / history length.
// "Smart" mode caps the length at max_history_frames (RGBA16F history), "progressive" at progressive_max_samples
// (RGBA32F history).
// Split history: A / B reproject with the same weights, A takes the new sample on even frames, B on odd frames, each
// capped at half the max length (same time window as the full history).
[shader("compute")]
[numthreads(RT_LAB_TEMPORAL_X, RT_LAB_TEMPORAL_Y, 1)]
func entry_rt_lab_temporal(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_temporal_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    CameraInfo* prev_camera = &push.attach.globals.view_camera_prev_frame;
    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    const bool split = push.split_enabled != 0u;

    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        push.attach.accumulated.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        push.attach.variance.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (push.sh_enabled != 0u)
        {
            push.attach.sh_y_accumulated.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            push.attach.cocg_accumulated.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if (split)
        {
            push.attach.split_a_accumulated.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            push.attach.split_b_accumulated.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            push.attach.split_difference.get()[dtid] = float2(0.0f, 0.0f);
        }
        if (rt_lab_temporal_debug_mode_active(debug_mode))
        {
            rt_lab_temporal_write_debug(dtid, float3(0.0f, 0.0f, 0.0f));
        }
        return;
    }

    const float2 size = float2(push.size);
    const float2 inv_size = rcp(size);
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, inv_size), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, size.y, depth));

    // Where this surface point was on screen last frame.
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    const Bilinear bilinear = get_bilinear_filter(prev_uv, size);

    float4 tap_weights = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 occluder_taps = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 taps[4];
    float4 taps_a[4];
    float4 taps_b[4];
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        taps[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        taps_a[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        taps_b[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
        bool occluder;
        const float weight = rt_lab_temporal_tap(tap, position, normal, rcp_pixel_width_ws, inv_size, occluder);
        occluder_taps[i] = occluder ? 1.0f : 0.0f;
        if (weight <= 0.0f)
        {
            continue;
        }
        const float4 value = push.attach.history.get()[tap];
        // Uninitialized / broken history (first frames after a graph rebuild) counts as a disocclusion.
        if (weight <= 0.0f || any(!isfinite(value)) || value.a < 0.0f)
        {
            continue;
        }
        taps[i] = value;
        tap_weights[i] = weight;
        if (split)
        {
            const float4 value_a = push.attach.split_a_history.get()[tap];
            const float4 value_b = push.attach.split_b_history.get()[tap];
            if (all(isfinite(value_a)) && value_a.a >= 0.0f)
            {
                taps_a[i] = value_a;
            }
            if (all(isfinite(value_b)) && value_b.a >= 0.0f)
            {
                taps_b[i] = value_b;
            }
        }
    }
    const float4 weights = get_bilinear_custom_weights(bilinear, tap_weights);
    const float weight_sum = dot(weights, 1.0f);
    const bool has_history = weight_sum > 1e-3f && settings.reset_history == 0;

    const float max_length = rt_lab_max_history_length(&push.attach.globals.rt_lab_settings);
    // Temporal gradient: scale mode = the reprojected lengths shrink by (1 - lambda x scale), like in the ray demand; cap
    // mode = capped below (rt_lab_gradient_history_cap). .y = signed undilated lambda (history rescale).
    const bool gradient_cap = push.gradient_bound != 0u && settings.temporal_gradient_cap_mode != 0;
    // temporal_gradient_pixel_relative: made relative to this pixel's own previous history luminance (same as the demand).
    const float2 gradient = push.gradient_bound != 0u
        ? rt_lab_gradient_pixel_relative(settings,
              rt_lab_temporal_gradient_full(push.attach.temporal_gradient.get(), camera, push.size, dtid, position, normal, rcp_pixel_width_ws,
                  rt_lab_gradient_packed(settings)),
              push.attach.history.get(), &push.attach.globals.view_camera_prev_frame, push.size, position, push.attach.globals.exposure)
        : float2(0.0f, 0.0f);
    // .y (rescale) stays raw. Kalman: .x = the information loss L, no deadzone.
    const float gradient_lambda = (settings.temporal_gradient_cap_mode & RT_LAB_GRADIENT_CAP_KALMAN) != 0 ? max(gradient.x, 0.0f) : rt_lab_gradient_deadzone(settings, gradient.x);
    float gradient_keep = push.gradient_bound != 0u && !gradient_cap ? 1.0f - saturate(gradient_lambda * settings.temporal_gradient_scale) : 1.0f;
    const float split_max_length = max(max_length * 0.5f, 1.0f);
    // Bilinear results first, so the per-tap values are dead before the (register hungry) bicubic below.
    float4 reprojected = float4(0.0f, 0.0f, 0.0f, 0.0f); // .a = 0: disocclusion
    float4 split_a = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 split_b = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (has_history)
    {
        reprojected = apply_bilinear_custom_weights(taps[0], taps[1], taps[2], taps[3], weights);
        // History lengths: occluder aware soft normalize (tido RTGI) or soft normalize (old RTGI / ReLAX style).
        const float occluder_coverage = dot(get_bilinear_custom_weights(bilinear, float4(1.0f, 1.0f, 1.0f, 1.0f)), occluder_taps);
        const bool occluder_aware = settings.occluder_aware_soft_normalize != 0;
        reprojected.a = min(rt_lab_reproject_count(float4(taps[0].a, taps[1].a, taps[2].a, taps[3].a), weights, occluder_coverage, occluder_aware), max_length) * gradient_keep;
        if (split)
        {
            split_a = apply_bilinear_custom_weights(taps_a[0], taps_a[1], taps_a[2], taps_a[3], weights);
            split_b = apply_bilinear_custom_weights(taps_b[0], taps_b[1], taps_b[2], taps_b[3], weights);
            split_a.a = min(rt_lab_reproject_count(float4(taps_a[0].a, taps_a[1].a, taps_a[2].a, taps_a[3].a), weights, occluder_coverage, occluder_aware), split_max_length) * gradient_keep;
            split_b.a = min(rt_lab_reproject_count(float4(taps_b[0].a, taps_b[1].a, taps_b[2].a, taps_b[3].a), weights, occluder_coverage, occluder_aware), split_max_length) * gradient_keep;
        }
        if (gradient_cap && reprojected.a > 0.0f)
        {
            // Same cap as the ray demand (same inputs); the split lengths shrink by the same ratio.
            const float perceptual_variance = rt_lab_previous_perceptual_variance(push.attach.history.get(), push.attach.variance_history.get(),
                &push.attach.globals.view_camera_prev_frame, settings, push.size, position, push.attach.globals.exposure);
            gradient_keep = min(1.0f, rt_lab_gradient_history_cap(settings, gradient_lambda, perceptual_variance, reprojected.a) / reprojected.a);
            reprojected.a *= gradient_keep;
            split_a.a *= gradient_keep;
            split_b.a *= gradient_keep;
        }
    }

    // Bicubic reprojection: 12-tap Catmull-Rom over the 4x4 footprint without its corners (NRD ReBLUR), only when all 12
    // taps lie on this surface, else the bilinear results above stay. Same filter for the split histories (linear, so
    // the split difference stays the error estimate). Clamped >= 0 (negative lobes). The lengths keep the bilinear value.
    // On a texel center (fraction 0 or 1) Catmull-Rom equals bilinear exactly: skipped.
    // Two lean loops (validity, then accumulation with reloads) instead of one: keeps the register peak low, the
    // shader's register count (occupancy) is set by its peak even when this branch is not taken.
    const float2 fraction_distance = abs(bilinear.weights - round(bilinear.weights));
    bool bicubic = has_history && settings.bicubic_reprojection != 0 && any(fraction_distance > 1e-3f);
    if (bicubic)
    {
        [loop] for (int t = 0; t < 16 && bicubic; ++t)
        {
            const int x = t & 3;
            const int y = t >> 2;
            if ((x == 0 || x == 3) && (y == 0 || y == 3))
            {
                continue; // corner
            }
            const bool inner = x >= 1 && x <= 2 && y >= 1 && y <= 2;
            // The inner taps are the bilinear taps: already tested.
            bicubic = inner
                ? tap_weights[uint(x - 1) + 2u * uint(y - 1)] > 0.0f
                : rt_lab_temporal_tap_surface_weight(int2(bilinear.origin) + int2(x - 1, y - 1), position, normal, rcp_pixel_width_ws, inv_size) > 0.0f;
        }
    }
    if (bicubic)
    {
        const float4 weights_x = rt_lab_catrom_weights(bilinear.weights.x);
        const float4 weights_y = rt_lab_catrom_weights(bilinear.weights.y);
        float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f); // .w = weight sum
        float3 sum_a = float3(0.0f, 0.0f, 0.0f);
        float3 sum_b = float3(0.0f, 0.0f, 0.0f);
        [loop] for (int t = 0; t < 16; ++t)
        {
            const int x = t & 3;
            const int y = t >> 2;
            if ((x == 0 || x == 3) && (y == 0 || y == 3))
            {
                continue; // corner
            }
            const int2 tap = int2(bilinear.origin) + int2(x - 1, y - 1);
            const float weight = weights_x[x] * weights_y[y];
            sum += float4(push.attach.history.get()[tap].rgb * weight, weight);
            if (split)
            {
                sum_a += push.attach.split_a_history.get()[tap].rgb * weight;
                sum_b += push.attach.split_b_history.get()[tap].rgb * weight;
            }
        }
        const float3 value = sum.rgb / sum.w;
        const float3 value_a = sum_a / sum.w;
        const float3 value_b = sum_b / sum.w;
        // Broken history (non finite) anywhere in the footprint: keep bilinear.
        if (sum.w > 1e-4f && all(isfinite(value)) && all(isfinite(value_a)) && all(isfinite(value_b)))
        {
            reprojected.rgb = max(value, 0.0f);
            split_a.rgb = max(value_a, 0.0f);
            split_b.rgb = max(value_b, 0.0f);
        }
    }

    // History rescale (temporal_gradient_rescale): the reprojected histories jump to the gradient's luminance ratio. Also
    // the coarse anti-lag's clip ratio (RT_LAB_ANTI_LAG_CLIP). gradient_rescale = the total factor (variance, SH below).
    float gradient_rescale = 1.0f;
    if (push.gradient_bound != 0u && settings.temporal_gradient_rescale != 0 && has_history)
    {
        gradient_rescale = rt_lab_gradient_rescale(settings, gradient.y);
    }
    if (push.anti_lag_bound != 0u && has_history)
    {
        gradient_rescale *= rt_lab_coarse_anti_lag_ratio(push.attach.coarse_anti_lag.get(), camera, push.size, dtid, position, normal, rcp_pixel_width_ws);
    }
    if (gradient_rescale != 1.0f)
    {
        reprojected.rgb *= gradient_rescale;
        split_a.rgb *= gradient_rescale;
        split_b.rgb *= gradient_rescale;
    }

    // Coarse temporal anti-lag: cap the reprojected length (never the color) at the coarse cell's RTGI style anti-lag cap
    // (rt_lab_anti_lag.hlsl, alpha = 1 / (cap + 1)); the split lengths shrink by the same ratio. REDUCE_SAMPLES: the
    // stored lengths are cut. BLEND_ONLY: only this frame's blend uses the cut length (blend_keep).
    float anti_lag_cut = 0.0f;     // debug: fraction of the history length removed
    float blend_keep = 1.0f;
    float2 anti_lag_stats = float2(0.0f, 0.0f);
    bool anti_lag_valid = false;
    if (push.anti_lag_bound != 0u)
    {
        anti_lag_valid = rt_lab_coarse_anti_lag_stats(push.attach.coarse_anti_lag.get(), camera, push.size, dtid, position, normal, rcp_pixel_width_ws, anti_lag_stats);
        if (anti_lag_valid && has_history && reprojected.a > 0.0f)
        {
            // anti_lag_stats = (coarse temporal mean, alpha of the coarse test): only the length is capped.
            const float clamped_length = rt_lab_coarse_anti_lag_cap(reprojected.a, anti_lag_stats.y);
            const float keep = clamped_length / reprojected.a;
            anti_lag_cut = 1.0f - keep;
            if ((settings.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MODE_MASK) == RT_LAB_ANTI_LAG_BLEND_ONLY)
            {
                blend_keep = keep;
            }
            else
            {
                reprojected.a = clamped_length;
                split_a.a *= keep;
                split_b.a *= keep;
            }
        }
    }

    // This frame's rays (ray redistribution: 0 .. max_rays_per_pixel), their mean.
    const float4 raw = push.attach.lab_raw_diffuse.get()[dtid];
    const float3 new_sample = raw.rgb;
    const float new_count = raw.a;
    float accumulated_length;
    const float blend = rt_lab_temporal_blend_factor(reprojected.a * blend_keep, new_count, max_length, accumulated_length);
    accumulated_length = min(reprojected.a + new_count, max_length);
    const float4 accumulated = float4(lerp(reprojected.rgb, new_sample, blend), accumulated_length);
    push.attach.accumulated.get()[dtid] = accumulated;

    // Per sample variance: this frame's 5x5 estimate blended into the reprojected one like the color (n / history length),
    // so it converges to the mean of the per frame estimates. Taps reloaded here, the bilinear tap arrays are dead.
    // No ray this frame: no frame estimate (the history's is kept, blend 0).
    // .a: firefly energy weighting, the accumulated RAW mean luminance (same reprojection + blend as the filtered color,
    // so .a / luminance(color) = the history's energy_pre / energy_post). 0 without the ray filter.
    const float3 frame_variance = new_count > 0.0f ? rt_lab_temporal_spatial_variance(dtid, position, normal, rcp_pixel_width_ws, inv_size, new_sample) : float3(0.0f, 0.0f, 0.0f);
    const float frame_raw_luminance = push.raw_luminance_bound != 0u ? push.attach.lab_raw_luminance.get()[dtid] : 0.0f;
    float4 variance = float4(frame_variance, frame_raw_luminance);
    if (has_history)
    {
        float4 reprojected_variance = float4(0.0f, 0.0f, 0.0f, 0.0f);
        [unroll] for (uint i = 0u; i < 4u; ++i)
        {
            if (weights[i] > 0.0f)
            {
                const float4 tap_variance = push.attach.variance_history.get()[int2(bilinear.origin) + int2(i & 1u, i >> 1u)];
                reprojected_variance += (all(isfinite(tap_variance)) ? max(tap_variance, 0.0f) : float4(frame_variance, frame_raw_luminance)) * weights[i];
            }
        }
        reprojected_variance /= weight_sum;
        reprojected_variance *= float4(square(gradient_rescale).xxx, gradient_rescale); // rgb variances, raw luminance
        variance = lerp(reprojected_variance, float4(frame_variance, frame_raw_luminance), blend);
    }
    push.attach.variance.get()[dtid] = push.raw_luminance_bound != 0u ? variance : float4(variance.rgb, 0.0f);

    // Ray length guide: this frame's 5x5 estimate blended into the reprojected one with the color's blend factor (no
    // estimate this frame: the history's is kept). Taps reloaded, bilinear.
    float2 ray_length = float2(-1.0f, 0.0f); // .x < 0: no estimate
    if (push.ray_length_bound != 0u)
    {
        float frame_rays;
        const float2 frame_ray_length = rt_lab_temporal_ray_length(dtid, position, normal, rcp_pixel_width_ws, inv_size, frame_rays);
        float2 reprojected_ray_length = float2(0.0f, 0.0f);
        float reprojected_weight = 0.0f;
        if (has_history)
        {
            [unroll] for (uint i = 0u; i < 4u; ++i)
            {
                if (weights[i] > 0.0f)
                {
                    const float2 tap_ray_length = push.attach.ray_length_history.get()[int2(bilinear.origin) + int2(i & 1u, i >> 1u)];
                    if (all(isfinite(tap_ray_length)) && tap_ray_length.x >= 0.0f)
                    {
                        reprojected_ray_length += tap_ray_length * weights[i];
                        reprojected_weight += weights[i];
                    }
                }
            }
        }
        if (reprojected_weight > 0.0f)
        {
            reprojected_ray_length /= reprojected_weight;
            ray_length = frame_rays > 0.0f ? lerp(reprojected_ray_length, frame_ray_length, blend) : reprojected_ray_length;
        }
        else
        {
            ray_length = frame_rays > 0.0f ? frame_ray_length : float2(-1.0f, 0.0f);
        }
        push.attach.ray_length_accumulated.get()[dtid] = ray_length;
    }

    // SH-Y + CoCg of the coarse fallback: same weights and blend factor as the color (bilinear only), taps reloaded.
    if (push.sh_enabled != 0u)
    {
        // Mean SH of the pixel's rays this frame.
        float4 new_sh_y = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float2 new_cocg = float2(0.0f, 0.0f);
        const uint2 alloc = push.attach.ray_alloc.get()[dtid];
        for (uint k = 0u; k < alloc.y; ++k)
        {
            const RtLabRay ray = push.attach.rays[alloc.x + k];
            float4 ray_sh_y;
            float2 ray_cocg;
            radiance_to_y_co_cg_sh(ray.radiance, uncompress_normal_octahedral_32(ray.direction), ray_sh_y, ray_cocg);
            new_sh_y += ray_sh_y;
            new_cocg += ray_cocg;
        }
        if (alloc.y > 0u)
        {
            new_sh_y /= float(alloc.y);
            new_cocg /= float(alloc.y);
        }
        float4 sh_y = new_sh_y;
        float2 cocg = new_cocg;
        if (has_history)
        {
            float4 reprojected_sh_y = float4(0.0f, 0.0f, 0.0f, 0.0f);
            float2 reprojected_cocg = float2(0.0f, 0.0f);
            [unroll] for (uint i = 0u; i < 4u; ++i)
            {
                if (weights[i] > 0.0f)
                {
                    const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
                    const float4 tap_sh_y = push.attach.sh_y_history.get()[tap];
                    const float4 tap_cocg = push.attach.cocg_history.get()[tap];
                    const bool tap_valid = all(isfinite(tap_sh_y)) && all(isfinite(tap_cocg));
                    reprojected_sh_y += (tap_valid ? tap_sh_y : new_sh_y) * weights[i];
                    reprojected_cocg += (tap_valid ? tap_cocg.xy : new_cocg) * weights[i];
                }
            }
            sh_y = lerp(reprojected_sh_y / weight_sum * gradient_rescale, new_sh_y, blend);
            cocg = lerp(reprojected_cocg / weight_sum * gradient_rescale, new_cocg, blend);
        }
        push.attach.sh_y_accumulated.get()[dtid] = sh_y;
        push.attach.cocg_accumulated.get()[dtid] = float4(cocg, accumulated.a, 0.0f);
    }

    // Split history: A takes the new sample on even frames, B on odd frames.
    if (split)
    {
        if ((push.attach.globals.frame_index & 1u) == 0u)
        {
            split_a = rt_lab_temporal_blend_kept(split_a, new_sample, new_count, split_max_length, blend_keep);
        }
        else
        {
            split_b = rt_lab_temporal_blend_kept(split_b, new_sample, new_count, split_max_length, blend_keep);
        }
        push.attach.split_a_accumulated.get()[dtid] = split_a;
        push.attach.split_b_accumulated.get()[dtid] = split_b;
        // Signed half difference (RTGI_RADIANCE_SCALE scaled) + validity (both halves hold samples).
        const bool split_valid = split_a.a > 0.0f && split_b.a > 0.0f;
        const float half_difference = split_valid ? (rt_lab_luminance(split_a.rgb) - rt_lab_luminance(split_b.rgb)) * 0.5f : 0.0f;
        push.attach.split_difference.get()[dtid] = float2(half_difference, split_valid ? 1.0f : 0.0f);
    }

    // Debug views.
    if (rt_lab_temporal_debug_mode_active(debug_mode))
    {
        // Sample count relative to the fast convergence target (1 = converged, no extra rays requested any more);
        // the full history length beyond it only suppresses long term boiling.
        float3 color = Heatmap(saturate(accumulated.a * rcp(max(push.attach.globals.rt_lab_settings.fast_convergence_samples, 1.0f))));
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_HISTORY_CUTS)
        {
            // R = stored history length / max, G = fraction of the reprojected length the temporal gradient removed,
            // B = fraction the coarse anti-lag removed (blend only: from this frame's blend, the stored length is not cut).
            color = float3(saturate(accumulated.a / max_length), has_history ? 1.0f - gradient_keep : 0.0f, anti_lag_cut);
        }
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_RAY_LENGTH_GUIDE)
        {
            // Accumulated AO guide (1 = open, 0 = occluded within the max distance), heatmap like the RTGI AO guide view.
            // Magenta: guide off. Black: no estimate yet.
            color = push.ray_length_bound == 0u ? float3(1.0f, 0.0f, 1.0f) * 0.25f
                  : (ray_length.x >= 0.0f ? Heatmap(saturate(ray_length.x)) : float3(0.0f, 0.0f, 0.0f));
        }
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_VARIANCE)
        {
            // Relative std of the temporal mean: sqrt(per sample luma variance / history length) / luma, log heatmap
            // 0.1 % .. 100 % (luma variance from the channel variances, channel covariance ignored).
            const float luma_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * push.attach.globals.exposure);
            const float luma_variance = dot(variance.rgb, square(float3(0.2126f, 0.7152f, 0.0722f)));
            color = rt_lab_relative_difference_heatmap(sqrt(luma_variance / max(accumulated.a, 1.0f)) / max(rt_lab_luminance(accumulated.rgb), luma_floor));
        }
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG)
        {
            // Fraction of the reprojected history length the coarse clamp removed: black = none, heatmap to 1 = all.
            // Magenta: clamp off / no coarse statistics.
            color = (push.anti_lag_bound == 0u || !anti_lag_valid) ? float3(1.0f, 0.0f, 1.0f) * 0.25f : (anti_lag_cut > 0.0f ? Heatmap(anti_lag_cut) : float3(0.0f, 0.0f, 0.0f));
        }
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_MEAN)
        {
            // The coarse mean luminance, exposed, x / (1 + x) gray (no clipping). Magenta: clamp off / no coarse statistics.
            const float exposed = rt_lab_exposed(anti_lag_stats.x, push.attach.globals.exposure);
            color = (push.anti_lag_bound == 0u || !anti_lag_valid) ? float3(1.0f, 0.0f, 1.0f) * 0.25f : float3(1.0f, 1.0f, 1.0f) * (exposed / (1.0f + exposed));
        }
        rt_lab_temporal_write_debug(dtid, color);
    }
}
