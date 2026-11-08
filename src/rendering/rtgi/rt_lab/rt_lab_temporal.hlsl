#pragma once

#include "rt_lab_temporal.inl"

#include "rt_lab_shared.hlsl"
#include "rt_lab_reproject.hlsl"
#include "rt_lab_distribution.hlsl"
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtLabTemporalPush rt_lab_temporal_push;

func rt_lab_temporal_debug_mode_active(int mode) -> bool
{
    // The split error views are drawn post-spatial by the resolve.
    return mode == DEBUG_DRAW_MODE_RT_LAB_SAMPLE_COUNT || mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_SAMPLE_COUNT ||
           mode == DEBUG_DRAW_MODE_RT_LAB_VARIANCE || mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG ||
           mode == DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_MEAN || mode == DEBUG_DRAW_MODE_RT_LAB_HISTORY_CUTS ||
           mode == DEBUG_DRAW_MODE_RT_LAB_SH_DIRECTION ||
           mode == DEBUG_DRAW_MODE_RT_LAB_SH_DIRECTION_SCALED || mode == DEBUG_DRAW_MODE_RT_LAB_SH_CONCENTRATION ||
           mode == DEBUG_DRAW_MODE_RT_LAB_SH_VARIANCE ||
           (mode >= DEBUG_DRAW_MODE_RT_LAB_DIST_BODY_MEAN && mode <= DEBUG_DRAW_MODE_RT_LAB_DIST_MODEL_ERROR);
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

// Largest log variance the perceptual moments may report (std factor e^2 = 7.4).
#define RT_LAB_MOMENTS_MAX_LOG_VARIANCE 4.0f

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

// Perceptual luma moments of this frame (RtLabSettings::luma_moments_enabled): x = log(luminance of the pixel's mean ray
// + floor), weighted over the 3x3 pixels with rays on this surface (same test as the variance estimate): the center is
// half of the total weight, the 8 neighbors share the other half in proportion to their relevance 1 / distance (edge
// taps 1, corner taps 1 / sqrt(2)); a neighbor that does not qualify gives its share to the center. Returns (m1, m2) = filtered (x, x^2); .z = 0 when no pixel in the footprint has a ray.
func rt_lab_temporal_frame_moments(uint2 dtid, float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size, float luminance_floor) -> float3
{
    let push = rt_lab_temporal_push;
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    float2 ring_sum = float2(0.0f, 0.0f);
    float ring_weight = 0.0f;
    float2 center = float2(0.0f, 0.0f);
    bool center_valid = false;
    [unroll] for (int y = -1; y <= 1; ++y)
    {
        [unroll] for (int x = -1; x <= 1; ++x)
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
            const float perceptual = log(1.0f + rt_lab_luminance(value.rgb) / luminance_floor);
            const float2 moments = float2(perceptual, perceptual * perceptual);
            if (x == 0 && y == 0)
            {
                center = moments;
                center_valid = true;
            }
            else
            {
                const float weight = rsqrt(float(x * x + y * y)); // 1 / distance
                ring_sum += moments * weight;
                ring_weight += weight;
            }
        }
    }
    // Center weight = the ring's total weight (half of everything).
    const float center_weight = center_valid ? max(ring_weight, 1.0f) : 0.0f;
    const float total_weight = ring_weight + center_weight;
    return total_weight > 0.0f ? float3((ring_sum + center * center_weight) / total_weight, 1.0f) : float3(0.0f, 0.0f, 0.0f);
}

// Second moments of this frame's light direction distribution: per pixel the mean over its rays of Y x (d.x^2, d.y^2, d.z^2,
// d.x d.y | d.x d.z, d.y d.z) and Y^2 (A = first four, B = the rest + Y^2), weighted over the 3x3 pixels with rays on this
// surface like rt_lab_temporal_frame_moments (center half of the total weight, the ring shares the other half by 1 / distance).
// valid = false: no pixel in the footprint has a ray.
func rt_lab_temporal_frame_sh_moments(uint2 dtid, float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size, out float4 moments_b, out bool valid) -> float4
{
    let push = rt_lab_temporal_push;
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    float4 ring_a = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ring_b = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float ring_weight = 0.0f;
    float4 center_a = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 center_b = float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool center_valid = false;
    [unroll] for (int y = -1; y <= 1; ++y)
    {
        [unroll] for (int x = -1; x <= 1; ++x)
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
                continue; // no ray this frame
            }
            float4 a = float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 b = float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (uint k = 0u; k < alloc.y; ++k)
            {
                const RtLabRay ray = push.attach.rays[alloc.x + k];
                const float3 direction = uncompress_normal_octahedral_32(ray.direction);
                float4 ray_sh_y;
                float2 ray_cocg;
                radiance_to_y_co_cg_sh(ray.radiance, direction, ray_sh_y, ray_cocg);
                const float ray_y = ray_sh_y.w;
                a += ray_y * float4(direction.x * direction.x, direction.y * direction.y, direction.z * direction.z, direction.x * direction.y);
                b += float4(ray_y * direction.x * direction.z, ray_y * direction.y * direction.z, ray_y * ray_y, 0.0f);
            }
            a /= float(alloc.y);
            b /= float(alloc.y);
            if (x == 0 && y == 0)
            {
                center_a = a;
                center_b = b;
                center_valid = true;
            }
            else
            {
                const float weight = rsqrt(float(x * x + y * y)); // 1 / distance
                ring_a += a * weight;
                ring_b += b * weight;
                ring_weight += weight;
            }
        }
    }
    const float center_weight = center_valid ? max(ring_weight, 1.0f) : 0.0f;
    const float total_weight = ring_weight + center_weight;
    valid = total_weight > 0.0f;
    moments_b = valid ? (ring_b + center_b * center_weight) / total_weight : float4(0.0f, 0.0f, 0.0f, 0.0f);
    return valid ? (ring_a + center_a * center_weight) / total_weight : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// Distribution guide (RtLabSettings::distribution_guide_mode): the pixel's OWN sample of this frame (the mean of its rays) as y = log(1 + luma / floor),
// < 0 = no ray. Only the pixel's own rays feed its statistics: a neighborhood (even on the same plane) mixes the light across shadow edges and texture detail
// into the statistics before any filter has decided which neighbors belong to the pixel, which blurs everything built on them.
func rt_lab_dist_own_sample(uint2 dtid, float luminance_floor) -> float
{
    let push = rt_lab_temporal_push;
    const float4 value = push.attach.lab_raw_diffuse.get()[dtid];
    return value.a > 0.0f ? log(1.0f + rt_lab_luminance(value.rgb) / luminance_floor) : -1.0f;
}

// Body / tail tracking of the pixel's incoming light (rt_lab.md "Distribution guide"). y = log(1 + luma / floor). The state is a
// mixture (dark tail) + body + (bright tail), all lognormal, kept per pixel and reprojected with the color's bilinear weights:
//   A = (body mean, body variance, PERCEPTUAL MEAN, body sample count), B = (p, tail mean, tail variance, tail count), C = the dark tail like B.
// The perceptual mean is the plain mean of y over ALL samples (no classification), an EMA in frame time like the tail fractions. The tail thresholds
// hang on it and on nothing the tail estimates: tau = mean + RT_LAB_DIST_TAIL_OFFSET, tau_low = mean - RT_LAB_DIST_TAIL_OFFSET, so there is no feedback
// loop between the classification and its own threshold. The mean of the PREVIOUS frames sets the thresholds (a bright sample does not move its own
// threshold); a pixel without state uses this frame's mean.
// Per frame:
//   all of it from the pixel's OWN sample (no neighborhood: it would blur the statistics across shadow edges and texture detail):
//   body: the sample if tau_low <= y <= tau, in body sample time (rate 1 / count, count <= RT_LAB_DIST_MAX_COUNT): mean + variance,
//   tail: the sample if y > tau, in tail sample time (count += 1, alpha = 1 / count), (dark tail: y < tau_low); the tails keep a window
//         RT_LAB_DIST_TAIL_WINDOW_FACTOR times longer than the body's (sample count cap, history cut cap), they are rare,
//   p:    EMA of "the sample was in the tail" with the color's blend factor / that factor,
//   perceptual mean: EMA of the sample.
// History cuts (temporal gradient, anti-lag) cap the counts at the cut history length of the color: a reset history forgets body and tails. A history
// rescale shifts the means by ln(rescale) like the luma moments. Returns A (debug views).
func rt_lab_temporal_distribution(uint2 dtid, float luminance_floor,
    bool has_history, Bilinear bilinear, float4 weights, float blend, float new_count, float history_length, float rescale) -> float4
{
    let push = rt_lab_temporal_push;
    const bool three = push.dist_mode >= RT_LAB_DISTRIBUTION_LOGNORMAL_3; // dark tail + body + bright tail

    // Reprojected state: the body from the taps that hold a body, the tail from the taps that hold a tail (different weights).
    float4 sum_a = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float sum_p = 0.0f;
    float3 sum_t = float3(0.0f, 0.0f, 0.0f); // tail mean, tail variance, tail count
    float weight_body = 0.0f;
    float weight_tail = 0.0f;
    float sum_low_fraction = 0.0f;
    float3 sum_low = float3(0.0f, 0.0f, 0.0f); // dark tail mean, variance, count
    float weight_low = 0.0f;
    if (has_history)
    {
        [unroll] for (uint i = 0u; i < 4u; ++i)
        {
            if (weights[i] > 0.0f)
            {
                const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
                const float4 tap_a = push.attach.dist_a_history.get()[tap];
                const float4 tap_b = push.attach.dist_b_history.get()[tap];
                if (all(isfinite(tap_a)) && all(isfinite(tap_b)) && tap_a.w >= 1.0f)
                {
                    sum_a += tap_a * weights[i];
                    sum_p += tap_b.x * weights[i];
                    weight_body += weights[i];
                    if (three)
                    {
                        const float4 tap_c = push.attach.dist_c_history.get()[tap];
                        if (all(isfinite(tap_c)))
                        {
                            sum_low_fraction += tap_c.x * weights[i];
                            if (tap_c.w >= 1.0f)
                            {
                                sum_low += float3(tap_c.y, tap_c.z, tap_c.w) * weights[i];
                                weight_low += weights[i];
                            }
                        }
                    }
                    if (tap_b.w >= 1.0f)
                    {
                        sum_t += float3(tap_b.y, tap_b.z, tap_b.w) * weights[i];
                        weight_tail += weights[i];
                    }
                }
            }
        }
    }
    float body_mean = 0.0f, body_variance = 0.0f, perceptual_mean = 0.0f, body_count = 0.0f;
    float tail_fraction = 0.0f, tail_mean = 0.0f, tail_variance = 0.0f, tail_count = 0.0f;
    float low_fraction = 0.0f, low_mean = 0.0f, low_variance = 0.0f, low_count = 0.0f;
    const bool had_body = weight_body > 1e-3f;
    if (had_body)
    {
        const float4 reprojected_a = sum_a / weight_body;
        body_mean = reprojected_a.x;
        body_variance = max(reprojected_a.y, 0.0f);
        perceptual_mean = reprojected_a.z;
        body_count = reprojected_a.w;
        tail_fraction = saturate(sum_p / weight_body);
        low_fraction = saturate(sum_low_fraction / weight_body);
        if (weight_low > 1e-3f)
        {
            const float3 reprojected_low = sum_low / weight_low;
            low_mean = reprojected_low.x;
            low_variance = max(reprojected_low.y, 0.0f);
            low_count = reprojected_low.z;
        }
        if (weight_tail > 1e-3f)
        {
            const float3 reprojected_tail = sum_t / weight_tail;
            tail_mean = reprojected_tail.x;
            tail_variance = max(reprojected_tail.y, 0.0f);
            tail_count = reprojected_tail.z;
        }
    }
    // History cuts (temporal gradient, anti-lag): the counts are CAPPED at the cut history length of the color. (Scaling them by the cut
    // factor every frame wiped the state: with sparse rays a pixel's counts decayed below one sample between its samples.)
    body_count = min(body_count, history_length);
    tail_count = min(tail_count, history_length * RT_LAB_DIST_TAIL_WINDOW_FACTOR);
    low_count = min(low_count, history_length * RT_LAB_DIST_TAIL_WINDOW_FACTOR);
    if (body_count < 1.0f)
    {
        body_count = 0.0f;
        body_mean = 0.0f;
        body_variance = 0.0f;
        perceptual_mean = 0.0f;
    }
    if (tail_count < 1.0f)
    {
        tail_count = 0.0f;
        tail_mean = 0.0f;
        tail_variance = 0.0f;
    }
    if (low_count < 1.0f)
    {
        low_count = 0.0f;
        low_mean = 0.0f;
        low_variance = 0.0f;
        low_fraction = 0.0f;
    }
    // The luminance scales by `rescale`: the log means shift (none for a ~black pixel, never below 0), the variances stay.
    if (rescale != 1.0f)
    {
        const float log_rescale = log(max(rescale, 1e-6f));
        body_mean += max(log_rescale * saturate(body_mean), -body_mean);
        perceptual_mean += max(log_rescale * saturate(perceptual_mean), -perceptual_mean);
        tail_mean += max(log_rescale * saturate(tail_mean), -tail_mean);
        low_mean += max(log_rescale * saturate(low_mean), -low_mean);
    }

    const bool had_state = body_count >= 1.0f; // after the history cuts
    const float own = rt_lab_dist_own_sample(dtid, luminance_floor);
    if (new_count > 0.0f && own >= 0.0f)
    {
        // The thresholds hang on the perceptual mean of the previous frames only (this frame's sample for a pixel without state).
        const float reference = had_state ? perceptual_mean : own;
        const float tau = reference + RT_LAB_DIST_TAIL_OFFSET;
        const float tau_low = three ? reference - RT_LAB_DIST_TAIL_OFFSET : -1e30f;
        const bool in_tail = own > tau;
        const bool in_low = three && own < tau_low;

        // Body: the sample between the thresholds, body sample time. A body without count starts at the sample (rate 1 overwrites the zero mean).
        if (!in_tail && !in_low)
        {
            body_count = min(body_count + 1.0f, RT_LAB_DIST_MAX_COUNT);
            const float rate = 1.0f / body_count;
            const float d = own - body_mean;
            body_variance = (1.0f - rate) * (body_variance + rate * d * d);
            body_mean += rate * d;
        }
        // Bright tail: the sample above tau, tail sample time.
        if (in_tail)
        {
            tail_count = min(tail_count + 1.0f, RT_LAB_DIST_MAX_COUNT * RT_LAB_DIST_TAIL_WINDOW_FACTOR);
            const float alpha = 1.0f / tail_count;
            const float new_mean = lerp(tail_mean, own, alpha);
            tail_variance = (1.0f - alpha) * (tail_variance + square(tail_mean - new_mean)) + alpha * square(own - new_mean);
            tail_mean = new_mean;
        }
        // Dark tail: the sample below tau_low, dark tail sample time.
        if (in_low)
        {
            low_count = min(low_count + 1.0f, RT_LAB_DIST_MAX_COUNT * RT_LAB_DIST_TAIL_WINDOW_FACTOR);
            const float alpha = 1.0f / low_count;
            const float new_mean = lerp(low_mean, own, alpha);
            low_variance = (1.0f - alpha) * (low_variance + square(low_mean - new_mean)) + alpha * square(own - new_mean);
            low_mean = new_mean;
        }
        // Perceptual mean, frame time (window at most RT_LAB_DIST_MAX_COUNT frames). The tail fractions run on the tails' longer window.
        const float frame_rate = max(blend, 1.0f / RT_LAB_DIST_MAX_COUNT);
        const float tail_rate = max(blend / RT_LAB_DIST_TAIL_WINDOW_FACTOR, 1.0f / (RT_LAB_DIST_MAX_COUNT * RT_LAB_DIST_TAIL_WINDOW_FACTOR));
        tail_fraction = had_body ? lerp(tail_fraction, in_tail ? 1.0f : 0.0f, tail_rate) : (in_tail ? 1.0f : 0.0f);
        low_fraction = !three ? 0.0f : (had_body ? lerp(low_fraction, in_low ? 1.0f : 0.0f, tail_rate) : (in_low ? 1.0f : 0.0f));
        perceptual_mean = had_state ? lerp(perceptual_mean, own, frame_rate) : own;
        // The light changed (brightened by more than the threshold): the body only takes samples below tau and would stay on the old level while
        // the tail grows to the majority. Once the tail holds the majority, it BECOMES the body (the old body is dropped).
        if (tail_count >= RT_LAB_DIST_PROMOTE_COUNT && tail_fraction > RT_LAB_DIST_PROMOTE_FRACTION)
        {
            body_mean = tail_mean;
            body_variance = tail_variance;
            body_count = tail_count;
            tail_fraction = 0.0f;
            tail_mean = 0.0f;
            tail_variance = 0.0f;
            tail_count = 0.0f;
        }
        else if (three && low_count >= RT_LAB_DIST_PROMOTE_COUNT && low_fraction > RT_LAB_DIST_PROMOTE_FRACTION)
        {
            // The light darkened by more than the threshold: the dark tail became the majority and replaces the body.
            body_mean = low_mean;
            body_variance = low_variance;
            body_count = low_count;
            low_fraction = 0.0f;
            low_mean = 0.0f;
            low_variance = 0.0f;
            low_count = 0.0f;
        }
    }
    const bool seeded = body_count >= 1.0f;
    const float4 a = seeded ? float4(body_mean, body_variance, perceptual_mean, body_count) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    push.attach.dist_a.get()[dtid] = a;
    push.attach.dist_b.get()[dtid] = seeded ? float4(tail_fraction, tail_mean, tail_variance, tail_count) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (three)
    {
        push.attach.dist_c.get()[dtid] = seeded ? float4(low_fraction, low_mean, low_variance, low_count) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return a;
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
        if (push.moments_bound != 0u)
        {
            push.attach.moments.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if (push.dist_mode != 0u)
        {
            push.attach.dist_a.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            push.attach.dist_b.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (push.dist_mode >= RT_LAB_DISTRIBUTION_LOGNORMAL_3)
            {
                push.attach.dist_c.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
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
    // Temporal gradient 2 cut of the reprojected lengths (1 = none), applied below once the lengths are known.
    float gradient_keep = 1.0f;
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
        reprojected.a = min(rt_lab_reproject_count(float4(taps[0].a, taps[1].a, taps[2].a, taps[3].a), weights, occluder_coverage, occluder_aware), max_length);
        if (split)
        {
            split_a = apply_bilinear_custom_weights(taps_a[0], taps_a[1], taps_a[2], taps_a[3], weights);
            split_b = apply_bilinear_custom_weights(taps_b[0], taps_b[1], taps_b[2], taps_b[3], weights);
            split_a.a = min(rt_lab_reproject_count(float4(taps_a[0].a, taps_a[1].a, taps_a[2].a, taps_a[3].a), weights, occluder_coverage, occluder_aware), split_max_length);
            split_b.a = min(rt_lab_reproject_count(float4(taps_b[0].a, taps_b[1].a, taps_b[2].a, taps_b[3].a), weights, occluder_coverage, occluder_aware), split_max_length);
        }
        // Temporal gradient 2 (rt_lab_gradient_oct.hlsl): only the history LENGTH is cut by the pyramid's per pixel change (the color stays).
        if (push.gradient2_bound != 0u && settings.temporal_gradient2_scale_history != 0 && reprojected.a > 0.0f)
        {
            const float perceptual_variance = rt_lab_previous_perceptual_variance(push.attach.history.get(), push.attach.variance_history.get(),
                &push.attach.globals.view_camera_prev_frame, settings, push.size, position, push.attach.globals.exposure);
            const float gradient2_keep = rt_lab_gradient2_history_keep(push.attach.gradient2.get()[dtid], reprojected.a, perceptual_variance);
            gradient_keep = gradient2_keep;
            reprojected.a *= gradient2_keep;
            split_a.a *= gradient2_keep;
            split_b.a *= gradient2_keep;
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

    // The coarse anti-lag's clip ratio (rt_lab_anti_lag.hlsl) rescales the reprojected histories (color, split, variance, SH below).
    // The temporal gradient never rescales a history, it only cuts its length.
    float gradient_rescale = 1.0f;
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
    // The color step accumulates the filtered frame, the statistics step the raw rays; the ray count is the pixel's own either way.
    const float3 new_sample = push.color_pass != 0u ? push.attach.lab_prefiltered.get()[dtid].rgb : raw.rgb;
    const float new_count = raw.a;
    float accumulated_length;
    const float blend = rt_lab_temporal_blend_factor(reprojected.a * blend_keep, new_count, max_length, accumulated_length);
    accumulated_length = min(reprojected.a + new_count, max_length);
    float3 accumulated_rgb = lerp(reprojected.rgb, new_sample, blend);

    const float4 accumulated = float4(accumulated_rgb, accumulated_length);
    push.attach.accumulated.get()[dtid] = accumulated;

    // Per sample variance: this frame's 5x5 estimate blended into the reprojected one like the color (n / history length),
    // so it converges to the mean of the per frame estimates. Taps reloaded here, the bilinear tap arrays are dead.
    // No ray this frame: no frame estimate (the history's is kept, blend 0). .a unused (0).
    const float3 frame_variance = new_count > 0.0f ? rt_lab_temporal_spatial_variance(dtid, position, normal, rcp_pixel_width_ws, inv_size, new_sample) : float3(0.0f, 0.0f, 0.0f);
    float3 variance = frame_variance;
    if (has_history)
    {
        float3 reprojected_variance = float3(0.0f, 0.0f, 0.0f);
        [unroll] for (uint i = 0u; i < 4u; ++i)
        {
            if (weights[i] > 0.0f)
            {
                const float3 tap_variance = push.attach.variance_history.get()[int2(bilinear.origin) + int2(i & 1u, i >> 1u)].rgb;
                reprojected_variance += (all(isfinite(tap_variance)) ? max(tap_variance, 0.0f) : frame_variance) * weights[i];
            }
        }
        reprojected_variance /= weight_sum;
        reprojected_variance *= square(gradient_rescale);
        variance = lerp(reprojected_variance, frame_variance, blend);
    }

    // Perceptual luma moments (RtLabSettings::luma_moments_enabled): the weighted m1 = x and m2 = x^2 of x = log(1 + luma /
    // floor) >= 0 over the 3x3 neighborhood (rt_lab_temporal_frame_moments) are blended into the reprojected ones with the
    // color's blend factor (taps reloaded, bilinear). They replace the 5x5 variance above: log variance m2 - m1^2, back to
    // linear with the delta method (luma + floor)^2 x log variance, spread over rgb in proportion to the color's channels.
    // Stored (m1, m2, valid).
    if (push.moments_bound != 0u)
    {
        const float luminance_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * push.attach.globals.exposure);
        const float3 frame_moments = rt_lab_temporal_frame_moments(dtid, position, normal, rcp_pixel_width_ws, inv_size, luminance_floor);
        float3 moments = frame_moments;
        float reprojected_weight = 0.0f;
        float3 reprojected_moments = float3(0.0f, 0.0f, 0.0f);
        if (has_history)
        {
            [unroll] for (uint i = 0u; i < 4u; ++i)
            {
                if (weights[i] > 0.0f)
                {
                    const float3 tap_moments = push.attach.moments_history.get()[int2(bilinear.origin) + int2(i & 1u, i >> 1u)].xyz;
                    if (all(isfinite(tap_moments)) && tap_moments.z > 0.0f)
                    {
                        reprojected_moments += tap_moments * weights[i];
                        reprojected_weight += weights[i];
                    }
                }
            }
        }
        if (reprojected_weight > 0.0f)
        {
            reprojected_moments /= reprojected_weight;
            // History rescale: the luminance scales by r = a shift s = ln(r) of x for x >> 1 (log range), none for x ~ 0
            // (linear range, black stays black); never below x = 0. The shift keeps the variance exact.
            const float shift = max(log(max(gradient_rescale, 1e-6f)) * saturate(reprojected_moments.x), -reprojected_moments.x);
            reprojected_moments.xy = float2(reprojected_moments.x + shift, reprojected_moments.y + 2.0f * shift * reprojected_moments.x + shift * shift);
            moments = frame_moments.z > 0.0f ? float3(lerp(reprojected_moments.xy, frame_moments.xy, blend), 1.0f) : float3(reprojected_moments.xy, 1.0f);
        }
        push.attach.moments.get()[dtid] = float4(moments, 0.0f);
        if (moments.z > 0.0f)
        {
            const float luminance = rt_lab_luminance(accumulated.rgb);
            // Capped: a handful of bright rays among dark ones would blow the log variance (and the delta method) up.
            const float log_variance = min(max(moments.y - moments.x * moments.x, 0.0f), RT_LAB_MOMENTS_MAX_LOG_VARIANCE);
            const float luma_variance = log_variance * square(luminance + luminance_floor);
            // Dark pixels fall back to a gray spread.
            const float3 channel_ratio = luminance > luminance_floor ? accumulated.rgb / luminance : float3(1.0f, 1.0f, 1.0f);
            variance = luma_variance * channel_ratio * channel_ratio;
        }
    }
    push.attach.variance.get()[dtid] = float4(variance, 0.0f);

    // Distribution guide state (body / tail mixture). The history cuts of the temporal gradient and the coarse anti-lag cap the sample counts.
    float4 debug_dist_a = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (push.dist_mode != 0u)
    {
        const float dist_luminance_floor = RT_LAB_DIST_LUMINANCE_FLOOR;
        const float dist_history_length = reprojected.a * blend_keep; // after the gradient cap + the anti-lag cut
        debug_dist_a = rt_lab_temporal_distribution(dtid, dist_luminance_floor, has_history, bilinear, weights,
            blend, new_count, dist_history_length, gradient_rescale);
    }

    // SH-Y + CoCg (SH guides): same weights and blend factor as the color (bilinear only), taps reloaded.
    float4 debug_sh_y = float4(0.0f, 0.0f, 0.0f, 0.0f); // debug views
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
        debug_sh_y = sh_y;
        push.attach.cocg_accumulated.get()[dtid] = float4(cocg, accumulated.a, 0.0f);
    }

    // Squared moments of the ray SH (sq, neural training data): the per pixel mean of Y^2, Co^2, Cg^2 over this frame's rays blended into the
    // reprojected moments with the color's blend factor (bilinear, same weights as the SH-Y). The history rescale scales radiance, squares by r^2.
    if (push.sq_bound != 0u)
    {
        float3 new_sq = float3(0.0f, 0.0f, 0.0f);
        const uint2 sq_alloc = push.attach.ray_alloc.get()[dtid];
        for (uint k = 0u; k < sq_alloc.y; ++k)
        {
            const RtLabRay ray = push.attach.rays[sq_alloc.x + k];
            float4 ray_sh_y;
            float2 ray_cocg;
            radiance_to_y_co_cg_sh(ray.radiance, uncompress_normal_octahedral_32(ray.direction), ray_sh_y, ray_cocg);
            new_sq += float3(ray_sh_y.w * ray_sh_y.w, ray_cocg.x * ray_cocg.x, ray_cocg.y * ray_cocg.y);
        }
        if (sq_alloc.y > 0u)
        {
            new_sq /= float(sq_alloc.y);
        }
        float3 sq = new_sq;
        if (has_history)
        {
            float3 reprojected_sq = float3(0.0f, 0.0f, 0.0f);
            [unroll] for (uint i = 0u; i < 4u; ++i)
            {
                if (weights[i] > 0.0f)
                {
                    const float3 tap_sq = push.attach.sq_history.get()[int2(bilinear.origin) + int2(i & 1u, i >> 1u)].xyz;
                    reprojected_sq += (all(isfinite(tap_sq)) ? tap_sq : new_sq) * weights[i];
                }
            }
            sq = lerp(reprojected_sq / weight_sum * square(gradient_rescale), new_sq, blend);
        }
        push.attach.sq_accumulated.get()[dtid] = float4(sq, 0.0f);
    }

    // SH second moments (sh_m2, sh_m2b): this frame's weighted moments of the light direction distribution blended into the
    // reprojected ones with the color's blend factor (bilinear, same weights as the SH-Y). The radiance rescale of the histories
    // scales Y x d d^T linearly and Y^2 squared.
    float4 debug_sh_m2b = float4(0.0f, 0.0f, 0.0f, 0.0f); // debug views
    if (push.sh_m2_bound != 0u)
    {
        float4 frame_sh_m2b;
        bool frame_sh_m2_valid;
        const float4 frame_sh_m2 = rt_lab_temporal_frame_sh_moments(dtid, position, normal, rcp_pixel_width_ws, inv_size, frame_sh_m2b, frame_sh_m2_valid);
        float4 sh_m2 = frame_sh_m2;
        float4 sh_m2b = frame_sh_m2b;
        if (has_history)
        {
            float4 reprojected_sh_m2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 reprojected_sh_m2b = float4(0.0f, 0.0f, 0.0f, 0.0f);
            [unroll] for (uint i = 0u; i < 4u; ++i)
            {
                if (weights[i] > 0.0f)
                {
                    const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
                    const float4 tap_sh_m2 = push.attach.sh_m2_history.get()[tap];
                    const float4 tap_sh_m2b = push.attach.sh_m2b_history.get()[tap];
                    const bool tap_valid = all(isfinite(tap_sh_m2)) && all(isfinite(tap_sh_m2b));
                    reprojected_sh_m2 += (tap_valid ? tap_sh_m2 : frame_sh_m2) * weights[i];
                    reprojected_sh_m2b += (tap_valid ? tap_sh_m2b : frame_sh_m2b) * weights[i];
                }
            }
            reprojected_sh_m2 = reprojected_sh_m2 / weight_sum * gradient_rescale;
            reprojected_sh_m2b = reprojected_sh_m2b / weight_sum * float4(gradient_rescale, gradient_rescale, square(gradient_rescale), 1.0f);
            sh_m2 = frame_sh_m2_valid ? lerp(reprojected_sh_m2, frame_sh_m2, blend) : reprojected_sh_m2;
            sh_m2b = frame_sh_m2_valid ? lerp(reprojected_sh_m2b, frame_sh_m2b, blend) : reprojected_sh_m2b;
        }
        push.attach.sh_m2_accumulated.get()[dtid] = sh_m2;
        push.attach.sh_m2b_accumulated.get()[dtid] = sh_m2b;
        debug_sh_m2b = sh_m2b;
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
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_VARIANCE)
        {
            // Relative std of the temporal mean: sqrt(per sample luma variance / history length) / luma, log heatmap
            // 0.1 % .. 100 % (luma variance from the channel variances, channel covariance ignored).
            const float luma_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * push.attach.globals.exposure);
            const float luma_variance = dot(variance.rgb, square(float3(0.2126f, 0.7152f, 0.0722f)));
            color = rt_lab_relative_difference_heatmap(sqrt(luma_variance / max(accumulated.a, 1.0f)) / max(rt_lab_luminance(accumulated.rgb), luma_floor));
        }
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_SH_DIRECTION || debug_mode == DEBUG_DRAW_MODE_RT_LAB_SH_DIRECTION_SCALED ||
            debug_mode == DEBUG_DRAW_MODE_RT_LAB_SH_CONCENTRATION)
        {
            // Dominant light direction of the accumulated SH (world space, normal style dir * 0.5 + 0.5), x the debiased
            // concentration for _SCALED (gray = no clear direction), or the concentration alone as a heatmap.
            // Magenta: no SH history (needs an SH guide). Black: no rays yet.
            color = float3(1.0f, 0.0f, 1.0f) * 0.25f;
            if (push.sh_enabled != 0u)
            {
                color = float3(0.0f, 0.0f, 0.0f);
                if (debug_sh_y.w > 0.0f)
                {
                    const float concentration = rt_lab_sh_concentration(debug_sh_y, accumulated.a);
                    const float3 direction = debug_sh_y.xyz * rsqrt(max(dot(debug_sh_y.xyz, debug_sh_y.xyz), 1e-12f));
                    if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_SH_DIRECTION)
                    {
                        color = direction * 0.5f + 0.5f;
                    }
                    else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_SH_DIRECTION_SCALED)
                    {
                        color = (direction * 0.5f + 0.5f) * concentration;
                    }
                    else
                    {
                        color = Heatmap(concentration);
                    }
                }
            }
        }
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_SH_VARIANCE)
        {
            // Relative std of the SH-Y mean (the Y channel): sqrt((E[Y^2] - E[Y]^2) / history length) / E[Y], the same log heatmap as
            // RT_LAB_VARIANCE. Magenta: no SH second moments (needs a variance aware SH guide mode). Black: no samples yet.
            color = float3(1.0f, 0.0f, 1.0f) * 0.25f;
            if (push.sh_m2_bound != 0u)
            {
                color = float3(0.0f, 0.0f, 0.0f);
                if (accumulated.a > 0.0f)
                {
                    const float luma_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * push.attach.globals.exposure);
                    color = rt_lab_relative_difference_heatmap(sqrt(max(debug_sh_m2b.z - debug_sh_y.w * debug_sh_y.w, 0.0f) / max(accumulated.a, 1.0f)) / max(debug_sh_y.w, luma_floor));
                }
            }
        }
        if (debug_mode >= DEBUG_DRAW_MODE_RT_LAB_DIST_BODY_MEAN && debug_mode <= DEBUG_DRAW_MODE_RT_LAB_DIST_MODEL_ERROR)
        {
            // Per pixel state of the body + bright tail guide (A = body, B = tail, y = ln(1 + luma / floor), floor = 1 in RTGI_RADIANCE_SCALE units).
            // Dim magenta: guide off. Black: no body yet (count < 1: the guide treats the pixel as neutral, it accepts every tap), or for the tail views no tail.
            //   BODY_MEAN: body log mean, 0 .. 12.   BODY_STD: body log std, 0 .. 2.   TAIL_FRACTION: p, 0 .. 50 %.
            //   TAIL_HEIGHT: tail mean - body mean (ln), 0 .. 6.   BODY_COUNT / TAIL_COUNT: samples / window (RT_LAB_DIST_MAX_COUNT).
            //   TAIL_ENERGY_SHARE: share of the mixture's mean energy the tail carries, 0 .. 1.
            //   MODEL_ERROR: log2(mixture mean luminance / the accumulated color's luminance), blue = the model is darker, red = brighter, gray = agrees,
            //   +-3 stops. A pixel whose tracked state is stuck on an old light level lights up here while the color has moved on.
            color = float3(1.0f, 0.0f, 1.0f) * 0.25f;
            if (push.dist_mode != 0u)
            {
                color = float3(0.0f, 0.0f, 0.0f);
                if (debug_dist_a.w >= 1.0f)
                {
                    const float4 dist_b = push.attach.dist_b.get()[dtid];
                    const bool has_tail = dist_b.w >= 1.0f && all(isfinite(dist_b));
                    const RtLabTailMixture mixture = rt_lab_tail_mixture(debug_dist_a, dist_b);
                    if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_DIST_BODY_MEAN)
                    {
                        color = Heatmap(saturate(debug_dist_a.x / 12.0f));
                    }
                    else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_DIST_BODY_STD)
                    {
                        color = Heatmap(saturate(0.5f * sqrt(max(debug_dist_a.y, 0.0f))));
                    }
                    else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_DIST_TAIL_FRACTION)
                    {
                        color = has_tail ? Heatmap(saturate(2.0f * dist_b.x)) : float3(0.0f, 0.0f, 0.0f);
                    }
                    else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_DIST_TAIL_HEIGHT)
                    {
                        color = has_tail ? Heatmap(saturate((dist_b.y - debug_dist_a.x) / 6.0f)) : float3(0.0f, 0.0f, 0.0f);
                    }
                    else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_DIST_BODY_COUNT)
                    {
                        color = Heatmap(saturate(debug_dist_a.w / RT_LAB_DIST_MAX_COUNT));
                    }
                    else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_DIST_TAIL_COUNT)
                    {
                        color = has_tail ? Heatmap(saturate(dist_b.w / RT_LAB_DIST_MAX_COUNT)) : float3(0.0f, 0.0f, 0.0f);
                    }
                    else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_DIST_TAIL_ENERGY_SHARE)
                    {
                        color = has_tail ? Heatmap(saturate(mixture.energy.y / max(mixture.mean_energy, 1e-6f))) : float3(0.0f, 0.0f, 0.0f);
                    }
                    else
                    {
                        // E - 1 is the mixture's mean luminance in floor units = RTGI_RADIANCE_SCALE units; + 0.5 keeps dark pixels from exploding the ratio.
                        const float model_luminance = max(mixture.mean_energy - 1.0f, 0.0f);
                        const float stops = log2((model_luminance + 0.5f) / (rt_lab_luminance(accumulated.rgb) + 0.5f));
                        const float signed_stops = clamp(stops / 3.0f, -1.0f, 1.0f);
                        color = signed_stops >= 0.0f ? lerp(float3(0.5f, 0.5f, 0.5f), float3(1.0f, 0.15f, 0.1f), signed_stops)
                                                     : lerp(float3(0.5f, 0.5f, 0.5f), float3(0.1f, 0.3f, 1.0f), -signed_stops);
                    }
                }
            }
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
