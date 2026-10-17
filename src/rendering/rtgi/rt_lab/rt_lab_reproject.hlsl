#pragma once

#include "rt_lab_shared.hlsl"

// History reprojection shared by the temporal pass (rt_lab_temporal.hlsl) and the ray demand (rt_lab_rays.hlsl), so the
// ray demand sees exactly the history length the temporal pass will reproject.

// Surface test of one previous frame lab pixel: > 0 = it lies on this pixel's surface (plane distance weight), 0 = out of
// bounds, sky, or another surface (normal / plane distance).
// occluder (out): the tap failed the surface test but lay more than 2 pixel widths IN FRONT of the expected point,
// seen from the previous camera: it hid the point last frame (real disocclusion). Failed taps behind it / sky only mean
// the footprint overhangs an edge.
func rt_lab_reproject_tap(Texture2D<float> depth_history, Texture2D<uint> face_normal_history, CameraInfo* prev_camera,
    RtLabSettings* settings, int2 size, int2 tap, float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size,
    out bool occluder) -> float
{
    occluder = false;
    if (any(tap < 0) || any(tap >= size))
    {
        return 0.0f;
    }
    const float tap_depth = depth_history[tap];
    if (tap_depth == 0.0f)
    {
        return 0.0f;
    }
    const float3 tap_normal = uncompress_normal_octahedral_32(face_normal_history[tap]);
    const float3 tap_position = linear_depth_to_world_space(*prev_camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
    float weight = 0.0f;
    if (dot(normal, tap_normal) >= settings->disocclusion_normal_threshold)
    {
        weight = rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, settings->disocclusion_plane_tolerance);
    }
    if (weight <= 0.0f)
    {
        const float expected_distance = distance(prev_camera->position, position);
        occluder = distance(prev_camera->position, tap_position) < expected_distance - 2.0f / rcp_pixel_width_ws;
    }
    return weight;
}

// Reprojected history length of a partially valid footprint (counts / weights per bilinear tap), see
// RtLabSettings::occluder_aware_soft_normalize.
func rt_lab_reproject_count(float4 counts, float4 weights, float occluder_coverage, bool occluder_aware) -> float
{
    float max_valid = 0.0f;
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        max_valid = max(max_valid, weights[i] > 0.1f ? counts[i] : 0.0f);
    }
    const float weight_sum = dot(weights, 1.0f);
    float count = 0.0f;
    if (occluder_aware)
    {
        const float normalized = weight_sum > 1e-6f ? dot(counts, weights) / weight_sum : 0.0f;
        count = normalized * pow(saturate(1.0f - occluder_coverage), 0.34f);
    }
    else
    {
        count = weight_sum > 1e-6f ? dot(counts, weights) / pow(weight_sum, 0.66f) : 0.0f;
    }
    return max(min(max_valid, 4.0f), count);
}

// History length cap of the active temporal mode.
func rt_lab_max_history_length(RtLabSettings* settings) -> float
{
    return float(max(settings->temporal_progressive != 0 ? settings->progressive_max_samples : settings->max_history_frames, 1));
}

// The temporal pass's reprojected history length (main history), without the colors: bilinear footprint, same surface
// test and tap validity, same length reprojection (occluder aware soft normalize or soft normalize), capped like it.
// 0 = disocclusion.
func rt_lab_reproject_history_length(Texture2D<float4> history, Texture2D<float> depth_history, Texture2D<uint> face_normal_history,
    CameraInfo* prev_camera, RtLabSettings* settings, int2 size, float3 position, float3 normal, float rcp_pixel_width_ws) -> float
{
    if (settings->reset_history != 0)
    {
        return 0.0f;
    }
    const float2 size_f = float2(size);
    const float2 inv_size = rcp(size_f);
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    const Bilinear bilinear = get_bilinear_filter(prev_uv, size_f);

    float4 tap_weights = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 occluder_taps = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 counts = float4(0.0f, 0.0f, 0.0f, 0.0f);
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
        bool occluder;
        const float weight = rt_lab_reproject_tap(depth_history, face_normal_history, prev_camera, settings, size, tap, position, normal,
            rcp_pixel_width_ws, inv_size, occluder);
        occluder_taps[i] = occluder ? 1.0f : 0.0f;
        if (weight <= 0.0f)
        {
            continue;
        }
        // Uninitialized / broken history counts as a disocclusion (same test as the temporal pass).
        const float4 value = history[tap];
        if (any(!isfinite(value)) || value.a < 0.0f)
        {
            continue;
        }
        counts[i] = value.a;
        tap_weights[i] = weight;
    }
    const float4 weights = get_bilinear_custom_weights(bilinear, tap_weights);
    if (dot(weights, 1.0f) <= 1e-3f)
    {
        return 0.0f;
    }
    const float occluder_coverage = dot(get_bilinear_custom_weights(bilinear, float4(1.0f, 1.0f, 1.0f, 1.0f)), occluder_taps);
    return min(rt_lab_reproject_count(counts, weights, occluder_coverage, settings->occluder_aware_soft_normalize != 0), rt_lab_max_history_length(settings));
}
