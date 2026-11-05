#pragma once

#include "rt_lab_integrator.inl"

#include "rt_lab_shared.hlsl"

[[vk::push_constant]] RtLabIntegratorPush rt_lab_integrator_push;

// Reprojected history (bilinear, surface tested). weight (out) = valid footprint weight, 0 = disocclusion.
func rt_lab_integrator_reproject(Texture2D<float4> history, Texture2D<float> depth_history, Texture2D<uint> normal_history, CameraInfo* prev_camera,
    int2 size, float3 position, float3 normal, float rcp_pixel_width_ws, RtLabIntegratorSettings s, out float weight) -> float4
{
    weight = 0.0f;
    const float2 size_f = float2(size);
    const float2 inv_size = rcp(size_f);
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    if (prev_clip.w <= 0.0f)
    {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    const Bilinear bilinear = get_bilinear_filter(prev_uv, size_f);
    float4 tap_weights = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 values[4];
    const float4 bilinear_only = get_bilinear_custom_weights(bilinear, float4(1.0f, 1.0f, 1.0f, 1.0f));
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        values[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
        if (bilinear_only[i] <= 0.0f || any(tap < 0) || any(tap >= size))
        {
            continue; // zero bilinear weight: cannot contribute (static camera: 3 of the 4 taps), no fetch, no unprojection
        }
        const float tap_depth = depth_history[tap];
        if (tap_depth <= 0.0f)
        {
            continue;
        }
        const float3 tap_normal = uncompress_normal_octahedral_32(normal_history[tap]);
        if (dot(normal, tap_normal) < s.normal_threshold)
        {
            continue;
        }
        const float3 tap_position = linear_depth_to_world_space(*prev_camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
        const float plane = rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, s.plane_tolerance);
        const float4 value = history[tap];
        if (plane <= 0.0f || any(!isfinite(value)) || value.a <= 0.0f)
        {
            continue;
        }
        values[i] = value;
        tap_weights[i] = plane;
    }
    const float4 weights = get_bilinear_custom_weights(bilinear, tap_weights);
    weight = dot(weights, 1.0f);
    if (weight <= 1e-3f)
    {
        weight = 0.0f;
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        sum += values[i] * weights[i];
    }
    return sum / weight;
}

// The integrator of one pixel (geometry pixel, the caller handles sky). valid_threshold: an input texel is valid when its .a is above it.
// history_scale multiplies the reprojected history (a lab's own correction of it, e.g. the RT Lab 2 temporal gradient), 1 = none.
// noise (out): window relative std / variance_reference, saturated. Returns .rgb radiance, .a history length (0 = no value).
func rt_lab_integrate(Texture2D<float4> input, Texture2D<float4> history, Texture2D<float> lab_depth, Texture2D<uint> lab_face_normal,
    Texture2D<float> depth_history, Texture2D<uint> normal_history, RenderGlobalData* globals, uint2 dtid, uint2 size, RtLabIntegratorSettings s,
    float valid_threshold, float history_scale, out float noise) -> float4
{
    const CameraInfo camera = globals->view_camera;
    CameraInfo* prev_camera = &globals->view_camera_prev_frame;
    const float depth = lab_depth[dtid];
    const float4 center = input[dtid];
    const float2 inv_size = rcp(float2(size));
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, inv_size), depth);
    const float3 normal = uncompress_normal_octahedral_32(lab_face_normal[dtid]);
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(size.y), depth));
    const float exposure_floor = 0.01f * RTGI_RADIANCE_SCALE / (PI * max(globals->exposure, 1e-6f));

    // Window statistics of this frame's input on the pixel's surface.
    const int radius = clamp(s.window_radius, 0, RT_LAB_INTEGRATOR_MAX_WINDOW_RADIUS);
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float3 sum_squared = float3(0.0f, 0.0f, 0.0f);
    float luma_sum = 0.0f;
    float luma_squared_sum = 0.0f;
    float count = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int2 tap = int2(dtid) + int2(dx, dy);
            if (any(tap < 0) || any(tap >= int2(size)))
            {
                continue;
            }
            const float4 value = input[tap];
            if (!(value.a > valid_threshold) || any(!isfinite(value.rgb)))
            {
                continue;
            }
            if (dx != 0 || dy != 0)
            {
                const float tap_depth = lab_depth[tap];
                if (tap_depth <= 0.0f)
                {
                    continue;
                }
                const float3 tap_normal = uncompress_normal_octahedral_32(lab_face_normal[tap]);
                if (dot(normal, tap_normal) < s.normal_threshold)
                {
                    continue;
                }
                const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
                if (rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, s.plane_tolerance) <= 0.0f)
                {
                    continue;
                }
            }
            const float luma = rt_lab_luminance(value.rgb);
            sum += value.rgb;
            sum_squared += value.rgb * value.rgb;
            luma_sum += luma;
            luma_squared_sum += luma * luma;
            count += 1.0f;
        }
    }
    const bool center_valid = center.a > valid_threshold && all(isfinite(center.rgb));
    float3 window_mean = center_valid ? center.rgb : float3(0.0f, 0.0f, 0.0f);
    float3 window_std = float3(0.0f, 0.0f, 0.0f);
    float relative_std = 0.0f;
    if (count > 0.0f)
    {
        window_mean = sum / count;
        window_std = sqrt(max(sum_squared / count - window_mean * window_mean, 0.0f));
        const float luma_mean = luma_sum / count;
        relative_std = sqrt(max(luma_squared_sum / count - luma_mean * luma_mean, 0.0f)) / (luma_mean + exposure_floor);
    }
    noise = saturate(relative_std / max(s.variance_reference, 1e-4f));

    // The new value: the input, moved toward the window mean when the window is noisy.
    const float3 new_value = center_valid ? lerp(center.rgb, window_mean, saturate(s.spatial_strength) * noise) : window_mean;
    const bool new_valid = center_valid || count > 0.0f;

    float reprojection_weight = 0.0f;
    float4 reprojected = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (s.reset == 0)
    {
        reprojected = rt_lab_integrator_reproject(history, depth_history, normal_history, prev_camera, int2(size), position, normal, rcp_pixel_width_ws, s, reprojection_weight);
    }

    float3 result_value = new_value;
    float frames = new_valid ? 1.0f : 0.0f;
    if (reprojection_weight > 0.0f && new_valid)
    {
        float3 history_value = any(!isfinite(reprojected.rgb)) ? new_value : reprojected.rgb * history_scale;
        if (s.variance_gamma > 0.0f && count > 0.0f)
        {
            history_value = clamp(history_value, window_mean - window_std * s.variance_gamma, window_mean + window_std * s.variance_gamma);
        }
        const float history_frames = min(reprojected.a + 1.0f, float(max(s.max_frames, 1)));
        const float alpha = max(lerp(s.alpha_max, s.alpha_min, noise), 1.0f / history_frames);
        result_value = lerp(history_value, new_value, saturate(alpha));
        frames = history_frames;
    }
    else if (reprojection_weight > 0.0f)
    {
        result_value = reprojected.rgb * history_scale; // no new data: keep the history
        frames = reprojected.a;
    }
    if (any(!isfinite(result_value)))
    {
        result_value = new_valid ? new_value : float3(0.0f, 0.0f, 0.0f);
        frames = new_valid ? 1.0f : 0.0f;
    }
    return float4(result_value, frames);
}

[shader("compute")]
[numthreads(RT_LAB_INTEGRATOR_X, RT_LAB_INTEGRATOR_Y, 1)]
func entry_rt_lab_integrator(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_integrator_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    if (push.attach.lab_depth.get()[dtid] <= 0.0f)
    {
        push.attach.integrated.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    float noise;
    push.attach.integrated.get()[dtid] = rt_lab_integrate(push.attach.input.get(), push.attach.history.get(), push.attach.lab_depth.get(), push.attach.lab_face_normal.get(),
        push.attach.lab_depth_history.get(), push.attach.lab_face_normal_history.get(), push.attach.globals, dtid, push.size, *push.settings, 0.0f, 1.0f, noise);
}
