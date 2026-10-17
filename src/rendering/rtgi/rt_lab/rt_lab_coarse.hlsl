#pragma once

#include "rt_lab_coarse.inl"

#include "rt_lab_shared.hlsl"

[[vk::push_constant]] RtLabCoarsePush rt_lab_coarse_push;

// 5x5 B3 spline kernel weights by |offset| (1/16, 1/4, 3/8, 1/4, 1/16 per axis).
static const float RT_LAB_COARSE_KERNEL[3] = {3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};

[shader("compute")]
[numthreads(RT_LAB_COARSE_X, RT_LAB_COARSE_Y, 1)]
func entry_rt_lab_coarse(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_coarse_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const bool split = push.split_enabled != 0u;
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        push.attach.dst_sh_y.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        push.attach.dst_cocg_count.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (split)
        {
            push.attach.split_dst.get()[dtid] = float2(0.0f, 0.0f);
        }
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const float2 inv_size = rcp(float2(push.size));
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, inv_size), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.size.y), depth));
    const bool clamp_count = push.clamp_input_count != 0u;

    // Colors normalized by the filter weights, the count by the weight relative to the center (not normalized).
    const float center_kernel = RT_LAB_COARSE_KERNEL[0] * RT_LAB_COARSE_KERNEL[0];
    const float4 center_cocg_count = push.attach.src_cocg_count.get()[dtid];
    float count = rt_lab_spatial_read_count(center_cocg_count.z, clamp_count, settings.sample_count_input_clamp);
    // A center without samples (ray redistribution left it without a ray) takes no weight.
    const float center_weight = count > 0.0f ? center_kernel : 0.0f;
    const float4 center_sh_y = push.attach.src_sh_y.get()[dtid];
    float4 sh_y = center_sh_y * center_weight;
    float2 cocg = center_cocg_count.xy * center_weight;
    float2 split_difference = split ? push.attach.split_src.get()[dtid] * center_weight : float2(0.0f, 0.0f);
    float weight_sum = center_weight;

    // Variance guide: the mean rgb is the YCoCg of the SH's Y (sh_y.w) and the CoCg.
    const bool variance_guide = settings.coarse_variance_guide_enabled != 0 && count > 0.0f; // no center samples: nothing to compare with
    float3 center_mean = float3(0.0f, 0.0f, 0.0f);
    float3 center_mean_variance = float3(0.0f, 0.0f, 0.0f);
    float3 variance_floor = float3(0.0f, 0.0f, 0.0f);
    float rcp_two_k_sq = 0.0f;
    if (variance_guide)
    {
        center_mean = y_co_cg_to_linear(float3(center_sh_y.w, center_cocg_count.xy));
        center_mean_variance = max(push.attach.lab_variance.get()[dtid].rgb, 0.0f) / max(count, 1e-3f);
        variance_floor = rt_lab_variance_guide_floor(push.attach.globals.exposure);
        const float k = max(settings.coarse_variance_guide_sigmas, 1e-3f);
        rcp_two_k_sq = rcp(2.0f * k * k);
    }
    [unroll] for (int y = -2; y <= 2; ++y)
    {
        [unroll] for (int x = -2; x <= 2; ++x)
        {
            if (x == 0 && y == 0)
            {
                continue;
            }
            const int2 tap = int2(dtid) + int2(x, y) * push.step;
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
            float weight = RT_LAB_COARSE_KERNEL[abs(x)] * RT_LAB_COARSE_KERNEL[abs(y)] *
                                 rt_lab_coarse_edge_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws,
                                     length(float2(x, y)) * float(push.step), settings.coarse_distance_tolerance, settings.coarse_normal_power);
            if (weight <= 0.0f)
            {
                continue;
            }
            const float4 tap_cocg_count = push.attach.src_cocg_count.get()[tap];
            const float4 tap_sh_y = push.attach.src_sh_y.get()[tap];
            const float tap_count = rt_lab_spatial_read_count(tap_cocg_count.z, clamp_count, settings.sample_count_input_clamp);
            if (tap_count <= 0.0f)
            {
                continue; // no samples
            }
            // A rejected tap is not gathered: the variance guide weight also scales the effective sample count.
            if (variance_guide)
            {
                const float3 tap_mean = y_co_cg_to_linear(float3(tap_sh_y.w, tap_cocg_count.xy));
                const float3 tap_mean_variance = max(push.attach.lab_variance.get()[tap].rgb, 0.0f) / max(tap_count, 1e-3f);
                weight *= rt_lab_variance_guide_likelihood(center_mean, center_mean_variance, tap_mean, tap_mean_variance, variance_floor, rcp_two_k_sq);
                if (weight <= 1e-6f)
                {
                    continue;
                }
            }
            sh_y += tap_sh_y * weight;
            cocg += tap_cocg_count.xy * weight;
            weight_sum += weight;
            count += tap_count * weight / center_kernel;
            if (split)
            {
                split_difference += push.attach.split_src.get()[tap] * weight;
            }
        }
    }
    const float rcp_weight = weight_sum > 0.0f ? rcp(weight_sum) : 0.0f;
    push.attach.dst_sh_y.get()[dtid] = sh_y * rcp_weight;
    push.attach.dst_cocg_count.get()[dtid] = float4(cocg * rcp_weight, count / RT_LAB_SPATIAL_COUNT_SCALE, 0.0f);
    if (split)
    {
        push.attach.split_dst.get()[dtid] = split_difference * rcp_weight;
    }
}
