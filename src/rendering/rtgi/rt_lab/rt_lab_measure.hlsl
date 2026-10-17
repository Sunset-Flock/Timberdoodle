#pragma once

#include "rt_lab_measure.inl"

#include "rt_lab_shared.hlsl"

[[vk::push_constant]] RtLabMeasureAccumulatePush rt_lab_measure_accumulate_push;
[[vk::push_constant]] RtLabMeasureStorePush rt_lab_measure_store_push;
[[vk::push_constant]] RtLabMeasureLagPush rt_lab_measure_lag_push;

groupshared uint gs_measure_row[RT_LAB_MEASURE_ROW_SIZE];

func rt_lab_measure_luminance(float3 v) -> float
{
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

// Slot of a metric value within its metric row: a bin, RT_LAB_MEASURE_UNDER or RT_LAB_MEASURE_OVER.
func rt_lab_measure_slot(float value) -> uint
{
    if (!(value > 0.0f))
    {
        return RT_LAB_MEASURE_UNDER;
    }
    if (!isfinite(value))
    {
        return RT_LAB_MEASURE_OVER;
    }
    const float t = (log10(value) - RT_LAB_MEASURE_LOG_MIN) / (RT_LAB_MEASURE_LOG_MAX - RT_LAB_MEASURE_LOG_MIN);
    if (t < 0.0f)
    {
        return RT_LAB_MEASURE_UNDER;
    }
    if (t >= 1.0f)
    {
        return RT_LAB_MEASURE_OVER;
    }
    return min(uint(t * float(RT_LAB_MEASURE_BINS)), RT_LAB_MEASURE_BINS - 1u);
}

// Plane distance (pixel widths) up to which a previous frame pixel counts as the same surface for the luma statistics.
#define RT_LAB_MEASURE_REPROJECT_PLANE_TOLERANCE 2.0f

// Last frame's running luma statistics (mean, variance, frames, linear depth) of this pixel's surface point: bilinear
// footprint at the previous camera's projection, taps of another surface (plane distance) or sky dropped, weights
// renormalized. A static camera reads the pixel's own texel with weight 1. frames = 0: nothing valid (disocclusion,
// off screen last frame), the statistics restart.
func rt_lab_measure_reproject_statistics(Texture2D<float4> history, CameraInfo* camera, CameraInfo* prev_camera, uint2 size,
    uint2 pixel, float linear_depth, float3 normal) -> float4
{
    const float2 inv_size = rcp(float2(size));
    const float3 position = linear_depth_to_world_space(*camera, rt_lab_ndc(pixel, inv_size), linear_depth);
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(*camera, float(size.y), linear_depth));
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    if (!(prev_clip.w > 0.0f))
    {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    const Bilinear bilinear = get_bilinear_filter(prev_uv, float2(size));
    const float4 bilinear_weights = get_bilinear_custom_weights(bilinear, float4(1.0f, 1.0f, 1.0f, 1.0f));
    float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
        if (any(tap < 0) || any(tap >= int2(size)))
        {
            continue;
        }
        const float4 statistics = history[tap];
        if (!(statistics.w > 0.0f) || !(statistics.z > 0.0f) || !all(isfinite(statistics)))
        {
            continue;
        }
        const float3 tap_position = linear_depth_to_world_space(*prev_camera, rt_lab_ndc(uint2(tap), inv_size), statistics.w);
        const float weight = bilinear_weights[i] *
            rt_lab_plane_weight(position, normal, tap_position, normal, rcp_pixel_width_ws, RT_LAB_MEASURE_REPROJECT_PLANE_TOLERANCE);
        if (weight <= 0.0f)
        {
            continue;
        }
        sum += statistics * weight;
        weight_sum += weight;
    }
    if (weight_sum < 1e-3f)
    {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return sum / weight_sum;
}

[shader("compute")]
[numthreads(RT_LAB_MEASURE_X, RT_LAB_MEASURE_Y, 1)]
func entry_rt_lab_measure_accumulate(uint2 dtid : SV_DispatchThreadID, uint group_index : SV_GroupIndex)
{
    let push = rt_lab_measure_accumulate_push;
    for (uint i = group_index; i < RT_LAB_MEASURE_ROW_SIZE; i += RT_LAB_MEASURE_X * RT_LAB_MEASURE_Y)
    {
        gs_measure_row[i] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    // No early outs: every thread has to reach the barriers.
    if (all(dtid < push.size))
    {
        const uint2 image_size = push.attach.globals.settings.render_target_size;
        const float exposure_scale = PI * push.attach.globals.exposure;
        CameraInfo* camera = &push.attach.globals.view_camera;
        CameraInfo* prev_camera = &push.attach.globals.view_camera_prev_frame;
        uint geometry_pixels = 0u;
        uint error_pixels = 0u;
        float error_squared_sum = 0.0f;
        float instability_squared_sum = 0.0f;
        for (uint y = 0u; y < RT_LAB_MEASURE_TILE; ++y)
        {
            for (uint x = 0u; x < RT_LAB_MEASURE_TILE; ++x)
            {
                const uint2 pixel = dtid * RT_LAB_MEASURE_TILE + uint2(x, y);
                if (any(pixel >= image_size))
                {
                    continue;
                }
                const float ndc_depth = push.attach.view_cam_depth.get()[pixel];
                if (ndc_depth == 0.0f)
                {
                    push.attach.luma_statistics.get()[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f); // sky: no tap may match it next frame
                    continue;
                }
                geometry_pixels += 1u;
                const float luminance = rt_lab_measure_luminance(push.attach.resolved.get()[pixel].rgb) * exposure_scale;

                // Convergence error: |A - B| / min(A, B), A, B ~ output +- |d| (post-spatial split difference).
                const float2 split_difference = push.attach.split_difference_resolved.get()[pixel];
                if (split_difference.y >= 0.999f)
                {
                    const float error = abs(split_difference.x) / RTGI_RADIANCE_SCALE * exposure_scale;
                    error_squared_sum += square(rt_lab_relative_difference(luminance + error, luminance - error));
                    error_pixels += 1u;
                }

                // Temporal instability: relative temporal std of the output luma. On UNEXPOSED luma: std / mean is scale
                // free, and auto exposure still adapting during the run would otherwise show up as fake flicker. Only the
                // dark pixel floor is in exposed units. The statistics follow the surface point (reprojected), so camera
                // motion is not measured as flicker; disoccluded pixels restart.
                const float raw_luminance = luminance / exposure_scale;
                const float linear_depth = camera.near_plane / ndc_depth;
                const float3 normal = uncompress_normal_octahedral_32(push.attach.view_cam_face_normals.get()[pixel]);
                float4 statistics = push.frame == 0u
                    ? float4(0.0f, 0.0f, 0.0f, 0.0f)
                    : rt_lab_measure_reproject_statistics(push.attach.luma_statistics_history.get(), camera, prev_camera, image_size, pixel, linear_depth, normal);
                if (statistics.z < 0.5f)
                {
                    statistics.xyz = float3(raw_luminance, 0.0f, 1.0f);
                }
                else
                {
                    // Plain average for the first `window` frames, then an EMA over `window` frames.
                    const float alpha = max(1.0f / (statistics.z + 1.0f), 1.0f / max(push.window, 1.0f));
                    const float difference = raw_luminance - statistics.x;
                    statistics.x += alpha * difference;
                    statistics.y = (1.0f - alpha) * (statistics.y + alpha * difference * difference);
                    statistics.z = min(statistics.z + 1.0f, 65536.0f);
                }
                statistics.w = linear_depth;
                push.attach.luma_statistics.get()[pixel] = statistics;
                instability_squared_sum += square(sqrt(max(statistics.y, 0.0f)) / max(statistics.x, RT_LAB_RELATIVE_DIFFERENCE_FLOOR / exposure_scale));
            }
        }
        if (geometry_pixels > 0u)
        {
            const uint error_slot = error_pixels > 0u ? rt_lab_measure_slot(sqrt(error_squared_sum / float(error_pixels))) : RT_LAB_MEASURE_OVER;
            const uint instability_slot = rt_lab_measure_slot(sqrt(instability_squared_sum / float(geometry_pixels)));
            const uint error_base = RT_LAB_MEASURE_METRIC_ERROR * RT_LAB_MEASURE_METRIC_STRIDE;
            const uint instability_base = RT_LAB_MEASURE_METRIC_INSTABILITY * RT_LAB_MEASURE_METRIC_STRIDE;
            InterlockedAdd(gs_measure_row[error_base + error_slot], 1u);
            InterlockedAdd(gs_measure_row[error_base + RT_LAB_MEASURE_TILES], 1u);
            InterlockedAdd(gs_measure_row[instability_base + instability_slot], 1u);
            InterlockedAdd(gs_measure_row[instability_base + RT_LAB_MEASURE_TILES], 1u);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    for (uint i = group_index; i < RT_LAB_MEASURE_ROW_SIZE; i += RT_LAB_MEASURE_X * RT_LAB_MEASURE_Y)
    {
        if (gs_measure_row[i] > 0u)
        {
            InterlockedAdd(push.attach.frame_histogram[i], gs_measure_row[i]);
        }
    }
}

[shader("compute")]
[numthreads(RT_LAB_MEASURE_STORE_X, 1, 1)]
func entry_rt_lab_measure_store(uint dtid : SV_DispatchThreadID)
{
    let push = rt_lab_measure_store_push;
    if (dtid >= RT_LAB_MEASURE_ROW_SIZE)
    {
        return;
    }
    push.attach.run_histograms[push.row * RT_LAB_MEASURE_ROW_SIZE + dtid] = push.attach.frame_histogram[dtid];
}

// Temporal lag: mean unexposed output luminance of this thread's tile (-1 = no geometry).
[shader("compute")]
[numthreads(RT_LAB_MEASURE_X, RT_LAB_MEASURE_Y, 1)]
func entry_rt_lab_measure_lag(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_measure_lag_push;
    if (any(dtid >= push.tiles))
    {
        return;
    }
    const uint2 image_size = push.attach.globals.settings.render_target_size;
    float sum = 0.0f;
    float count = 0.0f;
    for (uint y = 0u; y < RT_LAB_MEASURE_LAG_TILE; ++y)
    {
        for (uint x = 0u; x < RT_LAB_MEASURE_LAG_TILE; ++x)
        {
            const uint2 pixel = dtid * RT_LAB_MEASURE_LAG_TILE + uint2(x, y);
            if (any(pixel >= image_size) || push.attach.view_cam_depth.get()[pixel] == 0.0f)
            {
                continue;
            }
            const float luminance = rt_lab_measure_luminance(push.attach.resolved.get()[pixel].rgb);
            if (isfinite(luminance))
            {
                sum += luminance;
                count += 1.0f;
            }
        }
    }
    push.attach.lag_run[push.row * (push.tiles.x * push.tiles.y) + dtid.y * push.tiles.x + dtid.x] = count > 0.0f ? sum / count : -1.0f;
}
