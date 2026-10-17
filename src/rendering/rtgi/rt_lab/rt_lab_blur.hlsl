#pragma once

#include "rt_lab_blur.inl"

#include "rt_lab_shared.hlsl"

[[vk::push_constant]] RtLabBlurPush rt_lab_blur_push;

// 5x5 B3 spline kernel weights by |offset| (1/16, 1/4, 3/8, 1/4, 1/16 per axis).
static const float RT_LAB_ATROUS_KERNEL[3] = {3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};

// Weighted sums of a group of taps: color / split by the filter weight (normalized at the end), count by the weight
// relative to the center weight (the effective sample count, not normalized).
struct RtLabSpatialSum
{
    float3 color;
    float2 split;
    float weight;
    float count;
};

func rt_lab_spatial_sum_zero() -> RtLabSpatialSum
{
    return RtLabSpatialSum(float3(0.0f, 0.0f, 0.0f), float2(0.0f, 0.0f), 0.0f, 0.0f);
}

// Sample count budget (fine filter, RtLabSettings::fine_target_samples > 0): taps are accepted ring by ring from the
// center outwards (closest first). A ring that fits is accepted fully; the ring that would cross the target is scaled
// so the pass reaches the target exactly; the rest is dropped. Returns true when the budget is used up.
func rt_lab_accept_ring(inout RtLabSpatialSum total, RtLabSpatialSum ring, float target) -> bool
{
    float accept = 1.0f;
    if (target > 0.0f && ring.count > 0.0f)
    {
        accept = saturate((target - total.count) / ring.count);
    }
    total.color += ring.color * accept;
    total.split += ring.split * accept;
    total.weight += ring.weight * accept;
    total.count += ring.count * accept;
    return target > 0.0f && total.count >= target * 0.9999f;
}

// Variance guide of the fine filter (RtLabSettings::variance_guide_*): the center's mean and the uncertainty of it.
struct RtLabVarianceGuide
{
    bool enabled;
    float3 mean;
    float3 mean_variance; // per sample variance / effective sample count
    float rcp_two_k_sq;   // 1 / (2 k^2)
    float3 variance_floor;
};

func rt_lab_variance_guide_none() -> RtLabVarianceGuide
{
    return RtLabVarianceGuide(false, float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), 0.0f, float3(0.0f, 0.0f, 0.0f));
}

func rt_lab_variance_guide_begin(uint2 dtid, float3 mean, float count) -> RtLabVarianceGuide
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (push.variance_bound == 0u || settings.variance_guide_enabled == 0 || count <= 0.0f)
    {
        return rt_lab_variance_guide_none();
    }
    RtLabVarianceGuide guide;
    guide.enabled = true;
    guide.mean = mean;
    guide.mean_variance = max(push.attach.lab_variance.get()[dtid].rgb, 0.0f) / max(count, 1e-3f);
    const float k = max(settings.variance_guide_sigmas, 1e-3f);
    guide.rcp_two_k_sq = rcp(2.0f * k * k);
    guide.variance_floor = rt_lab_variance_guide_floor(push.attach.globals.exposure);
    return guide;
}

// Likelihood that center and tap estimate the same mean (rt_lab_variance_guide_likelihood).
func rt_lab_variance_guide_weight(RtLabVarianceGuide guide, int2 tap, float3 tap_mean, float tap_count) -> float
{
    let push = rt_lab_blur_push;
    const float3 tap_mean_variance = max(push.attach.lab_variance.get()[tap].rgb, 0.0f) / max(tap_count, 1e-3f);
    return rt_lab_variance_guide_likelihood(guide.mean, guide.mean_variance, tap_mean, tap_mean_variance, guide.variance_floor, guide.rcp_two_k_sq);
}

// One spatial tap: filter weight (kernel x edge stopping (plane distance + normal power), 0 = rejected) and its color /
// count / split added to ring.
func rt_lab_spatial_tap(inout RtLabSpatialSum ring, int2 tap, float kernel, float count_scale,
    float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size, bool clamp_count, RtLabVarianceGuide guide)
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (any(tap < 0) || any(tap >= int2(push.size)))
    {
        return;
    }
    const float tap_depth = push.attach.lab_depth.get()[tap];
    if (tap_depth == 0.0f)
    {
        return;
    }
    const CameraInfo camera = push.attach.globals.view_camera;
    const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[tap]);
    const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
    float weight = kernel *
                   rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, settings.blur_plane_tolerance) *
                   pow(saturate(dot(normal, tap_normal)), settings.blur_normal_power);
    if (weight <= 0.0f)
    {
        return;
    }
    const float4 value = push.attach.src.get()[tap];
    const float tap_count = rt_lab_spatial_read_count(value.a, clamp_count, settings.sample_count_input_clamp);
    if (tap_count <= 0.0f)
    {
        return; // no samples (ray redistribution left the pixel without a ray)
    }
    // A rejected tap is not gathered: the variance guide weight also scales the effective sample count.
    if (guide.enabled)
    {
        weight *= rt_lab_variance_guide_weight(guide, tap, value.rgb, tap_count);
        if (weight <= 1e-6f)
        {
            return;
        }
    }
    ring.color += value.rgb * weight;
    ring.weight += weight;
    ring.count += tap_count * weight * count_scale;
    if (push.split_enabled != 0u)
    {
        ring.split += push.attach.split_src.get()[tap] * weight;
    }
}

// Shared prologue of both filters: false = sky (output written).
func rt_lab_spatial_begin(uint2 dtid, out float4 center, out float3 position, out float3 normal, out float rcp_pixel_width_ws, out float2 inv_size) -> bool
{
    let push = rt_lab_blur_push;
    center = push.attach.src.get()[dtid];
    position = float3(0.0f, 0.0f, 0.0f);
    normal = float3(0.0f, 0.0f, 1.0f);
    rcp_pixel_width_ws = 0.0f;
    inv_size = rcp(float2(push.size));
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        push.attach.dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (push.split_enabled != 0u)
        {
            push.attach.split_dst.get()[dtid] = float2(0.0f, 0.0f);
        }
        return false;
    }
    const CameraInfo camera = push.attach.globals.view_camera;
    position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, inv_size), depth);
    normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.size.y), depth));
    return true;
}

func rt_lab_spatial_end(uint2 dtid, RtLabSpatialSum total)
{
    let push = rt_lab_blur_push;
    // No samples anywhere in the footprint: count 0, the next pass / the resolve skip it.
    const float rcp_weight = total.weight > 0.0f ? rcp(total.weight) : 0.0f;
    push.attach.dst.get()[dtid] = float4(total.color * rcp_weight, total.count / RT_LAB_SPATIAL_COUNT_SCALE);
    if (push.split_enabled != 0u)
    {
        push.attach.split_dst.get()[dtid] = total.split * rcp_weight;
    }
}

// The center tap: always fully accepted (relative count weight 1), unless it holds no samples (weight 0).
func rt_lab_spatial_center(uint2 dtid, float4 center, float weight_in, bool clamp_count) -> RtLabSpatialSum
{
    let push = rt_lab_blur_push;
    RtLabSpatialSum total = rt_lab_spatial_sum_zero();
    total.count = rt_lab_spatial_read_count(center.a, clamp_count, push.attach.globals.rt_lab_settings.sample_count_input_clamp);
    const float weight = total.count > 0.0f ? weight_in : 0.0f;
    total.color = center.rgb * weight;
    total.weight = weight;
    if (push.split_enabled != 0u)
    {
        total.split = push.attach.split_src.get()[dtid] * weight;
    }
    return total;
}

// One a-trous pass: 5x5 B3 spline taps spaced push.step pixels apart, rings (max(|x|, |y|) = 1, then 2) accepted from
// the center outwards under the fine filter's sample count budget. The split difference takes the same weights.
// Alpha: effective sample count.
[shader("compute")]
[numthreads(RT_LAB_BLUR_X, RT_LAB_BLUR_Y, 1)]
func entry_rt_lab_atrous(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_blur_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    float4 center;
    float3 position;
    float3 normal;
    float rcp_pixel_width_ws;
    float2 inv_size;
    if (!rt_lab_spatial_begin(dtid, center, position, normal, rcp_pixel_width_ws, inv_size))
    {
        return;
    }
    const bool clamp_count = push.clamp_input_count != 0u;
    const float target = push.attach.globals.rt_lab_settings.fine_target_samples;
    const float center_kernel = RT_LAB_ATROUS_KERNEL[0] * RT_LAB_ATROUS_KERNEL[0];
    RtLabSpatialSum total = rt_lab_spatial_center(dtid, center, center_kernel, clamp_count);
    const RtLabVarianceGuide guide = rt_lab_variance_guide_begin(dtid, center.rgb, total.count);
    if (!(target > 0.0f && total.count >= target))
    {
        [unroll] for (int r = 1; r <= 2; ++r)
        {
            RtLabSpatialSum ring = rt_lab_spatial_sum_zero();
            [unroll] for (int y = -2; y <= 2; ++y)
            {
                [unroll] for (int x = -2; x <= 2; ++x)
                {
                    if (max(abs(x), abs(y)) != r)
                    {
                        continue;
                    }
                    const float kernel = RT_LAB_ATROUS_KERNEL[abs(x)] * RT_LAB_ATROUS_KERNEL[abs(y)];
                    rt_lab_spatial_tap(ring, int2(dtid) + int2(x, y) * push.step, kernel, 1.0f / center_kernel,
                        position, normal, rcp_pixel_width_ws, inv_size, clamp_count, guide);
                }
            }
            if (rt_lab_accept_ring(total, ring, target))
            {
                break;
            }
        }
    }
    rt_lab_spatial_end(dtid, total);
}

// One pass of the separable bilateral gaussian. Edge stopping is purely geometric: plane distance (in pixel widths)
// and face normal similarity. No luminance, AO or variance guides. Taps +-d are accepted from d = 1 outwards under the
// fine filter's sample count budget. Alpha: effective sample count (rt_lab_blur.inl).
[shader("compute")]
[numthreads(RT_LAB_BLUR_X, RT_LAB_BLUR_Y, 1)]
func entry_rt_lab_blur(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_blur_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    float4 center;
    float3 position;
    float3 normal;
    float rcp_pixel_width_ws;
    float2 inv_size;
    if (!rt_lab_spatial_begin(dtid, center, position, normal, rcp_pixel_width_ws, inv_size))
    {
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const int radius = clamp(settings.blur_radius, 0, 32);
    const float sigma = max(float(radius) * 0.5f, 0.5f);
    const float rcp_two_sigma_sq = rcp(2.0f * sigma * sigma);
    const bool clamp_count = push.clamp_input_count != 0u;
    const float target = settings.fine_target_samples;

    // The center weight is 1, so the relative count weight is the weight itself.
    RtLabSpatialSum total = rt_lab_spatial_center(dtid, center, 1.0f, clamp_count);
    const RtLabVarianceGuide guide = rt_lab_variance_guide_begin(dtid, center.rgb, total.count);
    if (!(target > 0.0f && total.count >= target))
    {
        [loop] for (int d = 1; d <= radius; ++d)
        {
            const float kernel = exp(-float(d * d) * rcp_two_sigma_sq);
            RtLabSpatialSum ring = rt_lab_spatial_sum_zero();
            rt_lab_spatial_tap(ring, int2(dtid) - push.direction * d, kernel, 1.0f, position, normal, rcp_pixel_width_ws, inv_size, clamp_count, guide);
            rt_lab_spatial_tap(ring, int2(dtid) + push.direction * d, kernel, 1.0f, position, normal, rcp_pixel_width_ws, inv_size, clamp_count, guide);
            if (rt_lab_accept_ring(total, ring, target))
            {
                break;
            }
        }
    }
    rt_lab_spatial_end(dtid, total);
}
