#pragma once

#include "rt_lab_probe.inl"

#include "rt_lab_shared.hlsl"
#include "rt_lab_distribution.hlsl" // the distribution guide comparisons of the spatial passes

[[vk::push_constant]] RtLabProbePush rt_lab_probe_push;

func rt_lab_probe_store(uint index, float value)
{
    rt_lab_probe_push.attach.probe[index] = asuint(value);
}

func rt_lab_probe_store(uint index, uint value)
{
    rt_lab_probe_push.attach.probe[index] = value;
}

func rt_lab_probe_store(uint index, float3 value)
{
    rt_lab_probe_store(index + 0u, value.x);
    rt_lab_probe_store(index + 1u, value.y);
    rt_lab_probe_store(index + 2u, value.z);
}

func rt_lab_probe_store(uint index, float4 value)
{
    rt_lab_probe_store(index, value.xyz);
    rt_lab_probe_store(index + 3u, value.w);
}

// What the first spatial pass (rt_lab_blur.hlsl, input = the temporal output) knows about one lab pixel.
struct RtLabProbeSide
{
    bool geometry;
    float3 position;
    float3 normal;
    float rcp_pixel_width_ws;
    float4 temporal; // rgb mean (RTGI_RADIANCE_SCALE units), history length
    float3 variance; // per sample variance
    float4 dist_a;
    float4 dist_b;
    float4 dist_c;
};

func rt_lab_probe_side(uint2 lab_pixel) -> RtLabProbeSide
{
    let push = rt_lab_probe_push;
    RtLabProbeSide side = {};
    uint2 size;
    push.attach.lab_depth.get().GetDimensions(size.x, size.y);
    const float depth = push.attach.lab_depth.get()[lab_pixel];
    side.geometry = depth > 0.0f;
    const CameraInfo camera = push.attach.globals.view_camera;
    side.position = side.geometry ? linear_depth_to_world_space(camera, rt_lab_ndc(lab_pixel, rcp(float2(size))), depth) : float3(0.0f, 0.0f, 0.0f);
    side.normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[lab_pixel]);
    side.rcp_pixel_width_ws = side.geometry ? rcp(rt_lab_pixel_width_ws(camera, float(size.y), depth)) : 0.0f;
    side.temporal = push.attach.lab_temporal.get()[lab_pixel];
    side.variance = (push.bound & RT_LAB_PROBE_FLAG_VARIANCE) != 0u ? max(push.attach.lab_variance.get()[lab_pixel].rgb, 0.0f) : float3(0.0f, 0.0f, 0.0f);
    side.dist_a = (push.bound & RT_LAB_PROBE_FLAG_DIST_AB) != 0u ? push.attach.dist_a.get()[lab_pixel] : float4(0.0f, 0.0f, 0.0f, 0.0f);
    side.dist_b = (push.bound & RT_LAB_PROBE_FLAG_DIST_AB) != 0u ? push.attach.dist_b.get()[lab_pixel] : float4(0.0f, 0.0f, 0.0f, 0.0f);
    side.dist_c = (push.bound & RT_LAB_PROBE_FLAG_DIST_C) != 0u ? push.attach.dist_c.get()[lab_pixel] : float4(0.0f, 0.0f, 0.0f, 0.0f);
    return side;
}

// The distribution guide weight of tap t seen from center c: the same formulas as rt_lab_distribution_guide_weight (rt_lab_blur.hlsl).
func rt_lab_probe_distribution_weight(RtLabProbeSide c, RtLabProbeSide t, bool multi, int compare, float sigmas) -> float
{
    if (!(c.dist_a.w >= 1.0f) || !(t.dist_a.w >= 1.0f))
    {
        return 1.0f;
    }
    const float k = max(sigmas, 1e-3f);
    if (multi)
    {
        float3 weight_c, mean_c, variance_c, weight_t, mean_t, variance_t;
        rt_lab_distribution_components(c.dist_a, c.dist_b, c.dist_c, weight_c, mean_c, variance_c);
        rt_lab_distribution_components(t.dist_a, t.dist_b, t.dist_c, weight_t, mean_t, variance_t);
        const float cross = rt_lab_distribution_overlap(weight_c, mean_c, variance_c, weight_t, mean_t, variance_t, k);
        const float self_c = rt_lab_distribution_self_overlap(weight_c, mean_c, variance_c, k);
        const float self_t = rt_lab_distribution_self_overlap(weight_t, mean_t, variance_t, k);
        return cross * rsqrt(max(self_c * self_t, 1e-8f)) *
               rt_lab_distribution_energy_weight(rt_lab_distribution_energy(c.dist_a, c.dist_b, c.dist_c), rt_lab_distribution_energy(t.dist_a, t.dist_b, t.dist_c), k);
    }
    return rt_lab_distribution_compare(compare, c.dist_a, c.dist_b, t.dist_a, t.dist_b, k);
}

[shader("compute")]
[numthreads(RT_LAB_PROBE_COUNT, 1, 1)]
func entry_rt_lab_probe(uint probe_index : SV_GroupIndex)
{
    let push = rt_lab_probe_push;
    const uint slot_base = push.slot * RT_LAB_PROBE_SLOT_WORDS;
    if (probe_index == 0u)
    {
        rt_lab_probe_store(slot_base + 0u, push.stamp);
        rt_lab_probe_store(slot_base + 1u, push.active_mask);
    }
    if (((push.active_mask >> probe_index) & 1u) == 0u)
    {
        return;
    }
    const uint base = slot_base + RT_LAB_PROBE_HEADER_WORDS + probe_index * RT_LAB_PROBE_WORDS;
    const uint2 screen_pixel = push.screen_pixel[probe_index];
    const uint2 lab_pixel = push.lab_pixel[probe_index];
    const float depth = push.attach.lab_depth.get()[lab_pixel];
    uint flags = depth > 0.0f ? RT_LAB_PROBE_FLAG_GEOMETRY : 0u;
    const float inv_scale = 1.0f / RTGI_RADIANCE_SCALE;

    const float4 temporal = push.attach.lab_temporal.get()[lab_pixel];
    float4 filtered = push.attach.lab_filtered.get()[lab_pixel];
    const float4 raw = push.attach.lab_raw.get()[lab_pixel];
    // The spatial passes store the effective sample count / RT_LAB_SPATIAL_COUNT_SCALE in alpha, without them it is the temporal length.
    filtered.a = (push.bound & RT_LAB_PROBE_FLAG_SPATIAL_RAN) != 0u ? filtered.a * RT_LAB_SPATIAL_COUNT_SCALE : filtered.a;
    rt_lab_probe_store(base + 6u, float4(temporal.rgb * inv_scale, temporal.a));
    rt_lab_probe_store(base + 10u, float4(filtered.rgb * inv_scale, filtered.a));
    rt_lab_probe_store(base + 14u, float4(raw.rgb * inv_scale, raw.a));
    rt_lab_probe_store(base + 18u, push.attach.resolved.get()[screen_pixel].rgb);
    rt_lab_probe_store(base + 25u, push.attach.globals.exposure);
    if ((push.bound & RT_LAB_PROBE_FLAG_VARIANCE) != 0u)
    {
        flags |= RT_LAB_PROBE_FLAG_VARIANCE;
        rt_lab_probe_store(base + 22u, push.attach.lab_variance.get()[lab_pixel].rgb * (inv_scale * inv_scale));
    }
    if ((push.bound & RT_LAB_PROBE_FLAG_DIST_AB) != 0u)
    {
        flags |= RT_LAB_PROBE_FLAG_DIST_AB;
        rt_lab_probe_store(base + 26u, push.attach.dist_a.get()[lab_pixel]);
        rt_lab_probe_store(base + 30u, push.attach.dist_b.get()[lab_pixel]);
    }
    if ((push.bound & RT_LAB_PROBE_FLAG_DIST_C) != 0u)
    {
        flags |= RT_LAB_PROBE_FLAG_DIST_C;
        rt_lab_probe_store(base + 34u, push.attach.dist_c.get()[lab_pixel]);
    }
    // Weights of the first spatial pass between this probe (center) and the other one (tap).
    const uint other = 1u - probe_index;
    uint pair_flags = 0u;
    float geometry_weight = 1.0f;
    float variance_weight = 1.0f;
    float distribution_weight = 1.0f;
    if (((push.active_mask >> other) & 1u) != 0u)
    {
        const RtLabProbeSide center = rt_lab_probe_side(lab_pixel);
        const RtLabProbeSide tap = rt_lab_probe_side(push.lab_pixel[other]);
        if (center.geometry && tap.geometry)
        {
            pair_flags |= 1u;
            let settings = push.attach.globals.rt_lab_settings;
            geometry_weight = rt_lab_plane_weight(center.position, center.normal, tap.position, tap.normal, center.rcp_pixel_width_ws, settings.blur_plane_tolerance) *
                              pow(saturate(dot(center.normal, tap.normal)), settings.blur_normal_power);
            const float clamp_value = settings.sample_count_input_clamp;
            const float count_c = clamp_value > 0.0f ? min(center.temporal.a, clamp_value) : center.temporal.a;
            const float count_t = clamp_value > 0.0f ? min(tap.temporal.a, clamp_value) : tap.temporal.a;
            if ((push.bound & RT_LAB_PROBE_FLAG_VARIANCE) != 0u && settings.variance_guide_enabled != 0 && count_c > 0.0f && count_t > 0.0f)
            {
                pair_flags |= 2u;
                const float k = max(settings.variance_guide_sigmas, 1e-3f);
                variance_weight = rt_lab_variance_guide_likelihood(center.temporal.rgb, center.variance / max(count_c, 1e-3f), tap.temporal.rgb, tap.variance / max(count_t, 1e-3f),
                    rt_lab_variance_guide_floor(push.attach.globals.exposure), rcp(2.0f * k * k));
            }
            if ((push.bound & RT_LAB_PROBE_FLAG_DIST_AB) != 0u && settings.distribution_guide_mode != 0)
            {
                pair_flags |= 4u;
                distribution_weight = rt_lab_probe_distribution_weight(center, tap, settings.distribution_guide_mode >= RT_LAB_DISTRIBUTION_LOGNORMAL_3, settings.distribution_guide_compare, settings.distribution_guide_sigmas);
            }
        }
    }
    rt_lab_probe_store(base + 38u, geometry_weight);
    rt_lab_probe_store(base + 39u, variance_weight);
    rt_lab_probe_store(base + 40u, distribution_weight);
    rt_lab_probe_store(base + 41u, geometry_weight * variance_weight * distribution_weight);
    rt_lab_probe_store(base + 42u, pair_flags);
    rt_lab_probe_store(base + 0u, flags);
    rt_lab_probe_store(base + 1u, screen_pixel.x);
    rt_lab_probe_store(base + 2u, screen_pixel.y);
    rt_lab_probe_store(base + 3u, lab_pixel.x);
    rt_lab_probe_store(base + 4u, lab_pixel.y);
    rt_lab_probe_store(base + 5u, depth);
}
