#pragma once

#include "rt_lab_resolve.inl"

#include "rt_lab_shared.hlsl"
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtLabResolvePush rt_lab_resolve_push;

func rt_lab_resolve_luminance(float3 v) -> float
{
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

// Live post-spatial split history error views: the split difference went through the same blur + upsample weights as
// the diffuse, so |d| is the error estimate of THIS output.
// Effective sample count view: heatmap of count / sample_count_debug_max.
func rt_lab_resolve_sample_count_debug(uint2 dtid, float count, bool sky)
{
    let push = rt_lab_resolve_push;
    if (push.attach.globals.settings.debug_draw_mode != DEBUG_DRAW_MODE_RT_LAB_SPATIAL_SAMPLE_COUNT)
    {
        return;
    }
    const float maximum = max(push.attach.globals.rt_lab_settings.sample_count_debug_max, 1e-3f);
    const float3 color = sky ? float3(0.0f, 0.0f, 0.0f) : Heatmap(saturate(count / maximum));
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
}

func rt_lab_resolve_split_debug(uint2 dtid, float3 resolved_radiance, float2 split_difference, bool sky)
{
    let push = rt_lab_resolve_push;
    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    if (debug_mode != DEBUG_DRAW_MODE_RT_LAB_SPLIT_ERROR && debug_mode != DEBUG_DRAW_MODE_RT_LAB_SPLIT_RELATIVE_ERROR)
    {
        return;
    }
    float3 color = float3(0.0f, 0.0f, 0.0f);
    if (sky)
    {
        color = float3(0.0f, 0.0f, 0.0f);
    }
    else if (push.split_enabled == 0u)
    {
        color = float3(1.0f, 0.0f, 1.0f) * 0.25f; // split history off
    }
    else if (split_difference.y < 0.999f)
    {
        color = float3(1.0f, 0.0f, 1.0f); // part of the footprint has a half without samples: error unknown
    }
    else
    {
        const float exposure_scale = PI * push.attach.globals.exposure;
        const float error = abs(split_difference.x) / RTGI_RADIANCE_SCALE * exposure_scale;
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_SPLIT_ERROR)
        {
            color = Heatmap(error * push.attach.globals.settings.debug_visualization_scale);
        }
        else
        {
            // |A - B| / min(A, B). Post-spatial only d = (A - B) / 2 is carried; the output is ~(A + B) / 2, so
            // A, B ~ output +- |d|.
            const float luminance = rt_lab_resolve_luminance(resolved_radiance * exposure_scale);
            color = rt_lab_relative_difference_heatmap(rt_lab_relative_difference(luminance + error, luminance - error));
        }
    }
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
}

// Resolve of one full res pixel (no writes): the diffuse the opaque shading reads (mean incoming radiance,
// RTGI_RADIANCE_SCALE divided out), the split difference, the effective sample counts.
// Full res lab: a copy. Half res lab: the RTGI upscale weights (3x3 wide gaussian footprint x plane x normal, see below).
// The split difference (split history on) takes the same path with the same weights (still RTGI_RADIANCE_SCALE scaled).
#define RT_LAB_UPSCALE_SIGMA 1.0f           // = RTGI_UPSCALE_WIDE_FOOTPRINT_SIGMA
#define RT_LAB_UPSCALE_PLANE_TOLERANCE 3.0f // half res pixel widths, = rtgi_upscale.hlsl

struct RtLabResolved
{
    float3 radiance;
    float2 split_difference;
    float count;         // effective sample count of the output
    bool sky;
};

func rt_lab_resolve_pixel(uint2 dtid) -> RtLabResolved
{
    let push = rt_lab_resolve_push;
    RtLabResolved r = {};
    r.radiance = float3(0.0f, 0.0f, 0.0f);
    r.split_difference = float2(0.0f, 0.0f);
    r.count = 0.0f;
    r.sky = false;
    const CameraInfo camera = push.attach.globals.view_camera;
    const bool split = push.split_enabled != 0u;

    const float ndc_depth = push.attach.view_cam_depth.get()[dtid];
    if (ndc_depth == 0.0f)
    {
        r.sky = true;
        return r;
    }
    const bool clamp_count = push.clamp_input_count != 0u;
    const float count_clamp = push.attach.globals.rt_lab_settings.sample_count_input_clamp;

    uint2 lab_size;
    push.attach.lab_diffuse.get().GetDimensions(lab_size.x, lab_size.y);
    if (all(lab_size == push.size))
    {
        const float4 lab_value = push.attach.lab_diffuse.get()[dtid];
        float3 value = lab_value.rgb;
        r.count = rt_lab_spatial_read_count(lab_value.a, clamp_count, count_clamp);
        r.split_difference = split ? push.attach.lab_split_difference.get()[dtid] : float2(0.0f, 0.0f);
        r.radiance = value / RTGI_RADIANCE_SCALE;
        return r;
    }

    // Half res lab: the RTGI upscale weights (rtgi_upscale.hlsl, wide footprint): 3x3 half res taps around the pixel,
    // per axis gaussian (sigma RT_LAB_UPSCALE_SIGMA half res px) of the distance sub-pixel center -> tap center,
    // x binary plane test (RT_LAB_UPSCALE_PLANE_TOLERANCE half res pixel widths) x face normal dot^4. No matching tap:
    // RTGI fallback weights (gaussian x 1 / (1 + dist^4) x (0.1 + dot)).
    const float2 size = float2(push.size);
    const float depth = camera.near_plane / ndc_depth;
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(size)), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.view_cam_face_normals.get()[dtid]);
    // = rtgi_upscale.hlsl: calc_pixel_width_ws(inv half res size, near, ndc depth) = 2 x linear depth / half res height.
    const float rcp_half_pixel_width_ws = rcp(calc_pixel_width_ws(rcp(float2(lab_size)), camera.near_plane, ndc_depth));
    const float2 inv_lab_size = rcp(float2(lab_size));

    // Distance (half res px) from a left / top sub-pixel center to the taps at offset -1, 0, +1; mirrored for right / bottom.
    const float3 tap_distances_left = float3(0.75f, 0.25f, 1.25f);
    const float3 gauss_left_unnormalized = exp(-(tap_distances_left * tap_distances_left) / (2.0f * RT_LAB_UPSCALE_SIGMA * RT_LAB_UPSCALE_SIGMA));
    const float3 gauss_left = gauss_left_unnormalized / dot(gauss_left_unnormalized, 1.0f);
    const float3 gauss_x = (dtid.x & 1u) == 0u ? gauss_left : gauss_left.zyx;
    const float3 gauss_y = (dtid.y & 1u) == 0u ? gauss_left : gauss_left.zyx;

    float3 acc = float3(0.0f, 0.0f, 0.0f);
    float2 acc_split = float2(0.0f, 0.0f);
    float acc_count = 0.0f;
    float acc_weight = 0.0f;
    float3 fallback_acc = float3(0.0f, 0.0f, 0.0f);
    float2 fallback_acc_split = float2(0.0f, 0.0f);
    float fallback_acc_count = 0.0f;
    float fallback_acc_weight = 0.0f;
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            const int2 tap = clamp(int2(dtid / 2u) + int2(x - 1, y - 1), int2(0, 0), int2(lab_size) - 1);
            const float tap_depth = push.attach.lab_depth.get()[tap];
            if (tap_depth == 0.0f)
            {
                continue;
            }
            const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[tap]);
            const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_lab_size), tap_depth);
            const float3 offset = tap_position - position;
            const float gauss_weight = gauss_x[x] * gauss_y[y];
            const float plane_distance = abs(dot(offset, normal)) * rcp_half_pixel_width_ws;
            const float geometry_weight = plane_distance < RT_LAB_UPSCALE_PLANE_TOLERANCE ? 1.0f : 0.0f;
            const float normal_dot = max(0.0f, dot(tap_normal, normal));
            const float normal_weight = (normal_dot * normal_dot) * (normal_dot * normal_dot);
            const float weight = gauss_weight * geometry_weight * normal_weight;
            const float distance_squared = dot(offset, offset);
            const float fallback_weight = gauss_weight * rcp(1.0f + distance_squared * distance_squared) * (0.1f + normal_dot);

            const float4 lab_value = push.attach.lab_diffuse.get()[tap];
            const float count = rt_lab_spatial_read_count(lab_value.a, clamp_count, count_clamp);
            acc += weight * lab_value.rgb;
            acc_count += weight * count;
            fallback_acc += fallback_weight * lab_value.rgb;
            fallback_acc_count += fallback_weight * count;
            if (split)
            {
                const float2 split_value = push.attach.lab_split_difference.get()[tap];
                acc_split += weight * split_value;
                fallback_acc_split += fallback_weight * split_value;
            }
            acc_weight += weight;
            fallback_acc_weight += fallback_weight;
        }
    }
    // Only fall back when NO tap matched (thin geometry that does not exist at half res).
    if (!(acc_weight > 1e-5f))
    {
        acc = fallback_acc;
        acc_split = fallback_acc_split;
        acc_count = fallback_acc_count;
        acc_weight = fallback_acc_weight;
    }
    if (!(acc_weight > 0.0f))
    {
        return r; // all taps sky
    }
    // Normalized weights: upsampling interpolates the counts, no new samples.
    const float rcp_weight = rcp(acc_weight);
    float3 result = acc * rcp_weight;
    r.count = acc_count * rcp_weight;
    if (split)
    {
        r.split_difference = acc_split * rcp_weight;
    }
    r.radiance = result / RTGI_RADIANCE_SCALE;
    return r;
}

[shader("compute")]
[numthreads(RT_LAB_RESOLVE_X, RT_LAB_RESOLVE_Y, 1)]
func entry_rt_lab_resolve(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_resolve_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const RtLabResolved r = rt_lab_resolve_pixel(dtid);

    push.attach.resolved.get()[dtid] = float4(r.radiance, 1.0f);
    if (push.split_enabled != 0u)
    {
        push.attach.split_difference_resolved.get()[dtid] = r.split_difference;
    }
    rt_lab_resolve_split_debug(dtid, r.radiance, r.split_difference, r.sky);
    rt_lab_resolve_sample_count_debug(dtid, r.count, r.sky);
}
