#pragma once

#include "rtgi_blend_rays.inl"

#include "shader_lib/misc.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/debug.glsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiBlendRaysPush rtgi_blend_rays_push;

// Debug ray picker: draws the rays shot from the picked pixel this frame and writes their mean / variance into the general readback.
func rtgi_debug_pick_pixel_rays(uint2 pixel, float depth, uint ray_offset, uint ray_count, uint specular_ray_count)
{
    let push = rtgi_blend_rays_push;
    const CameraInfo camera = push.attach.globals.view_camera;
    const float2 ndc = (float2(pixel * 2u) + 1.0f) * camera.inv_screen_size * 2.0f - 1.0f;
    const float3 origin = rtgi_half_res_depth_to_world_space(camera, ndc, depth);
    const float max_length = push.attach.globals.rtgi_settings.debug_ray_length;
    const float3 luma_weights = float3(0.2126f, 0.7152f, 0.0722f);

    float3 sum = float3(0, 0, 0);
    float3 sum_squared = float3(0, 0, 0);
    float luma_sum = 0.0f;
    float luma_sum_squared = 0.0f;
    float hit_distance_sum = 0.0f;
    uint hit_count = 0u;
    uint miss_count = 0u;
    for (uint i = 0u; i < ray_count + specular_ray_count; ++i)
    {
        const RtgiRayResult res = push.attach.ray_result[ray_offset + i];
        const float3 rgb = res.radiance * rcp(RTGI_RADIANCE_SCALE);
        const bool specular = i >= ray_count;
        const bool miss = res.t >= RTGI_SPECULAR_MAX_HIT_DISTANCE;
        if (!specular)
        {
            const float luma = dot(rgb, luma_weights);
            sum += rgb;
            sum_squared += rgb * rgb;
            luma_sum += luma;
            luma_sum_squared += luma * luma;
            if (miss)
            {
                miss_count += 1u;
            }
            else
            {
                hit_count += 1u;
                hit_distance_sum += res.t;
            }
        }
        const float3 direction = uncompress_normal_octahedral_32(res.packed_dir);
        const float length = min(res.t, max_length);
        ShaderDebugLineDraw line = {};
        line.start = origin;
        line.end = origin + direction * length;
        line.color = rgb + 0.02f;
        line.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        debug_draw_line(push.attach.globals.debug, line);
        if (res.t < max_length)
        {
            ShaderDebugCircleDraw hit = {};
            hit.position = line.end;
            hit.color = line.color;
            hit.radius = 0.02f;
            hit.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
            debug_draw_circle(push.attach.globals.debug, hit);
        }
    }

    const float n = float(ray_count);
    const float3 mean = sum * rcp(max(n, 1.0f));
    const float luma_mean = luma_sum * rcp(max(n, 1.0f));
    // Unbiased sample variance: (sum x^2 - n mean^2) / (n - 1).
    const float inv_n_minus_1 = ray_count > 1u ? rcp(n - 1.0f) : 0.0f;
    const float3 variance = max(sum_squared - n * mean * mean, 0.0f) * inv_n_minus_1;
    const float luma_variance = max(luma_sum_squared - n * luma_mean * luma_mean, 0.0f) * inv_n_minus_1;
    push.attach.globals.readback.rtgi_debug_ray_mean = float4(mean, n);
    push.attach.globals.readback.rtgi_debug_ray_variance = float4(variance, luma_variance);
    push.attach.globals.readback.rtgi_debug_ray_info = float4(hit_count > 0u ? hit_distance_sum * rcp(float(hit_count)) : 0.0f, float(miss_count), float(specular_ray_count), 1.0f);
}

[shader("compute")]
[numthreads(RTGI_BLEND_RAYS_X, RTGI_BLEND_RAYS_Y, 1)]
func entry_blend_rays(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_blend_rays_push;
    if (any(dtid >= push.size)) return;

    // The picked pixel: under the cursor or frozen (full res coordinates -> half res).
    bool debug_pick = false;
    if (push.attach.globals.rtgi_settings.debug_ray_pick_enabled != 0)
    {
        const int2 full_res_pixel = push.attach.globals.rtgi_settings.debug_ray_pick_frozen != 0
            ? int2(push.attach.globals.rtgi_settings.debug_ray_pixel_x, push.attach.globals.rtgi_settings.debug_ray_pixel_y)
            : int2(floor(push.attach.globals.cursor_uv * push.attach.globals.settings.render_target_size));
        debug_pick = all(int2(dtid) == (full_res_pixel >> 1));
        if (debug_pick)
        {
            // Red crosshair on the picked full res pixel (NDC of the main camera, slightly in front of the surface).
            const CameraInfo cross_camera = push.attach.globals.view_camera;
            const float cross_depth = push.attach.view_cam_half_res_depth.get()[dtid] * 1.001f;
            const float2 center = (float2(full_res_pixel) + 0.5f) * cross_camera.inv_screen_size * 2.0f - 1.0f;
            const float2 arm = float2(10.0f, 10.0f) * cross_camera.inv_screen_size * 2.0f;
            const float2 gap = float2(3.0f, 3.0f) * cross_camera.inv_screen_size * 2.0f;
            for (uint i = 0u; i < 4u; ++i)
            {
                const float2 dir = i == 0u ? float2(1, 0) : (i == 1u ? float2(-1, 0) : (i == 2u ? float2(0, 1) : float2(0, -1)));
                ShaderDebugLineDraw line = {};
                line.start = float3(center + dir * gap, cross_depth);
                line.end = float3(center + dir * arm, cross_depth);
                line.color = float3(1.0f, 0.0f, 0.0f);
                line.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC_MAIN_CAMERA;
                debug_draw_line(push.attach.globals.debug, line);
            }
        }
    }

    const float depth = push.attach.view_cam_half_res_depth.get()[dtid];
    if (depth == 0.0f)
    {
        if (debug_pick)
        {
            push.attach.globals.readback.rtgi_debug_ray_mean = float4(0, 0, 0, 0);
            push.attach.globals.readback.rtgi_debug_ray_variance = float4(0, 0, 0, 0);
            push.attach.globals.readback.rtgi_debug_ray_info = float4(0, 0, 0, 0);
        }
        push.attach.perceptual_rgb_shortness.get()[dtid] = float4(0, 0, 0, 0);
        push.attach.specular_perceptual_rgb_hit.get()[dtid] = float4(0, 0, 0, 0);
        return;
    }

    let rtgi_settings = push.attach.globals.rtgi_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    // World-space pixel width at this depth — needed to convert raw ray hit distance to bounded shortness.
    const float2 half_res_inv_render_target_size = push.attach.globals.settings.render_target_size_inv * 2.0f;
    const float ws_px_size = rtgi_half_res_pixel_width_ws(half_res_inv_render_target_size, camera.near_plane, depth);

    const uint ray_offset  = push.attach.pixel_ray_alloc.get()[dtid];
    const uint ray_count   = push.attach.ray_count_image.get()[dtid];

    const uint specular_ray_count = push.attach.specular_ray_count_image.get()[dtid];
    if (debug_pick)
    {
        rtgi_debug_pick_pixel_rays(dtid, depth, ray_offset, ray_count, specular_ray_count);
    }
    float3 mean_specular_perceptual_rgb = float3(0, 0, 0);
    float  mean_specular_hit = 0.0f;
    for (uint s = 0u; s < specular_ray_count; s++)
    {
        const RtgiRayResult spec = push.attach.ray_result[ray_offset + ray_count + s];
        mean_specular_perceptual_rgb += linear_to_perceptual_rgb(spec.radiance, push.attach.globals.inv_exposure) * rcp(float(specular_ray_count));
        mean_specular_hit += spec.t * rcp(float(specular_ray_count));
    }
    push.attach.specular_perceptual_rgb_hit.get()[dtid] = float4(mean_specular_perceptual_rgb, mean_specular_hit);

    if (ray_count == 0u)
    {
        push.attach.perceptual_rgb_shortness.get()[dtid] = float4(0, 0, 0, 0);
        return;
    }

    const float inv_count = rcp(float(ray_count));

    // Mean log(rgb) geometric mean and mean ray shortness, averaged over the rays.
    float3 mean_perceptual_rgb = float3(0, 0, 0);
    float  acc_ray_shortness   = 0.0f; // res.t holds the raw hit distance; convert to shortness [0,1] and average over all rays

    for (uint s = 0u; s < ray_count; s++)
    {
        const RtgiRayResult res = push.attach.ray_result[ray_offset + s];
        const float3 ray_rgb    = res.radiance;

        mean_perceptual_rgb += linear_to_perceptual_rgb(ray_rgb, push.attach.globals.inv_exposure) * inv_count;
        acc_ray_shortness += calc_ray_shortness(res.t, ws_px_size, rtgi_settings.max_visibility_pixel_range) * inv_count;
    }

    push.attach.perceptual_rgb_shortness.get()[dtid] = float4(mean_perceptual_rgb, acc_ray_shortness);
}
