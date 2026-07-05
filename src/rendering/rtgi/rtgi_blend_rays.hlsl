#pragma once

#include "rtgi_blend_rays.inl"

#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiBlendRaysPush rtgi_blend_rays_push;

[shader("compute")]
[numthreads(RTGI_BLEND_RAYS_X, RTGI_BLEND_RAYS_Y, 1)]
func entry_blend_rays(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_blend_rays_push;
    if (any(dtid >= push.size)) return;

    const float depth = push.attach.view_cam_half_res_depth.get()[dtid];
    if (depth == 0.0f)
    {
        push.attach.perceptual_rgb_shortness.get()[dtid] = float4(0, 0, 0, 0);
        return;
    }

    let rtgi_settings = push.attach.globals.rtgi_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    // World-space pixel width at this depth — needed to convert raw ray hit distance to bounded shortness.
    const float2 half_res_inv_render_target_size = push.attach.globals.settings.render_target_size_inv * 2.0f;
    const float ws_px_size = calc_pixel_width_ws(half_res_inv_render_target_size, camera.near_plane, depth);

    const uint ray_offset  = push.attach.pixel_ray_alloc.get()[dtid];
    const uint ray_count   = push.attach.ray_count_image.get()[dtid];

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

        mean_perceptual_rgb += linear_to_perceptual(ray_rgb, push.attach.globals.inv_exposure) * inv_count;
        acc_ray_shortness += calc_ray_shortness(res.t, ws_px_size, rtgi_settings.max_visibility_pixel_range) * inv_count;
    }

    push.attach.perceptual_rgb_shortness.get()[dtid] = float4(mean_perceptual_rgb, acc_ray_shortness);
}
