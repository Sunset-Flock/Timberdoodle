#pragma once

#include "rtgi_adaptive_blur.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

#define POWER_PRE_BLUR 0
#define POWER_SAMPLE_STRENGTH 1.5f

#if POWER_PRE_BLUR
#define POWER_SAMPLE(X) pow(X, (1.0f/POWER_SAMPLE_STRENGTH))
#define DE_POWER_SAMPLE(X) pow(X, POWER_SAMPLE_STRENGTH)
#else
#define POWER_SAMPLE(X) X
#define DE_POWER_SAMPLE(X) X
#endif

[[vk::push_constant]] RtgiAdaptiveBlurPush rtgi_adaptive_blur_push;

float2 rand_concentric_sample_disc_center_focus()
{
    float r = rand();
    float theta = rand() * 2 * PI;
    return float2(cos(theta), sin(theta)) * r;
}

[shader("compute")]
[numthreads(RTGI_ADAPTIVE_BLUR_DIFFUSE_X,RTGI_ADAPTIVE_BLUR_DIFFUSE_Y,1)]
func entry_blur_diffuse(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_adaptive_blur_push;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Load and precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const uint2 halfres_pixel_index = dtid;

    #if RTGI_SPATIAL_PASSTHROUGH
        push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index] = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];
        push.attach.rtgi_diffuse2_blurred.get()[halfres_pixel_index] = push.attach.rtgi_diffuse2_before.get()[halfres_pixel_index];
        return;
    #endif

    // Load half res depth, normal and sample count
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[halfres_pixel_index];
    const float pixel_vs_depth = linearise_depth(pixel_depth, camera.near_plane);
    const float pixel_samplecnt = push.attach.rtgi_samplecnt.get()[halfres_pixel_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[halfres_pixel_index]);

    if (pixel_depth == 0.0f)
    {
        push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index] = float4(0,0,0,0);
        return;
    }
    
    // reconstruct pixel positions based on depth
    const float2 uv = (float2(dtid.xy) + 0.5f) * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 vs_position = mul(camera.view, float4(world_position, 1.0f)).xyz;
    const float3 vs_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;

    const uint thread_seed = (dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y) * push.attach.globals.frame_index;
    rand_seed(thread_seed);

    // Load pixels diffuse before value, used for width estimation and fallback diffuse
    const float4 pixel_value = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];

    // Determine if the blur radius has to be increased for the post blur
    float post_blur_min_radius_scale = 0.0f;
    const bool is_post_blur = !push.attach.rtgi_diffuse_accumulated.id.is_empty();
    #if RTGI_POST_BLUR_LUMA_DIFF_RADIUS_SCALE
        if (is_post_blur)
        {
            // Calculate relative absolute luma difference
            // Scale the blur radius up when the luma difference is large
            // At a relative luma difference of 5%, the radius is scaled to RTGI_SPATIAL_FILTER_RADIUS_MAX
            // Practically a poor mans version of variance estimation :)
            const float4 reprojected_diffuse = push.attach.rtgi_diffuse_accumulated.get()[halfres_pixel_index].rgba;
            const float luma_pixel = pixel_value.w;
            const float luma_reprojected = reprojected_diffuse.w;
            const float min_luma = min(luma_pixel, luma_reprojected);
            const float relative_luma_difference = 0.5f * (luma_pixel + luma_reprojected) * rcp(min_luma) - 1.0f;
            const float relative_luma_difference_scaling = 0.1f;
            post_blur_min_radius_scale = saturate(relative_luma_difference * rcp(relative_luma_difference_scaling));
        }
    #endif

    // Sample disc around normal
    const float pixel_ws_size = inv_half_res_render_target_size.y * camera.near_plane * rcp(pixel_depth + 0.000000001f);

    const float validity = min(1.0f, pixel_samplecnt * rcp(RTGI_SPATIAL_FILTER_DISOCCLUSION_FIX_FRAMES));
    float blur_radius = lerp(RTGI_SPATIAL_FILTER_RADIUS_MAX, push.attach.globals.rtgi_settings.spatial_filter_width, validity);

    // We want the kernel to align with the surface, 
    // but on shallow angles we would loose too much pixel footprint, 
    // so we bias the normal to face the camera more.
    const float ss_gradient_view_bias = 0.1;
    const float3 biased_vs_normal = lerp(vs_normal, float3(0,0,1), ss_gradient_view_bias);
    const float2 ss_gradient = float2(
        sin(acos(biased_vs_normal.x)),
        sin(acos(biased_vs_normal.y)),
    ) * inv_half_res_render_target_size;

    float valid_sample_count = 0.0f;
    float weight_accum = 0.0f;
    float4 blurred_accum = float4( 0.0f, 0.0f, 0.0f, 0.0f );
    float2 blurred_accum2 = float2( 0.0f, 0.0f );
    for (uint s = 0; s < RTGI_SPATIAL_FILTER_SAMPLES; ++s)
    {        
        const float2 disc_noise = rand_concentric_sample_disc_center_focus();
        const float2 sample_2d = disc_noise * blur_radius;
        const float3 sample_ndc = ndc + float3(ss_gradient * sample_2d, 0.0f);
        // Wiggle does two things:
        // - rand_concentric_sample_disc_center_focus creates a LOT of samples at the center texel, the wiggle moves many of those out by one pixel
        // - at shallow angles the kernel can become 1 pixel wide due to the gradient based shape, the wiggle breaks this up.
        const float2 pixel_index_wiggle = disc_noise * 0.5f; 
        const float2 sample_uv = sample_ndc.xy * 0.5f + 0.5f;
        const uint2 sample_index = uint2(sample_uv * half_res_render_target_size + pixel_index_wiggle);
        
        // if (all(dtid.xy == half_res_render_target_size/2))
        // {
        //     push.attach.debug_image.get()[sample_index] = float4(1,0,0,1);
        // }

        // Load sample data
        const float4 sample_sh_y = push.attach.rtgi_diffuse_before.get()[sample_index];
        const float2 sample_cocg = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;
        const float3 sample_value_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[sample_index]);
        const float sample_value_samplecnt = push.attach.rtgi_samplecnt.get()[sample_index];
        const float sample_value_depth = push.attach.view_cam_half_res_depth.get()[sample_index];
        const float3 sample_value_ndc = float3(sample_ndc.xy, sample_value_depth);
        const float4 sample_value_vs_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
        const float3 sample_value_vs = sample_value_vs_pre_div.xyz * rcp(sample_value_vs_pre_div.w);

        // Calculate validity weights
        const float depth_valid_weight = sample_value_depth != 0.0f ? 1.0f : 0.0f;
        const float geometric_weight = planar_surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, vs_position, vs_normal, sample_value_vs);
        const float normal_weight = normal_similarity_weight(pixel_face_normal, sample_value_normal);
        const float smplcnt_weight = sample_value_samplecnt >= pixel_samplecnt;
        const float weight = depth_valid_weight * geometric_weight * normal_weight * smplcnt_weight;

        // Accumulate blurred diffuse
        weight_accum += weight;
        blurred_accum += weight * sample_sh_y;
        blurred_accum2 += weight * sample_cocg;
        valid_sample_count += geometric_weight > 0.0f;
    }

    // Calculate blurred diffuse and fallback blending
    // Some pixels find nearly no suitable spacial samples,
    // if less than 1/4th of the samples matter, we start to fallback to the original diffuse
    const float low_weight_fallback_blend = max(0.0f, 1.0f - (weight_accum / (RTGI_SPATIAL_FILTER_SAMPLES/4.0f))); 
    const float4 blurry_sh_y = blurred_accum * rcp(weight_accum + 0.0001f);
    const float2 blurry_cocg = blurred_accum2 * rcp(weight_accum + 0.0001f);
    const float4 pixel_sh_y = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];
    const float2 pixel_cocg = push.attach.rtgi_diffuse2_before.get()[halfres_pixel_index].rg;
    const float4 blurred_sh_y = lerp(blurry_sh_y, pixel_sh_y, low_weight_fallback_blend);
    const float2 blurred_cocg = lerp(blurry_cocg, pixel_cocg, low_weight_fallback_blend);

    push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index] = blurred_sh_y;
    push.attach.rtgi_diffuse2_blurred.get()[halfres_pixel_index] = blurred_cocg;
}