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
[[vk::push_constant]] RtgiPreBlurPush rtgi_pre_blur_push;
[[vk::push_constant]] RtgiAtrousBlurPush rtgi_atrous_blur_push;

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
    const float blur_radius_smplcnt_scale = 1.0f / (pixel_samplecnt + 1.0f);

    const float max_blur_radius = RTGI_SPATIAL_FILTER_RADIUS_MAX - 1u /*subtract pixel wiggle*/;
    const float min_blur_radius = max(post_blur_min_radius_scale * max_blur_radius, RTGI_SPATIAL_FILTER_RADIUS_MIN);
    const float blur_radius = max(min_blur_radius, max_blur_radius * blur_radius_smplcnt_scale);

    const float3 pixel_to_camera_dir = normalize(world_position - push.attach.globals.view_camera.position);
    const float2 ss_gradient = float2(
        sin(acos(vs_normal.x)),
        sin(acos(vs_normal.y)),
    ) * inv_half_res_render_target_size;

    // push.attach.debug_image.get()[dtid] = float4(ss_gradient,0,1);

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
        const float geometric_weight = get_geometry_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, vs_position, vs_normal, sample_value_vs);
        const float normal_weight = get_normal_diffuse_weight(pixel_face_normal, sample_value_normal);
        const float weight = depth_valid_weight * geometric_weight * normal_weight;

        // Accumulate blurred diffuse
        weight_accum += weight;
            blurred_accum += weight * sample_sh_y;
            blurred_accum2 += weight * sample_cocg;
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

[shader("compute")]
[numthreads(RTGI_PRE_BLUR_DIFFUSE_X,RTGI_PRE_BLUR_DIFFUSE_Y,1)]
func entry_pre_blur_diffuse(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_pre_blur_push;
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
        push.attach.rtgi_diffuse_raw_pre_blurred.get()[halfres_pixel_index] = push.attach.rtgi_diffuse_raw.get()[halfres_pixel_index];
        push.attach.rtgi_diffuse2_raw_pre_blurred.get()[halfres_pixel_index] = push.attach.rtgi_diffuse2_raw.get()[halfres_pixel_index];
        return;
    #endif

    // Load half res depth and normal
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[halfres_pixel_index];
    const float pixel_vs_depth = linearise_depth(pixel_depth, camera.near_plane);
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[halfres_pixel_index]);
    
    // reconstruct pixel positions based on depth
    const float2 uv = (float2(dtid.xy) + 0.5f) * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 vs_position = mul(camera.view, float4(world_position, 1.0f)).xyz;
    const float3 vs_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;

    if (pixel_depth == 0.0f)
    {
        return;
    }

    // Construct tangent bases matrix and setup rand for sample generation
    const float3 world_tangent = normalize(cross(pixel_face_normal, float3(0,0,1) + 0.0001));
    const float3 world_bitangent = cross(world_tangent, pixel_face_normal);
    // const uint thread_seed = ((dtid.x & 0x1) * 2 + (dtid.y & 0x1)) * push.attach.globals.frame_index;
    // const uint thread_seed = push.attach.globals.frame_index;
    const uint thread_seed = (dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y) * push.attach.globals.frame_index;
    rand_seed(thread_seed);

    // Sample disc around normal
    const uint SAMPLE_COUNT = RTGI_SPATIAL_FILTER_SAMPLES_PRE_BLUR;
    const float BLUR_PIXEL_RADIUS = RTGI_SPATIAL_FILTER_RADIUS_PRE_BLUR_MAX;
    const float pixel_ws_size = inv_half_res_render_target_size.y * camera.near_plane * rcp(pixel_depth + 0.000000001f);
    const float scaled_pixel_radius = BLUR_PIXEL_RADIUS;
    const float blur_radius = max(RTGI_SPATIAL_FILTER_RADIUS_PRE_BLUR_MIN, scaled_pixel_radius);
    float weight_accum = 0.0f;
    float4 blurred_accum = float4( 0.0f, 0.0f, 0.0f, 0.0f );
    float2 blurred_accum2 = float2( 0.0f, 0.0f );
    float luma_squared_accum = 0.0f;
    float max_luma = 0.0f;
    float max_luma_squared = 0.0f;
    float max_luma_weight = 0.0f;
    for (uint s = 0; s < SAMPLE_COUNT; ++s)
    {
        const float sample_s = float(s) * rcp(SAMPLE_COUNT);
        const float sample_r = rand();
        const float2 sample_disc_dir = float2(cos(sample_r * 3.14 * 2), sin(sample_r * 3.14 * 2));

        // Calculate sample position
        const float sample_weighting = 1.0f;
        const float2 sample_2d = sample_s * blur_radius * pixel_ws_size * sample_disc_dir;
        const float3 sample_ws = world_position + world_tangent * sample_2d.x + world_bitangent * sample_2d.y;
        const float4 sample_ndc_prev_div = mul(camera.view_proj, float4(sample_ws, 1.0f));
        const float3 sample_ndc = sample_ndc_prev_div.xyz / sample_ndc_prev_div.w;
        const float2 sample_uv = sample_ndc.xy * 0.5f + 0.5f;
        const uint2 sample_index = uint2(sample_uv * half_res_render_target_size);

        // Load sample data
        const float4 sample_value_sh_y = push.attach.rtgi_diffuse_high_energy.get()[sample_index];
        const float2 sample_value_cocg = push.attach.rtgi_diffuse2_high_energy.get()[sample_index].rg;
        const float3 sample_value_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[sample_index]);
        const float sample_value_depth = push.attach.view_cam_half_res_depth.get()[sample_index];
        const float3 sample_value_ndc = float3(sample_ndc.xy, sample_value_depth);
        const float4 sample_value_ws_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
        const float3 sample_value_ws = sample_value_ws_pre_div.xyz / sample_value_ws_pre_div.w;

        // Calculate validity weights
        const float depth_valid_weight = sample_value_depth != 0.0f ? 1.0f : 0.0f;
        const float geometric_weight = get_geometry_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, vs_position, vs_normal, sample_value_ws);
        const float normal_weight = get_normal_diffuse_weight(pixel_face_normal, sample_value_normal);
        const float weight = sample_weighting * depth_valid_weight * geometric_weight * normal_weight;

        // Accumulate blurred diffuse
        blurred_accum += weight * sample_value_sh_y * rcp(SAMPLE_COUNT);
        blurred_accum2 += weight * sample_value_cocg * rcp(SAMPLE_COUNT);
    }

    push.attach.debug_image.get()[dtid] = float4(rcp(blurred_accum.w).xxx, 1.0f);

    const float4 pixel_sh_y = push.attach.rtgi_diffuse_reprojected.get()[halfres_pixel_index];
    const float2 pixel_cocg = push.attach.rtgi_diffuse2_reprojected.get()[halfres_pixel_index].rg;
    push.attach.rtgi_diffuse_reprojected.get()[halfres_pixel_index] = pixel_sh_y + blurred_accum;
    push.attach.rtgi_diffuse2_reprojected.get()[halfres_pixel_index] = pixel_cocg + blurred_accum2;
}