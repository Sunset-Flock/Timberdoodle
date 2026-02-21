#pragma once

#include "rtgi_pre_blur.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiPreBlurPush rtgi_pre_blur_push;

float2 rand_concentric_sample_disc_center_focus()
{
    float r = rand();
    float theta = rand() * 2 * PI;
    return float2(cos(theta), sin(theta)) * r;
}

[shader("compute")]
[numthreads(RTGI_PRE_BLUR_X,RTGI_PRE_BLUR_Y,1)]
func entry_adaptive_blur(uint2 dtid : SV_DispatchThreadID)
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

    // Load half res depth, normal and sample count
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[halfres_pixel_index];
    const float pixel_vs_depth = linearise_depth(pixel_depth, camera.near_plane);
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

    const uint thread_seed = push.attach.globals.frame_index;
    rand_seed(thread_seed);

    // Load pixels diffuse before value, used for width estimation and fallback diffuse
    const float4 pixel_value = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];

    // Sample disc around normal
    const float pixel_ws_size = inv_half_res_render_target_size.y * camera.near_plane * rcp(pixel_depth + 0.000000001f);

    const float pixel_std_dev = push.attach.spatial_std_dev_image.get()[dtid.xy];

    const bool variance_guiding_enabled = push.attach.globals.rtgi_settings.pre_blur_variance_guiding != 0;

    // Footprint Quality Scaling
    // * for a low quality footprint, most far radius samples will be rejected -> poor temporal stability
    // * a lot quality footprint indicates complex geometry -> large radius will lead to overblur
    // * scaling down the radius for low quality footprints increases temporal stability and reduces light leak.
    const float pixel_footprint_quality = (push.attach.footprint_quality_image.get()[dtid.xy]);

    float px_size = ws_pixel_size(inv_half_res_render_target_size, camera.near_plane, pixel_depth);
    float px_size_radius_scale = 1.0f / (px_size * 25.0f);
    float blur_radius = max(2.0f, push.attach.globals.rtgi_settings.pre_blur_base_width * px_size_radius_scale * pixel_footprint_quality);

    // We want the kernel to align with the surface, 
    // but on shallow angles we would loose too much pixel footprint, 
    // so we bias the normal to face the camera more.
    const float ss_gradient_view_bias = 0.1;
    const float3 biased_vs_normal = lerp(vs_normal, float3(0,0,1), ss_gradient_view_bias);
    const float2 ss_gradient = float2(
        sin(acos(biased_vs_normal.x)),
        sin(acos(biased_vs_normal.y)),
    ) * inv_half_res_render_target_size;

    uint samples = 8u;

    float4 blurred_accum = push.attach.rtgi_diffuse_before.get()[dtid.xy];
    const float pixel_y = blurred_accum.w;
    float2 blurred_accum2 = push.attach.rtgi_diffuse2_before.get()[dtid.xy].rg;
    float valid_sample_count = 1.0f;
    float weight_accum = push.attach.firefly_factor_image.get()[dtid.xy] * RTGI_MAX_FIREFLY_FACTOR;

    for (uint s = 0; s < samples; ++s)
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

        // Load sample data
        const float4 sample_sh_y = push.attach.rtgi_diffuse_before.get()[sample_index];
        const float2 sample_cocg = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;
        const float3 sample_value_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[sample_index]);
        const float sample_value_depth = push.attach.view_cam_half_res_depth.get()[sample_index];
        const float3 sample_value_ndc = float3(sample_ndc.xy, sample_value_depth);
        const float4 sample_value_vs_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
        const float3 sample_value_vs = sample_value_vs_pre_div.xyz * rcp(sample_value_vs_pre_div.w);

        // Calculate validity weights
        const float geometric_weight = planar_surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, vs_position, vs_normal, sample_value_vs);
        const float normal_weight = normal_similarity_weight(pixel_face_normal, sample_value_normal);

        // Not physically accurate, but does two very important things:
        // * clamped fireflys are distributed to more pixels, this recovers lost energy from the firefly clamp
        // * as bright pixels are spread much more than others, this increases their temporal stability by a lot, allowing a higher firefly ceiling and more temporal stability
        const float firefly_power = push.attach.firefly_factor_image.get()[sample_index].x * RTGI_MAX_FIREFLY_FACTOR;
        
        // Variance guiding:
        // * anything before two std deviations is weighted 1.0f, past that it decreases by 1 / (1.0f + y_difference)
        // * improves shadowing contact detail
        const float relative_sample_y_deviation = (max(1.0f, max(0.0f, sample_sh_y.w - pixel_y) / (pixel_std_dev + pixel_y * 0.00001f)) * 0.5f);
        const float relative_sample_y_weight = select(variance_guiding_enabled, 1.0f / relative_sample_y_deviation, 1.0f);

        const float weight = geometric_weight * normal_weight * firefly_power * relative_sample_y_weight;
        
        // if (all(dtid.xy == half_res_render_target_size/2))
        // {
        //     push.attach.debug_image.get()[sample_index] = lerp(float4(0,1,0,1), float4(1,1,1,1), weight);
        // }
        
        // Sky pixels contain garbage, prevent writing anything that involved them in calculations.
        const bool is_sky = sample_value_depth == 0.0f;
        if (!is_sky)
        {
            // Accumulate blurred diffuse
            weight_accum += weight;
            blurred_accum += weight * sample_sh_y;
            blurred_accum2 += weight * sample_cocg;
            valid_sample_count += geometric_weight > 0.0f;
        }
    }

    float4 blurry_sh_y = blurred_accum * rcp(weight_accum + 0.00001f);
    float2 blurry_cocg = blurred_accum2 * rcp(weight_accum + 0.00001f);

    push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index] = blurry_sh_y;
    push.attach.rtgi_diffuse2_blurred.get()[halfres_pixel_index] = blurry_cocg;
}