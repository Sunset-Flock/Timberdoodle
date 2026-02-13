#pragma once

#include "rtgi_post_blur.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiPostBlurPush rtgi_post_blur_push;

[shader("compute")]
[numthreads(RTGI_POST_BLUR_X,RTGI_POST_BLUR_Y,1)]
func entry_post_blur(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_post_blur_push;
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
    const float pixel_samplecnt = push.attach.rtgi_sample_count.get()[halfres_pixel_index];
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

    // Load pixels diffuse before value, used for width estimation and fallback diffuse
    const float4 pixel_value = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];

    // Sample disc around normal
    const float pixel_ws_size = inv_half_res_render_target_size.y * camera.near_plane * rcp(pixel_depth + 0.000000001f);

#if RTGI_SPATIAL_FILTER_DISOCCLUSION_EXPANSION
    const float validity = min(1.0f, pixel_samplecnt * rcp(push.attach.globals.rtgi_settings.history_frames));
#else
    const float validity = 1.0f;
#endif
    float px_size = ws_pixel_size(inv_half_res_render_target_size, camera.near_plane, pixel_depth);
    float px_size_radius_scale = 1.0f / (px_size * 25.0f);
    float blur_radius = max(6.0f, lerp(RTGI_SPATIAL_FILTER_RADIUS_MAX, push.attach.globals.rtgi_settings.pre_blur_base_width * px_size_radius_scale, validity));

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

    const float max_sample_count = push.attach.globals.rtgi_settings.history_frames;

    const int FILTER_WIDTH = int(lerp(12.0f, 4.0f, min(1.0f, pixel_samplecnt * rcp(12.0f))));
    const int FILTER_STRIDE = 1;

    for (int i = -FILTER_WIDTH; i <= FILTER_WIDTH; ++i)
    {
        int2 xy = push.pass == 0 ? int2(0, i) : int2(i, 0);

        const float2 sample_ndc = ndc.xy + float2(xy) * inv_half_res_render_target_size * 2 * FILTER_STRIDE;
        const int2 sample_index = clamp(int2(dtid.xy) + xy * FILTER_STRIDE, int2(0, 0), int2(push.size - 1));
        
        // if (all(dtid.xy == half_res_render_target_size/2))
        // {
        //     push.attach.debug_image.get()[sample_index] = float4(1,0,0,1);
        // }

        // Load sample data
        const float4 sample_sh_y = push.attach.rtgi_diffuse_before.get()[sample_index];
        const float2 sample_cocg = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;
        const float3 sample_value_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[sample_index]);
        const float sample_value_samplecnt = push.attach.rtgi_sample_count.get()[sample_index];
        const float sample_value_depth = push.attach.view_cam_half_res_depth.get()[sample_index];
        const float3 sample_value_ndc = float3(sample_ndc, sample_value_depth);
        const float4 sample_value_vs_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
        const float3 sample_value_vs = sample_value_vs_pre_div.xyz * rcp(sample_value_vs_pre_div.w);

        // Calculate validity weights
        const float geometric_weight = planar_surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, vs_position, vs_normal, sample_value_vs);
        const float normal_weight = normal_similarity_weight(pixel_face_normal, sample_value_normal);
        const float gauss_weight = pixel_samplecnt < (max_sample_count * 0.75f) ? 1.0f : get_gaussian_weight(float(abs(i))/float(FILTER_WIDTH));
        const float sample_count_weight = square(sample_value_samplecnt + 1); // hides disocclusion flicker
        const float weight = geometric_weight * normal_weight * gauss_weight * sample_count_weight;
        
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