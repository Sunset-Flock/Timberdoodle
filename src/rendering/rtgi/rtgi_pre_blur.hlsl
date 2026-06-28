#pragma once

#include "rtgi_pre_blur.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/debug.glsl"
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
    let rtgi_settings = push.attach.globals.rtgi_settings;
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
        return;
    }
    
    // reconstruct pixel positions based on depth
    const float2 uv = (float2(dtid.xy) + 0.5f) * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 vs_position = mul(camera.view, float4(world_position, 1.0f)).xyz;
    const float3 vs_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;

    const uint frame_shift = 1743216;
    const uint prime_shift0 = 257;   // just over typical period of frame time roughly (32 - 255 accum frames)
    const uint prime_shift1 = 9629;  // just over typical period of frame width x (480 - 8192)
    const uint prime_shift2 = 10069; // just over typical period of frame height y (480 - 8192)
    const uint frame_seed = rtgi_settings.animate_noise ? push.attach.globals.frame_index * prime_shift0 : 0u;
    const uint thread_seed =
        frame_shift * push.iteration +
        frame_seed +
        // dtid.x * prime_shift1 + dtid.y* prime_shift2 +
        // (dtid.y & 0x1) * 2 + (dtid.x & 0x1) +
        0;
    // const uint thread_seed = push.attach.globals.frame_index;
    // const uint thread_seed = 0;
    // const uint thread_seed = push.attach.globals.frame_index * 4 + (dtid.x & 1) * 2 + (dtid.y & 1);
    // const uint thread_seed = 2 * (dtid.x & 1) + (dtid.y & 1);
    rand_seed(thread_seed);

    // Load pixels diffuse before value, used for width estimation and fallback diffuse
    const float4 pixel_value = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];

    // Sample disc around normal
    const float pixel_ws_size = ws_pixel_size(inv_half_res_render_target_size.xy, camera.near_plane, pixel_depth);
    const float rcp_pixel_ws_size = rcp(pixel_ws_size);

    const float2 spatial_std_dev_data = push.attach.spatial_std_dev_image.get()[dtid.xy];
    const float pixel_log_std_dev        = spatial_std_dev_data.x;
    const float pixel_luma_mean_geometric = spatial_std_dev_data.y;

    // -1 mean signals that this pixel doesn't belong to its quad's representative surface (e.g. a depth discontinuity within the 2x2 blur-cell).
    // Pre-blur can't do anything meaningful with these pixels since their geometry context is mismatched — pass through raw and let post-blur clean them up.
    if (pixel_luma_mean_geometric < 0.0f)
    {
        push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index]  = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];
        push.attach.rtgi_diffuse2_blurred.get()[halfres_pixel_index] = push.attach.rtgi_diffuse2_before.get()[halfres_pixel_index].rg;
        return;
    }

    const bool variance_guiding_enabled = true;

    // Filter guide: R16_UINT packed — both channels [0,1] scale, sRGB compressed
    const float filter_guide    = unpack_filter_guide(push.attach.filter_guide_image.get()[dtid.xy]);
    const float geometric_guide = rtgi_settings.pre_blur_geometric_guiding ? lerp(rtgi_settings.geometric_guide_floor, 1.0f, filter_guide) : 1.0f;

    //debug_image_tile_draw(push.attach.debug_image.get(), 1, dtid, float4(TurboColormap(min(geometric_guide, detail_guide)), 2), 2);

    const float iteration_scaling = rsqrt((float)rtgi_settings.pre_blur_iterations);
    float blur_radius = max(1.0f, (float)rtgi_settings.pre_blur_base_width * geometric_guide * iteration_scaling);

    // debug_image_tile_draw(push.attach.debug_image.get(), 4, dtid, float4(TurboColormap(blur_radius * rcp((float)rtgi_settings.pre_blur_base_width * iteration_scaling)), 2.0f), 2);
    //(push.attach.debug_image.get(), 4, dtid, float4((blur_radius).xxx * rcp(32), 2), 2);

    // We want the kernel to align with the surface, 
    // but on shallow angles we would loose too much pixel footprint, 
    // so we bias the normal to face the camera more.
    const float ss_gradient_view_bias = 0.01f;
    const float3 biased_vs_normal = lerp(vs_normal, float3(0,0,1), ss_gradient_view_bias);
    const float2 ss_gradient = float2(
        sin(acos(biased_vs_normal.x)),
        sin(acos(biased_vs_normal.y)),
    );

    uint samples = max(1u, (uint)lerp(
        (float)rtgi_settings.pre_blur_sample_count_min * rcp(rtgi_settings.pre_blur_iterations),
        (float)rtgi_settings.pre_blur_sample_count_max * rcp(rtgi_settings.pre_blur_iterations),
        geometric_guide
    ));

    const bool firefly_energy_compensation_allowed = push.iteration == 0;

    float valid_sample_count = 1.0f;
    float weight_accum = ( firefly_energy_compensation_allowed ? push.attach.firefly_factor_image.get()[dtid.xy] : 1.0f ) * valid_sample_count;
    float4 blurred_accum = push.attach.rtgi_diffuse_before.get()[dtid.xy] * weight_accum;
    float2 blurred_accum2 = push.attach.rtgi_diffuse2_before.get()[dtid.xy].rg * weight_accum;

    //debug_image_tile_draw(push.attach.debug_image.get(), -1, dtid, float4(pixel_luma_mean_geometric.xxx * 3, 2), 2);

    for (uint s = 0; s < samples - 1; ++s)
    {
        const float2 disc_noise = rand_concentric_sample_disc_center_focus();
        const float2 sample_2d = disc_noise * blur_radius;
        const float3 sample_ndc = ndc + float3(ss_gradient * sample_2d * inv_half_res_render_target_size * 2.0f, 0.0f);
        // Wiggle does two things:
        // - rand_concentric_sample_disc_center_focus creates a LOT of samples at the center texel, the wiggle moves many of those out by one pixel
        // - at shallow angles the kernel can become 1 pixel wide due to the gradient based shape, the wiggle breaks this up.
        const float2 pixel_index_wiggle = disc_noise * 0.5f;
        const float2 sample_uv = sample_ndc.xy * 0.5f + 0.5f;
        const int2 sample_index_i = clamp(int2(sample_uv * half_res_render_target_size + pixel_index_wiggle), int2(0, 0), int2(half_res_render_target_size) - 1);
        const uint2 sample_index = uint2(sample_index_i);

        // Load sample data
        const float4 sample_sh_y = push.attach.rtgi_diffuse_before.get()[sample_index];
        const float2 sample_cocg = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;
        const float3 sample_value_normal_ws = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[sample_index]);
        const float3 sample_value_normal = mul(camera.view, float4(sample_value_normal_ws, 0.0f)).xyz;
        const float sample_value_depth = push.attach.view_cam_half_res_depth.get()[sample_index];
        const float3 sample_value_ndc = float3(sample_ndc.xy, sample_value_depth);
        const float4 sample_value_vs_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
        const float3 sample_value_vs = sample_value_vs_pre_div.xyz * rcp(sample_value_vs_pre_div.w);
        const float sample_prefilter_luma_mean_geometric = push.attach.spatial_std_dev_image.get()[sample_index.xy].y;
        
        // Calculate validity weights
        const float geometric_weight = same_surface_weight(rcp_pixel_ws_size, vs_position, vs_normal, sample_value_vs, sample_value_normal);
        const float normal_weight = normal_similarity_weight(vs_normal, sample_value_normal);

        // Hacky, but works:
        // * clamped fireflys are distributed to more pixels, this recovers lost energy from the firefly clamp
        // * as bright pixels are spread much more than others, this increases their temporal stability by a lot, allowing a higher firefly ceiling and more temporal stability
        const float firefly_power = firefly_energy_compensation_allowed ? push.attach.firefly_factor_image.get()[sample_index.xy] : 1.0f;

        const bool  means_valid     = pixel_luma_mean_geometric >= 0.0f && sample_prefilter_luma_mean_geometric >= 0.0f;
        const float log_ratio       = means_valid ? log(max(sample_prefilter_luma_mean_geometric, 1e-8f)) - log(max(pixel_luma_mean_geometric, 1e-8f)) : 0.0f;
        const float variance_weight = (rtgi_settings.pre_blur_geometric_mean_guiding && means_valid) ? exp(-square(rtgi_settings.pre_blur_geometric_mean_guiding_factor * 2.0f * log_ratio)) : 1.0f;

        const float weight = geometric_weight * normal_weight * firefly_power * variance_weight;
        
        #if 1
        if (all(dtid.xy == half_res_render_target_size/2))
        {
            // push.attach.debug_image.get()[sample_index] = lerp(float4(0,1,0,1), float4(1,1,1,1), weight);
            
            debug_image_tile_draw(push.attach.debug_image.get(), -1, sample_index, lerp(float4(1,0,0,2), float4(0,1,0,2), variance_weight), 2);
        }
        #endif
        
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
