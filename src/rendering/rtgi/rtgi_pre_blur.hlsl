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

    const PixelData pixel = calc_pixel_data(dtid, inv_half_res_render_target_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
    const float pixel_width_ws = calc_pixel_width_ws(inv_half_res_render_target_size, camera.near_plane, pixel.ndc.z);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);

    if (pixel.ndc.z == 0.0f)
    {
        return;
    }

    // Per pixel seed way too expensive.
    const uint iter_shift = 1743216;
    const uint prime_shift0 = 257;   // just over typical period of frame time roughly (32 - 255 accum frames)
    const uint prime_shift1 = 9629;  // just over typical period of frame width x (480 - 8192)
    const uint prime_shift2 = 10069; // just over typical period of frame height y (480 - 8192)
    const uint frame_seed = rtgi_settings.animate_noise ? push.attach.globals.frame_index * prime_shift0 : 0u;
    const uint thread_seed =
        iter_shift * push.iteration +
        frame_seed +
        // dtid.x * prime_shift1 + dtid.y* prime_shift2 +
        // (dtid.y & 0x1) * 2 + (dtid.x & 0x1) +
        0;
    rand_seed(thread_seed);

    // inf mean signals that this pixel doesn't belong to its quad's representative surface (e.g. a depth discontinuity within the 2x2 blur-cell).
    // Pre-blur can't do anything meaningful with these pixels since their geometry context is mismatched — pass through raw and let post-blur clean them up.
    const float pixel_luma_mean_geometric = push.attach.geo_mean_perceptual_image.get()[dtid.xy];
    if (isinf(pixel_luma_mean_geometric))
    {
        push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index]  = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];
        push.attach.rtgi_diffuse2_blurred.get()[halfres_pixel_index] = push.attach.rtgi_diffuse2_before.get()[halfres_pixel_index].rg;
        return;
    }

    // Filter guide: R16_UINT packed — both channels [0,1] scale, sRGB compressed
    const float filter_guide    = push.attach.filter_guide_image.get()[dtid.xy];
    const float raylength_guide = rtgi_settings.pre_blur_raylength_guiding ? lerp(rtgi_settings.raylength_guide_floor, 1.0f, filter_guide) : 1.0f;

    const float iteration_scaling = rsqrt((float)rtgi_settings.pre_blur_iterations);
    float blur_radius = max(1.0f, (float)rtgi_settings.pre_blur_base_width * raylength_guide * iteration_scaling);

    // We want the kernel to align with the surface, 
    // but on shallow angles we would loose too much pixel footprint, 
    // so we bias the normal to face the camera more.
    const float ss_gradient_view_bias = 0.01f;
    const float3 biased_vs_normal = lerp(pixel.normal_vs, float3(0,0,1), ss_gradient_view_bias);
    const float2 ss_gradient = float2(
        sin(acos(biased_vs_normal.x)),
        sin(acos(biased_vs_normal.y)),
    );

    uint samples = max(1u, (uint)((float)rtgi_settings.pre_blur_sample_count * rcp(rtgi_settings.pre_blur_iterations)));

    // Disc rotation: identical for all pixels within an iteration, but differs between
    // frames and between iterations. This reorients the (spatially uniform) disc pattern
    // so that temporal accumulation and successive blur iterations average over many
    // orientations instead of the same one, without adding any per-pixel cost.
    const float golden_angle = 2.39996323f; // radians, spreads successive indices maximally
    const uint frame_rot_index = rtgi_settings.animate_noise ? push.attach.globals.frame_index : 0u;
    const float rot_angle = ((float)frame_rot_index + (float)push.iteration * 0.61803399f) * golden_angle;
    float rot_sin, rot_cos;
    sincos(rot_angle, rot_sin, rot_cos);
    const float2x2 disc_rotation = float2x2(rot_cos, -rot_sin, rot_sin, rot_cos);

    const bool firefly_energy_compensation_allowed = push.iteration == 0;

    float valid_sample_count = 1.0f;
    float weight_accum = ( firefly_energy_compensation_allowed ? push.attach.firefly_factor_image.get()[dtid.xy] : 1.0f ) * valid_sample_count;
    float4 blurred_accum = push.attach.rtgi_diffuse_before.get()[dtid.xy] * weight_accum;
    float2 blurred_accum2 = push.attach.rtgi_diffuse2_before.get()[dtid.xy].rg * weight_accum;

    for (uint s = 0; s < samples - 1; ++s)
    {
        //const float2 disc_noise = rand_concentric_sample_disc_center_focus();
        const float2 disc_noise = mul(disc_rotation, g_Poisson8[min(s,7)].xy);
        const float2 sample_2d = disc_noise * blur_radius;
        const float3 sample_ndc = pixel.ndc + float3(ss_gradient * sample_2d * inv_half_res_render_target_size * 2.0f, 0.0f);
        // Wiggle does two things:
        // - rand_concentric_sample_disc_center_focus creates a LOT of samples at the center texel, the wiggle moves many of those out by one pixel
        // - at shallow angles the kernel can become 1 pixel wide due to the gradient based shape, the wiggle breaks this up.
        const float2 pixel_index_wiggle = disc_noise * 0.5f;
        const float2 sample_uv = sample_ndc.xy * 0.5f + 0.5f;
        const int2 sample_index_i = clamp(int2(sample_uv * half_res_render_target_size + pixel_index_wiggle), int2(0, 0), int2(half_res_render_target_size) - 1);
        const uint2 sample_index = uint2(sample_index_i);

        // Load sample data
        const PixelData sample = calc_pixel_data(sample_index, inv_half_res_render_target_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
        const float4 sample_sh_y = push.attach.rtgi_diffuse_before.get()[sample_index];
        const float2 sample_cocg = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;
        const float sample_luma_mean_geometric = push.attach.geo_mean_perceptual_image.get()[sample_index.xy];

        const float geometric_weight = calc_similar_surface_weight(pixel_width_ws_rcp, pixel.position_ws, pixel.normal_ws, sample.position_ws, sample.normal_ws);
        const float normal_weight = calc_similar_normal_weight(pixel.normal_ws, sample.normal_ws);

        // Hacky, but works:
        // * clamped fireflys are distributed to more pixels, this recovers lost energy from the firefly clamp
        // * as bright pixels are spread much more than others, this increases their temporal stability by a lot, allowing a higher firefly ceiling and more temporal stability
        const float firefly_power = firefly_energy_compensation_allowed ? push.attach.firefly_factor_image.get()[sample_index.xy] : 1.0f;

        const bool  means_valid                     = !(isinf(sample_luma_mean_geometric));
        const float log_ratio                       = means_valid ? sample_luma_mean_geometric - pixel_luma_mean_geometric : 0.0f;
        const float relative_geometric_luma_weight  = (rtgi_settings.geometric_luma_guiding && means_valid) ? exp(-square(rtgi_settings.geometric_luma_guiding_factor * 2.0f * log_ratio)) : 1.0f;

        const float weight = geometric_weight * normal_weight * firefly_power * relative_geometric_luma_weight;
        
        #if 1
        if (all(dtid.xy == half_res_render_target_size/2))
        {
            // push.attach.debug_image.get()[sample_index] = lerp(float4(0,1,0,1), float4(1,1,1,1), weight);
            
            debug_image_tile_draw(push.attach.debug_image.get(), -1, sample_index, lerp(float4(1,0,0,2), float4(0,1,0,2), relative_geometric_luma_weight), 2);
        }
        #endif
        
        // Sky pixels contain garbage, prevent writing anything that involved them in calculations.
        const bool is_sky = sample.ndc.z == 0.0f;
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
