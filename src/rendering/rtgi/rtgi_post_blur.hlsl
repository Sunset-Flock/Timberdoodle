#pragma once

#include "rtgi_post_blur.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiPostBlurPush rtgi_post_blur_push;
[[vk::push_constant]] RtgiAtrousPostBlurPush rtgi_atrous_post_blur_push;



// groupshared float gs_depth_preload[PRELAOD_WIDTH][RTGI_POST_BLUR_X];
// groupshared float4 gs_diffuse_preload[PRELAOD_WIDTH][RTGI_POST_BLUR_X];
// groupshared float2 gs_diffuse2_preload[PRELAOD_WIDTH][RTGI_POST_BLUR_X];

[shader("compute")]
[numthreads(RTGI_POST_BLUR_X,RTGI_POST_BLUR_Y,1)]
func entry_post_blur(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_post_blur_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    // Pass 0 = horizontal (16 threads along X, 4 along Y — no swizzle needed).
    // Pass 1 = vertical: swizzle so the 16-wide thread dimension aligns with Y reads.
    //   GroupID.x = dtid.x / RTGI_POST_BLUR_X,  LocalID.x = dtid.x % RTGI_POST_BLUR_X  (0..15)
    //   GroupID.y = dtid.y / RTGI_POST_BLUR_Y,  LocalID.y = dtid.y % RTGI_POST_BLUR_Y  (0..3)
    //   pixel.x = GroupID.x * RTGI_POST_BLUR_Y + LocalID.y
    //   pixel.y = GroupID.y * RTGI_POST_BLUR_X + LocalID.x
    const uint2 pixel_coord = push.pass == 0
        ? dtid
        : uint2((dtid.x / RTGI_POST_BLUR_X) * RTGI_POST_BLUR_Y + (dtid.y % RTGI_POST_BLUR_Y),
                (dtid.y / RTGI_POST_BLUR_Y) * RTGI_POST_BLUR_X + (dtid.x % RTGI_POST_BLUR_X));

    if (any(pixel_coord >= push.size))
    {
        return;
    }

    // Load and precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const uint2 halfres_pixel_index = pixel_coord;

    const PixelData pixel = calc_pixel_data(pixel_coord, inv_half_res_render_target_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
    const float pixel_width_ws = calc_pixel_width_ws(inv_half_res_render_target_size, camera.near_plane, pixel.ndc.z);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);

    const float pixel_samplecnt = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[halfres_pixel_index]);

    if (pixel.ndc.z == 0.0f)
    {
        return;
    }


    // We want the kernel to align with the surface,
    // but on shallow angles we would loose too much pixel footprint,
    // so we bias the normal to face the camera more.
    const float ss_gradient_view_bias = 0.1f;
    const float3 biased_vs_normal = lerp(pixel.normal_vs, float3(0,0,1), ss_gradient_view_bias);
    const float2 ss_gradient = float2(
        sin(acos(biased_vs_normal.x)),
        sin(acos(biased_vs_normal.y)),
    ) * inv_half_res_render_target_size;

    float valid_sample_count = 0.0f;
    float weight_accum = 0.0f;
    float4 blurred_accum = float4( 0.0f, 0.0f, 0.0f, 0.0f );
    float2 blurred_accum2 = float2( 0.0f, 0.0f );

    const float max_sample_count = rtgi_settings.max_temporal_samples;

    // Ambient occlusion guide × footprint quality (footprint kept separate to avoid temporal streaking; combined here).
    const float ao_guide    = push.attach.ao_guide_image.get()[pixel_coord] * push.attach.footprint_quality_image.get()[pixel_coord];
    const float ao_guide_radius_scale = rtgi_settings.post_blur_ao_guiding ? lerp(rtgi_settings.post_blur_ao_guide_floor, 1.0f, ao_guide) : 1.0f;
    // Disabled -> 0 (not 1): frame_scale only *widens* the filter on disocclusion. At 1 it would force full
    // width every frame and override the guides; at 0 the width is driven purely by the guides from frame one.
    const float frame_scale = rtgi_settings.post_blur_disocclusion_blur_enabled ? lerp(1.0f, 0.0f, square(saturate(pixel_samplecnt / 16.0f))) : 0.0f;

    const int filter_width = max(1, int((float)rtgi_settings.post_blur_max_width * max(frame_scale, ao_guide_radius_scale)));

    const float pixel_perceptual_radiance = push.attach.temporal_perceptual_radiance.get()[halfres_pixel_index];
    const float sample_count_ramp = rtgi_settings.max_temporal_samples * 0.1f;
    const float guide_ramp = saturate((pixel_samplecnt - sample_count_ramp) / max(sample_count_ramp, 1.0f));

    // write_debug_image(push.attach.debug_image.get(), 0, dtid, float4(Heatmap(filter_width * rcp((float)rtgi_settings.post_blur_max_width)), 2.0f), 2);

    for (int i = -filter_width; i <= filter_width; )
    {
        int2 xy = push.pass == 0 ? int2(i, 0) : int2(0, i);

        const int2 sample_index = clamp(int2(pixel_coord) + xy, int2(0, 0), int2(push.size - 1));

        // Load sample data
        const PixelData sample = calc_pixel_data(uint2(sample_index), inv_half_res_render_target_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
        const float4 sample_sh_y = push.attach.rtgi_diffuse_before.get()[sample_index];
        const float2 sample_cocg = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;

        // Calculate validity weights
        const float geometric_weight = calc_similar_surface_weight(pixel_width_ws_rcp, pixel.position_ws, pixel.normal_ws, sample.position_ws, sample.normal_ws);
        const float gauss_weight = calc_gaussian_weight(float(abs(i))/float(filter_width));

        // The 5 nearest taps (center ± 2) are always sampled at stride 1 regardless of post_blur_stride.
        // The quad-based pre-filter creates 2x2-cell boundary artifacts that need these nearby samples.
        const bool is_near_center = abs(i) <= 2;

        // Advance: outside the center zone use stride, but clamp so we land on -2 rather than
        // skipping past it. Inside [-2, 2) walk one pixel at a time, then resume stride after 2.
        if (i < -2)
            i = min(i + rtgi_settings.post_blur_stride, -2);
        else if (i < 2)
            i += 1;
        else
            i += rtgi_settings.post_blur_stride;

        float extra_weight = 1.0f;
        if (!is_near_center)
        {
            float raw_weight = 1.0f;
            if (rtgi_settings.post_blur_perceptual_difference_guiding)
            {
                const float sample_perceptual_radiance = push.attach.temporal_perceptual_radiance.get()[sample_index];
                raw_weight = calc_perceptual_difference_weight(pixel_perceptual_radiance, sample_perceptual_radiance, rtgi_settings.post_blur_perceptual_radiance_guide_tolerance);
            }
            extra_weight = lerp(1.0f, raw_weight, guide_ramp);
        }

        const float weight = geometric_weight * gauss_weight * extra_weight;

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

    // Calculate blurred diffuse and fallback blending
    // Some pixels find nearly no suitable spacial samples,
    // if less than 1/4th of the samples matter, we start to fallback to the original diffuse
    const float low_weight_fallback_blend = max(0.0f, 1.0f - (weight_accum / (filter_width/4.0f))); 
    const float4 blurry_sh_y = blurred_accum * rcp(weight_accum + 0.0001f);
    const float2 blurry_cocg = blurred_accum2 * rcp(weight_accum + 0.0001f);
    const float4 pixel_sh_y = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];
    const float2 pixel_cocg = push.attach.rtgi_diffuse2_before.get()[halfres_pixel_index].rg;
    const float4 blurred_sh_y = lerp(blurry_sh_y, pixel_sh_y, low_weight_fallback_blend);
    const float2 blurred_cocg = lerp(blurry_cocg, pixel_cocg, low_weight_fallback_blend);

    push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index] = blurred_sh_y;
    push.attach.rtgi_diffuse2_blurred.get()[halfres_pixel_index] = blurred_cocg;
}

// ===========================================================================================
// Groupshared (LDS) variant of the horizontal/vertical separable post blur. Identical math to
// entry_post_blur. Only the world-space POSITION + depth is preloaded into LDS (as a float4), so the
// per-tap unproject is done once per texel instead of once per tap. Everything else (normal, diffuse,
// CoCg, perceptual guide) is fetched per tap from its texture as in the plain variant. Toggled via
// post_blur_use_lds. The filter radius is capped at RTGI_PB_LDS_RADIUS (the preloaded halo).
static const int RTGI_PB_LDS_RADIUS = 16;
static const int RTGI_PB_LDS_SPAN   = RTGI_POST_BLUR_X + RTGI_PB_LDS_RADIUS * 2; // 48 texels along the blur axis
groupshared float4 gs_pb_pos_depth[RTGI_PB_LDS_SPAN][RTGI_POST_BLUR_Y]; // .xyz = world-space position, .w = ndc.z (0 == sky)

[shader("compute")]
[numthreads(RTGI_POST_BLUR_X, RTGI_POST_BLUR_Y, 1)]
func entry_post_blur_lds(uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rtgi_post_blur_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);

    const uint bl = gtid.x; // 0..RTGI_POST_BLUR_X-1, position ALONG the blur axis
    const uint pp = gtid.y; // 0..RTGI_POST_BLUR_Y-1, perpendicular position

    // Map the (swizzled for the vertical pass) thread to its pixel + the group's blur-axis start.
    int group_blur_start;
    int perp_global;
    uint2 pixel_coord;
    if (push.pass == 0)
    {
        group_blur_start = int(gid.x) * RTGI_POST_BLUR_X;
        perp_global      = int(gid.y) * RTGI_POST_BLUR_Y + int(pp);
        pixel_coord      = uint2(group_blur_start + int(bl), perp_global);
    }
    else
    {
        group_blur_start = int(gid.y) * RTGI_POST_BLUR_X;
        perp_global      = int(gid.x) * RTGI_POST_BLUR_Y + int(pp);
        pixel_coord      = uint2(perp_global, group_blur_start + int(bl));
    }

    // Cooperative preload of the world-space position + depth for [group_blur_start - R, +tile + R) along
    // the blur axis (the only thing that needs the per-texel unproject). Edges clamped to match the plain
    // variant's per-tap clamp.
    const int2 max_index = int2(push.size) - 1;
    for (uint s = bl; s < uint(RTGI_PB_LDS_SPAN); s += RTGI_POST_BLUR_X)
    {
        const int blur_global = group_blur_start - RTGI_PB_LDS_RADIUS + int(s);
        int2 tex = push.pass == 0 ? int2(blur_global, perp_global) : int2(perp_global, blur_global);
        tex = clamp(tex, int2(0, 0), max_index);
        const float  d   = push.attach.view_cam_half_res_depth.get()[uint2(tex)];
        const float2 uv  = (float2(tex) + 0.5f) * inv_half_res_render_target_size;
        const float4 ppd = mul(camera.inv_view_proj, float4(uv * 2.0f - 1.0f, d, 1.0f));
        gs_pb_pos_depth[s][pp] = float4(ppd.xyz / ppd.w, d);
    }
    GroupMemoryBarrierWithGroupSync();

    const int   center_span  = RTGI_PB_LDS_RADIUS + int(bl);
    const float4 center_pd   = gs_pb_pos_depth[center_span][pp];
    const float center_depth = center_pd.w;
    if (any(pixel_coord >= push.size) || center_depth == 0.0f)
        return;

    const float3 center_pos   = center_pd.xyz;
    const float3 center_norm  = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[pixel_coord]);
    const float  pixel_width_ws     = calc_pixel_width_ws(inv_half_res_render_target_size, camera.near_plane, center_depth);
    const float  pixel_width_ws_rcp = rcp(pixel_width_ws);

    const uint2 halfres_pixel_index = pixel_coord;
    const float pixel_samplecnt = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[halfres_pixel_index]);

    const float ao_guide    = push.attach.ao_guide_image.get()[pixel_coord] * push.attach.footprint_quality_image.get()[pixel_coord];
    const float ao_guide_radius_scale = rtgi_settings.post_blur_ao_guiding ? lerp(rtgi_settings.post_blur_ao_guide_floor, 1.0f, ao_guide) : 1.0f;
    const float frame_scale = rtgi_settings.post_blur_disocclusion_blur_enabled ? lerp(1.0f, 0.0f, square(saturate(pixel_samplecnt / 16.0f))) : 0.0f;
    // Cap the width at the preloaded halo radius.
    int filter_width = max(1, int((float)rtgi_settings.post_blur_max_width * max(frame_scale, ao_guide_radius_scale)));
    filter_width = min(filter_width, RTGI_PB_LDS_RADIUS);

    const float pixel_perceptual_radiance = push.attach.temporal_perceptual_radiance.get()[halfres_pixel_index];
    const float sample_count_ramp = rtgi_settings.max_temporal_samples * 0.1f;
    const float guide_ramp = saturate((pixel_samplecnt - sample_count_ramp) / max(sample_count_ramp, 1.0f));

    float weight_accum = 0.0f;
    float4 blurred_accum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float2 blurred_accum2 = float2(0.0f, 0.0f);

    for (int i = -filter_width; i <= filter_width; )
    {
        // Position + depth come from LDS; the sample's texel coord (for the other, per-tap texture reads).
        const int span = center_span + i;
        const float4 sample_pd = gs_pb_pos_depth[span][pp];
        const float  sample_depth = sample_pd.w;
        const float3 sample_pos   = sample_pd.xyz;
        const int2 sample_index = clamp(int2(pixel_coord) + (push.pass == 0 ? int2(i, 0) : int2(0, i)), int2(0, 0), max_index);
        const float3 sample_norm  = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[uint2(sample_index)]);
        const float4 sample_sh_y  = push.attach.rtgi_diffuse_before.get()[sample_index];
        const float2 sample_cocg  = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;

        const float geometric_weight = calc_similar_surface_weight(pixel_width_ws_rcp, center_pos, center_norm, sample_pos, sample_norm);
        const float gauss_weight = calc_gaussian_weight(float(abs(i)) / float(filter_width));

        const bool is_near_center = abs(i) <= 2;
        
        if (i < -2)      i = min(i + rtgi_settings.post_blur_stride, -2);
        else if (i < 2)  i += 1;
        else             i += rtgi_settings.post_blur_stride;

        float extra_weight = 1.0f;
        if (!is_near_center)
        {
            float raw_weight = 1.0f;
            if (rtgi_settings.post_blur_perceptual_difference_guiding)
            {
                const float sample_perceptual_radiance = push.attach.temporal_perceptual_radiance.get()[sample_index];
                raw_weight = calc_perceptual_difference_weight(pixel_perceptual_radiance, sample_perceptual_radiance, rtgi_settings.post_blur_perceptual_radiance_guide_tolerance);
            }
            extra_weight = lerp(1.0f, raw_weight, guide_ramp);
        }

        const float weight = geometric_weight * gauss_weight * extra_weight;
        if (sample_depth != 0.0f) // skip sky
        {
            weight_accum   += weight;
            blurred_accum  += weight * sample_sh_y;
            blurred_accum2 += weight * sample_cocg;
        }
    }

    const float low_weight_fallback_blend = max(0.0f, 1.0f - (weight_accum / (filter_width / 4.0f)));
    const float4 blurry_sh_y = blurred_accum * rcp(weight_accum + 0.0001f);
    const float2 blurry_cocg = blurred_accum2 * rcp(weight_accum + 0.0001f);
    const float4 pixel_sh_y  = push.attach.rtgi_diffuse_before.get()[halfres_pixel_index];
    const float2 pixel_cocg  = push.attach.rtgi_diffuse2_before.get()[halfres_pixel_index].rg;

    push.attach.rtgi_diffuse_blurred.get()[halfres_pixel_index]  = lerp(blurry_sh_y, pixel_sh_y, low_weight_fallback_blend);
    push.attach.rtgi_diffuse2_blurred.get()[halfres_pixel_index] = lerp(blurry_cocg, pixel_cocg, low_weight_fallback_blend);
}

[shader("compute")]
[numthreads(RTGI_POST_BLUR_X, RTGI_POST_BLUR_Y, 1)]
func entry_atrous_post_blur(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_atrous_post_blur_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    if (any(dtid.xy >= push.size))
        return;

    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_size = rcp(half_res_size);
    const uint2 pixel_index = dtid;

    const PixelData pixel = calc_pixel_data(dtid, inv_half_res_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
    const float pixel_width_ws = calc_pixel_width_ws(inv_half_res_size, camera.near_plane, pixel.ndc.z);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);

    if (pixel.ndc.z == 0.0f)
        return;

    const float pixel_samplecnt = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[pixel_index]);

    const float temporal_stability_scale = rtgi_settings.post_blur_disocclusion_blur_enabled ? 1.0f - square(saturate(pixel_samplecnt * rcp(16.0f))) : 0.0f;
    // Ambient occlusion guide × footprint quality (footprint kept separate to avoid temporal streaking; combined here).
    const float ao_guide    = push.attach.ao_guide_image.get()[dtid] * push.attach.footprint_quality_image.get()[dtid];

    // Ambient occlusion guide is calibrated for the pre blur that has a radius of 64, we take the lowest 25% of the guide as that rouughly matches our range here of 8-16 pixels.
    const float ao_guide_radius_scale = min(0.25f, rtgi_settings.post_blur_ao_guiding ? lerp(rtgi_settings.post_blur_ao_guide_floor, 1.0f, ao_guide) : 1.0f) * 4.0f;
    // Temporal stability pushes toward full blur when sample count is low (disocclusion)
    const float effective_ao_guide_radius_scale = lerp(ao_guide_radius_scale, 1.0f, temporal_stability_scale);

    const float pixel_perceptual_radiance_atrous = push.attach.temporal_perceptual_radiance.get()[pixel_index];

    const float sample_count_ramp = rtgi_settings.max_temporal_samples * 0.1f;
    const float guide_ramp = saturate((pixel_samplecnt - sample_count_ramp) / max(sample_count_ramp, 1.0f));

    float weight_accum = 0.0f;
    float4 diffuse_accum = float4(0, 0, 0, 0);
    float2 diffuse2_accum = float2(0, 0);

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int2 sample_index = clamp(int2(pixel_index) + int2(dx, dy) * push.step_size, int2(0, 0), int2(push.size) - 1);

            const PixelData sample = calc_pixel_data(uint2(sample_index), inv_half_res_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
            if (sample.ndc.z == 0.0f) continue;

            const float4 sample_sh_y = push.attach.rtgi_diffuse_before.get()[sample_index];
            const float2 sample_cocg = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;
            const float sample_samplecnt = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[sample_index]);

            const float geometric_weight = calc_similar_surface_weight(pixel_width_ws_rcp, pixel.position_ws, pixel.normal_ws, sample.position_ws, sample.normal_ws);
            const float normal_weight = calc_similar_normal_weight(pixel.normal_ws, sample.normal_ws);
            const float sample_count_weight = sample_samplecnt + 1.0f;

            // 3x3 separable Gaussian: [0.25, 0.5, 0.25] per axis.
            // Geometric guide scales down non-center tap contributions.
            const bool is_center = (dx == 0 && dy == 0);
            const float gauss_1d_x = dx == 0 ? 0.5f : 0.25f;
            const float gauss_1d_y = dy == 0 ? 0.5f : 0.25f;
            const float geometric_factor = is_center ? 1.0f : effective_ao_guide_radius_scale;
            const float gauss_weight = gauss_1d_x * gauss_1d_y * geometric_factor;

            float extra_weight = 1.0f;
            if (!is_center)
            {
                float raw_weight = 1.0f;
                if (rtgi_settings.post_blur_perceptual_difference_guiding)
                {
                    const float sample_perceptual_radiance = push.attach.temporal_perceptual_radiance.get()[sample_index];
                    raw_weight = calc_perceptual_difference_weight(pixel_perceptual_radiance_atrous, sample_perceptual_radiance, rtgi_settings.post_blur_perceptual_radiance_guide_tolerance);
                }
                extra_weight = lerp(1.0f, raw_weight, guide_ramp);
            }

            const float weight = geometric_weight * normal_weight * gauss_weight * sample_count_weight * extra_weight;

            weight_accum += weight;
            diffuse_accum += weight * sample_sh_y;
            diffuse2_accum += weight * sample_cocg;
        }
    }

    const float low_weight_fallback_blend = max(0.0f, 1.0f - (weight_accum / (8.0f / 4.0f)));
    const float4 blurry_sh_y = diffuse_accum * rcp(weight_accum + 0.0001f);
    const float2 blurry_cocg = diffuse2_accum * rcp(weight_accum + 0.0001f);
    const float4 pixel_sh_y = push.attach.rtgi_diffuse_before.get()[pixel_index];
    const float2 pixel_cocg = push.attach.rtgi_diffuse2_before.get()[pixel_index].rg;

    push.attach.rtgi_diffuse_blurred.get()[pixel_index] = lerp(blurry_sh_y, pixel_sh_y, low_weight_fallback_blend);
    push.attach.rtgi_diffuse2_blurred.get()[pixel_index] = lerp(blurry_cocg, pixel_cocg, low_weight_fallback_blend);
}
