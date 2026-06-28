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
    const float pixel_samplecnt = push.attach.rtgi_sample_count.get()[halfres_pixel_index];
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

    // Sample disc around normal
    const float pixel_ws_size = inv_half_res_render_target_size.y * camera.near_plane * rcp(pixel_depth + 0.000000001f);

    float px_size = ws_pixel_size(inv_half_res_render_target_size, camera.near_plane, pixel_depth);
    float px_size_radius_scale = 1.0f / (px_size * 25.0f);

    // We want the kernel to align with the surface, 
    // but on shallow angles we would loose too much pixel footprint, 
    // so we bias the normal to face the camera more.
    const float ss_gradient_view_bias = 0.1f;
    const float3 biased_vs_normal = lerp(vs_normal, float3(0,0,1), ss_gradient_view_bias);
    const float2 ss_gradient = float2(
        sin(acos(biased_vs_normal.x)),
        sin(acos(biased_vs_normal.y)),
    ) * inv_half_res_render_target_size;

    float valid_sample_count = 0.0f;
    float weight_accum = 0.0f;
    float4 blurred_accum = float4( 0.0f, 0.0f, 0.0f, 0.0f );
    float2 blurred_accum2 = float2( 0.0f, 0.0f );

    const float max_sample_count = rtgi_settings.history_frames;

    const float filter_guide    = unpack_filter_guide(push.attach.filter_guide_image.get()[dtid.xy]);
    const float geometric_guide = rtgi_settings.post_blur_geometric_guiding ? lerp(rtgi_settings.geometric_guide_floor, 1.0f, filter_guide) : 1.0f;
    const float frame_scale = rtgi_settings.post_blur_disocclusion_blur_enabled ? lerp(1.0f, 0.0f, square(saturate(pixel_samplecnt / 16.0f))) : 1.0f;

    const int filter_width = max(1, int((float)rtgi_settings.post_blur_max_width * max(frame_scale, geometric_guide)));

    const float pixel_log_geo_mean = push.attach.temporal_geometric_mean.get()[halfres_pixel_index];
    const float pixel_y = push.attach.rtgi_diffuse_before.get()[dtid.xy].w;
    // statistics_image: .x=fast_mean .y=fast_rel_var .z=slow_mean .w=slow_rel_var
    const float4 pixel_statistics = push.attach.statistics_image.get()[dtid.xy];
    const float slow_relative_std_dev = sqrt(pixel_statistics.w);
    const float half_frames_b = rtgi_settings.history_frames * 0.5f;
    const float geo_mean_guide_ramp = saturate((pixel_samplecnt - half_frames_b) / max(half_frames_b, 1.0f));

    // debug_image_tile_draw(push.attach.debug_image.get(), 0, dtid, float4(TurboColormap(filter_width * rcp((float)rtgi_settings.post_blur_max_width)), 2.0f), 2);

    for (int i = -filter_width; i <= filter_width; i += rtgi_settings.post_blur_stride)
    {
        int2 xy = push.pass == 0 ? int2(0, i) : int2(i, 0);

        const float2 sample_ndc = ndc.xy + float2(xy) * inv_half_res_render_target_size * 2;
        const int2 sample_index = clamp(int2(dtid.xy) + xy, int2(0, 0), int2(push.size - 1));
        
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
        const float gauss_weight = get_gaussian_weight(float(abs(i))/float(filter_width));
        const float sample_count_weight = (sample_value_samplecnt + 1); // hides disocclusion flicker

        const float sample_log_geo_mean = push.attach.temporal_geometric_mean.get()[sample_index];
        const float temporal_geo_log_ratio = sample_log_geo_mean - pixel_log_geo_mean;
        const float temporal_geo_factor = rtgi_settings.post_blur_geometric_mean_guiding_factor * geo_mean_guide_ramp;
        // The 5 nearest taps (center + 2 on each side) are trusted unconditionally.
        // The quad-based pre-filter creates 2x2-cell boundary artifacts; blurring freely over
        // those nearby samples is what smooths them out. Applying variance or geo-mean guiding
        // here would suppress exactly the samples needed to hide those artifacts.
#if RTGI_USE_QUAD
        const bool is_near_center = abs(i) <= 2 * rtgi_settings.post_blur_stride;
#else
        const bool is_near_center = false;
#endif
        const float temporal_geo_weight = (rtgi_settings.post_blur_geometric_mean_guiding && !is_near_center) ? exp(-square(temporal_geo_factor * 2.0f * temporal_geo_log_ratio)) : 1.0f;

        // One-sided luminance clamp: samples darker than center pass freely (valid shadows),
        // samples brighter than center are penalized — prevents bright background from bleeding
        // into darker foreground without penalising shadowed foreground samples.
        const float half_frames = rtgi_settings.history_frames * 0.5f;
        const float variance_guide_ramp = saturate((pixel_samplecnt - half_frames) / max(half_frames, 1.0f));
        const float max_allowed_y = pixel_y * (1.0f + 4.0f * slow_relative_std_dev * variance_guide_ramp);
        const float one_sided_luminance_weight = (rtgi_settings.post_blur_variance_guiding && !is_near_center) ?
            min(1.0f, max_allowed_y / (sample_sh_y.w + 1e-4f)) : 1.0f;

        const float weight = geometric_weight * normal_weight * gauss_weight * sample_count_weight * one_sided_luminance_weight * temporal_geo_weight;

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

    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[pixel_index];
    if (pixel_depth == 0.0f)
        return;

    const float pixel_samplecnt = push.attach.rtgi_sample_count.get()[pixel_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[pixel_index]);

    const float2 uv = (float2(dtid) + 0.5f) * inv_half_res_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 world_pos_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_pos = world_pos_pre_div.xyz / world_pos_pre_div.w;
    const float3 vs_pos = mul(camera.view, float4(world_pos, 1.0f)).xyz;
    const float3 vs_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;

    const float temporal_stability_scale = rtgi_settings.post_blur_disocclusion_blur_enabled ? 1.0f - square(saturate(pixel_samplecnt * rcp(16.0f))) : 0.0f;
    const float filter_guide    = unpack_filter_guide(push.attach.filter_guide_image.get()[dtid]);
    const float geometric_guide = rtgi_settings.post_blur_geometric_guiding ? lerp(rtgi_settings.geometric_guide_floor, 1.0f, filter_guide) : 1.0f;
    // Temporal stability pushes toward full blur when sample count is low (disocclusion)
    const float effective_geometric = lerp(geometric_guide, 1.0f, temporal_stability_scale);

    const float pixel_log_geo_mean_atrous = push.attach.temporal_geometric_mean.get()[pixel_index];
    const float pixel_y = push.attach.rtgi_diffuse_before.get()[dtid].w;
    const float4 pixel_statistics = push.attach.statistics_image.get()[dtid];
    const float slow_relative_std_dev = sqrt(pixel_statistics.w);

    const float half_frames = rtgi_settings.history_frames * 0.5f;
    const float variance_guide_ramp = saturate((pixel_samplecnt - half_frames) / max(half_frames, 1.0f));
    const float max_allowed_y = pixel_y * (1.0f + 4.0f * slow_relative_std_dev * variance_guide_ramp);

    float weight_accum = 0.0f;
    float4 diffuse_accum = float4(0, 0, 0, 0);
    float2 diffuse2_accum = float2(0, 0);

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int2 sample_index = clamp(int2(pixel_index) + int2(dx, dy) * push.step_size, int2(0, 0), int2(push.size) - 1);

            const float sample_depth = push.attach.view_cam_half_res_depth.get()[sample_index];
            if (sample_depth == 0.0f) continue;

            const float4 sample_sh_y = push.attach.rtgi_diffuse_before.get()[sample_index];
            const float2 sample_cocg = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;
            const float3 sample_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[sample_index]);
            const float sample_samplecnt = push.attach.rtgi_sample_count.get()[sample_index];

            const float2 sample_uv = (float2(sample_index) + 0.5f) * inv_half_res_size;
            const float3 sample_ndc = float3(sample_uv * 2.0f - 1.0f, sample_depth);
            const float4 sample_vs_pre_div = mul(camera.inv_proj, float4(sample_ndc, 1.0f));
            const float3 sample_vs = sample_vs_pre_div.xyz * rcp(sample_vs_pre_div.w);

            const float geometric_weight = planar_surface_weight(inv_half_res_size, camera.near_plane, pixel_depth, vs_pos, vs_normal, sample_vs);
            const float normal_weight = normal_similarity_weight(pixel_face_normal, sample_normal);
            const float sample_count_weight = sample_samplecnt + 1.0f;

            const float sample_log_geo_mean_atrous = push.attach.temporal_geometric_mean.get()[sample_index];
            const float atrous_geo_log_ratio = sample_log_geo_mean_atrous - pixel_log_geo_mean_atrous;
            const float atrous_geo_factor = rtgi_settings.post_blur_geometric_mean_guiding_factor * variance_guide_ramp;
            const float temporal_geo_weight = rtgi_settings.post_blur_geometric_mean_guiding ? exp(-square(atrous_geo_factor * 2.0f * atrous_geo_log_ratio)) : 1.0f;

            // 3x3 separable Gaussian: [0.25, 0.5, 0.25] per axis.
            // Geometric guide scales down non-center tap contributions.
            const bool is_center = (dx == 0 && dy == 0);
            const float gauss_1d_x = dx == 0 ? 0.5f : 0.25f;
            const float gauss_1d_y = dy == 0 ? 0.5f : 0.25f;
            const float geometric_factor = is_center ? 1.0f : effective_geometric;
            const float gauss_weight = gauss_1d_x * gauss_1d_y * geometric_factor;

            const float one_sided_luminance_weight = rtgi_settings.post_blur_variance_guiding ?
                min(1.0f, max_allowed_y / (sample_sh_y.w + 1e-4f)) : 1.0f;

            const float weight = geometric_weight * normal_weight * gauss_weight * sample_count_weight * one_sided_luminance_weight * temporal_geo_weight;

            weight_accum += weight;
            diffuse_accum += weight * sample_sh_y;
            diffuse2_accum += weight * sample_cocg;
        }
    }

    const float low_weight_fallback_blend = max(0.0f, 1.0f - (weight_accum / (RTGI_SPATIAL_FILTER_SAMPLES / 4.0f)));
    const float4 blurry_sh_y = diffuse_accum * rcp(weight_accum + 0.0001f);
    const float2 blurry_cocg = diffuse2_accum * rcp(weight_accum + 0.0001f);
    const float4 pixel_sh_y = push.attach.rtgi_diffuse_before.get()[pixel_index];
    const float2 pixel_cocg = push.attach.rtgi_diffuse2_before.get()[pixel_index].rg;

    push.attach.rtgi_diffuse_blurred.get()[pixel_index] = lerp(blurry_sh_y, pixel_sh_y, low_weight_fallback_blend);
    push.attach.rtgi_diffuse2_blurred.get()[pixel_index] = lerp(blurry_cocg, pixel_cocg, low_weight_fallback_blend);
}
