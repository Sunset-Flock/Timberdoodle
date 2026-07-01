#pragma once

#include "rtgi_temporal.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiTemporalPush rtgi_denoise_diffuse_reproject_push;

float apply_bilinear_custom_weights_soft_normalize( float s00, float s10, float s01, float s11, float4 w )
{
    float max_v = 0.0f;
    max_v = max(max_v, w.x > 0.1f ? s00 : 0.0f);
    max_v = max(max_v, w.y > 0.1f ? s10 : 0.0f);
    max_v = max(max_v, w.z > 0.1f ? s01 : 0.0f);
    max_v = max(max_v, w.w > 0.1f ? s11 : 0.0f);
    const float max_v_clamp_4 = min(max_v, 4.0f);
    const float v_acc = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    const float weight_sum = dot( w, 1.0f );
    // A larger exponent causes MORE normalization -> more streak artifacting.
    // A smaller exponent causes LESS normalization -> more temporal instability.
    const float SOFT_NORMALIZE_EXPONENT = 0.66f;
    const float soft_normalized_weight_sum = pow( weight_sum, SOFT_NORMALIZE_EXPONENT);
    return max(max_v_clamp_4, v_acc * rcp(soft_normalized_weight_sum));
    // return v_acc;
}

[shader("compute")]
[numthreads(RTGI_TEMPORAL_X,RTGI_TEMPORAL_Y,1)]
func entry_reproject_halfres(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_denoise_diffuse_reproject_push;
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

    const PixelData pixel = calc_pixel_data(dtid, inv_half_res_render_target_size, camera, push.attach.half_res_depth.get(), push.attach.half_res_normal.get());
    const float pixel_width_ws = calc_pixel_width_ws(inv_half_res_render_target_size, camera.near_plane, pixel.ndc.z);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);

    if (pixel.ndc.z == 0.0f)
    {
        push.attach.half_res_sample_count.get()[halfres_pixel_index] = 0;
        return;
    }

    bool disocclusion = false;
    float reprojected_sample_count = 0;
    float4 reprojected_diffuse = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float2 reprojected_diffuse2 = float2(0.0f, 0.0f);
    float reprojected_fast_temporal_mean = 0.0f;
    float reprojected_fast_temporal_variance = 0.0f;
    float reprojected_slow_mean = 0.0f;
    float reprojected_slow_relative_variance = 0.0f;
    float reprojected_filter_guide = 0.0f;
    float reprojected_log_geo_mean = 0.0f;
    {
        // Load relevant global data
        CameraInfo* previous_camera = &push.attach.globals->view_camera_prev_frame;

        const float3 expected_world_position_prev_frame = pixel.position_ws;
        const float4 ndc_prev_frame_pre_div = mul(previous_camera.view_proj, float4(expected_world_position_prev_frame, 1.0f));
        const float3 ndc_prev_frame = ndc_prev_frame_pre_div.xyz / ndc_prev_frame_pre_div.w;
        const float2 uv_prev_frame = ndc_prev_frame.xy * 0.5f + 0.5f;

        // Load previous frame half res depth
        const Bilinear bilinear_filter_at_prev_pos = get_bilinear_filter( saturate( uv_prev_frame ), half_res_render_target_size );
        const float2 reproject_gather_uv = ( float2( bilinear_filter_at_prev_pos.origin ) + 1.0 ) * inv_half_res_render_target_size;
        SamplerState linear_clamp_s = push.attach.globals.samplers.linear_clamp.get();
        const float4 depth_reprojected4 = push.attach.half_res_depth_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        const uint4 face_normals_packed_reprojected4 = push.attach.half_res_normal_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 samplecnt_reprojected4 = push.attach.half_res_sample_count_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;

        // Calculate plane distance based occlusion and normal similarity
        float4 occlusion = float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 normal_similarity = float4(1.0f, 1.0f, 1.0f, 1.0f);
        {
            const float in_screen = all(uv_prev_frame > 0.0f && uv_prev_frame < 1.0f) ? 1.0f : 0.0f;
            const float3 other_face_normals[] = {
                uncompress_normal_octahedral_32(face_normals_packed_reprojected4.x),
                uncompress_normal_octahedral_32(face_normals_packed_reprojected4.y),
                uncompress_normal_octahedral_32(face_normals_packed_reprojected4.z),
                uncompress_normal_octahedral_32(face_normals_packed_reprojected4.w),
            };
            // Note: hard normal weights (dot(other, pixel.normal) > -0.3) cause too much
            // dis-occlusion on fine detailed geometry, so only the soft normal_similarity is used.
            normal_similarity = {
                calc_similar_normal_weight(other_face_normals[0], pixel.normal_ws),
                calc_similar_normal_weight(other_face_normals[1], pixel.normal_ws),
                calc_similar_normal_weight(other_face_normals[2], pixel.normal_ws),
                calc_similar_normal_weight(other_face_normals[3], pixel.normal_ws),
            };

            // high quality geometric weights
            float4 surface_weights = float4( 0.0f, 0.0f, 0.0f, 0.0f );
            {
                const float3 texel_ndc_prev_frame[4] = {
                    float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(0,0)) * inv_half_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[0]),
                    float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(1,0)) * inv_half_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[1]),
                    float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(0,1)) * inv_half_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[2]),
                    float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(1,1)) * inv_half_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[3]),
                };
                const float4 texel_ws_prev_frame_pre_div[4] = {
                    mul(previous_camera.inv_view_proj, float4(texel_ndc_prev_frame[0], 1.0f)),
                    mul(previous_camera.inv_view_proj, float4(texel_ndc_prev_frame[1], 1.0f)),
                    mul(previous_camera.inv_view_proj, float4(texel_ndc_prev_frame[2], 1.0f)),
                    mul(previous_camera.inv_view_proj, float4(texel_ndc_prev_frame[3], 1.0f)),
                };
                const float3 texel_ws_prev_frame[4] = {
                    texel_ws_prev_frame_pre_div[0].xyz / texel_ws_prev_frame_pre_div[0].w,
                    texel_ws_prev_frame_pre_div[1].xyz / texel_ws_prev_frame_pre_div[1].w,
                    texel_ws_prev_frame_pre_div[2].xyz / texel_ws_prev_frame_pre_div[2].w,
                    texel_ws_prev_frame_pre_div[3].xyz / texel_ws_prev_frame_pre_div[3].w,
                };
                surface_weights = {
                    calc_similar_surface_weight_dist_limited(pixel_width_ws_rcp, expected_world_position_prev_frame, pixel.normal_ws, texel_ws_prev_frame[0], other_face_normals[0], 2),
                    calc_similar_surface_weight_dist_limited(pixel_width_ws_rcp, expected_world_position_prev_frame, pixel.normal_ws, texel_ws_prev_frame[1], other_face_normals[1], 2),
                    calc_similar_surface_weight_dist_limited(pixel_width_ws_rcp, expected_world_position_prev_frame, pixel.normal_ws, texel_ws_prev_frame[2], other_face_normals[2], 2),
                    calc_similar_surface_weight_dist_limited(pixel_width_ws_rcp, expected_world_position_prev_frame, pixel.normal_ws, texel_ws_prev_frame[3], other_face_normals[3], 2),
                };
                surface_weights[0] *= depth_reprojected4[0] != 0.0f;
                surface_weights[1] *= depth_reprojected4[1] != 0.0f;
                surface_weights[2] *= depth_reprojected4[2] != 0.0f;
                surface_weights[3] *= depth_reprojected4[3] != 0.0f;
            }

            occlusion = surface_weights * in_screen;
        }

        float4 sample_weights = get_bilinear_custom_weights( bilinear_filter_at_prev_pos, occlusion * normal_similarity );
        
        // For good quality reprojection we need multiple prev frame samples to properly avoid unwanted ghosting etc.
        // But for thin geometry its very hard or impossible to get 4 valid prev frame samples.
        // So we count the neighborhood pixels and scale the disocclusion threshold based on how easy it is to reproject.
        // So easy to reproject pixels have tight disocclusion, while thin things are allowed to have blurry ghosty reprojection.
        const float disocclusion_threshold = 0.025f;
        const float total_sample_weights = dot(1.0f, sample_weights);
        disocclusion = total_sample_weights < disocclusion_threshold;

        // Calc new sample count
        // MUST NOT NORMALIZE SAMPLECOUNT
        // WHEN SAMPLECOUNT IS NORMALIZED, PARTIAL DISOCCLUSIONS WILL GET FULL SAMPLECOUNT FROM THE VALID SAMPLES
        // THIS CAUSES THE PARTIALLY DISOCCLUDED SAMPLES TO IMMEDIATELY TAKE ON A FULL SAMPLECOUNT
        // THEY GET STUCK IN THEIR FIRST FRAME HISTORY IMMEDIATELY
        const bool NORMALIZE_SAMPLE_COUNT = false;
        // reprojected_sample_count = apply_bilinear_custom_weights( samplecnt_reprojected4.x, samplecnt_reprojected4.y, samplecnt_reprojected4.z, samplecnt_reprojected4.w, sample_weights, NORMALIZE_SAMPLE_COUNT ).x;

        // Reasoning behind the soft normalized sample count:
        // * to be correct, one has to not normalize the sample counter
        // * this however causes sample counters to be perpetually low on thin moving things because the temporal pass runs at half resolution :(.
        // * simply normalizing the sample count is also very bad, it causes "streaking" and crawling color on slow moving disocclusions :(
        // * as a compromise a mix of both is used, on partial disocclusion, the samplecount is partially normalized
        //   * this still causes the "streaking" artifacts but MUCH less so :)
        //   * its good enough to very significantly increase temporal stability :)
        //   * the streaking it causes is nearly completely hidden by the post blur :)
        reprojected_sample_count = apply_bilinear_custom_weights_soft_normalize( samplecnt_reprojected4.x, samplecnt_reprojected4.y, samplecnt_reprojected4.z, samplecnt_reprojected4.w, sample_weights ).x;
        if (any(isnan(reprojected_sample_count)))
        {
            reprojected_sample_count = {};
        }

        // Diffuse
        const float4 diffuse_r = push.attach.half_res_diffuse_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse_g = push.attach.half_res_diffuse_history.get().GatherGreen( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse_b = push.attach.half_res_diffuse_history.get().GatherBlue( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse_a = push.attach.half_res_diffuse_history.get().GatherAlpha( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse_samples[4] = {
            float4(diffuse_r[0], diffuse_g[0], diffuse_b[0], diffuse_a[0]),
            float4(diffuse_r[1], diffuse_g[1], diffuse_b[1], diffuse_a[1]),
            float4(diffuse_r[2], diffuse_g[2], diffuse_b[2], diffuse_a[2]),
            float4(diffuse_r[3], diffuse_g[3], diffuse_b[3], diffuse_a[3]),
        };
        reprojected_diffuse = apply_bilinear_custom_weights( diffuse_samples[0], diffuse_samples[1], diffuse_samples[2], diffuse_samples[3], sample_weights );
        if (any(isnan(reprojected_diffuse)))
        {
            reprojected_diffuse = {};
        }

        // Diffuse2
        const float4 diffuse2_r = push.attach.half_res_diffuse2_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse2_g = push.attach.half_res_diffuse2_history.get().GatherGreen( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float2 diffuse2_samples[4] = {
            float2(diffuse2_r[0], diffuse2_g[0]),
            float2(diffuse2_r[1], diffuse2_g[1]),
            float2(diffuse2_r[2], diffuse2_g[2]),
            float2(diffuse2_r[3], diffuse2_g[3]),
        };
        reprojected_diffuse2 = apply_bilinear_custom_weights( diffuse2_samples[0], diffuse2_samples[1], diffuse2_samples[2], diffuse2_samples[3], sample_weights );
        if (any(isnan(reprojected_diffuse2)))
        {
            reprojected_diffuse2 = {};
        }

        // Statistics History (f16x4): .x=fast_mean .y=fast_rel_var .z=slow_mean .w=slow_rel_var
        const float4 stat_r = push.attach.statistics_image_history.get().GatherRed(  linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 stat_g = push.attach.statistics_image_history.get().GatherGreen( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 stat_b = push.attach.statistics_image_history.get().GatherBlue(  linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 stat_a = push.attach.statistics_image_history.get().GatherAlpha( linear_clamp_s, reproject_gather_uv ).wzxy;
        reprojected_fast_temporal_mean     = apply_bilinear_custom_weights( stat_r[0], stat_r[1], stat_r[2], stat_r[3], sample_weights );
        reprojected_fast_temporal_variance = apply_bilinear_custom_weights( stat_g[0], stat_g[1], stat_g[2], stat_g[3], sample_weights );
        reprojected_slow_mean              = apply_bilinear_custom_weights( stat_b[0], stat_b[1], stat_b[2], stat_b[3], sample_weights );
        reprojected_slow_relative_variance = apply_bilinear_custom_weights( stat_a[0], stat_a[1], stat_a[2], stat_a[3], sample_weights );
        if (isnan(reprojected_fast_temporal_mean))     reprojected_fast_temporal_mean = 0.0f;
        if (isnan(reprojected_fast_temporal_variance)) reprojected_fast_temporal_variance = 0.0f;
        if (isnan(reprojected_slow_mean))              reprojected_slow_mean = 0.0f;
        if (isnan(reprojected_slow_relative_variance)) reprojected_slow_relative_variance = 0.0f;

        const float4 fg4 = push.attach.half_res_filter_guide_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        reprojected_filter_guide = apply_bilinear_custom_weights( fg4.x, fg4.y, fg4.z, fg4.w, sample_weights );
        if (isnan(reprojected_filter_guide)) { reprojected_filter_guide = 0.0f; }

        // Temporal geometric mean history (log-space R16_SFLOAT)
        const float4 gm4 = push.attach.temporal_geometric_mean_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        reprojected_log_geo_mean = apply_bilinear_custom_weights( gm4[0], gm4[1], gm4[2], gm4[3], sample_weights );
        if (isnan(reprojected_log_geo_mean)) { reprojected_log_geo_mean = 0.0f; }
    }
    //disocclusion = true;

    // Determine accumulated sample count
    float accumulated_sample_count = disocclusion ? 0u : min(rtgi_settings.history_frames, reprojected_sample_count + 1.0f);

    // Load new diffuse data
    const bool diffuse_pre_blurred_present = !push.attach.half_res_diffuse_new.index.is_empty();
    
    float4 new_diffuse = diffuse_pre_blurred_present ? push.attach.half_res_diffuse_new.get()[dtid.xy] : push.attach.pre_filtered_diffuse_new.get()[dtid.xy];
    float2 new_diffuse2 = diffuse_pre_blurred_present ? push.attach.half_res_diffuse2_new.get()[dtid.xy] : push.attach.pre_filtered_diffuse2_new.get()[dtid.xy];
    float new_filter_guide = push.attach.filter_guide_new.get()[dtid.xy];

    // Determine accumulated fast history

    const float FAST_HISTORY_FRAMES = 4.0f;

    // Accumulate statistics
    float fast_mean_diff_scaling = 1.0f;
    float fast_variance_scaling = 1.0f;
    float accumulated_fast_mean = 0.0f;
    float accumulated_fast_relative_variance = 0.0f;
    float fast_std_dev_relative = 0.0f;
    if (rtgi_settings.temporal_fast_history_enabled)
    {
        // Temporal Fast History inspired by [DD2018: Tomasz Stachowiak - Stochastic all the things](https://www.youtube.com/watch?v=MyTOGHqyquU)
        const float fast_blend_factor = (1.0f / (1.0f + min(accumulated_sample_count, FAST_HISTORY_FRAMES)));

        // Fast History only stores brightness to save space.
        const float new_fast_brightness = new_diffuse.w;
        accumulated_fast_mean = lerp(reprojected_fast_temporal_mean, new_fast_brightness, fast_blend_factor);

        // Relative variance EMA: point estimate uses OLD (reprojected) mean so the residual is
        // computed before the mean shifts — unbiased and dimensionless (fp16-safe at any radiance scale).
        // Clamped to 4.0 (2σ) to prevent fp16 overflow when the mean is uninitialized (zero).
        const float old_mean_safe = max(reprojected_fast_temporal_mean, 1e-6f);
        const float new_relative_variance_point = min(square((new_fast_brightness - reprojected_fast_temporal_mean) / old_mean_safe), 4.0f);
        accumulated_fast_relative_variance = lerp(reprojected_fast_temporal_variance, new_relative_variance_point, fast_blend_factor);
        fast_std_dev_relative = sqrt(accumulated_fast_relative_variance);

        // Recompute the original "wrong" point estimate (uses post-update mean) to drive fast_variance_scaling,
        // preserving the original adaptation model exactly.
        const float wrong_point_estimate = min(square(abs(accumulated_fast_mean - new_fast_brightness) / max(accumulated_fast_mean, 1e-6f)), 4.0f);
        const float adaptation_relative_variance = lerp(reprojected_fast_temporal_variance, wrong_point_estimate, fast_blend_factor);
        fast_variance_scaling = square(1.0f + sqrt(adaptation_relative_variance) * rtgi_settings.temporal_variance_fast_history_blend);

        const float slow_history_mean = reprojected_diffuse.w;
        const float slow_to_fast_mean_ratio = max(slow_history_mean, accumulated_fast_mean) / (min(slow_history_mean, accumulated_fast_mean) + 0.00000001f);
        const float relevant_fast_to_slow_mean_ratio = max(1.0f, slow_to_fast_mean_ratio);
        fast_mean_diff_scaling = square(1.0f / relevant_fast_to_slow_mean_ratio);
    }

    // Temporal firefly filter:
    if (accumulated_sample_count > FAST_HISTORY_FRAMES && rtgi_settings.temporal_fast_history_enabled && rtgi_settings.temporal_firefly_filter_enabled)
    {
        const float brightness_ratio = reprojected_fast_temporal_mean * (1.0f + sqrt(reprojected_fast_temporal_variance) * rtgi_settings.temporal_firefly_std_dev_clamp) / new_diffuse.w;
        const float clamp_factor = min(1.0f, brightness_ratio);
        new_diffuse = new_diffuse * clamp_factor;
        new_diffuse2 = new_diffuse2 * clamp_factor;
    }

    const float max_sample_count = rtgi_settings.history_frames;
    // Accumulate Color
    float history_confidence = accumulated_sample_count;
    if (accumulated_sample_count > FAST_HISTORY_FRAMES)
    {
        history_confidence = min(accumulated_sample_count * 2.0f, history_confidence * fast_variance_scaling * fast_mean_diff_scaling);
    }
    float blend = 1.0f / (1.0f + history_confidence);
    float co_cg_blend = blend;
    if (!rtgi_settings.temporal_accumulation_enabled)
    {
        blend = 1.0f;
        co_cg_blend = 1.0f;
    }

    // Determine accumulated diffuse
    float4 accumulated_diffuse = disocclusion ? new_diffuse : lerp(reprojected_diffuse, new_diffuse, blend);
    float2 accumulated_diffuse2 = disocclusion ? new_diffuse2 : lerp(reprojected_diffuse2, new_diffuse2, co_cg_blend);

    // Determine accumulated filter guide
    float accumulated_filter_guide = disocclusion ? new_filter_guide : lerp(reprojected_filter_guide, new_filter_guide, max(0.033f, blend)); // just enough to make it temporally stable

    // debug_image_tile_draw(push.attach.debug_image.get(), 0, dtid, float4(TurboColormap(accumulated_sample_count * rcp(64)), 2.0f), 2);

    // Slow statistics — same blend as color, tracks temporal mean/variance of the accumulated signal.
    // Uses pre-filtered (pre-blur bypassed) luma for a sharper variance signal.
    const float new_luma = push.attach.pre_filtered_diffuse_new.get()[dtid.xy].w;
    const float slow_old_mean_safe = max(reprojected_slow_mean, 1e-6f);
    const float slow_new_relative_variance = min(square((new_luma - reprojected_slow_mean) / slow_old_mean_safe), 4.0f);
    const float accumulated_slow_mean = disocclusion ? new_luma : lerp(reprojected_slow_mean, new_luma, blend);
    const float slow_variance_blend = max(blend, 1.0f / 8.0f);
    const float accumulated_slow_relative_variance = disocclusion ? slow_new_relative_variance : lerp(reprojected_slow_relative_variance, slow_new_relative_variance, slow_variance_blend);

    // Temporal geometric mean: EMA of log(luma) — same blend as slow_mean
    float new_log_luma = push.attach.geometric_mean_new.get()[dtid.xy];
    const bool invalid_new_log_luma = isinf(new_log_luma);
    if (invalid_new_log_luma)
    {
        new_log_luma = 0.0f;
    }
    const float accumulated_log_geo_mean = disocclusion ? new_log_luma : lerp(reprojected_log_geo_mean, new_log_luma, invalid_new_log_luma ? 0.0f : blend);

    // Write Textures
    push.attach.half_res_sample_count.get()[dtid.xy] = accumulated_sample_count;
    push.attach.half_res_diffuse_accumulated.get()[dtid.xy] = accumulated_diffuse;
    push.attach.half_res_diffuse2_accumulated.get()[dtid.xy] = accumulated_diffuse2;
    push.attach.statistics_image_accumulated.get()[dtid] = float4(accumulated_fast_mean, accumulated_fast_relative_variance, accumulated_slow_mean, accumulated_slow_relative_variance);
    push.attach.half_res_filter_guide_accumulated.get()[dtid.xy] = accumulated_filter_guide;
    push.attach.temporal_geometric_mean_accumulated.get()[dtid.xy] = accumulated_log_geo_mean;
}