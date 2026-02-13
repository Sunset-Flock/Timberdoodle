#pragma once

#include "rtgi_temporal.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/pack_unpack.hlsl"
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
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Load and precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const CameraInfo camera_prev_frame = push.attach.globals->view_camera_prev_frame;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const uint2 halfres_pixel_index = dtid;
    const float2 sv_xy = float2(halfres_pixel_index) + 0.5f;

    // Load half res depth and normal
    const float pixel_depth = push.attach.half_res_depth.get()[halfres_pixel_index];
    const float pixel_vs_depth = linearise_depth(pixel_depth, camera.near_plane);
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.half_res_normal.get()[halfres_pixel_index]);

    if (pixel_depth == 0.0f)
    {
        push.attach.half_res_sample_count.get()[halfres_pixel_index] = 0;
        return;
    }

    // Calculate pixel positions in cur and prev frame
    const float3 sv_pos = float3(sv_xy, pixel_depth);
    const float2 uv = sv_xy * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 world_position_prev_frame = world_position; // No support for dynamic object motion vectors yet.
    const float4 view_position_prev_frame_pre_div = mul(camera_prev_frame.view, float4(world_position_prev_frame, 1.0f));
    const float3 view_position_prev_frame = -view_position_prev_frame_pre_div.xyz / view_position_prev_frame_pre_div.w;
    const float4 ndc_prev_frame_pre_div = mul(camera_prev_frame.view_proj, float4(world_position_prev_frame, 1.0f));
    const float3 ndc_prev_frame = ndc_prev_frame_pre_div.xyz / ndc_prev_frame_pre_div.w;
    const float2 uv_prev_frame = ndc_prev_frame.xy * 0.5f + 0.5f;// - inv_half_res_render_target_size * 0.75;
    const float2 sv_xy_prev_frame = ( ndc_prev_frame.xy * 0.5f + 0.5f ) * half_res_render_target_size;
    const float3 sv_pos_prev_frame = float3( sv_xy_prev_frame, ndc_prev_frame.z );
    const float3 primary_ray_prev_frame = normalize(world_position - camera_prev_frame.position);

    bool disocclusion = false;
    float reprojected_sample_count = 0;
    float4 reprojected_diffuse = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float2 reprojected_diffuse2 = float2(0.0f, 0.0f);
    float reprojected_fast_temporal_mean = 0.0f;
    float reprojected_fast_temporal_variance = 0.0f;
    {
        // Load relevant global data
        CameraInfo* previous_camera = &push.attach.globals->view_camera_prev_frame;

        // Calculate pixel positions in cur and prev frame
        const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
        const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
        const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
        const float3 expected_world_position_prev_frame = world_position; // No support for dynamic object motion vectors yet.
        const float4 view_position_prev_frame_pre_div = mul(previous_camera.view, float4(expected_world_position_prev_frame, 1.0f));
        const float3 view_position_prev_frame = -view_position_prev_frame_pre_div.xyz / view_position_prev_frame_pre_div.w;
        const float4 ndc_prev_frame_pre_div = mul(previous_camera.view_proj, float4(expected_world_position_prev_frame, 1.0f));
        const float3 ndc_prev_frame = ndc_prev_frame_pre_div.xyz / ndc_prev_frame_pre_div.w;
        const float2 uv_prev_frame = ndc_prev_frame.xy * 0.5f + 0.5f;
        const float2 sv_xy_prev_frame = ( ndc_prev_frame.xy * 0.5f + 0.5f ) * inv_half_res_render_target_size;
        const float3 sv_pos_prev_frame = float3( sv_xy_prev_frame, ndc_prev_frame.z );
        const float3 primary_ray_prev_frame = normalize(world_position - previous_camera.position);

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
            // Normal weights cause too much dis-occlusion on fine detailed geometry!
            float4 normal_weights = {
                1,//dot(other_face_normals[0], pixel_face_normal) > -0.3f,
                1,//dot(other_face_normals[1], pixel_face_normal) > -0.3f,
                1,//dot(other_face_normals[2], pixel_face_normal) > -0.3f,
                1,//dot(other_face_normals[3], pixel_face_normal) > -0.3f,
            };
            normal_similarity = {
                normal_similarity_weight(other_face_normals[0], pixel_face_normal),
                normal_similarity_weight(other_face_normals[1], pixel_face_normal),
                normal_similarity_weight(other_face_normals[2], pixel_face_normal),
                normal_similarity_weight(other_face_normals[3], pixel_face_normal),
            };
            // normalize normal weights so it does not interact with SAMPLE_WEIGHT_DISSOCCLUSION_THRESHOLD.
            normal_weights;// *= rcp(normal_weights.x + normal_weights.y + normal_weights.z + normal_weights.w + 0.001f);

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
                    surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, expected_world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[0], other_face_normals[0], 1),
                    surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, expected_world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[1], other_face_normals[1], 1),
                    surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, expected_world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[2], other_face_normals[2], 1),
                    surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, expected_world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[3], other_face_normals[3], 1),
                };
                surface_weights[0] *= depth_reprojected4[0] != 0.0f;
                surface_weights[1] *= depth_reprojected4[1] != 0.0f;
                surface_weights[2] *= depth_reprojected4[2] != 0.0f;
                surface_weights[3] *= depth_reprojected4[3] != 0.0f;
            }

            occlusion = surface_weights * in_screen * normal_weights;
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

        // Fast Mean Fast Varaince History
        const uint4 packed_statistics_history_samples = push.attach.half_res_statistics_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        float2 statistics_history_samples[4] = {
            unpack_2x16f_uint(packed_statistics_history_samples[0]),
            unpack_2x16f_uint(packed_statistics_history_samples[1]),
            unpack_2x16f_uint(packed_statistics_history_samples[2]),
            unpack_2x16f_uint(packed_statistics_history_samples[3]),
        };
        float2 reprojected_statistics_history = apply_bilinear_custom_weights( statistics_history_samples[0], statistics_history_samples[1], statistics_history_samples[2], statistics_history_samples[3], sample_weights );
        if (any(isnan(reprojected_statistics_history)))
        {
            reprojected_statistics_history = {};
        }
        reprojected_fast_temporal_mean = reprojected_statistics_history[0];
        reprojected_fast_temporal_variance = reprojected_statistics_history[1];
    }
    //disocclusion = true;

    // Determine accumulated sample count
    float accumulated_sample_count = disocclusion ? 0u : min(push.attach.globals.rtgi_settings.history_frames, reprojected_sample_count + 1.0f);

    // Load new diffuse data
    float4 new_diffuse = push.attach.half_res_diffuse_new.get()[dtid.xy];
    float2 new_diffuse2 = push.attach.half_res_diffuse2_new.get()[dtid.xy];

    // Determine accumulated fast history

    const float FAST_HISTORY_FRAMES = 4.0f;

    // Accumulate statistics
    float fast_mean_diff_scaling = 1.0f;
    float fast_variance_scaling = 1.0f;
    float accumulated_fast_mean = 0.0f;
    float accumulated_fast_relative_variance = 0.0f;
    float fast_std_dev_relative = 0.0f;
    if (push.attach.globals.rtgi_settings.temporal_fast_history_enabled)
    {
        const float reprojected_fast_std_dev = sqrt(reprojected_fast_temporal_variance);

        // Temporal Fast History inspired by [DD2018: Tomasz Stachowiak - Stochastic all the things](https://www.youtube.com/watch?v=MyTOGHqyquU)
        const float fast_mean_blend_factor = (1.0f / (1.0f + min(accumulated_sample_count, FAST_HISTORY_FRAMES)));

        const float fast_variance_history_frames = 4.0f;
        const float fast_variance_blend_factor = (1.0f / (1.0f + min(accumulated_sample_count, fast_variance_history_frames)));
        
        // Fast History only stores brightness to save space.
        const float new_fast_brightness = new_diffuse.w;
        // Perceptual lerp uses power scaling to give lower values more weight.
        // For normal accumulation this causes too much variance.
        // But for the fast mean we want high reactivity to changes to dark.
        accumulated_fast_mean = lerp(reprojected_fast_temporal_mean, new_fast_brightness, fast_mean_blend_factor);

        // Variance is stored in relative units.
        // This makes its much more usable than raw radiance variance as fp16 is not precise enough for small values otherwise.
        const float new_fast_relative_variance = square(abs(accumulated_fast_mean - new_fast_brightness) / accumulated_fast_mean);
        accumulated_fast_relative_variance = lerp(reprojected_fast_temporal_variance, new_fast_relative_variance, fast_variance_blend_factor);
        fast_std_dev_relative = sqrt(accumulated_fast_relative_variance);
        const float accumulated_relative_std_dev = sqrt(accumulated_fast_relative_variance);
        fast_variance_scaling = square(1.0f + accumulated_relative_std_dev * 2);

        const float slow_history_mean = reprojected_diffuse.w;
        const float slow_to_fast_mean_ratio = max(slow_history_mean, accumulated_fast_mean) / (min(slow_history_mean, accumulated_fast_mean) + 0.00000001f);
        const float relevant_fast_to_slow_mean_ratio = max(1.0f, slow_to_fast_mean_ratio);
        fast_mean_diff_scaling = square(1.0f / relevant_fast_to_slow_mean_ratio);
    }

    // Temporal firefly filter:
    if (accumulated_sample_count > FAST_HISTORY_FRAMES && push.attach.globals.rtgi_settings.temporal_fast_history_enabled && push.attach.globals.rtgi_settings.temporal_firefly_filter_enabled) 
    {
        const float fast_relative_std_dev = sqrt(reprojected_fast_temporal_variance);
        const float brightness_ratio = reprojected_fast_temporal_mean * (1.0f + fast_relative_std_dev * 0.5f) / new_diffuse.w;
        const float clamp_factor = min(1.0f, brightness_ratio);
        new_diffuse = new_diffuse * clamp_factor;
        new_diffuse2 = new_diffuse2 * clamp_factor;
    }

    const float max_sample_count = push.attach.globals.rtgi_settings.history_frames;
    // Accumulate Color
    float history_confidence = accumulated_sample_count;
    if (accumulated_sample_count > FAST_HISTORY_FRAMES)
    {
        history_confidence *= fast_mean_diff_scaling;                                                         // decreased confidence based on relative difference
        history_confidence = min(accumulated_sample_count * 2, history_confidence * fast_variance_scaling);   // increases conficende based on temporal variance
    }
    const float blend = 1.0f / (1.0f + history_confidence);

    // Determine accumulated diffuse
    float4 accumulated_diffuse = disocclusion ? new_diffuse : lerp(reprojected_diffuse, new_diffuse, blend);
    float2 accumulated_diffuse2 = disocclusion ? new_diffuse2 : lerp(reprojected_diffuse2, new_diffuse2, blend);

    // Write Textures
    push.attach.half_res_sample_count.get()[dtid.xy] = accumulated_sample_count;
    push.attach.half_res_diffuse_accumulated.get()[dtid.xy] = accumulated_diffuse;
    push.attach.half_res_diffuse2_accumulated.get()[dtid.xy] = accumulated_diffuse2;
    push.attach.half_res_statistics_accumulated.get()[dtid] = pack_2x16f_uint(float2(accumulated_fast_mean, accumulated_fast_relative_variance));
}