#pragma once

#include "rtgi_upscale.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/pack_unpack.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiUpscaleDiffusePush rtgi_upscale_diffuse_push;

#define GS_PRELOAD_WIDTH (RTGI_UPSCALE_DIFFUSE_X/2+2)
groupshared float4 gs_half_diffuse_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float2 gs_half_diffuse2_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float4 gs_half_normals_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float4 gs_half_vs_positions[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float gs_half_samplecount[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];

#define RTGI_PERCEPTUAL_EXPONENT 4.0f
func perceptual_lerp(float a, float b, float v) -> float
{
    return pow(lerp(pow(a + 0.0000001f, (1.0f/RTGI_PERCEPTUAL_EXPONENT)), pow(b + 0.0000001f, (1.0f/RTGI_PERCEPTUAL_EXPONENT)), v), RTGI_PERCEPTUAL_EXPONENT);
}
func perceptual_lerp(float3 a, float3 b, float v) -> float3
{
    return pow(lerp(pow(a + 0.0000001f, (1.0f/RTGI_PERCEPTUAL_EXPONENT)), pow(b + 0.0000001f, (1.0f/RTGI_PERCEPTUAL_EXPONENT)), v), RTGI_PERCEPTUAL_EXPONENT);
}

[shader("compute")]
[numthreads(RTGI_UPSCALE_DIFFUSE_X,RTGI_UPSCALE_DIFFUSE_Y,1)]
func entry_upscale_diffuse(uint2 dtid : SV_DispatchThreadID, uint in_group_index : SV_GroupIndex, int2 group_id : SV_GroupID, int2 in_group_id : SV_GroupThreadID)
{
    let push = rtgi_upscale_diffuse_push;

    // Precalculate constants
    CameraInfo* camera = &push.attach.globals->view_camera;
    const uint2 full_res_pixel_index = min(dtid.xy, push.size-1);
    const float2 full_res_render_target_size = push.attach.globals.settings.render_target_size.xy;
    const float2 inv_half_res_render_target_size = rcp(float2(full_res_render_target_size / 2));
    const float2 inv_full_res_render_target_size = rcp(full_res_render_target_size);
    const float2 sv_xy = float2(full_res_pixel_index) + 0.5f;
    const float2 uv = sv_xy * inv_full_res_render_target_size;
    
    // Load pixel depth and face normal
    const float pixel_depth = push.attach.view_cam_depth.get()[full_res_pixel_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_face_normals.get()[full_res_pixel_index]);
    const float3 pixel_detail_normal = uncompress_normal_octahedral_32(push.attach.view_camera_detail_normal_image.get()[full_res_pixel_index]);

    // Calc pixel view attributes
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 position_vs_pre_div = mul(camera.inv_proj, float4(ndc, 1.0f));
    const float3 position_vs = -position_vs_pre_div.xyz / position_vs_pre_div.w;
    const float3 pixel_face_normal_vs = mul(camera.view, float4(pixel_face_normal,0.0f)).xyz;

    // Upscale Spatial Result
    float3 upscaled_diffuse = (float3)0;
    {

        // Preload Surrounding half res rtgi values
        // Each Group works on a 8x8 full res tile.
        // To full reconstruct the full res tile we need a (8/2 + 2)^2 section of the half res diffuse.
        {
            Texture2D<float4> half_res_diffuse_tex = push.attach.rtgi_diffuse_half_res.get();
            Texture2D<float2> half_res_diffuse2_tex = push.attach.rtgi_diffuse2_half_res.get();
            Texture2D<float> half_res_depth_tex = push.attach.view_cam_half_res_depth.get();
            Texture2D<uint> half_res_face_normal_tex = push.attach.view_cam_half_res_face_normals.get();

            const int2 group_base_half_index = (group_id * int(RTGI_UPSCALE_DIFFUSE_X/2)) - 1;
            const int2 preload_index = in_group_id;
            if (all(preload_index < GS_PRELOAD_WIDTH))
            {
                const int2 load_index = clamp(preload_index + group_base_half_index, int2(0,0), int2(push.size/2-1));
                const float depth = half_res_depth_tex[load_index];
                const float4 sh_y = half_res_diffuse_tex[load_index];
                const float2 cocg = half_res_diffuse2_tex[load_index];
                gs_half_diffuse_preload[preload_index.x][preload_index.y] = sh_y;
                gs_half_diffuse2_preload[preload_index.x][preload_index.y] = cocg;
                
                const float3 half_normal = uncompress_normal_octahedral_32(half_res_face_normal_tex[load_index]);
                gs_half_normals_preload[preload_index.x][preload_index.y] = float4(half_normal, 0.0f);

                const int2 sample_half_res_idx = load_index;
                const float2 sample_uv = float2(sample_half_res_idx + 0.5f) * inv_half_res_render_target_size;
                const float3 sample_ndc = float3(sample_uv * 2.0f - 1.0f, depth);
                const float4 sample_vs_pre_div = mul(camera.inv_proj, float4(sample_ndc,1.0f));
                const float3 sample_vs = -sample_vs_pre_div.xyz / sample_vs_pre_div.w;
                gs_half_vs_positions[preload_index.x][preload_index.y] = float4(sample_vs, 0.0f);

                gs_half_samplecount[preload_index.x][preload_index.y] = push.attach.rtgi_samplecount_half_res.get()[load_index];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        
        if (any(dtid.xy >= push.size))
        {
            return;
        }

        if (pixel_depth == 0.0f)
        {
            push.attach.rtgi_samplecount_full_res.get()[dtid] = 0.0f;
            return;
        }

        // Tent 3x3 filter the preloaded values
        static const uint TENT_WIDTH = 3;
        static const float GAUSS_WEIGHTS_5[5] = { 1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f };
        static const float3 TENT_WEIGHTS_LEFT_3 = { GAUSS_WEIGHTS_5[0] + GAUSS_WEIGHTS_5[1], GAUSS_WEIGHTS_5[2] + GAUSS_WEIGHTS_5[3], GAUSS_WEIGHTS_5[4] + 0.0f };
        // As each pixel is only half the size of a rtgi diffuse pixel, each rtgi diffuse pixel holds exactly 4 screen pixels.
        // We calculate the position of our screen pixel within the rtgi diffuse pixel to get a better sample weighting.
        const uint2 rtgi_subpixel_index = (full_res_pixel_index & 0x1);
        const float3 tent_weights_x = rtgi_subpixel_index.x == 0 ? TENT_WEIGHTS_LEFT_3 : TENT_WEIGHTS_LEFT_3.zyx;
        const float3 tent_weights_y = rtgi_subpixel_index.y == 0 ? TENT_WEIGHTS_LEFT_3 : TENT_WEIGHTS_LEFT_3.zyx;
        float4 acc_diffuse = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        float2 acc_diffuse2 = float2( 0.0f, 0.0f );
        float acc_weight = 0.0f;
        float4 fallback_acc_diffuse = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        float2 fallback_acc_diffuse2 = float2( 0.0f, 0.0f );
        float fallback_acc_weight = 0.0f;
        float acc_geo_weight = 0.0f;
        for (int col = 0; col < TENT_WIDTH; col++)
        {
            for (int row = 0; row < TENT_WIDTH; row++)
            {
                const int2 offset = int2(row - 1, col - 1);
                const int2 pos = int2(dtid/2) + offset;

                // Load values from gs
                const int2 sample_gs_index = in_group_id/2 + offset + int2(1,1);
                const float4 sample_sh_y = gs_half_diffuse_preload[sample_gs_index.x][sample_gs_index.y];
                const float2 sample_cocg = gs_half_diffuse2_preload[sample_gs_index.x][sample_gs_index.y];
                const float3 sample_face_normal = gs_half_normals_preload[sample_gs_index.x][sample_gs_index.y].xyz;

                // Calculate sample position
                const float3 sample_vs = gs_half_vs_positions[sample_gs_index.x][sample_gs_index.y].xyz;
                
                const float samplecount = gs_half_samplecount[sample_gs_index.x][sample_gs_index.y];

                // Calculate weights
                const float tent_weight = tent_weights_x[row] * tent_weights_y[col];
                const float geometry_weight = planar_surface_weight(inv_full_res_render_target_size, camera.near_plane, pixel_depth, position_vs, pixel_face_normal_vs, sample_vs, 3.0f);
                const float normal_weight = square(square(max(0.0f, dot(sample_face_normal, pixel_face_normal))));
                const float samplecount_weight = square(samplecount);
                const float weight = tent_weight * geometry_weight * normal_weight * samplecount_weight;


                acc_diffuse += weight * sample_sh_y;
                acc_diffuse2 += weight * sample_cocg;
                acc_weight += weight;
                acc_geo_weight += geometry_weight;

                // Fallback calculation:
                const float vs_dst_weight = 1.0f * rcp( 1.0f + square(dot(position_vs - sample_vs, position_vs - sample_vs)));
                const float fallback_weight = tent_weight * vs_dst_weight * (0.1f + max(0.0f, dot(sample_face_normal, pixel_face_normal)));
                fallback_acc_diffuse += fallback_weight * sample_sh_y;
                fallback_acc_diffuse2 += fallback_weight * sample_cocg;
                fallback_acc_weight += fallback_weight;
            }
        }

        // Write upscaled diffuse:
        float4 upscaled_sh_y = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        float2 upscaled_cocg = float2( 0.0f, 0.0f );
        if (acc_weight > 1.0f)
        {
            upscaled_sh_y = acc_diffuse * rcp(acc_weight + 0.0000001f);
            upscaled_cocg = acc_diffuse2 * rcp(acc_weight + 0.0000001f);
        }
        else
        {
            upscaled_sh_y = fallback_acc_diffuse * rcp(fallback_acc_weight + 0.0000001f);
            upscaled_cocg = fallback_acc_diffuse2 * rcp(fallback_acc_weight + 0.0000001f);
        }

        if (!push.attach.globals.rtgi_settings.upscale_enabled)
        {
            const int2 sample_gs_index = in_group_id/2 + int2(1,1);
            upscaled_sh_y = gs_half_diffuse_preload[sample_gs_index.x][sample_gs_index.y];
            upscaled_cocg = gs_half_diffuse2_preload[sample_gs_index.x][sample_gs_index.y];
        }

        if (push.attach.globals.rtgi_settings.sh_resolve_enabled)
        {
            upscaled_diffuse = sh_resolve_diffuse(upscaled_sh_y, upscaled_cocg, pixel_detail_normal);
        }
        else
        {
            upscaled_diffuse = y_co_cg_to_linear(float3(upscaled_sh_y.w, upscaled_cocg));
        }

    }

    // reproject history data
    float reprojected_samplecount = 0;
    float3 reprojected_color_history0 = float3(0.0f, 0.0f, 0.0f);
    float3 reprojected_stable_history = float3(0.0f, 0.0f, 0.0f);
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
        const float2 sv_xy_prev_frame = ( ndc_prev_frame.xy * 0.5f + 0.5f ) * inv_full_res_render_target_size;
        const float3 sv_pos_prev_frame = float3( sv_xy_prev_frame, ndc_prev_frame.z );
        const float3 primary_ray_prev_frame = normalize(world_position - previous_camera.position);

        // Load previous frame half res depth
        const Bilinear bilinear_filter_at_prev_pos = get_bilinear_filter( saturate( uv_prev_frame ), full_res_render_target_size );
        const float2 reproject_gather_uv = ( float2( bilinear_filter_at_prev_pos.origin ) + 1.0 ) * inv_full_res_render_target_size;
        SamplerState nearest_clamp_s = push.attach.globals.samplers.nearest_clamp.get();
        const float4 depth_reprojected4 = push.attach.rtgi_depth_history_full_res.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
        const uint4 face_normals_packed_reprojected4 = push.attach.rtgi_face_normal_history_full_res.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
        const float4 samplecnt_reprojected4 = push.attach.rtgi_samplecount_history_full_res.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;

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
            // Normal weights cause too much disocclusion on fine detailed geometry!
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
                    float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(0,0)) * inv_full_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[0]),
                    float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(1,0)) * inv_full_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[1]),
                    float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(0,1)) * inv_full_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[2]),
                    float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(1,1)) * inv_full_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[3]),
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
                    surface_weight(inv_full_res_render_target_size, camera.near_plane, pixel_depth, expected_world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[0], other_face_normals[0], 8),
                    surface_weight(inv_full_res_render_target_size, camera.near_plane, pixel_depth, expected_world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[1], other_face_normals[1], 8),
                    surface_weight(inv_full_res_render_target_size, camera.near_plane, pixel_depth, expected_world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[2], other_face_normals[2], 8),
                    surface_weight(inv_full_res_render_target_size, camera.near_plane, pixel_depth, expected_world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[3], other_face_normals[3], 8),
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
        // So we count the neighbiorhood pixels and scale the disocclusion threshold based on how easy it is to reproject.
        // So easy to reproject pixels have tight disocclusion, while thin things are allowed to have blurry ghosty reprojection.
        const float disocclusion_threshold = 0.025f;
        const float total_sample_weights = dot(1.0f, sample_weights);
        const bool disocclusion = total_sample_weights < disocclusion_threshold;

        // Calc new sample count
        // MUST NOT NORMALIZE SAMPLECOUNT
        // WHEN SAMPLECOUNT IS NORMALIZED, PARTIAL DISOCCLUSIONS WILL GET FULL SAMPLECOUNT FROM THE VALID SAMPLES
        // THIS CAUSES THE PARTIALLY DISOCCLUDED SAMPLES TO IMMEDIATELY TAKE ON A FULL SAMPLECOUNT
        // THEY GET STUCK IN THEIR FIRST FRAME HISTORY IMMEDIATELY
        const bool NORMALIZE_SAMPLE_COUNT = false;
        float samplecnt = apply_bilinear_custom_weights( samplecnt_reprojected4.x, samplecnt_reprojected4.y, samplecnt_reprojected4.z, samplecnt_reprojected4.w, sample_weights, NORMALIZE_SAMPLE_COUNT ).x;
        float blend_s = samplecnt;
        samplecnt = min( samplecnt + 1.0f, push.attach.globals.rtgi_settings.history_frames );
        samplecnt = disocclusion ? 0u : samplecnt;
        reprojected_samplecount = samplecnt;
        if (any(isnan(reprojected_samplecount)))
        {
            reprojected_samplecount = {};
        }
        
        // Stable Color History
        const uint4 color_history0 = push.attach.rtgi_color_history_full_res.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
        float3 decoded_color_history0[4] = {
            rgb9e5_to_float3(color_history0[0]),
            rgb9e5_to_float3(color_history0[1]),
            rgb9e5_to_float3(color_history0[2]),
            rgb9e5_to_float3(color_history0[3]),
        };
        reprojected_color_history0 = apply_bilinear_custom_weights( decoded_color_history0[0], decoded_color_history0[1], decoded_color_history0[2], decoded_color_history0[3], sample_weights );
        if (any(isnan(reprojected_color_history0)))
        {
            reprojected_color_history0 = {};
        }

        // Real Color History
        const uint4 color_history1 = push.attach.rtgi_color_history_full_res.get().GatherGreen( nearest_clamp_s, reproject_gather_uv ).wzxy;
        float3 decoded_color_history1[4] = {
            rgb9e5_to_float3(color_history1[0]),
            rgb9e5_to_float3(color_history1[1]),
            rgb9e5_to_float3(color_history1[2]),
            rgb9e5_to_float3(color_history1[3]),
        };
        reprojected_stable_history = apply_bilinear_custom_weights( decoded_color_history1[0], decoded_color_history1[1], decoded_color_history1[2], decoded_color_history1[3], sample_weights );
        if (any(isnan(reprojected_stable_history)))
        {
            reprojected_stable_history = {};
        }

        // Fast Mean Fast Varaince History
        const uint4 statistics_history = push.attach.rtgi_statistics_history_full_res.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
        float2 decoded_statistics_history[4] = {
            unpack_2x16f_uint(statistics_history[0]),
            unpack_2x16f_uint(statistics_history[1]),
            unpack_2x16f_uint(statistics_history[2]),
            unpack_2x16f_uint(statistics_history[3]),
        };
        float2 reprojected_statistics_history = apply_bilinear_custom_weights( decoded_statistics_history[0], decoded_statistics_history[1], decoded_statistics_history[2], decoded_statistics_history[3], sample_weights );
        if (any(isnan(reprojected_statistics_history)))
        {
            reprojected_statistics_history = {};
        }
        reprojected_fast_temporal_mean = reprojected_statistics_history[0];
        reprojected_fast_temporal_variance = reprojected_statistics_history[1];
    }

    const float FAST_HISTORY_FRAMES = 4.0f;

    // e5r9g9b9 banding fix:
    const float max_channel = max(upscaled_diffuse.x, max(upscaled_diffuse.y, upscaled_diffuse.z));
    upscaled_diffuse = max(upscaled_diffuse, max_channel * (1.0f / 7.0f));

    // Accumulate statistics
    float fast_mean_diff_scaling = 1.0f;
    float fast_variance_scaling = 1.0f;
    float fast_mean = 0.0f;
    float fast_std_dev_relative = 0.0f;
    if (push.attach.globals.rtgi_settings.temporal_fast_history_enabled)
    {
        const float reprojected_fast_std_dev = sqrt(reprojected_fast_temporal_variance);

        // Temporal Fast History inspired by [DD2018: Tomasz Stachowiak - Stochastic all the things](https://www.youtube.com/watch?v=MyTOGHqyquU)
        const float fast_mean_blend_factor = (1.0f / (1.0f + min(reprojected_samplecount, FAST_HISTORY_FRAMES)));

        const float fast_variance_history_frames = 4.0f;
        const float fast_variance_blend_factor = (1.0f / (1.0f + min(reprojected_samplecount, fast_variance_history_frames)));
        
        // Fast History only stores brightness to save space.
        const float new_fast_brightness = rgb_brightness(upscaled_diffuse);
        // Perceptual lerp uses power scaling to give lower values more weight.
        // For normal accumulation this causes too much variance.
        // But for the fast mean we want high reactivity to changes to dark.
        const float accumulated_fast_mean = perceptual_lerp(reprojected_fast_temporal_mean, new_fast_brightness, fast_mean_blend_factor);
        fast_mean = accumulated_fast_mean;

        // Variance is stored in relative units.
        // This makes its much more usable than raw radiance variance as fp16 is not precise enough for small values otherwise.
        const float new_fast_relative_variance = square(abs(accumulated_fast_mean - new_fast_brightness) / accumulated_fast_mean);
        const float accumulated_fast_relative_variance = perceptual_lerp(reprojected_fast_temporal_variance, new_fast_relative_variance, fast_variance_blend_factor);
        fast_std_dev_relative = sqrt(accumulated_fast_relative_variance);
        const float accumulated_relative_std_dev = sqrt(accumulated_fast_relative_variance);
        fast_variance_scaling = square(1.0f + accumulated_relative_std_dev * 2);

        const float slow_history_mean = rgb_brightness(reprojected_color_history0);
        const float slow_to_fast_mean_ratio = max(slow_history_mean, accumulated_fast_mean) / (min(slow_history_mean, accumulated_fast_mean) + 0.00000001f);
        const float relevant_fast_to_slow_mean_ratio = max(1.0f, slow_to_fast_mean_ratio);
        fast_mean_diff_scaling = square(1.0f / relevant_fast_to_slow_mean_ratio);

        push.attach.rtgi_accumulated_statistics_full_res.get()[dtid] = pack_2x16f_uint(float2(accumulated_fast_mean, accumulated_fast_relative_variance));
    }

    // Temporal firefly filter:
    if (reprojected_samplecount > FAST_HISTORY_FRAMES && push.attach.globals.rtgi_settings.temporal_fast_history_enabled && push.attach.globals.rtgi_settings.temporal_firefly_filter_enabled) 
    {
        const float fast_relative_std_dev = sqrt(reprojected_fast_temporal_variance);
        const float brightness_ratio = reprojected_fast_temporal_mean * (1.0f + fast_relative_std_dev * 0.5f) / rgb_brightness(upscaled_diffuse);
        const float clamp_factor = min(1.0f, brightness_ratio);
        upscaled_diffuse = clamp_factor * upscaled_diffuse;
        // push.attach.debug_image.get()[dtid.xy] = float4(reprojected_fast_temporal_mean, reprojected_fast_temporal_mean * fast_relative_std_dev, reprojected_fast_temporal_mean * reprojected_fast_temporal_variance, 0);
    }

    // Accumulate Color
    float history_confidence = reprojected_samplecount;
    if (reprojected_samplecount > FAST_HISTORY_FRAMES)
    {
        history_confidence *= fast_mean_diff_scaling;                                                // decreased confidence based on relative difference
        history_confidence = min(reprojected_samplecount * 2, history_confidence * fast_variance_scaling);   // increases conficende based on temporal variance
    }
    if (push.attach.globals.rtgi_settings.temporal_accumulation_enabled == 0)
    {
        history_confidence = 0.0f;
    }

    const float blend_factor = (1.0f / (1.0f + history_confidence));
    float3 accumulated_color = perceptual_lerp(reprojected_color_history0, upscaled_diffuse, blend_factor);
    
    float3 stable_history = accumulated_color;
    const bool at_max_samples = reprojected_samplecount > (push.attach.globals.rtgi_settings.history_frames - 1.0f);
    if (push.attach.globals.rtgi_settings.temporal_stabilization_enabled)
    {
        // With a high blend value, we take on MORE new values, and take LESS of the history.
        // We cannot trust history in this case because we could not accumulate for long OR there is a large difference between history and fast history.
        // In these cases we need to widen the clamp temporarily to keep the image stable.
        // One the blend becomes small again, we reduce the clamp to not get stuck on bad stable history.
        float clamp_range = lerp(0.1f, 1.0f, blend_factor); 

        stable_history = clamp(lerp(reprojected_stable_history, accumulated_color, reprojected_samplecount == 0.0f ? 1.0f : (blend_factor)), accumulated_color * rcp(1.0f + clamp_range), accumulated_color * (1.0f + clamp_range));
    }

    push.attach.rtgi_accumulated_color_full_res.get()[dtid] = uint2(
        float3_to_rgb9e5(accumulated_color),
        float3_to_rgb9e5(stable_history),
    );
    push.attach.rtgi_samplecount_full_res.get()[dtid] = reprojected_samplecount;
    push.attach.rtgi_diffuse_resolved.get()[dtid] = float4(stable_history / VALUE_MULTIPLIER, 1.0f);
}