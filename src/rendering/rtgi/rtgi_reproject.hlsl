#pragma once

#include "rtgi_reproject.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiReprojectDiffusePush rtgi_denoise_diffuse_reproject_push;
[[vk::push_constant]] RtgiDiffuseTemporalStabilizationPush rtgi_diffuse_stabilize_history_push;

[shader("compute")]
[numthreads(RTGI_DENOISE_DIFFUSE_X,RTGI_DENOISE_DIFFUSE_Y,1)]
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
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[halfres_pixel_index];
    const float pixel_vs_depth = linearise_depth(pixel_depth, camera.near_plane);
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[halfres_pixel_index]);

    if (pixel_depth == 0.0f)
    {
        push.attach.rtgi_samplecnt.get()[halfres_pixel_index] = 0;
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

    // Load previous frame half res depth
    const Bilinear bilinear_filter_at_prev_pos = get_bilinear_filter( saturate( uv_prev_frame ), half_res_render_target_size );
    const float2 reproject_gather_uv = ( float2( bilinear_filter_at_prev_pos.origin ) + 1.0 ) * inv_half_res_render_target_size;
    SamplerState nearest_clamp_s = push.attach.globals.samplers.nearest_clamp.get();
    const float4 depth_reprojected4 = push.attach.rtgi_depth_history.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float4 samplecnt_reprojected4 = push.attach.rtgi_samplecnt_history.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const uint4 face_normals_packed_reprojected4 = push.attach.rtgi_face_normal_history.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float3 view_space_normal = mul(camera_prev_frame.view, float4(pixel_face_normal, 0.0f)).xyz;

    const float4 ndc_space_normal_pre_div = mul(camera_prev_frame.view_proj, float4(pixel_face_normal, 0.0f));
    const float3 ndc_space_normal = ndc_space_normal_pre_div.xyz;

    // Calculate plane distance based occlusion and normal similarity
    float4 occlusion = float4(1.0f, 1.0f, 1.0f, 1.0f);
    {
        const float in_screen = all(uv_prev_frame > 0.0f && uv_prev_frame < 1.0f) ? 1.0f : 0.0f;
        const float3 other_face_normals[] = {
            uncompress_normal_octahedral_32(face_normals_packed_reprojected4.x),
            uncompress_normal_octahedral_32(face_normals_packed_reprojected4.y),
            uncompress_normal_octahedral_32(face_normals_packed_reprojected4.z),
            uncompress_normal_octahedral_32(face_normals_packed_reprojected4.w),
        };

        float4 normal_weights = {
            1,//dot(other_face_normals[0], pixel_face_normal) > 0.5f,
            1,//dot(other_face_normals[1], pixel_face_normal) > 0.5f,
            1,//dot(other_face_normals[2], pixel_face_normal) > 0.5f,
            1,//dot(other_face_normals[3], pixel_face_normal) > 0.5f,
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
                mul(camera_prev_frame.inv_view_proj, float4(texel_ndc_prev_frame[0], 1.0f)),
                mul(camera_prev_frame.inv_view_proj, float4(texel_ndc_prev_frame[1], 1.0f)),
                mul(camera_prev_frame.inv_view_proj, float4(texel_ndc_prev_frame[2], 1.0f)),
                mul(camera_prev_frame.inv_view_proj, float4(texel_ndc_prev_frame[3], 1.0f)),
            };
            const float3 texel_ws_prev_frame[4] = {
                texel_ws_prev_frame_pre_div[0].xyz / texel_ws_prev_frame_pre_div[0].w,
                texel_ws_prev_frame_pre_div[1].xyz / texel_ws_prev_frame_pre_div[1].w,
                texel_ws_prev_frame_pre_div[2].xyz / texel_ws_prev_frame_pre_div[2].w,
                texel_ws_prev_frame_pre_div[3].xyz / texel_ws_prev_frame_pre_div[3].w,
            };
            surface_weights = {
                surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[0], other_face_normals[0], 8),
                surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[1], other_face_normals[1], 8),
                surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[2], other_face_normals[2], 8),
                surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[3], other_face_normals[3], 8),
            };
            surface_weights[0] *= depth_reprojected4[0] != 0.0f;
            surface_weights[1] *= depth_reprojected4[1] != 0.0f;
            surface_weights[2] *= depth_reprojected4[2] != 0.0f;
            surface_weights[3] *= depth_reprojected4[3] != 0.0f;
        }

        occlusion = surface_weights * in_screen * normal_weights;
    }

    float4 sample_weights = get_bilinear_custom_weights( bilinear_filter_at_prev_pos, occlusion );
    
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
    push.attach.rtgi_samplecnt.get()[halfres_pixel_index] = samplecnt;
    
    #if 0
    // Read in diffuse history
    const float4 history_Y0 = push.attach.rtgi_diffuse_history.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float4 history_Y1 = push.attach.rtgi_diffuse_history.get().GatherGreen( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float4 history_Y2 = push.attach.rtgi_diffuse_history.get().GatherBlue( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float4 history_Y3 = push.attach.rtgi_diffuse_history.get().GatherAlpha( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float4 history_Co = push.attach.rtgi_diffuse2_history.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float4 history_Cg = push.attach.rtgi_diffuse2_history.get().GatherGreen( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float4 y00 = float4(history_Y0.x, history_Y1.x, history_Y2.x, history_Y3.x);
    const float4 y10 = float4(history_Y0.y, history_Y1.y, history_Y2.y, history_Y3.y);
    const float4 y01 = float4(history_Y0.z, history_Y1.z, history_Y2.z, history_Y3.z);
    const float4 y11 = float4(history_Y0.w, history_Y1.w, history_Y2.w, history_Y3.w);
    const float4 cocg00 = float4(history_Co.x, history_Cg.x, 0.0f, 0.0f);
    const float4 cocg10 = float4(history_Co.y, history_Cg.y, 0.0f, 0.0f);
    const float4 cocg01 = float4(history_Co.z, history_Cg.z, 0.0f, 0.0f);
    const float4 cocg11 = float4(history_Co.w, history_Cg.w, 0.0f, 0.0f);
    float4 y_history = apply_bilinear_custom_weights( y00, y10, y01, y11, sample_weights );
    float2 cocg_history = apply_bilinear_custom_weights( cocg00, cocg10, cocg01, cocg11, sample_weights ).rg;

    if (any(isnan(y_history)) || any(isnan(cocg_history)))
    {
        y_history = float4(0,0,0,0);
        cocg_history = float2(0,0);
    }

    push.attach.rtgi_diffuse_reprojected.get()[dtid] = y_history;
    push.attach.rtgi_diffuse2_reprojected.get()[dtid] = cocg_history;
    #endif
}

#define RTGI_TS_BORDER 2
#define PRELOAD_WIDTH (RTGI_DIFFUSE_TEMPORAL_STABILIZATION_X + RTGI_TS_BORDER * 2)
groupshared float gs_luma[PRELOAD_WIDTH][PRELOAD_WIDTH];
groupshared float4 gs_depth_normals_preload[PRELOAD_WIDTH][PRELOAD_WIDTH];
groupshared float gs_samplecount_preload[PRELOAD_WIDTH][PRELOAD_WIDTH];

func local_offset_to_gs_index(int2 in_group_id, int2 offset) -> int2
{
    return in_group_id + 2 + offset;
}

[shader("compute")]
[numthreads(RTGI_DENOISE_DIFFUSE_X,RTGI_DENOISE_DIFFUSE_Y,1)]
func entry_temporal_stabilization(uint2 dtid : SV_DispatchThreadID, uint in_group_index : SV_GroupIndex, int2 group_id : SV_GroupID, int2 in_group_id : SV_GroupThreadID)
{
    // Get Constants
    let push = rtgi_diffuse_stabilize_history_push;
    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[dtid];
    const float3 sv_pos = float3(dtid.xy, pixel_depth);
    const float2 uv = dtid.xy * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[dtid.xy]);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 world_position_prev_frame = world_position; // No support for dynamic object motion vectors yet.

    // Preload Luma
    for (int gx = 0; gx < 2; ++gx)
    {
        for (int gy = 0; gy < 2; ++gy)
        {
            int2 gs_idx = in_group_id + int2(gx,gy) * RTGI_DENOISE_DIFFUSE_X;
            int2 src_idx = int2(dtid) + int2(gx,gy) * RTGI_DENOISE_DIFFUSE_X - 2;
            src_idx = clamp(src_idx, int2(0,0), int2(half_res_render_target_size) - 1);

            if (all(gs_idx <  PRELOAD_WIDTH))
            {
                if (push.attach.view_cam_half_res_depth.get()[src_idx] == 0.0f)
                {
                    gs_luma[gs_idx.x][gs_idx.y] = 0.0f;
                    gs_depth_normals_preload[gs_idx.x][gs_idx.y] = 0.0f;
                }
                else
                {
                    gs_luma[gs_idx.x][gs_idx.y] = push.attach.rtgi_diffuse_blurred.get()[src_idx].w;
                    gs_samplecount_preload[gs_idx.x][gs_idx.y] = push.attach.rtgi_samplecnt.get()[src_idx];
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Calculate luma average and sigma
    float valid_surrounding_samples = 0.0f;
    float avg_luma = 0.0f;
    float sigma_luma = 0.0f;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            const int2 gs_index = local_offset_to_gs_index(in_group_id, int2(x,y) * 2);
            const float luma = gs_luma[gs_index.x][gs_index.y];
            const float samplecnt = gs_samplecount_preload[gs_index.x][gs_index.y];
            avg_luma += pow(luma, 1.0f / 5.0f);
            valid_surrounding_samples += samplecnt >= 1.0f;
        }
    }
    avg_luma = pow(avg_luma * rcp(9.0f), 5.0f);

    
    const uint thread_seed = dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y + push.attach.globals.frame_index * push.attach.globals->settings.render_target_size.y * push.attach.globals->settings.render_target_size.x;
    rand_seed(thread_seed);

    float4 diffuse_history = push.attach.rtgi_diffuse_reprojected.get()[dtid];
    float2 diffuse2_history = push.attach.rtgi_diffuse2_reprojected.get()[dtid];

    // floodfill history of disoccluded pixels within 2 pixel range
    // A lot of times, there will be a single pixel shimmer that does not resolve under movement.
    // These shimmers are completely replaced with their valid neighbor history.
    bool flood_fill_similar = false;
    float4 similar_converged_neighbor_diffuse = float4(0,0,0,0);
    float2 similar_converged_neighbor_diffuse2 = float2(0,0);
    if (false)
    {
        // The threshold prevents floodfilled pixels to be the src for other floodfills.
        // If this were not prevented, the floodfills would cascade bejond one pixel.
        const float SRC_DST_SAMPLECOUNT_FLOODFILL_THRESHOLD = push.attach.globals.rtgi_settings.history_frames * 0.75f;
        float similar_converged_pixel_neighbors = 0.0f;
        float converged_pixel_neighbors = 0.0f;

        const float NEIGHBORHOOD_SCALE = 1.0f;
        for (int x = -2; x <= 2; ++x)
        {
            for (int y = -2; y <= 2; ++y)
            {
                if (x == 0 && y == 0)
                    continue;

                const int2 neighbor_pixel_idx = clamp(dtid.xy + int2(x,y), int2(0,0), half_res_render_target_size - 1);
                const float neighbor_depth = push.attach.view_cam_half_res_depth.get()[neighbor_pixel_idx];
                const float3 neighbor_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[neighbor_pixel_idx]);
                
                const float3 neighbor_ndc = float3(ndc.xy + float2(inv_half_res_render_target_size * 2.0f * float2(x,y)), neighbor_depth);
                const float4 neighbor_ws_pre_div = mul(camera.inv_view_proj, float4(neighbor_ndc, 1.0f));
                const float3 neighbor_ws = neighbor_ws_pre_div.xyz / neighbor_ws_pre_div.w;

                const float neighbor_samplecnt = push.attach.rtgi_samplecnt.get()[neighbor_pixel_idx];

                const float neighbor_converged = (neighbor_samplecnt > SRC_DST_SAMPLECOUNT_FLOODFILL_THRESHOLD);
                const float neighbor_similar_converged = 
                    neighbor_converged *
                    max(0.0f, dot(neighbor_normal, pixel_face_normal)) *
                    surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, world_position_prev_frame, pixel_face_normal, neighbor_ws, neighbor_normal);
                converged_pixel_neighbors += neighbor_converged;
                similar_converged_pixel_neighbors += neighbor_similar_converged;
                
                const float4 diffuse_sample = push.attach.rtgi_diffuse_reprojected.get()[neighbor_pixel_idx];
                const float2 diffuse2_sample = push.attach.rtgi_diffuse2_reprojected.get()[neighbor_pixel_idx];
                similar_converged_neighbor_diffuse += neighbor_similar_converged * diffuse_sample;
                similar_converged_neighbor_diffuse2 += neighbor_similar_converged * diffuse2_sample;
            }
        }
        similar_converged_neighbor_diffuse *= rcp(similar_converged_pixel_neighbors);
        similar_converged_neighbor_diffuse2 *= rcp(similar_converged_pixel_neighbors);

        flood_fill_similar = similar_converged_pixel_neighbors >= 1.0f && push.attach.rtgi_samplecnt.get()[dtid] < SRC_DST_SAMPLECOUNT_FLOODFILL_THRESHOLD;
    }

    bool disocclusion = push.attach.rtgi_samplecnt.get()[dtid] == 0;
    float history_luma = diffuse_history.w;

    float4 diffuse_blurred = push.attach.rtgi_diffuse_blurred.get()[dtid];
    float2 diffuse2_blurred = push.attach.rtgi_diffuse2_blurred.get()[dtid];
    const float luma_diff_relative = (history_luma + avg_luma) / (min(history_luma, avg_luma)) * 0.5f - 1.0f;
    float luma_diff_history_scaling = 1.0f;
    if (false)
    {
        luma_diff_history_scaling = clamp(pow(0.5f, luma_diff_relative * 0.01f + 0.00001f) + 0.01f, 0.0f, 1.0f); // TODO: Add heavily firefly filtered fast history to clamp against!
    }
    
    const bool temp_stabilization =     // Prevents bubbling on converged pixels
        push.attach.globals.rtgi_settings.temporal_stabilization_enabled * 
        (push.attach.rtgi_samplecnt.get()[dtid] == push.attach.globals.rtgi_settings.history_frames);
    const float temp_jitter =           // Quantization artifacts over time are avoided by temporal jitter
        rand();
    const float samplecnt =             // Accumulated samples
        min(push.attach.rtgi_samplecnt.get()[dtid], push.attach.globals.rtgi_settings.history_frames);
    float boosted_samplecount =   // When we hit max samplecount we artificially boost it to prevent bubbling
        samplecnt * (temp_stabilization ? lerp(1.0f, 16.0f, temp_jitter) : 1.0f);
    //boosted_samplecount = min(4.0f, boosted_samplecount);
    float history_blend = 
        (1.0f - 1.0f / (1.0f + boosted_samplecount))
        * clamp(luma_diff_history_scaling, 0.0f, 1.0f);
    if (disocclusion)
    {
        history_blend = 0.0f;
    }
    if (!push.attach.globals.rtgi_settings.temporal_accumulation_enabled)
    {
        history_blend = 0.0f;
    }
    //diffuse_blurred = pixel_depth.xxxx;
    //diffuse2_blurred = pixel_depth.xx;
    const float4 new_diffuse = lerp(diffuse_blurred, diffuse_history, history_blend);
    const float2 new_diffuse2 = lerp(diffuse2_blurred, diffuse2_history, history_blend);

    push.attach.rtgi_diffuse_accumulated.get()[dtid] = new_diffuse;
    push.attach.rtgi_diffuse2_accumulated.get()[dtid] = new_diffuse2;

    if (flood_fill_similar)
    {
        push.attach.rtgi_diffuse_stable.get()[dtid] = similar_converged_neighbor_diffuse;
        push.attach.rtgi_diffuse2_stable.get()[dtid] = similar_converged_neighbor_diffuse2;
        push.attach.debug_image.get()[dtid * 2 + uint2(0,1)] = float4(1,1,0,1);
        push.attach.debug_image.get()[dtid * 2 + uint2(0,0)] = float4(1,1,0,1);
        push.attach.debug_image.get()[dtid * 2 + uint2(1,1)] = float4(1,1,0,1);
        push.attach.debug_image.get()[dtid * 2 + uint2(1,0)] = float4(1,1,0,1);
    }
    else
    {
        push.attach.rtgi_diffuse_stable.get()[dtid] = new_diffuse;
        push.attach.rtgi_diffuse2_stable.get()[dtid] = new_diffuse2;
        push.attach.debug_image.get()[dtid * 2 + uint2(0,0)] = float4((samplecnt > 0.5 * push.attach.globals.rtgi_settings.history_frames).xxx,1);
        push.attach.debug_image.get()[dtid * 2 + uint2(0,1)] = float4((samplecnt > 0.5 * push.attach.globals.rtgi_settings.history_frames).xxx,1);
        push.attach.debug_image.get()[dtid * 2 + uint2(1,0)] = float4((samplecnt > 0.5 * push.attach.globals.rtgi_settings.history_frames).xxx,1);
        push.attach.debug_image.get()[dtid * 2 + uint2(1,1)] = float4((samplecnt > 0.5 * push.attach.globals.rtgi_settings.history_frames).xxx,1);
    }
}