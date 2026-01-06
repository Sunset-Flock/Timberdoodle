#pragma once

#include "rtgi_reproject.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiReprojectDiffusePush rtgi_denoise_diffuse_reproject_push;
[[vk::push_constant]] RtgiDiffuseTemporalStabilizationPush rtgi_diffuse_stabilize_history_push;

[shader("compute")]
[numthreads(RTGI_DENOISE_DIFFUSE_X,RTGI_DENOISE_DIFFUSE_Y,1)]
func entry_reproject(uint2 dtid : SV_DispatchThreadID)
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
        push.attach.rtgi_diffuse_reprojected.get()[dtid] = float4(0,0,0,0);
        push.attach.rtgi_diffuse2_reprojected.get()[dtid] = float2(0,0);
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
    const float2 uv_prev_frame = ndc_prev_frame.xy * 0.5f + 0.5f;
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
                surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[0], other_face_normals[0]),
                surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[1], other_face_normals[1]),
                surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[2], other_face_normals[2]),
                surface_weight(inv_half_res_render_target_size, camera.near_plane, pixel_depth, world_position_prev_frame, pixel_face_normal, texel_ws_prev_frame[3], other_face_normals[3]),
            };
            surface_weights[0] *= depth_reprojected4[0] != 0.0f;
            surface_weights[1] *= depth_reprojected4[1] != 0.0f;
            surface_weights[2] *= depth_reprojected4[2] != 0.0f;
            surface_weights[3] *= depth_reprojected4[3] != 0.0f;
        }

        occlusion = surface_weights * in_screen;
    }

    float4 sample_weights = get_bilinear_custom_weights( bilinear_filter_at_prev_pos, occlusion );
    
    // In order to detect a disocclusion, test if the sum of sample weights is lower than 1/4 - epsilon.
    // If at the sum of sample weights is > (1/4-epsi), we have enough non-occluded samples with a relevant bilinear weight.
    // When the sum is lower, the visible samples do not have enough bilinear weight to be relevant.
    const float SAMPLE_WEIGHT_DISSOCCLUSION_THRESHOLD = 0.99f * 0.6f;
    const float total_sample_weight = dot(1.0f, sample_weights);
    bool disocclusion = total_sample_weight < SAMPLE_WEIGHT_DISSOCCLUSION_THRESHOLD;

    // Calc new sample count
    float samplecnt = apply_bilinear_custom_weights( samplecnt_reprojected4.x, samplecnt_reprojected4.y, samplecnt_reprojected4.z, samplecnt_reprojected4.w, sample_weights ).x;
    samplecnt = min( samplecnt + 1.0f, push.attach.globals.rtgi_settings.history_frames );
    samplecnt = disocclusion ? 0u : samplecnt;
    push.attach.rtgi_samplecnt.get()[halfres_pixel_index] = samplecnt;
    

    // Read in diffuse history
    const float4 history_Y0 = push.attach.rtgi_diffuse_history.get().GatherRed( push.attach.globals.samplers.nearest_clamp.get(), reproject_gather_uv ).wzxy;
    const float4 history_Y1 = push.attach.rtgi_diffuse_history.get().GatherGreen( push.attach.globals.samplers.nearest_clamp.get(), reproject_gather_uv ).wzxy;
    const float4 history_Y2 = push.attach.rtgi_diffuse_history.get().GatherBlue( push.attach.globals.samplers.nearest_clamp.get(), reproject_gather_uv ).wzxy;
    const float4 history_Y3 = push.attach.rtgi_diffuse_history.get().GatherAlpha( push.attach.globals.samplers.nearest_clamp.get(), reproject_gather_uv ).wzxy;
    const float4 history_Co = push.attach.rtgi_diffuse2_history.get().GatherRed( push.attach.globals.samplers.nearest_clamp.get(), reproject_gather_uv ).wzxy;
    const float4 history_Cg = push.attach.rtgi_diffuse2_history.get().GatherGreen( push.attach.globals.samplers.nearest_clamp.get(), reproject_gather_uv ).wzxy;
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
            const int2 gs_index = local_offset_to_gs_index(in_group_id, int2(x,y));
            const float luma = gs_luma[gs_index.x][gs_index.y];
            const float samplecnt = gs_samplecount_preload[gs_index.x][gs_index.y];
            avg_luma += luma;
            sigma_luma += luma * luma;
            valid_surrounding_samples += samplecnt >= 24.0f;
        }
    }
    avg_luma *= rcp(25.0f);
    sigma_luma *= rcp(25.0f);
    
    const uint thread_seed = dtid.x * push.attach.globals->settings.render_target_size.y + dtid.y + push.attach.globals.frame_index * push.attach.globals->settings.render_target_size.y * push.attach.globals->settings.render_target_size.x;
    rand_seed(thread_seed);

    const float4 diffuse_history = push.attach.rtgi_diffuse_reprojected.get()[dtid];
    const float2 diffuse2_history = push.attach.rtgi_diffuse2_reprojected.get()[dtid];
    const float history_luma = diffuse_history.w;
    

    
    bool pixel_problematic = false;//push.attach.rtgi_samplecnt.get()[dtid] == 0 && (valid_surrounding_samples > (3*3 * 0.8));
    if (pixel_problematic)
    {
        push.attach.rtgi_samplecnt.get()[dtid] = push.attach.globals.rtgi_settings.history_frames;
    }
    bool disocclusion = push.attach.rtgi_samplecnt.get()[dtid] == 0;

    const float4 diffuse_blurred = push.attach.rtgi_diffuse_blurred.get()[dtid];
    const float2 diffuse2_blurred = push.attach.rtgi_diffuse2_blurred.get()[dtid];
    const float luma_diff_relative = (history_luma + avg_luma) / (min(history_luma, avg_luma)) * 0.5f - 1.0f;
    const float luma_diff_history_scaling = pow(0.5f, luma_diff_relative * 0.01f + 0.00001f);
    const float temp_stabilization = 
        (-1.0f + 0.1f) * 
        push.attach.globals.rtgi_settings.temporal_stabilization_enabled;
    const float temp_jitter = -square(rand()) * rcp(64);
    const float boosted_samplecnt = min(push.attach.rtgi_samplecnt.get()[dtid] * 4.0f, push.attach.globals.rtgi_settings.history_frames);
    float history_blend = 
        boosted_samplecnt * rcp(push.attach.globals.rtgi_settings.history_frames + 1.0f + temp_stabilization) * 
        clamp(luma_diff_history_scaling, 0.0f, 1.0f) + 
        temp_jitter;
    if (disocclusion || !push.attach.globals.rtgi_settings.temporal_accumulation_enabled)
    {
        history_blend = 0.0f;
    }
    const float4 new_diffuse = lerp(diffuse_blurred, diffuse_history, history_blend);
    const float2 new_diffuse2 = lerp(diffuse2_blurred, diffuse2_history, history_blend);

    push.attach.rtgi_diffuse_stable.get()[dtid] = new_diffuse;
    push.attach.rtgi_diffuse2_stable.get()[dtid] = new_diffuse2;
}