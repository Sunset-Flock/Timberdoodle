#pragma once

#include "rtgi_upscale.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"

[[vk::push_constant]] RtgiUpscaleDiffusePush rtgi_upscale_diffuse_push;

func get_geometry_weight(float2 inv_render_target_size, float near_plane, float depth, float3 vs_position, float3 vs_normal, float3 other_vs_position) -> float
{
    const float plane_distance = abs(dot(other_vs_position - vs_position, vs_normal));
    // The further away the pixel is, the larger difference we allow.
    // The scale is proportional to the size the pixel takes up in world space.
    const float pixel_size_on_near_plane = inv_render_target_size.y;
    const float near_plane_ws_size = near_plane * 2;
    const float pixel_ws_size = pixel_size_on_near_plane * near_plane_ws_size * rcp(depth + 0.0000001f);
    const float threshold_scale = 3.0f; // a larger factor leads to more bleeding across edges but also less noise on small details
    const float threshold = pixel_ws_size * threshold_scale; 

    const float validity = step( plane_distance, threshold );
    return validity;
}

func get_normal_diffuse_weight(float3 normal, float3 other_normal) -> float
{
    const float validity = max(0.0f, dot(normal, other_normal));
    const float tight_validity = pow(validity, 8.0f);
    return tight_validity;
}

groupshared float4 gs_half_diffuse_depth_preload[RTGI_UPSCALE_DIFFUSE_X+2][RTGI_UPSCALE_DIFFUSE_Y+2];
groupshared float4 gs_half_normals_preload[RTGI_UPSCALE_DIFFUSE_X+2][RTGI_UPSCALE_DIFFUSE_Y+2];

[shader("compute")]
[numthreads(RTGI_UPSCALE_DIFFUSE_X,RTGI_UPSCALE_DIFFUSE_Y,1)]
func entry_upscale_diffuse(uint2 dtid : SV_DispatchThreadID, uint in_group_index : SV_GroupIndex, int2 group_id : SV_GroupID, int2 in_group_id : SV_GroupThreadID)
{
    let push = rtgi_upscale_diffuse_push;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const uint2 full_res_pixel_index = dtid.xy;
    const float2 full_res_render_target_size = push.attach.globals.settings.render_target_size.xy;
    const float2 inv_half_res_render_target_size = rcp(float2(full_res_render_target_size / 2));
    const float2 inv_full_res_render_target_size = rcp(full_res_render_target_size);
    const float2 sv_xy = float2(full_res_pixel_index) + 0.5f;
    const float2 uv = sv_xy * inv_full_res_render_target_size;
    
    // Load pixel depth and face normal
    const float pixel_depth = push.attach.view_cam_depth.get()[full_res_pixel_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_face_normals.get()[full_res_pixel_index]);
    const float3 pixel_detail_normal = uncompress_normal_octahedral_32(push.attach.view_camera_detail_normal_image.get()[full_res_pixel_index]);

    // Calc pixel world position
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;

    // Preload Surrounding half res rtgi values
    {
        Texture2D<float4> half_res_diffuse_tex = push.attach.rtgi_diffuse_half_res.get();
        Texture2D<float> half_res_depth_tex = push.attach.view_cam_half_res_depth.get();
        Texture2D<uint> half_res_face_normal_tex = push.attach.view_cam_half_res_face_normals.get();

        const int needed_preload_threads = square(RTGI_UPSCALE_DIFFUSE_X+2);
        const int group_threads = square(RTGI_UPSCALE_DIFFUSE_X);
        const int group_iters = round_up_div(needed_preload_threads, group_threads);
        const int2 group_base_half_index = (group_id * int(RTGI_UPSCALE_DIFFUSE_X/2)) - 1;
        for (int group_iter = 0; group_iter < group_iters; ++group_iter)
        {
            int preload_flat_index = group_iter * group_threads + in_group_index;
            if (preload_flat_index >= needed_preload_threads)
                break;

            int2 preload_index = int2(preload_flat_index % (RTGI_UPSCALE_DIFFUSE_X+2), preload_flat_index / (RTGI_UPSCALE_DIFFUSE_X+2));
            int2 load_index = clamp(preload_index + group_base_half_index, int2(0,0), int2(push.size/2-1));
            float3 diffuse = half_res_diffuse_tex[load_index].rgb;
            float depth = half_res_depth_tex[load_index];
            gs_half_diffuse_depth_preload[preload_index.x][preload_index.y] = float4(diffuse, depth);
            float3 half_normal = uncompress_normal_octahedral_32(half_res_face_normal_tex[load_index]);
            gs_half_normals_preload[preload_index.x][preload_index.y] = float4(half_normal, 0.0f);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Early out for sky pixel
    if (pixel_depth == 0.0f)
    {
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
    float3 acc_diffuse = (float3)0;
    float acc_weight = 0.0f;
    float3 fallback_acc_diffuse = (float3)0;
    float fallback_acc_weight = 0.0f;
    for (int col = 0; col < TENT_WIDTH; col++)
    {
        for (int row = 0; row < TENT_WIDTH; row++)
        {
            const int2 offset = int2(row - 1, col - 1);
            const int2 pos = int2(dtid/2) + offset;
            if (any(pos >= int2(push.size/2)) || any(pos < int2(0,0)))
            {
                continue;
            }


            // Load values from gs
            const int2 sample_gs_index = in_group_id/2 + offset + int2(1,1);
            const float4 sample_diffuse_depth = gs_half_diffuse_depth_preload[sample_gs_index.x][sample_gs_index.y];
            const float3 sample_face_normal = gs_half_normals_preload[sample_gs_index.x][sample_gs_index.y].xyz;

            // Calculate sample position
            const int2 sample_half_res_idx = (full_res_pixel_index/2) + int2(col,row);
            const float2 sample_uv = float2(sample_half_res_idx) * inv_half_res_render_target_size;
            const float3 sample_ndc = float3(sample_uv * 2.0f - 1.0f, sample_diffuse_depth.w);
            const float4 sample_ws_pre_div = mul(camera.inv_view_proj, float4(sample_ndc,1.0f));
            const float3 sample_ws = sample_ws_pre_div.xyz / sample_ws_pre_div.w;

            // Calculate weights
            const float tent_weight = tent_weights_x[row] * tent_weights_y[col];
            const float geometry_weight = get_geometry_weight(inv_full_res_render_target_size, camera.near_plane, pixel_depth, world_position, pixel_face_normal, sample_ws);
            const float normal_weight = square(max(0.0f, dot(sample_face_normal, pixel_face_normal)));
            const float weight = tent_weight * geometry_weight * normal_weight;

            acc_diffuse += weight * sample_diffuse_depth.rgb;
            acc_weight += weight;

            // Fallback calculation:
            const float ws_dst_weight = exp(-abs(length(world_position - sample_ws)));
            const float fallback_weight = tent_weight * ws_dst_weight;
            fallback_acc_diffuse += fallback_weight * sample_diffuse_depth.rgb;
            fallback_acc_weight += fallback_weight;
        }
    }

    // Write upscaled diffuse:
    float3 upscaled_diffuse = (float3)0;
    if (acc_weight > 0.1f)
    {
        upscaled_diffuse = acc_diffuse * rcp(acc_weight + 0.0000001f);
    }
    else
    {
        upscaled_diffuse = fallback_acc_diffuse * rcp(fallback_acc_weight);
    }
    push.attach.rtgi_diffuse_full_res.get()[full_res_pixel_index] = float4(upscaled_diffuse, 1.0f);
}