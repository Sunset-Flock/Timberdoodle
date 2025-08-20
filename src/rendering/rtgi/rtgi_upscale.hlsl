#pragma once

#include "rtgi_upscale.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiUpscaleDiffusePush rtgi_upscale_diffuse_push;

#define GS_PRELOAD_WIDTH (RTGI_UPSCALE_DIFFUSE_X/2+2)
groupshared float4 gs_half_diffuse_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float2 gs_half_diffuse2_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float4 gs_half_normals_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float4 gs_half_vs_positions[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];

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

    // Calc pixel view attributes
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 position_vs_pre_div = mul(camera.inv_proj, float4(ndc, 1.0f));
    const float3 position_vs = -position_vs_pre_div.xyz / position_vs_pre_div.w;
    const float3 pixel_face_normal_vs = mul(camera.view, float4(pixel_face_normal,0.0f)).xyz;

    // Preload Surrounding half res rtgi values
    // Each Group works on a 8x8 full res tile.
    // To full reconstruct the full res tile we need a (8/2 + 2)^2 section of the half res diffuse.
    {
        Texture2D<float4> half_res_diffuse_tex = push.attach.rtgi_diffuse_half_res.get();
        Texture2D<float2> half_res_diffuse2_tex = push.attach.rtgi_diffuse2_half_res.get();
        Texture2D<float> half_res_depth_tex = push.attach.view_cam_half_res_depth.get();
        Texture2D<uint> half_res_face_normal_tex = push.attach.view_cam_half_res_face_normals.get();

        const int2 group_base_half_index = (group_id * int(RTGI_UPSCALE_DIFFUSE_X/2)) - 1;
        if (all(in_group_id < GS_PRELOAD_WIDTH))
        {
            const int2 preload_index = in_group_id;
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

            // Calculate weights
            const float tent_weight = tent_weights_x[row] * tent_weights_y[col];
            const float geometry_weight = get_geometry_weight(inv_full_res_render_target_size, camera.near_plane, pixel_depth, position_vs, pixel_face_normal_vs, sample_vs, 3.0f);
            const float normal_weight = square(square(square(max(0.0f, dot(sample_face_normal, pixel_face_normal)))));
            const float weight = tent_weight * geometry_weight * normal_weight;

            acc_diffuse += weight * sample_sh_y;
            acc_diffuse2 += weight * sample_cocg;
            acc_weight += weight;
            acc_geo_weight += geometry_weight;

            // Fallback calculation:
            const float vs_dst_weight = exp(-abs(length(position_vs - sample_vs)));
            const float fallback_weight = tent_weight * vs_dst_weight * (0.1f + max(0.0f, dot(sample_face_normal, pixel_face_normal)));
            fallback_acc_diffuse += fallback_weight * sample_sh_y;
            fallback_acc_diffuse2 += fallback_weight * sample_cocg;
            fallback_acc_weight += fallback_weight;
        }
    }

    // Write upscaled diffuse:
    float3 upscaled_diffuse = (float3)0;
    float4 upscaled_sh_y = float4( 0.0f, 0.0f, 0.0f, 0.0f );
    float2 upscaled_cocg = float2( 0.0f, 0.0f );
    if (acc_weight > 0.1f)
    {
        upscaled_sh_y = acc_diffuse * rcp(acc_weight + 0.0000001f);
        upscaled_cocg = acc_diffuse2 * rcp(acc_weight + 0.0000001f);
    }
    else
    {
        upscaled_sh_y = fallback_acc_diffuse * rcp(fallback_acc_weight);
        upscaled_cocg = fallback_acc_diffuse2 * rcp(fallback_acc_weight);
    }
    upscaled_diffuse = sh_resolve_diffuse(upscaled_sh_y * 1e-3f, upscaled_cocg * 1e-4f, pixel_detail_normal);

    push.attach.rtgi_diffuse_full_res.get()[full_res_pixel_index] = float4(upscaled_diffuse, 1.0f);
}