#pragma once

#include "rtgi_reconstruct_history.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"

[[vk::push_constant]] RtgiReconstructHistoryGenMipsDiffusePush rtgi_reconstruct_history_gen_mips_diffuse_push;
[[vk::push_constant]] RtgiReconstructHistoryApplyDiffusePush rtgi_reconstruct_history_apply_diffuse_push;

groupshared float4 gs_diffuse_depth[RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X][RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_Y];

func downsample_mip_linear(uint2 thread_index, uint2 group_thread_index, uint mip)
{
    let mip_factor = 2u << mip;
    let push = rtgi_reconstruct_history_gen_mips_diffuse_push;
    let remaining_block_size = RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X/mip_factor;
    if (all(group_thread_index.xy < remaining_block_size))
    {
        // ignore sky pixels
        float4 weights = {
            (gs_diffuse_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0].a != 0.0f ? 1.0f : 0.0f),
            (gs_diffuse_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0].a != 0.0f ? 1.0f : 0.0f),
            (gs_diffuse_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1].a != 0.0f ? 1.0f : 0.0f),
            (gs_diffuse_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1].a != 0.0f ? 1.0f : 0.0f),
        };
        const float weight_sum = (weights.x + weights.y + weights.z + weights.w);
        weights *= rcp(weight_sum + 0.00000001f);
        const float4 diffuse_depth = 
            weights.x * gs_diffuse_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0] +
            weights.y * gs_diffuse_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0] +
            weights.z * gs_diffuse_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1] +
            weights.w * gs_diffuse_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1];
            
        const uint2 group_index = thread_index / RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X;
        const uint2 mip_group_base_index = group_index * remaining_block_size;
        push.attach.rtgi_reconstructed_diffuse_history[mip].get()[mip_group_base_index + group_thread_index] = diffuse_depth;
        gs_diffuse_depth[group_thread_index.x][group_thread_index.y] = diffuse_depth;
    }
}

[shader("compute")]
[numthreads(RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X,RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_Y,1)]
func entry_gen_mips_diffuse(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID)
{
    let push = rtgi_reconstruct_history_gen_mips_diffuse_push;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    uint2 half_res_index = dtid.xy;

    float3 diffuse = push.attach.rtgi_diffuse_accumulated.get()[half_res_index].rgb;
    float depth = push.attach.view_cam_half_res_depth.get()[half_res_index];
    gs_diffuse_depth[gtid.x][gtid.y] = float4(diffuse, depth);

    // Mip 0:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid,gtid,0);

    // Mip 1:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid,gtid,1);

    // Mip 2:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid,gtid,2);

    // Mip 3:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid,gtid,3);
}

[shader("compute")]
[numthreads(RTGI_RECONSTRUCT_HISTORY_APPLY_DIFFUSE_X,RTGI_RECONSTRUCT_HISTORY_APPLY_DIFFUSE_Y,1)]
func entry_apply_diffuse(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID)
{
    let push = rtgi_reconstruct_history_apply_diffuse_push;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Precalculate constants
    const uint2 halfres_pixel_index = dtid.xy;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const float2 sv_xy = float2(halfres_pixel_index) + 0.5f;
    const float2 uv = sv_xy * inv_half_res_render_target_size;
    
    // Load pixel depth and samplecnt
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[halfres_pixel_index];
    const float pixel_samplecnt = push.attach.rtgi_samplecnt.get()[halfres_pixel_index];

    if (pixel_depth == 0.0f)
    {
        return;
    }

    // Freshly disoccluded areas (pixel_samplecnt < 5) are replaced with reconstructed history
    if (pixel_samplecnt < 5) 
    {
        const float mip = clamp(3.0f - (pixel_samplecnt-1.0f), 0.0f, 3.0f);
        const float2 mip_size = float2(uint2(half_res_render_target_size) >> uint(mip+1));
        const Bilinear bilinear_filter_reconstruct = get_bilinear_filter( saturate( uv ), mip_size );
        
        const float4 diffuse_depth_reconstruct00 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,0), mip));
        const float4 diffuse_depth_reconstruct10 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,0), mip));
        const float4 diffuse_depth_reconstruct01 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,1), mip));
        const float4 diffuse_depth_reconstruct11 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,1), mip));

        const float4 depth_distances = {
            abs(diffuse_depth_reconstruct00.w - pixel_depth),
            abs(diffuse_depth_reconstruct10.w - pixel_depth),
            abs(diffuse_depth_reconstruct01.w - pixel_depth),
            abs(diffuse_depth_reconstruct11.w - pixel_depth)
        };
        const float max_depth_distance = max(max(depth_distances.x,depth_distances.y),max(depth_distances.z,depth_distances.w));
        const float4 depth_weights = rcp(max_depth_distance + 0.00001f) * 4;

        const float4 weights_reconstruct = get_bilinear_custom_weights( bilinear_filter_reconstruct, depth_weights );
        const float3 reconstructed_diffuse = apply_bilinear_custom_weights( diffuse_depth_reconstruct00, diffuse_depth_reconstruct10, diffuse_depth_reconstruct01, diffuse_depth_reconstruct11, weights_reconstruct ).rgb;
        
        push.attach.rtgi_diffuse_accumulated.get()[halfres_pixel_index] = float4(reconstructed_diffuse, 1.0f);
    }
}