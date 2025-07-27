#pragma once

#include "rtgi_reconstruct_history.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"

[[vk::push_constant]] RtgiDiffuseReconstructHistoryPush rtgi_diffuse_reconstruct_history_push;

groupshared float4 gs_diffuse_depth[RTGI_RECONSTRUCT_HISTORY_DIFFUSE_X][RTGI_RECONSTRUCT_HISTORY_DIFFUSE_Y];

func downsample_mip_linear(uint2 thread_index, uint2 group_thread_index, uint mip)
{
    let mip_factor = 2u << mip;
    let push = rtgi_diffuse_reconstruct_history_push;
    if (all(group_thread_index.xy < RTGI_RECONSTRUCT_HISTORY_DIFFUSE_X/mip_factor))
    {
        float4 diffuse_depth = 
            0.25f * gs_diffuse_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0] +
            0.25f * gs_diffuse_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0] +
            0.25f * gs_diffuse_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1] +
            0.25f * gs_diffuse_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1];
            
        let mip_group_base_index = ((thread_index.xy / RTGI_RECONSTRUCT_HISTORY_DIFFUSE_X) * RTGI_RECONSTRUCT_HISTORY_DIFFUSE_X)/mip_factor;
        push.attach.rtgi_reconstructed_diffuse_history[mip].get()[mip_group_base_index + group_thread_index] = diffuse_depth;
    }
}

[shader("compute")]
[numthreads(RTGI_RECONSTRUCT_HISTORY_DIFFUSE_X,RTGI_RECONSTRUCT_HISTORY_DIFFUSE_Y,1)]
func entry_reconstruct_history_diffuse(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID)
{
    let push = rtgi_diffuse_reconstruct_history_push;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    uint2 half_res_index = dtid.xy;

    float3 diffuse = push.attach.rtgi_diffuse_raw.get()[half_res_index].rgb;
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