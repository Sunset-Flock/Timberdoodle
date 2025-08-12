#pragma once

#include "rtgi_reconstruct_history.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiReconstructHistoryGenMipsDiffusePush rtgi_reconstruct_history_gen_mips_diffuse_push;
[[vk::push_constant]] RtgiReconstructHistoryApplyDiffusePush rtgi_reconstruct_history_apply_diffuse_push;

groupshared float4 gs_diffuse[RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X][RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_Y][2];
groupshared float gs_depth[RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X][RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_Y][2];
#if RTGI_USE_SH
    groupshared float2 gs_diffuse2[RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X][RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_Y][2];
#endif

func downsample_mip_linear(uint2 thread_index, uint2 group_thread_index, uint mip, uint gs_src)
{
    let mip_factor = 2u << mip;
    let push = rtgi_reconstruct_history_gen_mips_diffuse_push;
    let remaining_block_size = RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X/mip_factor;
    if (all(group_thread_index.xy < remaining_block_size))
    {
        // ignore sky pixels
        float4 weights = {
            (gs_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0][gs_src] != 0.0f ? 1.0f : 0.0f),
            (gs_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0][gs_src] != 0.0f ? 1.0f : 0.0f),
            (gs_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1][gs_src] != 0.0f ? 1.0f : 0.0f),
            (gs_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1][gs_src] != 0.0f ? 1.0f : 0.0f),
        };
        const float weight_sum = (weights.x + weights.y + weights.z + weights.w);
        weights *= rcp(weight_sum + 0.00000001f);
        const float4 diffuse = 
            weights.x * gs_diffuse[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0][gs_src] +
            weights.y * gs_diffuse[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0][gs_src] +
            weights.z * gs_diffuse[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1][gs_src] +
            weights.w * gs_diffuse[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1][gs_src];
        const float depth = 
            weights.x * gs_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0][gs_src] +
            weights.y * gs_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0][gs_src] +
            weights.z * gs_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1][gs_src] +
            weights.w * gs_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1][gs_src];
        #if RTGI_USE_SH
            const float2 diffuse2 = 
                weights.x * gs_diffuse2[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0][gs_src] +
                weights.y * gs_diffuse2[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0][gs_src] +
                weights.z * gs_diffuse2[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1][gs_src] +
                weights.w * gs_diffuse2[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1][gs_src];
        #endif
            
        const uint2 group_index = thread_index / RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X;
        const uint2 mip_group_base_index = group_index * remaining_block_size;

        const uint gs_dst = (gs_src + 1u) & 0x1u;

        gs_depth[group_thread_index.x][group_thread_index.y][gs_dst] = depth;
        gs_diffuse[group_thread_index.x][group_thread_index.y][gs_dst] = diffuse;

        #if RTGI_USE_SH
            gs_diffuse2[group_thread_index.x][group_thread_index.y][gs_dst] = diffuse2;
            push.attach.rtgi_reconstructed_diffuse_history[mip].get()[mip_group_base_index + group_thread_index] = diffuse;
            push.attach.rtgi_reconstructed_diffuse2_history[mip].get()[mip_group_base_index + group_thread_index] = float4(diffuse2, depth, 0.0f);
        #else
            push.attach.rtgi_reconstructed_diffuse_history[mip].get()[mip_group_base_index + group_thread_index] = float4(diffuse.rgb, depth);
        #endif
    }
}

[shader("compute")]
[numthreads(RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X,RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_Y,1)]
func entry_gen_mips_diffuse(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID)
{
    let push = rtgi_reconstruct_history_gen_mips_diffuse_push;

    // No early out needed, as the texture mip chain is guaranteed to be 

    const uint2 half_res_index = dtid.xy;
    const uint2 clamped_index = min( half_res_index, push.size - 1u );
    const float depth = push.attach.view_cam_half_res_depth.get()[clamped_index];
    gs_depth[gtid.x][gtid.y][0] = depth;
    #if RTGI_USE_SH
        const float4 diffuse = push.attach.rtgi_diffuse_accumulated.get()[clamped_index];
        const float2 diffuse2 = push.attach.rtgi_diffuse2_accumulated.get()[clamped_index];
        gs_diffuse[gtid.x][gtid.y][0] = diffuse;
        gs_diffuse2[gtid.x][gtid.y][0] = diffuse2;
    #else
        const float3 diffuse = push.attach.rtgi_diffuse_accumulated.get()[clamped_index].rgb;
        gs_diffuse[gtid.x][gtid.y][0] = float4( diffuse, 0.0f );
    #endif

    // Mip 0:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid, gtid, 0, 0);

    // Mip 1:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid, gtid, 1, 1);

    // Mip 2:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid, gtid, 2, 0);

    // Mip 3:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid, gtid, 3, 1);
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

    // Calculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const uint2 halfres_pixel_index = dtid.xy;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const float2 sv_xy = float2(halfres_pixel_index) + 0.5f;
    const float2 uv = sv_xy * inv_half_res_render_target_size;
    
    // Load pixel depth and samplecnt and normal
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[halfres_pixel_index];
    const float pixel_samplecnt = push.attach.rtgi_samplecnt.get()[halfres_pixel_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_normals.get()[halfres_pixel_index]);

    // Calculate pixel attributes
    const float3 vs_pixel_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 vs_position_pre_div = mul(camera.inv_proj, float4(ndc, 1.0f));
    const float3 vs_position = -vs_position_pre_div.xyz / vs_position_pre_div.w;

    if (pixel_depth == 0.0f)
    {
        return;
    }

    // Freshly disoccluded areas (pixel_samplecnt < 5) are replaced with reconstructed history
    if (pixel_samplecnt < 5) 
    {
        // The mip chain base size is round up to 8 to ensure all texels have an exact 2x2 -> 1 match between mip levels.
        // Due to this, the uv is shifted by some amount as there are padding texels on the border of the reconstructed history mip chain.
        // We round to multiple of 16, not 8, as the uv are calculated for the half size. The mip chain is quater size, so going from quater to half means increasing the alignment from 8 to 16.
        const float2 half_size_ru16 = float2(round_up_to_multiple(half_res_render_target_size.x, 16), round_up_to_multiple(half_res_render_target_size.y, 16));
        const float2 corrected_uv = sv_xy * rcp(half_size_ru16);

        const float mip = clamp(3.0f - (pixel_samplecnt), 0.0f, 3.0f);
        const float2 mip_size = float2(uint2(half_size_ru16) >> uint(mip+1));
        const float2 inv_mip_size = rcp(mip_size);
        const Bilinear bilinear_filter_reconstruct = get_bilinear_filter( saturate( corrected_uv ), mip_size );
        
        #if RTGI_USE_SH
            const float4 diffuse00 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,0), mip));
            const float4 diffuse10 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,0), mip));
            const float4 diffuse01 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,1), mip));
            const float4 diffuse11 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,1), mip));
            const float4 reconstruct00_2 = push.attach.rtgi_reconstructed_diffuse2_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,0), mip));
            const float4 reconstruct10_2 = push.attach.rtgi_reconstructed_diffuse2_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,0), mip));
            const float4 reconstruct01_2 = push.attach.rtgi_reconstructed_diffuse2_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,1), mip));
            const float4 reconstruct11_2 = push.attach.rtgi_reconstructed_diffuse2_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,1), mip));
            const float4 diffuse00_2 = float4(reconstruct00_2.xy, 0.0f, 0.0f);
            const float4 diffuse10_2 = float4(reconstruct10_2.xy, 0.0f, 0.0f);
            const float4 diffuse01_2 = float4(reconstruct01_2.xy, 0.0f, 0.0f);
            const float4 diffuse11_2 = float4(reconstruct11_2.xy, 0.0f, 0.0f);
            const float depth00 = reconstruct00_2.z;
            const float depth10 = reconstruct10_2.z;
            const float depth01 = reconstruct01_2.z;
            const float depth11 = reconstruct11_2.z;
        #else
            const float4 diffuse_depth_reconstruct00 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,0), mip));
            const float4 diffuse_depth_reconstruct10 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,0), mip));
            const float4 diffuse_depth_reconstruct01 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,1), mip));
            const float4 diffuse_depth_reconstruct11 = push.attach.rtgi_reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,1), mip));
            const float4 diffuse00 = float4(diffuse_depth_reconstruct00.rgb, 0.0f);
            const float4 diffuse10 = float4(diffuse_depth_reconstruct10.rgb, 0.0f);
            const float4 diffuse01 = float4(diffuse_depth_reconstruct01.rgb, 0.0f);
            const float4 diffuse11 = float4(diffuse_depth_reconstruct11.rgb, 0.0f);
            const float depth00 = diffuse_depth_reconstruct00.w;
            const float depth10 = diffuse_depth_reconstruct10.w;
            const float depth01 = diffuse_depth_reconstruct01.w;
            const float depth11 = diffuse_depth_reconstruct11.w;
        #endif

        const float4 depths = float4(depth00, depth10, depth01, depth11);
        const float4 geometric_weight4 = get_geometry_weight4(inv_mip_size, camera.near_plane, pixel_depth, vs_position, vs_pixel_normal, depths, 0.5f);

        const float4 weights_reconstruct = get_bilinear_custom_weights( bilinear_filter_reconstruct, geometric_weight4 );
        const float4 reconstructed_diffuse = apply_bilinear_custom_weights( diffuse00, diffuse10, diffuse01, diffuse11, weights_reconstruct );
        #if RTGI_USE_SH
            const float2 reconstructed_diffuse2 = apply_bilinear_custom_weights( diffuse00_2, diffuse10_2, diffuse01_2, diffuse11_2, weights_reconstruct ).rg;
        #endif 
        
        if (dot(geometric_weight4, 1) > 0.0f)
        {
            #if RTGI_USE_SH
                push.attach.rtgi_diffuse_accumulated.get()[halfres_pixel_index] = reconstructed_diffuse;
                push.attach.rtgi_diffuse2_accumulated.get()[halfres_pixel_index] = reconstructed_diffuse2;
            #else
                push.attach.rtgi_diffuse_accumulated.get()[halfres_pixel_index] = float4(reconstructed_diffuse.rgb, 1.0f);
            #endif
        }
    }
}