#include <daxa/daxa.inl>
#include "vsm.inl"
#include "shader_lib/vsm_util.glsl"

[[vk::push_constant]] GenPointDirtyBitHizPush push;

float2 make_gather_uv(float2 inv_size, uint2 top_left_index)
{
    return (float2(top_left_index + 1.0f) * inv_size);
}

// TODO:    There is ub here.
//          The writes and reads should have bounds checks.
groupshared bool s_mins[2][GEN_DIRTY_BIT_HIZ_X_DISPATCH][GEN_DIRTY_BIT_HIZ_Y_DISPATCH];
// START_MIP_LEVEL is basically the "clip level" - aka which vsm level we are using.
void downsample_64x64<let START_MIP_LEVEL : int>(
    daxa.RWTexture2DArrayId<uint>[7 - START_MIP_LEVEL] hiz,
    uint2 local_index,
    uint2 grid_index,
    uint array_idx,
    int mip_count
)
{
    bool4 quad_values = bool4(0);
    const int start_resolution = (VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION / (1 << START_MIP_LEVEL));
    const float2 inv_size = 1.0f / float2(start_resolution);

    [[unroll]]
    for(uint quad_i = 0; quad_i < 4; ++quad_i)
    {
        int2 sub_i = int2(quad_i >> 1, quad_i & 1);
        int2 src_i = int2((grid_index * GEN_DIRTY_BIT_HIZ_X_DISPATCH + local_index.xy) * 2 + sub_i) * 2;

        // const uint4 fetch = push.attachments.vsm_point_page_table.get().GatherRed(
        //     push.attachments.globals.samplers.linear_clamp.get(),
        //     float3(make_gather_uv(inv_size, src_i), array_idx), 0);

        uint4 fetch;
        fetch.w = push.attachments.vsm_point_spot_page_table.get().Load(int4(src_i + int2(0, 0), array_idx, START_MIP_LEVEL), int2(0)).x;
        hiz[0].get()[uint3(src_i + uint2(0, 0), array_idx)] = uint(get_is_dirty(fetch.w));
        if(START_MIP_LEVEL == 6) { return; }

        fetch.x = push.attachments.vsm_point_spot_page_table.get().Load(int4(src_i + int2(0, 1), array_idx, START_MIP_LEVEL), int2(0)).x;
        fetch.y = push.attachments.vsm_point_spot_page_table.get().Load(int4(src_i + int2(1, 1), array_idx, START_MIP_LEVEL), int2(0)).x;
        fetch.z = push.attachments.vsm_point_spot_page_table.get().Load(int4(src_i + int2(1, 0), array_idx, START_MIP_LEVEL), int2(0)).x;

        const bool is_dirty = get_is_dirty(fetch.x) || get_is_dirty(fetch.y) || get_is_dirty(fetch.z) || get_is_dirty(fetch.w);
        
        hiz[0].get()[uint3(src_i + uint2(1, 0), array_idx)] = uint(get_is_dirty(fetch.z));
        hiz[0].get()[uint3(src_i + uint2(0, 1), array_idx)] = uint(get_is_dirty(fetch.x));
        hiz[0].get()[uint3(src_i + uint2(1, 1), array_idx)] = uint(get_is_dirty(fetch.y));
                                                                        

        int2 dst_i = int2((grid_index * GEN_DIRTY_BIT_HIZ_X_DISPATCH + local_index.xy) * 2) + sub_i;
        hiz[1].get()[uint3(dst_i, array_idx)].x = uint(is_dirty);
        quad_values[quad_i] = is_dirty;
    }
    if(START_MIP_LEVEL == 5) { return; }
    {
        const bool is_dirty = quad_values.x || quad_values.y || quad_values.z || quad_values.w;
        const int2 dst_i = int2(grid_index * GEN_DIRTY_BIT_HIZ_X_DISPATCH + local_index.xy);
        hiz[2].get()[uint3(dst_i, array_idx)].x = uint(is_dirty);
        s_mins[0][local_index.y][local_index.x] = is_dirty;
    }
    if(START_MIP_LEVEL == 4) { return; }
    const uint2 glob_wg_dst_offset0 = (uint2(GEN_DIRTY_BIT_HIZ_X_WINDOW, GEN_DIRTY_BIT_HIZ_Y_WINDOW) * grid_index.xy) / 2;
    [[unroll]]
    for(uint i = 2; i < mip_count - 1; ++i)
    {
        const uint ping_pong_src_index = (i & 1u);
        const uint dst_mip = i + 1;
        const uint ping_pong_dst_index = ((dst_mip) & 1u);
        GroupMemoryBarrierWithGroupSync();
        const bool invoc_active = 
            local_index.x < (GEN_DIRTY_BIT_HIZ_X_WINDOW >> (dst_mip)) && 
            local_index.y < (GEN_DIRTY_BIT_HIZ_Y_WINDOW >> (dst_mip));
        if(invoc_active)
        {
            const uint2 glob_wg_offset = glob_wg_dst_offset0 >> i;
            const uint2 src_i          = local_index.xy * 2;
            const bool4 fetch          = bool4(
                s_mins[ping_pong_src_index][src_i.y + 0][src_i.x+0],
                s_mins[ping_pong_src_index][src_i.y + 0][src_i.x+1],
                s_mins[ping_pong_src_index][src_i.y + 1][src_i.x+0],
                s_mins[ping_pong_src_index][src_i.y + 1][src_i.x+1]
            );
            const bool is_dirty = fetch.x || fetch.y || fetch.z || fetch.w;
            hiz[dst_mip].get()[uint3(glob_wg_offset + local_index.xy, array_idx)].x = uint(is_dirty);
            s_mins[ping_pong_dst_index][local_index.y][local_index.x] = is_dirty;
        }
    }
}

[numthreads(GEN_DIRTY_BIT_HIZ_X_DISPATCH, GEN_DIRTY_BIT_HIZ_Y_DISPATCH, 1)]
[shader("compute")]
void entry_gen_dirty_bit_hiz(uint2 liid : SV_GroupThreadID, uint3 wgid : SV_GroupID)
{
    const uint per_light_images = (7 * 6); // 7 mips 6 cubemap faces.
    const uint point_light_images = push.attachments.globals.vsm_settings.point_light_count * per_light_images;
    const bool is_point_light = (wgid.z < point_light_images);

    uint array_layer_idx = -1;
    uint mip_index = -1;

    if(is_point_light) 
    {
        const uint light_index = wgid.z / per_light_images;
        const uint in_light_index = wgid.z - (light_index * per_light_images);
        mip_index = in_light_index / 7;

        const uint face_index = in_light_index - (7 * mip_index);
        array_layer_idx = get_vsm_point_page_array_idx(face_index, light_index);
    }
    else 
    {
        const uint spot_linear_idx = wgid.z - point_light_images;
        const uint spot_light_idx = spot_linear_idx / 7; // Mip count
        mip_index = spot_linear_idx - (spot_light_idx * 7);
        array_layer_idx = VSM_SPOT_LIGHT_OFFSET + spot_light_idx;
    }

    switch(mip_index)
    {
        case 0: downsample_64x64<0>(push.attachments.vsm_dirty_bit_hiz_mip0, liid, wgid.xy, array_layer_idx, 7);
        case 1: downsample_64x64<1>(push.attachments.vsm_dirty_bit_hiz_mip1, liid, wgid.xy, array_layer_idx, 6);
        case 2: downsample_64x64<2>(push.attachments.vsm_dirty_bit_hiz_mip2, liid, wgid.xy, array_layer_idx, 5);
        case 3: downsample_64x64<3>(push.attachments.vsm_dirty_bit_hiz_mip3, liid, wgid.xy, array_layer_idx, 4);
        case 4: downsample_64x64<4>(push.attachments.vsm_dirty_bit_hiz_mip4, liid, wgid.xy, array_layer_idx, 3);
        case 5: downsample_64x64<5>(push.attachments.vsm_dirty_bit_hiz_mip5, liid, wgid.xy, array_layer_idx, 2);
        case 6: downsample_64x64<6>(push.attachments.vsm_dirty_bit_hiz_mip6, liid, wgid.xy, array_layer_idx, 1);
    }
}