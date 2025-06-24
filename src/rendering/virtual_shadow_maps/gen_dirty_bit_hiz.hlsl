#include <daxa/daxa.inl>
#include "vsm.inl"
#include "shader_lib/vsm_util.glsl"

[[vk::push_constant]]GenDirtyBitHizPush push;

float2 make_gather_uv(float2 inv_size, uint2 top_left_index)
{
    return (float2(top_left_index + 1.0f) * inv_size);
}

// TODO:    There is ub here.
//          The writes and reads should have bounds checks.
groupshared bool s_mins[2][GEN_DIRTY_BIT_HIZ_X_DISPATCH][GEN_DIRTY_BIT_HIZ_Y_DISPATCH];
void downsample_64x64(
    uint2 local_index,
    uint clip_level,
    uint2 grid_index,
    uint2 min_mip_size,
    int mip_count
)
{
    const float2 inv_size = float2(1.0f) / float2(min_mip_size);
    bool4 quad_values = bool4(0);
    [[unroll]]
    for(uint quad_i = 0; quad_i < 4; ++quad_i)
    {
        int2 sub_i = int2(quad_i >> 1, quad_i & 1);
        int2 src_i = int2((grid_index * GEN_DIRTY_BIT_HIZ_X_DISPATCH + local_index.xy) * 2 + sub_i) * 2;

        const int2 wrapped_coords_1 = vsm_page_coords_to_wrapped_coords(int3(src_i + int2(0,0), clip_level), push.attachments.vsm_clip_projections).xy;
        const int2 wrapped_coords_2 = vsm_page_coords_to_wrapped_coords(int3(src_i + int2(1,0), clip_level), push.attachments.vsm_clip_projections).xy;
        const int2 wrapped_coords_3 = vsm_page_coords_to_wrapped_coords(int3(src_i + int2(0,1), clip_level), push.attachments.vsm_clip_projections).xy;
        const int2 wrapped_coords_4 = vsm_page_coords_to_wrapped_coords(int3(src_i + int2(1,1), clip_level), push.attachments.vsm_clip_projections).xy;

        uint4 fetch;
        fetch.x = Texture2DArray<uint>::get(push.attachments.vsm_page_table).Load(int4(wrapped_coords_1, clip_level, 0), int2(0)).x;
        fetch.y = Texture2DArray<uint>::get(push.attachments.vsm_page_table).Load(int4(wrapped_coords_2, clip_level, 0), int2(0)).x;
        fetch.z = Texture2DArray<uint>::get(push.attachments.vsm_page_table).Load(int4(wrapped_coords_3, clip_level, 0), int2(0)).x;
        fetch.w = Texture2DArray<uint>::get(push.attachments.vsm_page_table).Load(int4(wrapped_coords_4, clip_level, 0), int2(0)).x;

        const bool is_dirty = 
            get_is_dirty(fetch.x) ||
            get_is_dirty(fetch.y) ||
            get_is_dirty(fetch.z) ||
            get_is_dirty(fetch.w);
        
        int2 dst_i = int2((grid_index * GEN_DIRTY_BIT_HIZ_X_DISPATCH + local_index.xy) * 2) + sub_i;

        RWTexture2DArray<uint>::get(push.attachments.vsm_dirty_bit_hiz[0])[uint3(src_i + uint2(0, 0), clip_level)].x = uint(get_is_dirty(fetch.x));
        RWTexture2DArray<uint>::get(push.attachments.vsm_dirty_bit_hiz[0])[uint3(src_i + uint2(1, 0), clip_level)].x = uint(get_is_dirty(fetch.y));
        RWTexture2DArray<uint>::get(push.attachments.vsm_dirty_bit_hiz[0])[uint3(src_i + uint2(0, 1), clip_level)].x = uint(get_is_dirty(fetch.z));
        RWTexture2DArray<uint>::get(push.attachments.vsm_dirty_bit_hiz[0])[uint3(src_i + uint2(1, 1), clip_level)].x = uint(get_is_dirty(fetch.w));

        RWTexture2DArray<uint>::get(push.attachments.vsm_dirty_bit_hiz[1])[uint3(dst_i, clip_level)].x = uint(is_dirty);
        quad_values[quad_i] = is_dirty;
    }
    {
        const bool is_dirty = quad_values.x || quad_values.y || quad_values.z || quad_values.w;
        const int2 dst_i = int2(grid_index * GEN_DIRTY_BIT_HIZ_X_DISPATCH + local_index.xy);
        RWTexture2DArray<uint>::get(push.attachments.vsm_dirty_bit_hiz[2])[uint3(dst_i, clip_level)].x = uint(is_dirty);
        s_mins[0][local_index.y][local_index.x] = is_dirty;
    }
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
            RWTexture2DArray<uint>::get(push.attachments.vsm_dirty_bit_hiz[dst_mip])[uint3(glob_wg_offset + local_index.xy, clip_level)].x = uint(is_dirty);
            s_mins[ping_pong_dst_index][local_index.y][local_index.x] = is_dirty;
        }
    }
}

[numthreads(GEN_DIRTY_BIT_HIZ_X_DISPATCH, GEN_DIRTY_BIT_HIZ_Y_DISPATCH, 1)]
[shader("compute")]
void main(uint2 liid : SV_GroupThreadID, uint3 wgid : SV_GroupID)
{
    downsample_64x64(liid, wgid.z, wgid.xy, VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION, push.mip_count);
}
