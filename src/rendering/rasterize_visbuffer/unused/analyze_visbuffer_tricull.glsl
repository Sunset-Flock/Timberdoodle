#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "analyze_visbuffer.inl"
#include "visbuffer.glsl"

void add_bit_to_mask(inout uvec4 mask, uint index)
{
    const uint vec_i = (index >> 5u);
    mask[vec_i] = mask[vec_i] | (1u << (index - (vec_i << 5u)));
}

void get_quad_unique_meshlet_masks(uint meshlet_ids[4], uvec4 triangle_indices, out uint unique_meshlet_ids[4], out uvec4 unique_masks[4], out uint count)
{
    count = 0;
    [[unroll]]
    for (uint i = 0; i < 4; ++i)
    {
        if (meshlet_ids[i] != INVALID_MESHLET_INDEX)
        {
            bool unique = true;
            for (uint o = 0; o < count; ++o)
            {
                if (meshlet_ids[i] == unique_meshlet_ids[o])
                {
                    add_bit_to_mask(unique_masks[o], triangle_indices[i]);
                    unique = false;
                }
            }
            if (unique)
            {
                unique_meshlet_ids[count] = meshlet_ids[i];
                unique_masks[count] = uvec4(0,0,0,0);
                add_bit_to_mask(unique_masks[count], triangle_indices[i]);
                ++count;
            }
        }
    }
}

DAXA_DECL_PUSH_CONSTANT(AnalyzeVisbufferPush, push)
layout(local_size_x = ANALYZE_VIS_BUFFER_WORKGROUP_X, local_size_y = ANALYZE_VIS_BUFFER_WORKGROUP_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy) * 2;

    const ivec2 quad_indices[4] = { index + ivec2(0,1), index + ivec2(0,0), index + ivec2(1,1), index + ivec2(1,0) };
    uint inst_meshlet_indices[4] = { INVALID_MESHLET_INDEX, INVALID_MESHLET_INDEX, INVALID_MESHLET_INDEX, INVALID_MESHLET_INDEX };
    uvec4 triangle_indices = uvec4(0, 0, 0, 0);
    [[unroll]]
    for (uint quad_i = 0; quad_i < 4; ++quad_i)
    {
        if (quad_indices[quad_i].x < push.width && quad_indices[quad_i].y < push.height)
        {
            // TODO: make this a texture gather.
            const uint vis_id = texelFetch(daxa_utexture2D(u_visbuffer), quad_indices[quad_i], 0).x;
            if (vis_id != INVALID_TRIANGLE_ID)
            {
                uint instantiated_meshlet_index;
                uint triangle_index;
                decode_triangle_id(vis_id, instantiated_meshlet_index, triangle_index);
                inst_meshlet_indices[quad_i] = instantiated_meshlet_index;
                triangle_indices[quad_i] = triangle_index;
            }
        }
    }
    uint quad_unique_meshlet_id_count;
    uint quad_unique_meshlet_ids[4];
    uvec4 quad_unqiue_meshlet_masks[4];
    get_quad_unique_meshlet_masks(inst_meshlet_indices, triangle_indices, quad_unique_meshlet_ids, quad_unqiue_meshlet_masks, quad_unique_meshlet_id_count);

    // In each subgroup, vote on uniform meshlet id, then do the masking. Loop as long as unique ids remain.
    uint left_meshlet_ids = quad_unique_meshlet_id_count;
    const uint debug_buffer_index = globals.frame_index & 1;
    while (left_meshlet_ids > 0)
    {
        const uint uniform_meshlet_id = subgroupBroadcastFirst(quad_unique_meshlet_ids[left_meshlet_ids-1]);
        uint quad_list_index = (~0u);
        for (uint i = 0; i < left_meshlet_ids; ++i)
        {
            if (uniform_meshlet_id == quad_unique_meshlet_ids[i])
            {
                quad_list_index = i;
                break;
            }
        }
        if (quad_list_index != (~0u))
        {
            const uvec4 combined_bitmask = subgroupOr(quad_unqiue_meshlet_masks[quad_list_index]);
            if (subgroupElect())
            {
                uvec4 ret_mask;
                ret_mask[0] = atomicOr(deref(u_meshlet_visibility_bitfields[uniform_meshlet_id])[0], combined_bitmask[0]);
                ret_mask[1] = atomicOr(deref(u_meshlet_visibility_bitfields[uniform_meshlet_id])[1], combined_bitmask[1]);
                ret_mask[2] = atomicOr(deref(u_meshlet_visibility_bitfields[uniform_meshlet_id])[2], combined_bitmask[2]);
                ret_mask[3] = atomicOr(deref(u_meshlet_visibility_bitfields[uniform_meshlet_id])[3], combined_bitmask[3]);
            }
            // Remove element from quad list, by swapping it with the last element and then decrementing the list size.
            if (/*if not last element in list*/quad_list_index != (left_meshlet_ids-1))
            {
                // Swap element with last element, this way we remove the element from the list and preserve the last element,
                // when we decrement the list size.
                quad_unique_meshlet_ids[quad_list_index] = quad_unique_meshlet_ids[left_meshlet_ids-1];
                quad_unqiue_meshlet_masks[quad_list_index] = quad_unqiue_meshlet_masks[left_meshlet_ids-1];
            }
            --left_meshlet_ids;
        }
    }
    #if 0
    if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0)
    {
        //for (uint i = 0; i < 32; ++i)
        //{
        //    deref(u_debug_buffer[(debug_buffer_index^1)*32 + i]) = 0;
        //}
        deref(u_debug_buffer[debug_buffer_index^1]) = 0;
    }
    for (uint y = 0; y < 2; ++y)
    {
        for (uint x = 0; x < 2; ++x)
        {
            const ivec2 test_index = ivec2(gl_GlobalInvocationID.xy * 2 + uvec2(x,y));
            if (test_index.x < push.width && test_index.y < push.height)
            {
                const uint test_id = texelFetch(daxa_utexture2D(u_visbuffer), test_index, 0).x;
                const uint test_flat_index = test_index.x + test_index.y * push.width;
                bool unique = true;
                if (test_id == INVALID_TRIANGLE_ID)
                {
                    unique = false;
                }
                for (uint search_i = 0; search_i < test_flat_index; ++search_i)
                {
                    const ivec2 search_index = ivec2(search_i % push.width, search_i / push.width);
                    const uint search_id = texelFetch(daxa_utexture2D(u_visbuffer), search_index, 0).x;
                    if (search_id == test_id)
                    {
                        unique = false;
                        break;
                    }
                }
                if (unique)
                {
                    atomicAdd(deref(u_debug_buffer[debug_buffer_index]), 1);
                }
            }
        }
    }
    #endif 
}