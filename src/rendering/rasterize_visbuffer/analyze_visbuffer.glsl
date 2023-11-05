#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "analyze_visbuffer.inl"
#include "shader_shared/shared.inl"

#include "shader_lib/visbuffer.glsl"

// MUST BE SMALLER EQUAL TO WARP_SIZE!
#define COALESE_MESHLET_INSTANCE_WRITE_COUNT 32

vec2 make_gather_uv(vec2 inv_size, uvec2 top_left_index)
{
    return (vec2(top_left_index) + 1.0f) * inv_size;
}

void update_visibility_masks_and_list(uint meshlet_instance_index, uint triangle_mask)
{
    const uint prev_value = atomicOr(deref(u_meshlet_visibility_bitfield[meshlet_instance_index]), triangle_mask);
    if (prev_value == 0)
    {
        // prev value == zero means, that we are the first thread to ever see this meshlet visible.
        // As this condition only happens once per meshlet that is marked visible,
        // this thread in the position to uniquely write out this meshlets index to the visible meshlet list.
        const uint offset = atomicAdd(deref(u_visible_meshlets).count, 1);
        deref(u_visible_meshlets).meshlet_ids[offset] = meshlet_instance_index;
    }
}

DAXA_DECL_PUSH_CONSTANT(AnalyzeVisbufferPush2, push)
layout(local_size_x = ANALYZE_VIS_BUFFER_WORKGROUP_X, local_size_y = ANALYZE_VIS_BUFFER_WORKGROUP_Y) in;
void main()
{
    // How does this Work?
    // Problem:
    //   We need to atomic or the visibility mask of each meshlet instance to mark it as visible.
    //   Then, we test if it not already visible before that atomic or.
    //   If not, we write its id to the visible meshlet list.
    // Naive Solution:
    //   Launch an invocation per pixel, do the atomicOr and conditional append per thread.
    //   VERY slow, as it has brutal atomic contention.
    // Better Solution:
    //   Launch one invoc per pixel, then find the minimal list of unique values from the values of each thread in the warp.
    //     Find unique value list:
    //       In a loop, elect a value from the threads,
    //       each thread checks if they have the elected value, if so mark to be done (when marked as done, thread does not participate in vote anymore),
    //       write out value to list.
    //       early out when all threads are marked done.
    //     Writeout:
    //       Each thread in the warp takes 1 value from the list and does the atomic ops coherently.
    // Even Better Solution:
    //   Take the better solution with the twist, that each thread takes 4 pixels instead of one, tracking them with a bitmask per thread in the unique list generation.
    //   Still do only up to WARP_SIZE iterations in the loop with a list size of WARP_SIZE.
    //   In the end threads can still have values left in a worst case (because we read WARPSIZE*4 pixels but the list can only hold N values), so write the rest out divergently but coherent.
    //   In 90% of warps the list covers all WARPSIZE*4 pixels, so the writeout is very coherent and atomic op count greatly reduced as a result.
    //   Around 26x faster even in scenes with lots of small meshlets.
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 sampleIndex = min(index << 1, ivec2(push.size) - 1);
    uvec4 vis_ids = textureGather(daxa_usampler2D(u_visbuffer, globals.samplers.linear_clamp), make_gather_uv(1.0f / push.size, sampleIndex), 0);
    uint list_mask = (vis_ids[0] != INVALID_TRIANGLE_ID ? 1 : 0) |
                     (vis_ids[1] != INVALID_TRIANGLE_ID ? 2 : 0) |
                     (vis_ids[2] != INVALID_TRIANGLE_ID ? 4 : 0) |
                     (vis_ids[3] != INVALID_TRIANGLE_ID ? 8 : 0);
    uvec4 triangle_masks;
    uvec4 meshlet_instance_indices;
    [[unroll]] for (uint i = 0; i < 4; ++i)
    {
        triangle_masks[i] = triangle_mask_bit_from_triangle_index(triangle_index_from_triangle_id(vis_ids[i]));
        meshlet_instance_indices[i] = meshlet_instance_index_from_triangle_id(vis_ids[i]);
    }
    uint assigned_meshlet_instance_index = ~0;
    uint assigned_triangle_mask = 0;
    uint assigned_meshlet_index_count = 0;
    for (; assigned_meshlet_index_count < COALESE_MESHLET_INSTANCE_WRITE_COUNT && subgroupAny(list_mask != 0); ++assigned_meshlet_index_count)
    {
        const bool lane_on = list_mask != 0;
        const uint voted_for_id = lane_on ? meshlet_instance_indices[findLSB(list_mask)] : ~0;
        const uint elected_meshlet_instance_index = subgroupBroadcast(voted_for_id, subgroupBallotFindLSB(subgroupBallot(lane_on)));
        // If we have the elected id in our list, remove it.
        uint triangle_mask_contribution = 0;
        [[unroll]] for (uint i = 0; i < 4; ++i)
        {
            if (meshlet_instance_indices[i] == elected_meshlet_instance_index)
            {
                triangle_mask_contribution |= triangle_masks[i];
                list_mask &= ~(1u << i);
            }
        }
        const uint warp_merged_triangle_mask = subgroupOr(triangle_mask_contribution);
        if (assigned_meshlet_index_count == gl_SubgroupInvocationID.x)
        {
            assigned_meshlet_instance_index = elected_meshlet_instance_index;
            assigned_triangle_mask = warp_merged_triangle_mask;
        }
    }
    // Write out
    if (gl_SubgroupInvocationID.x < assigned_meshlet_index_count)
    {
        update_visibility_masks_and_list(assigned_meshlet_instance_index, assigned_triangle_mask);
    }
    // Write out rest of local meshlet list:
    [[loop]] while (list_mask != 0)
    {
        const uint lsb = findLSB(list_mask);
        const uint meshlet_instance_index = meshlet_instance_indices[lsb];
        const uint triangle_index_mask = triangle_masks[lsb];
        list_mask &= ~(1 << lsb);
        update_visibility_masks_and_list(meshlet_instance_index, triangle_index_mask);
    }
}