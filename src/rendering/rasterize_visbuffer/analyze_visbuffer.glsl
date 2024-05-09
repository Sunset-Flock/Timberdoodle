#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "analyze_visbuffer.inl"
#include "shader_shared/shared.inl"
#include "shader_shared/globals.inl"

#include "shader_lib/visbuffer.glsl"

// MUST BE SMALLER EQUAL TO WARP_SIZE!
#define COALESE_MESHLET_INSTANCE_WRITE_COUNT 32

vec2 make_gather_uv(vec2 inv_size, uvec2 top_left_index)
{
    return (vec2(top_left_index) + 1.0f) * inv_size;
}

DAXA_DECL_PUSH_CONSTANT(AnalyzeVisbufferPush2, push)
void update_visibility_masks_and_list(uint meshlet_instance_index, uint triangle_mask)
{
    const uint prev_value = atomicOr(deref(push.uses.meshlet_visibility_bitfield[meshlet_instance_index]), triangle_mask);
    if (prev_value == 0)
    {
        // prev value == zero means, that we are the first thread to ever see this meshlet visible.
        // As this condition only happens once per meshlet that is marked visible,
        // this thread in the position to uniquely write out this meshlets index to the visible meshlet list.
        const uint offset = atomicAdd(deref(push.uses.visible_meshlets).count, 1);
        deref(push.uses.visible_meshlets).meshlet_ids[offset] = meshlet_instance_index;

        
        // TODO: Profile and optimize this:
        MeshletInstance meshlet_instance = deref_i(deref(push.uses.meshlet_instances).meshlets, meshlet_instance_index);
        uint mesh_instance_index = meshlet_instance.mesh_instance_index;
        uint bitfield_index = mesh_instance_index / 32;
        uint bitfield_mask = 1u << (mesh_instance_index % 32);
        uint ret_bitfield_mask = atomicOr(deref_i(push.uses.mesh_visibility_bitfield, bitfield_index), bitfield_mask);
        if ((bitfield_mask & ret_bitfield_mask) == 0)
        {
            // First to see mesh. Append to list of visible meshes.
            uint visible_mesh_list_index = atomicAdd(deref(push.uses.visible_meshes).count, 1);
            deref(push.uses.visible_meshes).mesh_ids[visible_mesh_list_index] = meshlet_instance.mesh_instance_index;
        }
    }
    #if 0
    #endif
}

void main_old()
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
    uvec4 vis_ids = textureGather(daxa_usampler2D(push.uses.visbuffer, deref(push.uses.globals).samplers.linear_clamp), make_gather_uv(1.0f / push.size, sampleIndex), 0);
    uint list_mask = (vis_ids[0] != INVALID_TRIANGLE_ID ? (1u) : 0) |
                     (vis_ids[1] != INVALID_TRIANGLE_ID ? (1u << 1u) : 0) |
                     (vis_ids[2] != INVALID_TRIANGLE_ID ? (1u << 2u) : 0) |
                     (vis_ids[3] != INVALID_TRIANGLE_ID ? (1u << 3u) : 0);
    uvec4 meshlet_bitfield_masks;
    uvec4 meshlet_instance_indices;
    [[unroll]] for (uint i = 0; i < 4; ++i)
    {
        meshlet_instance_indices[i] = meshlet_instance_index_from_triangle_id(vis_ids[i]);
        meshlet_bitfield_masks[i] = triangle_mask_bit_from_triangle_index(triangle_index_from_triangle_id(vis_ids[i]));
        #if SHADER_DEBUG_VISBUFFER
            if (meshlet_instance_indices[i] >= MAX_MESHLET_INSTANCES && vis_ids[i] != INVALID_TRIANGLE_ID)
            {
                debugPrintfEXT("Detected invalid meshlet instance index: %i\n", meshlet_instance_indices[i]);
                list_mask = list_mask & (~(1u << i));
            }
        #endif
    }




    #if 1
    // if (any_over)
    // {
    // }
    uint assigned_meshlet_bitfield_index = ~0;
    uint assigned_meshlet_bitfield_mask = 0;
    uint assigned_meshlet_bitfield_index_count = 0;
    for (; assigned_meshlet_bitfield_index_count < COALESE_MESHLET_INSTANCE_WRITE_COUNT && subgroupAny(list_mask != 0); ++assigned_meshlet_bitfield_index_count)
    {
        const bool lane_on = list_mask != 0;
        const uint voted_for_id = lane_on ? meshlet_instance_indices[findLSB(list_mask)] : ~0;
        const uint elected_meshlet_instance_index = subgroupBroadcast(voted_for_id, subgroupBallotFindLSB(subgroupBallot(lane_on)));
        subgroupBarrier();
        // If we have the elected id in our list, remove it.
        uint meshlet_bitfield_mask_contribution = 0;
        [[unroll]] for (uint i = 0; i < 4; ++i)
        {
            if (meshlet_instance_indices[i] == elected_meshlet_instance_index)
            {
                meshlet_bitfield_mask_contribution |= meshlet_bitfield_masks[i];
                list_mask &= ~(1u << i);
            }
        }
        subgroupBarrier();
        const uint warp_merged_meshlet_bit_mask = subgroupOr(meshlet_bitfield_mask_contribution);
        if (assigned_meshlet_bitfield_index_count == gl_SubgroupInvocationID.x)
        {
            assigned_meshlet_bitfield_index = elected_meshlet_instance_index;
            assigned_meshlet_bitfield_mask = warp_merged_meshlet_bit_mask;
        }
    }
    subgroupBarrier();
    // uint mask2 = list_mask;
    // bool any_over = assigned_meshlet_bitfield_index != ~0 && assigned_meshlet_bitfield_index > 400000;
    // while(mask2 != 0)
    // {
    //     const uint lsb = findLSB(mask2);
    //     const uint meshlet_instance_index = meshlet_instance_indices[lsb];
    //     mask2 &= ~(1 << lsb);
    //     any_over = any_over || meshlet_instance_index > 400000;
    // }
    // atomicOr(deref(deref(push.uses.globals).debug).gpu_output.debug_ivec4.x, any_over ? 1 : 0);

    // if (subgroupAny(list_mask != 0))
    // {
    //     return;
    // }
    // Write out
    if (gl_SubgroupInvocationID.x < assigned_meshlet_bitfield_index_count)
    {
        update_visibility_masks_and_list(assigned_meshlet_bitfield_index, assigned_meshlet_bitfield_mask);
    }
    #endif
    // Write out rest of local meshlet list:
    [[loop]] while (list_mask != 0)
    {
        const uint lsb = findLSB(list_mask);
        const uint meshlet_instance_index = meshlet_instance_indices[lsb];
        const uint triangle_index_mask = meshlet_bitfield_masks[lsb];
        list_mask &= ~(1 << lsb);
        update_visibility_masks_and_list(meshlet_instance_index, triangle_index_mask);
        atomicAdd(deref(deref(push.uses.globals).debug).gpu_output.debug_ivec4.x, 1);
    }
}

uint meshlet_bitfield_index(uint meshlet_instance_index) { return meshlet_instance_index / 32; }
uint meshlet_bitfield_bit(uint meshlet_instance_index) { return 1u << (meshlet_instance_index % 32); }
uint meshlet_bitfield_bit_to_index(uint meshlet_bitfield_index, uint meshlet_bitfield_bit_index) 
{
    return meshlet_bitfield_index * 32 + meshlet_bitfield_bit_index;
}
void update_visibility_masks_and_list2(uint meshlet_bitfield_index, uint meshlet_bitfield_mask)
{
    const uint prev_mask = atomicOr(deref(push.uses.meshlet_visibility_bitfield[meshlet_bitfield_index]), meshlet_bitfield_mask);
    const uint prev_unset_bits = ~prev_mask;
    uint newly_set_bits = prev_unset_bits & meshlet_bitfield_mask;
    const uint number_of_newly_set_bits = bitCount(newly_set_bits);
    uint visible_meshlets_writeout_offset = 0;
    if (newly_set_bits != 0)
    {
        visible_meshlets_writeout_offset = atomicAdd(deref(push.uses.visible_meshlets).count, number_of_newly_set_bits);
    }
    uint iter = 0;
    while (newly_set_bits != 0)
    {
        const uint lsb = findLSB(newly_set_bits);
        newly_set_bits &= ~(1 << lsb);
        const uint meshlet_instance_index = meshlet_bitfield_bit_to_index(meshlet_bitfield_index, lsb);

        const uint local_writeout_index = iter + visible_meshlets_writeout_offset;
        deref(push.uses.visible_meshlets).meshlet_ids[local_writeout_index] = meshlet_instance_index;

        
        #if 0
            // TODO: Profile and optimize this:
            MeshletInstance meshlet_instance = deref_i(deref(push.uses.meshlet_instances).meshlets, meshlet_instance_index);
            uint mesh_instance_index = meshlet_instance.mesh_instance_index;
            uint bitfield_index = mesh_instance_index / 32;
            uint bitfield_mask = 1u << (mesh_instance_index % 32);
            uint ret_bitfield_mask = atomicOr(deref_i(push.uses.mesh_visibility_bitfield, bitfield_index), bitfield_mask);
            if ((bitfield_mask & ret_bitfield_mask) == 0)
            {
                // First to see mesh. Append to list of visible meshes.
                uint visible_mesh_list_index = atomicAdd(deref(push.uses.visible_meshes).count, 1);
                deref(push.uses.visible_meshes).mesh_ids[visible_mesh_list_index] = meshlet_instance.mesh_instance_index;
            }
        #endif
        ++iter;
    }
}

vec3 hsv2rgb(vec3 c) {
    vec4 k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

// Note:
// Bug we encountered before is that nvidia sometimes makes warps 16 size instead of 32.
// This meant that relying on all 32 thread indices caused bugs.
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
    uvec4 vis_ids = textureGather(daxa_usampler2D(push.uses.visbuffer, deref(push.uses.globals).samplers.linear_clamp), make_gather_uv(1.0f / push.size, sampleIndex), 0);
    uint list_mask = (vis_ids[0] != INVALID_TRIANGLE_ID ? (1u) : 0) |
                     (vis_ids[1] != INVALID_TRIANGLE_ID ? (1u << 1u) : 0) |
                     (vis_ids[2] != INVALID_TRIANGLE_ID ? (1u << 2u) : 0) |
                     (vis_ids[3] != INVALID_TRIANGLE_ID ? (1u << 3u) : 0);
    uvec4 meshlet_bitfield_masks;
    uvec4 meshlet_bitfield_indices;
    [[unroll]] for (uint i = 0; i < 4; ++i)
    {
        uint meshlet_instance_index = meshlet_instance_index_from_triangle_id(vis_ids[i]);
        meshlet_bitfield_masks[i] = meshlet_bitfield_bit(meshlet_instance_index);
        meshlet_bitfield_indices[i] = meshlet_bitfield_index(meshlet_instance_index);
        ivec2 sample_locations[4] = { ivec2(0,1), ivec2(1,1), ivec2(1,0), ivec2(0,0), };
        ivec2 quad_offset = sample_locations[i];

        imageStore(daxa_image2D(push.uses.debug_image), sampleIndex + quad_offset, vec4(hsv2rgb(vec3(float(meshlet_bitfield_indices[i]) * 0.123424243523, 1, 1)),1));
    }        

    #if 1
    uint assigned_meshlet_bitfield_index = ~0;
    uint assigned_meshlet_bitfield_mask = 0;
    while (subgroupAny(assigned_meshlet_bitfield_index == ~0) && subgroupAny(list_mask != 0))
    {
        const bool lane_on = list_mask != 0;
        const uint voted_for_id = lane_on ? meshlet_bitfield_indices[findLSB(list_mask)] : ~0;
        const uint elected_meshlet_bitfield_index = subgroupBroadcast(voted_for_id, subgroupBallotFindLSB(subgroupBallot(lane_on)));
        // If we have the elected id in our list, remove it.
        uint meshlet_bitfield_mask_contribution = 0;
        [[unroll]] for (uint i = 0; i < 4; ++i)
        {
            if (meshlet_bitfield_indices[i] == elected_meshlet_bitfield_index)
            {
                meshlet_bitfield_mask_contribution |= meshlet_bitfield_masks[i];
                list_mask &= ~(1u << i);
            }
        }
        const uint warp_merged_meshlet_bit_mask = subgroupOr(meshlet_bitfield_mask_contribution);
        if ((assigned_meshlet_bitfield_index == ~0))
        {
            if (subgroupElect())
            {
                assigned_meshlet_bitfield_index = elected_meshlet_bitfield_index;
                assigned_meshlet_bitfield_mask = warp_merged_meshlet_bit_mask;
            }
        }
    }
    // Write out
    if (assigned_meshlet_bitfield_index != ~0)
    {
        update_visibility_masks_and_list2(assigned_meshlet_bitfield_index, assigned_meshlet_bitfield_mask);
    }
    #endif
    // Write out rest of local meshlet list:
    [[loop]] while (list_mask != 0)
    {
        const uint lsb = findLSB(list_mask);
        const uint meshlet_bitfield_index = meshlet_bitfield_indices[lsb];
        const uint meshlet_bitfield_mask = meshlet_bitfield_masks[lsb];
        list_mask &= ~(1 << lsb);
        update_visibility_masks_and_list2(meshlet_bitfield_index, meshlet_bitfield_mask);
        atomicAdd(deref(deref(push.uses.globals).debug).gpu_output.debug_ivec4.x, 1);
    }
}