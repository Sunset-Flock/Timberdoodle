#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "filter_visible_triangles.inl"
#include "cull_util.glsl"

// Notes:
// - wg_ = work group
// - div and mod by constant is fast
// - div and mod by pow of 2 constant is blazing fast.
// - 1 thread per triangle

#define WORKGROUP_SIZE 1024
#define WORK_THREADS_PER_MESHLET (128)
#define MESHLETS_PER_WORKGROUP (WORKGROUP_SIZE / WORK_THREADS_PER_MESHLET)
#define BITS_PER_UINT (32)
#define UINTS_PER_MASK (WORK_THREADS_PER_MESHLET / BITS_PER_UINT)

#if defined(FilterVisibleTrianglesWriteCommand_COMMAND)
layout(local_size_x = 1) in;
void main()
{
    const uint needed_threads = WORK_THREADS_PER_MESHLET * deref(u_instantiated_meshlets).count;
    const uint needed_workgroups = (needed_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    DispatchIndirectStruct command;
    command.x = needed_workgroups;
    command.y = 1;
    command.z = 1;
    deref(u_command) = command;
}
#else
shared uvec4 s_masks[MESHLETS_PER_WORKGROUP];
shared uint s_visible_triangles;
shared uint s_wg_visible_triangle_out_offset;
layout(local_size_x = WORKGROUP_SIZE) in;
void main()
{
    const uint meshlet_count = deref(u_instantiated_meshlets).count;
    const uint wg_invoc_id = gl_LocalInvocationID.x;
    const uint wg_meshlets_offset = (gl_WorkGroupID.x * MESHLETS_PER_WORKGROUP);
    // Coalese WORK_THREADS_PER_MESHLET reads into one each. Reducing 1024 loads to 8.
    if (wg_invoc_id == 0)
    {
        s_visible_triangles = 0;
    }
    if (wg_invoc_id < MESHLETS_PER_WORKGROUP)
    {
        const uint src_meshlet_index = wg_meshlets_offset + wg_invoc_id;
        // Avoid reading out of bounds.
        // Simply write 0's to fool the following threads and make them not do any work.
        if (src_meshlet_index < meshlet_count)
        {
            s_masks[wg_invoc_id] = deref(u_meshlet_visibility_bitfields[src_meshlet_index]);
        }
        else
        {
            s_masks[wg_invoc_id] = uvec4(0,0,0,0);
        }
    }
    memoryBarrierShared();
    barrier();
    const uint wg_mask_index = (wg_invoc_id / WORK_THREADS_PER_MESHLET) % MESHLETS_PER_WORKGROUP;
    const uint mask_vec_index = (wg_invoc_id / BITS_PER_UINT) % UINTS_PER_MASK;
    const uint bit_in_mask = 1u << (wg_invoc_id % BITS_PER_UINT);
    const bool triangle_visible = (s_masks[wg_mask_index][mask_vec_index] & bit_in_mask) != 0;
    const uint local_offset = atomicAdd(s_visible_triangles, (triangle_visible ? 1 : 0));
    memoryBarrierShared();
    barrier();
    if (wg_invoc_id == 0)
    {
        s_wg_visible_triangle_out_offset = atomicAdd(deref(u_visible_triangles).count, s_visible_triangles);
    }
    memoryBarrierShared();
    barrier();
    if (triangle_visible)
    {
        const uint out_index = s_wg_visible_triangle_out_offset + local_offset;
        const uint inst_meshlet_index = wg_meshlets_offset + (wg_invoc_id / WORK_THREADS_PER_MESHLET);
        const uint triangle_index = wg_invoc_id % WORK_THREADS_PER_MESHLET;
        uint triangle_id;
        encode_triangle_id(inst_meshlet_index, triangle_index, triangle_id);
        deref(u_visible_triangles).triangle_ids[out_index] = triangle_id;
    }
}
#endif