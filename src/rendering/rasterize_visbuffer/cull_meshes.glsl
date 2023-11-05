#include <daxa/daxa.inl>

#include "cull_meshes.inl"
#include "shader_shared/cull_util.inl"

#define DEBUG_MESH_CULL 0
#define DEBUG_MESH_CULL1 0

#extension GL_EXT_debug_printf : enable

#if defined(CullMeshesCommand_COMMAND)
layout(local_size_x = 32) in;
void main()
{
    if (gl_GlobalInvocationID.x == 0)
    {
        const uint entity_count = deref(u_entity_meta).entity_count;
        const uint dispatch_x = (entity_count + CULL_MESHES_WORKGROUP_X - 1) / CULL_MESHES_WORKGROUP_X;
        deref(u_command).x = dispatch_x;
        deref(u_command).y = 1;
        deref(u_command).z = 1;
    }
    const uint index = gl_GlobalInvocationID.x;
    uint64_t sizeof_arg_table = my_sizeof(MeshletCullIndirectArgTable);
    uint64_t sizeof_arg = my_sizeof(MeshletCullIndirectArg);
    // Each table is half the size of the previous one.
    // Table power 0 is the size of MAX_MESHLET_INSTANCES.
    // index 0 starts at sizeof_arg_table + sizeof_arg * MAX_MESHLET_INSTANCES,
    // indes 1 starts at sizeof_arg_table + sizeof_arg * MAX_MESHLET_INSTANCES * 1/2,
    // index 2 ...
    daxa_u64 addr = daxa_u64(u_meshlet_cull_indirect_args) + sizeof_arg_table + (sizeof_arg * MAX_MESHLET_INSTANCES >> index);
    deref(u_meshlet_cull_indirect_args).indirect_arg_ptrs[index] = daxa_RWBufferPtr(MeshletCullIndirectArg)(addr);
    deref(u_meshlet_cull_indirect_args).indirect_arg_counts[index] = 0;
    deref(u_cull_meshlets_commands[index]).x = 0;
    deref(u_cull_meshlets_commands[index]).y = 1;
    deref(u_cull_meshlets_commands[index]).z = 1;
}
#else
layout(local_size_x = CULL_MESHES_WORKGROUP_X, local_size_y = CULL_MESHES_WORKGROUP_Y) in;
void main()
{
    const uint entity_index = gl_GlobalInvocationID.x;
    const uint in_meshgroup_index = gl_LocalInvocationID.y;
    if (entity_index >= deref(u_entity_meta).entity_count)
    {
        return;
    }
    const uint meshgroup_index = deref(u_entity_meshgroup_indices[entity_index]);
    if (meshgroup_index == INVALID_MANIFEST_INDEX)
    {
        return;
    }
    const GPUMeshGroup mesh_group = deref(u_meshgroups + meshgroup_index);
    if (in_meshgroup_index >= mesh_group.count)
    {
        return;
    }
    const uint mesh_index = mesh_group.mesh_manifest_indices[in_meshgroup_index];
    const uint meshlet_count = deref(u_meshes[mesh_index]).meshlet_count;
    if (meshlet_count == 0)
    {
        return;
    }
    // if (entity_index != (globals.frame_index / 10) % 2480)
    // {
    //     return;
    // }
    if (entity_index == 0)
    {
        return;
    }

    // How does this work?
    // - this is an asymertric work distribution problem
    // - each mesh cull thread needs x followup threads where x is the number of meshlets for the mesh
    // - writing x times to some argument buffer to dispatch over later is extreamly divergent and inefficient
    //   - solution is to combine writeouts in powers of two:
    //   - instead of x writeouts, only do log2(x), one writeout per set bit in the meshletcount.
    //   - when you want to write out 17 meshlet work units, instead of writing 7 args into a buffer,
    //     you write one 1x arg, no 2x arg, no 4x arg, no 8x arg and one 16x arg. the 1x and the 16x args together contain 17 work units.
    // - still not good enough, in large cases like 2^16 - 1 meshlets it would need 15 writeouts, that is too much!
    //   - solution is to limit the writeouts to some smaller number (i chose 5, as it has a max thread waste of < 5%)
    //   - A strong compromise is to round up invocation count from meshletcount in such a way that the round up value only has 4 bits set at most.
    //   - as we do one writeout per bit set in meshlet count, this limits the writeout to 5.
    // - in worst case this can go down from thousands of divergent writeouts down to 5 while only wasting < 5% of invocations.
    const uint MAX_BITS = 5;
    uint meshlet_count_msb = findMSB(meshlet_count);
    const uint shift = uint(max(0, int(meshlet_count_msb) + 1 - int(MAX_BITS)));
    // clip off all bits below the 5 most significant ones.
    uint clipped_bits_meshlet_count = (meshlet_count >> shift) << shift;
    // Need to round up if there were bits clipped.
    if (clipped_bits_meshlet_count < meshlet_count)
    {
        clipped_bits_meshlet_count += (1 << shift);
    }
    // Now bit by bit, do one writeout of an indirect command:
    uint bucket_bit_mask = clipped_bits_meshlet_count;
#if DEBUG_MESH_CULL
    if (bitCount(meshlet_count) >= MAX_BITS || clipped_bits_meshlet_count != meshlet_count)
    {
        const float wasted = (1.0f - float(meshlet_count) / float(clipped_bits_meshlet_count)) * 100.0f;
        debugPrintfEXT("cull mesh index %u for entity %u:\n in mesh group index: %u\n  meshletcount: (%u)->(%u)\n  bitCount: (%u)->(%u)\n  new bitCount <= old bitCount? %u\n  new meshletcount >= old meshletcount?: %u\n  wasted: %f%%\n\n",
                       mesh_index,
                       entity_index,
                       in_meshgroup_index,
                       meshlet_count,
                       clipped_bits_meshlet_count,
                       bitCount(meshlet_count),
                       bitCount(clipped_bits_meshlet_count),
                       ((bitCount(clipped_bits_meshlet_count) <= bitCount(meshlet_count)) ? 1 : 0),
                       ((clipped_bits_meshlet_count >= meshlet_count) ? 1 : 0),
                       wasted);
    }
#endif
    // Each time we write out a command we add on the number of meshlets processed by that arg.
    uint meshlet_offset = 0;
    while (bucket_bit_mask != 0)
    {
        const uint bucket_index = findMSB(bucket_bit_mask);
        const uint indirect_arg_meshlet_count = 1 << (bucket_index);
        // Mask out bit.
        bucket_bit_mask &= ~indirect_arg_meshlet_count;
        const uint arg_array_offset = atomicAdd(deref(u_meshlet_cull_indirect_args).indirect_arg_counts[bucket_index], 1);
        // Update indirect args for meshlet cull
        {
            const uint threads_per_indirect_arg = 1 << bucket_index;

            const uint work_group_size = (globals.settings.enable_mesh_shader == 1) ? TASK_SHADER_WORKGROUP_X : CULL_MESHLETS_WORKGROUP_X;
            const uint prev_indirect_arg_count = arg_array_offset;
            const uint prev_needed_threads = threads_per_indirect_arg * prev_indirect_arg_count;
            const uint prev_needed_workgroups = (prev_needed_threads + work_group_size - 1) / work_group_size;
            const uint cur_indirect_arg_count = arg_array_offset + 1;
            const uint cur_needed_threads = threads_per_indirect_arg * cur_indirect_arg_count;
            const uint cur_needed_workgroups = (cur_needed_threads + work_group_size - 1) / work_group_size;

            const bool update_cull_meshlets_dispatch = prev_needed_workgroups != cur_needed_workgroups;
            if (update_cull_meshlets_dispatch)
            {
                atomicMax(deref(u_cull_meshlets_commands[bucket_index]).x, cur_needed_workgroups);
            }
        }
        MeshletCullIndirectArg arg;
        arg.entity_index = entity_index;
        arg.mesh_index = mesh_index;
        arg.material_index = 0; // TODO: hook up materials.
        arg.in_meshgroup_index = in_meshgroup_index;
        arg.meshlet_indices_offset = meshlet_offset;
        deref(deref(u_meshlet_cull_indirect_args).indirect_arg_ptrs[bucket_index][arg_array_offset]) = arg;
        // debugPrintfEXT("test\n");
        meshlet_offset += indirect_arg_meshlet_count;
    }
#if DEBUG_MESH_CULL1
    if (meshlet_count > 33)
    {
        uint meshlet_offset = 0;
        uint bucket_bit_mask = clipped_bits_meshlet_count;
        uint powers[5] = {33, 33, 33, 33, 33};
        uint counts[5] = {0, 0, 0, 0, 0};
        uint index = 0;
        while (bucket_bit_mask != 0)
        {
            const uint bucket_index = findMSB(bucket_bit_mask);
            const uint indirect_arg_meshlet_count = 1 << (bucket_index);
            // Mask out bit.
            bucket_bit_mask &= ~indirect_arg_meshlet_count;
            powers[index] = bucket_index;
            counts[index] = indirect_arg_meshlet_count;
            ++index;
        }
        debugPrintfEXT("wrote out %u work units for %u meshlets,\n  meshlet count rounded up to fit 4 consequtive bits in uint (%u)->(%u),\n  power[0]: %u, meshlets: %u\n  power[1]: %u, meshlets: %u\n  power[2]: %u, meshlets: %u\n  power[3]: %u, meshlets: %u\n  power[4]: %u, meshlets: %u\n",
                       meshlet_offset, meshlet_count,
                       meshlet_count, clipped_bits_meshlet_count,
                       powers[0], counts[0], powers[1], counts[1], powers[2], counts[2], powers[3], counts[3], powers[4], counts[4]);
    }
#endif
}
#endif
