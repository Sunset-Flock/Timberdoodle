#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "prepopulate_meshlets.inl"

#if defined(PrepopMeshletInstancesCommWH_SHADER)
DAXA_DECL_PUSH_CONSTANT(PrepopMeshletInstancesCommWPush, push)
#elif defined(WriteFirstPassMeshletsAndBitfields_SHADER)
DAXA_DECL_PUSH_CONSTANT(WriteFirstPassMeshletsAndBitfieldsPush, push)
#elif defined(AllocEntToMeshInstOffsetsOffsets_SHADER)
DAXA_DECL_PUSH_CONSTANT(AllocEntToMeshInstOffsetsOffsetsPush, push)
#elif defined(AllocMeshletInstBitfields_SHADER)
DAXA_DECL_PUSH_CONSTANT(AllocMeshletInstBitfieldsPush, push)
#endif

#include "shader_lib/cull_util.glsl"

#define WORKGROUP_SIZE PREPOPULATE_MESHLET_INSTANCES_X

#if defined(AllocEntToMeshInstOffsetsOffsets_SHADER)
layout(local_size_x = ALLOC_ENT_TO_MESH_INST_OFFSETS_OFFSETS_X) in;
void main()
{
    uint mesh_draw_index = gl_GlobalInvocationID.x;
    uint opaque_draw_list_index = 0;
    // We do a single dispatch to cull both lists, the shader threads simply check if their id overruns the first list,
    // then assign themselves to an element in the second list.
    if (mesh_draw_index >= deref(push.uses.opaque_mesh_draw_lists).list_sizes[0])
    {
        mesh_draw_index -= deref(push.uses.opaque_mesh_draw_lists).list_sizes[0];
        opaque_draw_list_index = 1;
        if (mesh_draw_index >= deref(push.uses.opaque_mesh_draw_lists).list_sizes[1])
        {
            return;
        }
    }

    const MeshDrawTuple mesh_draw = deref(deref(push.uses.opaque_mesh_draw_lists).mesh_draw_tuples[opaque_draw_list_index][mesh_draw_index]);
    const uint mesh_group_index = deref(push.uses.entity_mesh_groups[mesh_draw.entity_index]);
    if (mesh_group_index == INVALID_MANIFEST_INDEX)
    {
        // Entity has no mesh group.
        return;
    }
    GPUMeshGroup mesh_group = deref(push.uses.mesh_groups[mesh_group_index]);
    if (mesh_group.count == 0)
    {
        // Broken mesh group
        return;
    }

    const bool locked_entities_offset = FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID == atomicCompSwap(
        deref(push.uses.ent_to_mesh_inst_offsets_offsets[mesh_draw.entity_index]),
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID,
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED
    );
    if (!locked_entities_offset)
    {
        return;
    }

    uint allocation_offset = atomicAdd(
        push.uses.bitfield_arena.offsets_section_size, 
        mesh_group.count
    );
    const uint offsets_section_size = allocation_offset + mesh_group.count;
    if (offsets_section_size < (FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE - 2))
    {
        atomicExchange(
            deref(push.uses.ent_to_mesh_inst_offsets_offsets[mesh_draw.entity_index]),
            allocation_offset
        );
        // Write indirect clear command, clearing the section used to store the mesh bitfield offsets.
        atomicMax(
            deref(push.uses.clear_arena_command).size, 
            offsets_section_size
        );
        const uint needed_clear_workgroups = round_up_div(offsets_section_size, INDIRECT_MEMSET_BUFFER_X);
        atomicMax(
            deref(push.uses.clear_arena_command).dispatch.x, 
            needed_clear_workgroups
        );
    }
}
#endif // #if defined(AllocEntToMeshInstOffsetsOffsets_SHADER)

#if defined(PrepopMeshletInstancesCommWH_SHADER)
layout(local_size_x = 1) in;
void main()
{
    const uint needed_threads = deref(push.uses.visible_meshlets_prev).count;
    const uint needed_workgroups = round_up_div(needed_threads, WORKGROUP_SIZE);
    DispatchIndirectStruct command;
    command.x = needed_workgroups;
    command.y = 1;
    command.z = 1;
    deref(push.uses.command) = command;
}
#endif

#if defined(WriteFirstPassMeshletsAndBitfields_SHADER)
layout(local_size_x = WORKGROUP_SIZE) in;
void main()
{    
    const uint count = deref(push.uses.visible_meshlets_prev).count;
    const uint thread_index = gl_GlobalInvocationID.x;
    if (thread_index >= count)
    {
        return;
    }

    const uint prev_frame_meshlet_idx = deref(push.uses.visible_meshlets_prev).meshlet_ids[thread_index];
    const MeshletInstance prev_frame_vis_meshlet = deref(deref(push.uses.meshlet_instances_prev).meshlets[prev_frame_meshlet_idx]);

    const uint first_pass_meshgroup_bitfield_offset = deref(push.uses.ent_to_mesh_inst_offsets_offsets[prev_frame_vis_meshlet.entity_index]);
    if ((first_pass_meshgroup_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID) && 
        (first_pass_meshgroup_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED))
    {
        const uint mesh_instance_bitfield_offset_offset = first_pass_meshgroup_bitfield_offset + prev_frame_vis_meshlet.in_mesh_group_index;
        // Offset is valid, need to check if mesh instance offset is valid now.
        const uint first_pass_mesh_instance_bitfield_offset = push.uses.bitfield_arena.uints[mesh_instance_bitfield_offset_offset];
        if ((first_pass_mesh_instance_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID) && 
            (first_pass_mesh_instance_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED))
        {
            // Try to allocate new meshlet instance:
            const uint meshlet_instance_index = atomicAdd(deref(push.uses.meshlet_instances).first_count, 1);
            if (meshlet_instance_index >= MAX_MESHLET_INSTANCES)
            {
                // Overrun Meshlet instance buffer!
                return;
            }

            const uint in_bitfield_u32_index = prev_frame_vis_meshlet.meshlet_index / 32 + first_pass_mesh_instance_bitfield_offset;
            const uint in_u32_bit = prev_frame_vis_meshlet.meshlet_index % 32;
            const uint in_u32_mask = 1u << in_u32_bit;
            const uint prior_bitfield_u32 = atomicOr(push.uses.bitfield_arena.uints[in_bitfield_u32_index], in_u32_mask);
            if ((prior_bitfield_u32 & in_u32_mask) != 0)
            {
                // SHOULD NEVER HAPPEN:
                // - THIS MEANS WE HAVE DUPLICATE MESHLETS IN THE VISIBLE MESHLET LIST!
                // - THIS MEANS WE HAVE DRAWN THOSE MESHLETS MULTIPLE TIMES SOMEHOW!
                debugPrintfEXT("GPU ASSERT FAILED: DUPLICATE MESHLET %i\n", prev_frame_meshlet_idx);
                return;
            }
            
            // Write meshlet instance into draw list and instance list:
            deref(deref(push.uses.meshlet_instances).meshlets[meshlet_instance_index]) = prev_frame_vis_meshlet;
            uint opaque_draw_list_type_index = OPAQUE_DRAW_LIST_MASKED;
            if (prev_frame_vis_meshlet.material_index != INVALID_MANIFEST_INDEX)
            {
                GPUMaterial material = deref(push.uses.materials[prev_frame_vis_meshlet.material_index]);
                opaque_draw_list_type_index = material.alpha_discard_enabled ? OPAQUE_DRAW_LIST_MASKED : OPAQUE_DRAW_LIST_SOLID;
            }
            // Scalarize appends to the draw lists.
            // Scalarized atomics probably give consecutive retrun values for each thread within the warp (true on RTX4080).
            // This allows for scalar atomic ops and packed writeouts.
            // Drawlist type count are low, scalarization will most likely always improve perf.
            [[unroll]]
            for (uint draw_list_type = 0; draw_list_type < OPAQUE_DRAW_LIST_COUNT; ++draw_list_type)
            {
                if (opaque_draw_list_type_index != draw_list_type)
                {
                    continue;
                }
                // NOTE: Can never overrun buffer here as this is always <= meshlet_instances.first_count!
                const uint opaque_draw_list_index = atomicAdd(deref(push.uses.meshlet_instances).draw_lists[draw_list_type].first_count, 1);
                deref(deref(push.uses.meshlet_instances).draw_lists[draw_list_type].instances[opaque_draw_list_index]) = meshlet_instance_index;
            } 
        }
    }
}
#endif

#if defined(AllocMeshletInstBitfields_SHADER)
layout(local_size_x = WORKGROUP_SIZE) in;
void main()
{
    const uint bitfield_arena_base_offset = push.uses.bitfield_arena.offsets_section_size;
    if (gl_GlobalInvocationID.x == 0)
    {
        // One thread must write the offset past the mesh instance offset section.
        deref(push.uses.clear_arena_command).offset = bitfield_arena_base_offset + 2 /*buffer head*/;
        // debugPrintfEXT("deref(push.uses.visible_meshlets_prev).count %i\n", deref(push.uses.visible_meshlets_prev).count);
    }

    const uint count = deref(push.uses.visible_meshlets_prev).count;
    const uint thread_index = gl_GlobalInvocationID.x;
    if (thread_index >= count)
    {
        //debugPrintfEXT("thread_index >= count: %i >= %i\n", thread_index, count);
        return;
    }

    const uint prev_frame_meshlet_idx = deref(push.uses.visible_meshlets_prev).meshlet_ids[thread_index];
    const MeshletInstance prev_frame_vis_meshlet = deref(deref(push.uses.meshlet_instances_prev).meshlets[prev_frame_meshlet_idx]);

    const uint entity_to_meshgroup_bitfield_offset = deref(push.uses.ent_to_mesh_inst_offsets_offsets[prev_frame_vis_meshlet.entity_index]);
    if (entity_to_meshgroup_bitfield_offset == FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID || 
        entity_to_meshgroup_bitfield_offset == FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED)
    {
        // Entity and its mesh group are not present in the current bitfield.
        return;
    }

    // Try to lock the mesh instance offset:
    const uint mesh_instance_bitfield_offset_offset = entity_to_meshgroup_bitfield_offset + prev_frame_vis_meshlet.in_mesh_group_index;
    const bool locked_mesh_instance_offset = FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID == atomicCompSwap(
        push.uses.bitfield_arena.uints[mesh_instance_bitfield_offset_offset],
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID,
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED
    );
    if (!locked_mesh_instance_offset)
    {
        // Another thread locked the mesh instance offset, we can return
        return;
    }

    GPUMesh mesh = deref(push.uses.meshes[prev_frame_vis_meshlet.mesh_index]);

    // Try to allocate bitfield space:
    const uint needed_bitfield_u32_size = round_up_div(mesh.meshlet_count, 32);
    const uint bitfield_section_local_offset = atomicAdd(push.uses.bitfield_arena.bitfield_section_size, needed_bitfield_u32_size);
    const uint bitfield_allocation_offset = bitfield_arena_base_offset + bitfield_section_local_offset;
    const uint potentially_used_bitfield_size = bitfield_allocation_offset + needed_bitfield_u32_size;
    const uint potentially_used_bitfield_size_section_local = bitfield_section_local_offset + needed_bitfield_u32_size;
    if (potentially_used_bitfield_size >= (FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE - 2))
    {
        // Allocation failed, as we do not fit into the arena!
        return;
    }

    // We succeeded in allocating and locking the offset.
    // - Write bitfield offset of meshlet instance
    // - Write clear bitfield section indirect command 

    // Write allocated bitfield offset to mesh instances offset
    atomicExchange(push.uses.bitfield_arena.uints[mesh_instance_bitfield_offset_offset], bitfield_allocation_offset);

    // Write clear bitfield section indirect command:
    const uint needed_clear_workgroups = round_up_div(potentially_used_bitfield_size_section_local, INDIRECT_MEMSET_BUFFER_X);
    atomicMax(deref(push.uses.clear_arena_command).dispatch.x, needed_clear_workgroups);
    atomicMax(deref(push.uses.clear_arena_command).size, potentially_used_bitfield_size_section_local);
}
#endif