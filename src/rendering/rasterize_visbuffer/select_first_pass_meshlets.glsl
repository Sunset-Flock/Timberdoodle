#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "select_first_pass_meshlets.inl"

#if defined(WriteFirstPassMeshletsAndBitfields_SHADER)
DAXA_DECL_PUSH_CONSTANT(WriteFirstPassMeshletsAndBitfieldsPush, push)
#elif defined(AllocEntToMeshInstOffsetsOffsets_SHADER)
DAXA_DECL_PUSH_CONSTANT(AllocEntBitfieldListsPush, push)
#elif defined(AllocMeshletInstBitfields_SHADER)
DAXA_DECL_PUSH_CONSTANT(AllocMeshletInstBitfieldsPush, push)
#endif

#define WORKGROUP_SIZE SFPM_X

#if defined(AllocEntToMeshInstOffsetsOffsets_SHADER)
layout(local_size_x = SFPM_X) in;
void main()
{
    if (all(equal(gl_GlobalInvocationID, uvec3(0,0,0))))
    {
        const uint needed_threads = deref(push.attach.visible_meshlets_prev).count;
        const uint needed_workgroups = round_up_div(needed_threads, WORKGROUP_SIZE);
        DispatchIndirectStruct command;
        command.x = needed_workgroups;
        command.y = 1;
        command.z = 1;
        deref(push.attach.command) = command;
        // Initialize offset to be past the dynamic_offset counter and entity offsets.
        push.attach.bitfield_arena.dynamic_offset = FIRST_PASS_MESHLET_BITFIELD_OFFSET_SECTION_START;
    }
    uint mesh_instance_index = gl_GlobalInvocationID.x;
    uint mesh_instance_count = min(deref(push.attach.mesh_instances).count, MAX_MESH_INSTANCES);
    if (mesh_instance_index >= mesh_instance_count)
    {
        return;
    }
    MeshInstance mesh_instance = deref_i(deref(push.attach.mesh_instances).instances, mesh_instance_index);
    uint opaque_draw_list_index = ((mesh_instance.flags & MESH_INSTANCE_FLAG_OPAQUE) != 0) ? PREPASS_DRAW_LIST_OPAQUE : PREPASS_DRAW_LIST_MASKED;

    const uint mesh_group_index = deref(push.attach.entity_mesh_groups[mesh_instance.entity_index]);
    if (mesh_group_index == INVALID_MANIFEST_INDEX)
    {
        // Entity has no mesh group.
        return;
    }

    if (mesh_instance.entity_index >= MAX_ENTITIES)
    {
        // Entity index out of bounds.
        return;
    }
    
    GPUMeshGroup mesh_group = deref(push.attach.mesh_groups[mesh_group_index]);
    if (mesh_group.count == 0)
    {
        // Broken mesh group
        return;
    }

    // /// bug zone begin
    // const bool locked_entities_offset = FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID == atomicCompSwap(
    //     deref(push.attach.ent_to_mesh_inst_offsets_offsets[mesh_instance.entity_index]),
    //     FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID,
    //     FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED
    // );
    // if (!locked_entities_offset)
    // {
    //     return;
    // }    
    
    const bool locked_entities_offset = FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID == atomicCompSwap(
        push.attach.bitfield_arena.entity_to_meshlist_offsets[mesh_instance.entity_index],
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID,
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED
    );
    if (!locked_entities_offset)
    {
        return;
    }
    /// bug zone end

    uint allocation_offset = atomicAdd(
        push.attach.bitfield_arena.dynamic_offset, 
        mesh_group.count
    );
    const uint offsets_section_size = allocation_offset + mesh_group.count;
    if (offsets_section_size < (FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE))
    {
        atomicExchange(
            push.attach.bitfield_arena.entity_to_meshlist_offsets[mesh_instance.entity_index],
            allocation_offset
        );
    }
}
#endif // #if defined(AllocEntToMeshInstOffsetsOffsets_SHADER)

#if defined(AllocMeshletInstBitfields_SHADER)
layout(local_size_x = WORKGROUP_SIZE) in;
void main()
{
    uint entity_index = 0;
    uint mesh_index = 0;
    uint in_mesh_group_index = 0;
    if (deref(push.attach.globals).settings.enable_visbuffer_two_pass_culling)
    {
        const uint count = deref(push.attach.visible_meshlets_prev).count;
        const uint thread_index = gl_GlobalInvocationID.x;
        if (thread_index >= count)
        {
            //debugPrintfEXT("thread_index >= count: %i >= %i\n", thread_index, count);
            return;
        }
        const uint prev_frame_meshlet_idx = deref(push.attach.visible_meshlets_prev).meshlet_ids[thread_index];
        const MeshletInstance prev_frame_vis_meshlet = deref(deref(push.attach.meshlet_instances_prev).meshlets[prev_frame_meshlet_idx]);

        entity_index = prev_frame_vis_meshlet.entity_index;
        in_mesh_group_index = prev_frame_vis_meshlet.in_mesh_group_index;
        mesh_index = prev_frame_vis_meshlet.mesh_index;
    }
    else // For non visbuffer two pass culling, we have to allocate a bit for every possible meshlet
    {
        uint mesh_instance_index = gl_GlobalInvocationID.x;
        uint mesh_instance_count = min(deref(push.attach.mesh_instances).count, MAX_MESH_INSTANCES);
        if (mesh_instance_index >= mesh_instance_count)
        {
            return;
        }
        MeshInstance mesh_instance = deref_i(deref(push.attach.mesh_instances).instances, mesh_instance_index);

        entity_index = mesh_instance.entity_index;
        in_mesh_group_index = mesh_instance.in_mesh_group_index;
        mesh_index = mesh_instance.mesh_index;
    }

    const uint entity_to_meshgroup_bitfield_offset = push.attach.bitfield_arena.entity_to_meshlist_offsets[entity_index];
    if (entity_to_meshgroup_bitfield_offset == FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID || 
        entity_to_meshgroup_bitfield_offset == FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED)
    {
        // Entity and its mesh group are not present in the current bitfield.
        return;
    }

    // Try to lock the mesh instance offset:
    const uint mesh_instance_bitfield_offset_offset = entity_to_meshgroup_bitfield_offset + in_mesh_group_index;
    const bool locked_mesh_instance_offset = FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID == atomicCompSwap(
        push.attach.bitfield_arena.dynamic_section[mesh_instance_bitfield_offset_offset],
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID,
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED
    );
    if (!locked_mesh_instance_offset)
    {
        // Another thread locked the mesh instance offset, we can return
        return;
    }

    GPUMesh mesh = deref(push.attach.meshes[mesh_index]);
    if (mesh.mesh_buffer.value == 0)
    {
        // Unloaded Mesh
        return;
    }

    // Try to allocate bitfield space:
    const uint needed_bitfield_u32_size = round_up_div(mesh.meshlet_count, 32);
    const uint bitfield_section_local_offset = atomicAdd(push.attach.bitfield_arena.dynamic_offset, needed_bitfield_u32_size);
    const uint potentially_used_bitfield_size = bitfield_section_local_offset + needed_bitfield_u32_size;
    const uint potentially_used_bitfield_size_section_local = bitfield_section_local_offset + needed_bitfield_u32_size;
    if (potentially_used_bitfield_size >= (FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE))
    {
        // Allocation failed, as we do not fit into the arena!
        return;
    }

    // We succeeded in allocating and locking the offset.
    // - Write bitfield offset of meshlet instance
    // - Write clear bitfield section indirect command 

    // Write allocated bitfield offset to mesh instances offset
    atomicExchange(push.attach.bitfield_arena.dynamic_section[mesh_instance_bitfield_offset_offset], bitfield_section_local_offset);
}
#endif


#if defined(WriteFirstPassMeshletsAndBitfields_SHADER)
layout(local_size_x = WORKGROUP_SIZE) in;
void main()
{    
    const uint count = deref(push.attach.visible_meshlets_prev).count; 
    const uint thread_index = gl_GlobalInvocationID.x;
    if (thread_index >= count)
    {
        return;
    }

    if (thread_index == 0)
    {
        deref(deref(push.attach.globals).readback).sfpm_bitfield_arena_requested = push.attach.bitfield_arena.dynamic_offset;
    }

    const uint prev_frame_meshlet_idx = deref(push.attach.visible_meshlets_prev).meshlet_ids[thread_index];
    const MeshletInstance prev_frame_vis_meshlet = deref(deref(push.attach.meshlet_instances_prev).meshlets[prev_frame_meshlet_idx]);

    const uint first_pass_meshgroup_bitfield_offset = push.attach.bitfield_arena.entity_to_meshlist_offsets[prev_frame_vis_meshlet.entity_index];
    if ((first_pass_meshgroup_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID) && 
        (first_pass_meshgroup_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED))
    {
        const uint mesh_instance_bitfield_offset_offset = first_pass_meshgroup_bitfield_offset + prev_frame_vis_meshlet.in_mesh_group_index;
        // Offset is valid, need to check if mesh instance offset is valid now.
        const uint first_pass_mesh_instance_bitfield_offset = push.attach.bitfield_arena.dynamic_section[mesh_instance_bitfield_offset_offset];
        if ((first_pass_mesh_instance_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID) && 
            (first_pass_mesh_instance_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED))
        {
            // Try to allocate new meshlet instance:
            const uint meshlet_instance_index = atomicAdd(deref(push.attach.meshlet_instances).pass_counts[0], 1);
            if (meshlet_instance_index >= MAX_MESHLET_INSTANCES)
            {
                // Overrun Meshlet instance buffer!
                return;
            }

            const uint in_bitfield_u32_index = prev_frame_vis_meshlet.meshlet_index / 32 + first_pass_mesh_instance_bitfield_offset;
            const uint in_u32_bit = prev_frame_vis_meshlet.meshlet_index % 32;
            const uint in_u32_mask = 1u << in_u32_bit;
            const uint prior_bitfield_u32 = atomicOr(push.attach.bitfield_arena.dynamic_section[in_bitfield_u32_index], in_u32_mask);
            if ((prior_bitfield_u32 & in_u32_mask) != 0)
            {
                // SHOULD NEVER HAPPEN:
                // - THIS MEANS WE HAVE DUPLICATE MESHLETS IN THE VISIBLE MESHLET LIST!
                // - THIS MEANS WE HAVE DRAWN THOSE MESHLETS MULTIPLE TIMES SOMEHOW!
                debugPrintfEXT("GPU ASSERT FAILED: DUPLICATE MESHLET %i\n", prev_frame_meshlet_idx);
                return;
            }
            

            // debugPrintfEXT("process prev meshlet vis index %i, meshlet id %i\n", thread_index, prev_frame_meshlet_idx);

            // Write meshlet instance into draw list and instance list:
            deref(deref(push.attach.meshlet_instances).meshlets[meshlet_instance_index]) = prev_frame_vis_meshlet;
            uint opaque_draw_list_type_index = PREPASS_DRAW_LIST_OPAQUE;
            if (prev_frame_vis_meshlet.material_index != INVALID_MANIFEST_INDEX)
            {
                GPUMaterial material = deref(push.attach.materials[prev_frame_vis_meshlet.material_index]);
                opaque_draw_list_type_index = material.alpha_discard_enabled ? PREPASS_DRAW_LIST_MASKED : PREPASS_DRAW_LIST_OPAQUE;
            }
            // Scalarize appends to the draw lists.
            // Scalarized atomics probably give consecutive retrun values for each thread within the warp (true on RTX4080).
            // This allows for scalar atomic ops and packed writeouts.
            // Drawlist type count are low, scalarization will most likely always improve perf.
            [[unroll]]
            for (uint draw_list_type = 0; draw_list_type < PREPASS_DRAW_LIST_TYPE_COUNT; ++draw_list_type)
            {
                if (opaque_draw_list_type_index != draw_list_type)
                {
                    continue;
                }
                // NOTE: Can never overrun buffer here as this is always <= meshlet_instances.first_count!
                const uint opaque_draw_list_index = atomicAdd(deref(push.attach.meshlet_instances).prepass_draw_lists[draw_list_type].pass_counts[0], 1);
                deref(deref(push.attach.meshlet_instances).prepass_draw_lists[draw_list_type].instances[opaque_draw_list_index]) = meshlet_instance_index;
            } 
        }
    }
}
#endif
