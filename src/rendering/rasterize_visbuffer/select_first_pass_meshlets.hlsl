#include <daxa/daxa.inl>

#include "select_first_pass_meshlets.inl"
#include "shader_lib/debug.glsl"

// Problems:
// As we iterate over the draw list in the allocation for meshlists,
// We ALWAYS allocate worst case mesh offsets...
// This is super wasteful as that will include ALL possible entities.
// We should only allocate enough for the visible entities!

[[vk::push_constant]] AllocEntBitfieldListsPush alloc_ent_mesh_offset_lists_push;
[numthreads(SFPM_ALLOC_ENT_BITFIELD_LISTS_X,1,1)]
func entry_alloc_ent_bitfield_lists(uint dtid : SV_DispatchThreadID)
{
    let push = alloc_ent_mesh_offset_lists_push;
    
    // First thread writes command for following SFPM dispatches.
    if (dtid == 0)
    {
        const uint needed_threads = deref(push.attach.visible_meshlets_prev).count;
        const uint needed_workgroups = round_up_div(needed_threads, SFPM_X);
        DispatchIndirectStruct command;
        command.x = needed_workgroups;
        command.y = 1;
        command.z = 1;
        *push.attach.command = command;
        // As 0 is used as a clear value for the arena, we must have the 0 offset be invalid.
        // The first index reserved should therefor be 1.
        push.attach.bitfield_arena.dynamic_offset = 1;
    }

    uint threads_mapped = 0;
    uint draw_list_type = 0;
    uint draw_list_index = ~0u;
    for (uint i = 0; i < PREPASS_DRAW_LIST_TYPE_COUNT; ++i)
    {
        let potential_draw_list_index = dtid - threads_mapped;
        let draw_list_instance_count = push.attach.mesh_instances.prepass_draw_lists[i].count;
        if (potential_draw_list_index < draw_list_instance_count)
        {
            draw_list_index = potential_draw_list_index;
            draw_list_type = i;
            break;
        }
        threads_mapped += draw_list_instance_count;
    }
    if (draw_list_index == ~0u)
    {
        return;
    }

    let mesh_instance_index = push.attach.mesh_instances.prepass_draw_lists[draw_list_type].instances[draw_list_index];
    let mesh_instance_count = min(push.attach.mesh_instances.count, MAX_MESH_INSTANCES);
    if (mesh_instance_index >= mesh_instance_count)
    {
        printf(GPU_ASSERT_STRING"mesh instance index (%u) out of bounds (%u)!\n", mesh_instance_index, mesh_instance_count);
        return;
    }
    MeshInstance mesh_instance = push.attach.mesh_instances.instances[mesh_instance_index];

    if (mesh_instance.entity_index >= MAX_ENTITIES)
    {
        printf(GPU_ASSERT_STRING"entity index (%u) out of bounds (%u)!\n", mesh_instance.entity_index, MAX_ENTITIES);
        return;
    }

    const uint mesh_group_index = push.attach.globals.scene.entity_to_meshgroup[mesh_instance.entity_index];
    if (mesh_group_index == INVALID_MANIFEST_INDEX)
    {
        return;
    }
    
    GPUMeshGroup mesh_group = push.attach.globals.scene.mesh_groups[mesh_group_index];
    if (mesh_group.mesh_lod_group_count == 0)
    {
        printf(GPU_ASSERT_STRING"entity index (%u) has mesh group (%u) with 0 mesh_lod_group_count!\n", mesh_instance.entity_index, mesh_group.mesh_lod_group_count);
        return;
    }
    
    // Only a single thread is elected to perform the allocation for an entity in the drawlists.
    uint prev_value = 0;
    InterlockedCompareExchange(
        push.attach.bitfield_arena.entity_to_meshlist_offsets[mesh_instance.entity_index],
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID,
        FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED,
        prev_value);
    let is_thread_elected = prev_value == FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID;
    if (!is_thread_elected)
    {
        return;
    }

    uint allocation_offset = 0;
    InterlockedAdd(push.attach.bitfield_arena.dynamic_offset, mesh_group.mesh_lod_group_count, allocation_offset);
    // allocation_offset += FIRST_PASS_MESHLET_BITFIELD_OFFSET_SECTION_START;
    let offsets_section_size = allocation_offset + mesh_group.mesh_lod_group_count;
    if (offsets_section_size < FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE)
    {
        InterlockedExchange(push.attach.bitfield_arena.entity_to_meshlist_offsets[mesh_instance.entity_index], allocation_offset);
    }
    else
    {
        InterlockedAdd(push.attach.globals.readback.sfpm_bitfield_arena_allocation_failures_ent_pass, 1);
    }
}



[[vk::push_constant]] AllocMeshletInstBitfieldsPush alloc_mesh_offsets_push;
[[vk::push_constant]] WriteFirstPassMeshletsAndBitfieldsPush write_meshlets_push;