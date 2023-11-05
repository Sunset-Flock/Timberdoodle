#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "prepopulate_inst_meshlets.inl"

#include "shader_lib/cull_util.glsl"

#define WORKGROUP_SIZE PREPOPULATE_INST_MESHLETS_X

#if defined(PrepopulateInstantiatedMeshletsCommandWrite_COMMAND)
layout(local_size_x = 1) in;
void main()
{
    const uint needed_threads = deref(u_visible_meshlets_prev).count;
    const uint needed_workgroups = (needed_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    DispatchIndirectStruct command;
    command.x = needed_workgroups;
    command.y = 1;
    command.z = 1;
    deref(u_command) = command;
}
#elif defined(SetEntityMeshletVisibilityBitMasks_SHADER)
layout(local_size_x = WORKGROUP_SIZE) in;
void main()
{
    const uint count = deref(u_instantiated_meshlets).first_count;
    const uint inst_meshlet_index = gl_GlobalInvocationID.x;
    const bool thread_active = inst_meshlet_index < count;
    // TODO: check if entity, its mesh id and meshlets are valid
    if (thread_active)
    {
        MeshletInstance inst_meshlet = unpack_meshlet_instance(deref(u_instantiated_meshlets).meshlets[inst_meshlet_index]);
        const uint mask_bit = 1u << (inst_meshlet.meshlet_index % 32);
        const uint local_mask_offset = inst_meshlet.meshlet_index / 32;
        const uint global_mask_offset = u_entity_meshlet_visibility_bitfield_offsets
                                            .entity_offsets[inst_meshlet.entity_index]
                                            .mesh_bitfield_offset[inst_meshlet.in_meshgroup_index];
        const uint offset = global_mask_offset + local_mask_offset;

        atomicOr(deref(u_entity_meshlet_visibility_bitfield_arena[offset]), mask_bit);
    }
}
#else
shared uint s_out_count;
shared uint s_out_offset;
layout(local_size_x = WORKGROUP_SIZE) in;
void main()
{
    const uint count = deref(u_visible_meshlets_prev).count;
    const bool thread_active = gl_GlobalInvocationID.x < count;
    // TODO: check if entity, its mesh id and meshlets are valid
    uint inst_meshlet_index_prev = 0;
    MeshletInstance inst_meshlet;
    if (thread_active)
    {
        inst_meshlet_index_prev = deref(u_visible_meshlets_prev).meshlet_ids[gl_GlobalInvocationID.x];
        inst_meshlet = unpack_meshlet_instance(deref(u_instantiated_meshlets_prev).meshlets[inst_meshlet_index_prev]);
        const uint counters_index = inst_meshlet.entity_index * 8 + inst_meshlet.in_meshgroup_index;
        const uint prev_value = atomicCompSwap(
            u_entity_meshlet_visibility_bitfield_offsets.entity_offsets[inst_meshlet.entity_index].mesh_bitfield_offset[inst_meshlet.in_meshgroup_index],
            ENT_MESHLET_VIS_OFFSET_UNALLOCATED,
            ENT_MESHLET_VIS_OFFSET_EMPTY);
        // Saw entity (and entity mesh index) the first time -> allocate bitfield offset
        if (prev_value == ENT_MESHLET_VIS_OFFSET_UNALLOCATED)
        {
            const uint meshlets_in_mesh = deref(u_meshes[inst_meshlet.mesh_index]).meshlet_count;
            if (meshlets_in_mesh > 0)
            {
                const uint needed_uints_in_bitfield = (meshlets_in_mesh + 32 - 1) / 32;
                const uint bitfield_arena_offset = atomicAdd(u_entity_meshlet_visibility_bitfield_offsets.back_offset, needed_uints_in_bitfield);
                atomicExchange(u_entity_meshlet_visibility_bitfield_offsets.entity_offsets[inst_meshlet.entity_index].mesh_bitfield_offset[inst_meshlet.in_meshgroup_index], bitfield_arena_offset);
            }
        }
    }

    if (gl_LocalInvocationID.x == 0)
    {
        s_out_count = 0;
    }
    memoryBarrierShared();
    barrier();
    uint local_offset = 0;
    if (thread_active)
    {
        local_offset = atomicAdd(s_out_count, 1);
    }
    memoryBarrierShared();
    barrier();
    if (gl_LocalInvocationID.x == 0)
    {
        s_out_offset = atomicAdd(deref(u_instantiated_meshlets).first_count, s_out_count);
    }
    memoryBarrierShared();
    barrier();
    if (thread_active)
    {
        const uint meshlet_instance_index = s_out_offset + local_offset;
        // Write out meshlet instance to the meshlet instance list of the first pass:
        deref(u_instantiated_meshlets).meshlets[meshlet_instance_index] = pack_meshlet_instance(inst_meshlet);
    }
}
#endif

// 
// First Pass Preparation: 50 mics
// First Pass Draw: 250 mics
// Cull and prep second Pass: 70 mics
// Draw Second pass: 110 mics
// Post analyze: 44mics
// TOTAL: 526
// TOTAL FIRST PASS PREP: 95mics
// 
// Raw draw prep: 53 mics
// Raw Draw: 450mics
// TOTAL: 503
// 
// 
// 
// 
// 