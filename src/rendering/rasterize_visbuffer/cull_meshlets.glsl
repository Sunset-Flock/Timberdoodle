#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "cull_meshlets.inl"

#include "shader_lib/debug.glsl"

DAXA_DECL_PUSH_CONSTANT(CullMeshletsPush,push)

#define GLOBALS deref(push.uses.globals)

#include "shader_lib/cull_util.glsl"

layout(local_size_x = CULL_MESHLETS_WORKGROUP_X) in;
void main()
{
    MeshletInstance instanced_meshlet;
    const bool valid_meshlet = get_meshlet_instance_from_arg(gl_GlobalInvocationID.x, push.indirect_args_table_id, push.uses.meshlets_cull_arg_buckets_buffer, instanced_meshlet);
    if (!valid_meshlet)
    {
        return;
    }
#if ENABLE_MESHLET_CULLING
    const bool occluded = is_meshlet_occluded(
        deref(push.uses.globals).camera,
        instanced_meshlet,
        push.uses.first_pass_meshlets_bitfield_offsets,
        push.uses.first_pass_meshlets_bitfield_arena,
        push.uses.entity_combined_transforms,
        push.uses.meshes,
        push.uses.hiz);
        
#else
    const bool occluded = false;
#endif
    if (!occluded)
    {
        const uint out_index = atomicAdd(deref(push.uses.meshlet_instances).second_count, 1);
        const uint offset = deref(push.uses.meshlet_instances).first_count;
        const uint meshlet_instance_idx = out_index + offset;
        deref(deref(push.uses.meshlet_instances).meshlets[meshlet_instance_idx]) = instanced_meshlet;
        // Scalarize atomic appends.
        [[unroll]]
        for (uint draw_list_type = 0; draw_list_type < OPAQUE_DRAW_LIST_COUNT; ++draw_list_type)
        {
            if (push.opaque_or_discard != draw_list_type) continue;
            atomicAdd(deref(push.uses.draw_commands[draw_list_type]).instance_count, 1);
            const uint draw_list_element_offset = 
                deref(push.uses.meshlet_instances).draw_lists[draw_list_type].first_count;
            const uint draw_list_element_index = draw_list_element_offset +
                atomicAdd(deref(push.uses.meshlet_instances).draw_lists[draw_list_type].second_count, 1);
            deref(deref(push.uses.meshlet_instances).draw_lists[draw_list_type].instances[draw_list_element_index]) = meshlet_instance_idx;
        }
    }
}