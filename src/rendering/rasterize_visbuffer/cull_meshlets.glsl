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
    if (gl_GlobalInvocationID.x == 0)
    {
        deref(push.uses.draw_command).vertex_count = 3 * MAX_TRIANGLES_PER_MESHLET;
    }
    MeshletInstance instanced_meshlet;
    const bool valid_meshlet = get_meshlet_instance_from_arg(gl_GlobalInvocationID.x, push.indirect_args_table_id, push.uses.meshlet_cull_indirect_args, instanced_meshlet);
    if (!valid_meshlet)
    {
        return;
    }
#if ENABLE_MESHLET_CULLING
    const bool occluded = is_meshlet_occluded(
        instanced_meshlet,
        push.uses.entity_meshlet_visibility_bitfield_offsets,
        push.uses.entity_meshlet_visibility_bitfield_arena,
        push.uses.entity_combined_transforms,
        push.uses.meshes,
        push.uses.hiz);
        
#else
    const bool occluded = false;
#endif
    if (!occluded)
    {
        const uint out_index = atomicAdd(deref(push.uses.instantiated_meshlets).second_count, 1);
        const uint offset = deref(push.uses.instantiated_meshlets).first_count;
        deref(push.uses.instantiated_meshlets).meshlets[out_index + offset] = pack_meshlet_instance(instanced_meshlet);
        atomicAdd(deref(push.uses.draw_command).instance_count, 1);
    }
}