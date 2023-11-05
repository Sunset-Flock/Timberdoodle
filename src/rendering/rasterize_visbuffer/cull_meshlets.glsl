#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "cull_meshlets.inl"

#include "shader_lib/cull_util.glsl"

DAXA_DECL_PUSH_CONSTANT(CullMeshletsPush,push)
layout(local_size_x = CULL_MESHLETS_WORKGROUP_X) in;
void main()
{
    if (gl_GlobalInvocationID.x == 0)
    {
        deref(u_draw_command).vertex_count = 3 * MAX_TRIANGLES_PER_MESHLET;
    }
    MeshletInstance instanced_meshlet;
    const bool valid_meshlet = get_meshlet_instance_from_arg(gl_GlobalInvocationID.x, push.indirect_args_table_id, u_meshlet_cull_indirect_args, instanced_meshlet);
    if (!valid_meshlet)
    {
        return;
    }
#if ENABLE_MESHLET_CULLING
    if (is_meshlet_occluded(
        instanced_meshlet,
        u_entity_meshlet_visibility_bitfield_offsets,
        u_entity_meshlet_visibility_bitfield_arena,
        u_entity_combined_transforms,
        u_meshes,
        u_hiz))
    {
        return;
    }
#endif
    const uint out_index = atomicAdd(deref(u_instantiated_meshlets).second_count, 1);
    const uint offset = deref(u_instantiated_meshlets).first_count;
    deref(u_instantiated_meshlets).meshlets[out_index + offset] = pack_meshlet_instance(instanced_meshlet);
    atomicAdd(deref(u_draw_command).instance_count, 1);
}