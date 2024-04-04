#include <daxa/daxa.inl>

#include "cull_meshes.inl"

#extension GL_EXT_debug_printf : enable

#include "shader_lib/cull_util.hlsl"

DAXA_DECL_PUSH_CONSTANT(CullMeshesPush, push)
layout(local_size_x = CULL_MESHES_WORKGROUP_X) in;
void main()
{
    uint mesh_draw_index = gl_GlobalInvocationID.x;
    uint draw_list_type = 0;
    // We do a single dispatch to cull both lists, the shader threads simply check if their id overruns the first list,
    // then assign themselves to an element in the second list.
    if (mesh_draw_index >= deref(push.uses.opaque_mesh_draw_lists).list_sizes[0])
    {
        mesh_draw_index -= deref(push.uses.opaque_mesh_draw_lists).list_sizes[0];
        draw_list_type = 1;
        // As the dispatch thread count is a multiple of CULL_MESHES_WORKGROUP_X, there will also be threads overstepping
        // the second draw list.
        if (mesh_draw_index >= deref(push.uses.opaque_mesh_draw_lists).list_sizes[1])
        {
            return;
        }
    }
    MeshDrawTuple mesh_draw = deref(deref(push.uses.opaque_mesh_draw_lists).mesh_draw_tuples[draw_list_type][mesh_draw_index]);
    GPUMesh mesh = deref(push.uses.meshes[mesh_draw.mesh_index]);
    if (mesh.meshlet_count == 0)
    {
        return;
    }
    
    const uint cull_shader_workgroup_size = TASK_SHADER_WORKGROUP_X;
    const uint cull_shader_workgroup_log2 = uint(log2(TASK_SHADER_WORKGROUP_X));
    
    write_meshlet_cull_arg_buckets(
        mesh,
        mesh_draw,
        push.uses.meshlets_cull_arg_buckets_buffers,
        draw_list_type,
        cull_shader_workgroup_size,
        cull_shader_workgroup_log2
    );
}
