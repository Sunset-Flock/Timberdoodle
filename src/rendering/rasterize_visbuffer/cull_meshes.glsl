#include <daxa/daxa.inl>

#include "cull_meshes.inl"

#extension GL_EXT_debug_printf : enable

#include "shader_lib/cull_util.glsl"

#define DEBUG_MESH_CULL 0
#define DEBUG_MESH_CULL1 0

DAXA_DECL_PUSH_CONSTANT(CullMeshesPush, push)
layout(local_size_x = CULL_MESHES_WORKGROUP_X) in;
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
    }
    else
    {
    }
    // As the dispatch thread count is a multiple of CULL_MESHES_WORKGROUP_X, there will also be threads overstepping
    // the second draw list.
    if (mesh_draw_index >= deref(push.uses.opaque_mesh_draw_lists).list_sizes[1])
    {
        return;
    }

    MeshDrawTuple mesh_draw = deref(deref(push.uses.opaque_mesh_draw_lists).mesh_draw_tuples[opaque_draw_list_index][mesh_draw_index]);
    GPUMesh mesh = deref(push.uses.meshes[mesh_draw.mesh_index]);
    if (mesh.meshlet_count == 0)
    {
        return;
    }

    GPUMaterial material = deref(push.uses.materials[mesh.material_index]);
    daxa_RWBufferPtr(MeshletCullArgBucketsBufferHead) cull_buckets = 
        opaque_draw_list_index == 0 ? 
        push.uses.meshlet_cull_arg_buckets_opaque :
        push.uses.meshlet_cull_arg_buckets_discard;
    const uint cull_shader_workgroup_size = 
        (deref(push.uses.globals).settings.enable_mesh_shader == 1) ? 
        TASK_SHADER_WORKGROUP_X : 
        CULL_MESHLETS_WORKGROUP_X;
    const uint cull_shader_workgroup_log2 = 
        (deref(push.uses.globals).settings.enable_mesh_shader == 1) ? 
        uint(log2(TASK_SHADER_WORKGROUP_X)) : 
        uint(log2(CULL_MESHLETS_WORKGROUP_X));
    
    write_meshlet_cull_arg_buckets(
        mesh,
        mesh_draw,
        cull_buckets,
        cull_shader_workgroup_size,
        cull_shader_workgroup_log2
    );
}
