#include <daxa/daxa.inl>

#include "cull_meshes.inl"

#include "shader_lib/cull_util.hlsl"
#include "shader_lib/po2_expansion.hlsl"

[[vk::push_constant]] ExpandMeshesToMeshletsPush push;

[numthreads(CULL_MESHES_WORKGROUP_X,1,1)]
void main(uint thread_id : SV_DispatchThreadID)
{
    uint mesh_instance_index = thread_id;
    uint mesh_instance_count = deref(push.uses.mesh_instances).count;
    if (mesh_instance_index >= mesh_instance_count)
    {
        return;
    }
    MeshInstance mesh_instance = deref_i(deref(push.uses.mesh_instances).instances, mesh_instance_index);
    uint draw_list_type = ((mesh_instance.flags & MESH_INSTANCE_FLAG_OPAQUE) != 0) ? DRAW_LIST_OPAQUE : DRAW_LIST_MASKED;
    GPUMesh mesh = deref_i(push.uses.meshes, mesh_instance.mesh_index);
    if (mesh.meshlet_count == 0)
    {
        return;
    }

    // Currently only used by main visbuffer path:
    if (push.cull_meshes && push.uses.hiz.value != 0 && push.uses.globals.settings.enable_mesh_cull)
    {
        if (is_mesh_occluded(
            push.uses.globals->debug,
            push.uses.globals.camera,
            mesh_instance,
            push.uses.entity_combined_transforms,
            push.uses.meshes,
            push.uses.globals.settings.next_lower_po2_render_target_size,
            push.uses.hiz))
        {
            return;
        }
    }

    if (push.uses.vsm_clip_projections && push.cull_meshes && push.uses.hip.value != 0)
    {
        if (is_mesh_occluded_vsm(
            push.uses.vsm_clip_projections[push.cascade].camera,
            mesh_instance,
            push.uses.entity_combined_transforms,
            push.uses.meshes,
            push.uses.hip,
            push.cascade))
        {
            return;
        }
    }

    Po2WorkExpansionBufferHead * po2expansion = (Po2WorkExpansionBufferHead *)(draw_list_type == 
        DRAW_LIST_OPAQUE ? 
        (uint64_t)push.uses.opaque_po2expansion : 
        (uint64_t)push.uses.masked_opaque_po2expansion);

    expand_work_items(po2expansion, mesh.meshlet_count, mesh_instance_index);
}