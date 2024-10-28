#include <daxa/daxa.inl>

#include "cull_meshes.inl"

#include "shader_lib/cull_util.hlsl"
#include "shader_lib/po2_expansion.hlsl"

[[vk::push_constant]] ExpandMeshesToMeshletsPush push;

[numthreads(CULL_MESHES_WORKGROUP_X,1,1)]
void main(uint thread_id : SV_DispatchThreadID)
{
    uint mesh_instance_index = thread_id;
    uint mesh_instance_count = min(MAX_MESH_INSTANCES, deref(push.attach.mesh_instances).count);
    if (mesh_instance_index >= mesh_instance_count)
    {
        return;
    }
    MeshInstance mesh_instance = deref_i(deref(push.attach.mesh_instances).instances, mesh_instance_index);
    uint draw_list_type = ((mesh_instance.flags & MESH_INSTANCE_FLAG_OPAQUE) != 0) ? PREPASS_DRAW_LIST_OPAQUE : PREPASS_DRAW_LIST_MASKED;
    GPUMesh mesh = deref_i(push.attach.meshes, mesh_instance.mesh_index);
    if (mesh.meshlet_count == 0 || mesh.mesh_buffer.value == 0)
    {
        return;
    }

    // Currently only used by main visbuffer path:
    if (push.cull_meshes && push.attach.hiz.value != 0 && push.attach.globals.settings.enable_mesh_cull)
    {
        if (is_mesh_occluded(
            push.attach.globals->debug,
            push.attach.globals.camera,
            mesh_instance,
            push.attach.entity_combined_transforms,
            push.attach.meshes,
            push.attach.globals.cull_data,
            push.attach.hiz))
        {
            return;
        }
    }

    if (push.attach.vsm_clip_projections && push.cull_meshes && push.attach.hip.value != 0)
    {
        if (is_mesh_occluded_vsm(
            push.attach.vsm_clip_projections[push.cascade].camera,
            mesh_instance,
            push.attach.entity_combined_transforms,
            push.attach.meshes,
            push.attach.hip,
            push.cascade))
        {
            return;
        }
    }

    Po2WorkExpansionBufferHead * po2expansion = (Po2WorkExpansionBufferHead *)(draw_list_type == 
        PREPASS_DRAW_LIST_OPAQUE ? 
        (uint64_t)push.attach.opaque_po2expansion : 
        (uint64_t)push.attach.masked_opaque_po2expansion);

    expand_work_items(po2expansion, mesh.meshlet_count, mesh_instance_index);
}