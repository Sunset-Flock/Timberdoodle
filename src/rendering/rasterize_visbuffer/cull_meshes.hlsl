#include <daxa/daxa.inl>

#include "cull_meshes.inl"

#include "shader_lib/cull_util.hlsl"
#include "shader_lib/gpu_work_expansion.hlsl"

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
    const uint mesh_lod = 0;
    const uint mesh_index = mesh_instance.mesh_index;

    uint draw_list_type = ((mesh_instance.flags & MESH_INSTANCE_FLAG_OPAQUE) != 0) ? PREPASS_DRAW_LIST_OPAQUE : PREPASS_DRAW_LIST_MASKED;
    //if (draw_list_type == PREPASS_DRAW_LIST_MASKED) return;

    GPUMesh mesh = deref_i(push.meshes, mesh_index);
    if (mesh.meshlet_count == 0 || mesh.mesh_buffer.value == 0)
    {
        return;
    }

    // Currently only used by main visbuffer path:
    if (push.cull_meshes && push.attach.hiz.value != 0 && push.attach.globals.settings.enable_mesh_cull)
    {
        let cull_camera = push.cull_against_last_frame ? push.attach.globals.camera_prev_frame : push.attach.globals.camera;
        if (is_mesh_occluded(
            push.attach.globals->debug,
            cull_camera,
            mesh_instance,
            mesh,
            push.entity_combined_transforms,
            push.meshes,
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
            mesh,
            push.entity_combined_transforms,
            push.meshes,
            push.attach.hip,
            push.cascade))
        {
            return;
        }
    }

    let vsm = !push.attach.hip.is_empty();
    let separate_compute_meshlet_cull = (push.attach.globals.settings.enable_separate_compute_meshlet_culling) && !vsm;

    if (push.attach.globals.settings.enable_prefix_sum_work_expansion)
    {
        PrefixSumWorkExpansionBufferHead * prefixsum_expansion = (PrefixSumWorkExpansionBufferHead *)(draw_list_type == 
            PREPASS_DRAW_LIST_OPAQUE ? 
            (uint64_t)push.attach.opaque_expansion : 
            (uint64_t)push.attach.masked_expansion);
        let dst_workgroup_size_log2 = separate_compute_meshlet_cull ? uint(log2(MESHLET_CULL_WORKGROUP_X)) : uint(log2(MESH_SHADER_WORKGROUP_X));
        prefix_sum_expansion_add_workitems(prefixsum_expansion, mesh.meshlet_count, mesh_instance_index, dst_workgroup_size_log2);
    }
    else
    {
        Po2PackedWorkExpansionBufferHead * po2packed_expansion = (Po2PackedWorkExpansionBufferHead *)(draw_list_type == 
            PREPASS_DRAW_LIST_OPAQUE ? 
            (uint64_t)push.attach.opaque_expansion : 
            (uint64_t)push.attach.masked_expansion);
        let dst_workgroup_size_log2 = separate_compute_meshlet_cull ? uint(log2(MESHLET_CULL_WORKGROUP_X)) : uint(log2(MESH_SHADER_WORKGROUP_X));
        po2packed_expansion_add_workitems(po2packed_expansion, mesh.meshlet_count, mesh_instance_index, dst_workgroup_size_log2);
    }
}