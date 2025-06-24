#include <daxa/daxa.inl>

#include "cull_meshes.inl"

#include "shader_lib/cull_util.hlsl"
#include "shader_lib/gpu_work_expansion.hlsl"
#include "shader_lib/vsm_util.glsl"

[[vk::push_constant]] ExpandMeshesToMeshletsPush push;

#define AT deref(push.attachments).attachments

[numthreads(CULL_MESHES_WORKGROUP_X,1,1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint mesh_instance_index = thread_id.x;
    uint source_mesh_instance_index = mesh_instance_index;

    uint mesh_instance_count = min(MAX_MESH_INSTANCES, deref(AT.mesh_instances).count);
    if (mesh_instance_index >= mesh_instance_count)
    {
        return;
    }
    MeshInstance mesh_instance = deref_i(deref(AT.mesh_instances).instances, mesh_instance_index);
    const uint mesh_index = mesh_instance.mesh_index;

    uint draw_list_type = ((mesh_instance.flags & MESH_INSTANCE_FLAG_OPAQUE) != 0) ? PREPASS_DRAW_LIST_OPAQUE : PREPASS_DRAW_LIST_MASKED;
    //if (draw_list_type == PREPASS_DRAW_LIST_MASKED) return;

    GPUMesh mesh = deref_i(push.meshes, mesh_index);
    if (mesh.meshlet_count == 0 || mesh.mesh_buffer.value == 0)
    {
        return;
    }

    // Currently only used by main visbuffer path:
    if (push.cull_meshes && AT.hiz.value != 0 && AT.globals.settings.enable_mesh_cull)
    {
        let cull_camera = push.cull_against_last_frame ? AT.globals.main_camera_prev_frame : AT.globals.main_camera;
        if (is_mesh_occluded(
            AT.globals->debug,
            cull_camera,
            mesh_instance,
            mesh,
            push.entity_combined_transforms,
            push.meshes,
            AT.globals.cull_data,
            AT.hiz))
        {
            return;
        }
    }

    if (AT.vsm_clip_projections && push.cull_meshes && AT.hip.value != 0)
    {
        let cascade = thread_id.y;
        const VSMDirectionalIndirections indirections = VSMDirectionalIndirections(
            cascade,                // cascade
            mesh_instance_index     // mesh_instance_index
        );
        source_mesh_instance_index = pack_vsm_directional_light_indirections(indirections);
        if (is_mesh_occluded_vsm(
            AT.vsm_clip_projections[cascade].camera,
            mesh_instance,
            mesh,
            push.entity_combined_transforms,
            push.meshes,
            AT.hip,
            cascade))
        {
            return;
        }
    }

    if (AT.vsm_point_lights && (push.mip_level != -1) && push.cull_meshes)
    {
        let point_light_prefix = AT.globals.vsm_settings.point_light_count * 6;

        VSMPointSpotIndirections indirections = VSMPointSpotIndirections(
            push.mip_level,      // mip_level
            -1,                  // FILLED OUT LATER
            mesh_instance_index  // mesh_instance_index
        );

        const float2 base_resolution = VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION / (1 << indirections.mip_level);
        CameraInfo * camera_info;
        float cutoff = 0.0f;
        uint light_idx = 0;
        // Point light
        if (thread_id.y < point_light_prefix)
        {
            let point_light_index  = thread_id.y / 6;
            let face_index = thread_id.y - (point_light_index * 6);

            indirections.array_layer_index = thread_id.y;
            camera_info = &(AT.vsm_point_lights[point_light_index].face_cameras[face_index]);
            cutoff = AT.vsm_point_lights[point_light_index].light.cutoff;
            light_idx = thread_id.y;
        }
        // Spot light
        else
        {
            let spot_light_index = thread_id.y - point_light_prefix;

            camera_info = &(AT.vsm_spot_lights[spot_light_index].camera);
            cutoff = AT.vsm_spot_lights[spot_light_index].light.cutoff;
            light_idx = VSM_SPOT_LIGHT_OFFSET + spot_light_index;
            indirections.array_layer_index = light_idx;
        }

        if(is_mesh_occluded_point_spot_vsm(
            *camera_info,
            mesh_instance,
            mesh,
            cutoff,
            push.entity_combined_transforms,
            push.meshes,
            AT.point_hip,
            light_idx,
            base_resolution,
            AT.globals))
        {
            return;
        }
        source_mesh_instance_index = pack_vsm_point_spot_light_indirections(indirections);
    }

    let vsm = !AT.hip.is_empty() || (push.mip_level != -1);
    let separate_compute_meshlet_cull = (AT.globals.settings.enable_separate_compute_meshlet_culling) && !vsm;

    if (AT.first_pass_meshlet_bitfield != (FirstPassMeshletBitfield*)0 && source_mesh_instance_index < FIRST_PASS_MESHLET_BITFIELD_U32_OFFSETS_SIZE)
    {
        uint allocated_offset = 0;
        uint meshlet_u32_bitfield_size = round_up_div(mesh.meshlet_count, 32);
        InterlockedAdd(AT.globals.readback.first_pass_meshlet_bitfield_requested_dynamic_size, meshlet_u32_bitfield_size, allocated_offset);
        if ((allocated_offset + meshlet_u32_bitfield_size) < FIRST_PASS_MESHLET_BITFIELD_U32_BITFIELD_SIZE)
        {
            AT.first_pass_meshlet_bitfield.meshlet_instance_offsets[source_mesh_instance_index] = allocated_offset + 1;
        }
    }

    if (AT.globals.settings.enable_prefix_sum_work_expansion)
    {
        PrefixSumWorkExpansionBufferHead * prefixsum_expansion = (PrefixSumWorkExpansionBufferHead *)(draw_list_type == 
            PREPASS_DRAW_LIST_OPAQUE ? 
            (uint64_t)AT.opaque_expansion : 
            (uint64_t)AT.masked_expansion);
        let dst_workgroup_size_log2 = separate_compute_meshlet_cull ? uint(log2(MESHLET_CULL_WORKGROUP_X)) : uint(log2(MESH_SHADER_WORKGROUP_X));
        prefix_sum_expansion_add_workitems(prefixsum_expansion, mesh.meshlet_count, source_mesh_instance_index, dst_workgroup_size_log2);
    }
    else
    {
        Po2PackedWorkExpansionBufferHead * po2packed_expansion = (Po2PackedWorkExpansionBufferHead *)(draw_list_type == 
            PREPASS_DRAW_LIST_OPAQUE ? 
            (uint64_t)AT.opaque_expansion : 
            (uint64_t)AT.masked_expansion);
        let dst_workgroup_size_log2 = separate_compute_meshlet_cull ? uint(log2(MESHLET_CULL_WORKGROUP_X)) : uint(log2(MESH_SHADER_WORKGROUP_X));
        po2packed_expansion_add_workitems(po2packed_expansion, mesh.meshlet_count, source_mesh_instance_index, dst_workgroup_size_log2);
    }
}