#include "daxa/daxa.inl"

#include "cull_and_draw_pages_shared.hlsl"

#include "vsm.inl"
#include "shader_lib/cull_util.hlsl"
#include "shader_lib/vsm_util.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/glsl_to_slang.glsl"
#include "../rasterize_visbuffer/draw_visbuffer.hlsl"

[[vk::push_constant]] CullAndDrawPointPagesPush point_vsm_push;

uint64_t get_expansion_buffer()
{
    uint64_t* expansion_array = &point_vsm_push.attachments.po2expansion_mip0;
    return expansion_array[point_vsm_push.draw_list_type + 2 * point_vsm_push.mip_level];
}

[shader("amplification")]
[numthreads(32, 1, 1)]
func point_vsm_entry_task(
    uint3 svtid : SV_DispatchThreadID,
    uint3 svgid : SV_GroupId
)
{
    let push = point_vsm_push;
    const uint64_t expansion = get_expansion_buffer();

    MeshletInstance instanced_meshlet;
    VSMPointSpotIndirections point_spot_indirections;

    bool valid_meshlet = get_vsm_point_spot_meshlet_instance_from_work_item(
        push.attachments.globals.settings.enable_prefix_sum_work_expansion,
        expansion,
        push.attachments.mesh_instances,
        push.attachments.meshes,
        svtid.x,
        instanced_meshlet,
        point_spot_indirections,
    );

    // We still continue to run the task shader even with invalid meshlets.
    // We simple set the occluded value to true for these invalida meshlets.
    // This is done so that the following WaveOps are well formed and have all threads active. 
    if (valid_meshlet)
    {

        CameraInfo * camera_info;
        float cutoff;
        // Point light
        if (point_spot_indirections.array_layer_index < VSM_SPOT_LIGHT_OFFSET)
        {
            let point_light_index = point_spot_indirections.array_layer_index / 6;
            let face_index = point_spot_indirections.array_layer_index - (point_light_index * 6);

            camera_info = &(push.attachments.vsm_point_lights[point_light_index].face_cameras[face_index]);
            cutoff = push.attachments.vsm_point_lights[point_light_index].light.cutoff;
        }
        // Spot light
        else
        {
            let spot_light_index = point_spot_indirections.array_layer_index - VSM_SPOT_LIGHT_OFFSET;

            camera_info = &(push.attachments.vsm_spot_lights[spot_light_index].camera);
            cutoff = push.attachments.vsm_spot_lights[spot_light_index].light.cutoff;
        }

        const float2 base_resolution = VSM_PAGE_TABLE_RESOLUTION / (1 << point_spot_indirections.mip_level);
        valid_meshlet = valid_meshlet && !is_meshlet_occluded_point_spot_vsm(
            *camera_info,
            instanced_meshlet,
            push.attachments.entity_combined_transforms,
            push.attachments.meshes,
            cutoff,
            push.hpb_view,
            point_spot_indirections.array_layer_index,
            base_resolution
        );
    }

    CullMeshletsDrawPagesPayload payload;
    payload.task_shader_wg_meshlet_args_offset = svgid.x * MESH_SHADER_WORKGROUP_X;
    payload.task_shader_surviving_meshlets_mask = WaveActiveBallot(valid_meshlet).x;
    let surviving_meshlet_count = WaveActiveSum(valid_meshlet ? 1u : 0u);
    // When not occluded, this value determines the new packed index for each thread in the wave:
    let local_survivor_index = WavePrefixSum(valid_meshlet ? 1u : 0u);

    bool enable_backface_culling = false;
    if (valid_meshlet)
    {
        if (instanced_meshlet.material_index != INVALID_MANIFEST_INDEX)
        {
            GPUMaterial material = push.attachments.material_manifest[instanced_meshlet.material_index];
            enable_backface_culling = !material.alpha_discard_enabled && !material.double_sided_enabled;
        }
    }
    payload.enable_backface_culling = WaveActiveBallot(enable_backface_culling).x;

    DispatchMesh(1, surviving_meshlet_count, 1, payload);
}

func point_vsm_mesh_cull_draw<V: MeshShaderVertexT, P: VSMMeshShaderPrimitiveT>(
    in uint3 svtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in CullMeshletsDrawPagesPayload payload)
{
    let push = point_vsm_push;

    // The payloads packed survivor indices go from 0-survivor_count.
    // These indices map to the meshlet instance indices.
    let local_meshlet_instance_index = svtid.y;

    // Meshlet instance indices are the task allocated offset into the meshlet instances + the packed survivor index.

    // We need to know the thread index of the task shader that ran for this meshlet.
    // With its thread id we can read the argument buffer just like the task shader did.
    // From the argument we construct the meshlet and any other data that we need.
    let task_shader_local_index = wave32_find_nth_set_bit(payload.task_shader_surviving_meshlets_mask, local_meshlet_instance_index);
    let meshlet_cull_arg_index = payload.task_shader_wg_meshlet_args_offset + task_shader_local_index;
    let task_shader_local_bit = (1u << task_shader_local_index);

    const uint64_t expansion = get_expansion_buffer();

    MeshletInstance instanced_meshlet;
    VSMPointSpotIndirections indirections;
    let valid_meshlet = get_vsm_point_spot_meshlet_instance_from_work_item(
        push.attachments.globals.settings.enable_prefix_sum_work_expansion,
        expansion,
        push.attachments.mesh_instances,
        push.attachments.meshes,
        meshlet_cull_arg_index,
        instanced_meshlet,
        indirections,
    );
    
    let cull_backfaces = (payload.enable_backface_culling & task_shader_local_bit) != 0;
    CameraInfo camera;
    if (indirections.array_layer_index < VSM_SPOT_LIGHT_OFFSET)
    {
        let point_light_index = indirections.array_layer_index / 6;
        let face_index = indirections.array_layer_index - (point_light_index * 6);
        camera = push.attachments.vsm_point_lights[point_light_index].face_cameras[face_index];
    }
    else
    {
        let spot_light_index = indirections.array_layer_index - VSM_SPOT_LIGHT_OFFSET;
        camera = push.attachments.vsm_spot_lights[spot_light_index].camera;
    }

    generic_vsm_mesh(
        svtid,
        out_indices,
        out_vertices,
        out_primitives,
        push.attachments.entity_combined_transforms,
        push.attachments.meshes,
        pack_vsm_point_spot_light_indirections(indirections),
        cull_backfaces,
        camera,
        instanced_meshlet,
        push.hpb_view
    );
}

[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func point_vsm_entry_mesh_opaque(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderOpaqueVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<PointVSMOpaqueMeshShaderPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in payload CullMeshletsDrawPagesPayload payload)
{
    point_vsm_mesh_cull_draw(svtid, out_indices, out_vertices, out_primitives, payload);
}

[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func point_vsm_entry_mesh_masked(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderMaskVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<PointVSMMaskMeshShaderPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in payload CullMeshletsDrawPagesPayload payload)
{
    point_vsm_mesh_cull_draw(svtid, out_indices, out_vertices, out_primitives, payload);
}

[shader("fragment")]
void point_vsm_entry_fragment_opaque(
    in MeshShaderOpaqueVertex vert,
    in PointVSMOpaqueMeshShaderPrimitive prim)
{
    let push = point_vsm_push;

    let page_coords = uint2(vert.position.xy) / VSM_PAGE_SIZE;
    let indirections = unpack_vsm_point_spot_light_indirections(prim.vsm_meta_info);

    let vsm_page_entry = push.attachments.vsm_point_spot_page_table[indirections.mip_level].get()[int3(page_coords.xy, indirections.array_layer_index)];
    if(get_is_allocated(vsm_page_entry) && get_is_dirty(vsm_page_entry))
    {
        let memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);

        const uint2 in_page_texel_coord = uint2(vert.position.xy) % VSM_PAGE_SIZE;
        const uint2 in_memory_offset = memory_page_coords * VSM_PAGE_SIZE;
        const uint2 memory_texel_coord = in_memory_offset + in_page_texel_coord;

        InterlockedMax(
            push.daxa_uint_vsm_memory_view.get_formatted()[memory_texel_coord],
            asuint(clamp(vert.position.z, 0.0f, 1.0f))
        );
    }
}

[shader("fragment")]
void point_vsm_entry_fragment_masked(
    in MeshShaderMaskVertex vert,
    in PointVSMMaskMeshShaderPrimitive prim)
{

    let push = point_vsm_push;

    let page_coords = uint2(vert.position.xy) / VSM_PAGE_SIZE;
    let indirections = unpack_vsm_point_spot_light_indirections(prim.vsm_meta_info);

    let vsm_page_entry = push.attachments.vsm_point_spot_page_table[indirections.mip_level].get()[int3(page_coords.xy, indirections.array_layer_index)];
    if(get_is_allocated(vsm_page_entry) && get_is_dirty(vsm_page_entry))
    {
        let memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);

        const uint2 in_page_texel_coord = uint2(vert.position.xy) % VSM_PAGE_SIZE;
        const uint2 in_memory_offset = memory_page_coords * VSM_PAGE_SIZE;
        const uint2 memory_texel_coord = in_memory_offset + in_page_texel_coord;

        if(prim.material_index != INVALID_MANIFEST_INDEX)
        {
            const GPUMaterial material = deref_i(push.attachments.material_manifest, prim.material_index);
            float alpha = 1.0;
            if(material.opacity_texture_id.value != 0 && material.alpha_discard_enabled)
            {
                alpha = Texture2D<float>::get(material.opacity_texture_id)
                    .SampleLevel(SamplerState::get(push.attachments.globals->samplers.linear_repeat), vert.uv, 2).r;
            }
            else if(material.diffuse_texture_id.value != 0 && material.alpha_discard_enabled)
            {
                alpha = Texture2D<float4>::get(material.diffuse_texture_id)
                    .SampleLevel(SamplerState::get(push.attachments.globals->samplers.linear_repeat), vert.uv, 2).a;
            }
            if(alpha < 0.5) { discard; }
        }
        InterlockedMax(
            push.daxa_uint_vsm_memory_view.get_formatted()[memory_texel_coord],
            asuint(clamp(vert.position.z, 0.0f, 1.0f))
        );
    }
}