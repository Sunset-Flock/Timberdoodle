#include "daxa/daxa.inl"

#include "cull_and_draw_pages_shared.hlsl"

#include "vsm.inl"
#include "shader_lib/cull_util.hlsl"
#include "shader_lib/vsm_util.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/glsl_to_slang.glsl"
#include "../rasterize_visbuffer/draw_visbuffer.hlsl"

[[vk::push_constant]] CullAndDrawPagesPush vsm_push;

uint64_t get_expansion_buffer()
{
    uint64_t* expansion_array = &vsm_push.attachments.po2expansion;
    return expansion_array[vsm_push.draw_list_type];
}

[shader("amplification")]
[numthreads(32, 1, 1)]
func directional_vsm_entry_task(
    uint3 svtid : SV_DispatchThreadID,
    uint3 svgid : SV_GroupID
)
{
    let push = vsm_push;

    uint64_t expansion = get_expansion_buffer();
    MeshletInstance instanced_meshlet;
    VSMDirectionalIndirections indirections;
    let valid_meshlet = get_vsm_directional_meshlet_instance_from_work_item(
        push.attachments.globals.settings.enable_prefix_sum_work_expansion,
        expansion,
        push.attachments.mesh_instances,
        push.attachments.meshes,
        svtid.x,
        instanced_meshlet,
        indirections,
    );
    
    bool draw_meshlet = valid_meshlet;
    // We still continue to run the task shader even with invalid meshlets.
    // We simple set the occluded value to true for these invalida meshlets.
    // This is done so that the following WaveOps are well formed and have all threads active. 
    if (valid_meshlet)
    {
        draw_meshlet = draw_meshlet && !is_meshlet_occluded_vsm(
            deref_i(push.attachments.vsm_clip_projections, indirections.cascade).camera,
            instanced_meshlet,
            push.attachments.entity_combined_transforms,
            push.attachments.meshes,
            push.attachments.vsm_dirty_bit_hiz,
            indirections.cascade
        );
    }

    CullMeshletsDrawPagesPayload payload;
    payload.task_shader_wg_meshlet_args_offset = svgid.x * MESH_SHADER_WORKGROUP_X;
    payload.task_shader_surviving_meshlets_mask = WaveActiveBallot(draw_meshlet).x;
    let surviving_meshlet_count = WaveActiveSum(draw_meshlet ? 1u : 0u);
    // When not occluded, this value determines the new packed index for each thread in the wave:
    let local_survivor_index = WavePrefixSum(draw_meshlet ? 1u : 0u);

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


func directional_vsm_mesh_cull_draw<V: MeshShaderVertexT, P: VSMMeshShaderPrimitiveT>(
    in uint3 svtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in CullMeshletsDrawPagesPayload payload)
{
    let push = vsm_push;
    let group_index = svtid.y;

    // The payloads packed survivor indices go from 0-survivor_count.
    // These indices map to the meshlet instance indices.
    let local_meshlet_instance_index = group_index;
    // Meshlet instance indices are the task allocated offset into the meshlet instances + the packed survivor index.

    // We need to know the thread index of the task shader that ran for this meshlet.
    // With its thread id we can read the argument buffer just like the task shader did.
    // From the argument we construct the meshlet and any other data that we need.
    let task_shader_local_index = wave32_find_nth_set_bit(payload.task_shader_surviving_meshlets_mask, group_index);
    let meshlet_cull_arg_index = payload.task_shader_wg_meshlet_args_offset + task_shader_local_index;
    let task_shader_local_bit = (1u << task_shader_local_index);

    uint64_t expansion = get_expansion_buffer();
    MeshletInstance meshlet_inst;
    VSMDirectionalIndirections indirections;
    let valid_meshlet = get_vsm_directional_meshlet_instance_from_work_item(
        push.attachments.globals.settings.enable_prefix_sum_work_expansion,
        expansion,
        push.attachments.mesh_instances,
        push.attachments.meshes,
        meshlet_cull_arg_index,
        meshlet_inst,
        indirections,
    );
    
    let cull_backfaces = (payload.enable_backface_culling & task_shader_local_bit) != 0;
    generic_vsm_mesh(
        svtid,
        out_indices,
        out_vertices,
        out_primitives,
        push.attachments.entity_combined_transforms,
        push.attachments.meshes,
        pack_vsm_directional_light_indirections(indirections),
        cull_backfaces,
        push.attachments.vsm_clip_projections[indirections.cascade].camera,
        meshlet_inst,
        push.attachments.vsm_dirty_bit_hiz
    );
}

[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func directional_vsm_entry_mesh_opaque(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderOpaqueVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<DirectionalVSMOpaqueMeshShaderPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in payload CullMeshletsDrawPagesPayload payload)
{
    directional_vsm_mesh_cull_draw(svtid, out_indices, out_vertices, out_primitives, payload);
}

[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func directional_vsm_entry_mesh_masked(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderMaskVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<DirectionalVSMMaskMeshShaderPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in payload CullMeshletsDrawPagesPayload payload)
{
    directional_vsm_mesh_cull_draw(svtid, out_indices, out_vertices, out_primitives, payload);
}

[shader("fragment")]
void directional_vsm_entry_fragment_opaque(
    in MeshShaderOpaqueVertex vert,
    in DirectionalVSMOpaqueMeshShaderPrimitive prim)
{
    let push = vsm_push;
    const float2 virtual_uv = vert.position.xy / VSM_TEXTURE_RESOLUTION;
    let indirections = unpack_vsm_directional_light_indirections(prim.vsm_meta_info);

    let wrapped_coords = vsm_clip_info_to_wrapped_coords(
        {indirections.cascade, virtual_uv},
        push.attachments.vsm_clip_projections);

    let vsm_page_entry = RWTexture2DArray<uint>::get(push.attachments.vsm_page_table)[uint3(wrapped_coords)].x;
    if(get_is_allocated(vsm_page_entry) && get_is_dirty(vsm_page_entry))
    {
        let memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
        let physical_texel_coords = virtual_uv_to_physical_texel(virtual_uv, memory_page_coords);

        InterlockedMin(
            push.daxa_uint_vsm_memory_view.get_formatted()[physical_texel_coords],
            asuint(clamp(vert.position.z / vert.position.w, 0.0f, 1.0f))
        );
        if (push.attachments.vsm_overdraw_debug.index() != 0)
        {
            InterlockedAdd(RWTexture2D<uint>::get_formatted(push.attachments.vsm_overdraw_debug)[physical_texel_coords], 1);
        }
    }
}

[shader("fragment")]
void directional_vsm_entry_fragment_masked(
    in MeshShaderMaskVertex vert,
    in DirectionalVSMMaskMeshShaderPrimitive prim)
{
    let push = vsm_push;
    const float2 virtual_uv = vert.position.xy / VSM_TEXTURE_RESOLUTION;
    let indirections = unpack_vsm_directional_light_indirections(prim.vsm_meta_info);

    let wrapped_coords = vsm_clip_info_to_wrapped_coords(
        {indirections.cascade, virtual_uv},
        push.attachments.vsm_clip_projections);

    let vsm_page_entry = RWTexture2DArray<uint>::get(push.attachments.vsm_page_table)[uint3(wrapped_coords)].x;
    if(get_is_allocated(vsm_page_entry) && get_is_dirty(vsm_page_entry))
    {
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

        let memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
        let physical_texel_coords = virtual_uv_to_physical_texel(virtual_uv, memory_page_coords);
        InterlockedMin(
            push.daxa_uint_vsm_memory_view.get_formatted()[physical_texel_coords],
            asuint(clamp(vert.position.z / vert.position.w, 0.0f, 1.0f))
        );
        if (push.attachments.vsm_overdraw_debug.index() != 0)
        {
            InterlockedAdd(RWTexture2D<uint>::get_formatted(push.attachments.vsm_overdraw_debug)[physical_texel_coords], 1);
        }
    }
}