#include "daxa/daxa.inl"

#include "../rasterize_visbuffer/draw_visbuffer.hlsl"

struct CullMeshletsDrawPagesPayload
{
    uint task_shader_wg_meshlet_args_offset;
    uint task_shader_surviving_meshlets_mask;
    uint enable_backface_culling;
};

interface VSMMeshShaderPrimitiveT
{
    DECL_GET_SET(uint, vsm_meta_info)
    DECL_GET_SET(bool, cull_primitive)
}

struct DirectionalVSMOpaqueMeshShaderPrimitive : VSMMeshShaderPrimitiveT
{
    bool cull_primitive : SV_CullPrimitive;
    IMPL_GET_SET(bool, cull_primitive)
    nointerpolation [[vk::location(0)]] uint vsm_meta_info;
    IMPL_GET_SET(uint, vsm_meta_info)
};

struct DirectionalVSMMaskMeshShaderPrimitive : VSMMeshShaderPrimitiveT
{
    bool cull_primitive : SV_CullPrimitive;
    IMPL_GET_SET(bool, cull_primitive)
    nointerpolation [[vk::location(0)]] uint vsm_meta_info;
    IMPL_GET_SET(uint, vsm_meta_info)
    nointerpolation [[vk::location(1)]] uint material_index;
    IMPL_GET_SET(uint, material_index)
}

struct PointVSMOpaqueMeshShaderPrimitive : VSMMeshShaderPrimitiveT
{
    bool cull_primitive : SV_CullPrimitive;
    IMPL_GET_SET(bool, cull_primitive)
    nointerpolation [[vk::location(0)]] uint vsm_meta_info;
    IMPL_GET_SET(uint, vsm_meta_info)
};

struct PointVSMMaskMeshShaderPrimitive : VSMMeshShaderPrimitiveT
{
    bool cull_primitive : SV_CullPrimitive;
    IMPL_GET_SET(bool, cull_primitive)
    nointerpolation [[vk::location(0)]] uint vsm_meta_info;
    IMPL_GET_SET(uint, vsm_meta_info)
    nointerpolation [[vk::location(1)]] uint material_index;
    IMPL_GET_SET(uint, material_index)
}

func generic_vsm_mesh<V: MeshShaderVertexT, P: VSMMeshShaderPrimitiveT>(
    in uint3 svtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    daxa_BufferPtr(daxa_f32mat4x3) combined_entity_transforms,
    daxa_BufferPtr(GPUMesh) meshes,
    const uint vsm_meta_info,
    const bool cull_backfaces,
    const CameraInfo camera,
    const MeshletInstance meshlet_inst,
    const daxa_ImageViewId hiz)
{    
    uint2 render_target_size = uint2(VSM_TEXTURE_RESOLUTION,VSM_TEXTURE_RESOLUTION);
    const GPUMesh mesh = meshes[meshlet_inst.mesh_index];
    const Meshlet meshlet = mesh.meshlets[meshlet_inst.meshlet_index];
    daxa_BufferPtr(daxa_u32) micro_index_buffer = mesh.micro_indices;

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

    const daxa_f32mat4x3 model_mat4x3 = combined_entity_transforms[meshlet_inst.entity_index];
    const daxa_f32mat4x4 model_mat = mat_4x3_to_4x4(model_mat4x3);
    const bool is_directional_vsm = (P is DirectionalVSMMaskMeshShaderPrimitive) || (P is DirectionalVSMOpaqueMeshShaderPrimitive);

    for (uint vertex_offset = 0; vertex_offset < meshlet.vertex_count; vertex_offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint in_meshlet_vertex_index = svtid.x + vertex_offset;
        if (in_meshlet_vertex_index >= meshlet.vertex_count) break;

        const uint in_mesh_vertex_index = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + in_meshlet_vertex_index);
        if (in_mesh_vertex_index >= mesh.vertex_count)
        {
            /// TODO: ASSERT HERE. 
            continue;
        }
        const daxa_f32vec4 vertex_position = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index), 1);
        const daxa_f32vec4 pos = mul(camera.view_proj, mul(model_mat, vertex_position));

        V vertex;
        vertex.set_position(pos);
        if (V is MeshShaderMaskVertex)
        {
            var mvertex = reinterpret<MeshShaderMaskVertex>(vertex);
            mvertex.uv = float2(0,0);
            if (as_address(mesh.vertex_uvs) != 0)
            {
                mvertex.uv = deref_i(mesh.vertex_uvs, in_mesh_vertex_index);
            }
            vertex = reinterpret<V>(mvertex);
        }
        out_vertices[in_meshlet_vertex_index] = vertex;
    }

    for (uint triangle_offset = 0; triangle_offset < meshlet.triangle_count; triangle_offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint in_meshlet_triangle_index = svtid.x + triangle_offset;
        if (in_meshlet_triangle_index >= meshlet.triangle_count) break;

        const uint3 tri_in_meshlet_vertex_indices = uint3(
            get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 0),
            get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 1),
            get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 2));
        
        out_indices[in_meshlet_triangle_index] = tri_in_meshlet_vertex_indices;

        const uint in_meshlet_vertex_index_0 = tri_in_meshlet_vertex_indices.x;
        const uint in_mesh_vertex_index_0 = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + in_meshlet_vertex_index_0);
        let vert_0_world_pos = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index_0), 1);
        let vert_0_ndc_pos = mul(camera.view_proj, mul(model_mat, vert_0_world_pos));

        const uint in_meshlet_vertex_index_1 = tri_in_meshlet_vertex_indices.y;
        const uint in_mesh_vertex_index_1 = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + in_meshlet_vertex_index_1);
        let vert_1_world_pos = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index_1), 1);
        let vert_1_ndc_pos = mul(camera.view_proj, mul(model_mat, vert_1_world_pos));

        const uint in_meshlet_vertex_index_2 = tri_in_meshlet_vertex_indices.z;
        const uint in_mesh_vertex_index_2 = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + in_meshlet_vertex_index_2);
        let vert_2_world_pos = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index_2), 1);
        let vert_2_ndc_pos = mul(camera.view_proj, mul(model_mat, vert_2_world_pos));

        const bool max_behind_near_plane = 
            (vert_0_ndc_pos.z > vert_0_ndc_pos.w) ||
            (vert_1_ndc_pos.z > vert_1_ndc_pos.w) ||
            (vert_2_ndc_pos.z > vert_2_ndc_pos.w);
        let vert_0_clip_pos = vert_0_ndc_pos.xyz / vert_0_ndc_pos.w;
        let vert_1_clip_pos = vert_1_ndc_pos.xyz / vert_1_ndc_pos.w;
        let vert_2_clip_pos = vert_2_ndc_pos.xyz / vert_2_ndc_pos.w;
        NdcAABB tri_aabb = {
            min(vert_0_clip_pos.xyz, min(vert_1_clip_pos.xyz, vert_2_clip_pos.xyz)),
            max(vert_0_clip_pos.xyz, max(vert_1_clip_pos.xyz, vert_2_clip_pos.xyz))
        };

        float4 tri_vert_clip_positions[3] = {vert_0_ndc_pos, vert_1_ndc_pos, vert_2_ndc_pos};
        bool cull_primitive = false;
        #if 1
        cull_primitive = cull_backfaces ? is_triangle_backfacing(tri_vert_clip_positions) : false;
        if (!cull_primitive)
        {
            const float3[3] tri_vert_ndc_positions = float3[3](
                tri_vert_clip_positions[0].xyz / (tri_vert_clip_positions[0].w),
                tri_vert_clip_positions[1].xyz / (tri_vert_clip_positions[1].w),
                tri_vert_clip_positions[2].xyz / (tri_vert_clip_positions[2].w)
            );

            float2 ndc_min = min(min(tri_vert_ndc_positions[0].xy, tri_vert_ndc_positions[1].xy), tri_vert_ndc_positions[2].xy);
            float2 ndc_max = max(max(tri_vert_ndc_positions[0].xy, tri_vert_ndc_positions[1].xy), tri_vert_ndc_positions[2].xy);
            let cull_micro_poly_invisible = false; //is_triangle_invisible_micro_triangle( ndc_min, ndc_max, float2(render_target_size));
            cull_primitive = cull_micro_poly_invisible;
        }
        #endif


        P primitive;
        if(!cull_primitive) {
            uint array_index;
            float2 base_resolution;
            if(is_directional_vsm) 
            {
                VSMDirectionalIndirections indirections = unpack_vsm_directional_light_indirections(vsm_meta_info);
                array_index = indirections.cascade;
                base_resolution = float2(camera.screen_size >> 1);
            }
            else 
            {
                VSMPointIndirections indirections = unpack_vsm_point_light_indirections(vsm_meta_info);
                array_index = get_vsm_point_page_array_idx(indirections.face_index, indirections.point_light_index);
                base_resolution = VSM_PAGE_TABLE_RESOLUTION / (1 << indirections.mip_level);
            }
            if(! max_behind_near_plane)
            {
                cull_primitive = is_ndc_aabb_hiz_opacity_occluded(tri_aabb, hiz, base_resolution, array_index);
            }
        }

        primitive.set_vsm_meta_info(vsm_meta_info);
        primitive.set_cull_primitive(cull_primitive);

        if(P is DirectionalVSMMaskMeshShaderPrimitive) 
        {
            var mprim = reinterpret<DirectionalVSMMaskMeshShaderPrimitive>(primitive);
            mprim.material_index = meshlet_inst.material_index;
            primitive = reinterpret<P>(mprim);
        }
        else if (P is PointVSMMaskMeshShaderPrimitive)
        {
            var mprim = reinterpret<PointVSMMaskMeshShaderPrimitive>(primitive);
            mprim.material_index = meshlet_inst.material_index;
            primitive = reinterpret<P>(mprim);
        }
        out_primitives[in_meshlet_triangle_index] = primitive;
    }
}