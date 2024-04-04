#include "daxa/daxa.inl"

#include "vsm.inl"
#include "shader_lib/cull_util.hlsl"
#include "shader_lib/vsm_util.glsl"
#include "../rasterize_visbuffer/draw_visbuffer.hlsl"

[[vk::push_constant]] CullAndDrawPagesPush vsm_push;
[[vk::push_constant]] CullAndDrawPages_WriteCommandH::AttachmentShaderBlob write_command_push;

struct CullMeshletsDrawPagesPayload
{
    uint task_shader_wg_meshlet_args_offset;
    uint task_shader_surviving_meshlets_mask;
    uint task_shader_clip_level;
};
// 32 is the number of buckets - if this number is changed this also needs to be changed to match it
[numthreads(32, 1, 1)]
[shader("compute")]
void vsm_entry_write_commands(
    uint3 svgtid : SV_GroupThreadID,
    uint3 svgid : SV_GroupID
)
{
    let push = write_command_push;
    push.vsm_meshlets_cull_arg_buckets.draw_list_arg_buckets[svgid.x].commands[svgtid.x].y = VSM_CLIP_LEVELS;
}

bool is_meshlet_occluded_vsm2(
    uint tid,
    CameraInfo camera,
    MeshletInstance meshlet_inst,
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms,
    daxa_BufferPtr(GPUMesh) meshes,
    daxa_ImageViewId hiz,
    daxa_u32 cascade)
{
    GPUMesh mesh_data = deref_i(meshes, meshlet_inst.mesh_index);

    if (meshlet_inst.meshlet_index >= mesh_data.meshlet_count)
    {
        return true;
    }

    daxa_f32mat4x4 model_matrix = mat_4x3_to_4x4(deref_i(entity_combined_transforms, meshlet_inst.entity_index));
    BoundingSphere model_bounding_sphere = deref_i(mesh_data.meshlet_bounds, meshlet_inst.meshlet_index);
    if (is_sphere_out_of_frustum(camera, model_matrix, model_bounding_sphere))
    {
        #if defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus)
            ShaderDebugCircleDraw circle;
            circle.position = ws_center;
            circle.radius = scaled_radius;
            circle.color = daxa_f32vec3(1,1,0);
            circle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
            debug_draw_circle(GLOBALS.debug, circle);
        #endif
        return true;
    }
    return false;

    AABB meshlet_aabb = deref_i(mesh_data.meshlet_aabbs, meshlet_inst.meshlet_index);
    NdcAABB meshlet_ndc_aabb = calculate_meshlet_ndc_aabb(camera, meshlet_inst, model_matrix, meshlet_aabb);
    const bool page_opacity_cull = is_ndc_aabb_hiz_opacity_occluded(camera, meshlet_ndc_aabb, hiz, cascade);

    #if (defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus))
    if (page_opacity_cull)
    {
        ShaderDebugAABBDraw aabb1;
        aabb1.position = mul(model_matrix, daxa_f32vec4(meshlet_aabb.center,1)).xyz;
        aabb1.size = mul(model_matrix, daxa_f32vec4(meshlet_aabb.size,0)).xyz;
        aabb1.color = daxa_f32vec3(0.1, 0.5, 1);
        aabb1.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        debug_draw_aabb(GLOBALS.debug, aabb1);
        {
            ShaderDebugRectangleDraw rectangle;
            const daxa_f32vec3 rec_size = (ndc_max - ndc_min);
            rectangle.center = ndc_min + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = daxa_f32vec3(0, 1, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(GLOBALS.debug, rectangle);
        }
        {
            const daxa_f32vec2 min_r = quad_corner_texel << imip;
            const daxa_f32vec2 max_r = (quad_corner_texel + 2) << imip;
            const daxa_f32vec2 min_r_uv = min_r / f_hiz_resolution;
            const daxa_f32vec2 max_r_uv = max_r / f_hiz_resolution;
            const daxa_f32vec2 min_r_ndc = min_r_uv * 2.0f - 1.0f;
            const daxa_f32vec2 max_r_ndc = max_r_uv * 2.0f - 1.0f;
            ShaderDebugRectangleDraw rectangle;
            const daxa_f32vec3 rec_size = (daxa_f32vec3(max_r_ndc, ndc_max.z) - daxa_f32vec3(min_r_ndc, ndc_min.z));
            rectangle.center = daxa_f32vec3(min_r_ndc, ndc_min.z) + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = daxa_f32vec3(1, 0, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(GLOBALS.debug, rectangle);
        }
    }
    #endif

    return page_opacity_cull;
}

bool get_meshlet_instance_from_arg_buckets2(
    uint thread_id, 
    uint arg_bucket_index, 
    daxa_BufferPtr(MeshletCullArgBucketsBufferHead) meshlet_cull_indirect_args,
    uint opaque_draw_list_type, 
    out MeshletInstance meshlet_inst)
{
    const uint indirect_arg_index = thread_id >> arg_bucket_index;
    const uint valid_arg_count = 
        deref(meshlet_cull_indirect_args)
        .draw_list_arg_buckets[opaque_draw_list_type]
        .indirect_arg_counts[arg_bucket_index];
    // As work groups are launched in multiples of 128 (or 32 in the case of task shaders), 
    // there may be threads with indices greater then the arg count for a bucket.
    if (indirect_arg_index >= valid_arg_count)
    {
        return false;
    }
    const uint in_arg_meshlet_index = thread_id - (indirect_arg_index << arg_bucket_index);
    daxa_RWBufferPtr(MeshletCullIndirectArg) args_ptr = 
        (deref(meshlet_cull_indirect_args)
        .draw_list_arg_buckets[opaque_draw_list_type]
        .indirect_arg_ptrs[arg_bucket_index]);
    const MeshletCullIndirectArg arg = deref_i(args_ptr, indirect_arg_index);
    
    meshlet_inst.entity_index = arg.entity_index;
    meshlet_inst.material_index = arg.material_index;
    meshlet_inst.mesh_index = arg.mesh_index;
    meshlet_inst.meshlet_index = arg.meshlet_indices_offset + in_arg_meshlet_index;
    meshlet_inst.in_mesh_group_index = arg.in_mesh_group_index;
    if (meshlet_inst.mesh_index > 5000 && opaque_draw_list_type == 0) {
        printf("args_ptr %ull,\nopaque_draw_list_type %u,\n arg_bucket_index %u,\n indirect_arg_index: %u\n\n", uint64_t(args_ptr), opaque_draw_list_type, arg_bucket_index, indirect_arg_index);
    }
    // Work argument may work on less then 1<<bucket_index meshlets.
    // In this case we cull threads with an index over meshlet_count.
    return in_arg_meshlet_index < arg.meshlet_count;
}


[shader("amplification")]
[numthreads(32, 1, 1)]
func vsm_entry_task(
    uint3 svtid : SV_DispatchThreadID,
    uint3 svgid : SV_GroupID
)
{
    let push = vsm_push;
    let clip_level = svgid.y;
    MeshletInstance instanced_meshlet;
    const bool valid_meshlet = get_meshlet_instance_from_arg_buckets(
        svtid.x,
        push.bucket_index,
        push.attachments.meshlets_cull_arg_buckets,
        push.draw_list_type,
        instanced_meshlet);
    bool draw_meshlet = valid_meshlet;
#if ENABLE_MESHLET_CULLING == 1
    // We still continue to run the task shader even with invalid meshlets.
    // We simple set the occluded value to true for these invalida meshlets.
    // This is done so that the following WaveOps are well formed and have all threads active. 
    if (valid_meshlet)
    {
        draw_meshlet = draw_meshlet && !is_meshlet_occluded_vsm2(
            svtid.x,
            deref_i(push.attachments.vsm_clip_projections, clip_level).camera,
            instanced_meshlet,
            push.attachments.entity_combined_transforms,
            push.attachments.meshes,
            push.attachments.vsm_dirty_bit_hiz,
            clip_level
        );
    }
#endif

    CullMeshletsDrawPagesPayload payload;
    payload.task_shader_wg_meshlet_args_offset = svgid.x * MESH_SHADER_WORKGROUP_X;
    payload.task_shader_surviving_meshlets_mask = WaveActiveBallot(draw_meshlet).x;
    payload.task_shader_clip_level = clip_level;
    let surviving_meshlet_count = WaveActiveSum(draw_meshlet ? 1u : 0u);
    // When not occluded, this value determines the new packed index for each thread in the wave:
    let local_survivor_index = WavePrefixSum(draw_meshlet ? 1u : 0u);

    // DispatchMesh(1, 0, 1, payload);
    DispatchMesh(1, surviving_meshlet_count, 1, payload);
}

interface VSMMeshShaderPrimitiveT
{
    DECL_GET_SET(uint, clip_level)
}

struct VSMOpaqueMeshShaderPrimitive : VSMMeshShaderPrimitiveT
{
    nointerpolation [[vk::location(0)]] uint clip_level;
    IMPL_GET_SET(uint, clip_level)
};

struct VSMMaskMeshShaderPrimitive : VSMMeshShaderPrimitiveT
{
    nointerpolation [[vk::location(0)]] uint clip_level;
    nointerpolation [[vk::location(1)]] uint material_index;
    IMPL_GET_SET(uint, clip_level)
}

func generic_vsm_mesh<V: MeshShaderVertexT, P: VSMMeshShaderPrimitiveT>(
    in uint3 svtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    uint clip_level,
    MeshletInstance meshlet_inst)
{    
    let push = vsm_push;
    const GPUMesh mesh = deref_i(push.attachments.meshes, meshlet_inst.mesh_index);
    const Meshlet meshlet = deref_i(mesh.meshlets, meshlet_inst.meshlet_index);
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref_i(push.attachments.meshes, meshlet_inst.mesh_index).micro_indices;
    const daxa_f32mat4x4 view_proj = deref_i(push.attachments.vsm_clip_projections, clip_level).camera.view_proj;

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

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
        const daxa_f32mat4x3 model_mat4x3 = deref_i(push.attachments.entity_combined_transforms, meshlet_inst.entity_index);
        const daxa_f32mat4x4 model_mat = mat_4x3_to_4x4(model_mat4x3);
        const daxa_f32vec4 pos = mul(view_proj, mul(model_mat, vertex_position));

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

        P primitive;
        primitive.set_clip_level(clip_level);
        if (P is VSMMaskMeshShaderPrimitive)
        {
            var mprim = reinterpret<VSMMaskMeshShaderPrimitive>(primitive);
            mprim.material_index = meshlet_inst.material_index;
            primitive = reinterpret<P>(mprim);
        }
        out_primitives[in_meshlet_triangle_index] = primitive;
    }
}

func vsm_mesh_cull_draw<V: MeshShaderVertexT, P: VSMMeshShaderPrimitiveT>(
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
    MeshletInstance meshlet_inst;
    // The meshlet should always be valid here, 
    // as otherwise the task shader would not have dispatched this mesh shader.
    let meshlet_valid = get_meshlet_instance_from_arg_buckets(
        meshlet_cull_arg_index, 
        push.bucket_index, 
        push.attachments.meshlets_cull_arg_buckets, 
        push.draw_list_type, 
        meshlet_inst);
    
    generic_vsm_mesh(svtid, out_indices, out_vertices, out_primitives, payload.task_shader_clip_level, meshlet_inst);
}

[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func vsm_entry_mesh_opaque(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderOpaqueVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<VSMOpaqueMeshShaderPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in payload CullMeshletsDrawPagesPayload payload)
{
    vsm_mesh_cull_draw(svtid, out_indices, out_vertices, out_primitives, payload);
}

[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func vsm_entry_mesh_masked(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderMaskVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<VSMMaskMeshShaderPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in payload CullMeshletsDrawPagesPayload payload)
{
    vsm_mesh_cull_draw(svtid, out_indices, out_vertices, out_primitives, payload);
}

[[vk::binding(DAXA_STORAGE_IMAGE_BINDING, 0)]] RWTexture2D<daxa::u32> RWTexture2D_utable[];

[shader("fragment")]
void vsm_entry_fragment_opaque(
    in MeshShaderOpaqueVertex vert,
    in VSMOpaqueMeshShaderPrimitive prim)
{
    let push = vsm_push;
    const float2 virtual_uv = vert.position.xy / VSM_TEXTURE_RESOLUTION;

    let wrapped_coords = vsm_clip_info_to_wrapped_coords(
        {prim.clip_level, virtual_uv},
        push.attachments.vsm_clip_projections);

    let vsm_page_entry = RWTexture2DArray<uint>::get(push.attachments.vsm_page_table)[uint3(wrapped_coords)].x;
    if(get_is_allocated(vsm_page_entry) && get_is_dirty(vsm_page_entry))
    {
        let memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
        let physical_texel_coords = virtual_uv_to_physical_texel(virtual_uv, memory_page_coords);

        InterlockedMin(
            RWTexture2D_utable[push.daxa_u32_vsm_memory_view.index()][physical_texel_coords],
            asuint(get_page_offset_depth(
                {prim.clip_level, virtual_uv}, 
                vert.position.z / vert.position.w,
                push.attachments.vsm_clip_projections
            ))
        );
    }
}

[shader("fragment")]
void vsm_entry_fragment_masked(
    in MeshShaderMaskVertex vert,
    in VSMMaskMeshShaderPrimitive prim)
{
    let push = vsm_push;
    const float2 virtual_uv = vert.position.xy / VSM_TEXTURE_RESOLUTION;

    let wrapped_coords = vsm_clip_info_to_wrapped_coords(
        {prim.clip_level, virtual_uv},
        push.attachments.vsm_clip_projections);

    let vsm_page_entry = RWTexture2DArray<uint>::get(push.attachments.vsm_page_table)[uint3(wrapped_coords)].x;
    if(get_is_allocated(vsm_page_entry) && get_is_dirty(vsm_page_entry))
    {
        if(prim.material_index != INVALID_MANIFEST_INDEX)
        {
            const GPUMaterial material = deref_i(push.attachments.material_manifest, prim.material_index);
            if(material.diffuse_texture_id.value != 0 && material.alpha_discard_enabled)
            {
                float alpha = Texture2D<float>::get(material.diffuse_texture_id)
                    .Sample(SamplerState::get(push.attachments.globals->samplers.linear_repeat), vert.uv).a;
                if(alpha < 0.5f) { discard; }
            }
        }

        let memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
        let physical_texel_coords = virtual_uv_to_physical_texel(virtual_uv, memory_page_coords);

        InterlockedMin(
            RWTexture2D_utable[push.daxa_u32_vsm_memory_view.index()][physical_texel_coords],
            asuint(get_page_offset_depth(
                {prim.clip_level, virtual_uv}, 
                vert.position.z / vert.position.w,
                push.attachments.vsm_clip_projections
            ))
        );
    }
}