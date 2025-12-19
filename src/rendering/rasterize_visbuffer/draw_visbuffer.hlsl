#pragma once

#include "daxa/daxa.inl"
#include "draw_visbuffer.inl"

[[vk::push_constant]] DrawVisbufferPush_WriteCommand write_cmd_p;
[[vk::push_constant]] DrawVisbufferPush draw_p;
[[vk::push_constant]] CullMeshletsDrawVisbufferPush cull_meshlets_draw_visbuffer_push;

// #define GLOBALS cull_meshlets_draw_visbuffer_push.attach.globals

#include "shader_shared/cull_util.inl"

#include "shader_lib/depth_util.glsl"
#include "shader_lib/cull_util.hlsl"
#include "shader_lib/pass_logic.glsl"
#include "shader_lib/gpu_work_expansion.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/vsm_util.glsl"

[shader("compute")]
[numthreads(1,1,1)]
void entry_write_commands(uint3 dtid : SV_DispatchThreadID)
{
    DrawVisbufferPush_WriteCommand push = write_cmd_p;
    for (uint draw_list_type = 0; draw_list_type < PREPASS_DRAW_LIST_TYPE_COUNT; ++draw_list_type)
    {
        uint meshlets_to_draw = get_meshlet_draw_count(
            push.attach.globals,
            push.attach.meshlet_instances,
            push.pass,
            draw_list_type);
            
        meshlets_to_draw = min(meshlets_to_draw, MAX_MESHLET_INSTANCES);
            DispatchIndirectStruct command;
            command.x = meshlets_to_draw;
            command.y = 1;
            command.z = 1;
            ((DispatchIndirectStruct*)(push.attach.draw_commands))[draw_list_type] = command;
    }
}

#define DECL_GET_SET(TYPE, FIELD)\
    [mutating]\
    func set_##FIELD(TYPE v);\
    func get_##FIELD() -> TYPE;

#define IMPL_GET_SET(TYPE, FIELD)\
    [mutating]\
    func set_##FIELD(TYPE v) { FIELD = v; }\
    func get_##FIELD() -> TYPE { return FIELD; };


interface IFragmentOut
{
    DECL_GET_SET(uint, visibility_id)
}

struct FragmentOut : IFragmentOut
{
    [[vk::location(0)]] uint visibility_id;
    IMPL_GET_SET(uint, visibility_id)
};

interface IFragmentExtraData
{

};
struct FragmentMaskedData : IFragmentExtraData
{
    uint material_index;
    GPUMaterial* materials;
    daxa::SamplerId sampler;
    float2 uv;
};
struct NoIFragmentExtraData : IFragmentExtraData
{

};

func generic_fragment<ExtraData : IFragmentExtraData, FragOutT : IFragmentOut>(out FragOutT ret, uint2 index, uint vis_id, daxa::ImageViewId overdraw_image, ExtraData extra, float depth)
{
    ret.set_visibility_id(vis_id);
    if (ExtraData is FragmentMaskedData)
    {
        let masked_data = reinterpret<FragmentMaskedData>(extra);
        if (masked_data.material_index != INVALID_MANIFEST_INDEX)
        {
            GPUMaterial material = deref_i(masked_data.materials, masked_data.material_index);
            float alpha = 1.0;
            if (material.opacity_texture_id.value != 0 && material.alpha_discard_enabled)
            {
                // TODO: WHAT THE FUCK IS THIS BUG? WHY ARE WE SAMPLING diffuse_texture_id IN THIS BRANCH??
                alpha = Texture2D<float4>::get(material.opacity_texture_id)
                    .Sample( SamplerState::get(masked_data.sampler), masked_data.uv).r; 
            }
            else if (material.diffuse_texture_id.value != 0 && material.alpha_discard_enabled)
            {
                alpha = Texture2D<float4>::get(material.diffuse_texture_id)
                    .Sample( SamplerState::get(masked_data.sampler), masked_data.uv).a; 
            }
            // const float max_obj_space_deriv_len = max(length(ddx(mvertex.object_space_position)), length(ddy(mvertex.object_space_position)));
            // const float threshold = compute_hashed_alpha_threshold(mvertex.object_space_position, max_obj_space_deriv_len, 0.3);
            // if (alpha < clamp(threshold, 0.001, 1.0)) // { discard; }
            if(alpha < 0.1) { discard; }
        }
    }
    if (overdraw_image.value != 0)
    {
        uint prev_val;
        InterlockedAdd(RWTexture2D<uint>::get_formatted(overdraw_image)[index], 1, prev_val);
    }
}

// Interface:
interface MeshShaderVertexT
{
    DECL_GET_SET(float4, position)
    static const uint PREPASS_DRAW_LIST_TYPE;
}
interface MeshShaderPrimitiveT
{
    DECL_GET_SET(uint, visibility_id)
    DECL_GET_SET(bool, cull_primitive)
}


// Opaque:
struct MeshShaderOpaqueVertex : MeshShaderVertexT
{
    float4 position : SV_Position;
    IMPL_GET_SET(float4, position)
    static const uint PREPASS_DRAW_LIST_TYPE = PREPASS_DRAW_LIST_OPAQUE;
};
struct MeshShaderOpaquePrimitive : MeshShaderPrimitiveT
{
    nointerpolation [[vk::location(0)]] uint visibility_id;
    IMPL_GET_SET(uint, visibility_id)
    bool cull_primitive : SV_CullPrimitive;
    IMPL_GET_SET(bool, cull_primitive)
};


// Mask:
struct MeshShaderMaskVertex : MeshShaderVertexT
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
    IMPL_GET_SET(float4, position)
    static const uint PREPASS_DRAW_LIST_TYPE = PREPASS_DRAW_LIST_MASKED;
}
struct MeshShaderMaskPrimitive : MeshShaderPrimitiveT
{
    nointerpolation [[vk::location(0)]] uint visibility_id;
    nointerpolation [[vk::location(1)]] uint material_index;
    bool cull_primitive : SV_CullPrimitive;
    IMPL_GET_SET(uint, visibility_id)
    IMPL_GET_SET(bool, cull_primitive)
};

func shuffle_arr2(float4 local_values[2], uint load_index) -> float4
{
    let load0 = WaveReadLaneAt(local_values[0], (load_index % 32));
    let load1 = WaveReadLaneAt(local_values[1], (uint(max(0u, int(load_index) - 32))));
    return load_index > 31 ? load1 : load0;
}

// --- Mesh shader opaque ---

func generic_mesh<V: MeshShaderVertexT, P: MeshShaderPrimitiveT>(
    in DrawVisbufferPush push,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in GPUMesh mesh,
    in uint meshlet_thread_index,
    in uint meshlet_instance_index,
    in MeshletInstance meshlet_instance,
    in bool cull_backfaces,
    in bool cull_hiz_occluded)
{          
    const Meshlet meshlet = deref_i(mesh.meshlets, meshlet_instance.meshlet_index);
    const bool observer_pass = push.draw_data.observer;
    cull_hiz_occluded = cull_hiz_occluded && !(observer_pass);
    const daxa_f32mat4x4 view_proj = 
        observer_pass ? 
        deref(push.attach.globals).view_camera.view_proj : 
        deref(push.attach.globals).main_camera.view_proj;

    const daxa_f32mat4x3 model_mat4x3 = deref_i(push.entity_combined_transforms, meshlet_instance.entity_index);
    const daxa_f32mat4x4 model_mat = mat_4x3_to_4x4(model_mat4x3);
    let mvp = mul(view_proj, model_mat);

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);
    if (meshlet_instance_index >= MAX_MESHLET_INSTANCES)
    {
        printf("GPU ERROR: Invalid meshlet passed to mesh shader! Meshlet instance index %i exceeded max meshlet instance count %i\n", meshlet_instance_index, MAX_MESHLET_INSTANCES);
    }

    float4 local_clip_vertices[2];
    float3 local_ndc_vertices[2];

    for (uint l = 0; l < 2; ++l)
    {
        uint vertex_offset = MESH_SHADER_WORKGROUP_X * l;
        const uint in_meshlet_vertex_index = meshlet_thread_index + vertex_offset;
        if (in_meshlet_vertex_index >= meshlet.vertex_count) continue;

        // Very slow fetch, as its incoherent memory address across warps.
        const uint in_mesh_vertex_index = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + in_meshlet_vertex_index);
        if (in_mesh_vertex_index >= mesh.vertex_count)
        {
            /// TODO: ASSERT HERE. 
            continue;
        }
        
        // Very slow fetch, as its incoherent memory address across warps.
        const daxa_f32vec4 vertex_position = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index), 1);
        const daxa_f32vec4 pos = mul(mvp, vertex_position);

        V vertex;
        local_clip_vertices[l] = pos;
        vertex.set_position(pos);
        if (V is MeshShaderMaskVertex)
        {
            var mvertex = reinterpret<MeshShaderMaskVertex>(vertex);
            mvertex.uv = float2(0,0);
            if (as_address(mesh.vertex_uvs) != 0)
            {
                mvertex.uv = mesh.vertex_uvs[in_mesh_vertex_index];
            }
            vertex = reinterpret<V>(mvertex);
        }
        out_vertices[in_meshlet_vertex_index] = vertex;
    }

    for (uint triangle_offset = 0; triangle_offset < meshlet.triangle_count; triangle_offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint in_meshlet_triangle_index = meshlet_thread_index + triangle_offset;
        uint3 tri_in_meshlet_vertex_indices = uint3(0,0,0);
        if (in_meshlet_triangle_index < meshlet.triangle_count)
        {
            tri_in_meshlet_vertex_indices = uint3(
                get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 0),
                get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 1),
                get_micro_index(mesh.micro_indices, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 2)
            );
        }
        float4[3] tri_vert_clip_positions = float4[3](
            shuffle_arr2(local_clip_vertices, tri_in_meshlet_vertex_indices[0]),
            shuffle_arr2(local_clip_vertices, tri_in_meshlet_vertex_indices[1]),
            shuffle_arr2(local_clip_vertices, tri_in_meshlet_vertex_indices[2])
        );

        if (in_meshlet_triangle_index < meshlet.triangle_count)
        {
            bool cull_primitive = false;

            // Observer culls triangles from the perspective of the main camera.
            if (push.draw_data.observer)
            {        
                for (uint c = 0; c < 3; ++c)
                {
                    const uint in_mesh_vertex_index = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + tri_in_meshlet_vertex_indices[c]);
                    const daxa_f32vec4 vertex_position = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index), 1);
                    let main_camera_view_proj = push.attach.globals.main_camera.view_proj;
                    const daxa_f32vec4 pos = mul(main_camera_view_proj, mul(model_mat, vertex_position));
                    tri_vert_clip_positions[c] = pos;
                }
            }

            if (push.attach.globals.settings.enable_triangle_cull)
            {
                if (cull_backfaces)
                {
                    cull_primitive = is_triangle_backfacing(tri_vert_clip_positions);
                }
                if (!cull_primitive)
                {
                    const float3[3] tri_vert_ndc_positions = float3[3](
                        tri_vert_clip_positions[0].xyz / (tri_vert_clip_positions[0].w),
                        tri_vert_clip_positions[1].xyz / (tri_vert_clip_positions[1].w),
                        tri_vert_clip_positions[2].xyz / (tri_vert_clip_positions[2].w)
                    );

                    float2 ndc_min = min(min(tri_vert_ndc_positions[0].xy, tri_vert_ndc_positions[1].xy), tri_vert_ndc_positions[2].xy);
                    float2 ndc_max = max(max(tri_vert_ndc_positions[0].xy, tri_vert_ndc_positions[1].xy), tri_vert_ndc_positions[2].xy);
                    let cull_micro_poly_invisible = is_triangle_invisible_micro_triangle( ndc_min, ndc_max, float2(push.attach.globals.settings.render_target_size));
                    cull_primitive = cull_micro_poly_invisible;

                    const float2 ndc_size = (ndc_max - ndc_min);
                    const float2 ndc_pixel_size = 0.5f * ndc_size * push.attach.globals.settings.render_target_size;
                    const float ndc_pixel_area_size = ndc_pixel_size.x * ndc_pixel_size.y;
                    bool large_triangle = ndc_pixel_area_size > 128;
                    if (large_triangle && push.attach.globals.settings.enable_triangle_cull && (push.attach.hiz.value != 0) && !cull_primitive && cull_hiz_occluded)
                    {
                        let is_hiz_occluded = is_triangle_hiz_occluded(
                            push.attach.globals.debug,
                            push.attach.globals.main_camera,
                            tri_vert_ndc_positions,
                            push.attach.globals.cull_data,
                            push.attach.hiz);
                        cull_primitive = is_hiz_occluded;
                    }
                }
            }
            
            P primitive;
            primitive.set_cull_primitive(cull_primitive);
            if (!cull_primitive)
            {
                uint visibility_id = TRIANGLE_ID_MAKE(meshlet_instance_index, in_meshlet_triangle_index);
                primitive.set_visibility_id(cull_primitive ? ~0u : visibility_id);
                if (P is MeshShaderMaskPrimitive)
                {
                    var mprim = reinterpret<MeshShaderMaskPrimitive>(primitive);
                    mprim.material_index = meshlet_instance.material_index;
                    primitive = reinterpret<P>(mprim);
                }
                out_indices[in_meshlet_triangle_index] = tri_in_meshlet_vertex_indices;
            }
            out_primitives[in_meshlet_triangle_index] = primitive;
        }
    }
}

func generic_mesh_draw_only<V: MeshShaderVertexT, P: MeshShaderPrimitiveT>(
    in uint gid,
    in uint gtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives)
{
    const uint meshlet_instance_index = get_meshlet_instance_index(
        draw_p.attach.globals,
        draw_p.attach.meshlet_instances, 
        draw_p.draw_data.pass_index, 
        V::PREPASS_DRAW_LIST_TYPE,
        gid);
    if (meshlet_instance_index >= MAX_MESHLET_INSTANCES)
    {
        SetMeshOutputCounts(0,0);
        return;
    }
    const uint total_meshlet_count = 
        deref(draw_p.attach.meshlet_instances).prepass_draw_lists[0].pass_counts[0] + 
        deref(draw_p.attach.meshlet_instances).prepass_draw_lists[0].pass_counts[1];
    const MeshletInstance meshlet_instance = deref_i(deref(draw_p.attach.meshlet_instances).meshlets, meshlet_instance_index);

    bool cull_backfaces = !GPU_MATERIAL_FALLBACK.alpha_discard_enabled;
    if (meshlet_instance.material_index != INVALID_MANIFEST_INDEX)
    {
        GPUMaterial material = draw_p.materials[meshlet_instance.material_index];
        cull_backfaces = !material.alpha_discard_enabled && !material.double_sided_enabled;
    }

    // Culling in first pass would require calculating tri vertex positions in old camera space.
    // That would require a shitload of extra processing power, 2x vertex work.
    // Because of this we simply dont hiz cull tris in first pass ever.
    let cull_hiz_occluded = draw_p.draw_data.pass_index != VISBUF_FIRST_PASS;

    if (meshlet_instance.mesh_index == INVALID_MANIFEST_INDEX)
    {
        SetMeshOutputCounts(0,0);
        return;
    }
    const GPUMesh mesh = draw_p.meshes[meshlet_instance.mesh_index];
    if (mesh.mesh_buffer.value == 0) // Unloaded Mesh
    {
        SetMeshOutputCounts(0,0);
        return;
    }
    generic_mesh(draw_p, out_indices, out_vertices, out_primitives, mesh, gtid, meshlet_instance_index, meshlet_instance, cull_backfaces, cull_hiz_occluded);
}

// --- Mesh shader opaque ---
[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func entry_mesh_opaque(
    uint gtid : SV_GroupThreadID,
    uint gid : SV_GroupID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderOpaqueVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<MeshShaderOpaquePrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives)
{

    generic_mesh_draw_only(gid, gtid, out_indices, out_vertices, out_primitives);
}

// Didnt seem to do much.
// [earlydepthstencil]
[shader("fragment")]
FragmentOut entry_fragment_opaque(in MeshShaderOpaqueVertex vert, in MeshShaderOpaquePrimitive prim)
{
    FragmentOut ret;
    generic_fragment(
        ret,
        uint2(vert.position.xy),
        prim.visibility_id,
        draw_p.attach.overdraw_image,
        NoIFragmentExtraData(),
        0
    );
    return ret;
}
// --- Mesh shader opaque ---


// --- Mesh shader mask ---

[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func entry_mesh_masked(
    uint gtid : SV_GroupThreadID,
    uint gid : SV_GroupID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderMaskVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<MeshShaderMaskPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives)
{
    generic_mesh_draw_only(gid, gtid, out_indices, out_vertices, out_primitives);
}

[shader("fragment")]
FragmentOut entry_fragment_masked(in MeshShaderMaskVertex vert, in MeshShaderMaskPrimitive prim)
{
    FragmentOut ret;
    generic_fragment(
        ret,
        uint2(vert.position.xy),
        prim.visibility_id,
        draw_p.attach.overdraw_image,
        FragmentMaskedData(
            prim.material_index,
            draw_p.materials,
            draw_p.attach.globals->samplers.linear_repeat_ani,
            // draw_p.attach.globals->samplers.nearest_repeat,
            vert.uv
        ),
        0
    );
    return ret;
}

// --- Mesh shader mask ---

// --- Cull Meshlets Draw Visbuffer

struct CullMeshletsDrawVisbufferPayload
{
    uint task_shader_wg_meshlet_args_offset;
    uint task_shader_meshlet_instances_offset;
    uint task_shader_surviving_meshlets_mask;
    uint cull_backfaces_mask;
};

// TODO: This should not be three functions but only one. (Maybe make it a template?)
bool get_vsm_directional_meshlet_instance_from_work_item(
    bool prefix_sum_expansion,
    uint64_t expansion_buffer_ptr,
    MeshInstancesBufferHead * mesh_instances,
    GPUMesh * meshes,
    uint thread_index, 
    out MeshletInstance meshlet_instance,
    out VSMDirectionalIndirections indirections
)
{
    DstItemInfo workitem;
    bool valid_meshlet = false;
    if (prefix_sum_expansion)
    {
        PrefixSumWorkExpansionBufferHead * prefix_expansion = (PrefixSumWorkExpansionBufferHead *)expansion_buffer_ptr;
        valid_meshlet = prefix_sum_expansion_get_workitem(prefix_expansion, thread_index, workitem);
    }
    else
    {
        Po2BucketWorkExpansionBufferHead * po2bucket_expansion = (Po2BucketWorkExpansionBufferHead *)expansion_buffer_ptr;
        valid_meshlet = po2bucket_expansion_get_workitem(po2bucket_expansion, thread_index, workitem);
    }
    if (valid_meshlet)
    {
        indirections = unpack_vsm_directional_light_indirections(workitem.payload.x);
        MeshInstance mesh_instance = mesh_instances.instances[indirections.mesh_instance_index];
        const uint mesh_index = mesh_instance.mesh_index;
        GPUMesh mesh = meshes[mesh_index];    
        if (mesh.mesh_buffer.value == 0) // Unloaded Mesh
        {
            return false;
        }
        meshlet_instance.entity_index = mesh_instance.entity_index;
        meshlet_instance.in_mesh_group_index = mesh_instance.in_mesh_group_index;
        meshlet_instance.material_index = mesh.material_index;
        meshlet_instance.mesh_index = mesh_index;
        meshlet_instance.meshlet_index = workitem.work_item_index;
        meshlet_instance.mesh_instance_index = indirections.mesh_instance_index;
    }
    return valid_meshlet;
}

bool get_vsm_point_spot_meshlet_instance_from_work_item(
    bool prefix_sum_expansion,
    uint64_t expansion_buffer_ptr,
    MeshInstancesBufferHead * mesh_instances,
    GPUMesh * meshes,
    uint thread_index, 
    out MeshletInstance meshlet_instance,
    out VSMPointSpotIndirections point_spot_indirections,
)
{
    DstItemInfo workitem;
    bool valid_meshlet = false;
    if (prefix_sum_expansion)
    {
        PrefixSumWorkExpansionBufferHead * prefix_expansion = (PrefixSumWorkExpansionBufferHead *)expansion_buffer_ptr;
        valid_meshlet = prefix_sum_expansion_get_workitem(prefix_expansion, thread_index, workitem);
    }
    else
    {
        Po2BucketWorkExpansionBufferHead * po2bucket_expansion = (Po2BucketWorkExpansionBufferHead *)expansion_buffer_ptr;
        valid_meshlet = po2bucket_expansion_get_workitem(po2bucket_expansion, thread_index, workitem);
    }
    if (valid_meshlet)
    {
        point_spot_indirections = unpack_vsm_point_spot_light_indirections(workitem.payload.x);

        MeshInstance mesh_instance = mesh_instances.instances[point_spot_indirections.mesh_instance_index];
        // WARNING: WE IGNORE THE MESH INSTANCE MESH INDEX AND USE THE ONE FROM THE PAYLOAD HERE!
        const uint mesh_index = workitem.payload.y;
        GPUMesh mesh = meshes[mesh_index];    
        if (mesh.mesh_buffer.value == 0) // Unloaded Mesh
        {
            return false;
        }
        meshlet_instance.entity_index = mesh_instance.entity_index;
        meshlet_instance.in_mesh_group_index = mesh_instance.in_mesh_group_index;
        meshlet_instance.material_index = mesh.material_index;
        meshlet_instance.mesh_index = mesh_index;
        meshlet_instance.meshlet_index = workitem.work_item_index;
        meshlet_instance.mesh_instance_index = point_spot_indirections.mesh_instance_index;
    }
    return valid_meshlet;
}

bool get_meshlet_instance_from_workitem(
    bool prefix_sum_expansion,
    uint64_t expansion_buffer_ptr,
    MeshInstancesBufferHead * mesh_instances,
    GPUMesh * meshes,
    uint thread_index, 
    out MeshletInstance meshlet_instance)
{
    DstItemInfo workitem;
    bool valid_meshlet = false;
    if (prefix_sum_expansion)
    {
        PrefixSumWorkExpansionBufferHead * prefix_expansion = (PrefixSumWorkExpansionBufferHead *)expansion_buffer_ptr;
        valid_meshlet = prefix_sum_expansion_get_workitem(prefix_expansion, thread_index, workitem);
    }
    else
    {
        Po2BucketWorkExpansionBufferHead * po2bucket_expansion = (Po2BucketWorkExpansionBufferHead *)expansion_buffer_ptr;
        valid_meshlet = po2bucket_expansion_get_workitem(po2bucket_expansion, thread_index, workitem);
    }
    if (valid_meshlet)
    {
        MeshInstance mesh_instance = mesh_instances.instances[workitem.payload.x];
        const uint mesh_index = mesh_instance.mesh_index;
        GPUMesh mesh = meshes[mesh_index];    
        if (mesh.mesh_buffer.value == 0) // Unloaded Mesh
        {
            return false;
        }
        meshlet_instance.entity_index = mesh_instance.entity_index;
        meshlet_instance.in_mesh_group_index = mesh_instance.in_mesh_group_index;
        meshlet_instance.material_index = mesh.material_index;
        meshlet_instance.mesh_index = mesh_index;
        meshlet_instance.meshlet_index = workitem.work_item_index;
        meshlet_instance.mesh_instance_index = workitem.payload.x;
    }
    return valid_meshlet;
}

struct MeshletCullWriteoutResult
{
    uint warp_meshlet_instances_offset; // WARP UNIFORM
}

/// WARNING: FUNCTION EXPECTS ALL THREADS IN THE WARP TO BE ACTIVE.
func cull_and_writeout_meshlet(inout bool draw_meshlet, MeshletInstance meshlet_instance) -> MeshletCullWriteoutResult
{
    let push = cull_meshlets_draw_visbuffer_push;    

    if (draw_meshlet && (push.draw_data.pass_index == VISBUF_SECOND_PASS))
    {
        draw_meshlet = draw_meshlet && !mehslet_bitfield_check_bit( meshlet_instance, push.attach.first_pass_meshlet_bitfield );
    }
    
    if (draw_meshlet)
    {
        GPUMesh mesh_data = deref_i(push.meshes, meshlet_instance.mesh_index);
        draw_meshlet = draw_meshlet && mesh_data.mesh_buffer.value != 0; // Check if mesh is loaded.
        draw_meshlet = draw_meshlet && (meshlet_instance.meshlet_index < mesh_data.meshlet_count);
    }
    
    // We still continue to run the task shader even with invalid meshlets.
    // We simple set the occluded value to true for these invalida meshlets.
    // This is done so that the following WaveOps are well formed and have all threads active. 
    if (draw_meshlet && push.attach.globals.settings.enable_meshlet_cull)
    {
        let cull_camera = (push.draw_data.pass_index == VISBUF_FIRST_PASS) ? deref(push.attach.globals).main_camera_prev_frame : deref(push.attach.globals).main_camera;

        draw_meshlet = draw_meshlet && !is_meshlet_occluded(
            push.attach.globals.debug,
            cull_camera,
            meshlet_instance,
            push.entity_combined_transforms,
            push.meshes,
            push.attach.globals.cull_data,
            push.attach.hiz);
    }

    // Only mark as drawn if it passes the visibility test!
    if (draw_meshlet && (push.draw_data.pass_index == VISBUF_FIRST_PASS))
    {
        mehslet_bitfield_set_bit( meshlet_instance, push.attach.first_pass_meshlet_bitfield );
    }

    uint surviving_meshlet_count = WaveActiveSum(draw_meshlet ? 1u : 0u);
    uint warp_meshlet_instances_offset = 0;
    bool allocation_failed = false;
    if (surviving_meshlet_count > 0) 
    {
        // When not occluded, this value determines the new packed index for each thread in the wave:
        let local_survivor_index = WavePrefixSum(draw_meshlet ? 1u : 0u);
        uint global_draws_offsets;
        if (WaveIsFirstLane())
        {
            if (push.draw_data.pass_index == VISBUF_FIRST_PASS)
            {
                warp_meshlet_instances_offset =
                    atomicAdd(push.attach.meshlet_instances->pass_counts[0], surviving_meshlet_count);
                global_draws_offsets = 
                    atomicAdd(push.attach.meshlet_instances->prepass_draw_lists[push.draw_data.draw_list_section_index].pass_counts[0], surviving_meshlet_count);
            }
            else
            {
                warp_meshlet_instances_offset = 
                    push.attach.meshlet_instances->pass_counts[0] + 
                    atomicAdd(push.attach.meshlet_instances->pass_counts[1], surviving_meshlet_count);
                global_draws_offsets = 
                    push.attach.meshlet_instances->prepass_draw_lists[push.draw_data.draw_list_section_index].pass_counts[0] + 
                    atomicAdd(push.attach.meshlet_instances->prepass_draw_lists[push.draw_data.draw_list_section_index].pass_counts[1], surviving_meshlet_count);
            }
        }
        warp_meshlet_instances_offset = WaveBroadcastLaneAt(warp_meshlet_instances_offset, 0);
        global_draws_offsets = WaveBroadcastLaneAt(global_draws_offsets, 0);
        
        if (draw_meshlet)
        {
            const uint meshlet_instance_idx = warp_meshlet_instances_offset + local_survivor_index;
            // When we fail to push back into the meshlet instances we dont need to do anything extra.
            // get_meshlet_instance_from_arg_buckets will make sure that no meshlet indices past the max number are attempted to be drawn.

            if (meshlet_instance_idx < MAX_MESHLET_INSTANCES)
            {
                deref_i(deref(push.attach.meshlet_instances).meshlets, meshlet_instance_idx) = meshlet_instance;
            }
            else
            {
                allocation_failed = true;
                //printf("ERROR: Exceeded max meshlet instances! Entity: %i\n", meshlet_instance.entity_index);
            }

            // Only needed for observer:
            const uint draw_list_element_index = global_draws_offsets + local_survivor_index;
            if (draw_list_element_index < MAX_MESHLET_INSTANCES)
            {
                deref_i(deref(push.attach.meshlet_instances).prepass_draw_lists[push.draw_data.draw_list_section_index].instances, draw_list_element_index) = 
                    (meshlet_instance_idx < MAX_MESHLET_INSTANCES) ? 
                    meshlet_instance_idx : 
                    (~0u);
            }
        }
    }

    // Remove all meshlets that couldnt be allocated.
    draw_meshlet = draw_meshlet && !allocation_failed;

    return MeshletCullWriteoutResult( warp_meshlet_instances_offset );
}

[shader("compute")]
[numthreads(MESHLET_CULL_WORKGROUP_X, 1, 1)]
func entry_compute_meshlet_cull(
    uint3 svtid : SV_DispatchThreadID,
    uint3 svgid : SV_GroupID
)
{
    let push = cull_meshlets_draw_visbuffer_push;
    uint64_t expansion = (push.draw_data.draw_list_section_index == PREPASS_DRAW_LIST_OPAQUE ? push.attach.po2expansion : push.attach.masked_po2expansion);
        
    if (svtid.x == 0)
    {
        uint meshlets_pre_cull = 0;
        uint meshes_post_cull = 0;
        if (push.attach.globals.settings.enable_prefix_sum_work_expansion)
        {
            PrefixSumWorkExpansionBufferHead * prefix_expansion = (PrefixSumWorkExpansionBufferHead *)expansion;
            meshlets_pre_cull = prefix_expansion.expansions_inclusive_prefix_sum[max(1,prefix_expansion.expansion_count)-1];
            meshes_post_cull = prefix_expansion.expansion_count;
        }
        else
        {
            Po2BucketWorkExpansionBufferHead * po2bucket_expansion = (Po2BucketWorkExpansionBufferHead *)expansion;
            for (uint i = 0; i < PO2_WORK_EXPANSION_BUCKET_COUNT; ++i)
            {
                meshlets_pre_cull += po2bucket_expansion.bucket_thread_counts[i];
            }
            meshes_post_cull = po2bucket_expansion.expansion_count;
        }
        if (push.draw_data.pass_index == 0)
        {
            push.attach.globals.readback.first_pass_meshlet_count_pre_cull[push.draw_data.draw_list_section_index] = meshlets_pre_cull;
            push.attach.globals.readback.first_pass_mesh_count_post_cull[push.draw_data.draw_list_section_index] = meshes_post_cull;
        }
        else
        {
            push.attach.globals.readback.second_pass_meshlet_count_pre_cull[push.draw_data.draw_list_section_index] = meshlets_pre_cull;
            push.attach.globals.readback.second_pass_mesh_count_post_cull[push.draw_data.draw_list_section_index] = meshes_post_cull;
        }
    }

    MeshletInstance meshlet_instance;
    bool valid_meshlet = get_meshlet_instance_from_workitem(
        push.attach.globals.settings.enable_prefix_sum_work_expansion,
        expansion,
        push.attach.mesh_instances,
        push.meshes,
        svtid.x,
        meshlet_instance
    );
    
    MeshletCullWriteoutResult cull_result = cull_and_writeout_meshlet(valid_meshlet, meshlet_instance);
}

[shader("amplification")]
[numthreads(MESH_SHADER_WORKGROUP_X, 1, 1)]
func entry_task_meshlet_cull(
    uint3 svtid : SV_DispatchThreadID,
    uint3 svgid : SV_GroupID
)
{
    let push = cull_meshlets_draw_visbuffer_push;

    uint64_t expansion = (push.draw_data.draw_list_section_index == PREPASS_DRAW_LIST_OPAQUE ? push.attach.po2expansion : push.attach.masked_po2expansion);

    if (svtid.x == 0)
    {
        uint meshlets_pre_cull = 0;
        uint meshes_post_cull = 0;
        if (push.attach.globals.settings.enable_prefix_sum_work_expansion)
        {
            PrefixSumWorkExpansionBufferHead * prefix_expansion = (PrefixSumWorkExpansionBufferHead *)expansion;
            meshlets_pre_cull = prefix_expansion.expansions_inclusive_prefix_sum[max(1,prefix_expansion.expansion_count)-1];
            meshes_post_cull = prefix_expansion.expansion_count;
        }
        else
        {
            Po2BucketWorkExpansionBufferHead * po2bucket_expansion = (Po2BucketWorkExpansionBufferHead *)expansion;
            for (uint i = 0; i < PO2_WORK_EXPANSION_BUCKET_COUNT; ++i)
            {
                meshlets_pre_cull += po2bucket_expansion.bucket_thread_counts[i];
            }
            meshes_post_cull = po2bucket_expansion.expansion_count;
        }
        if (push.draw_data.pass_index == 0)
        {
            push.attach.globals.readback.first_pass_meshlet_count_pre_cull[push.draw_data.draw_list_section_index] = meshlets_pre_cull;
            push.attach.globals.readback.first_pass_mesh_count_post_cull[push.draw_data.draw_list_section_index] = meshes_post_cull;
        }
        else
        {
            push.attach.globals.readback.second_pass_meshlet_count_pre_cull[push.draw_data.draw_list_section_index] = meshlets_pre_cull;
            push.attach.globals.readback.second_pass_mesh_count_post_cull[push.draw_data.draw_list_section_index] = meshes_post_cull;
        }
    }

    MeshletInstance meshlet_instance;
    bool valid_meshlet = get_meshlet_instance_from_workitem(
        push.attach.globals.settings.enable_prefix_sum_work_expansion,
        expansion,
        push.attach.mesh_instances,
        push.meshes,
        svtid.x,
        meshlet_instance
    );
    
    MeshletCullWriteoutResult cull_result = cull_and_writeout_meshlet(valid_meshlet, meshlet_instance);

    let surviving_meshlet_count = WaveActiveSum(valid_meshlet ? 1u : 0u);

    CullMeshletsDrawVisbufferPayload payload;
    payload.task_shader_wg_meshlet_args_offset = svgid.x * MESH_SHADER_WORKGROUP_X;
    payload.task_shader_meshlet_instances_offset = cull_result.warp_meshlet_instances_offset;
    payload.task_shader_surviving_meshlets_mask = WaveActiveBallot(valid_meshlet).x;  

    bool cull_backfaces = !GPU_MATERIAL_FALLBACK.alpha_discard_enabled;
    if (valid_meshlet)
    {
        if (meshlet_instance.material_index != INVALID_MANIFEST_INDEX)
        {
            GPUMaterial material = push.materials[meshlet_instance.material_index];
            cull_backfaces = !material.alpha_discard_enabled && !material.double_sided_enabled;
        }
    }
    payload.cull_backfaces_mask = WaveActiveBallot(cull_backfaces).x;

    DispatchMesh(1, surviving_meshlet_count, 1, payload);
}

// Starts counting bits from 0
func wave32_find_nth_set_bit(uint mask, uint n) -> uint
{
    // We assign one bit in the mask to one thread.
    // Each thread calculates bits up to its own bit.
    // If the number of bits before the threads bit is equal to n AND the bit is set, 
    // this means that the threads bit is the nth set bit.
    let lane_mask = ~((~0u) << (WaveGetLaneIndex())); // mask away all bits greater or equal to the tested bit
    let bits_set_up_to_lanebit = countbits(mask & lane_mask);
    let lane_bit_set = (mask & (1u << WaveGetLaneIndex())) != 0;
    let is_nth_bit = (bits_set_up_to_lanebit == n) && lane_bit_set;
    let nth_bit_finder_ballot = WaveActiveBallot(is_nth_bit).x;
    if (nth_bit_finder_ballot == 0) // no bits were set
    {
        return ~0u;
    }
    let nth_set_bit = firstbitlow(nth_bit_finder_ballot);
    return nth_set_bit;
}

func generic_mesh_cull_draw<V: MeshShaderVertexT, P: MeshShaderPrimitiveT>(
    in uint3 svtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in CullMeshletsDrawVisbufferPayload payload)
{
    let push = cull_meshlets_draw_visbuffer_push;
    let wave_lane_index = svtid.x;
    let group_index = svtid.y;
    // The payloads packed survivor indices go from 0-survivor_count.
    // These indices map to the meshlet instance indices.
    let local_meshlet_instance_index = group_index;
    // Meshlet instance indices are the task allocated offset into the meshlet instances + the packed survivor index.
    let meshlet_instance_index = payload.task_shader_meshlet_instances_offset + local_meshlet_instance_index;

    if (meshlet_instance_index >= MAX_MESHLET_INSTANCES)
    {
        SetMeshOutputCounts(0,0);
        return;
    }

    // We need to know the thread index of the task shader that ran for this meshlet.
    // With its thread id we can read the argument buffer just like the task shader did.
    // From the argument we construct the meshlet and any other data that we need.
    let task_shader_local_index = wave32_find_nth_set_bit(payload.task_shader_surviving_meshlets_mask, group_index);
    let task_shader_local_bit = (1u << task_shader_local_index);
    let meshlet_cull_arg_index = payload.task_shader_wg_meshlet_args_offset + task_shader_local_index;
    // The meshlet should always be valid here, 
    // as otherwise the task shader would not have dispatched this mesh shader.
    uint64_t expansion = (push.draw_data.draw_list_section_index == PREPASS_DRAW_LIST_OPAQUE ? (uint64_t)push.attach.po2expansion : (uint64_t)push.attach.masked_po2expansion);
    MeshletInstance meshlet_instance;
    let valid_meshlet = get_meshlet_instance_from_workitem(
        push.attach.globals.settings.enable_prefix_sum_work_expansion,
        expansion,
        push.attach.mesh_instances,
        push.meshes,
        meshlet_cull_arg_index,
        meshlet_instance
    );
    DrawVisbufferPush fake_draw_p;
    fake_draw_p.attach.hiz = push.attach.hiz;
    fake_draw_p.draw_data = push.draw_data; // Can only be the second pass.
    fake_draw_p.attach.globals = push.attach.globals;
    fake_draw_p.attach.meshlet_instances = push.attach.meshlet_instances;
    fake_draw_p.meshes = push.meshes;
    fake_draw_p.entity_combined_transforms = push.entity_combined_transforms;
    fake_draw_p.materials = push.materials;
    
    let cull_backfaces = (payload.cull_backfaces_mask & task_shader_local_bit) != 0;
    const GPUMesh mesh = push.meshes[meshlet_instance.mesh_index];
    if (mesh.mesh_buffer.value == 0) // Unloaded Mesh
    {
        SetMeshOutputCounts(0,0);
        return;
    }
    let cull_hiz_occluded = push.draw_data.pass_index != VISBUF_FIRST_PASS;
    generic_mesh(fake_draw_p, out_indices, out_vertices, out_primitives, mesh, svtid.x, meshlet_instance_index, meshlet_instance, cull_backfaces, cull_hiz_occluded);
}

[outputtopology("triangle")]
[shader("mesh")]
[numthreads(MESH_SHADER_WORKGROUP_X, 1, 1)]
func entry_mesh_meshlet_cull_opaque(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderOpaqueVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<MeshShaderOpaquePrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in payload CullMeshletsDrawVisbufferPayload payload)
{
    generic_mesh_cull_draw(svtid, out_indices, out_vertices, out_primitives, payload);
}

[outputtopology("triangle")]
[shader("mesh")]
[numthreads(MESH_SHADER_WORKGROUP_X, 1, 1)]
func entry_mesh_meshlet_cull_masked(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderMaskVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<MeshShaderMaskPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in payload CullMeshletsDrawVisbufferPayload payload)
{
    generic_mesh_cull_draw(svtid, out_indices, out_vertices, out_primitives, payload);
}

[shader("fragment")]
FragmentOut entry_fragment_meshlet_cull_opaque(in MeshShaderOpaqueVertex vert, in MeshShaderOpaquePrimitive prim)
{
    FragmentOut ret;
    generic_fragment(
        ret,
        uint2(vert.position.xy),
        prim.visibility_id,
        cull_meshlets_draw_visbuffer_push.attach.overdraw_image,
        NoIFragmentExtraData(),
        0
    );
    return ret;
}

[shader("fragment")]
FragmentOut entry_fragment_meshlet_cull_masked(in MeshShaderMaskVertex vert, in MeshShaderMaskPrimitive prim)
{  
    FragmentOut ret;
    generic_fragment(
        ret,
        uint2(vert.position.xy),
        prim.visibility_id,
        cull_meshlets_draw_visbuffer_push.attach.overdraw_image,
        FragmentMaskedData(
            prim.material_index,
            cull_meshlets_draw_visbuffer_push.materials,
            cull_meshlets_draw_visbuffer_push.attach.globals.samplers.linear_repeat_ani,
            vert.uv
        ),
        0
    );
    return ret;
}