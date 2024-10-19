#include "daxa/daxa.inl"

#include "draw_visbuffer.inl"

[[vk::push_constant]] DrawVisbufferPush_WriteCommand write_cmd_p;
[[vk::push_constant]] SplitAtomicVisbufferPush split_atimic_visbuffer_p;
[[vk::push_constant]] DrawVisbufferPush draw_p;
[[vk::push_constant]] CullMeshletsDrawVisbufferPush cull_meshlets_draw_visbuffer_push;

#define GLOBALS cull_meshlets_draw_visbuffer_push.attach.globals

#include "shader_shared/cull_util.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/cull_util.hlsl"
#include "shader_lib/pass_logic.glsl"
#include "shader_lib/po2_expansion.hlsl"
#include "shader_lib/misc.hlsl"

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
            command.x = 1;
            command.y = meshlets_to_draw;
            command.z = 1;
            ((DispatchIndirectStruct*)(push.attach.draw_commands))[draw_list_type] = command;
    }
}

[shader("compute")]
[numthreads(SPLIT_ATOMIC_VISBUFFER_X, SPLIT_ATOMIC_VISBUFFER_Y, 1)]
void entry_split_atomic_visbuffer(uint3 dtid : SV_DispatchThreadID)
{
    let push = split_atimic_visbuffer_p;
    if (any(dtid.xy >= split_atimic_visbuffer_p.size))
    {
        return;
    }

    daxa::u64 visdepth = tex_rw_u64_table[push.attach.atomic_visbuffer.index()][dtid.xy];
    RWTexture2D<uint>::get(push.attach.visbuffer)[dtid.xy] = uint(visdepth);
    RWTexture2D<float>::get(push.attach.depth)[dtid.xy] = asfloat(uint(visdepth >> 32));
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

func generic_fragment<ExtraData : IFragmentExtraData, FragOutT : IFragmentOut>(out FragOutT ret, uint2 index, uint vis_id, daxa::ImageViewId overdraw_image, ExtraData extra, daxa::ImageViewId atomic_visbuffer, float depth)
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
                alpha = Texture2D<float4>::get(material.diffuse_texture_id)
                    .Sample( SamplerState::get(masked_data.sampler), masked_data.uv).a; 
            }
            else if (material.diffuse_texture_id.value != 0 && material.alpha_discard_enabled)
            {
                alpha = Texture2D<float4>::get(material.diffuse_texture_id)
                    .Sample( SamplerState::get(masked_data.sampler), masked_data.uv).a; 
            }
            // const float max_obj_space_deriv_len = max(length(ddx(mvertex.object_space_position)), length(ddy(mvertex.object_space_position)));
            // const float threshold = compute_hashed_alpha_threshold(mvertex.object_space_position, max_obj_space_deriv_len, 0.3);
            // if (alpha < clamp(threshold, 0.001, 1.0)) // { discard; }
            if(alpha < 0.5) { discard; }
        }
    }
    if (overdraw_image.value != 0)
    {
        uint prev_val;
        InterlockedAdd(RWTexture2D_utable[overdraw_image.index()][index], 1, prev_val);
    }
    if (atomic_visbuffer.value != 0)
    {
        daxa::u64 visdepth = (daxa::u64(asuint(depth)) << 32) | daxa::u64(vis_id);
        AtomicMaxU64(tex_rw_u64_table[atomic_visbuffer.index()][index], visdepth);
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
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] float3 object_space_position;
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

func generic_mesh<V: MeshShaderVertexT, P: MeshShaderPrimitiveT>(
    DrawVisbufferPush push,
    in uint3 svtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    uint meshlet_inst_index,
    MeshletInstance meshlet_inst,
    bool cull_backfaces)
{            
    const GPUMesh mesh = deref_i(push.attach.meshes, meshlet_inst.mesh_index);
    if (mesh.mesh_buffer.value == 0) // Unloaded Mesh
    {
        return;
    }
    const Meshlet meshlet = deref_i(mesh.meshlets, meshlet_inst.meshlet_index);
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref_i(push.attach.meshes, meshlet_inst.mesh_index).micro_indices;
    const daxa_f32mat4x4 view_proj = 
        (push.pass > PASS1_DRAW_POST_CULL) ? 
        deref(push.attach.globals).observer_camera.view_proj : 
        deref(push.attach.globals).camera.view_proj;

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);
    if (meshlet_inst_index >= MAX_MESHLET_INSTANCES)
    {
        printf("fuck\n");
    }

    float4 local_clip_vertices[2];
    float3 local_ndc_vertices[2];

    const daxa_f32mat4x3 model_mat4x3 = deref_i(push.attach.entity_combined_transforms, meshlet_inst.entity_index);
    const daxa_f32mat4x4 model_mat = mat_4x3_to_4x4(model_mat4x3);
    for (uint l = 0; l < 2; ++l)
    {
        uint vertex_offset = MESH_SHADER_WORKGROUP_X * l;
        const uint in_meshlet_vertex_index = svtid.x + vertex_offset;
        if (in_meshlet_vertex_index >= meshlet.vertex_count) continue;

        const uint in_mesh_vertex_index = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + in_meshlet_vertex_index);
        if (in_mesh_vertex_index >= mesh.vertex_count)
        {
            /// TODO: ASSERT HERE. 
            continue;
        }
        const daxa_f32vec4 vertex_position = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index), 1);
        const daxa_f32vec4 pos = mul(view_proj, mul(model_mat, vertex_position));

        V vertex;
        local_clip_vertices[l] = pos;
        vertex.set_position(pos);
        if (V is MeshShaderMaskVertex)
        {
            var mvertex = reinterpret<MeshShaderMaskVertex>(vertex);
            mvertex.uv = float2(0,0);
            mvertex.object_space_position = vertex_position.xyz;
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
        uint3 tri_in_meshlet_vertex_indices = uint3(0,0,0);
        if (in_meshlet_triangle_index < meshlet.triangle_count)
        {
            tri_in_meshlet_vertex_indices = uint3(
                get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 0),
                get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 1),
                get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 2)
            );
        }
        float4[3] tri_vert_clip_positions = float4[3](
            shuffle_arr2(local_clip_vertices, tri_in_meshlet_vertex_indices[0]),
            shuffle_arr2(local_clip_vertices, tri_in_meshlet_vertex_indices[1]),
            shuffle_arr2(local_clip_vertices, tri_in_meshlet_vertex_indices[2])
        );

        if (in_meshlet_triangle_index < meshlet.triangle_count)
        {
            // From: https://zeux.io/2023/04/28/triangle-backface-culling/#fnref:3
            bool cull_primitive = false;

            // Observer culls triangles from the perspective of the main camera.
            if (push.pass >= PASS2_OBSERVER_DRAW_VISIBLE_LAST_FRAME)
            {        
                for (uint c = 0; c < 3; ++c)
                {
                    const uint in_mesh_vertex_index = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + tri_in_meshlet_vertex_indices[c]);
                    const daxa_f32vec4 vertex_position = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index), 1);
                    let main_camera_view_proj = push.attach.globals.camera.view_proj;
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

                    if (push.attach.hiz.value != 0 && !cull_primitive)
                    {
                        let cull_hiz_occluded = is_triangle_hiz_occluded(
                            push.attach.globals.camera,
                            tri_vert_ndc_positions,
                            push.attach.globals.settings.next_lower_po2_render_target_size,
                            push.attach.hiz);
                        cull_primitive = cull_hiz_occluded;
                    }
                }
            }
            
            P primitive;
            primitive.set_cull_primitive(cull_primitive);
            if (!cull_primitive)
            {
                uint visibility_id = TRIANGLE_ID_MAKE(meshlet_inst_index, in_meshlet_triangle_index);
                primitive.set_visibility_id(cull_primitive ? ~0u : visibility_id);
                if (P is MeshShaderMaskPrimitive)
                {
                    var mprim = reinterpret<MeshShaderMaskPrimitive>(primitive);
                    mprim.material_index = meshlet_inst.material_index;
                    primitive = reinterpret<P>(mprim);
                }
                out_indices[in_meshlet_triangle_index] = tri_in_meshlet_vertex_indices;
            }
            out_primitives[in_meshlet_triangle_index] = primitive;
        }
    }
}

func generic_mesh_draw_only<V: MeshShaderVertexT, P: MeshShaderPrimitiveT>(
    in uint3 svtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives)
{
    const uint inst_meshlet_index = get_meshlet_instance_index(
        draw_p.attach.globals,
        draw_p.attach.meshlet_instances, 
        draw_p.pass, 
        V::PREPASS_DRAW_LIST_TYPE,
        svtid.y);
    if (inst_meshlet_index >= MAX_MESHLET_INSTANCES)
    {
        SetMeshOutputCounts(0,0);
        return;
    }
    const uint total_meshlet_count = 
        deref(draw_p.attach.meshlet_instances).prepass_draw_lists[0].first_count + 
        deref(draw_p.attach.meshlet_instances).prepass_draw_lists[0].second_count;
    const MeshletInstance meshlet_inst = deref_i(deref(draw_p.attach.meshlet_instances).meshlets, inst_meshlet_index);

    bool cull_backfaces = false;
    if (meshlet_inst.material_index != INVALID_MANIFEST_INDEX)
    {
        GPUMaterial material = draw_p.attach.material_manifest[meshlet_inst.material_index];
        cull_backfaces = !material.alpha_discard_enabled;
    }
    generic_mesh(draw_p, svtid, out_indices, out_vertices, out_primitives, inst_meshlet_index, meshlet_inst, cull_backfaces);
}

// --- Mesh shader opaque ---
[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func entry_mesh_opaque(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderOpaqueVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<MeshShaderOpaquePrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives)
{
    generic_mesh_draw_only(svtid, out_indices, out_vertices, out_primitives);
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
        daxa::ImageViewId(0),
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
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderMaskVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<MeshShaderMaskPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives)
{
    generic_mesh_draw_only(svtid, out_indices, out_vertices, out_primitives);
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
            draw_p.attach.material_manifest,
            draw_p.attach.globals->samplers.linear_repeat,
            // draw_p.attach.globals->samplers.nearest_repeat,
            vert.uv
        ),
        daxa::ImageViewId(0),
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
    uint enable_backface_culling;
};

struct MeshInstanceWorkItems : IPo2SrcWorkItems
{
    MeshInstance * mesh_instances;
    GPUMesh * meshes;
    func get_itemcount(uint src_item_index) -> uint
    {
        let mesh_instance = mesh_instances[src_item_index];
        let mesh = meshes[mesh_instance.mesh_index];
        return mesh.meshlet_count;
    }
}

bool get_meshlet_instance_from_workitem(
    Po2WorkExpansionBufferHead * po2expansion,
    MeshInstancesBufferHead * mesh_instances,
    GPUMesh * meshes,
    uint bucket_index,
    uint thread_index, 
    out MeshletInstance meshlet_instance)
{
    Po2ExpandedWorkItem workitem;
    let valid_meshlet = get_expanded_work_item(po2expansion, MeshInstanceWorkItems(mesh_instances.instances, meshes), thread_index, bucket_index, workitem);
    if (valid_meshlet)
    {
        MeshInstance mesh_instance = mesh_instances.instances[workitem.src_work_item_index];
        GPUMesh mesh = meshes[mesh_instance.mesh_index];    
        if (mesh.mesh_buffer.value == 0) // Unloaded Mesh
        {
            return false;
        }
        meshlet_instance.entity_index = mesh_instance.entity_index;
        meshlet_instance.in_mesh_group_index = mesh_instance.in_mesh_group_index;
        meshlet_instance.material_index = mesh.material_index;
        meshlet_instance.mesh_index = mesh_instance.mesh_index;
        meshlet_instance.meshlet_index = workitem.dst_work_item_index;
        meshlet_instance.mesh_instance_index = workitem.src_work_item_index;
    }
    return valid_meshlet;
}

[shader("amplification")]
[numthreads(MESH_SHADER_WORKGROUP_X, 1, 1)]
func entry_task_meshlet_cull(
    uint3 svtid : SV_DispatchThreadID,
    uint3 svgid : SV_GroupID
)
{
    let push = cull_meshlets_draw_visbuffer_push;

    Po2WorkExpansionBufferHead * po2expansion = (Po2WorkExpansionBufferHead *)(push.draw_list_type == PREPASS_DRAW_LIST_OPAQUE ? (uint64_t)push.attach.po2expansion : (uint64_t)push.attach.masked_po2expansion);
    MeshletInstance instanced_meshlet;
    bool valid_meshlet = get_meshlet_instance_from_workitem(
        po2expansion,
        push.attach.mesh_instances,
        push.attach.meshes,
        push.bucket_index,
        svtid.x,
        instanced_meshlet
    );
    
    if (valid_meshlet)
    {
        GPUMesh mesh_data = deref_i(push.attach.meshes, instanced_meshlet.mesh_index);
        valid_meshlet = valid_meshlet && mesh_data.mesh_buffer.value != 0; // Check if mesh is loaded.
        valid_meshlet = valid_meshlet && (instanced_meshlet.meshlet_index < mesh_data.meshlet_count);
    }
    if (valid_meshlet)
    {
        valid_meshlet = valid_meshlet && !is_meshlet_drawn_in_first_pass( instanced_meshlet, push.attach.first_pass_meshlets_bitfield_arena );
    }

    bool draw_meshlet = valid_meshlet;
    // We still continue to run the task shader even with invalid meshlets.
    // We simple set the occluded value to true for these invalida meshlets.
    // This is done so that the following WaveOps are well formed and have all threads active. 
    if (valid_meshlet && push.attach.globals.settings.enable_meshlet_cull)
    {
        draw_meshlet = draw_meshlet && !is_meshlet_occluded(
            push.attach.globals.debug,
            deref(push.attach.globals).camera,
            instanced_meshlet,
            push.attach.entity_combined_transforms,
            push.attach.meshes,
            push.attach.globals->settings.next_lower_po2_render_target_size,
            push.attach.hiz);
    }
    CullMeshletsDrawVisbufferPayload payload;
    payload.task_shader_wg_meshlet_args_offset = svgid.x * MESH_SHADER_WORKGROUP_X;
    payload.task_shader_surviving_meshlets_mask = WaveActiveBallot(draw_meshlet).x;  
    uint surviving_meshlet_count = WaveActiveSum(draw_meshlet ? 1u : 0u);
    // When not occluded, this value determines the new packed index for each thread in the wave:
    let local_survivor_index = WavePrefixSum(draw_meshlet ? 1u : 0u);
    uint global_draws_offsets;
    if (WaveIsFirstLane())
    {
        payload.task_shader_meshlet_instances_offset = 
            push.attach.meshlet_instances->first_count + 
            atomicAdd(push.attach.meshlet_instances->second_count, surviving_meshlet_count);
        global_draws_offsets = 
            push.attach.meshlet_instances->prepass_draw_lists[push.draw_list_type].first_count + 
            atomicAdd(push.attach.meshlet_instances->prepass_draw_lists[push.draw_list_type].second_count, surviving_meshlet_count);
    }
    payload.task_shader_meshlet_instances_offset = WaveBroadcastLaneAt(payload.task_shader_meshlet_instances_offset, 0);
    global_draws_offsets = WaveBroadcastLaneAt(global_draws_offsets, 0);
    
    bool allocation_failed = false;
    if (draw_meshlet)
    {
        const uint meshlet_instance_idx = payload.task_shader_meshlet_instances_offset + local_survivor_index;
        // When we fail to push back into the meshlet instances we dont need to do anything extra.
        // get_meshlet_instance_from_arg_buckets will make sure that no meshlet indices past the max number are attempted to be drawn.

        if (meshlet_instance_idx < MAX_MESHLET_INSTANCES)
        {
            deref_i(deref(push.attach.meshlet_instances).meshlets, meshlet_instance_idx) = instanced_meshlet;
        }
        else
        {
            allocation_failed = true;
            //printf("ERROR: Exceeded max meshlet instances! Entity: %i\n", instanced_meshlet.entity_index);
        }

        // Only needed for observer:
        const uint draw_list_element_index = global_draws_offsets + local_survivor_index;
        if (draw_list_element_index < MAX_MESHLET_INSTANCES)
        {
            deref_i(deref(push.attach.meshlet_instances).prepass_draw_lists[push.draw_list_type].instances, draw_list_element_index) = 
                (meshlet_instance_idx < MAX_MESHLET_INSTANCES) ? 
                meshlet_instance_idx : 
                (~0u);
        }
    }


    // Remove all meshlets that couldnt be allocated.
    draw_meshlet = draw_meshlet && !allocation_failed;
    if (WaveActiveAnyTrue(allocation_failed))
    {
        payload.task_shader_surviving_meshlets_mask = WaveActiveBallot(draw_meshlet).x;  
        surviving_meshlet_count = WaveActiveSum(draw_meshlet ? 1u : 0u);
    }

    bool enable_backface_culling = false;
    if (valid_meshlet)
    {
        if (instanced_meshlet.material_index != INVALID_MANIFEST_INDEX)
        {
            GPUMaterial material = push.attach.material_manifest[instanced_meshlet.material_index];
            enable_backface_culling = !material.alpha_discard_enabled;
        }
    }
    payload.enable_backface_culling = WaveActiveBallot(enable_backface_culling).x;

    DispatchMesh(1, surviving_meshlet_count, 1, payload);
}

func wave32_find_nth_set_bit(uint mask, uint bit) -> uint
{
    // Each thread tests a bit in the mask.
    // The nth bit is the nth thread.
    let wave_lane_bit_mask = 1u << WaveGetLaneIndex();
    let is_nth_bit_set = ((mask & wave_lane_bit_mask) != 0) ? 1u : 0u;
    let set_bits_prefix_sum = WavePrefixSum(is_nth_bit_set) + is_nth_bit_set;

    let does_nth_bit_match_group = set_bits_prefix_sum == (bit + 1);
    uint ret;
    uint4 mask = WaveActiveBallot(does_nth_bit_match_group);
    uint first_set_bit = WaveActiveMin((mask.x & wave_lane_bit_mask) != 0 ? WaveGetLaneIndex() : ~0u);
    return first_set_bit;
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
    Po2WorkExpansionBufferHead * po2expansion = (Po2WorkExpansionBufferHead *)(push.draw_list_type == PREPASS_DRAW_LIST_OPAQUE ? (uint64_t)push.attach.po2expansion : (uint64_t)push.attach.masked_po2expansion);
    MeshletInstance meshlet_inst;
    let valid_meshlet = get_meshlet_instance_from_workitem(
        po2expansion,
        push.attach.mesh_instances,
        push.attach.meshes,
        push.bucket_index,
        meshlet_cull_arg_index,
        meshlet_inst
    );
    DrawVisbufferPush fake_draw_p;
    fake_draw_p.attach.hiz = push.attach.hiz;
    fake_draw_p.pass = PASS1_DRAW_POST_CULL; // Can only be the second pass.
    fake_draw_p.attach.globals = push.attach.globals;
    fake_draw_p.attach.meshlet_instances = push.attach.meshlet_instances;
    fake_draw_p.attach.meshes = push.attach.meshes;
    fake_draw_p.attach.entity_combined_transforms = push.attach.entity_combined_transforms;
    fake_draw_p.attach.material_manifest = push.attach.material_manifest;
    
    let cull_backfaces = (payload.enable_backface_culling & task_shader_local_bit) != 0;
    generic_mesh(fake_draw_p, svtid, out_indices, out_vertices, out_primitives, meshlet_instance_index, meshlet_inst, cull_backfaces);
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
        daxa::ImageViewId(0),
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
            cull_meshlets_draw_visbuffer_push.attach.material_manifest,
            cull_meshlets_draw_visbuffer_push.attach.globals->samplers.linear_repeat,
            // cull_meshlets_draw_visbuffer_push.attach.globals->samplers.nearest_repeat,
            vert.uv
        ),
        daxa::ImageViewId(0),
        0
    );
    return ret;
}

/// --- Atomic Visbuffer Begin ---

struct DummyFragmentOut : IFragmentOut
{
    func set_visibility_id(uint v) {}
    func get_visibility_id() -> uint{ return 0; }
};

[shader("fragment")]
void entry_fragment_opaque_atomicvis(in MeshShaderOpaqueVertex vert, in MeshShaderOpaquePrimitive prim)
{
    DummyFragmentOut ret;
    generic_fragment(
        ret,
        uint2(vert.position.xy),
        prim.visibility_id,
        draw_p.attach.overdraw_image,
        NoIFragmentExtraData(),
        draw_p.attach.atomic_visbuffer,
        vert.get_position().z
    );
}

[shader("fragment")]
void entry_fragment_masked_atomicvis(in MeshShaderMaskVertex vert, in MeshShaderMaskPrimitive prim)
{
    DummyFragmentOut ret;
    generic_fragment(
        ret,
        uint2(vert.position.xy),
        prim.visibility_id,
        draw_p.attach.overdraw_image,
        FragmentMaskedData(
            prim.material_index,
            draw_p.attach.material_manifest,
            draw_p.attach.globals->samplers.linear_repeat,
            // draw_p.attach.globals->samplers.nearest_repeat,
            vert.uv
        ),
        draw_p.attach.atomic_visbuffer,
        vert.get_position().z
    );
}

[shader("fragment")]
void entry_fragment_meshlet_cull_opaque_atomicvis(in MeshShaderOpaqueVertex vert, in MeshShaderOpaquePrimitive prim)
{
    DummyFragmentOut ret;
    generic_fragment(
        ret,
        uint2(vert.position.xy),
        prim.visibility_id,
        cull_meshlets_draw_visbuffer_push.attach.overdraw_image,
        NoIFragmentExtraData(),
        cull_meshlets_draw_visbuffer_push.attach.atomic_visbuffer,
        vert.get_position().z
    );
}

[shader("fragment")]
void entry_fragment_meshlet_cull_masked_atomicvis(in MeshShaderMaskVertex vert, in MeshShaderMaskPrimitive prim)
{  
    DummyFragmentOut ret;
    generic_fragment(
        ret,
        uint2(vert.position.xy),
        prim.visibility_id,
        cull_meshlets_draw_visbuffer_push.attach.overdraw_image,
        FragmentMaskedData(
            prim.material_index,
            cull_meshlets_draw_visbuffer_push.attach.material_manifest,
            cull_meshlets_draw_visbuffer_push.attach.globals->samplers.linear_repeat,
            // cull_meshlets_draw_visbuffer_push.attach.globals->samplers.nearest_repeat,
            vert.uv
        ),
        cull_meshlets_draw_visbuffer_push.attach.atomic_visbuffer,
        vert.get_position().z
    );
}

/// --- Atomic Visbuffer End ---