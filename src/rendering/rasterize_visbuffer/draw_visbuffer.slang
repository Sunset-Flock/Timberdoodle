#include "daxa/daxa.inl"

#include "draw_visbuffer.inl"

#include "shader_shared/cull_util.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/cull_util.glsl"
#include "shader_lib/pass_logic.glsl"

[[vk::push_constant]] DrawVisbufferPush_WriteCommand write_cmd_p;
[[vk::push_constant]] DrawVisbufferPush draw_p;
[[vk::push_constant]] CullMeshletsDrawVisbufferPush cull_meshlets_draw_visbuffer_push;

[shader("compute")]
[numthreads(1,1,1)]
void entry_write_commands(uint3 dtid : SV_DispatchThreadID)
{
    DrawVisbufferPush_WriteCommand push = write_cmd_p;
    for (uint draw_list_type = 0; draw_list_type < DRAW_LIST_TYPES; ++draw_list_type)
    {
        uint meshlets_to_draw = get_meshlet_draw_count(
            push.uses.globals,
            push.uses.meshlet_instances,
            push.pass,
            draw_list_type);
        if (push.mesh_shader == 1)
        {
            DispatchIndirectStruct command;
            command.x = 1;
            command.y = meshlets_to_draw;
            command.z = 1;
            ((DispatchIndirectStruct*)(push.uses.draw_commands))[draw_list_type] = command;
        }
        else
        {
            DrawIndirectStruct command;
            command.vertex_count = MAX_TRIANGLES_PER_MESHLET * 3;
            command.instance_count = meshlets_to_draw;
            command.first_vertex = 0;
            command.first_instance = 0;
            ((DrawIndirectStruct*)(push.uses.draw_commands))[draw_list_type] = command;
        }
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

struct FragmentOut
{
    [[vk::location(0)]] uint visibility_id;
};

interface VertexT
{
    DECL_GET_SET(float4, position)
    DECL_GET_SET(uint, visibility_id)
    static const uint DRAW_LIST_TYPE;
}

struct OpaqueVertex : VertexT
{
    float4 position : SV_Position;
    [[vk::location(0)]] nointerpolation uint visibility_id;
    IMPL_GET_SET(float4, position)
    IMPL_GET_SET(uint, visibility_id)
    static const uint DRAW_LIST_TYPE = DRAW_LIST_OPAQUE;
};

struct MaskedVertex : VertexT
{
    float4 position : SV_Position;
    [[vk::location(0)]] nointerpolation uint visibility_id;
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] nointerpolation uint material_index;
    IMPL_GET_SET(float4, position)
    IMPL_GET_SET(uint, visibility_id)
    static const uint DRAW_LIST_TYPE = DRAW_LIST_MASK;
}

func generic_vertex<V : VertexT>(
    uint sv_vertex_index,
    uint sv_instance_index) -> V
{
    const uint triangle_corner_index = sv_vertex_index % 3;
    const uint inst_meshlet_index = get_meshlet_instance_index(
        draw_p.uses.globals,
        draw_p.uses.meshlet_instances, 
        draw_p.pass, 
        V::DRAW_LIST_TYPE,
        sv_instance_index);
    const uint triangle_index = sv_vertex_index / 3;
    const MeshletInstance meshlet_inst = deref_i(deref(draw_p.uses.meshlet_instances).meshlets, inst_meshlet_index);
    const GPUMesh mesh = deref_i(draw_p.uses.meshes, meshlet_inst.mesh_index);
    const Meshlet meshlet = deref_i(mesh.meshlets, meshlet_inst.meshlet_index);

    // Discard triangle indices that are out of bounds of the meshlets triangle list.
    if (triangle_index >= meshlet.triangle_count)
    {
        V vertex;
        vertex.set_position(float4(2, 2, 2, 1));
        return vertex;
    }
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref_i(draw_p.uses.meshes, meshlet_inst.mesh_index).micro_indices;
    const uint micro_index = get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + triangle_index * 3 + triangle_corner_index);
    uint vertex_index = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + micro_index);

    vertex_index = min(vertex_index, mesh.vertex_count - 1);
    const daxa_f32vec4 vertex_position = daxa_f32vec4(deref_i(mesh.vertex_positions, vertex_index), 1);
    const daxa_f32mat4x4 view_proj = (draw_p.pass > PASS1_DRAW_POST_CULL) ? deref(draw_p.uses.globals).observer_camera.view_proj : deref(draw_p.uses.globals).camera.view_proj;
    const daxa_f32mat4x3 model_mat4x3 = deref_i(draw_p.uses.entity_combined_transforms, meshlet_inst.entity_index);
    const daxa_f32mat4x4 model_mat = mat_4x3_to_4x4(model_mat4x3);
    const daxa_f32vec4 pos = mul(view_proj, mul(model_mat, vertex_position));

    uint vis_id = 0;
    encode_triangle_id(inst_meshlet_index, triangle_index, vis_id);
    V vertex;
    vertex.set_position(pos);
    vertex.set_visibility_id(vis_id);
    if (V is MaskedVertex)
    {
        MaskedVertex mvertex = reinterpret<MaskedVertex>(vertex);
        mvertex.material_index = meshlet_inst.material_index;
        mvertex.uv = float2(0,0);
        if (as_address(mesh.vertex_uvs) != 0)
        {
            mvertex.uv = deref_i(mesh.vertex_uvs, vertex_index);
        }
        vertex = reinterpret<V>(mvertex);
    }
    return vertex;
}

// SamplerState::get(deref(draw_p.uses.globals).samplers.linear_clamp), 
func generic_fragment<V : VertexT>(V vertex, GPUMaterial* materials, daxa::SamplerId sampler) -> FragmentOut
{
    FragmentOut ret;
    ret.visibility_id = vertex.get_visibility_id();
    if (vertex is MaskedVertex)
    {
        if (WaveIsFirstLane())
        {
            // printf("am masked\n");
        }
    }
    if (V is MaskedVertex && daxa::u64(materials) != 0)
    {
        MaskedVertex mvertex = reinterpret<MaskedVertex>(vertex);
        if (mvertex.material_index != INVALID_MANIFEST_INDEX)
        {
            GPUMaterial material = deref_i(materials, mvertex.material_index);
            if (material.diffuse_texture_id.value != 0 && material.alpha_discard_enabled)
            {
                float alpha = Texture2D<float>::get(material.diffuse_texture_id)
                    .Sample(
                        SamplerState::get(sampler), 
                        mvertex.uv
                    ).a; 
                if (alpha < 0.5f)
                {
                    discard;
                }
            }
        }
    }
    return ret;
}

// --- Opaque ---
[shader("vertex")]
OpaqueVertex entry_vertex_opaque(
    uint sv_vertex_index : SV_VertexID,
    uint sv_instance_index : SV_InstanceID)
{
    return generic_vertex<OpaqueVertex>(
        sv_vertex_index,
        sv_instance_index
    );
}

[shader("fragment")]
FragmentOut entry_fragment_opaque(OpaqueVertex vertex)
{
    return generic_fragment(vertex, (Ptr<GPUMaterial>)(0), daxa::SamplerId(0));
}
// --- Opaque ---


// --- Masked ---
[shader("vertex")]
MaskedVertex entry_vertex_masked(
    uint sv_vertex_index : SV_VertexID,
    uint sv_instance_index : SV_InstanceID)
{
    return generic_vertex<MaskedVertex>(
        sv_vertex_index,
        sv_instance_index
    );
}

[shader("fragment")]
FragmentOut entry_fragment_masked(MaskedVertex vertex)
{
    return generic_fragment(vertex, draw_p.uses.material_manifest, draw_p.uses.globals->samplers.linear_repeat);
}
// --- Masked ---

// Interface:
interface MeshShaderVertexT
{
    DECL_GET_SET(float4, position)
    static const uint DRAW_LIST_TYPE;
}
interface MeshShaderPrimitiveT
{
    DECL_GET_SET(uint, visibility_id)
}


// Opaque:
struct MeshShaderOpaqueVertex : MeshShaderVertexT
{
    float4 position : SV_Position;
    IMPL_GET_SET(float4, position)
    static const uint DRAW_LIST_TYPE = DRAW_LIST_OPAQUE;
};
struct MeshShaderOpaquePrimitive : MeshShaderPrimitiveT
{
    nointerpolation [[vk::location(0)]] uint visibility_id;
    IMPL_GET_SET(uint, visibility_id)
};


// Mask:
struct MeshShaderMaskVertex : MeshShaderVertexT
{
    float4 position : SV_Position;
    [[vk::location(1)]] float2 uv;
    IMPL_GET_SET(float4, position)
    static const uint DRAW_LIST_TYPE = DRAW_LIST_MASK;
}
struct MeshShaderMaskPrimitive : MeshShaderPrimitiveT
{
    nointerpolation [[vk::location(0)]] uint visibility_id;
    nointerpolation [[vk::location(1)]] uint material_index;
    IMPL_GET_SET(uint, visibility_id)
};

func generic_mesh<V: MeshShaderVertexT, P: MeshShaderPrimitiveT>(
    DrawVisbufferPush push,
    in uint3 svtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    uint meshlet_inst_index,
    MeshletInstance meshlet_inst)
{    
    const GPUMesh mesh = deref_i(push.uses.meshes, meshlet_inst.mesh_index);
    const Meshlet meshlet = deref_i(mesh.meshlets, meshlet_inst.meshlet_index);
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref_i(push.uses.meshes, meshlet_inst.mesh_index).micro_indices;
    const daxa_f32mat4x4 view_proj = 
        (push.pass > PASS1_DRAW_POST_CULL) ? 
        deref(push.uses.globals).observer_camera.view_proj : 
        deref(push.uses.globals).camera.view_proj;

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
        const daxa_f32mat4x3 model_mat4x3 = deref_i(push.uses.entity_combined_transforms, meshlet_inst.entity_index);
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
        uint visibility_id;
        encode_triangle_id(meshlet_inst_index, in_meshlet_triangle_index, visibility_id);

        P primitive;
        primitive.set_visibility_id(visibility_id);
        if (P is MeshShaderMaskPrimitive)
        {
            var mprim = reinterpret<MeshShaderMaskPrimitive>(primitive);
            mprim.material_index = meshlet_inst.material_index;
            primitive = reinterpret<P>(mprim);
        }
        out_primitives[in_meshlet_triangle_index] = primitive;
    }
}

func generic_mesh_draw_only<V: MeshShaderVertexT, P: MeshShaderPrimitiveT>(
    in uint3 svtid,
    out OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    out OutputVertices<V, MAX_VERTICES_PER_MESHLET> out_vertices,
    out OutputPrimitives<P, MAX_TRIANGLES_PER_MESHLET> out_primitives)
{
    const uint inst_meshlet_index = get_meshlet_instance_index(
        draw_p.uses.globals,
        draw_p.uses.meshlet_instances, 
        draw_p.pass, 
        V::DRAW_LIST_TYPE,
        svtid.y);
    const uint total_meshlet_count = 
        deref(draw_p.uses.meshlet_instances).draw_lists[0].first_count + 
        deref(draw_p.uses.meshlet_instances).draw_lists[0].second_count;
    const MeshletInstance meshlet_inst = deref_i(deref(draw_p.uses.meshlet_instances).meshlets, inst_meshlet_index);
    generic_mesh(draw_p, svtid, out_indices, out_vertices, out_primitives, inst_meshlet_index, meshlet_inst);
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

[shader("fragment")]
FragmentOut entry_mesh_fragment_opaque(in MeshShaderOpaqueVertex vert, in MeshShaderOpaquePrimitive prim)
{
    OpaqueVertex o_vert;
    o_vert.position = vert.position;
    o_vert.visibility_id = prim.visibility_id;
    return generic_fragment(o_vert, Ptr<GPUMaterial>(0), daxa::SamplerId(0));
}
// --- Mesh shader opaque ---


// --- Mesh shader mask ---

[outputtopology("triangle")]
[numthreads(MESH_SHADER_WORKGROUP_X,1,1)]
[shader("mesh")]
func entry_mesh_mask(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderMaskVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<MeshShaderMaskPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives)
{
    generic_mesh_draw_only(svtid, out_indices, out_vertices, out_primitives);
}

[shader("fragment")]
FragmentOut entry_mesh_fragment_mask(in MeshShaderMaskVertex vert, in MeshShaderMaskPrimitive prim)
{
    MaskedVertex o_vert;
    o_vert.position = vert.position;
    o_vert.visibility_id = prim.visibility_id;
    o_vert.uv = vert.uv;
    o_vert.material_index = prim.material_index;
    return generic_fragment(o_vert, draw_p.uses.material_manifest, draw_p.uses.globals->samplers.linear_repeat);
}

// --- Mesh shader mask ---

// --- Cull Meshlets Draw Visbuffer

struct CullMeshletsDrawVisbufferPayload
{
    uint task_shader_wg_meshlet_args_offset;
    uint task_shader_meshlet_instances_offset;
    uint task_shader_surviving_meshlets_mask;
};

[shader("amplification")]
[numthreads(MESH_SHADER_WORKGROUP_X, 1, 1)]
func entry_task_cull_draw_opaque_and_mask(
    uint3 svtid : SV_DispatchThreadID,
    uint3 svgid : SV_GroupID
)
{
    let push = cull_meshlets_draw_visbuffer_push;
    MeshletInstance instanced_meshlet;
    const bool valid_meshlet = get_meshlet_instance_from_arg_buckets(
        svtid.x,
        push.bucket_index,
        push.uses.meshlets_cull_arg_buckets,
        push.draw_list_type,
        instanced_meshlet);
    bool draw_meshlet = valid_meshlet;
#if ENABLE_MESHLET_CULLING == 1
    // We still continue to run the task shader even with invalid meshlets.
    // We simple set the occluded value to true for these invalida meshlets.
    // This is done so that the following WaveOps are well formed and have all threads active. 
    if (valid_meshlet)
    {
        draw_meshlet = draw_meshlet && !is_meshlet_occluded(
            deref(push.uses.globals).camera,
            instanced_meshlet,
            push.uses.first_pass_meshlets_bitfield_offsets,
            push.uses.first_pass_meshlets_bitfield_arena,
            push.uses.entity_combined_transforms,
            push.uses.meshes,
            push.uses.hiz);
    }
#endif
    CullMeshletsDrawVisbufferPayload payload;
    payload.task_shader_wg_meshlet_args_offset = svgid.x * MESH_SHADER_WORKGROUP_X;
    payload.task_shader_surviving_meshlets_mask = WaveActiveBallot(draw_meshlet).x;  
    let surviving_meshlet_count = WaveActiveSum(draw_meshlet ? 1u : 0u);
    // When not occluded, this value determines the new packed index for each thread in the wave:
    let local_survivor_index = WavePrefixSum(draw_meshlet ? 1u : 0u);
    uint global_draws_offsets;
    if (WaveIsFirstLane())
    {
        payload.task_shader_meshlet_instances_offset = 
            push.uses.meshlet_instances->first_count + 
            atomicAdd(push.uses.meshlet_instances->second_count, surviving_meshlet_count);
        global_draws_offsets = 
            push.uses.meshlet_instances->draw_lists[push.draw_list_type].first_count + 
            atomicAdd(push.uses.meshlet_instances->draw_lists[push.draw_list_type].second_count, surviving_meshlet_count);
    }
    payload.task_shader_meshlet_instances_offset = WaveBroadcastLaneAt(payload.task_shader_meshlet_instances_offset, 0);
    global_draws_offsets = WaveBroadcastLaneAt(global_draws_offsets, 0);
    
    if (draw_meshlet)
    {
        const uint meshlet_instance_idx = payload.task_shader_meshlet_instances_offset + local_survivor_index;
        deref_i(deref(push.uses.meshlet_instances).meshlets, meshlet_instance_idx) = instanced_meshlet;

        // Only needed for observer:
        const uint draw_list_element_index = global_draws_offsets + local_survivor_index;
        deref_i(deref(push.uses.meshlet_instances).draw_lists[push.draw_list_type].instances, draw_list_element_index) = meshlet_instance_idx;
    }

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
    uint first_set_bit = WaveActiveMin((mask.x & wave_lane_bit_mask) != 0 ? WaveGetLaneIndex() : 100);
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
        push.uses.meshlets_cull_arg_buckets, 
        push.draw_list_type, 
        meshlet_inst);
    DrawVisbufferPush fake_draw_p;
    fake_draw_p.pass = PASS1_DRAW_POST_CULL; // Can only be the second pass.
    fake_draw_p.uses.globals = push.uses.globals;
    fake_draw_p.uses.meshlet_instances = push.uses.meshlet_instances;
    fake_draw_p.uses.meshes = push.uses.meshes;
    fake_draw_p.uses.entity_combined_transforms = push.uses.entity_combined_transforms;
    fake_draw_p.uses.material_manifest = push.uses.material_manifest;
    

    // SetMeshOutputCounts(0,0);
    generic_mesh(fake_draw_p, svtid, out_indices, out_vertices, out_primitives, meshlet_instance_index, meshlet_inst);
}

[outputtopology("triangle")]
[shader("mesh")]
[numthreads(MESH_SHADER_WORKGROUP_X, 1, 1)]
func entry_mesh_cull_draw_opaque(
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
func entry_mesh_cull_draw_mask(
    in uint3 svtid : SV_DispatchThreadID,
    OutputIndices<uint3, MAX_TRIANGLES_PER_MESHLET> out_indices,
    OutputVertices<MeshShaderMaskVertex, MAX_VERTICES_PER_MESHLET> out_vertices,
    OutputPrimitives<MeshShaderMaskPrimitive, MAX_TRIANGLES_PER_MESHLET> out_primitives,
    in payload CullMeshletsDrawVisbufferPayload payload)
{
    generic_mesh_cull_draw(svtid, out_indices, out_vertices, out_primitives, payload);
}

[shader("fragment")]
FragmentOut entry_mesh_fragment_cull_draw_opaque(in MeshShaderOpaqueVertex vert, in MeshShaderOpaquePrimitive prim)
{
    OpaqueVertex o_vert;
    o_vert.position = vert.position;
    o_vert.visibility_id = prim.visibility_id;
    return generic_fragment(o_vert, Ptr<GPUMaterial>(0), daxa::SamplerId(0));
}

[shader("fragment")]
FragmentOut entry_mesh_fragment_cull_draw_mask(in MeshShaderMaskVertex vert, in MeshShaderMaskPrimitive prim)
{
    MaskedVertex o_vert;
    o_vert.position = vert.position;
    o_vert.visibility_id = prim.visibility_id;
    o_vert.uv = vert.uv;
    o_vert.material_index = prim.material_index;
    return generic_fragment(
        o_vert, 
        cull_meshlets_draw_visbuffer_push.uses.material_manifest, 
        cull_meshlets_draw_visbuffer_push.uses.globals->samplers.linear_repeat
    );

    // OpaqueVertex o_vert;
    // o_vert.position = vert.position;
    // o_vert.visibility_id = prim.visibility_id;
    // return generic_fragment(o_vert, Ptr<GPUMaterial>(0), daxa::SamplerId(0));
}


// --- Cull Meshlets Draw Visbuffer