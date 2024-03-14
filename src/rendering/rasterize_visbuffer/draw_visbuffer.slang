#include "daxa/daxa.inl"
#include "draw_visbuffer.inl"
#include "shader_shared/cull_util.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/cull_util.glsl"
#include "shader_lib/pass_logic.glsl"

[[vk::push_constant]] DrawVisbufferPush_WriteCommand write_cmd_p;
[[vk::push_constant]] DrawVisbufferPush draw_p;

#if defined(OPAQUE)
#define OPAQUE_DRAW_LIST_TYPE OPAQUE_DRAW_LIST_SOLID
#elif defined(DISCARD)
#define OPAQUE_DRAW_LIST_TYPE OPAQUE_DRAW_LIST_MASKED
#else
#define OPAQUE_DRAW_LIST_TYPE 0
#endif

import core;

[shader("compute")]
[numthreads(1,1,1)]
void entry_write_commands(uint3 dtid : SV_DispatchThreadID)
{
    DrawVisbufferPush_WriteCommand push = write_cmd_p;
    for (uint draw_list_type = 0; draw_list_type < OPAQUE_DRAW_LIST_COUNT; ++draw_list_type)
    {
        uint meshlets_to_draw = get_meshlet_draw_count(
            push.uses.globals,
            push.uses.meshlet_instances,
            push.pass,
            draw_list_type);
        if (push.mesh_shader == 1)
        {
            DispatchIndirectStruct command;
            command.x = meshlets_to_draw;
            command.y = 1;
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

struct VertexOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] nointerpolation uint triangle_id : TRI_ID;
    #if OPAQUE_DRAW_LIST_TYPE == OPAQUE_DRAW_LIST_MASKED
        [[vk::location(1)]] float2 uv : UV;
        [[vk::location(2)]] nointerpolation uint material_index : MATERIAL_ID;
    #endif // #if OPAQUE_DRAW_LIST_TYPE == OPAQUE_DRAW_LIST_MASKED
};

[shader("vertex")]
VertexOut entry_vertex(
    uint sv_vertex_index : SV_VertexID,
    uint sv_instance_index : SV_InstanceID)
{
    const uint triangle_corner_index = sv_vertex_index % 3;
    const uint inst_meshlet_index = get_meshlet_instance_index(
        draw_p.uses.globals,
        draw_p.uses.meshlet_instances, 
        draw_p.pass, 
        OPAQUE_DRAW_LIST_TYPE,
        sv_instance_index);
    const uint triangle_index = sv_vertex_index / 3;
    const MeshletInstance meshlet_inst = deref_i(deref(draw_p.uses.meshlet_instances).meshlets, inst_meshlet_index);
    const GPUMesh mesh = deref_i(draw_p.uses.meshes, meshlet_inst.mesh_index);
    const Meshlet meshlet = deref_i(mesh.meshlets, meshlet_inst.meshlet_index);

    // Discard triangle indices that are out of bounds of the meshlets triangle list.
    if (triangle_index >= meshlet.triangle_count)
    {
        VertexOut frag_in;
        frag_in.position = float4(2, 2, 2, 1);
        return frag_in;
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

    VertexOut frag_in;
    frag_in.position = pos;
    encode_triangle_id(inst_meshlet_index, triangle_index, frag_in.triangle_id);
    #if OPAQUE_DRAW_LIST_TYPE == OPAQUE_DRAW_LIST_MASKED
        frag_in.material_index = meshlet_inst.material_index;
        frag_in.uv = float2(0,0);
        if (as_address(mesh.vertex_uvs) != 0)
        {
            frag_in.uv = deref_i(mesh.vertex_uvs, vertex_index);
        }
    #endif // #if OPAQUE_DRAW_LIST_TYPE == OPAQUE_DRAW_LIST_MASKED
    return frag_in;
}

struct FragmentOut
{
    [[vk::location(0)]] uint triangle_id;
};

[shader("fragment")]
FragmentOut entry_fragment(VertexOut frag_in)
{
    FragmentOut frag_out;
    #if OPAQUE_DRAW_LIST_TYPE == OPAQUE_DRAW_LIST_MASKED
        GPUMaterial material = deref_i(draw_p.uses.material_manifest, frag_in.material_index);
        if (material.diffuse_texture_id.value != 0 && material.alpha_discard_enabled)
        {
            float alpha = 
                daxa_Texture2D(float4, material.diffuse_texture_id)
                .SampleLevel(daxa_SamplerState(deref(draw_p.uses.globals).samplers.linear_clamp), frag_in.uv, 0).a; 
            if (alpha < 0.5f)
            {
                discard;
            }
        }
    #endif // #if OPAQUE_DRAW_LIST_TYPE == OPAQUE_DRAW_LIST_MASKED
    frag_out.triangle_id = frag_in.triangle_id;
    return frag_out;
}

struct Push
{
    float3x4* ptr;
};

[[vk::push_constant]] Push push;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{    
    // This matrix is in memry column major. Slang respects this here and load it properly!
    float3x4 correctly_read_matrix = *push.ptr;
    printf("(%f,%f,%f,%f)\n(%f,%f,%f,%f)\n",
        correctly_read_matrix[0][0], correctly_read_matrix[0][1], correctly_read_matrix[0][2], correctly_read_matrix[0][3],
        correctly_read_matrix[1][0], correctly_read_matrix[1][1], correctly_read_matrix[1][2], correctly_read_matrix[1][3]
    );
    printf("(%f,%f,%f,%f)\n\n",
        correctly_read_matrix[2][0], correctly_read_matrix[2][1], correctly_read_matrix[2][2], correctly_read_matrix[2][3]
    );
    // With this syntax however, Slang ignores the column major setting and loads it as it it was row major!
    float3x4 broken_matrix = *push.ptr;
    printf("(%f,%f,%f,%f)\n(%f,%f,%f,%f)\n",
        broken_matrix[0][0], broken_matrix[0][1], broken_matrix[0][2], broken_matrix[0][3],
        broken_matrix[1][0], broken_matrix[1][1], broken_matrix[1][2], broken_matrix[1][3]
    );
    printf("(%f,%f,%f,%f)\n\n",
        broken_matrix[2][0], broken_matrix[2][1], broken_matrix[2][2], broken_matrix[2][3]
    );
}