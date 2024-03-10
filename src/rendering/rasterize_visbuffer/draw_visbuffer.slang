#include "daxa/daxa.inl"
#include "draw_visbuffer.inl"
#include "shader_shared/cull_util.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/cull_util.glsl"
#include "shader_lib/pass_logic.glsl"

[[vk::push_constant]] DrawVisbufferPush_WriteCommand write_cmd_p;
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