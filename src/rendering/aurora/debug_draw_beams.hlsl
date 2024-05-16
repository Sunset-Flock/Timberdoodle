#include "daxa/daxa.inl"
#include "shader_lib/debug.glsl"
#include "shader_lib/aurora_util.glsl"

#include "aurora.inl"
[[vk::push_constant]] DebugDrawBeamOriginsH::AttachmentShaderBlob push;

[numthreads(DEBUG_DRAW_BEAM_ORIGINS_WG, 1, 1)]
[shader("compute")]
void main(
    uint3 svdtid : SV_DispatchThreadID
)
{
    if(svdtid.x < push.aurora_globals.beam_count)
    {
        let start_segment_buff_idx = beam_segment_buffer_idx(svdtid.x, 0, push.aurora_globals);
        for(int segment = 1; segment < push.aurora_globals.beam_path_segment_count; segment++)
        {
            let start_position = push.beam_paths[start_segment_buff_idx + segment - 1];
            let end_position = push.beam_paths[start_segment_buff_idx + segment];
            ShaderDebugLineDraw draw = ShaderDebugLineDraw(
                float3[2](start_position, end_position),
                float3[2](float3(0.0, 0.6, 0.7), float3(0.0, 0.9, 0.6)),
                DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
            );
            debug_draw_line(push.globals.debug, draw);
        }
    }
}