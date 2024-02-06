#include <daxa/daxa.inl>

#include "shader_debug_draws.inl"
#include "shader_shared/debug.inl"

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX || __cplusplus

DAXA_DECL_PUSH_CONSTANT(DebugDrawCircles, attachments)
void main()
{
    const uint circle_idx = gl_InstanceIndex;
    const uint vertex_idx = gl_VertexIndex;

    const ShaderDebugCircleDraw circle = deref(deref(deref(attachments.globals).debug_draw_info).circle_draws + circle_idx);

    const float rotation = float(vertex_idx) * (1.0f / 64.0f);
    const vec3 model_position = circle.radius * vec3(cos(rotation),sin(rotation), 0);
    const vec4 clipspace_position = deref(attachments.globals).camera.view_proj * vec4(model_position, 1);

    gl_Position = clipspace_position;
}

#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT || __cplusplus

layout(location = 0) out vec4 color;
void main()
{
    color = vec4(1,0,0,1);
}

#endif