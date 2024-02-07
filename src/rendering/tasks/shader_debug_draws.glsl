#include <daxa/daxa.inl>

#include "shader_debug_draws.inl"
#include "shader_shared/debug.inl"

#if defined(DRAW_CIRCLE)
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX  || __cplusplus

layout(location = 0) out vec3 vtf_color;

DAXA_DECL_PUSH_CONSTANT(DebugDraw, attachments)
void main()
{
    const uint circle_idx = gl_InstanceIndex;
    const uint vertex_idx = gl_VertexIndex;

    const ShaderDebugCircleDraw circle = deref(deref(deref(attachments.globals).debug_draw_info).circle_draws + circle_idx);

    const float rotation = float(vertex_idx) * (1.0f / (64.0f - 1.0f)) * 2.0f * 3.14f;
    // Make circle in world space.
    vec4 model_position = vec4(circle.radius * vec3(cos(rotation),sin(rotation), 0.0f), 1);
    if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        mat4 view = deref(attachments.globals).camera.view;
        // Remove position aspect of the view matrix.
        view[3] = vec4(0,0,0,1);
        // View matrix only has rotation left in it. Reverse rotation to rotate the circle to face the camera.
        mat4 inv_view_rotation = inverse(view);
        // Rotate circle to face camera.
        model_position = inv_view_rotation * model_position;
        // Add on world position of circle
        model_position = model_position + vec4(circle.position, 0);
        const vec4 clipspace_position = deref(attachments.globals).camera.view_proj * model_position;

        vtf_color = circle.color;
        gl_Position = clipspace_position;
    }
    else if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        vtf_color = circle.color;
        gl_Position = vec4(model_position.xyz,0) + vec4(circle.position, 1);
    }
}

#endif // DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX  || __cplusplus

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT || __cplusplus

layout(location = 0) in vec3 vtf_color;

layout(location = 0) out vec4 color;
void main()
{
    color = vec4(vtf_color,1);
}
#endif // DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT || __cplusplus
#endif // defined(DRAW_CIRCLE)

#if defined(DRAW_RECTANGLE)
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
layout(location = 0) out vec3 vtf_color;

DAXA_DECL_PUSH_CONSTANT(DebugDraw, attachments)
const vec2 rectangle_pos[] = vec2[](
    vec2(-0.5, -0.5),
    vec2( 0.5, -0.5),
    vec2( 0.5,  0.5),
    vec2(-0.5,  0.5),
    vec2(-0.5, -0.5)
);

void main()
{
    const uint rectangle_idx = gl_InstanceIndex;
    const uint vertex_idx = gl_VertexIndex;

    const ShaderDebugRectangleDraw rectangle = deref(deref(deref(attachments.globals).debug_draw_info).rectangle_draws + rectangle_idx);
    const vec2 scaled_position = rectangle_pos[gl_VertexIndex] * rectangle.span;

    if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        const mat4 view = deref(attachments.globals).camera.view;
        const mat4 projection = deref(attachments.globals).camera.proj;
        const vec3 view_center_position = (view * vec4(rectangle.center, 1.0)).xyz;
        const vec3 view_offset_position = vec3(view_center_position.xy + scaled_position, view_center_position.z);
        const vec4 clipspace_position = projection * vec4(view_offset_position, 1.0);
        vtf_color = rectangle.color;
        gl_Position = clipspace_position;
    }
    else if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        vtf_color = rectangle.color;
        gl_Position = vec4(rectangle.center.xyz, 0.0) + vec4(scaled_position, 0.0, 1.0);
    }
}
#endif //DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT
layout(location = 0) in vec3 vtf_color;

layout(location = 0) out vec4 color;
void main()
{
    color = vec4(vtf_color,1);
}
#endif //DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT
#endif //defined(DRAW_RECTANGLE)


