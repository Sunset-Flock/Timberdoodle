#include <daxa/daxa.inl>

#include "shader_debug_draws.inl"
#include "shader_shared/debug.inl"

#if defined(DRAW_CIRCLE) || __cplusplus
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX  || __cplusplus

layout(location = 0) out vec3 vtf_color;

DAXA_DECL_PUSH_CONSTANT(DebugDrawPush, push)
void main()
{
    const uint circle_idx = gl_InstanceIndex;
    const uint vertex_idx = gl_VertexIndex;

    const ShaderDebugCircleDraw circle = deref(deref(deref(push.attachments.globals).debug_draw_info).circle_draws + circle_idx);

    const float rotation = float(vertex_idx) * (1.0f / (64.0f - 1.0f)) * 2.0f * 3.14f;
    // Make circle in world space.
    vec4 model_position = vec4(circle.radius * vec3(cos(rotation),sin(rotation), 0.0f), 1);
    vec4 out_position;
    if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        mat4 view = deref(push.attachments.globals).camera.view;
        // Remove position aspect of the view matrix.
        view[3] = vec4(0,0,0,1);
        // View matrix only has rotation left in it. Reverse rotation to rotate the circle to face the camera.
        mat4 inv_view_rotation = inverse(view);
        // Rotate circle to face camera.
        model_position = inv_view_rotation * model_position;
        // Add on world position of circle
        model_position = model_position + vec4(circle.position, 0);
        const vec4 clipspace_position = deref(push.attachments.globals).camera.view_proj * model_position;

        vtf_color = circle.color;
        out_position = clipspace_position;
    }
    else if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        vtf_color = circle.color;
        out_position = vec4(model_position.xyz,0) + vec4(circle.position, 1);
    }
    if (push.draw_as_observer == 1)
    {
        out_position = deref(push.attachments.globals).observer_camera.view_proj * (inverse(deref(push.attachments.globals).camera.view_proj) * out_position);
    }
    gl_Position = out_position;
}

#endif // DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX  || __cplusplus
#endif // defined(DRAW_CIRCLE)

#if defined(DRAW_RECTANGLE) || __cplusplus
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
layout(location = 0) out vec3 vtf_color;

DAXA_DECL_PUSH_CONSTANT(DebugDrawPush, push)
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

    const ShaderDebugRectangleDraw rectangle = deref(deref(deref(push.attachments.globals).debug_draw_info).rectangle_draws + rectangle_idx);
    const vec2 scaled_position = rectangle_pos[gl_VertexIndex] * rectangle.span;

    vec4 out_position;
    if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        const mat4 view = deref(push.attachments.globals).camera.view;
        const mat4 projection = deref(push.attachments.globals).camera.proj;
        const vec3 view_center_position = (view * vec4(rectangle.center, 1.0)).xyz;
        const vec3 view_offset_position = vec3(view_center_position.xy + scaled_position, view_center_position.z);
        const vec4 clipspace_position = projection * vec4(view_offset_position, 1.0);
        vtf_color = rectangle.color;
        out_position = clipspace_position;
    }
    else if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        vtf_color = rectangle.color;
        out_position = vec4(rectangle.center.xyz, 0.0) + vec4(scaled_position, 0.0, 1.0);
    }
    if (push.draw_as_observer == 1)
    {
        out_position = deref(push.attachments.globals).observer_camera.view_proj * (inverse(deref(push.attachments.globals).camera.view_proj) * out_position);
    }
    gl_Position = out_position;
}
#endif //DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
#endif //defined(DRAW_RECTANGLE)


#if defined(DRAW_AABB) || __cplusplus
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
layout(location = 0) out vec3 vtf_color;

DAXA_DECL_PUSH_CONSTANT(DebugDrawPush, push)
const vec3 aabb_vertex_base_offsets[] = vec3[](
    // Bottom rectangle:
    vec3(-1,-1,-1), vec3( 1,-1,-1),
    vec3(-1,-1,-1), vec3(-1, 1,-1),
    vec3(-1, 1,-1), vec3( 1, 1,-1),
    vec3( 1,-1,-1), vec3( 1, 1,-1),
    // Top rectangle:
    vec3(-1,-1, 1), vec3( 1,-1, 1),
    vec3(-1,-1, 1), vec3(-1, 1, 1),
    vec3(-1, 1, 1), vec3( 1, 1, 1),
    vec3( 1,-1, 1), vec3( 1, 1, 1),
    // Connecting lines:
    vec3(-1,-1,-1), vec3(-1,-1, 1),
    vec3( 1,-1,-1), vec3( 1,-1, 1),
    vec3(-1, 1,-1), vec3(-1, 1, 1),
    vec3( 1, 1,-1), vec3( 1, 1, 1)
);

void main()
{
    const uint aabb_idx = gl_InstanceIndex;
    const uint vertex_idx = gl_VertexIndex;

    const ShaderDebugAABBDraw aabb = deref(deref(deref(push.attachments.globals).debug_draw_info).aabb_draws + aabb_idx);

    vec4 out_position;
    const vec3 model_position = aabb_vertex_base_offsets[vertex_idx] * 0.5f * aabb.size + aabb.position;
    vtf_color = aabb.color;
    out_position = deref(push.attachments.globals).camera.view_proj * vec4(model_position, 1);
    if (push.draw_as_observer == 1)
    {
        out_position = deref(push.attachments.globals).observer_camera.view_proj * (inverse(deref(push.attachments.globals).camera.view_proj) * out_position);
    }
    gl_Position = out_position;
}
#endif //DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

#endif //defined(DRAW_RECTANGLE)


#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT || __cplusplus
layout(location = 0) in vec3 vtf_color;

layout(location = 0) out vec4 color;
void main()
{
    color = vec4(vtf_color,1);
}
#endif //DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT