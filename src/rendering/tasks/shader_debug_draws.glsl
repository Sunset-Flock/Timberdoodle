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

    const ShaderDebugCircleDraw circle = deref(deref(deref(push.attachments.globals).debug).circle_draws + circle_idx);

    const float rotation = float(vertex_idx) * (1.0f / (64.0f - 1.0f)) * 2.0f * 3.14f;
    // Make circle in world space.
    vec4 model_position = vec4(circle.radius * vec3(cos(rotation),sin(rotation), 0.0f), 1);
    vec4 out_position;
    vtf_color = circle.color;
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

        out_position = clipspace_position;
    }
    else if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        out_position = vec4(model_position.xyz,0) + vec4(circle.position, 1);
    }
    else if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = vec4(model_position.xyz,0) + vec4(circle.position, 1);
    }
    // If we draw in ndc of the main camera, we must translate it from main camera to observer.
    if (push.draw_as_observer == 1 && circle.coord_space != DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = deref(push.attachments.globals).observer_camera.view_proj * (deref(push.attachments.globals).camera.inv_view_proj * out_position);
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

    const ShaderDebugRectangleDraw rectangle = deref(deref(deref(push.attachments.globals).debug).rectangle_draws + rectangle_idx);
    const vec2 scaled_position = rectangle_pos[gl_VertexIndex] * rectangle.span;

    vec4 out_position;
    vtf_color = rectangle.color;
    if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        const mat4 view = deref(push.attachments.globals).camera.view;
        const mat4 projection = deref(push.attachments.globals).camera.proj;
        const vec3 view_center_position = (view * vec4(rectangle.center, 1.0)).xyz;
        const vec3 view_offset_position = vec3(view_center_position.xy + scaled_position, view_center_position.z);
        const vec4 clipspace_position = projection * vec4(view_offset_position, 1.0);
        out_position = clipspace_position;
    }
    else if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        out_position = vec4(rectangle.center.xyz, 0.0) + vec4(scaled_position, 0.0, 1.0);
    }
    else if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = vec4(rectangle.center.xyz, 0.0) + vec4(scaled_position, 0.0, 1.0);
    }
    // If we draw in ndc of the main camera, we must translate it from main camera to observer.
    if (push.draw_as_observer == 1 && rectangle.coord_space != DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = deref(push.attachments.globals).observer_camera.view_proj * (deref(push.attachments.globals).camera.inv_view_proj * out_position);
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

    const ShaderDebugAABBDraw aabb = deref(deref(deref(push.attachments.globals).debug).aabb_draws + aabb_idx);

    vec4 out_position;
    const vec3 model_position = aabb_vertex_base_offsets[vertex_idx] * 0.5f * aabb.size + aabb.position;
    vtf_color = aabb.color;
    if (aabb.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        out_position = deref(push.attachments.globals).camera.view_proj * vec4(model_position, 1);
    }
    else if (aabb.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        out_position = vec4(model_position, 1);
    }
    else if (aabb.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = vec4(model_position, 1);
    }
    // If we draw in ndc of the main camera, we must translate it from main camera to observer.
    if (push.draw_as_observer == 1 && aabb.coord_space != DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = deref(push.attachments.globals).observer_camera.view_proj * (deref(push.attachments.globals).camera.inv_view_proj * out_position);
    }
    gl_Position = out_position;
}
#endif //DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
#endif //defined(DRAW_AABB)

#if defined(DRAW_BOX) || __cplusplus
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
layout(location = 0) out vec3 vtf_color;

DAXA_DECL_PUSH_CONSTANT(DebugDrawPush, push)
const uint box_vertex_indices[] = uint[](
    // Bottom rectangle
    4, 5, 5, 6, 6, 7, 7, 4,
    // Top rectangle
    0, 1, 1, 2, 2, 3, 3, 0,
    // Connecting lines
    0, 4, 1, 5, 2, 6, 3, 7
);

void main()
{
    const uint box_idx = gl_InstanceIndex;
    const uint vertex_idx = gl_VertexIndex;

    const ShaderDebugBoxDraw box = deref(deref(deref(push.attachments.globals).debug).box_draws + box_idx);

    vec4 out_position;
    const vec3 model_position = box.vertices[box_vertex_indices[vertex_idx]];
    vtf_color = box.color;
    if (box.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        out_position = deref(push.attachments.globals).camera.view_proj * vec4(model_position, 1);
    }
    else if (box.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        out_position = vec4(model_position, 1);
    }
    else if (box.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = vec4(model_position, 1);
    }
    // If we draw in ndc of the main camera, we must translate it from main camera to observer.
    if (push.draw_as_observer == 1 && box.coord_space != DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = deref(push.attachments.globals).observer_camera.view_proj * (deref(push.attachments.globals).camera.inv_view_proj * out_position);
    }
    gl_Position = out_position;
}
#endif //DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
#endif //defined(DRAW_BOX)

#if defined(DRAW_LINE) || __cplusplus
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
layout(location = 0) out vec3 vtf_color;

DAXA_DECL_PUSH_CONSTANT(DebugDrawPush, push)

void main()
{
    const uint line_idx = gl_InstanceIndex;
    const uint vertex_idx = gl_VertexIndex;

    const ShaderDebugLineDraw line = deref(deref(deref(push.attachments.globals).debug).line_draws + line_idx);

    vec4 out_position;
    const vec3 model_position = line.vertices[vertex_idx];
    vtf_color = line.colors[vertex_idx];
    if (line.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        out_position = deref(push.attachments.globals).camera.view_proj * vec4(model_position, 1);
    }
    else if (line.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        out_position = vec4(model_position, 1);
    }
    else if (line.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = vec4(model_position, 1);
    }
    // If we draw in ndc of the main camera, we must translate it from main camera to observer.
    if (push.draw_as_observer == 1 && line.coord_space != DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        out_position = deref(push.attachments.globals).observer_camera.view_proj * (deref(push.attachments.globals).camera.inv_view_proj * out_position);
    }
    gl_Position = out_position;
}
#endif //DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
#endif //defined(DRAW_LINE)

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT || __cplusplus
DAXA_DECL_PUSH_CONSTANT(DebugDrawPush, push)
layout(location = 0) in vec3 vtf_color;

layout(location = 0) out vec4 color;
void main()
{
    const float depth_bufer_depth = texelFetch(daxa_texture2D(push.attachments.depth_image), ivec2(gl_FragCoord.xy), 0).x;
    // if(depth_bufer_depth > gl_FragCoord.z) { discard; }
    // color = vec4(vtf_color,1);
}
#endif //DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT