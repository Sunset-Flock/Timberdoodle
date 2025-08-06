#include <daxa/daxa.inl>

#include "shader_debug_draws.inl"
#include "shader_shared/debug.inl"
#include "shader_lib/depth_util.glsl"

struct VertexToPixel
{
    float3 color;
    float4 position : SV_Position;
};

[[vk::push_constant]] DebugDrawPush push;

func vertex_rotate_to_main_cam(float3 orbit_position) -> float4
{
    CameraInfo cam = push.attachments.globals->main_camera;
    float4x4 inv_view_rotation = cam.inv_view;
    // Remove position aspect of the view matrix.
    inv_view_rotation[0][3] = 0;
    inv_view_rotation[1][3] = 0;
    inv_view_rotation[2][3] = 0;
    inv_view_rotation[3][3] = 1;
    return mul(inv_view_rotation, float4(orbit_position, 1.0f));
}

func vertex_transform(uint coord_space, float4 model_position) -> float4
{
    float4 ret = (float4)0;
    if (coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        ret = mul(push.attachments.globals->view_camera.view_proj, model_position);
    }
    else
    {
        ret = model_position;
    }
    if (coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_MAIN_CAMERA)
    {
        ret = mul(push.attachments.globals->view_camera.view_proj, mul(push.attachments.globals->main_camera.inv_view_proj, ret));
    }
    return ret;
}

func line_vertex(uint vertex_index, uint instance_index, out float4 position, out uint coord_space, out float3 color)
{
    const ShaderDebugLineDraw line = push.attachments.globals->debug->line_draws.draws[instance_index];
    position = float4(vertex_index == 0 ? line.start : line.end, 1.0f);
    coord_space = line.coord_space;
    color = line.color;
}

func circle_line_vertex(uint vertex_index, uint instance_index, out float4 position, out uint coord_space, out float3 color)
{    
    const ShaderDebugCircleDraw circle = push.attachments.globals->debug->circle_draws.draws[instance_index];

    uint segment = (vertex_index + 1) / 2;
    const float rotation = float(segment) * (1.0f / (64.0f - 1.0f)) * 2.0f * 3.14f;
    // Make circle in world space.
    position = float4(circle.radius * float3(cos(rotation),sin(rotation), 0.0f), 1.0f);
    
    if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        position = vertex_rotate_to_main_cam(position.xyz);
    }
    position = position + float4(circle.position, 0.0f);
    coord_space = circle.coord_space;
    color = circle.color;
}

func rect_line_vertex(uint vertex_index, uint instance_index, out float4 position, out uint coord_space, out float3 color)
{
    static const float2 rectangle_pos[8] = float2[8](
        float2(-0.5, -0.5), float2( 0.5, -0.5),
        float2( 0.5, -0.5), float2( 0.5,  0.5),
        float2( 0.5,  0.5), float2(-0.5,  0.5),
        float2(-0.5,  0.5), float2(-0.5, -0.5)
    );
    const ShaderDebugRectangleDraw rectangle = push.attachments.globals->debug->rectangle_draws.draws[instance_index];
    position = float4(rectangle_pos[vertex_index] * rectangle.span, 0, 1.0f);

    if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        position = vertex_rotate_to_main_cam(position.xyz);
    }
    position = position + float4(rectangle.center, 0.0f);
    coord_space = rectangle.coord_space;
    color = rectangle.color;
}

func aabb_line_vertex(uint vertex_index, uint instance_index, out float4 position, out uint coord_space, out float3 color)
{
    const ShaderDebugAABBDraw aabb = push.attachments.globals->debug->aabb_draws.draws[instance_index];

    static const float3 aabb_vertex_base_offsets[24] = float3[24](
        // Bottom rectangle:
        float3(-1,-1,-1), float3( 1,-1,-1),
        float3(-1,-1,-1), float3(-1, 1,-1),
        float3(-1, 1,-1), float3( 1, 1,-1),
        float3( 1,-1,-1), float3( 1, 1,-1),
        // Top rectangle:
        float3(-1,-1, 1), float3( 1,-1, 1),
        float3(-1,-1, 1), float3(-1, 1, 1),
        float3(-1, 1, 1), float3( 1, 1, 1),
        float3( 1,-1, 1), float3( 1, 1, 1),
        // Connecting lines:
        float3(-1,-1,-1), float3(-1,-1, 1),
        float3( 1,-1,-1), float3( 1,-1, 1),
        float3(-1, 1,-1), float3(-1, 1, 1),
        float3( 1, 1,-1), float3( 1, 1, 1)
    );
    position = float4(aabb_vertex_base_offsets[vertex_index] * 0.5f * aabb.size, 1.0f);

    position = position + float4(aabb.position, 0.0f);
    coord_space = aabb.coord_space;
    color = aabb.color;
}

func box_line_vertex(uint vertex_index, uint instance_index, out float4 position, out uint coord_space, out float3 color)
{
    static const uint box_vertex_indices[24] = uint[24](
        // Bottom rectangle
        4, 5, 5, 6, 6, 7, 7, 4,
        // Top rectangle
        0, 1, 1, 2, 2, 3, 3, 0,
        // Connecting lines
        0, 4, 1, 5, 2, 6, 3, 7
    );
    const ShaderDebugBoxDraw box = push.attachments.globals->debug->box_draws.draws[instance_index];
    position = float4(box.vertices[box_vertex_indices[vertex_index]], 1.0f);

    coord_space = box.coord_space;
    color = box.color;
}

func cone_line_vertex(uint vertex_index, uint instance_index, out float4 position, out uint coord_space, out float3 color)
{
    const ShaderDebugConeDraw cone = push.attachments.globals->debug->cone_draws.draws[instance_index];

    static let SEGMENTS = DEBUG_OBJECT_CONE_VERTICES/4;
    static let BASE_VERTICES = DEBUG_OBJECT_CONE_VERTICES/2;
    
    float3 tangent_side = normalize(cross(float3(0,0,1), cone.direction));
    float3 tangent_up = normalize(cross(cone.direction, tangent_side));
    let tan_angle = tan(cone.angle);

    uint segment = vertex_index/2;
    uint is_first = vertex_index%2;
    let is_tip_vertex = !is_first && vertex_index < BASE_VERTICES;

    segment = is_first ? segment : segment + 1;
    let c = cos( 3.14 * 2 * (float(segment) * rcp(SEGMENTS)));
    let s = sin( 3.14 * 2 * (float(segment) * rcp(SEGMENTS)));
    float3 segment_base_pos = ((tangent_side * c + tangent_up * s) * tan_angle + cone.direction) * cone.size;
    position = float4(is_tip_vertex ? float3(0,0,0) : segment_base_pos, 1.0f);
    position.xyz += cone.position;

    coord_space = cone.coord_space;
    color = cone.color;
}

func sphere_line_vertex(uint vertex_index, uint instance_index, out float4 position, out uint coord_space, out float3 color)
{
    const ShaderDebugSphereDraw sphere = push.attachments.globals->debug->sphere_draws.draws[instance_index];

    let CIRCLE_SEGMENTS = (DEBUG_OBJECT_SPHERE_VERTICES/5)/2;
    let CIRCLE_VERTICES = DEBUG_OBJECT_SPHERE_VERTICES/5;

    uint segment = ((vertex_index % CIRCLE_VERTICES) + 1) / 2;
    const float rotation = float(segment) * (1.0f / (CIRCLE_VERTICES/2 - 1.0f)) * 2.0f * 3.14f;

    let c = cos(rotation);
    let s = sin(rotation);

    let circle = vertex_index / CIRCLE_VERTICES;

    if (circle == 0)
    {
        position = float4(float3(c, s, 0.0f), 1.0f);
    }
    if (circle == 1)
    {
        position = float4(float3(c * rsqrt(2), c * rsqrt(2), s), 1.0f);
    }
    if (circle == 2)
    {
        position = float4(float3(s * rsqrt(2), -s * rsqrt(2), c), 1.0f);
    }
    if (circle == 3)
    {
        position = float4(float3(0.0f, c, s), 1.0f);
    }
    if (circle == 4)
    {
        position = float4(float3(s, 0.0f, c), 1.0f);
    }
    position.xyz = position.xyz * sphere.radius;

    position = position + float4(sphere.position, 0.0f);
    coord_space = sphere.coord_space;
    color = sphere.color;
}

[shader("vertex")]
func entry_vertex_line(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> VertexToPixel
{
    VertexToPixel ret = {};

    uint coord_space;
    switch (push.mode)
    {
        case DEBUG_OBJECT_DRAW_MODE_LINE:
        {
            line_vertex(vertex_index, instance_index, ret.position, coord_space, ret.color);
            break;
        }
        case DEBUG_OBJECT_DRAW_MODE_CIRCLE:
        {
            circle_line_vertex(vertex_index, instance_index, ret.position, coord_space, ret.color);
            break;
        }
        case DEBUG_OBJECT_DRAW_MODE_RECTANGLE:
        {
            rect_line_vertex(vertex_index, instance_index, ret.position, coord_space, ret.color);
            break;
        }
        case DEBUG_OBJECT_DRAW_MODE_AABB:
        {
            aabb_line_vertex(vertex_index, instance_index, ret.position, coord_space, ret.color);
            break;
        }
        case DEBUG_OBJECT_DRAW_MODE_BOX:
        {
            box_line_vertex(vertex_index, instance_index, ret.position, coord_space, ret.color);
            break;
        }
        case DEBUG_OBJECT_DRAW_MODE_CONE:
        {
            cone_line_vertex(vertex_index, instance_index, ret.position, coord_space, ret.color);
            break;
        }
        case DEBUG_OBJECT_DRAW_MODE_SPHERE:
        {
            sphere_line_vertex(vertex_index, instance_index, ret.position, coord_space, ret.color);
            break;
        }
    }
    ret.position = vertex_transform(coord_space, ret.position);

    return ret;
}

struct FragmentOut
{
    float4 color : SV_Target;
};

[shader("fragment")]
func entry_fragment(VertexToPixel vertToPix) -> FragmentOut
{
    return FragmentOut(float4(vertToPix.color,1));
}


[[vk::push_constant]] DebugTaskDrawDebugDisplayPush draw_debug_clone_push;

float3 hsv2rgb(float3 c) {
    float4 k = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * lerp(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

float3 rainbow_maker(uint i)
{
    return (0.2987123 * float(i), 1.0f, 1.0f);
}
float3 rainbow_maker(int i)
{
    return (0.2987123 * float(i), 1.0f, 1.0f);
}

[shader("compute")]
[numthreads(DEBUG_DRAW_CLONE_X,DEBUG_DRAW_CLONE_Y,1)]
func entry_draw_debug_display(uint2 thread_index : SV_DispatchThreadID)
{
    let p = draw_debug_clone_push;

    if (any(thread_index >= p.src_size))
        return;

    float4 sample_color = float4(0,0,0,0);

    let readback_pixel = all(thread_index == p.mouse_over_index);

    switch (p.format)
    {
        case 0: 
        {
            var sample = RWTexture2D<float4>::get(p.src)[thread_index];
            if (readback_pixel)
            {
                ((float4*)p.readback_ptr)[p.readback_index * 2] = sample;
            }
            sample_color = float4((sample.rgb - p.float_min) * rcp(p.float_max - p.float_min), sample.a);
        }
        break;
        case 1: 
        {
            var sample = RWTexture2D<int4>::get(p.src)[thread_index];
            if (readback_pixel)
            {
                ((int4*)p.readback_ptr)[p.readback_index * 2] = sample;
            }
            if (p.rainbow_ints)
                sample_color = float4(rainbow_maker(sample.x), 1);
            else
                sample_color = float4((sample.rgb - p.int_min) * rcp(p.int_max - p.int_min), sample.a);
        }
        break;
        case 2: 
        {
            var sample = RWTexture2D<uint4>::get(p.src)[thread_index];
            if (readback_pixel)
            {
                ((uint4*)p.readback_ptr)[p.readback_index * 2] = sample;
            }
            if (p.rainbow_ints)
                sample_color = float4(rainbow_maker(sample.x), 1);
            else
                sample_color = float4((sample.rgb - p.uint_min) * rcp(p.uint_max - p.uint_min), sample.a);    
        }
        break;
    }
    
    let one_channel_active = (p.enabled_channels[0] + p.enabled_channels[1] + p.enabled_channels[2] + p.enabled_channels[3]) == 1;
    let only_alpha_active = one_channel_active && p.enabled_channels[3];

    if (only_alpha_active)
    {
        sample_color[3] = (sample_color[3] - p.float_min) * rcp(p.float_max - p.float_min);
    }

    sample_color[0] = p.enabled_channels[0] != 0 ? sample_color[0] : 0.0f;
    sample_color[1] = p.enabled_channels[1] != 0 ? sample_color[1] : 0.0f;
    sample_color[2] = p.enabled_channels[2] != 0 ? sample_color[2] : 0.0f;
    sample_color[3] = p.enabled_channels[3] != 0 ? sample_color[3] : 1.0f;

    if (one_channel_active)
    {
        let single_channel_color = 
            (p.enabled_channels[0] * sample_color[0]) + 
            (p.enabled_channels[1] * sample_color[1]) + 
            (p.enabled_channels[2] * sample_color[2]) + 
            (p.enabled_channels[3] * sample_color[3]);
        sample_color.xyz = single_channel_color;
        sample_color[3] = 1.0f;
    }

    if (readback_pixel)
    {
        p.readback_ptr[p.readback_index * 2 + 1] = sample_color;
        let color_max = max(max(sample_color.x, sample_color.y), max(sample_color.z, sample_color.w));
        let color_max_int = uint(color_max); 
        let color_min = max(max(sample_color.x, sample_color.y), max(sample_color.z, sample_color.w));
    }

    let previous_value = p.dst.get()[thread_index];
    p.dst.get()[thread_index] = float4(lerp(previous_value.rgb, sample_color.rgb, sample_color.a), 1.0f);
}