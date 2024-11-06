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

[shader("vertex")]
func entry_vertex_circle(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> VertexToPixel
{
    VertexToPixel ret = {};
    const ShaderDebugCircleDraw circle = push.attachments.globals->debug->circle_draws[instance_index];

    const float rotation = float(vertex_index) * (1.0f / (64.0f - 1.0f)) * 2.0f * 3.14f;
    // Make circle in world space.
    float4 model_position = float4(circle.radius * float3(cos(rotation),sin(rotation), 0.0f), 1);
    ret.color = circle.color;
    if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        float4x4 inv_view_rotation = push.attachments.globals->camera.inv_view;
        // Remove position aspect of the view matrix.
        inv_view_rotation[0][3] = 0;
        inv_view_rotation[0][3] = 0;
        inv_view_rotation[0][3] = 0;
        inv_view_rotation[0][3] = 1;
        // Rotate circle to face camera.
        model_position = mul(inv_view_rotation, model_position);
        // Add on world position of circle
        model_position = model_position + float4(circle.position, 0);
        const float4 clipspace_position = mul(push.attachments.globals->camera.view_proj, model_position);

        ret.position = clipspace_position;
    }
    else if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        ret.position = float4(model_position.xyz,0) + float4(circle.position, 1);
    }
    else if (circle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        ret.position = float4(model_position.xyz,0) + float4(circle.position, 1);
    }
    // If we draw in ndc of the main camera, we must translate it from main camera to observer.
    if (push.draw_as_observer == 1 && circle.coord_space != DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        ret.position = mul(push.attachments.globals->observer_camera.view_proj, mul(push.attachments.globals->camera.inv_view_proj, ret.position));
    }
    return ret;
}

static const float2 rectangle_pos[6] = float2[6](
    float2(-0.5, -0.5),
    float2( 0.5, -0.5),
    float2( 0.5,  0.5),
    float2(-0.5,  0.5),
    float2(-0.5, -0.5)
);

[shader("vertex")]
func entry_vertex_rectangle(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> VertexToPixel
{
    VertexToPixel ret = {};
    const ShaderDebugRectangleDraw rectangle = push.attachments.globals->debug->rectangle_draws[instance_index];
    const float2 scaled_position = rectangle_pos[vertex_index] * rectangle.span;

    ret.color = rectangle.color;
    if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        const float4x4 view = push.attachments.globals->camera.view;
        const float4x4 projection = push.attachments.globals->camera.proj;
        const float3 view_center_position = mul(view, float4(rectangle.center, 1.0)).xyz;
        const float3 view_offset_position = float3(view_center_position.xy + scaled_position, view_center_position.z);
        const float4 clipspace_position = mul(projection, float4(view_offset_position, 1.0));
        ret.position = clipspace_position;
    }
    else if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        ret.position = float4(rectangle.center.xyz, 0.0) + float4(scaled_position, 0.0, 1.0);
    }
    else if (rectangle.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        ret.position = float4(rectangle.center.xyz, 0.0) + float4(scaled_position, 0.0, 1.0);
    }
    // If we draw in ndc of the main camera, we must translate it from main camera to observer.
    if (push.draw_as_observer == 1 && rectangle.coord_space != DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        ret.position = mul(push.attachments.globals->observer_camera.view_proj, mul(push.attachments.globals->camera.inv_view_proj, ret.position));
    }
    return ret;
}

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

[shader("vertex")]
func entry_vertex_aabb(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> VertexToPixel
{
    VertexToPixel ret = {};

    const ShaderDebugAABBDraw aabb = push.attachments.globals->debug->aabb_draws[instance_index];

    const float3 model_position = aabb_vertex_base_offsets[vertex_index] * 0.5f * aabb.size + aabb.position;
    ret.color = aabb.color;
    if (aabb.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        ret.position = mul(push.attachments.globals->camera.view_proj, float4(model_position, 1));
    }
    else if (aabb.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        ret.position = float4(model_position, 1);
    }
    else if (aabb.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        ret.position = float4(model_position, 1);
    }
    // If we draw in ndc of the main camera, we must translate it from main camera to observer.
    if (push.draw_as_observer == 1 && aabb.coord_space != DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        ret.position = mul(push.attachments.globals->observer_camera.view_proj, mul(push.attachments.globals->camera.inv_view_proj, ret.position));
    }
    return ret;
}

static const uint box_vertex_indices[24] = uint[24](
    // Bottom rectangle
    4, 5, 5, 6, 6, 7, 7, 4,
    // Top rectangle
    0, 1, 1, 2, 2, 3, 3, 0,
    // Connecting lines
    0, 4, 1, 5, 2, 6, 3, 7
);

[shader("vertex")]
func entry_vertex_box(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> VertexToPixel
{
    VertexToPixel ret = {};
    const ShaderDebugBoxDraw box = push.attachments.globals->debug->box_draws[instance_index];

    const float3 model_position = box.vertices[box_vertex_indices[vertex_index]];
    ret.color = box.color;
    if (box.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE)
    {
        ret.position = mul(push.attachments.globals->camera.view_proj, float4(model_position, 1));
    }
    else if (box.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC)
    {
        ret.position = float4(model_position, 1);
    }
    else if (box.coord_space == DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        ret.position = float4(model_position, 1);
    }
    // If we draw in ndc of the main camera, we must translate it from main camera to observer.
    if (push.draw_as_observer == 1 && box.coord_space != DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER)
    {
        ret.position = mul(push.attachments.globals->observer_camera.view_proj, mul(push.attachments.globals->camera.inv_view_proj, ret.position));
    }
    return ret;
}

struct FragmentOut
{
    float4 color : SV_Target;
};

[shader("fragment")]
func entry_fragment(VertexToPixel vertToPix) -> FragmentOut
{
    // const float depth_bufer_depth = texelFetch(daxa_texture2D(push.attachments.depth_image), ivec2(gl_FragCoord.xy), 0).x;
    // if(depth_bufer_depth > gl_FragCoord.z) { discard; }
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