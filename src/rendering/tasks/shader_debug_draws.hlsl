#include <daxa/daxa.inl>

#include "shader_debug_draws.inl"
#include "shader_shared/debug.inl"

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


[[vk::push_constant]] DrawDebugClonePush draw_debug_clone_push;

__generic<TVEC>
func generic_read_texture(uint2 thread_index) -> TVEC
{
    let p = draw_debug_clone_push;
    SamplerState sampler = p.globals->samplers.nearest_clamp.get();
    float2 uv = (float2(thread_index) + 0.5f) * rcp(float2(p.src_size));
    float uv1 = uv.x;
    return Texture2D<TVEC>::get(p.src).SampleLevel(sampler, uv, 0);
    // switch (p.format)
    // {
    //     /*REGULAR_1D*/ case 0: return Texture1D<TVEC>::get(p.src).SampleLevel(sampler, uv1, p.src_mip);
    //     /*REGULAR_2D*/ case 1: return Texture2D<TVEC>::get(p.src).SampleLevel(sampler, uv, p.src_mip);
    //     /*REGULAR_3D*/ case 2: return Texture3D<TVEC>::get(p.src).SampleLevel(sampler, float3(uv, p.src_layer), p.src_mip);
    //     /*CUBE*/ case 3: return TVEC(); // unimplemented, use a image2darray view instead!
    //     /*REGULAR_1D_ARRAY*/ case 4: return Texture1DArray<TVEC>::get(p.src).SampleLevel(sampler, float2(uv1, p.src_layer), p.src_mip);
    //     /*REGULAR_2D_ARRAY*/ case 5: return Texture2DArray<TVEC>::get(p.src).SampleLevel(sampler, float3(uv, p.src_layer), p.src_mip);
    //     /*CUBE_ARRAY*/  case 6: return TVEC();// unimplemented, use a image2darray view instead!
    // }
    // return TVEC();
}

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
func entry_draw_debug_clone(uint2 thread_index : SV_DispatchThreadID)
{
    let p = draw_debug_clone_push;

    if (any(thread_index >= p.src_size))
        return;

    float4 float_sample = float4(0,0,0,0);

    switch (p.format)
    {
        case DrawDebugClone_Format::DrawDebugClone_Format_FLOAT: 
        {
            var sample = generic_read_texture<float4>(thread_index);
            float_sample = (sample - p.float_min) * rcp(p.float_max - p.float_min);
        }
        break;
        case DrawDebugClone_Format::DrawDebugClone_Format_INT: 
        {
            var sample = generic_read_texture<int4>(thread_index);
            if (p.rainbow_ints)
                float_sample = float4(rainbow_maker(sample.x), 1);
            else
                float_sample = float4((sample - p.int_min) * rcp(p.int_max - p.int_min));
        }
        break;
        case DrawDebugClone_Format::DrawDebugClone_Format_UINT: 
        {
            var sample = generic_read_texture<uint4>(thread_index);        
            if (p.rainbow_ints)
                float_sample = float4(rainbow_maker(sample.x), 1);
            else
                float_sample = float4((sample - p.uint_min) * rcp(p.uint_max - p.uint_min));    
        }
        break;
    }

    // if (all(thread_index == uint2(0,0)))
    // {
    //     printf("x %i, y %i\n", p.src_size.x, p.src_size.y);
    // }

    float_sample[0] = p.enabled_channels[0] != 0 ? float_sample[0] : 0.0f;
    float_sample[1] = p.enabled_channels[1] != 0 ? float_sample[1] : 0.0f;
    float_sample[2] = p.enabled_channels[2] != 0 ? float_sample[2] : 0.0f;
    float_sample[3] = p.enabled_channels[3] != 0 ? float_sample[3] : 0.0f;

    p.dst.get()[thread_index] = float_sample;
}