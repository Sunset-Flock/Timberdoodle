#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "draw_asteroids.inl"
#include "../../shader_shared/asteroids.inl"
#include "../../shader_lib/misc.hlsl"

struct DrawDebugAsteroidVertexToPixel
{
    float4 position : SV_Position;
    float3 normal;
    nointerpolation int asteroid_index;
};

[[vk::push_constant]] DebugDrawAsteroidsPush debug_draw_asteroids_push;

#define DEBUG_ASTEROID_SCALE 1.0f

[shader("vertex")]
func entry_vertex_debug_draw_asteroids(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> DrawDebugAsteroidVertexToPixel
{
    let push = debug_draw_asteroids_push;
    var position = push.asteroid_mesh_positions[vertex_index] * DEBUG_ASTEROID_SCALE;
    var normal = normalize(position);

    float3 asteroid_position = push.attach.asteroids[instance_index].position * 0.001;
    position += asteroid_position;

    float4x4* viewproj = {};
    if (push.attach.globals.settings.draw_from_observer != 0)
    {
        viewproj = &push.attach.globals.observer_camera.view_proj;
    }
    else
    {
        viewproj = &push.attach.globals.camera.view_proj;
    }

    DrawDebugAsteroidVertexToPixel ret = {};
    ret.position = mul(*viewproj, float4(position, 1));
    ret.normal = normal;
    ret.asteroid_index = instance_index;
    return ret;
}

struct DrawDebugAsteroidFragmentOut
{
    float4 color : SV_Target;
};

[shader("fragment")]
func entry_fragment_debug_draw_asteroids(DrawDebugAsteroidVertexToPixel vertex_to_pixel) -> DrawDebugAsteroidFragmentOut
{
    let push = debug_draw_asteroids_push;

    let sun_direction = push.attach.globals.sky_settings.sun_direction;
    let ambient = 0.05f;
    let sun_norm_dot = clamp(dot(vertex_to_pixel.normal, sun_direction), 0.0, 1.0f);
    let lighting = sun_norm_dot + ambient;

    float3 color = lighting;
    switch(push.attach.globals.asteroid_settings.debug_draw_mode)
    {
        case ASTEROID_DEBUG_DRAW_MODE_NONE:
            break;
        case ASTEROID_DEBUG_DRAW_MODE_VELOCITY:
            const float min_value = 000.0f;
            const float max_value = 100000.0f;
            let velocity = push.attach.asteroids[vertex_to_pixel.asteroid_index].velocity;
            let velocity_size = length(velocity);
            let remapped_velocity = max(velocity_size - min_value, 0.0f) / (max_value - min_value);
            color = TurboColormap(remapped_velocity);
    }

    return DrawDebugAsteroidFragmentOut(float4(color, 1.0f));
}