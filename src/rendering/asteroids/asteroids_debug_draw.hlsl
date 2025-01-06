#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "draw_asteroids.inl"
#include "../../shader_shared/asteroids.inl"

struct DrawDebugAsteroidVertexToPixel
{
    float4 position : SV_Position;
    float3 asteroid_position;
    float3 normal;
};

[[vk::push_constant]] DebugDrawAsteroidsPush debug_draw_asteroids_push;

[shader("vertex")]
func entry_vertex_debug_draw_asteroids(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> DrawDebugAsteroidVertexToPixel
{
    let push = debug_draw_asteroids_push;
    var position = push.asteroid_mesh_positions[vertex_index];
    var normal = normalize(position);

    float3 asteroid_position = push.attach.asteroids[instance_index].position;
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
    ret.asteroid_position = asteroid_position;
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
    let sun_norm_dot = clamp(dot(vertex_to_pixel.normal, sun_direction), 0.0, 1.0f);
    return DrawDebugAsteroidFragmentOut(float4(float3(sun_norm_dot), 1.0f));
}