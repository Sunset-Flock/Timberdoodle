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
    float3 color_interp;
    nointerpolation int asteroid_index;
};

#define COLOR_COUNT 6
static const float3[COLOR_COUNT] ACCRETION_PALETTE = float3[](
    float3(0.43f, 0.70f, 1.f),
    float3(0.5f, 0.5f, 0.5f),
    float3(0.65f, 0.12f, 0.01f),
    float3(0.79f, 0.38f, 0.02f),
    float3(0.93f, 0.83f, 0.34f),
    float3(0.94f, 0.90f, 0.84f));

[[vk::push_constant]] DebugDrawAsteroidsPush debug_draw_asteroids_push;

float3 get_asteroid_float3_param<let PARAM : int>(int asteroid_index)
{
    let push = debug_draw_asteroids_push;
#if CPU_SIMULATION
    switch(PARAM)
    {
        case ASTEROID_POSITION            : return push.attach.asteroids[asteroid_index].position;
        case ASTEROID_VELOCITY            : return push.attach.asteroids[asteroid_index].velocity;
        case ASTEROID_VELOCITY_DIVERGENCE : return push.attach.asteroids[asteroid_index].velocity_divergence;
        case ASTEROID_VELOCITY_DERIVATIVE : return push.attach.asteroids[asteroid_index].acceleration;
        default                           : return 0.0f;
    }
#else
    switch(PARAM)
    {
        case ASTEROID_POSITION            : return push.parameters.position[asteroid_index];
        case ASTEROID_VELOCITY            : return push.parameters.velocity[asteroid_index];
        case ASTEROID_VELOCITY_DIVERGENCE : return push.parameters.velocity_divergence[asteroid_index];
        case ASTEROID_VELOCITY_DERIVATIVE : return push.parameters.velocity_derivative[asteroid_index];
        default                           : return 0.0f;
    }
#endif
}

float get_asteroid_float_param<let PARAM : int>(int asteroid_index)
{
    let push = debug_draw_asteroids_push;
#if CPU_SIMULATION
    switch(PARAM)
    {
        case ASTEROID_PRESSURE: return push.attach.asteroids[asteroid_index].pressure;
        case ASTEROID_DENSITY: return push.attach.asteroids[asteroid_index].density;
        case ASTEROID_SCALE: return push.attach.asteroids[asteroid_index].particle_scale;
        default: return 0.0f;
    }
#else
    switch(PARAM)
    {
        case ASTEROID_SMOOTHING_RADIUS    : return push.parameters.smoothing_radius[asteroid_index];
        case ASTEROID_MASS                : return push.parameters.mass[asteroid_index];
        case ASTEROID_DENSITY             : return push.parameters.density[asteroid_index];
        case ASTEROID_ENERGY              : return push.parameters.energy[asteroid_index];
        case ASTEROID_PRESSURE            : return push.parameters.pressure[asteroid_index];
        case ASTEROID_SCALE               : return push.parameters.scale[asteroid_index];
        default                           : return 0.0f;
    }
#endif
}


[shader("vertex")]
func entry_vertex_debug_draw_asteroids(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> DrawDebugAsteroidVertexToPixel
{
    let push = debug_draw_asteroids_push;
    var position = push.asteroid_mesh_positions[vertex_index] * get_asteroid_float_param<ASTEROID_SCALE>(instance_index);
    var normal = normalize(position);

    float3 asteroid_position = get_asteroid_float3_param<ASTEROID_POSITION>(instance_index) * POSITION_SCALING_FACTOR;
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
    ret.color_interp = 1.0f;

    float value;
    float min_value;
    float max_value;
    bool do_log10;
    if(push.attach.globals.asteroid_settings.simulation_started != 0)
    {
        switch(push.attach.globals.asteroid_settings.debug_draw_mode)
        {
            case ASTEROID_DEBUG_DRAW_MODE_NONE:
                break;
            case ASTEROID_DEBUG_DRAW_MODE_VELOCITY:
                min_value = 0.0f;
                max_value = 10000.0f;
                value = length(get_asteroid_float3_param<ASTEROID_VELOCITY>(instance_index));
                do_log10 = true;
                break;
            case ASTEROID_DEBUG_DRAW_MODE_ACCELERATION:
                min_value = 0.0f;
                max_value = 100.0f;
                value = length(get_asteroid_float3_param<ASTEROID_VELOCITY_DERIVATIVE>(instance_index));
                do_log10 = true;
                break;
            case ASTEROID_DEBUG_DRAW_MODE_VELOCITY_DIVERGENCE:
                min_value = -0.1f;
                max_value = 0.1f;
                value = length(get_asteroid_float3_param<ASTEROID_VELOCITY_DIVERGENCE>(instance_index));
                do_log10 = false;
                break;
            case ASTEROID_DEBUG_DRAW_MODE_PRESSURE:
                min_value = -100000.0f;
                max_value = 10e+10f;
                value = length(get_asteroid_float_param<ASTEROID_PRESSURE>(instance_index));
                do_log10 = true;
                break;
            case ASTEROID_DEBUG_DRAW_MODE_DENSITY:
                min_value = 2650.0f;
                max_value = 2750.0f;
                value = length(get_asteroid_float_param<ASTEROID_DENSITY>(instance_index));
                do_log10 = false;
                break;
        }

        if(push.attach.globals.asteroid_settings.debug_draw_mode != ASTEROID_DEBUG_DRAW_MODE_NONE)
        {
            float rescaled_value;
            if(do_log10) {
                rescaled_value = log10(clamp(value - min_value, 0.0001f, max_value - min_value)) / log10((max_value - min_value));
            }
            else {
                rescaled_value = clamp(value - min_value, 0.0f, max_value - min_value) / (max_value - min_value);
            }
            int lower_color_idx = clamp(floor(rescaled_value * float(COLOR_COUNT - 1)), 0, 5);
            int upper_color_idx = clamp(ceil(rescaled_value * float(COLOR_COUNT - 1)), 0, 5);
            float interp = fract(rescaled_value * float(COLOR_COUNT -1));
            ret.color_interp *= ACCRETION_PALETTE[lower_color_idx] * (1.0f - interp) + ACCRETION_PALETTE[upper_color_idx] * interp;
        }
    }
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
    let ambient = 0.5f;
    let sun_norm_dot = clamp(dot(vertex_to_pixel.normal, sun_direction), 0.0, 1.0f);
    let lighting = sun_norm_dot + ambient;


    float3 color = lighting * vertex_to_pixel.color_interp;
    if(push.attach.globals.asteroid_settings.selected_setup_asteroid == vertex_to_pixel.asteroid_index)
    {
        color = float3(1.0f, 1.0f, 0.0f);
    }

    return DrawDebugAsteroidFragmentOut(float4(pow(color, 2.2), 1.0f));
}