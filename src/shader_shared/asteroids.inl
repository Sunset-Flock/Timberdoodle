#pragma once

#include <daxa/daxa.inl>

#define MAX_ASTEROID_COUNT 20000

struct GPUAsteroid
{
    daxa_f32vec3 position;
    daxa_f32vec3 velocity;
    daxa_f32vec3 acceleration;
    daxa_f32 velocity_divergence;
    daxa_f32 pressure;
    daxa_f32 density;

    daxa_f32 particle_scale;
};