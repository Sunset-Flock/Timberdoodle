#pragma once

#include <daxa/daxa.inl>

// #define CPU_SIMULATION 1

#if CPU_SIMULATION
#define MAX_ASTEROID_COUNT 20000
#else
#define MAX_ASTEROID_COUNT 1
#endif

#define POSITION_SCALING_FACTOR 0.001

#define ASTEROID_POSITION            0
#define ASTEROID_VELOCITY            1
#define ASTEROID_VELOCITY_DERIVATIVE 2
#define ASTEROID_VELOCITY_DIVERGENCE 3
#define ASTEROID_SMOOTHING_RADIUS    4
#define ASTEROID_MASS                5
#define ASTEROID_DENSITY             6
#define ASTEROID_DENSITY_DERIVATIVE  7
#define ASTEROID_ENERGY              8
#define ASTEROID_ENERGY_DERIVATIVE   9
#define ASTEROID_PRESSURE            10
#define ASTEROID_SCALE               11

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