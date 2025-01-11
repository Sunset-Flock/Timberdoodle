#pragma once

#include "../timberdoodle.hpp"

using namespace tido::types;

#define PI 3.1415926535897932384f
#define INITAL_DENSITY 2700.0f

#define PLANET_RADIUS 50'000.0f
#define ASTEROID_RADIUS 5'000.0f

struct Asteroid
{
    f32vec3 position;
    f32vec3 velocity;
    f32vec3 velocity_derivative;
    f32 velocity_divergence;

    f32 smoothing_radius;
    f32 mass;

    f32 density;
    f32 density_derivative;

    f32 energy;
    f32 energy_derivative;

    f32 pressure;

    f32 speed_of_sound;
    f32vec3 gradient;
};