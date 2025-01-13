#pragma once

#include "../timberdoodle.hpp"

using namespace tido::types;

#define PI 3.1415926535897932384f
#define INITAL_DENSITY 2700.0f

#define PLANET_RADIUS 50'000.0f
#define ASTEROID_RADIUS 10'000.0f

struct Asteroid
{
    f64vec3 position = {};
    f64vec3 velocity = {};
    f64vec3 velocity_derivative = {};
    f64 velocity_divergence = {};

    f64 smoothing_radius = {};
    f64 mass = {};

    f64 density = {};
    f64 density_derivative = {};

    f64 energy = {};
    f64 energy_derivative = {};

    f64 pressure = {};

    f64 speed_of_sound = {};
    f64vec3 gradient = {};

    float particle_scale = {};
};