#pragma once

#include <daxa/daxa.inl>

#define MAX_ASTEROID_COUNT 1000
#define DOMAIN_BOUNDS 1000000

struct GPUAsteroid
{
    daxa_f32vec3 position;
};