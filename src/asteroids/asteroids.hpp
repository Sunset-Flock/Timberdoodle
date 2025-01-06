#pragma once

#include "../timberdoodle.hpp"
#include "../shader_shared/asteroids.inl"

using namespace tido::types;

struct Asteroid
{
    f32vec3 position;
    f32vec3 velocity;
};

struct AsteroidSimulation
{
    AsteroidSimulation();

    void update_asteroids(float const dt);
    auto get_asteroids() const -> std::array<Asteroid, MAX_ASTEROID_COUNT> const &;

    private:
        std::array<Asteroid, MAX_ASTEROID_COUNT> asteroids = {};
};
