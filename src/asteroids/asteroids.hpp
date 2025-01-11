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
    void draw_imgui();
    
    private:
        f32 speed_multiplier = 1.0f;
        std::array<Asteroid, MAX_ASTEROID_COUNT> asteroids = {};
};
