#pragma once

#include "../timberdoodle.hpp"
#include "../shader_shared/asteroids.inl"

using namespace tido::types;

struct Asteroid
{
    f32vec3 position;
    f32vec3 velocity;
    f32vec3 force;
    f32 mass;
    f32 pressure;
    f32 density;
};

struct AsteroidSimulation
{
    AsteroidSimulation();
    void update_asteroids(float const dt);
    auto get_asteroids() const -> std::array<Asteroid, MAX_ASTEROID_COUNT> const &;
    
    private:
        void calculateDensityAndPressure(Asteroid & asteroid);
        void calculateForce(Asteroid & asteroid);
        void calculateSPH(Asteroid & asteroid);
        void setConstants();
    private:
        std::array<Asteroid, MAX_ASTEROID_COUNT> asteroids = {};

        f32 poly6;
        f32 spikyGrad;
        f32 spikyLap;
        f32 gasConstant;
        f32 mass;
        f32 h2;
        f32 selfDensity;
        f32 restDensity;
        f32 viscosity;
        f32 h;
        //f32 g;
        f32 tension;
        f32 massPoly6Product;
};
