#pragma once

#include "../timberdoodle.hpp"
#include "../shader_shared/asteroids.inl"
#include "asteroids_shared.hpp"

using namespace tido::types;

struct Kernel
{
    f32 value(float const dist) const;
    f32 grad_value(float const dist) const;

    f32 smoothing_radius;

    private:
        static constexpr f32 NORMALIZATION = 1.0f / PI;
};

struct Material
{
    Material();
    struct EvaluateRet
    {
        float pressure;
        float speed_of_sound;
    };
    auto evaluate(f32 const density, f32 const energy) -> EvaluateRet;

    private:
        float start_density;
        float A;
        float c;
        float c_p;
};

struct Solver
{
    void integrate(std::array<Asteroid, MAX_ASTEROID_COUNT> & asteroids, f32 const dt);

    private:
        Material material = {};
        Kernel kernel = {};
};