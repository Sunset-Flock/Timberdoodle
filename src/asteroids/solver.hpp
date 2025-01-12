#pragma once

#include "../timberdoodle.hpp"
#include "../shader_shared/asteroids.inl"
#include "asteroids_shared.hpp"

using namespace tido::types;

struct Kernel
{
    f64 value(f64 const dist) const;
    f64 grad_value(f64 const dist) const;

    f64 smoothing_radius;

    private:
        static constexpr f64 NORMALIZATION = 1.0f / PI;
};

struct Material
{
    Material();
    struct EvaluateRet
    {
        f64 pressure;
        f64 speed_of_sound;
    };
    auto evaluate(f64 const density, f64 const energy) -> EvaluateRet;

    private:
        f64 start_density;
        f64 A;
        f64 c;
        f64 c_p;
};

struct Solver
{
    void integrate(std::vector<Asteroid> & asteroids, f64 const dt);

    private:
        Material material = {};
        Kernel kernel = {};
};