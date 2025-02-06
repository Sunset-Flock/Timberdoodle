#pragma once

#include "../timberdoodle.hpp"
#include "../shader_shared/asteroids.inl"
#include "asteroids_shared.hpp"
#include "../multithreading/thread_pool.hpp"

using namespace tido::types;

struct Kernel
{
    f32 value(f32 const dist) const;
    f32 grad_value(f32 const dist) const;

    f32 smoothing_radius;

    private:
        static constexpr f32 NORMALIZATION = 1.0f / PI;
};

struct Material
{
    Material();
    struct EvaluateRet
    {
        f32 pressure;
        f32 speed_of_sound;
    };
    auto evaluate(f32 const density, f32 const energy) const -> EvaluateRet;

    private:
        f32 start_density;
        f32 A;
        f32 c;
        f32 c_p;
};

struct Solver
{
    void integrate(AsteroidsWrapper & asteroids, f32 const dt, ThreadPool & threadpool);

    private:
        Material material = {};
        Kernel kernel = {};
};