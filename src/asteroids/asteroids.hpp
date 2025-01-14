#pragma once

#include "../timberdoodle.hpp"
#include "../shader_shared/asteroids.inl"
#include "asteroids_shared.hpp"
#include "solver.hpp"

#include <mutex>

using namespace tido::types;

struct AsteroidSimulation
{
    std::atomic_bool should_run = true;
    std::mutex data_exchange_mutex;

    AsteroidSimulation(ThreadPool * the_threadpool);
    ~AsteroidSimulation();

    void run();
    auto get_asteroids() -> AsteroidsWrapper;
    void draw_imgui();
    
    private:
        ThreadPool * threadpool;
        f32 speed_multiplier = 1.0f;
        AsteroidsWrapper asteroids = {};
        AsteroidsWrapper last_update_asteroids = {};
        Solver solver = {};
        f64 dt = 0.0000001;

        std::thread run_thread;

        void explicit_euler_step();
        void update_asteroids(f64 const dt);
        void integrate_derivatives();
};
