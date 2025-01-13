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

    AsteroidSimulation();
    ~AsteroidSimulation();

    void run();
    auto get_asteroids() -> std::vector<Asteroid>;
    void draw_imgui();
    
    private:
        f32 speed_multiplier = 1.0f;
        std::vector<Asteroid> asteroids = {};
        std::vector<Asteroid> last_update_asteroids = {};
        Solver solver = {};
        f64 dt = 0.0000001;

        std::thread run_thread;

        void explicit_euler_step();
        void update_asteroids(f64 const dt);
        void integrate_derivatives();
};
