#pragma once

#include "../timberdoodle.hpp"
#include "../shader_shared/asteroids.inl"
#include "asteroids_shared.hpp"
#include "solver.hpp"

#include <mutex>

using namespace tido::types;

struct SimulationBodyInfo
{
    f32vec3 position;
    f32vec3 velocity;
    f32 radius;
    i32 particle_count;
    f32 particle_size;
};

struct AsteroidSimulation
{
    std::mutex data_exchange_mutex;

    AsteroidSimulation(ThreadPool * the_threadpool);
    ~AsteroidSimulation();

    auto get_asteroids() -> AsteroidsWrapper;
    void draw_imgui();

    void add_simulation_body(SimulationBodyInfo const & info);
    
    private:
        ThreadPool * threadpool;
        f32 speed_multiplier = 1.0f;
        AsteroidsWrapper asteroids = {};
        AsteroidsWrapper last_update_asteroids = {};
        Solver solver = {};
        f64 dt = 0.0000001;

        std::atomic_bool should_run = true;
        std::atomic_bool simulation_paused = true;
        std::atomic_bool deduce_timestep = true;
        std::atomic_bool simulation_started = false;

        std::vector<SimulationBodyInfo> simulation_bodies = {};

        std::thread run_thread;

        void initialize_simulation();
        void run();
};
