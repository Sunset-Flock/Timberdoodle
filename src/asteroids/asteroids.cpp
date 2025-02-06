#include "asteroids.hpp"
#include "../shader_shared/asteroids.inl"

#include <imgui.h>

#include <algorithm>
#include <random>
#include <chrono>

#include "../ui/widgets/helpers.hpp"
#include <imgui_stdlib.h>

using namespace std::chrono_literals;

struct DistributeAsteroidsInfo
{
    f32vec3 center;
    f32 radius;
    i32 asteroid_count;
    std::vector<f32vec3> * positions;
    std::vector<f32> * smoothing_radii;
    std::vector<f32> * masses;
};

// Distributes the particles using parameterized spiraling method.
auto distribute_particles_in_sphere(DistributeAsteroidsInfo const & info) -> i32
{
    info.positions->reserve(info.asteroid_count);
    f32 const sphere_volume = 1.33333333333f * PI * std::pow(info.radius, 3.0f);
    f32 const h = std::cbrt(sphere_volume / info.asteroid_count);

    // Determine the number of shells
    i32 num_shells = info.radius / h;
    std::vector<f32> shells(num_shells);
    f32 total = 0.0f;

    // Determine the number of particles for each shell.
    for(i32 shell_index = 0; shell_index < num_shells; ++shell_index)
    {
        shells.at(shell_index) = std::pow((shell_index + 1) * h, 2.0f);
        total += shells.at(shell_index);
    }

    f32 const mult = info.asteroid_count / total;
    std::for_each(shells.begin(), shells.end(), [mult](f32 & shell){ shell *= mult; });

    std::mt19937_64 mersenne_engine(1234);
    std::uniform_real_distribution<f32> uniform_distribution;

    // Random sphere sampling helper.
    auto get_random_sphere_dir = [&]() -> f32vec3 {
        f32 const phi = uniform_distribution(mersenne_engine) * 2.0f * PI;
        f32 const z = uniform_distribution(mersenne_engine) * 2.0f - 1.0f;
        f32 const u = std::sqrt(1.0f - pow(z, 2.0f));

        return f32vec3(u * std::cos(phi), u * std::sin(phi), z);
    };

    // Helper to convert from spherical to cartesian coordinates.
    auto spherical_to_cartesian = [](f32 const r, f32 const theta, f32 const phi) -> f32vec3
    {
        return r * f32vec3(
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            std::cos(theta)
        );
    };

    i32 shell_index = 0;
    i32 actually_generated = 0;
    f32 phi = 0.0f;
    // Loop through all shells and distribute particles in each one.
    for(f32 r = h; r <= info.radius; r += h, ++shell_index)
    {
        // Used to offset the particles inside the shell.
        f32 const rotation = 2.0f * PI * uniform_distribution(mersenne_engine);
        f32vec3 const dir = get_random_sphere_dir();
        f32mat3x3 const rotator = glm::rotate(glm::identity<f32mat4x4>(), glm::radians(rotation), dir);

        i32 const m = std::ceil(shells.at(shell_index));
        for(i32 k = 1; k < m; ++k)
        {
            f32 const hk = -1.0f + 2.0f * f32(k) / m;
            f32 const theta = std::acos(hk);
            phi += 3.8f / std::sqrt(m * (1.0f - pow(hk, 2.0f)));
            f32vec3 const pos = info.center + rotator * spherical_to_cartesian(r, theta, phi);
            // It can happen the particle falls outside of the desired radius. We do not accept it in that case.
            if(length(pos - info.center) <= info.radius)
            {
                info.positions->push_back(f32vec3(pos));
                info.smoothing_radii->push_back(h);
                actually_generated += 1;
            }
        }
    }

    // Distribute the masses equally to each of the particle.
    info.masses->reserve(actually_generated);

    f32 const total_mass = sphere_volume * INITAL_DENSITY;

    f32 prelim_mass = 0.0f;
    i32 const start = info.positions->size() - actually_generated;
    for(i32 i = start; i < info.positions->size(); ++i)
    {
        info.masses->push_back(std::pow(info.smoothing_radii->at(i), 3.0f));
        prelim_mass += info.masses->back();
    }

    f32 const normalization = total_mass / prelim_mass;
    for (i32 i = start; i < info.positions->size(); ++i)
    {
        info.masses->at(i) *= normalization;
    }

    return actually_generated;
}

AsteroidSimulation::AsteroidSimulation(ThreadPool * the_threadpool) : threadpool(the_threadpool)
{
    asteroids.simulation_bodies.push_back({
        .position = f32vec3(0.0),
        .velocity_vector = f32vec3(0.0),
        .velocity_magnitude = f32(0.0),
        .radius = PLANET_RADIUS,
        .particle_count = 5000,
        .particle_size = 2.5f,
        .name = "large static asteroid"
    });

    asteroids.simulation_bodies.push_back({
        .position = f32vec3(75'000) / f32vec3(std::sqrt(3)),
        .velocity_vector = normalize(f32vec3(-1.0, -1.0, 0.0)),
        .velocity_magnitude = 3000.0,
        .radius = ASTEROID_RADIUS,
        .particle_count = 2000,
        .particle_size = 1.0f,
        .name = "small dynamic asteroid"
    });

#if CPU_SIMULATION
    last_update_asteroids.simulation_bodies = asteroids.simulation_bodies;
    run_thread = std::thread([=, this]() {AsteroidSimulation::run(); });
#endif
}

AsteroidSimulation::~AsteroidSimulation()
{
    should_run.store(false);
    run_thread.join();
}

void AsteroidSimulation::run()
{
    while(should_run)
    {
        // Do nothing if the simulation is paused.
        if(!simulation_paused.load(std::memory_order_relaxed))
        {
            // Overwrite the time step if user requests.
            if(deduce_timestep.load(std::memory_order_relaxed))
            {
                dt = 0.01;
                for(auto const & velocity_divergence : asteroids.velocity_divergences)
                {
                    f32 const factor = 0.001;
                    if(velocity_divergence > 10e-13)
                    {
                        dt = std::min(dt, factor / velocity_divergence);
                    }
                }
            }
            // Run the simulation.
            solver.integrate(asteroids, dt, *threadpool);
        }

        // Copy the data only if the simulation is running!
        if(simulation_started.load(std::memory_order_relaxed) && !simulation_paused.load(std::memory_order_relaxed))
        {
            // Critical section because the data can be simultaneously read by the main thread.
            std::lock_guard<std::mutex> guard(data_exchange_mutex);
            AsteroidsWrapper::copy(last_update_asteroids, asteroids);
        }

        if(simulation_paused.load(std::memory_order_relaxed))
        {
            // Notify the main thread that we have seen the simulation paused and the data will no longer be in use.
            simulation_paused_ackowledged.store(true, std::memory_order_relaxed);
        }
    }

    return;
}

void AsteroidSimulation::initialize_simulation()
{
    // Clear parameters that might be left from previous simulation run.
    // The derivatives are cleared in the simulation step so we do not need to cover those.
    asteroids.positions.clear();
    asteroids.velocities.clear();
    asteroids.particle_scales.clear();
    asteroids.densities.clear();
    asteroids.energies.clear();
    asteroids.pressures.clear();
    asteroids.masses.clear();
    asteroids.smoothing_radii.clear();

    // Loop through the bodies that should be a part of the simulation.
    for(auto const & simulation_body : asteroids.simulation_bodies)
    {
        // Distribute the particles in the simulation body.
        i32 const generated_particles_count = distribute_particles_in_sphere({
            .center = simulation_body.position,
            .radius = simulation_body.radius,
            .asteroid_count = simulation_body.particle_count,
            .positions = &asteroids.positions,
            .smoothing_radii = &asteroids.smoothing_radii,
            .masses = &asteroids.masses,
        });

        // And fill the particle properties with intial values.
        asteroids.velocities.resize(asteroids.velocities.size() + generated_particles_count);
        std::fill(asteroids.velocities.end() - generated_particles_count, asteroids.velocities.end(), simulation_body.velocity_vector * simulation_body.velocity_magnitude);

        asteroids.particle_scales.resize(asteroids.particle_scales.size() + generated_particles_count);
        std::fill(asteroids.particle_scales.end() - generated_particles_count, asteroids.particle_scales.end(), simulation_body.particle_size);

        asteroids.densities.resize(asteroids.densities.size() + generated_particles_count);
        std::fill(asteroids.densities.end() - generated_particles_count, asteroids.densities.end(), INITAL_DENSITY);

        asteroids.energies.resize(asteroids.energies.size() + generated_particles_count);
        std::fill(asteroids.energies.end() - generated_particles_count, asteroids.energies.end(), 0.0f);

        asteroids.pressures.resize(asteroids.pressures.size() + generated_particles_count);
        std::fill(asteroids.pressures.end() - generated_particles_count, asteroids.pressures.end(), 0.0f);
    }

    // Calculate the maximum smoothing radius used by the spatial acceleration structure.
    asteroids.max_smoothing_radius = std::numeric_limits<f32>::min();
    for(auto const & smoothing_radius : asteroids.smoothing_radii)
    {
        asteroids.max_smoothing_radius = std::max(asteroids.max_smoothing_radius, smoothing_radius);
    }

    // And finally resize both the containers so that fields we did not initalize here are correct size (mainly the derivatives).
#if CPU_SIMULATION
    last_update_asteroids.resize(asteroids.positions.size());
#endif
    asteroids.resize(asteroids.positions.size());
}

// Draw the ui - this function is a bit messy, but it is ui code, can that really be super clean?
void AsteroidSimulation::draw_imgui(AsteroidSettings & settings)
{
    // If the simulation was started user should not be messing with the simluation initial parameters.
    ImGui::BeginDisabled(simulation_started);
    ImGui::SeparatorText("Simulation Setup");
    {
        ImGui::Indent(20.0f);
        // Simulation bodies overview.
        ImGui::SeparatorText("Simulation bodies");
        if(ImGui::BeginTable("Bodies table", 1, ImGuiTableFlags_Borders))
        {
            for(i32 simulation_body_index = 0; simulation_body_index < asteroids.simulation_bodies.size(); ++simulation_body_index)
            {
                auto const & simulation_body = asteroids.simulation_bodies.at(simulation_body_index);
                std::string name = fmt::format("{}##{}", simulation_body.name, simulation_body_index);
                ImGui::TableNextColumn();
                if(ImGui::Selectable(name.c_str(), settings.selected_setup_asteroid == simulation_body_index))
                {
                    settings.selected_setup_asteroid = simulation_body_index;
                }
            }
            ImGui::EndTable();
        }

        bool body_not_selected = settings.selected_setup_asteroid == -1;

        if(ImGui::Button("Add sim body"))
        {
            asteroids.simulation_bodies.push_back({
                .name = "new sim body"
            });
            settings.selected_setup_asteroid = asteroids.simulation_bodies.size() - 1;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(body_not_selected);
        if(ImGui::Button("Remove selected body"))
        {
            asteroids.simulation_bodies.erase(asteroids.simulation_bodies.begin() + settings.selected_setup_asteroid);
            settings.selected_setup_asteroid = -1;
            body_not_selected = true;
        }
        ImGui::EndDisabled();

        SimulationBodyInfo default_sim_body = {};
        auto & selected_simulation_body = body_not_selected ? default_sim_body : asteroids.simulation_bodies.at(settings.selected_setup_asteroid);


        // Properties of the individual body.
        ImGui::SeparatorText("Selected body properties");
        ImGui::BeginDisabled(body_not_selected);
        {
            ImGui::InputText("Name", &selected_simulation_body.name);
            ImGui::SliderFloat3("Position", s_cast<f32*>(&selected_simulation_body.position.x), -200'000.0f, 200'000.0f);
            ImGui::InputFloat("Radius", &selected_simulation_body.radius);
            ImGui::InputInt("Particle count", &selected_simulation_body.particle_count, 100, 1000);
            ImGui::InputFloat("Visual particle size", &selected_simulation_body.particle_size);

            ImGui::SliderFloat3("Velocity vector", s_cast<f32*>(&selected_simulation_body.velocity_vector.x), -1.0f, 1.0f);
            ImGui::InputFloat("Velocity magnitude", &selected_simulation_body.velocity_magnitude);
            selected_simulation_body.velocity_vector = glm::length(selected_simulation_body.velocity_vector) ? 
                glm::normalize(selected_simulation_body.velocity_vector) : 
                selected_simulation_body.velocity_vector;
            f32 magnitude = selected_simulation_body.velocity_magnitude;
        }
        ImGui::EndDisabled();

        last_update_asteroids.simulation_bodies = asteroids.simulation_bodies;
        ImGui::Unindent(20.0f);
    }
    ImGui::EndDisabled();


    ImGui::SeparatorText("Simulation run settings");
    auto modes = std::array{
        "NONE",
        "VELOCITY",
        "ACCELERATION",
        "VELOCITY DIVERGENCE",
        "PRESSURE",
        "DENSITY",
    };
    ImGui::Combo("debug visualization", &settings.debug_draw_mode, modes.data(), modes.size());
    bool deduce_timestep_tmp = deduce_timestep;
    ImGui::Checkbox("Deduce timestep", &deduce_timestep_tmp);
    deduce_timestep.store(deduce_timestep_tmp, std::memory_order_relaxed);

    ImGui::BeginDisabled(deduce_timestep_tmp);
    f32 tmp = dt;
    ImGui::SliderFloat("Manual timestep", &tmp, 0.001, 0.01);
    dt = tmp;
    ImGui::EndDisabled();

    // A bit convoluted logic here but let me try to explain.
    std::string_view button_text = simulation_paused ? "Start simulation" : "Pause simulation";

    if(ImGui::Button(button_text.data()))
    {
        if(simulation_paused)
        {
            if(!simulation_started)
            {
                // If the simluation is just starting we need to initalize it first.
                initialize_simulation();
            }
            // If the simulation was already start (and thus just paused) this does nothing,
            // otherwise this stores the information that the simulation was started until the user decides to restart it again.
            simulation_started.store(true, std::memory_order_relaxed);
            asteroids.simulation_started = true;
            settings.selected_setup_asteroid = -1;
        }
        // Flip the started/paused state.
        simulation_paused.store(!simulation_paused, std::memory_order_relaxed);
    }
    ImGui::SameLine();
    if(ImGui::Button("Reset simulation"))
    {
        // Reset the simulation acknowledged.
        simulation_paused_ackowledged.store(false, std::memory_order_relaxed);
        // Notify the simulation thread that the simulation should be paused.
        simulation_paused.store(true, std::memory_order_relaxed);

#if CPU_SIMULATION
        // Wait for the simulation thread to stop using the data - complete it's simulation step.
        while(!simulation_paused_ackowledged.load())
        {
            std::this_thread::sleep_for(7ms);
        }
#endif

        // We are back to initialization state - reset the simluation started flag.
        simulation_started.store(false, std::memory_order_relaxed);
        asteroids.simulation_started = false;
        last_update_asteroids.simulation_started = false;
    }
}

auto AsteroidSimulation::get_asteroids() -> AsteroidsWrapper
{
#if CPU_SIMULATION
    std::lock_guard<std::mutex> guard(data_exchange_mutex);
    return last_update_asteroids;
#else
    return asteroids;
#endif
}
