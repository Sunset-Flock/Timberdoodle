#include "asteroids.hpp"
#include "../shader_shared/asteroids.inl"

#include <imgui.h>

#include <algorithm>
#include <random>

struct DistributeAsteroidsInfo
{
    f64vec3 center;
    f64 radius;
    i32 asteroid_count;
};

auto distribute_asteroids_in_sphere(DistributeAsteroidsInfo const & info) -> std::vector<f64vec4>
{
    std::vector<f64vec4> positions = {};
    positions.reserve(info.asteroid_count);

    f64 const sphere_volume = 1.33333333333f * PI * std::pow(info.radius, 3.0f);
    f64 const h = std::cbrt(sphere_volume / info.asteroid_count);

    i32 num_shells = info.radius / h;
    std::vector<f64> shells(num_shells);
    f64 total = 0.0f;

    for(i32 shell_index = 0; shell_index < num_shells; ++shell_index)
    {
        shells.at(shell_index) = std::pow((shell_index + 1) * h, 2.0f);
        total += shells.at(shell_index);
    }

    f64 const mult = info.asteroid_count / total;
    std::for_each(shells.begin(), shells.end(), [mult](f64 & shell){ shell *= mult; });

    std::mt19937_64 mersenne_engine(1234);
    std::uniform_real_distribution<f64> uniform_distribution;

    i32 shell_index = 0;
    auto get_random_sphere_dir = [&]() -> f64vec3 {
        f64 const phi = uniform_distribution(mersenne_engine) * 2.0f * PI;
        f64 const z = uniform_distribution(mersenne_engine) * 2.0f - 1.0f;
        f64 const u = std::sqrt(1.0f - pow(z, 2.0f));

        return f64vec3(u * std::cos(phi), u * std::sin(phi), z);
    };

    auto spherical_to_cartesian = [](f64 const r, f64 const theta, f64 const phi) -> f64vec3
    {
        return r * f64vec3(
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            std::cos(theta)
        );
    };

    f64 phi = 0.0f;
    for(f64 r = h; r <= info.radius; r += h, ++shell_index)
    {
        f64 const rotation = 2.0f * PI * uniform_distribution(mersenne_engine);
        f64vec3 const dir = get_random_sphere_dir();
        f64mat3x3 const rotator = glm::rotate(glm::identity<f64mat4x4>(), glm::radians(rotation), dir);

        i32 const m = std::ceil(shells.at(shell_index));
        for(i32 k = 1; k < m; ++k)
        {
            f64 const hk = -1.0f + 2.0f * f64(k) / m;
            f64 const theta = std::acos(hk);
            phi += 3.8f / std::sqrt(m * (1.0f - pow(hk, 2.0f)));
            f64vec3 const pos = info.center + rotator * spherical_to_cartesian(r, theta, phi);
            if(length(pos - info.center) <= info.radius)
            {
                positions.push_back(f64vec4(pos, h));
            }
        }
    }

    return positions;
}

auto get_asteroid_masses(std::vector<f64vec4> const & positions, const f64 total_mass) -> std::vector<f64>
{
    std::vector<f64> masses = {};
    masses.reserve(positions.size());

    f64 prelim_mass = 0.0f;
    for(i32 i = 0; i < positions.size(); ++i)
    {
        masses.push_back(std::pow(positions.at(i)[3], 3.0f));
        prelim_mass += masses.back();
    }

    f64 const normalization = total_mass / prelim_mass;
    for (i32 i = 0; i < positions.size(); ++i)
    {
        masses.at(i) *= normalization;
    }
    return masses;
}

AsteroidSimulation::AsteroidSimulation()
{
    std::vector<f64vec4> asteroid_positions = distribute_asteroids_in_sphere({
        .center = f64vec3(0.0f),
        .radius = PLANET_RADIUS,
        .asteroid_count = 5000,
    });

    f64 const planet_sphere_volume = 1.33333333333f * PI * std::pow(PLANET_RADIUS, 3.0f);
    f64 const planet_total_mass = planet_sphere_volume * INITAL_DENSITY;
    std::vector<f64> asteroid_masses = get_asteroid_masses(asteroid_positions, planet_total_mass);

    std::vector<f64vec4> collider_positions = distribute_asteroids_in_sphere({
        .center = f64vec3(75'000) / f64vec3(std::sqrt(3)),
        .radius = ASTEROID_RADIUS,
        .asteroid_count = 2000,
    });

    f64 const collider_sphere_volume = 1.33333333333f * PI * std::pow(ASTEROID_RADIUS, 3.0f);
    f64 const collider_total_mass = collider_sphere_volume * INITAL_DENSITY;
    std::vector<f64> collider_masses = get_asteroid_masses(asteroid_positions, collider_total_mass);

    u32 const total_positions_count = asteroid_positions.size() + collider_positions.size();

    last_update_asteroids.resize(total_positions_count);
    asteroids.resize(total_positions_count);

    asteroids.max_smoothing_radius = std::numeric_limits<f64>::min();

    for(i32 asteroid_index = 0; asteroid_index < total_positions_count; ++asteroid_index)
    {
        bool const use_planet_params = asteroid_index < asteroid_positions.size();
        i32 const real_index = use_planet_params ? asteroid_index : asteroid_index - asteroid_positions.size();
        asteroids.positions.at(asteroid_index) = (use_planet_params ? asteroid_positions.at(real_index) : collider_positions.at(real_index));
        // asteroids.velocities.push_back(use_planet_params ? f64vec3(0.0) : 3000.0 * normalize(f64vec3(-5000) + f64vec3(0.0, 0.0, 5000.0)));
        asteroids.velocities.at(asteroid_index) = use_planet_params ? f64vec3(0.0) : 3000.0 * normalize(f64vec3(-5000));
        asteroids.smoothing_radii.at(asteroid_index) = use_planet_params ? asteroid_positions.at(real_index).w : collider_positions.at(real_index).w;
        asteroids.masses.at(asteroid_index) = use_planet_params ? asteroid_masses.at(real_index) : collider_masses.at(real_index);
        asteroids.particle_scales.at(asteroid_index) = use_planet_params ? 2.5f : 1.0f;

        asteroids.max_smoothing_radius = std::max(asteroids.max_smoothing_radius, asteroids.smoothing_radii.at(asteroid_index));
    }
    std::fill(asteroids.densities.begin(), asteroids.densities.end(), INITAL_DENSITY);
    std::fill(asteroids.energies.begin(), asteroids.energies.end(), 0.0);
    std::fill(asteroids.pressures.begin(), asteroids.pressures.end(), 0.0);

    run_thread = std::thread([=, this]() {AsteroidSimulation::run(); });
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
        update_asteroids(dt);
        // dt = 0.01;
        // for(auto const & asteroid : asteroids)
        // {
        //     f64 const factor = 0.001;
        //     if(asteroid.velocity_divergence > 10e-13)
        //     {
        //         dt = std::min(dt, factor / asteroid.velocity_divergence);
        //     }
        // }
        // dt = std::max(dt, 0.001);
        {
            std::lock_guard<std::mutex> guard(data_exchange_mutex);
            AsteroidsWrapper::copy(last_update_asteroids, asteroids);
        }
    }

    return;
}

void AsteroidSimulation::update_asteroids(f64 const dt)
{
    solver.integrate(asteroids, dt);
}

void AsteroidSimulation::draw_imgui()
{
    f32 tmp = dt;
    ImGui::SliderFloat("dt", &tmp, 0.001, 0.01);
    dt = tmp;
}

auto AsteroidSimulation::get_asteroids() -> AsteroidsWrapper
{
    std::lock_guard<std::mutex> guard(data_exchange_mutex);
    return last_update_asteroids;
}
