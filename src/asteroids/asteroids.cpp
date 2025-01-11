#include "asteroids.hpp"
#include "../shader_shared/asteroids.inl"

#include <imgui.h>

#include <algorithm>
#include <random>

struct DistributeAsteroidsInfo
{
    f32vec3 center;
    f32 radius;
    i32 asteroid_count;
};

auto distribute_asteroids_in_sphere(DistributeAsteroidsInfo const & info) -> std::vector<f32vec4>
{
    std::vector<f32vec4> positions = {};
    positions.reserve(info.asteroid_count);

    f32 const sphere_volume = 1.33333333333f * PI * std::pow(info.radius, 3.0f);
    f32 const h = std::cbrt(sphere_volume / info.asteroid_count);

    i32 num_shells = info.radius / h;
    std::vector<f32> shells(num_shells);
    f32 total = 0.0f;

    for(i32 shell_index = 0; shell_index < num_shells; ++shell_index)
    {
        shells.at(shell_index) = std::pow((shell_index + 1) * h, 2.0f);
        total += shells.at(shell_index);
    }

    f32 const mult = info.asteroid_count / total;
    std::for_each(shells.begin(), shells.end(), [mult](f32 & shell){ shell *= mult; });

    std::mt19937_64 mersenne_engine(1234);
    std::uniform_real_distribution<f32> uniform_distribution;

    i32 shell_index = 0;
    auto get_random_sphere_dir = [&]() -> f32vec3 {
        f32 const phi = uniform_distribution(mersenne_engine) * 2.0f * PI;
        f32 const z = uniform_distribution(mersenne_engine) * 2.0f - 1.0f;
        f32 const u = std::sqrt(1.0f - pow(z, 2.0f));

        return f32vec3(u * std::cos(phi), u * std::sin(phi), z);
    };

    auto spherical_to_cartesian = [](f32 const r, f32 const theta, f32 const phi) -> f32vec3
    {
        return r * f32vec3(
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            std::cos(theta)
        );
    };

    f32 phi = 0.0f;
    for(f32 r = h; r <= info.radius; r += h, ++shell_index)
    {
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
            if(length(pos - info.center) <= info.radius)
            {
                positions.push_back(f32vec4(pos, h));
            }
        }
    }

    return positions;
}

auto get_asteroid_masses(std::vector<f32vec4> const & positions, const f32 total_mass) -> std::vector<f32>
{
    std::vector<f32> masses = {};
    masses.reserve(positions.size());

    f32 prelim_mass = 0.0f;
    for(i32 i = 0; i < positions.size(); ++i)
    {
        masses.push_back(std::pow(positions.at(i)[3], 3.0f));
        prelim_mass += masses.back();
    }

    f32 const normalization = total_mass / prelim_mass;
    for (i32 i = 0; i < positions.size(); ++i)
    {
        masses.at(i) *= normalization;
    }
    return masses;
}

AsteroidSimulation::AsteroidSimulation()
{
    std::vector<f32vec4> asteroid_positions = distribute_asteroids_in_sphere({
        .center = f32vec3(0.0f),
        .radius = PLANET_RADIUS,
        .asteroid_count = 900,
    });

    f32 const planet_sphere_volume = 1.33333333333f * PI * std::pow(PLANET_RADIUS, 3.0f);
    f32 const planet_total_mass = planet_sphere_volume * INITAL_DENSITY;
    std::vector<f32> asteroid_masses = get_asteroid_masses(asteroid_positions, planet_total_mass);

    std::vector<f32vec4> collider_positions = distribute_asteroids_in_sphere({
        .center = f32vec3(50'200) / f32vec3(std::sqrt(3)),
        .radius = ASTEROID_RADIUS,
        .asteroid_count = 100,
    });

    f32 const collider_sphere_volume = 1.33333333333f * PI * std::pow(ASTEROID_RADIUS, 3.0f);
    f32 const collider_total_mass = collider_sphere_volume * INITAL_DENSITY;
    std::vector<f32> collider_masses = get_asteroid_masses(asteroid_positions, collider_total_mass);

    for(i32 asteroid_index = 0; asteroid_index < asteroid_positions.size() + collider_positions.size(); ++asteroid_index)
    {
        bool const use_planet_params = asteroid_index < asteroid_positions.size();
        i32 const real_index = use_planet_params ? asteroid_index : asteroid_index - asteroid_positions.size();
        asteroids.at(asteroid_index) = {
            .position = use_planet_params ? asteroid_positions.at(real_index) : collider_positions.at(real_index),
            .velocity = use_planet_params ? f32vec3(0.0f) : 5000.0f * -normalize(f32vec3(5000)),
            .smoothing_radius = use_planet_params ? asteroid_positions.at(real_index).w : collider_positions.at(real_index).w,
            .mass = use_planet_params ? asteroid_masses.at(real_index) : collider_masses.at(real_index),
            .density = INITAL_DENSITY,
            .energy = 0.0f
        };
    }

}

void AsteroidSimulation::update_asteroids(float const dt)
{
    solver.integrate(asteroids, dt * 0.01);
}

void AsteroidSimulation::draw_imgui()
{
    ImGui::SliderFloat("speed multiplier", &speed_multiplier, 0.0f, 10.0f);
}

auto AsteroidSimulation::get_asteroids() const -> std::array<Asteroid, MAX_ASTEROID_COUNT> const &
{
    return asteroids;
}
