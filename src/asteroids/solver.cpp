#include "solver.hpp"

float Kernel::value(f32 const dist) const
{
    DBG_ASSERT_TRUE_M(dist >= 0, "q must be not negative");
    f32 const q = dist / smoothing_radius;

    if(q < 1.0f) {
        return NORMALIZATION * (0.25f * std::pow(2.0f - q, 3.0f) - std::pow(1.0f - q, 3.0f));
    }
    else if (q < 2.0f) {
        return NORMALIZATION * (0.25f * std::pow(2.0f - q, 3.0f));
    }

    return 0.0f;
}

float Kernel::grad_value(f32 const dist) const
{
    f32 const q = dist / smoothing_radius;

    if (q == 0.0f) {
        return -3.0f * NORMALIZATION;
    }
    else if (q < 1.0f) {
        return (1.0f / q) * NORMALIZATION * (-0.75f * std::pow(2.0f - q, 2.0f) + 3.0f * std::pow(1.0f - q, 2.0f));
    }
    else if (q < 2.0f) {
        return (1.0f / q) * NORMALIZATION * (-0.75f * std::pow(2.0f - q, 2.0f));
    }
    return 0.0f;
}

Material::Material()
{
    c = 2.0f;
    start_density = INITAL_DENSITY;
    A = 26700000000.0f;
    c_p = 700.0f;
}

auto Material::evaluate(f32 const density, f32 const energy) -> EvaluateRet
{
    const f32 mu = density / start_density - 1.0f;
    const f32 pressure = c * density * energy + A * mu;
    const f32 speed_of_sound = std::sqrt(A / start_density);
    return {
        .pressure = pressure,
        .speed_of_sound = speed_of_sound
    };
}


void Solver::integrate(std::array<Asteroid, MAX_ASTEROID_COUNT> & asteroids, f32 const dt)
{
    for(i32 asteroid_idx = 0; asteroid_idx < 994; ++asteroid_idx)
    {
        auto & asteroid = asteroids.at(asteroid_idx);
        auto const & ret = material.evaluate(asteroid.density, asteroid.energy);
        // asteroid.density_derivative = 0.0f;
        // asteroid.velocity_divergence = 0.0f;
        // asteroid.energy_derivative = 0.0f;

        asteroid.pressure = ret.pressure;
        asteroid.speed_of_sound = ret.speed_of_sound;

        for(i32 neighbor_asteroid_idx = 0; neighbor_asteroid_idx < asteroids.size(); ++neighbor_asteroid_idx)
        {
            auto const & neighbor_asteroid = asteroids.at(neighbor_asteroid_idx);
            kernel.smoothing_radius = 0.5f * (asteroid.smoothing_radius + neighbor_asteroid.smoothing_radius);

            f32 const asteroid_distance = glm::length(asteroid.position - neighbor_asteroid.position);

            bool const asteroids_same = neighbor_asteroid_idx == asteroid_idx;
            bool const asteroids_too_far = asteroid_distance > kernel.smoothing_radius;

            if(asteroids_same || asteroids_too_far) { continue; }

            f32vec3 const asteroid_to_neighbor = asteroid.position - neighbor_asteroid.position;
            asteroid.gradient = asteroid_to_neighbor * std::pow(1.0f / kernel.smoothing_radius, 5.0f) * kernel.grad_value(asteroid_distance);

            // Velocity derivative
            {
                f32vec3 const force = (
                    (asteroid.pressure / std::pow(asteroid.density, 2.0f)) + 
                    (neighbor_asteroid.pressure / std::pow(neighbor_asteroid.density, 2.0f))) * asteroid.gradient * -1.0f;
            
                asteroid.velocity_derivative += neighbor_asteroid.mass * force;
            }
            // Velocity divergence
            {
                f32 dv = glm::dot(neighbor_asteroid.velocity - asteroid.velocity, asteroid.gradient);
                asteroid.velocity_divergence += neighbor_asteroid.mass / asteroid.density * dv;
            }
        }

        if(asteroid.velocity_derivative != f32vec3(0.0f))
        {
            int i = 0;
        }
        // Calculate continuity equation
        asteroid.density_derivative += -asteroid.density * asteroid.velocity_divergence;
        // Calculate pressure force
        asteroid.energy_derivative -= asteroid.pressure / asteroid.density * asteroid.velocity_divergence;

        // Second order step
        asteroid.velocity += asteroid.velocity_derivative * dt;
        asteroid.position += asteroid.velocity * dt;

        // First order step
        asteroid.density += asteroid.density_derivative * dt;
        asteroid.energy += asteroid.energy_derivative * dt;
    }
}