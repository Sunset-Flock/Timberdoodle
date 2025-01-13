#include "solver.hpp"

f64 Kernel::value(f64 const dist) const
{
    DBG_ASSERT_TRUE_M(dist >= 0, "q must be not negative");
    f64 const q = dist / smoothing_radius;

    if(q < 1.0) {
        return NORMALIZATION * (0.25 * std::pow(2.0 - q, 3.0) - std::pow(1.0 - q, 3.0));
    }
    else if (q < 2.0) {
        return NORMALIZATION * (0.25 * std::pow(2.0 - q, 3.0));
    }

    return 0.0f;
}

f64 Kernel::grad_value(f64 const dist) const
{
    f64 const q = dist / smoothing_radius;

    if (q == 0.0) {
        return -3.0f * NORMALIZATION;
    }
    else if (q < 1.0) {
        return (1.0 / q) * NORMALIZATION * (-0.75 * std::pow(2.0 - q, 2.0) + 3.0 * std::pow(1.0 - q, 2.0));
    }
    else if (q < 2.0) {
        return (1.0 / q) * NORMALIZATION * (-0.75 * std::pow(2.0 - q, 2.0));
    }
   return 0.0;
}

Material::Material()
{
    c = 2.0f;
    start_density = INITAL_DENSITY;
    A = 26700000000.0f;
    c_p = 700.0f;
}

auto Material::evaluate(f64 const density, f64 const energy) -> EvaluateRet
{
    const f64 mu = density / start_density - 1.0f;
    const f64 pressure = c * density * energy + A * mu;
    const f64 speed_of_sound = std::sqrt(A / start_density);

    return {
        .pressure = pressure,
        .speed_of_sound = speed_of_sound
    };
}


void Solver::integrate(std::vector<Asteroid> & asteroids, f64 const dt)
{
    // Zero out derivatives from last frame.
    for(i32 asteroid_idx = 0; asteroid_idx < asteroids.size(); ++asteroid_idx)
    {
        auto & asteroid = asteroids.at(asteroid_idx);
        asteroid.energy_derivative = 0.0f;
        asteroid.density_derivative = 0.0f;
        asteroid.velocity_derivative = f32vec3(0.0f);
        asteroid.velocity_divergence *= 0.7f;

        auto const & ret = material.evaluate(asteroid.density, asteroid.energy);
        asteroid.pressure = ret.pressure;
        asteroid.speed_of_sound = ret.speed_of_sound;
    }


    for(i32 asteroid_idx = 0; asteroid_idx < asteroids.size(); ++asteroid_idx)
    {
        auto & asteroid = asteroids.at(asteroid_idx);
        for(i32 neighbor_asteroid_idx = 0; neighbor_asteroid_idx < asteroids.size(); ++neighbor_asteroid_idx)
        {
            auto const & neighbor_asteroid = asteroids.at(neighbor_asteroid_idx);
            kernel.smoothing_radius = 0.5f * (asteroid.smoothing_radius + neighbor_asteroid.smoothing_radius);

            f64 const asteroid_distance = glm::length(asteroid.position - neighbor_asteroid.position);

            bool const asteroids_same = neighbor_asteroid_idx == asteroid_idx;
            bool const asteroids_too_far = asteroid_distance > (kernel.smoothing_radius * 2.0f);

            if(asteroids_same || asteroids_too_far) { continue; }

            f64vec3 const asteroid_to_neighbor = asteroid.position - neighbor_asteroid.position;
            asteroid.gradient = asteroid_to_neighbor * std::pow(1.0f / kernel.smoothing_radius, 5.0f) * kernel.grad_value(asteroid_distance);

            // Velocity derivative
            {
                f64vec3 const force = (
                    (asteroid.pressure / std::pow(asteroid.density, 2.0)) + 
                    (neighbor_asteroid.pressure / std::pow(neighbor_asteroid.density, 2.0))) * asteroid.gradient * -1.0;
            
                asteroid.velocity_derivative += neighbor_asteroid.mass * force;
            }
            // Velocity divergence
            {
                f64 dv = glm::dot(neighbor_asteroid.velocity - asteroid.velocity, asteroid.gradient);
                asteroid.velocity_divergence += neighbor_asteroid.mass / asteroid.density * dv;
            }
        }
    }

    for(i32 asteroid_idx = 0; asteroid_idx < asteroids.size(); ++asteroid_idx)
    {
        auto & asteroid = asteroids.at(asteroid_idx);
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