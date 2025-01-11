#include "asteroids.hpp"
#include "../shader_shared/asteroids.inl"

#include <imgui.h>

#include <random>
#include <algorithm>

AsteroidSimulation::AsteroidSimulation()
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    std::uniform_real_distribution<f32> position_distribution{ -DOMAIN_BOUNDS, DOMAIN_BOUNDS};
    std::uniform_real_distribution<f32> velocity_distribution{ -3.0f, 3.0f};

    auto gen = [&]() -> Asteroid {
        return {
            .position = {
                position_distribution(mersenne_engine),
                position_distribution(mersenne_engine),
                position_distribution(mersenne_engine)
            },
            .velocity = {
                velocity_distribution(mersenne_engine),
                velocity_distribution(mersenne_engine),
                velocity_distribution(mersenne_engine)
            }
        };
    };

    std::generate(asteroids.begin(), asteroids.end(), gen);
}

void AsteroidSimulation::update_asteroids(float const dt)
{
    auto asteroid_in_bounds = [](Asteroid const & asteroid) -> bool
    {
        return std::abs(asteroid.position.x) < DOMAIN_BOUNDS &&
               std::abs(asteroid.position.y) < DOMAIN_BOUNDS &&
               std::abs(asteroid.position.z) < DOMAIN_BOUNDS;
    };

    for(auto & asteroid : asteroids)
    {
        asteroid.position += asteroid.velocity * dt * speed_multiplier;
        if(!asteroid_in_bounds(asteroid))
        {
            asteroid.position = glm::clamp(asteroid.position, f32vec3(-DOMAIN_BOUNDS), f32vec3(DOMAIN_BOUNDS));
            asteroid.velocity = -asteroid.velocity;
        }
    }
}

void AsteroidSimulation::draw_imgui()
{
    ImGui::SliderFloat("speed multiplier", &speed_multiplier, 0.0f, 10.0f);
}

auto AsteroidSimulation::get_asteroids() const -> std::array<Asteroid, MAX_ASTEROID_COUNT> const &
{
    return asteroids;
}
