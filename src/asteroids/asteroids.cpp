#include "asteroids.hpp"
#include "../shader_shared/asteroids.inl"

#include <imgui.h>

#include <random>
#include <algorithm>

#define PI 3.14159265f
#define EPSILON 0.000001f
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

    setConstants();
}


void AsteroidSimulation::calculateDensityAndPressure(Asteroid & asteroid)
{
    f32 acumulatingAsteroidDensity = 0.f;
    for(auto & astNeighbour : asteroids)
    {
        f32 dist = glm::length(astNeighbour.position - asteroid.position);
        if(dist < EPSILON) continue;
        if (dist < h) {

            acumulatingAsteroidDensity += massPoly6Product * glm::pow(h2 - (dist * dist), 3);
        }
    }

    // Include self density (as itself isn't included in neighbour)
    asteroid.density = acumulatingAsteroidDensity + selfDensity;

    // 
    asteroid.pressure = gasConstant * (asteroid.density - restDensity);
}

void AsteroidSimulation::calculateForce(Asteroid & asteroid)
{
    for(auto & astNeighbour : asteroids)
    {
        f32 dist = glm::length(astNeighbour.position - asteroid.position);
        if(dist < 0.00001f) continue;
        if (dist < h) {

            //unit direction and length
            f32vec3 dir = glm::normalize(astNeighbour.position - asteroid.position);

            //apply pressure sforce
            f32vec3 pressureForce = -dir * mass * (asteroid.pressure + astNeighbour.pressure) / (2 * astNeighbour.density) * spikyGrad;
            pressureForce *= std::pow(h - dist, 2);
            asteroid.force += pressureForce;

            //apply viscosity force
            f32vec3 velocityDif = astNeighbour.velocity - asteroid.velocity;
            f32vec3 viscoForce = viscosity * mass * (velocityDif / astNeighbour.density) * spikyLap * (h - dist);
            asteroid.force += viscoForce;
        }
    }
}
void AsteroidSimulation::calculateSPH(Asteroid & asteroid)
{
    calculateDensityAndPressure(asteroid);
    calculateForce(asteroid);
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
        calculateSPH(asteroid);
        f32vec3 acceleration = asteroid.force / asteroid.density;
        asteroid.velocity += acceleration * dt;

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
    ImGui::SliderFloat("speed multiplier", &speed_multiplier, 0.0f, 2.0f);
}

auto AsteroidSimulation::get_asteroids() const -> std::array<Asteroid, MAX_ASTEROID_COUNT> const &
{
    return asteroids;
}


void AsteroidSimulation::setConstants()
{
    mass = 0.02f;
    gasConstant = 1.f;
    viscosity = 1.04f;
    h = 1.f;
    //g = -9.81f;
    tension = 0.2f;
    restDensity = 10.f;
    viscosity = 0.3f;


    poly6 = 315.0f / (64.0f * PI * std::pow(h, 9));
    spikyGrad = -45.0f / (PI * std::pow(h, 6));
    spikyLap = 45.0f / (PI * std::pow(h, 6));
    h2 = h * h;
    selfDensity = mass * poly6 * std::pow(h, 6);
    massPoly6Product = mass * poly6;


    for(auto & asteroid : asteroids)
    {
        asteroid.mass = mass;
    }
}
