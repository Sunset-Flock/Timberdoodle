#pragma once

#include "../timberdoodle.hpp"

using namespace tido::types;

#define PI 3.1415926535897932384f
#define INITAL_DENSITY 2700.0f

#define PLANET_RADIUS 50'000.0f
#define ASTEROID_RADIUS 20'000.0f

struct SimulationBodyInfo
{
    f32vec3 position = {};
    f32vec3 velocity_vector = {};
    f32 velocity_magnitude = {};
    f32 radius = 1000;
    i32 particle_count = 1;
    f32 particle_size = 1.0f;

    std::string name;
};

struct AsteroidsWrapper
{
    void resize(i32 size)
    {
        positions.resize(size);
        velocities.resize(size);
        velocity_derivatives.resize(size);
        velocity_divergences.resize(size);

        smoothing_radii.resize(size);
        masses.resize(size);

        densities.resize(size);
        density_derivatives.resize(size);

        energies.resize(size);
        energy_derivatives.resize(size);

        pressures.resize(size);

        particle_scales.resize(size);
    }

    template<typename Type>
    static void copy_vector(std::vector<Type> & dst, std::vector<Type> const & src)
    {
        DBG_ASSERT_TRUE_M(dst.size() == src.size(), "Sizes must match");
        std::memcpy(dst.data(), src.data(), sizeof(Type) * dst.size());
    }

    static void copy(AsteroidsWrapper & dst, AsteroidsWrapper const & src)
    {
        if(src.positions.size() != dst.positions.size())
        {
            dst.resize(src.positions.size());
        }

        copy_vector(dst.positions, src.positions);
        copy_vector(dst.velocities, src.velocities);
        copy_vector(dst.velocity_derivatives, src.velocity_derivatives);
        copy_vector(dst.velocity_divergences, src.velocity_divergences);

        copy_vector(dst.smoothing_radii, src.smoothing_radii);
        copy_vector(dst.masses, src.masses);

        copy_vector(dst.densities, src.densities);
        copy_vector(dst.density_derivatives, src.density_derivatives);

        copy_vector(dst.energies, src.energies);
        copy_vector(dst.energy_derivatives, src.energy_derivatives);

        copy_vector(dst.pressures, src.pressures);

        copy_vector(dst.particle_scales, src.particle_scales);


        dst.max_smoothing_radius = src.max_smoothing_radius;
        dst.simulation_started = src.simulation_started;
        dst.simulation_bodies = src.simulation_bodies;
    }

    std::vector<f32vec3> positions = {};
    std::vector<f32vec3> velocities = {};
    std::vector<f32vec3> velocity_derivatives = {};
    std::vector<f32> velocity_divergences = {};
    
    std::vector<f32> smoothing_radii = {};
    std::vector<f32> masses = {};

    std::vector<f32> densities = {};
    std::vector<f32> density_derivatives = {};

    std::vector<f32> energies = {};
    std::vector<f32> energy_derivatives = {};

    std::vector<f32> pressures = {};

    std::vector<f32> particle_scales = {};

    std::vector<SimulationBodyInfo> simulation_bodies = {};

    f32 max_smoothing_radius = {};
    bool simulation_started = {};
};