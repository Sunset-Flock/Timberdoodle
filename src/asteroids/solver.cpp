#include "solver.hpp"
#include "spatial_grid.hpp"

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
}

auto Material::evaluate(f64 const density, f64 const energy) const -> EvaluateRet
{
    const f64 mu = density / start_density - 1.0f;
    const f64 pressure = c * density * energy + A * mu;
    const f64 speed_of_sound = std::sqrt(A / start_density);

    return {
        .pressure = pressure,
        .speed_of_sound = speed_of_sound
    };
}

struct MaterialUpdateTask : Task
{
    AsteroidsWrapper * const asteroids;
    Material const * const material;

    MaterialUpdateTask(AsteroidsWrapper * const the_asteroids,
        Material const * const the_material,
        u32 const the_chunk_count,
        u32 const the_chunk_size) :
        asteroids(the_asteroids),
        material(the_material)
    {
        chunk_count = the_chunk_count;
        chunk_size = the_chunk_size;
    }

    void callback(u32 chunk_index, u32 thread_index) override
    {
        u32 const start = chunk_size * chunk_index;
        u32 const end = std::min(start + chunk_size, s_cast<u32>(asteroids->positions.size()));

        for(i32 asteroid_idx = start; asteroid_idx < end; ++asteroid_idx)
        {
            auto const & ret = material->evaluate(asteroids->densities.at(asteroid_idx), asteroids->energies.at(asteroid_idx));
            asteroids->pressures.at(asteroid_idx) = ret.pressure;
        }
    }
};

struct DerivativesTask : Task
{
    AsteroidsWrapper * const asteroids;
    SpatialGrid const * const grid;

    DerivativesTask(AsteroidsWrapper * const the_asteroids,
        SpatialGrid const * const the_grid,
        u32 const the_chunk_count,
        u32 const the_chunk_size) :
        asteroids(the_asteroids),
        grid(the_grid)
    {
        chunk_count = the_chunk_count;
        chunk_size = the_chunk_size;
    }

    void callback(u32 chunk_index, u32 thread_index) override
    {
        u32 const start = chunk_size * chunk_index;
        u32 const end = std::min(start + chunk_size, s_cast<u32>(asteroids->positions.size()));

        for(i32 asteroid_idx = start; asteroid_idx < end; ++asteroid_idx)
        {
            Kernel kernel;
            std::vector<u16> potential_neighbors = grid->get_neighbor_candidate_indices(
                asteroids->positions.at(asteroid_idx),
                0.5 * (asteroids->max_smoothing_radius + asteroids->smoothing_radii.at(asteroid_idx))
            );

            std::vector<u16> actual_neighbors = {};

            for(i32 potential_neighbors_vector_idx = 0; potential_neighbors_vector_idx < potential_neighbors.size(); ++potential_neighbors_vector_idx)
            {
                u16 const potential_neighbor_index = potential_neighbors.at(potential_neighbors_vector_idx);

                kernel.smoothing_radius = 0.5f * (asteroids->smoothing_radii.at(asteroid_idx) + asteroids->smoothing_radii.at(potential_neighbor_index));
                f64 const asteroid_distance = glm::length(asteroids->positions.at(asteroid_idx) - asteroids->positions.at(potential_neighbor_index));

                bool const asteroids_same = potential_neighbor_index == asteroid_idx;
                bool const asteroids_too_far = asteroid_distance > (kernel.smoothing_radius * 2.0f);
                if(!(asteroids_same || asteroids_too_far))
                {
                    actual_neighbors.push_back(potential_neighbor_index);
                }
            }


            for(i32 neighbor_asteroid_vector_idx = 0; neighbor_asteroid_vector_idx < actual_neighbors.size(); ++neighbor_asteroid_vector_idx)
            {
                u16 const neighbor_asteroid_idx = actual_neighbors.at(neighbor_asteroid_vector_idx);

                kernel.smoothing_radius = 0.5f * (asteroids->smoothing_radii.at(asteroid_idx) + asteroids->smoothing_radii.at(neighbor_asteroid_idx));

                f64 const asteroid_distance = glm::length(asteroids->positions.at(asteroid_idx) - asteroids->positions.at(neighbor_asteroid_idx));

                f64vec3 const asteroid_to_neighbor = asteroids->positions.at(asteroid_idx) - asteroids->positions.at(neighbor_asteroid_idx);
                f64vec3 const to_neighbor_gradient = asteroid_to_neighbor * std::pow(1.0f / kernel.smoothing_radius, 5.0f) * kernel.grad_value(asteroid_distance);

                // Velocity derivative
                f64vec3 const force = (
                    (asteroids->pressures.at(asteroid_idx) / std::pow(asteroids->densities.at(asteroid_idx), 2.0)) + 
                    (asteroids->pressures.at(neighbor_asteroid_idx) / std::pow(asteroids->densities.at(neighbor_asteroid_idx), 2.0)))
                    * to_neighbor_gradient * -1.0;
            
                asteroids->velocity_derivatives.at(asteroid_idx) += asteroids->masses.at(neighbor_asteroid_idx) * force;

                // Velocity divergence
                f64 dv = glm::dot(asteroids->velocities.at(neighbor_asteroid_idx) - asteroids->velocities.at(asteroid_idx), to_neighbor_gradient);
                asteroids->velocity_divergences.at(asteroid_idx) += asteroids->masses.at(neighbor_asteroid_idx) / asteroids->densities.at(asteroid_idx) * dv;
            }
        }
    }
};

struct EquationsAndUpdatesTask : Task
{
    AsteroidsWrapper * const asteroids;
    SpatialGrid const * const grid;
    f64 const dt;

    EquationsAndUpdatesTask(AsteroidsWrapper * const the_asteroids,
        SpatialGrid const * const the_grid,
        u32 const the_chunk_count,
        u32 const the_chunk_size,
        f64 const the_dt) :
        asteroids(the_asteroids),
        grid(the_grid),
        dt(the_dt)
    {
        chunk_count = the_chunk_count;
        chunk_size = the_chunk_size;
    }

    void callback(u32 chunk_index, u32 thread_index) override
    {
        u32 const start = chunk_size * chunk_index;
        u32 const end = std::min(start + chunk_size, s_cast<u32>(asteroids->positions.size()));
        // Calculate continuity equation
        for(i32 asteroid_idx = start; asteroid_idx < end; ++asteroid_idx)
        {
            asteroids->density_derivatives.at(asteroid_idx) += -asteroids->densities.at(asteroid_idx) * asteroids->velocity_divergences.at(asteroid_idx);
        }

        // Calculate pressure force
        for(i32 asteroid_idx = start; asteroid_idx < end; ++asteroid_idx)
        {
            asteroids->energy_derivatives.at(asteroid_idx) -= 
                asteroids->pressures.at(asteroid_idx) / asteroids->densities.at(asteroid_idx) * asteroids->velocity_divergences.at(asteroid_idx);
        }

        // Second order velocity step
        for(i32 asteroid_idx = start; asteroid_idx < end; ++asteroid_idx)
        {
            asteroids->velocities.at(asteroid_idx) += asteroids->velocity_derivatives.at(asteroid_idx) * dt;
        }
        // Second order position step
        for(i32 asteroid_idx = start; asteroid_idx < end; ++asteroid_idx)
        {
            asteroids->positions.at(asteroid_idx) += asteroids->velocities.at(asteroid_idx) * dt;
        }

        // First order step
        for(i32 asteroid_idx = start; asteroid_idx < end; ++asteroid_idx)
        {
            asteroids->densities.at(asteroid_idx) += asteroids->density_derivatives.at(asteroid_idx) * dt;
        }
        for(i32 asteroid_idx = start; asteroid_idx < end; ++asteroid_idx)
        {
            asteroids->energies.at(asteroid_idx) += asteroids->energy_derivatives.at(asteroid_idx) * dt;
        }
    }
};

void Solver::integrate(AsteroidsWrapper & asteroids, f64 const dt, ThreadPool & threadpool)
{
    const u32 asteroids_size = asteroids.positions.size();

    std::fill(asteroids.energy_derivatives.begin(), asteroids.energy_derivatives.end(), 0.0);
    std::fill(asteroids.density_derivatives.begin(), asteroids.density_derivatives.end(), 0.0);
    std::fill(asteroids.velocity_derivatives.begin(), asteroids.velocity_derivatives.end(), f32vec3(0.0));
    std::fill(asteroids.velocity_divergences.begin(), asteroids.velocity_divergences.end(), 0.0);
    // Zero out derivatives from last frame.


    SpatialGrid grid = SpatialGrid(asteroids.positions, 3000.0f);

    {
        const u32 chunk_size = 256u;
        const u32 chunk_count = (asteroids.positions.size() + chunk_size - 1) / chunk_size;
        std::shared_ptr<MaterialUpdateTask> material_update_task = std::make_shared<MaterialUpdateTask>(&asteroids, &material, chunk_count, chunk_size);
        threadpool.blocking_dispatch(material_update_task);
    }

    {
        const u32 chunk_size = 32u;
        const u32 chunk_count = (asteroids.positions.size() + chunk_size - 1) / chunk_size;
        std::shared_ptr<DerivativesTask> derivatives_task = std::make_shared<DerivativesTask>(&asteroids, &grid, chunk_count, chunk_size);
        threadpool.blocking_dispatch(derivatives_task);
    }

    {
        const u32 chunk_size = 256u;
        const u32 chunk_count = (asteroids.positions.size() + chunk_size - 1) / chunk_size;
        std::shared_ptr<EquationsAndUpdatesTask> equations_and_updates_task = std::make_shared<EquationsAndUpdatesTask>(&asteroids, &grid, chunk_count, chunk_size, dt);
        threadpool.blocking_dispatch(equations_and_updates_task);
    }
}