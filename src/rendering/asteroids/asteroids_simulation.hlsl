#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "draw_asteroids.inl"
#include "../../shader_shared/asteroids.inl"
#include "../../shader_lib/misc.hlsl"

[[vk::push_constant]] MaterialUpdatePush material_push;
[numthreads(MATERIAL_UPDATE_WORKGROUP_X, 1, 1)]
[shader("compute")]
void update_material(
    uint3 svdtid : SV_DispatchThreadID
)
{
    let push = material_push;

    if(svdtid.x < push.asteroid_count)
    {
        const float density = push.asteroid_density[svdtid.x];
        const float energy = push.asteroid_energy[svdtid.x];

        const float mu = density / push.start_density - 1.0f;
        const float pressure = push.c * density * energy + push.A * mu;
        push.asteroid_pressure[svdtid.x] = pressure;
    }
}

static const float NORMALIZATION = 0.31830988618379067153776752674f; // 1.0f / PI
float grad_value(const float distance, const float smoothing_radius)
{
    const float q = distance / smoothing_radius;

    if (q == 0.0f) {
        return -3.0f * NORMALIZATION;
    }
    else if (q < 1.0f) {
        return (1.0 / q) * NORMALIZATION * (-0.75 * pow(2.0 - q, 2.0) + 3.0 * pow(1.0 - q, 2.0));
    }
    else if (q < 2.0f) {
        return (1.0 / q) * NORMALIZATION * (-0.75 * pow(2.0 - q, 2.0));
    }
    return 0.0f;
}

groupshared float4 accum[DERIVATIVES_CALCULATION_WORKGROUP_X / 32];

static const uint HASH_KEY_1 = 15823;
static const uint HASH_KEY_2 = 9737333;
static const uint HASH_KEY_3 = 440817757;

[[vk::push_constant]] DerivativesCalculationPush derivatives_push;
[numthreads(DERIVATIVES_CALCULATION_WORKGROUP_X, 1, 1)]
[shader("compute")]
void calculate_derivatives(
    uint3 svdtid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 gtid : SV_GroupThreadID
)
{
    let push = derivatives_push;
    let asteroid_index = svdtid.x;

    float3 velocity_derivative = 0.0f;
    float velocity_divergence = 0.0f;

    if(asteroid_index < push.asteroid_count)
    {
        const float my_smoothing_radius = push.smoothing_radius[asteroid_index];
        const float3 my_position = push.position[asteroid_index];
        const float my_pressure = push.pressure[asteroid_index];
        const float3 my_velocity = push.velocity[asteroid_index];
        const float my_density = push.density[asteroid_index];

        const float my_force = push.pressure[asteroid_index] / pow(my_density, 2.0f);

        const float max_search_radius = (my_smoothing_radius + push.max_smoothing_radius) * 0.5f;
        const int search_radius_in_cells = int(ceil(max_search_radius / push.cell_size));

        const int3 start_cell_coordinates = int3(my_position / float3(push.cell_size));

        const uint target_asteroid_index = 6696;
        for(int cell_x_offset = -search_radius_in_cells; cell_x_offset <= search_radius_in_cells; ++cell_x_offset)
        {
            for(int cell_y_offset = -search_radius_in_cells; cell_y_offset <= search_radius_in_cells; ++cell_y_offset)
            {
                for(int cell_z_offset = -search_radius_in_cells; cell_z_offset <= search_radius_in_cells; ++cell_z_offset)
                {
                    const int3 cell_offset = int3(cell_x_offset, cell_y_offset, cell_z_offset);
                    const int3 offset_cell_coordinates = start_cell_coordinates + cell_offset;

                    // Calculate the cells key.
                    const uint hash = 
                        offset_cell_coordinates.x * HASH_KEY_1 +
                        offset_cell_coordinates.y * HASH_KEY_2 +
                        offset_cell_coordinates.z * HASH_KEY_3;

                    const uint key = hash % push.asteroid_count;

                    // Lookup its starting index.
                    const uint cell_start_index = push.cell_start_indices[key];

                    // And while the key matches the cell we are currently in add the point indices to the list of potential neighbors.
                    for (int index = cell_start_index; index < push.asteroid_count; ++index)
                    {
                        if(push.spatial_lookup[index].x != key) { break; }

                        const int neighbor_index = push.spatial_lookup[index].y;
                        if(neighbor_index == asteroid_index) { continue; }

                        const float3 neighbor_position = push.position[neighbor_index];
                        const float3 position_difference = my_position - neighbor_position;

                        const float average_smoothing_radius = (my_smoothing_radius + push.smoothing_radius[neighbor_index]) * 0.5f;

                        const float distance = length(position_difference);
                        if(distance > average_smoothing_radius * 2.0f) { continue; }

                        const float gradient_value = grad_value(distance, average_smoothing_radius);
                        const float3 to_neighbor_gradient = position_difference * pow(1.0f / average_smoothing_radius, 5.0f) * gradient_value;
                        if(asteroid_index == target_asteroid_index)
                        {
                            // push.velocity_derivative[neighbor_index] = 10.0f;
                            // printf("found neighbor %d with key %d my key %d offset: %d %d %d gradient value %f\n",
                            //     neighbor_index, push.spatial_lookup[index].x, key, cell_offset.x, cell_offset.y, cell_offset.z, gradient_value);
                        }

                        const float neighbor_force = push.pressure[neighbor_index] / pow(push.density[neighbor_index], 2.0f);

                        const float3 force = (my_force + neighbor_force) * to_neighbor_gradient * -1.0f;
                        velocity_derivative += push.mass[neighbor_index] * force;

                        const float dv = dot(push.velocity[neighbor_index] - my_velocity, to_neighbor_gradient);
                        velocity_divergence += push.mass[neighbor_index] / my_density * dv;
                    }
                }
            }
        }

        if(asteroid_index == target_asteroid_index)
        {
            // push.velocity_derivative[asteroid_index] = 100.0f;
            // printf("===============================\n");
        }
        // push.velocity_derivative[asteroid_index] = 0;
        push.velocity_derivative[asteroid_index] = velocity_derivative;
        push.velocity_divergence[asteroid_index] = velocity_divergence;
    }
}

[[vk::push_constant]] EquationUpdatePush equations_push;

[numthreads(EQUATION_UPDATE_WORKGROUP_X, 1, 1)]
[shader("compute")]
void equation_update(
    uint3 svdtid : SV_DispatchThreadID
)
{
    let push = equations_push;
    let asteroid_index = svdtid.x;

    if(svdtid.x < push.asteroid_count)
    {
        const float asteroid_density = push.params.density[asteroid_index];
        // Calculate continuity equation
        const float density_derivative = -asteroid_density * push.params.velocity_divergence[asteroid_index];

        // Calculate pressure force
        const float energy_derivative = -push.params.pressure[asteroid_index] / asteroid_density * push.params.velocity_divergence[asteroid_index];

        // Second order velocity step
        push.params.velocity[asteroid_index] += push.params.velocity_derivative[asteroid_index] * push.dt;
        // Second order position step
        push.params.position[asteroid_index] += push.params.velocity[asteroid_index] * push.dt;

        // First order step
        push.params.density[asteroid_index] += density_derivative * push.dt;
        push.params.energy[asteroid_index] += energy_derivative * push.dt;
    }
}