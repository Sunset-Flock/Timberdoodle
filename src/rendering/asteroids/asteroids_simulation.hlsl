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


struct WaveScalarParameters
{
    float my_pressure;
    float my_density;
    float my_smoothing_radius;
    float my_force;
    float3 my_position;
    float3 my_velocity;

    float max_search_radius;
    int search_radius_in_cells;
    int3 start_cell_coordinates;

    int cell_start_indices[27];
    int keys[27];
};
groupshared WaveScalarParameters gs_derivatives[DERIVATIVES_CALCULATION_WORKGROUP_X / 32];

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
    let wave_index = gtid.x >> 5;
    let waves_per_group = DERIVATIVES_CALCULATION_WORKGROUP_X / 32;
    let asteroid_index = (gid.x * waves_per_group) + wave_index;

    float3 velocity_derivative = 0.0f;
    float velocity_divergence = 0.0f;

    if(WaveIsFirstLane())
    {
        gs_derivatives[wave_index].my_smoothing_radius = push.smoothing_radius[asteroid_index];
        gs_derivatives[wave_index].my_pressure = push.pressure[asteroid_index];
        gs_derivatives[wave_index].my_density = push.density[asteroid_index];
        gs_derivatives[wave_index].my_force = push.pressure[asteroid_index] / pow(gs_derivatives[wave_index].my_density, 2.0f);

        gs_derivatives[wave_index].my_position = push.position[asteroid_index];
        gs_derivatives[wave_index].my_velocity = push.velocity[asteroid_index];

        gs_derivatives[wave_index].max_search_radius = (gs_derivatives[wave_index].my_smoothing_radius + push.max_smoothing_radius) * 0.5f;
        gs_derivatives[wave_index].search_radius_in_cells = int(ceil(gs_derivatives[wave_index].max_search_radius / push.cell_size));

        gs_derivatives[wave_index].start_cell_coordinates = int3(gs_derivatives[wave_index].my_position / float3(push.cell_size));
    }
    GroupMemoryBarrierWithWaveSync();

    let search_radius = gs_derivatives[wave_index].search_radius_in_cells;
    if(any(greaterThan(abs(search_radius), 1)))
    {
        printf("Search radius larger than I assumed wtf\n");
    }

    if(WaveGetLaneIndex() < 27)
    {
        let wave_lane_index = WaveGetLaneIndex();
        const int z_offset = wave_lane_index % 3;
        const int y_offset = (wave_lane_index / 3) % 3;
        const int x_offset = (wave_lane_index / 9);

        const int3 cell_offset = int3(x_offset - 1, y_offset - 1 , z_offset - 1);
        const int3 offset_cell_coordinates = gs_derivatives[wave_index].start_cell_coordinates + cell_offset;

        // Calculate the cells key.
        const uint hash = 
            offset_cell_coordinates.x * HASH_KEY_1 +
            offset_cell_coordinates.y * HASH_KEY_2 +
            offset_cell_coordinates.z * HASH_KEY_3;

        const uint key = hash % push.asteroid_count;
        const uint cell_start_index = push.cell_start_indices[key];

        gs_derivatives[wave_index].cell_start_indices[WaveGetLaneIndex()] = cell_start_index;
        gs_derivatives[wave_index].keys[WaveGetLaneIndex()] = key;
    }
    GroupMemoryBarrierWithGroupSync();

    for(int cells_index = 0; cells_index < 27; ++cells_index)
    {
        // And while the key matches the cell we are currently in add the point indices to the list of potential neighbors.
        for (int index = gs_derivatives[wave_index].cell_start_indices[cells_index]; index < push.asteroid_count; index += WaveGetLaneCount())
        {
            int real_index = index + WaveGetLaneIndex();
            bool is_key = false;
            if(real_index < push.asteroid_count)
            {
                is_key = push.spatial_lookup[real_index].x == gs_derivatives[wave_index].keys[cells_index];
            }

            if(is_key)
            {
                bool real_neighbor = true;
                const int neighbor_index = push.spatial_lookup[real_index].y;
                real_neighbor = (neighbor_index != asteroid_index);

                float average_smoothing_radius = 0.0f;
                float distance = 0.0f;
                float3 position_difference = 0.0f;
                if(real_neighbor)
                {
                    const float3 neighbor_position = push.position[neighbor_index];
                    position_difference = gs_derivatives[wave_index].my_position - neighbor_position;

                    average_smoothing_radius = (gs_derivatives[wave_index].my_smoothing_radius + push.smoothing_radius[neighbor_index]) * 0.5f;

                    distance = length(position_difference);
                    real_neighbor = distance <= (average_smoothing_radius * 2.0f);
                }

                if(real_neighbor)
                {
                    const float gradient_value = grad_value(distance, average_smoothing_radius);
                    const float3 to_neighbor_gradient = position_difference * pow(1.0f / average_smoothing_radius, 5.0f) * gradient_value;

                    const float neighbor_force = push.pressure[neighbor_index] / pow(push.density[neighbor_index], 2.0f);

                    const float3 force = (gs_derivatives[wave_index].my_force + neighbor_force) * to_neighbor_gradient * -1.0f;
                    velocity_derivative += push.mass[neighbor_index] * force;

                    const float dv = dot(push.velocity[neighbor_index] - gs_derivatives[wave_index].my_velocity, to_neighbor_gradient);
                    velocity_divergence += push.mass[neighbor_index] / gs_derivatives[wave_index].my_density * dv;
                }
            }
            if(WaveActiveAnyTrue(!is_key))
            {
                break;
            }
        }
    }

    let exclusive_velocity_derivative = WavePrefixSum(velocity_derivative);
    let exclusive_velocity_divergence = WavePrefixSum(velocity_divergence);
    if(WaveGetLaneIndex() == 31)
    {
        push.velocity_derivative[asteroid_index] = exclusive_velocity_derivative + velocity_derivative;
        push.velocity_divergence[asteroid_index] = exclusive_velocity_divergence + velocity_divergence;
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