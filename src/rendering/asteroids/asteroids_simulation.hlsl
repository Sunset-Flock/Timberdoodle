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
    let asteroid_index = gid.x;

    float3 velocity_derivative = 0.0f;
    float velocity_divergence = 0.0f;

    const float my_smoothing_radius = push.params.smoothing_radius[asteroid_index];
    const float3 my_position = push.params.position[asteroid_index];
    const float my_pressure = push.params.pressure[asteroid_index];
    const float3 my_velocity = push.params.velocity[asteroid_index];
    const float my_density = push.params.density[asteroid_index];

    const float my_force = push.params.pressure[asteroid_index] / pow(my_density, 2.0f);

    const int chunk_size = (push.asteroid_count + DERIVATIVES_CALCULATION_WORKGROUP_X - 1) / DERIVATIVES_CALCULATION_WORKGROUP_X;
    const int start = gtid.x * chunk_size;
    const int end = min((gtid.x + 1) * chunk_size, push.asteroid_count);

    for(int neighbor_index = start; neighbor_index < end; ++neighbor_index)
    {
        if(neighbor_index == asteroid_index) { continue; }
        const float3 neighbor_position = push.params.position[neighbor_index];
        const float3 position_difference = my_position - neighbor_position;

        const float average_smoothing_radius = (my_smoothing_radius + push.params.smoothing_radius[neighbor_index]) * 0.5f;

        const float distance = length(position_difference);
        if(distance > average_smoothing_radius * 2.0f) { continue; }

        const float gradient_value = grad_value(distance, average_smoothing_radius);
        const float3 to_neighbor_gradient = position_difference * pow(1.0f / average_smoothing_radius, 5.0f) * gradient_value;

        const float neighbor_force = push.params.pressure[neighbor_index] / pow(push.params.density[neighbor_index], 2.0f);

        const float3 force = (my_force + neighbor_force) * to_neighbor_gradient * -1.0f;
        velocity_derivative += push.params.mass[neighbor_index] * force;


        const float dv = dot(push.params.velocity[neighbor_index] - my_velocity, to_neighbor_gradient);
        velocity_divergence += push.params.mass[neighbor_index] / my_density * dv;
    }

    GroupMemoryBarrierWithWaveSync();
    velocity_derivative = WaveActiveSum(velocity_derivative);
    velocity_divergence = WaveActiveSum(velocity_divergence);

    if(WaveIsFirstLane())
    {
        accum[(gtid.x + 1) / 32] = float4(velocity_derivative, velocity_divergence);
    }

    GroupMemoryBarrierWithGroupSync();
    if(gtid.x < 32)
    {
        float4 value = 0.0f;
        if(gtid.x < (DERIVATIVES_CALCULATION_WORKGROUP_X / 32))
        {
            value = accum[gtid.x];
        }
        float4 final = WaveActiveSum(value);
        if(gtid.x == 0)
        {
            push.params.velocity_derivative[asteroid_index] = final.xyz;
            push.params.velocity_divergence[asteroid_index] = final.w;
        }
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