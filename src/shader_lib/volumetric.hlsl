#pragma once

#include "daxa/daxa.inl"

#include "shader_shared/volumetric.inl"

// Froxel volume format:
// rgb9e5 instatter color froxel volume
// rgb9e4 accumulated inscatter color

struct VolumetricResult
{
    float3 transmittance;
    float3 inscattering;
};
func volumetric_extinction_inscatter(float3 origin, float3 ray, float t, RaytracingAccelerationStructure tlas, float3 sun_direction) -> VolumetricResult
{
    VolumetricResult ret = {};

    float3 fog_albedo = float3(0.5f, 0.8f, 1.0f) * 0.1f;
    float fog_density = 0.05;

    float3 fog_extinction_coef = (1.0f - fog_albedo) * fog_density;
    float3 fog_scattering_coef = (fog_albedo) * fog_density;
    float3 transmittance = exp(-(fog_extinction_coef * t));
    ret.transmittance = transmittance;

    //ret.transmittance = 1.0f.xxx;
    ret.inscattering = 0.0f.xxx;

    // Shitty Raymarch
    let march_step_size = 1.5;
    for (uint i = 0; i < 32; ++i)
    {
        let march_t = i * march_step_size + march_step_size * rand();
        if (march_t >= t) 
            break;

        float3 ws_sample_pos = origin + ray * march_t;

        float3 inscatter = 1.0f - exp(-(fog_scattering_coef * march_step_size));

        RayQuery<RAY_FLAG_CULL_NON_OPAQUE |
        RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;

        const float t_min = 0.01f;
        const float t_max = 1000.0f;

        RayDesc my_ray = {
            ws_sample_pos,
            t_min,
            sun_direction,
            t_max,
        };

        // Set up a trace.  No work is done yet.
        q.TraceRayInline(
            tlas,
            0, // OR'd with flags above
            0xFFFF,
            my_ray);

        // Proceed() below is where behind-the-scenes traversal happens,
        // including the heaviest of any driver inlined code.
        // In this simplest of scenarios, Proceed() only needs
        // to be called once rather than a loop:
        // Based on the template specialization above,
        // traversal completion is guaranteed.
        q.Proceed();


        bool shadowed = false;
        // Examine and act on the result of the traversal.
        // Was a hit committed?
        if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            shadowed = true;
        }


        if (!shadowed)
        {
            ret.inscattering += inscatter * float3(1,1,1) * march_step_size;
        }
    }

    return ret;
}