#pragma once

#include "daxa/daxa.inl"
#include "../shader_shared/pgi.inl"
#include "../shader_lib/misc.hlsl"
#include "../shader_shared/globals.inl"
#include "../shader_lib/raytracing.hlsl"
#include "../shader_lib/SH.hlsl"

// ===== PGI Probe Grid =====
// 
// Example Probe Grid 4x4 probes in 2d
//
// O-----------O           O           O <- Probe
// |   probe   |
// |   grid    |
// |   cell    |
// O-----------O           O           O
//
//                   x <- main camera position
//
// O           O           O           O
//
//
//                                 
// O           O           O           O
// 
// - The probe count is always even
// - The probes are centered around the player (marked as x)
// - The probes form a grid. Each cell in the grid has 8 probes in its corners
// - The grid has probe_count-1 cells in each dimension
// - The probes positions are locked to multiples of the cell size in world space
// 
// ===== PGI Probe Grid =====

// ===== PGI Probe Texture Layouts =====
//
// Example: probe texel resolution = 4x4
//   0 1 2 3 4 5 6 7 ...
// 0 x x x x x x x x 
// 1 x     x x     x <= each probe gets a 4x4 xy section in the probes texture.
// 2 x     x x     x 
// 3 x x x x x x x x 
// 4 x x x x x x x x 
// 5 x     x x     x 
// 6 x     x x     x 
// 7 x x x x x x x x 
// 
// - Probe with index (x,y,z) gets a section in the texture in (xy*4 <-> xy*4 + 4, z)
//
// ===== PGI Probe Texture Layouts =====

float3 pgi_probe_index_to_worldspace(PGISettings settings, float3 probes_anchor, uint3 probe_index)
{
    float3 pgi_grid_cell_size = settings.probe_range / settings.probe_count;   // TODO: precalculate
    float3 center_grid_cell_min_probe_pos = float3(
        f32_round_down_to_multiple(probes_anchor.x, pgi_grid_cell_size.x),
        f32_round_down_to_multiple(probes_anchor.y, pgi_grid_cell_size.y),
        f32_round_down_to_multiple(probes_anchor.z, pgi_grid_cell_size.z),
    );
    return (int3(probe_index) - settings.probe_count/2) * pgi_grid_cell_size + center_grid_cell_min_probe_pos;


    float3 min_probe_pos = center_grid_cell_min_probe_pos - pgi_grid_cell_size * 0.5f * float3(settings.probe_count);
    float3 probe_pos = min_probe_pos + pgi_grid_cell_size * float3(probe_index);
    return probe_pos;
}

uint3 pgi_probe_texture_base_offset(PGISettings settings, uint3 probe_index)
{
    let probe_texture_base_xy = probe_index.xy * settings.probe_surface_resolution;
    let probe_texture_z = probe_index.z;

    var probe_texture_index = uint3(probe_texture_base_xy, probe_texture_z);
    return probe_texture_index;
}

uint3 pgi_probe_texture_base_offset_prev_frame(PGISettings settings, uint3 probe_index)
{
    let probe_texture_base_xy = probe_index.xy * settings.probe_surface_resolution;
    let probe_texture_z = probe_index.z;

    var probe_texture_index = uint3(probe_texture_base_xy, probe_texture_z);
    return probe_texture_index;
}

float2 pgi_probe_normal_to_probe_uv(float3 normal)
{
    return map_octahedral(normal);
}

float3 pgi_probe_uv_to_probe_normal(float2 uv)
{
    return unmap_octahedral(uv);
}



// Grid space starts at 0,0,0 at the min probe and ends at settings.probe_count
// floor(grid_space_pos) is the base probe of the position
// frac(grid_space_pos) are the interpolators for the probes around that position
func pgi_world_space_to_grid_coordinate(
    RenderGlobalData* globals,
    PGISettings settings,
    float3 position
) -> float3
{
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : globals.camera.position;
    float3 min_probe_world_position = pgi_probe_index_to_worldspace(settings, probe_anchor, uint3(0,0,0)); 
    float3 min_probe_relative_position = position - min_probe_world_position;
    float3 grid_space_coordinate = min_probe_relative_position * rcp(settings.probe_range) * settings.probe_count;
    return grid_space_coordinate;
}

func pgi_sample_nearest(
    RenderGlobalData* globals,
    PGISettings settings,
    float3 position,
    float3 direction,
    RWTexture2DArray<float4> probes
) -> float4
{
    float3 grid_coord = pgi_world_space_to_grid_coordinate(globals, settings, position);
    int3 base_probe = int3(grid_coord);
    float3 interpolants = frac(grid_coord);
    float3 probe_normal = direction;

    float4 accum = float4(0,0,0,0);
    for (int z = 0; z < 2; ++z)
    {
        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < 2; ++x)
            {
                int3 probe_index = base_probe + int3(x,y,z);
                float probe_weight = 
                    (x == 0 ? 1.0f - interpolants.x : interpolants.x) *
                    (y == 0 ? 1.0f - interpolants.y : interpolants.y) *
                    (z == 0 ? 1.0f - interpolants.z : interpolants.z);

                if (all(probe_index >= int3(0,0,0)) && all(probe_index < settings.probe_count))
                {
                    float2 octa_index = floor(float(settings.probe_surface_resolution) * map_octahedral(probe_normal));
                    uint3 probe_texture_base_index = pgi_probe_texture_base_offset(globals.pgi_settings, probe_index);
                    uint3 probe_texture_index = probe_texture_base_index + uint3(octa_index.x, octa_index.y, 0);
                    float4 probe_fetch = probes[probe_texture_index];
                    accum += probe_weight * probe_fetch;
                }
            }
        }
    }
    return accum * rcp(8.0f);
}

func pgi_sample_irradiance(
    RenderGlobalData* globals,
    PGISettings settings,
    float3 position,
    float3 direction,
    RaytracingAccelerationStructure tlas,
    Texture2DArray<float4> probes
) -> float3
{
    position += direction * 0.01f;
    float3 grid_coord = pgi_world_space_to_grid_coordinate(globals, settings, position);
    int3 base_probe = floor(grid_coord);
    float3 interpolants = frac(grid_coord);
    float3 probe_normal = direction;

    float3 cell_size = float3(settings.probe_range) / float3(settings.probe_count);
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : globals.camera.position;

    float3 accum = float3(0,0,0);
    float weight_accum = 0;
    for (int z = 0; z < 2; ++z)
    {
        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < 2; ++x)
            {
                int3 probe_index = base_probe + int3(x,y,z);
                float probe_weight = 
                    (x == 0 ? 1.0f - interpolants.x : interpolants.x) *
                    (y == 0 ? 1.0f - interpolants.y : interpolants.y) *
                    (z == 0 ? 1.0f - interpolants.z : interpolants.z);

                if (all(probe_index >= int3(0,0,0)) && all(probe_index < settings.probe_count))
                {
                    float3 probe_position = pgi_probe_index_to_worldspace(settings, probe_anchor, probe_index);

                    float distance = length(probe_position - position) * 1.01f;
                    float3 direction = normalize(probe_position - position);

                    float t = rt_free_path(tlas, position, direction, distance);
                    bool visible = t >= distance;
                    if (!visible) 
                    {
                        continue;
                    }

                    // Based on the texture index we linearly subsample the probes image with a 2x2 kernel.
                    float2 probe_octa_uv = map_octahedral(probe_normal);
                    float2 probe_local_texel = probe_octa_uv * float(settings.probe_surface_resolution);
                    // FLOORING IS REQUIRED HERE AS FLOAT TO INT CONVERSION ALWAYS ROUNDS TO 0, NOT TO THE LOWER NUMBER!
                    int2 probe_local_base_texel = int2(floor(probe_local_texel - 0.5f));
                    float2 xy_base_weights = frac(probe_local_texel - 0.5f + float(settings.probe_surface_resolution));
                    int3 base_offset = pgi_probe_texture_base_offset(settings, probe_index);

                    float3 linearly_filtered_samples = float3(0,0,0);
                    for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 2; ++x)
                    {
                        int2 xy_sample_offset = int2(x,y);
                        int2 probe_local_sample_texel = probe_local_base_texel + xy_sample_offset;

                        // Octahedral texel clamping is very strange..
                        if (probe_local_sample_texel.y >= settings.probe_surface_resolution || probe_local_sample_texel.y == -1)
                        {
                            probe_local_sample_texel.y = clamp(probe_local_sample_texel.y, 0, settings.probe_surface_resolution-1);
                            // Mirror x sample when y is out of bounds
                            probe_local_sample_texel.x = settings.probe_surface_resolution - 1 - probe_local_sample_texel.x;
                        }
                        if (probe_local_sample_texel.x >= settings.probe_surface_resolution || probe_local_sample_texel.x == -1)
                        {
                            probe_local_sample_texel.x = clamp(probe_local_sample_texel.x, 0, settings.probe_surface_resolution-1);
                            // Mirror y sample when x is out of bounds
                            probe_local_sample_texel.y = settings.probe_surface_resolution - 1 - probe_local_sample_texel.y;
                        }

                        int3 sample_texel = base_offset + int3(probe_local_sample_texel, 0);
                        float3 sample = probes[sample_texel].rgb;
                        float weight = 
                            (x != 0 ? xy_base_weights.x : 1.0f - xy_base_weights.x) *
                            (y != 0 ? xy_base_weights.y : 1.0f - xy_base_weights.y);
                        linearly_filtered_samples += weight * sample;
                    }
                    //if (any(int2(probe_local_texel) == int2(5,5)))
                    //{
                    //    linearly_filtered_samples = float3(1,0,0);
                    //}

                    //linearly_filtered_samples = float3(probe_octa_uv,0);//float3(float2(probe_local_base_texel + 1+1) * rcp(8), 0);

                    //float2 octa_index = floor(float(settings.probe_surface_resolution) * map_octahedral(probe_normal));
                    //uint3 probe_texture_base_index = pgi_probe_texture_base_offset(globals.pgi_settings, probe_index);
                    //uint3 probe_texture_index = probe_texture_base_index + uint3(octa_index.x, octa_index.y, 0);
                    //float4 probe_fetch = probes[probe_texture_index];
                    accum += probe_weight * linearly_filtered_samples.rgb;
                    weight_accum += probe_weight;
                }
            }
        }
    }
    if (weight_accum == 0)
    {
        return float3(0,0,0);
    }
    else
    {
        return accum * rcp(weight_accum);
    }
}