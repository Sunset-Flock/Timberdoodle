#pragma once

#include "daxa/daxa.inl"
#include "../shader_shared/ddgi.inl"
#include "../shader_lib/misc.hlsl"
#include "../shader_shared/globals.inl"
#include "../shader_lib/raytracing.hlsl"

// ===== DDGI Probe Grid =====
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
// ===== DDGI Probe Grid =====

float3 ddgi_probe_index_to_worldspace(DDGISettings settings, float3 probes_anchor, uint3 probe_index)
{
    float3 ddgi_grid_cell_size = settings.probe_range / settings.probe_count;   // TODO: precalculate
    float3 center_grid_cell_min_probe_pos = float3(
        f32_round_down_to_multiple(probes_anchor.x, ddgi_grid_cell_size.x),
        f32_round_down_to_multiple(probes_anchor.y, ddgi_grid_cell_size.y),
        f32_round_down_to_multiple(probes_anchor.z, ddgi_grid_cell_size.z),
    );
    return (int3(probe_index) - settings.probe_count/2) * ddgi_grid_cell_size + center_grid_cell_min_probe_pos;


    float3 min_probe_pos = center_grid_cell_min_probe_pos - ddgi_grid_cell_size * 0.5f * float3(settings.probe_count);
    float3 probe_pos = min_probe_pos + ddgi_grid_cell_size * float3(probe_index);
    return probe_pos;
}

uint3 ddgi_probe_base_texture_index(DDGISettings settings, uint3 probe_index, uint frame_index)
{
    let probe_texture_layer_offset = 0;//settings.probe_count.z * (frame_index & 0x1);
    let probe_texture_base_xy = probe_index.xy * settings.probe_surface_resolution;
    let probe_texture_z = probe_texture_layer_offset + probe_index.z;

    var probe_texture_index = uint3(probe_texture_base_xy, probe_texture_z);
    return probe_texture_index;
}

uint3 ddgi_probe_base_texture_index_prev_frame(DDGISettings settings, uint3 probe_index, uint frame_index)
{
    let probe_texture_layer_offset = 0;//settings.probe_count.z * ((frame_index+1) & 0x1);
    let probe_texture_base_xy = probe_index.xy * settings.probe_surface_resolution;
    let probe_texture_z = probe_texture_layer_offset + probe_index.z;

    var probe_texture_index = uint3(probe_texture_base_xy, probe_texture_z);
    return probe_texture_index;
}

float2 ddgi_probe_normal_to_probe_texel(float3 normal)
{
    return map_octahedral(normal);
}

float3 ddgi_probe_texel_to_probe_normal(float2 texel_index)
{
    return unmap_octahedral(texel_index);
}



// Grid space starts at 0,0,0 at the min probe and ends at settings.probe_count
// floor(grid_space_pos) is the base probe of the position
// frac(grid_space_pos) are the interpolators for the probes around that position
func ddgi_world_space_to_grid_coordinate(
    RenderGlobalData* globals,
    DDGISettings settings,
    float3 position
) -> float3
{
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : globals.camera.position;
    float3 min_probe_world_position = ddgi_probe_index_to_worldspace(settings, probe_anchor, uint3(0,0,0)); 
    float3 min_probe_relative_position = position - min_probe_world_position;
    float3 grid_space_coordinate = min_probe_relative_position * rcp(settings.probe_range) * settings.probe_count;
    return grid_space_coordinate;
}

func ddgi_sample_nearest(
    RenderGlobalData* globals,
    DDGISettings settings,
    float3 position,
    float3 direction,
    RWTexture2DArray<float4> probes
) -> float4
{
    float3 grid_coord = ddgi_world_space_to_grid_coordinate(globals, settings, position);
    int3 base_probe = floor(grid_coord);
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
                    uint3 probe_texture_base_index = ddgi_probe_base_texture_index(globals.ddgi_settings, probe_index, globals.frame_index);
                    uint3 probe_texture_index = probe_texture_base_index + uint3(octa_index.x, octa_index.y, 0);
                    float4 probe_fetch = probes[probe_texture_index];
                    accum += probe_weight * probe_fetch;
                }
            }
        }
    }
    return accum * rcp(8.0f);
}

func ddgi_sample_nearest_rt_occlusion(
    RenderGlobalData* globals,
    DDGISettings settings,
    float3 position,
    float3 direction,
    RaytracingAccelerationStructure tlas,
    RWTexture2DArray<float4> probes
) -> float4
{
    float3 grid_coord = ddgi_world_space_to_grid_coordinate(globals, settings, position);
    int3 base_probe = floor(grid_coord);
    float3 interpolants = frac(grid_coord);
    float3 probe_normal = direction;

    float3 cell_size = float3(settings.probe_range) / float3(settings.probe_count);
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : globals.camera.position;

    float4 accum = float4(0,0,0,0);
    float active_probes = 0;
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
                    float3 probe_position = ddgi_probe_index_to_worldspace(settings, probe_anchor, probe_index);

                    float distance = length(probe_position - position) * 1.01f;
                    float3 direction = normalize(probe_position - position);

                    float t = rt_free_path(tlas, position, direction, distance);

                    bool visible = t >= distance;
                    if (!visible) 
                        continue;

                    float2 octa_index = floor(float(settings.probe_surface_resolution) * map_octahedral(probe_normal));
                    uint3 probe_texture_base_index = ddgi_probe_base_texture_index(globals.ddgi_settings, probe_index, globals.frame_index);
                    uint3 probe_texture_index = probe_texture_base_index + uint3(octa_index.x, octa_index.y, 0);
                    float4 probe_fetch = probes[probe_texture_index];
                    accum += probe_weight * probe_fetch;
                    active_probes += 1;
                }
            }
        }
    }
    if (active_probes == 0)
    {
        return float4(0,0,0,0);
    }
    else
    {
        return accum * rcp(active_probes);
    }
}

func ddgi_visible_probes(
    RenderGlobalData* globals,
    DDGISettings settings,
    float3 position,
    float3 direction,
    RaytracingAccelerationStructure tlas,
    RWTexture2DArray<float4> probes
) -> float
{
    float3 grid_coord = ddgi_world_space_to_grid_coordinate(globals, settings, position);
    int3 base_probe = floor(grid_coord);
    float3 interpolants = frac(grid_coord);
    float3 probe_normal = direction;

    float3 cell_size = float3(settings.probe_range) / float3(settings.probe_count);
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : globals.camera.position;

    float4 accum = float4(0,0,0,0);
    float active_probes = 0;
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
                    float3 probe_position = ddgi_probe_index_to_worldspace(settings, probe_anchor, probe_index);

                    float distance = length(probe_position - position) * 1.01f;
                    float3 direction = normalize(probe_position - position);

                    float t = rt_free_path(tlas, position, direction, distance);

                    //bool visible = t >= distance;
                    //if (!visible) 
                    //    continue;

                    active_probes += 1;
                }
            }
        }
    }
    return active_probes * rcp(8);
}