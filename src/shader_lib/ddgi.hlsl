#pragma once

#include "daxa/daxa.inl"
#include "../shader_shared/ddgi.inl"
#include "../shader_lib/misc.hlsl"

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

float3 ddgi_probe_index_to_worldspace(DDGISettings settings, float3 main_camera_pos, uint3 probe_index)
{
    float3 ddgi_grid_cell_size = settings.probe_range / settings.probe_count;
    float3 center_grid_cell_min_probe_pos = float3(
        f32_round_down_to_multiple(main_camera_pos.x, ddgi_grid_cell_size.x),
        f32_round_down_to_multiple(main_camera_pos.y, ddgi_grid_cell_size.y),
        f32_round_down_to_multiple(main_camera_pos.z, ddgi_grid_cell_size.z),
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

float4 ddgi_cos_sample_probe(DDGISettings settings, Texture2DArray<float4> probe_radiance, uint3 probe_index, float3 sample_direction)
{
    // Sample half the probes texels,
    // Weigh them by the cosine of the sampledirection and texel world normal
    float3 sample_probe_normal = -sample_direction;
    float2 probe_texel = ddgi_probe_normal_to_probe_texel(sample_probe_normal);

    // To sample half the probes texels around the sample directions texel, we loop 
    uint sampling_width = uint(float(settings.probe_surface_resolution) * sqrt(2));
    for (uint x = 0; x < sampling_width; ++x)
    {
        for (uint y = 0; y < sampling_width; ++y)
        {
            
        }
    }
}