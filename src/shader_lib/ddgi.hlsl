#pragma once

#include "daxa/daxa.inl"
#include "../shader_shared/ddgi.inl"

// ===== DDGI Probe Grid =====
// 
// Example Probe Grid 4x4 probes in 2d
//
// O-----------O           O           O <- Probe
// |  probe    |
// |  grid     |
// |  cell     |
// O-----------O           O           O
//
//                   x <- main camera position
//
// O           O           O          O
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