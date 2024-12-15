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

#define PGI_BACKFACE_WALL_THICKNESS 0.2f

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

// The Texel res for trace, color and depth texture is different. Must pass the corresponding size here.
uint3 pgi_probe_texture_base_offset(PGISettings settings, int texel_res, int3 probe_index)
{
    let probe_texture_base_xy = probe_index.xy * texel_res;
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

float2 pgi_probe_trace_noise(int3 probe_index, int frame_index)
{
    const uint seed = (probe_index.x * 1823754 + probe_index.y * 5232 + probe_index.z * 21 + frame_index);
    rand_seed(seed);
    float2 in_texel_offset = { rand(), rand() };
    return in_texel_offset;
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

static uint debug_pixel = 0;

func octahedtral_texel_wrap(int2 index, int2 resolution) -> int2
{
    // Octahedral texel clamping is very strange..
    if (index.y >= resolution.y || index.y == -1)
    {
        index.y = clamp(index.y, 0, resolution.y-1);
        // Mirror x sample when y is out of bounds
        index.x = resolution.x - 1 - index.x;
    }
    if (index.x >= resolution.x|| index.x == -1)
    {
        index.x = clamp(index.x, 0, resolution.x-1);
        // Mirror y sample when x is out of bounds
        index.y = resolution.y - 1 - index.y;
    }
    return index;
}

func pgi_sample_probe_irradiance(
    RenderGlobalData* globals,
    PGISettings settings,
    float3 shading_normal,
    Texture2DArray<float4> probes,
    int3 probe_index) -> float3
{
    // Based on the texture index we linearly subsample the probes image with a 2x2 kernel.
    float2 probe_octa_uv = map_octahedral(shading_normal);
    float2 probe_local_texel = probe_octa_uv * float(settings.probe_surface_resolution);
    // FLOORING IS REQUIRED HERE AS FLOAT TO INT CONVERSION ALWAYS ROUNDS TO 0, NOT TO THE LOWER NUMBER!
    int2 probe_local_base_texel = int2(floor(probe_local_texel - 0.5f));
    float2 xy_base_weights = frac(probe_local_texel - 0.5f + float(settings.probe_surface_resolution));
    int3 base_offset = pgi_probe_texture_base_offset(settings, settings.probe_surface_resolution, probe_index);

    float3 linearly_filtered_samples = float3(0,0,0);
    for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 2; ++x)
    {
        int2 xy_sample_offset = int2(x,y);
        int2 probe_local_sample_texel = probe_local_base_texel + xy_sample_offset;
        probe_local_sample_texel = octahedtral_texel_wrap(probe_local_sample_texel, settings.probe_surface_resolution.xx);

        int3 sample_texel = base_offset + int3(probe_local_sample_texel, 0);
        float3 sample = probes[sample_texel].rgb;
        float weight = 
            (x != 0 ? xy_base_weights.x : 1.0f - xy_base_weights.x) *
            (y != 0 ? xy_base_weights.y : 1.0f - xy_base_weights.y);
        linearly_filtered_samples += weight * sample;
    }
    return linearly_filtered_samples;
}

func pgi_sample_probe_visibility(
    RenderGlobalData* globals,
    PGISettings settings,
    float3 shading_normal,
    Texture2DArray<float2> probe_visibility,
    int3 probe_index) -> float2 // returns visibility (x) and certainty (y)
{
    // Based on the texture index we linearly subsample the probes image with a 2x2 kernel.
    float2 probe_octa_uv = map_octahedral(shading_normal);
    float2 probe_local_texel = probe_octa_uv * float(settings.probe_visibility_resolution);
    // FLOORING IS REQUIRED HERE AS FLOAT TO INT CONVERSION ALWAYS ROUNDS TO 0, NOT TO THE LOWER NUMBER!
    int2 probe_local_base_texel = int2(floor(probe_local_texel - 0.5f));
    float2 xy_base_weights = frac(probe_local_texel - 0.5f + float(settings.probe_visibility_resolution));
    int3 base_offset = pgi_probe_texture_base_offset(settings, settings.probe_visibility_resolution, probe_index);

    float2 linearly_filtered_samples = float2(0,0);
    for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 2; ++x)
    {
        int2 xy_sample_offset = int2(x,y);
        int2 probe_local_sample_texel = probe_local_base_texel + xy_sample_offset;
        probe_local_sample_texel = octahedtral_texel_wrap(probe_local_sample_texel, settings.probe_visibility_resolution.xx);
        int3 sample_texel = base_offset + int3(probe_local_sample_texel, 0);
        float2 sample = probe_visibility[sample_texel].rg;
        float weight = 
            (x != 0 ? xy_base_weights.x : 1.0f - xy_base_weights.x) *
            (y != 0 ? xy_base_weights.y : 1.0f - xy_base_weights.y);
        linearly_filtered_samples += weight * sample;
    }
    return float2(linearly_filtered_samples.x, linearly_filtered_samples.y);
}

func pgi_sample_irradiance(
    RenderGlobalData* globals,
    PGISettings settings,
    float3 position,
    float3 geo_normal,
    float3 shading_normal,
    float3 view_direction,
    RaytracingAccelerationStructure tlas,
    Texture2DArray<float4> probes,
    Texture2DArray<float2> probe_visibility
) -> float3
{
    position = rt_calc_ray_start(position, geo_normal, view_direction);

    float3 grid_coord = pgi_world_space_to_grid_coordinate(globals, settings, position);
    int3 base_probe = floor(grid_coord);
    float3 interpolants = frac(grid_coord);
    float3 probe_normal = geo_normal;

    float3 cell_size = float3(settings.probe_range) / float3(settings.probe_count);
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : globals.camera.position;

    float3 accum = float3(0,0,0);
    float weight_accum = 0;
    int visible_probes = 0;
    for (int z = 0; z < 2; ++z)
    {
        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < 2; ++x)
            {
                int3 probe_index = base_probe + int3(x,y,z);
                float3 probe_weights = float3(
                    (x == 0 ? 1.0f - interpolants.x : interpolants.x),
                    (y == 0 ? 1.0f - interpolants.y : interpolants.y),
                    (z == 0 ? 1.0f - interpolants.z : interpolants.z)
                );
                probe_weights = float3(
                    smoothstep(0.0f, 1.0f, probe_weights.x),
                    smoothstep(0.0f, 1.0f, probe_weights.y),
                    smoothstep(0.0f, 1.0f, probe_weights.z),
                );
                float probe_weight = probe_weights.x * probe_weights.y * probe_weights.z;

                if (all(probe_index >= int3(0,0,0)) && all(probe_index < settings.probe_count))
                {
                    float3 probe_position = pgi_probe_index_to_worldspace(settings, probe_anchor, probe_index);
                    float3 to_probe_direction = normalize(probe_position - position);
                    float distance = length(probe_position - position) + RAY_MIN_POSITION_OFFSET;

                    float smooth_backface_term = square((1.0f + dot(shading_normal, to_probe_direction)) * 0.5f);
                    probe_weight *= smooth_backface_term;

                    //float t = rt_free_path(tlas, position, to_probe_direction, distance);
                    //bool visible = true;//t >= distance;
                    //if (!visible) 
                    //{
                    //    continue;
                    //}

                    float2 visibility = pgi_sample_probe_visibility(
                        globals,
                        settings,
                        -to_probe_direction,
                        probe_visibility,
                        probe_index
                    );
                    // visibility (Chebyshev)
                    float mean = 0.0f;
                    float std_dev = 0.1f;
                    float visibility_weight = 1.0f;
                    {
                        mean = max(visibility.x, 0.0f);
                        float average_mean_difference = sqrt(visibility.y);
                        // Technically wrong, but leads to much better results than averaging d^2.
                        std_dev = average_mean_difference * 1.5f;
                        float variance = square(std_dev);
                        if (distance > mean)
                        {
                            visibility_weight = variance / (variance + square(distance - mean));
                            visibility_weight = max(0.0001f, visibility_weight * visibility_weight * visibility_weight);
                            const float crushThreshold = 0.2f;
                            if (visibility_weight < crushThreshold)
                            {
                                visibility_weight *= (visibility_weight * visibility_weight * visibility_weight) * (1.f / (crushThreshold * crushThreshold * crushThreshold));
                            }
                        }
                    }
                    {
                        visible_probes += 1;
                        if (debug_pixel)
                        {
                            ShaderDebugLineDraw line = {};
                            line.start = probe_position;
                            line.end = probe_position - to_probe_direction * (mean + std_dev);
                            line.color = visibility_weight.rrr;
                            debug_draw_line(globals.debug, line);
                            ShaderDebugCircleDraw hit = {};
                            hit.position = probe_position - to_probe_direction * (mean);
                            hit.color = float3(0,1,0) * visibility_weight;
                            hit.radius = 0.11f;
                            debug_draw_circle(globals.debug, hit);
                            ShaderDebugCircleDraw start = {};
                            start.position = probe_position - to_probe_direction * (mean - std_dev);
                            start.color = float3(1,0,0) * visibility_weight;
                            start.radius = 0.11f;
                            debug_draw_circle(globals.debug, start);
                            ShaderDebugCircleDraw end = {};
                            end.position = probe_position - to_probe_direction * (mean + std_dev);
                            end.color = float3(0,0,1) * visibility_weight;
                            end.radius = 0.11f;
                            debug_draw_circle(globals.debug, end);
                        }
                    }
                    probe_weight *= visibility_weight;

                    float3 linearly_filtered_samples = pgi_sample_probe_irradiance(
                        globals,
                        settings,
                        shading_normal,
                        probes,
                        probe_index
                    );
                    
                    #if 0 // draw probe influence
                    if (debug_pixel)
                    {
                        ShaderDebugLineDraw line = {};
                        line.start = position;
                        line.end = probe_position;
                        line.color = probe_weight.rrr;
                        debug_draw_line(globals.debug, line);
                        line.start = probe_position;
                        line.end = probe_position + probe_normal * 0.2;
                        line.color = linearly_filtered_samples.rgb;
                        debug_draw_line(globals.debug, line);
                    }
                    #endif

                    accum += probe_weight * linearly_filtered_samples.rgb;
                    weight_accum += probe_weight;
                }
            }
        }
    }
    //return (float(visible_probes) * rcp(8)).xxx;
    if (weight_accum == 0)
    {
        return float3(0,0,0);
    }
    else
    {
        return accum * rcp(weight_accum);
    }
}