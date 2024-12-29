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

// Distances in In probe space
#define PGI_DESIRED_RELATIVE_DISTANCE 0.4f 
#define PGI_RELATIVE_REPOSITIONING_STEP 0.05f
#define PGI_RELATIVE_REPOSITIONING_MIN_STEP 0.1f
#define PGI_MAX_RELATIVE_REPOSITIONING 1.0f
#define PGI_BACKFACE_DIST_SCALE 10.0f

struct PGIProbeInfo
{
    float3 offset;
    float validity;

    static func load(Texture2DArray<float4> probe_info_tex, int3 probe_index) -> PGIProbeInfo
    {
        PGIProbeInfo ret = {};
        float4 fetch = probe_info_tex[probe_index];
        ret.offset = fetch.xyz;
        ret.validity = fetch.w;
        return ret;
    }

    static func load(RWTexture2DArray<float4> probe_info_tex, int3 probe_index) -> PGIProbeInfo
    {
        PGIProbeInfo ret = {};
        float4 fetch = probe_info_tex[probe_index];
        ret.offset = fetch.xyz;
        ret.validity = fetch.w;
        return ret;
    }
}

struct PGIProbeState
{
    float3 position_update_vector;
    float d;
};

float3 pgi_probe_index_to_worldspace(PGISettings settings, PGIProbeInfo probe_info, float3 probes_anchor, uint3 probe_index)
{
    float3 pgi_grid_cell_size = settings.probe_spacing;
    float3 center_grid_cell_min_probe_pos = float3(
        f32_round_down_to_multiple(probes_anchor.x, pgi_grid_cell_size.x),
        f32_round_down_to_multiple(probes_anchor.y, pgi_grid_cell_size.y),
        f32_round_down_to_multiple(probes_anchor.z, pgi_grid_cell_size.z),
    );
    return (float3(probe_index) - float3(uint3(settings.probe_count) >> 1) + probe_info.offset) * pgi_grid_cell_size + center_grid_cell_min_probe_pos + settings.fixed_center_position;
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
    //return float2(0.5f,0.5f);
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
    PGIProbeInfo info_dummy = {};
    float3 min_probe_world_position = pgi_probe_index_to_worldspace(settings, info_dummy, probe_anchor, uint3(0,0,0)); 
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
    float2 probe_local_texel = probe_octa_uv * float(settings.probe_radiance_resolution);
    // FLOORING IS REQUIRED HERE AS FLOAT TO INT CONVERSION ALWAYS ROUNDS TO 0, NOT TO THE LOWER NUMBER!
    int2 probe_local_base_texel = int2(floor(probe_local_texel - 0.5f));
    float2 xy_base_weights = frac(probe_local_texel - 0.5f + float(settings.probe_radiance_resolution));
    int3 base_offset = pgi_probe_texture_base_offset(settings, settings.probe_radiance_resolution, probe_index);

    float3 linearly_filtered_samples = float3(0,0,0);
    for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 2; ++x)
    {
        int2 xy_sample_offset = int2(x,y);
        int2 probe_local_sample_texel = probe_local_base_texel + xy_sample_offset;
        probe_local_sample_texel = octahedtral_texel_wrap(probe_local_sample_texel, settings.probe_radiance_resolution.xx);

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

// Moves the sample position back along view ray and offset by normal based on probe grid density.
// Greatly reduces self shadowing for corners.
func pgi_calc_biased_sample_position(PGISettings settings, float3 position, float3 geo_normal, float3 view_direction) -> float3
{
    const float BIAS_FACTOR = 0.15f;
    const float NORMAL_TO_VIEW_WEIGHT = 0.3f;
    return position + lerp(-view_direction, geo_normal, NORMAL_TO_VIEW_WEIGHT) * settings.probe_spacing * BIAS_FACTOR;
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
    Texture2DArray<float2> probe_visibility,
    Texture2DArray<float4> probe_infos
) -> float3 {
    float3 visibility_sample_position = pgi_calc_biased_sample_position(settings, position, geo_normal, view_direction);

    float3 grid_coord = pgi_world_space_to_grid_coordinate(globals, settings, position);
    int3 base_probe = int3(floor(grid_coord));
    float3 grid_interpolants = frac(grid_coord);
    float3 probe_normal = geo_normal;
    

    float3 cell_size = float3(settings.probe_range) / float3(settings.probe_count);
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : globals.camera.position;

    float3 accum = float3(0,0,0);
    float weight_accum = 0;
    for (uint probe = 0; probe < 8; ++probe)
    {
        int x = int((probe >> 0u) & 0x1u);
        int y = int((probe >> 1u) & 0x1u);
        int z = int((probe >> 2u) & 0x1u);
        int3 probe_index = base_probe + int3(x,y,z);

        if (all(probe_index >= int3(0,0,0)) && all(probe_index < settings.probe_count))
        {
            float probe_weight = 1.0f;

            PGIProbeInfo probe_info = PGIProbeInfo::load(probe_infos, probe_index);
            if (probe_info.validity < 1.0f)
            {
                probe_weight = 0.0f;
            }
            
            // Trilinear Probe Proximity Weighting
            {
                float3 probe_grid_coord = base_probe + float3(x,y,z) + probe_info.offset;
                float3 grid_coord_distance = abs(probe_grid_coord - grid_coord);
                float3 distance_probe_weights = float3(
                    sqrt(1.0f - clamp(0.1f, 0.9f, grid_coord_distance.x)),
                    sqrt(1.0f - clamp(0.1f, 0.9f, grid_coord_distance.y)),
                    sqrt(1.0f - clamp(0.1f, 0.9f, grid_coord_distance.z)),
                );
                float3 cell_probe_weights = float3(
                    sqrt(x == 0 ? 1.0f - grid_interpolants.x : grid_interpolants.x),
                    sqrt(y == 0 ? 1.0f - grid_interpolants.y : grid_interpolants.y),
                    sqrt(z == 0 ? 1.0f - grid_interpolants.z : grid_interpolants.z)
                );
                float3 probe_weights = cell_probe_weights * distance_probe_weights;
                probe_weights = smoothstep(float3(0,0,0), float3(1,1,1), probe_weights);
                probe_weight *= probe_weights.x * probe_weights.y * probe_weights.z;
            }

            float3 probe_position = pgi_probe_index_to_worldspace(settings, probe_info, probe_anchor, probe_index);
            float3 shading_to_probe_direction = normalize(probe_position - position);

            // Backface influence
            // - smooth backface used to ensure smooth transition between probes
            // - normal cosine influence causes hash cutoffs
            float smooth_backface_term = (1.0f + dot(shading_normal, shading_to_probe_direction)) * 0.5f;
            probe_weight *= square(smooth_backface_term);

            // visibility (Chebyshev)
            // ===== Shadow Map Visibility Test =====
            // - Shadowmap channel r contains average ray length
            // - Shadowmap channel g contains average *difference to average* of raylength
            // - Original DDGI stores distance^2 in g channel
            //   - I noticed that this leads to very bad results for certain exponential averaging values
            //   - Using the average difference to average raylength is much more stable and "close enough" to the std dev
            //   - Leads to better results in my testing
            float visibility_distance = length(probe_position - visibility_sample_position) + RAY_MIN_POSITION_OFFSET;
            float3 visibility_to_probe_direction = normalize(probe_position - visibility_sample_position);
            float average_distance = 0.0f;
            float average_distance_std_dev = 0.0f;
            float visibility_weight = 1.0f;
            {
                float2 visibility = pgi_sample_probe_visibility(
                    globals,
                    settings,
                    -visibility_to_probe_direction,
                    probe_visibility,
                    probe_index
                );

                average_distance = max(visibility.x, 0.0f); // Can contain negative values (back face disabling)
                float average_difference_to_average_distance = visibility.y;
                average_distance_std_dev = average_difference_to_average_distance; // Wrong but works better :P
                float variance = square(average_distance_std_dev);
                if (visibility_distance > average_distance)
                {
                    visibility_weight = variance / (variance + square(visibility_distance - average_distance));
                    const float min_visibility = 0.00001f; // Bias. If all probes are occluded we want to fallback to leaking.
                    visibility_weight = max(min_visibility, visibility_weight * visibility_weight * visibility_weight);

                    // Crushing tiny weights reduces leaking BUT does not reduce image blending smoothness.
                    const float crushThreshold = 0.2f;
                    if (visibility_weight < crushThreshold)
                    {
                        visibility_weight *= (visibility_weight * visibility_weight * visibility_weight) * (1.f / (crushThreshold * crushThreshold * crushThreshold));
                    }
                }
            }
            probe_weight *= visibility_weight;
            // ===== Shadow Map Visibility Test =====

            if (debug_pixel && settings.debug_probe_influence)
            {
                ShaderDebugLineDraw white_line = {};
                white_line.start = probe_position;
                white_line.end = probe_position - visibility_to_probe_direction * (average_distance - average_distance_std_dev);
                white_line.color = visibility_weight.rrr;
                debug_draw_line(globals.debug, white_line);

                ShaderDebugLineDraw green_line = {};
                green_line.start = probe_position - visibility_to_probe_direction * (average_distance - average_distance_std_dev);
                green_line.end = probe_position - visibility_to_probe_direction * (average_distance);
                green_line.color = visibility_weight.rrr * float3(0,1,0);
                debug_draw_line(globals.debug, green_line);

                ShaderDebugLineDraw blue_line = {};
                blue_line.start = probe_position - visibility_to_probe_direction * (average_distance);
                blue_line.end = probe_position - visibility_to_probe_direction * (average_distance + average_distance_std_dev);
                blue_line.color = visibility_weight.rrr * float3(0,0,1);
                debug_draw_line(globals.debug, blue_line);

                if (probe_weight < 0.001f)
                {
                    ShaderDebugLineDraw black_line = {};
                    black_line.start = probe_position - visibility_to_probe_direction * (average_distance + average_distance_std_dev);
                    black_line.end = visibility_sample_position;
                    black_line.color = float3(1,0.02,0.02) * 0.01;
                    debug_draw_line(globals.debug, black_line);
                }

                ShaderDebugCircleDraw start = {};
                start.position = probe_position - visibility_to_probe_direction * (average_distance - average_distance_std_dev);
                start.color = float3(0,1,0) * visibility_weight;
                start.radius = 0.01f;
                debug_draw_circle(globals.debug, start);
                ShaderDebugCircleDraw end = {};
                end.position = probe_position - visibility_to_probe_direction * (average_distance + average_distance_std_dev);
                end.color = float3(0,0,1) * visibility_weight;
                end.radius = 0.01f;
                debug_draw_circle(globals.debug, end);

                ShaderDebugLineDraw bias_offset_line = {};
                bias_offset_line.start = visibility_sample_position;
                bias_offset_line.end = position;
                bias_offset_line.color = float3(1,1,0);
                debug_draw_line(globals.debug, bias_offset_line);
                ShaderDebugCircleDraw sample_pos = {};
                sample_pos.position = position;
                sample_pos.color = float3(1,1,0);
                sample_pos.radius = 0.04f;
                debug_draw_circle(globals.debug, sample_pos);
                ShaderDebugCircleDraw sample_pos_offset = {};
                sample_pos_offset.position = visibility_sample_position;
                sample_pos_offset.color = float3(1,1,0);
                sample_pos_offset.radius = 0.01f;
                debug_draw_circle(globals.debug, sample_pos_offset);
            }

            float3 linearly_filtered_samples = pgi_sample_probe_irradiance(
                globals,
                settings,
                shading_normal,
                probes,
                probe_index
            );

            accum += (probe_weight) * sqrt(linearly_filtered_samples.rgb);
            weight_accum += probe_weight;
        }
    }

    if (weight_accum == 0)
    {
        return float3(0,0,0);
    }
    else
    {
        return square(accum * rcp(weight_accum));
    }
}