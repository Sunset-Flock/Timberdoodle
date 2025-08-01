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
#define PGI_BACKFACE_DIST_SCALE 10.0f

struct PGIProbeInfo
{
    float3 offset;
    float validity;

    static func null() -> PGIProbeInfo
    {
        return PGIProbeInfo(float3(0,0,0),0);
    }

    static func load(PGISettings* settings, Texture2DArray<float4> probe_info_tex, int4 probe_index) -> PGIProbeInfo
    {
        int3 stable_index = pgi_probe_to_stable_index(settings, probe_index);
        PGIProbeInfo ret = {};
        float4 fetch = probe_info_tex[stable_index];
        ret.offset = fetch.xyz;
        ret.validity = fetch.w;
        return ret;
    }

    static func load(PGISettings* settings, RWTexture2DArray<float4> probe_info_tex, int4 probe_index) -> PGIProbeInfo
    {
        int3 stable_index = pgi_probe_to_stable_index(settings, probe_index);
        PGIProbeInfo ret = {};
        float4 fetch = probe_info_tex[stable_index];
        ret.offset = fetch.xyz;
        ret.validity = fetch.w;
        return ret;
    }
}

uint2 pgi_indirect_index_to_trace_tex_offset(PGISettings* settings, uint indirect_index)
{
    uint row = indirect_index / PGI_TRACE_TEX_PROBES_X;
    uint col = indirect_index - (row * PGI_TRACE_TEX_PROBES_X);
    return uint2(
        settings.probe_trace_resolution * col,
        settings.probe_trace_resolution * row
    );
}

// The base probe of a cell is the probe with the smallest probe index.
// There are probe-count-1 cells, so any probe except for the last probe in each dimension is a base probe.
bool pgi_is_cell_base_probe(PGISettings* settings, int4 probe_index)
{ 
    return all(probe_index.xyz >= 0) && all(probe_index.xyz < (settings.probe_count-1));
}

int4 pgi_probe_index_to_prev_frame(PGISettings* settings, int4 probe_index)
{
    // As the window moves, the probe indices are going the opposite direction
    let cascade = probe_index.w;
    return int4(probe_index.xyz + settings.cascades[cascade].window_movement_frame_to_frame, cascade);
}

float3 pgi_probe_index_to_worldspace(PGISettings* settings, PGIProbeInfo probe_info, int4 probe_index)
{
    let cascade = probe_index.w;
    return (float3(probe_index.xyz) + probe_info.offset) * settings.cascades[cascade].probe_spacing + settings.cascades[cascade].window_base_position;
}

int3 pgi_probe_to_stable_index(PGISettings* settings, int4 probe_index)
{
    // This bitmasking for modulo works only due to the probe count always beeing a power of two.
    let cascade = probe_index.w;
    return ((probe_index.xyz + settings.cascades[cascade].window_to_stable_index_offset) & (settings.probe_count-1)) + int3(0,0,settings.probe_count.z * cascade);
}

static const bool HAS_BORDER = true;
static const bool NO_BORDER = false;

// The Texel res for trace, color and depth texture is different. Must pass the corresponding size here.
uint3 pgi_probe_texture_base_offset<let HAS_BORDER: bool>(PGISettings* settings, int texel_res, int4 probe_index)
{
    int3 stable_index = pgi_probe_to_stable_index(settings, probe_index);
    let probe_texture_base_xy = stable_index.xy * (texel_res + (HAS_BORDER ? 2 : 0)) + (HAS_BORDER ? 1 : 0);
    let probe_texture_z = stable_index.z;

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

float2 pgi_probe_trace_noise(int4 probe_index, int frame_index)
{
    const uint seed = (probe_index.x * 1823754 + probe_index.y * 5232 + probe_index.z * 21 + frame_index);
    rand_seed(seed);
    float2 in_texel_offset = { rand(), rand() };
    return in_texel_offset;
}

static bool debug_pixel = false;

func pgi_sample_probe_irradiance(
    RenderGlobalData* globals,
    PGISettings* settings,
    float3 shading_normal,
    Texture2DArray<float4> probes,
    int3 stable_index) -> float4
{
    // Based on the texture index we linearly subsample the probes image with a 2x2 kernel.
    float2 probe_octa_uv = map_octahedral(shading_normal);
    float2 inv_probe_cnt = settings.probe_count_rcp.xy;
    float probe_res = float(settings.probe_irradiance_resolution);
    float inv_border_probe_res = settings.irradiance_resolution_w_border_rcp;
    float probe_uv_to_border_uv = probe_res * inv_border_probe_res;

    float2 base_uv = stable_index.xy * inv_probe_cnt;
    float2 uv = base_uv + (probe_octa_uv * probe_uv_to_border_uv + inv_border_probe_res) * inv_probe_cnt;
    float4 linearly_filtered_samples = probes.SampleLevel(globals.samplers.linear_clamp.get(), float3(uv, stable_index.z), 0);
    return linearly_filtered_samples;
}

func pgi_sample_probe_visibility(
    RenderGlobalData* globals,
    PGISettings* settings,
    float3 shading_normal,
    Texture2DArray<float2> probe_visibility,
    int3 stable_index) -> float2 // returns visibility (x) and certainty (y)
{
    // Based on the texture index we linearly subsample the probes image with a 2x2 kernel.
    float2 probe_octa_uv = map_octahedral(shading_normal);
    
    float2 inv_probe_cnt = settings.probe_count_rcp.xy;
    float probe_res = float(settings.probe_visibility_resolution);
    float inv_border_probe_res = settings.visibility_resolution_w_border_rcp;
    float probe_uv_to_border_uv = probe_res * inv_border_probe_res;

    SamplerState sampler = globals.samplers.linear_clamp.get();
    
    #if defined(DEBUG_PROBE_TEXEL_UPDATE)
    bool debug_mode = any(settings.debug_probe_index != 0);
    if (debug_mode)
    {
        sampler = globals.samplers.nearest_clamp.get();
    }
    #endif

    float2 base_uv = stable_index.xy * inv_probe_cnt;
    float2 uv = base_uv + (probe_octa_uv * probe_uv_to_border_uv + inv_border_probe_res) * inv_probe_cnt;
    float2 linearly_filtered_samples = probe_visibility.SampleLevel(sampler, float3(uv, stable_index.z), 0);
    return linearly_filtered_samples;
}

// Moves the sample position back along view ray and offset by normal based on probe grid density.
// Greatly reduces self shadowing for corners.
func pgi_calc_biased_sample_position(
    PGISettings* settings, 
    float3 origin,          // Should usually be camera position to get the best possible bias into the local interior
    float3 position, 
    float3 geo_normal, 
    uint cascade) -> float3
{
    const float BIAS_FACTOR = 0.25f;
    const float NORMAL_TO_VIEW_WEIGHT = 0.3f;
    const float origin_to_sample_dst = length(origin - position);
    const float sample_offset = min(settings.cascades[cascade].probe_spacing.x * BIAS_FACTOR, origin_to_sample_dst * 0.5f);
    const float3 sample_to_origin = normalize(origin - position);
    return position + lerp(sample_to_origin, geo_normal, NORMAL_TO_VIEW_WEIGHT) * sample_offset;
}

#define PGI_PROBE_REQUEST_MODE_DIRECT 0
#define PGI_PROBE_REQUEST_MODE_INDIRECT 1
#define PGI_PROBE_REQUEST_MODE_NONE 2

// Pass non adjusted position here.
// This typically yields a lot fewer probe requests while not causing any new artifacts.
func pgi_request_probes(
    RenderGlobalData* globals,
    PGISettings* settings,
    RWTexture2DArray<uint> probe_requests,
    float3 position,
    uint probe_request_mode,
    int cascade)
{
    const bool direct_request_mode = probe_request_mode == PGI_PROBE_REQUEST_MODE_DIRECT;
    const bool indirect_request_mode = probe_request_mode == PGI_PROBE_REQUEST_MODE_INDIRECT;
    
    const float3 request_grid_coord = (position - settings.cascades[cascade].window_base_position) * settings.cascades[cascade].probe_spacing_rcp;
    const int4 request_base_probe = int4(floor(request_grid_coord), cascade);
    if ((direct_request_mode || indirect_request_mode) && all(request_base_probe.xyz >= int3(0,0,0) && request_base_probe.xyz < (settings.probe_count - int3(1,1,1))))
    {
        int3 base_probe_stable_index = pgi_probe_to_stable_index(settings, request_base_probe);
        uint request_package = probe_requests[base_probe_stable_index];
        uint direct_request_timer = request_package & 0xFF;
        uint indirect_request_timer = (request_package >> 8) & 0xFF;

        if (direct_request_mode && (direct_request_timer < 16))
        {
            InterlockedOr(probe_requests[base_probe_stable_index], 0x3F);
        }
        if (indirect_request_mode && (indirect_request_timer < 16))
        {
            InterlockedOr(probe_requests[base_probe_stable_index], 0xFF << 8);
        }
    }
}

func pgi_get_probe_request_mode(
    RenderGlobalData* globals,
    PGISettings* settings,
    RWTexture2DArray<uint> probe_requests,
    int4 probe_index) -> uint
{
    if (all(probe_index.xyz >= int3(0,0,0) && probe_index.xyz < (settings.probe_count - int3(1,1,1))))
    {
        int3 base_probe_stable_index = pgi_probe_to_stable_index(settings, probe_index);
        uint request_package = probe_requests[base_probe_stable_index];
        bool directly_requested = ((request_package >> 16) & 0x1) != 0;
        bool indirectly_requested = ((request_package >> 17) & 0x1) != 0;

        if (directly_requested)
        {
            return PGI_PROBE_REQUEST_MODE_DIRECT;
        }
        if (indirectly_requested)
        {
            return PGI_PROBE_REQUEST_MODE_INDIRECT;
        }
    }
    return PGI_PROBE_REQUEST_MODE_NONE;
}

func pgi_grid_coord_of_position(PGISettings* settings, float3 position, int cascade) -> float3
{
    float3 grid_coord = (position - settings.cascades[cascade].window_base_position) * settings.cascades[cascade].probe_spacing_rcp;
    return grid_coord;
}

func pgi_sample_irradiance(
    RenderGlobalData* globals,
    PGISettings* settings,
    float3 position,
    float3 geo_normal,
    float3 shading_normal, 
    float3 origin,           // Camera position or Ray Origin
    Texture2DArray<float4> probes,
    Texture2DArray<float2> probe_visibility,
    Texture2DArray<float4> probe_infos,
    RWTexture2DArray<uint> probe_requests,
    int probe_request_mode,
    bool stochastic_cascade_selection = false,
    bool low_certainty_fade_black = false) -> float3
{
    float cascade = pgi_select_cascade_smooth_spherical(settings, position - globals.main_camera.position);
    if (cascade > settings.cascade_count)
    {
        return float3(0,0,0);
    }

    if (stochastic_cascade_selection)
    {
        const float cascade_high = ceil(cascade);
        const float cascade_low = floor(cascade);
        const float cascade_frac = frac(cascade);
        const float r = rand();
        cascade = r < cascade_frac ? cascade_high : cascade_low;
    }

    const float min_weight_til_fade_to_black = low_certainty_fade_black ? 0.2f : 0.00000001f;

    let lower_cascade = int(floor(cascade));
    float4 lower_cascade_result = pgi_sample_irradiance_cascade(
        globals,
        settings,
        position,
        geo_normal,
        shading_normal,
        origin,
        probes,
        probe_visibility,
        probe_infos,
        probe_requests,
        probe_request_mode,
        lower_cascade,
        min_weight_til_fade_to_black
    );
    float3 color = lower_cascade_result.rgb;
    float weight = lower_cascade_result.w;

    let higher_cascade = int(ceil(cascade));
    let two_cascades = lower_cascade != higher_cascade && !stochastic_cascade_selection;
    if (two_cascades)
    {
        float4 higher_cascade_result = pgi_sample_irradiance_cascade(
            globals,
            settings,
            position,
            geo_normal,
            shading_normal,
            origin,
            probes,
            probe_visibility,
            probe_infos,
            probe_requests,
            probe_request_mode,
            higher_cascade
        );

        let lower_cascade_color = color;
        let higher_cascade_color = higher_cascade_result.rgb;
        // let higher_cascade_color = TurboColormap(float(higher_cascade) * rcp(12));
        let cascade_blend = frac(cascade);


        let lower_cascade_weight = weight * (1.0f - cascade_blend);
        let higher_cascade_weight = higher_cascade_result.w * cascade_blend;
        let weight_sum = lower_cascade_weight + higher_cascade_weight;
        color = (lower_cascade_color * lower_cascade_weight + higher_cascade_color * higher_cascade_weight) * rcp(weight_sum);
    }

    return color;
}

func pgi_sample_irradiance_cascade(
    RenderGlobalData* globals,
    PGISettings* settings,
    float3 position,
    float3 geo_normal,
    float3 shading_normal, 
    float3 origin,           // Usually camera position
    Texture2DArray<float4> probes,
    Texture2DArray<float2> probe_visibility,
    Texture2DArray<float4> probe_infos,
    RWTexture2DArray<uint> probe_requests,
    int probe_request_mode,
    int cascade,
    float min_weight_bias = 0.00001f
) -> float4 {
    float3 visibility_sample_position = pgi_calc_biased_sample_position(settings, origin, position, geo_normal, cascade);

    float3 grid_coord = pgi_grid_coord_of_position(settings, visibility_sample_position, cascade);
    int3 base_probe = int3(floor(grid_coord));

    pgi_request_probes(globals, settings, probe_requests, position, probe_request_mode, cascade);

    float3 grid_interpolants = frac(grid_coord);
    
    float3 cell_size = settings.cascades[cascade].probe_spacing;

    float3 accum = float3(0,0,0);
    float weight_accum = 0.00001f;
    for (uint probe = 0; probe < 8; ++probe)
    {
        int x = int((probe >> 0u) & 0x1u);
        int y = int((probe >> 1u) & 0x1u);
        int z = int((probe >> 2u) & 0x1u);
        int4 probe_index = int4(base_probe + int3(x,y,z), cascade);
        int3 stable_index = pgi_probe_to_stable_index(settings, probe_index);

        if (all(probe_index.xyz >= int3(0,0,0)) && all(probe_index.xyz < settings.probe_count))
        {
            float probe_weight = 1.0f;

            PGIProbeInfo probe_info = PGIProbeInfo::load(settings, probe_infos, probe_index);
            if (probe_info.validity < 0.5f)
            {
                probe_weight = 0.0f;
            }
            
            // Trilinear Probe Proximity Weighting
            {
                float3 cell_probe_weights = float3(
                    (x == 0 ? 1.0f - grid_interpolants.x : grid_interpolants.x),
                    (y == 0 ? 1.0f - grid_interpolants.y : grid_interpolants.y),
                    (z == 0 ? 1.0f - grid_interpolants.z : grid_interpolants.z)
                );
                probe_weight *= cell_probe_weights.x * cell_probe_weights.y * cell_probe_weights.z;
            }

            float3 probe_position = pgi_probe_index_to_worldspace(settings, probe_info, probe_index);
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
            float visibility_distance = length(probe_position - visibility_sample_position);
            float3 visibility_to_probe_direction = (probe_position - visibility_sample_position) * rcp(visibility_distance);
            visibility_distance += RAY_MIN_POSITION_OFFSET;
            float average_distance = 0.0f;
            float average_distance_std_dev = 0.0f;
            float visibility_weight = 1.0f;
            {
                float2 visibility = pgi_sample_probe_visibility(
                    globals,
                    settings,
                    -visibility_to_probe_direction,
                    probe_visibility,
                    stable_index
                );

                average_distance = max(visibility.x, 0.0f); // Can contain negative values (back face disabling)
                float average_difference_to_average_distance = visibility.y;
                average_distance_std_dev = average_difference_to_average_distance; // Wrong but works better :P
                float variance = (average_distance_std_dev);
                if (visibility_distance > average_distance)
                {
                    visibility_weight = variance / (variance + square(visibility_distance - average_distance));
                    visibility_weight = square(square(visibility_weight));
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
                stable_index
            ).rgb;

            accum += probe_weight * sqrt(linearly_filtered_samples.rgb);
            weight_accum += probe_weight;
        }
    }

    if (weight_accum == 0)
    {
        return float4(0,0,0,0);
    }
    else
    {
        weight_accum = max(min_weight_bias, weight_accum);
        return float4(clamp(square(accum * rcp(weight_accum)), float3(0,0,0), float3(1,1,1) * 100000.0f) * (2.0f * 3.141f), weight_accum);
    }
}

func pgi_sample_irradiance_nearest(
    RenderGlobalData* globals,
    PGISettings* settings,
    float3 position,
    float3 geo_normal,
    float3 shading_normal,
    float3 origin,          // Camera position or Ray Origin
    Texture2DArray<float4> probes,
    Texture2DArray<float2> probe_visibility,
    Texture2DArray<float4> probe_infos,
    RWTexture2DArray<uint> probe_requests,
    int probe_request_mode,
    int cascade_bias = 0
) -> float3 {

    int cascade = int(floor(pgi_select_cascade_smooth_spherical(settings, position - globals.main_camera.position))) + cascade_bias;
    if (cascade > settings.cascade_count)
    {
        return float3(0,0,0);
    }

    float3 visibility_sample_position = pgi_calc_biased_sample_position(settings, origin, position, geo_normal, cascade);

    float3 grid_coord_shifted = pgi_grid_coord_of_position(settings, visibility_sample_position, cascade) + 0.5f; // shifting here to get the closest probe and not the base probe
    int3 closest_probe = int3(floor(grid_coord_shifted));
    

    pgi_request_probes(globals, settings, probe_requests, position, probe_request_mode, cascade - 1);
    pgi_request_probes(globals, settings, probe_requests, position, probe_request_mode, cascade);
    
    float3 cell_size = settings.cascades[cascade].probe_spacing;

    float3 accum = float3(0,0,0);
    float weight_accum = 0.2f; // heavily underestimate light to prevent leaks
    for (uint probe = 0; probe < 1; ++probe)
    {
        int x = int((probe >> 0u) & 0x1u);
        int y = int((probe >> 1u) & 0x1u);
        int z = int((probe >> 2u) & 0x1u);
        int4 probe_index = int4(closest_probe + int3(x,y,z), cascade);
        int3 stable_index = pgi_probe_to_stable_index(settings, probe_index);

        if (all(probe_index.xyz >= int3(0,0,0)) && all(probe_index.xyz < settings.probe_count))
        {
            float probe_weight = 1.0f;

            PGIProbeInfo probe_info = PGIProbeInfo::load(settings, probe_infos, probe_index);
            if (probe_info.validity < 1.0f)
            {
                if (probe_info.validity < 0.5f)
                {
                    probe_weight = 0.0f;
                }
                else
                {
                    probe_weight *= square(probe_info.validity);
                }
            }

            float3 probe_position = pgi_probe_index_to_worldspace(settings, probe_info, probe_index);
            float3 shading_to_probe_direction = normalize(probe_position - position);

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
                    stable_index
                );

                average_distance = max(visibility.x, 0.0f); // Can contain negative values (back face disabling)
                float average_difference_to_average_distance = visibility.y;
                average_distance_std_dev = average_difference_to_average_distance; // Wrong but works better :P
                float variance = (average_distance_std_dev);
                if (visibility_distance > average_distance)
                {
                    visibility_weight = variance / (variance + square(visibility_distance - average_distance));
                    const float min_visibility = 0.00001f; // Bias. If all probes are occluded we want to fallback to leaking.
                    visibility_weight = max(min_visibility, visibility_weight * visibility_weight * visibility_weight);

                    // Crushing tiny weights reduces leaking BUT does not reduce image blending smoothness.
                    const float crushThreshold = 0.02f;
                    if (visibility_weight < crushThreshold)
                    {
                        visibility_weight *= (visibility_weight * visibility_weight) * (1.f / (crushThreshold * crushThreshold));
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
                stable_index
            ).rgb;

            accum += probe_weight * linearly_filtered_samples.rgb;
            weight_accum += probe_weight;
        }
    }

    if (weight_accum < 0.201f)
    {
        return pgi_sample_irradiance_cascade(
            globals,
            settings,
            position,
            geo_normal,
            shading_normal,
            origin,
            probes,
            probe_visibility,
            probe_infos,
            probe_requests,
            probe_request_mode,
            cascade
        ).rgb;
    }
    else
    {
        return clamp(accum * rcp(weight_accum), float3(0,0,0), float3(1,1,1) * 100000.0f) * (2.0f * 3.141f);
    }
}

func pgi_select_cascade_smooth_spherical(PGISettings* settings, /*viewrel = worldpos - campos*/ float3 viewrel_position) -> float
{
    if (settings.debug_force_cascade != -1)
    {
        return float(settings.debug_force_cascade);
    }
    float3 cascade0_safe_range = settings.cascades[0].probe_spacing * float3(settings.probe_count/2 - 2);
    float3 normalized_manhattan_cascade0_center_dist = viewrel_position * rcp(cascade0_safe_range);
    float normalized_cascade0_center_dist = length(normalized_manhattan_cascade0_center_dist);

    float blend_area_percentage = 1.0f - settings.cascade_blend;
    float cascade = max(0.0f, log2(normalized_cascade0_center_dist) + 1.0f);
    float cascade_blend = smoothstep(floor(cascade) + blend_area_percentage, ceil(cascade), cascade);
    float selected_cascade = lerp(floor(cascade), ceil(cascade), cascade_blend);
    return selected_cascade;
}

func pgi_is_pos_in_cascade(PGISettings* settings, /*viewrel = worldpos - campos*/ float3 world_position, int cascade) -> bool
{
    if (settings.debug_force_cascade != -1)
    {
        return cascade == settings.debug_force_cascade;
    }
    float3 grid_coord = pgi_grid_coord_of_position(settings, world_position, cascade);
    int3 base_probe = int3(floor(grid_coord));
    return all(base_probe >= int3(0,0,0)) && all(base_probe < (settings.probe_count-1));
}

func pgi_pack_indirect_probe(int4 probe_index) -> uint
{
    uint package = 0;
    package = package | (uint(probe_index.x & 0xFF) << 0u);
    package = package | (uint(probe_index.y & 0xFF) << 8u);
    package = package | (uint(probe_index.z & 0xFF) << 16u);
    package = package | (uint(probe_index.w & 0xFF) << 24u);
    return package;
}

func pgi_unpack_indirect_probe(uint package) -> int4
{
    int4 probe_index = (int4)0;
    probe_index.x = (package >> 0u) & 0xFF;
    probe_index.y = (package >> 8u) & 0xFF;
    probe_index.z = (package >> 16u) & 0xFF;
    probe_index.w = (package >> 24u) & 0xFF;
    return probe_index;
}