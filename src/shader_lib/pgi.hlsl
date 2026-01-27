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

    static func load(PGISettings settings, PGICascade cascade, Texture2DArray<float4> probe_info_tex, int4 probe_index) -> PGIProbeInfo
    {
        int3 stable_index = pgi_probe_to_stable_index(settings, cascade, probe_index);
        PGIProbeInfo ret = {};
        float4 fetch = probe_info_tex[stable_index];
        ret.offset = fetch.xyz;
        ret.validity = fetch.w;
        return ret;
    }

    static func load(PGISettings settings, PGICascade cascade, RWTexture2DArray<float4> probe_info_tex, int4 probe_index) -> PGIProbeInfo
    {
        int3 stable_index = pgi_probe_to_stable_index(settings, cascade, probe_index);
        PGIProbeInfo ret = {};
        float4 fetch = probe_info_tex[stable_index];
        ret.offset = fetch.xyz;
        ret.validity = fetch.w;
        return ret;
    }

    static func pack(PGIProbeInfo info) -> uint
    {
        return 
            (((reinterpret<uint>((int(clamp(info.offset.x, -2.0f, 2.0f) * 31.0f)))) & 0xFFu) << 0u) | 
            (((reinterpret<uint>((int(clamp(info.offset.y, -2.0f, 2.0f) * 31.0f)))) & 0xFFu) << 8u) | 
            (((reinterpret<uint>((int(clamp(info.offset.z, -2.0f, 2.0f) * 31.0f)))) & 0xFFu) << 16u) | 
            (((reinterpret<uint>((int(clamp(info.validity, 0.0f, 4.0f) * 63.0f)))) & 0xFFu) << 24u);
    }

    static func unpack(uint packed) -> PGIProbeInfo
    {
        PGIProbeInfo ret;
        ret.offset = float3(
            float(reinterpret<int>((packed >> 0u) & 0xFFu)) * rcp(31.0f),
            float(reinterpret<int>((packed >> 8u) & 0xFFu)) * rcp(31.0f),
            float(reinterpret<int>((packed >> 16u) & 0xFFu)) * rcp(31.0f)
        );
        ret.validity = float((packed >> 24u) & 0xFFu) * rcp(63.0f);
        return ret;
    }
}

uint2 pgi_indirect_index_to_trace_tex_offset(PGISettings settings, uint indirect_index)
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
bool pgi_is_cell_base_probe(PGISettings settings, int4 probe_index)
{ 
    return all(probe_index.xyz >= 0) && all(probe_index.xyz < (settings.probe_count-1));
}

int4 pgi_probe_index_to_prev_frame(PGISettings settings, PGICascade cascade, int4 probe_index)
{
    // As the window moves, the probe indices are going the opposite direction
    let cascade_index = probe_index.w;
    return int4(probe_index.xyz + cascade.window_movement_frame_to_frame, cascade_index);
}

float3 pgi_probe_index_to_worldspace(PGISettings settings, PGICascade cascade, PGIProbeInfo probe_info, int4 probe_index)
{
    return (float3(probe_index.xyz) + probe_info.offset) * cascade.probe_spacing + cascade.window_base_position;
}

int3 pgi_probe_to_stable_index(PGISettings settings, PGICascade cascade, int4 probe_index)
{
    // This bitmasking for modulo works only due to the probe count always beeing a power of two.
    let cascade_index = probe_index.w;
    return ((probe_index.xyz + cascade.window_to_stable_index_offset) & (settings.probe_count-1)) + int3(0,0,settings.probe_count.z * cascade_index);
}

static const bool HAS_BORDER = true;
static const bool NO_BORDER = false;

// The Texel res for trace, color and depth texture is different. Must pass the corresponding size here.
int3 pgi_probe_texture_base_offset<let HAS_BORDER: bool>(PGISettings settings, PGICascade cascade, int texel_res, int4 probe_index)
{
    const int3 stable_index = pgi_probe_to_stable_index(settings, cascade, probe_index);
    const int2 probe_texture_base_xy = stable_index.xy * (texel_res + (HAS_BORDER ? 2 : 0)) + (HAS_BORDER ? 1 : 0);
    const int probe_texture_z = stable_index.z;

    const int3 probe_texture_index = int3(probe_texture_base_xy, probe_texture_z);
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
    const float2 in_texel_offset = { rand(), rand() };
    return in_texel_offset;
}

static bool debug_pixel = false;

func pgi_sample_probe_color(
    RenderGlobalData* globals,
    PGISettings settings,
    float3 direction,
    Texture2DArray<float4> probes,
    int3 stable_index,
    bool color_filter_nearest = false) -> float4
{
    const float2 probe_octa_uv = map_octahedral(direction);
    const float2 inv_probe_cnt = settings.probe_count_rcp.xy;
    const float probe_res = float(settings.probe_color_resolution);
    const float inv_border_probe_res = settings.irradiance_resolution_w_border_rcp;
    const float probe_uv_to_border_uv = probe_res * inv_border_probe_res;
    const float2 base_uv = stable_index.xy * inv_probe_cnt;
    const float2 uv = base_uv + (probe_octa_uv * probe_uv_to_border_uv + inv_border_probe_res) * inv_probe_cnt;

    float4 sample = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (color_filter_nearest)
    {
        sample = probes.SampleLevel(globals.samplers.nearest_clamp.get(), float3(uv, stable_index.z), 0);
    }
    else
    {
        sample = probes.SampleLevel(globals.samplers.linear_clamp.get(), float3(uv, stable_index.z), 0);
    }
    return sample;
}

func pgi_sample_probe_visibility(
    RenderGlobalData* globals,
    PGISettings settings,
    float3 shading_normal,
    Texture2DArray<float2> probe_visibility,
    int3 stable_index) -> float2 // returns visibility (x) and certainty (y)
{
    const float2 probe_octa_uv = map_octahedral(shading_normal);
    const float2 inv_probe_cnt = settings.probe_count_rcp.xy;
    const float probe_res = float(settings.probe_visibility_resolution);
    const float inv_border_probe_res = settings.visibility_resolution_w_border_rcp;
    const float probe_uv_to_border_uv = probe_res * inv_border_probe_res;
    const float2 base_uv = stable_index.xy * inv_probe_cnt;
    const float2 uv = base_uv + (probe_octa_uv * probe_uv_to_border_uv + inv_border_probe_res) * inv_probe_cnt;

    const float2 sample = probe_visibility.SampleLevel(globals.samplers.linear_clamp.get(), float3(uv, stable_index.z), 0);
    return sample;
}

// Moves the sample position back along view ray and offset by normal based on probe grid density.
// Greatly reduces self shadowing for corners.
func pgi_calc_biased_sample_position(
    PGISettings settings, 
    PGICascade cascade,
    float3 camera_position,
    float3 position, 
    float3 geo_normal, 
    uint cascade_index) -> float3
{
    const float BIAS_FACTOR = 0.25f;
    const float NORMAL_TO_VIEW_WEIGHT = 0.3f;
    const float origin_to_sample_dst = length(camera_position - position);
    const float sample_offset = min(cascade.probe_spacing.x * BIAS_FACTOR, origin_to_sample_dst * 0.5f);
    const float3 sample_to_origin = normalize(camera_position - position);
    return position + lerp(sample_to_origin, geo_normal, NORMAL_TO_VIEW_WEIGHT) * sample_offset;
}

#define PGI_REQUEST_MODE_DIRECT 0
#define PGI_REQUEST_MODE_INDIRECT 1
#define PGI_REQUEST_MODE_NONE 2

// Pass non adjusted position here.
// This typically yields a lot fewer probe requests while not causing any new artifacts.
func pgi_request_probes(
    RenderGlobalData* globals,
    PGISettings settings,
    PGICascade cascade,
    RWTexture2DArray<uint> probe_requests,
    float3 position,
    uint probe_request_mode,
    int cascade_index)
{
    const bool direct_request_mode = probe_request_mode == PGI_REQUEST_MODE_DIRECT;
    const bool indirect_request_mode = probe_request_mode == PGI_REQUEST_MODE_INDIRECT;
    
    const float3 request_grid_coord = (position - cascade.window_base_position) * cascade.probe_spacing_rcp;
    const int4 request_base_probe = int4(int3(floor(request_grid_coord)), cascade_index);
    if ((direct_request_mode || indirect_request_mode) && all(request_base_probe.xyz >= int3(0,0,0) && request_base_probe.xyz < (settings.probe_count - int3(1,1,1))))
    {
        const int3 base_probe_stable_index = pgi_probe_to_stable_index(settings, cascade, request_base_probe);
        const uint request_package = probe_requests[base_probe_stable_index];
        const uint direct_request_timer = request_package & 0xFF;
        const uint indirect_request_timer = (request_package >> 8) & 0xFF;

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
    PGISettings settings,
    PGICascade cascade,
    RWTexture2DArray<uint> probe_requests,
    int4 probe_index) -> uint
{
    if (all(probe_index.xyz >= int3(0,0,0) && probe_index.xyz < (settings.probe_count - int3(1,1,1))))
    {
        const int3 base_probe_stable_index = pgi_probe_to_stable_index(settings, cascade, probe_index);
        const uint request_package = probe_requests[base_probe_stable_index];
        const bool directly_requested = ((request_package >> 16) & 0x1) != 0;
        const bool indirectly_requested = ((request_package >> 17) & 0x1) != 0;

        if (directly_requested)
        {
            return PGI_REQUEST_MODE_DIRECT;
        }
        if (indirectly_requested)
        {
            return PGI_REQUEST_MODE_INDIRECT;
        }
    }
    return PGI_REQUEST_MODE_NONE;
}

func pgi_grid_coord_of_position(PGISettings settings, PGICascade cascade, float3 position) -> float3
{
    float3 grid_coord = (position - cascade.window_base_position) * cascade.probe_spacing_rcp;
    return grid_coord;
}

#define PGI_SAMPLE_MODE_IRRADIANCE 0
#define PGI_SAMPLE_MODE_RADIANCE 1

#define PGI_CASCADE_MODE_BLEND 0
#define PGI_CASCADE_MODE_STOCHASTIC_BLEND 1
#define PGI_CASCADE_MODE_NEAREST 2

// Add normal based vs position based probe direction/uv/texel selection 
struct PGISampleInfo
{
    int request_mode = PGI_REQUEST_MODE_DIRECT;
    int sample_mode = PGI_SAMPLE_MODE_IRRADIANCE;
    int cascade_mode = PGI_CASCADE_MODE_BLEND;
    bool low_visibility_fade_black = false;
    bool color_filter_nearest = false;
    bool probe_blend_nearest = false;
    bool probe_relative_sample_dir = false;
};

PGISampleInfo PGISampleInfoNearestSurfaceRadiance()
{
    PGISampleInfo ret               = {};
    ret.request_mode                = PGI_REQUEST_MODE_INDIRECT;
    ret.sample_mode                 = PGI_SAMPLE_MODE_RADIANCE;
    ret.cascade_mode                = PGI_CASCADE_MODE_NEAREST;
    ret.low_visibility_fade_black   = true;
    ret.color_filter_nearest        = true;
    ret.probe_blend_nearest         = false;
    ret.probe_relative_sample_dir   = true;
    return ret;
}

func pgi_sample_probe_volume(
    RenderGlobalData* globals,
    PGISettings* settings,
    PGISampleInfo info,
    float3 sample_position,
    float3 camera_position,
    float3 sample_normal, 
    float3 offset_direction, // usually best to use face or geometry normal here.
    Texture2DArray<float4> probes,
    Texture2DArray<float2> probe_visibility,
    Texture2DArray<float4> probe_infos,
    RWTexture2DArray<uint> probe_requests) -> float3
{
    PGISettings reg_settings = *settings;

    float cascade = pgi_select_cascade_smooth_spherical(reg_settings, sample_position - camera_position);
    if (cascade > reg_settings.cascade_count)
    {
        return float3(0,0,0);
    }

    if (info.cascade_mode == PGI_CASCADE_MODE_STOCHASTIC_BLEND)
    {
        const float cascade_high    = ceil(cascade);
        const float cascade_low     = floor(cascade);
        const float cascade_frac    = frac(cascade);
        const float r               = rand();
        cascade                     = r < cascade_frac ? cascade_high : cascade_low;
    }

    const int lower_cascade             = int(floor(cascade));
    const float4 lower_cascade_result   = pgi_sample_probe_volume_cascade(
        globals, settings, info,
        sample_position, camera_position, sample_normal, offset_direction,
        probes, probe_visibility, probe_infos, probe_requests,
        lower_cascade
    );
    const float3 lower_cascade_color            = lower_cascade_result.rgb;
    const float lower_cascade_visibility        = lower_cascade_result.w;
    float3 color = lower_cascade_color;

    const int higher_cascade                    = lower_cascade + 1;
    const bool is_low_cascade_low_visibility    = lower_cascade_visibility < 0.02f;
    const bool is_cascade_blend_region          = lower_cascade != int(ceil(cascade));
    const bool allow_higher_cascade_blending    = info.cascade_mode == PGI_CASCADE_MODE_BLEND && higher_cascade < reg_settings.cascade_count;
    const bool sample_higher_cascade            = (is_low_cascade_low_visibility || is_cascade_blend_region) && allow_higher_cascade_blending;
    if (sample_higher_cascade)
    {
        const float4 higher_cascade_result      = pgi_sample_probe_volume_cascade(
            globals, settings, info,
            sample_position, camera_position, sample_normal, offset_direction,
            probes, probe_visibility, probe_infos, probe_requests,
            higher_cascade
        );
        const float3 higher_cascade_color       = higher_cascade_result.rgb;
        const float higher_cascade_visibility   = higher_cascade_result.w;

        const float cascade_distance_blend      = is_low_cascade_low_visibility ? 0.5f : frac(cascade);

        const float lower_cascade_weight        = lower_cascade_visibility * (1.0f - cascade_distance_blend);
        const float higher_casccade_weight      = higher_cascade_visibility * cascade_distance_blend;
        const float weight_sum_rcp              = rcp(lower_cascade_weight + higher_casccade_weight);
        const float higher_cascade_blend        = higher_casccade_weight * weight_sum_rcp;
        color = lerp(lower_cascade_color, higher_cascade_color, higher_cascade_blend);
    }

    return color;
}


func pgi_sample_probe_volume_cascade(
    RenderGlobalData* globals,
    PGISettings* settings,
    PGISampleInfo info,
    float3 sample_position,
    float3 camera_position,
    float3 sample_normal, 
    float3 offset_direction,
    Texture2DArray<float4> probes,
    Texture2DArray<float2> probe_visibility,
    Texture2DArray<float4> probe_infos,
    RWTexture2DArray<uint> probe_requests,
    int cascade,
) -> float4 {
    // Load settings and cascade data into registers.
    PGICascade reg_cascade = settings->cascades[cascade];
    PGISettings reg_settings = *settings;

    // Request probe in current sample position.
    pgi_request_probes(globals, reg_settings, reg_cascade, probe_requests, sample_position, info.request_mode, cascade);

    // Bias sample position away from surface and towards camera to avoid leaks.
    const float3 visibility_sample_position = pgi_calc_biased_sample_position(reg_settings, reg_cascade, camera_position, sample_position, offset_direction, cascade);
    const float probe_coord_shift = info.probe_blend_nearest ? 0.5f : 0.0f;
    const uint probe_count = info.probe_blend_nearest ? 1 : 8;
    const float3 grid_coord = pgi_grid_coord_of_position(reg_settings, reg_cascade, visibility_sample_position) + probe_coord_shift;
    const int3 base_probe = int3(floor(grid_coord));
    const float3 grid_interpolants = frac(grid_coord);
    const float3 cell_size = reg_cascade.probe_spacing;
    const float min_weight_bias = info.low_visibility_fade_black ? 0.1f : 0.00000001f;

    float3 accum = float3(0,0,0);
    float weight_accum = 0.00001f;
    for (uint probe = 0; probe < probe_count; ++probe)
    {
        const int x = int((probe >> 0u) & 0x1u);
        const int y = int((probe >> 1u) & 0x1u);
        const int z = int((probe >> 2u) & 0x1u);
        const int4 probe_index = int4(clamp(base_probe + int3(x,y,z), int3(0,0,0), reg_settings.probe_count), cascade);
        const int3 stable_index = pgi_probe_to_stable_index(reg_settings, reg_cascade, probe_index);

        {
            float probe_weight = 1.0f;

            PGIProbeInfo probe_info = PGIProbeInfo::load(reg_settings, reg_cascade, probe_infos, probe_index);
            //PGIProbeInfo probe_info = PGIProbeInfo::unpack(packed_probe_infos[probe]); // PGIProbeInfo::load(settings, probe_infos, probe_index);
            if (info.sample_mode != PGI_SAMPLE_MODE_RADIANCE && probe_info.validity < 0.5f)
            {
                probe_weight = 0.0f;
            }
            
            // Trilinear Probe Proximity Weighting
            if (!info.probe_blend_nearest)
            {
                const float3 cell_probe_weights = float3(
                    (x == 0 ? 1.0f - grid_interpolants.x : grid_interpolants.x),
                    (y == 0 ? 1.0f - grid_interpolants.y : grid_interpolants.y),
                    (z == 0 ? 1.0f - grid_interpolants.z : grid_interpolants.z)
                );
                probe_weight *= cell_probe_weights.x * cell_probe_weights.y * cell_probe_weights.z;
            }

            const float3 probe_position = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, probe_info, probe_index);
            const float3 shading_to_probe_direction = normalize(probe_position - sample_position);

            // Backface influence
            // - smooth backface used to ensure smooth transition between probes
            // - normal cosine influence causes hash cutoffs
            const float smooth_backface_term = (1.0f + dot(sample_normal, shading_to_probe_direction)) * 0.5f;
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
                    reg_settings,
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

            if (debug_pixel && reg_settings.debug_probe_influence)
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
                bias_offset_line.end = sample_position;
                bias_offset_line.color = float3(1,1,0);
                debug_draw_line(globals.debug, bias_offset_line);
                ShaderDebugCircleDraw sample_pos = {};
                sample_pos.position = sample_position;
                sample_pos.color = float3(1,1,0);
                sample_pos.radius = 0.04f;
                debug_draw_circle(globals.debug, sample_pos);
                ShaderDebugCircleDraw sample_pos_offset = {};
                sample_pos_offset.position = visibility_sample_position;
                sample_pos_offset.color = float3(1,1,0);
                sample_pos_offset.radius = 0.01f;
                debug_draw_circle(globals.debug, sample_pos_offset);
            }

            int3 color_sample_stable_index = stable_index;
            // Raw radiance is stored in later layers
            if (info.sample_mode == PGI_SAMPLE_MODE_RADIANCE)
            {
                color_sample_stable_index.z += reg_settings.cascade_count * reg_settings.probe_count.z;
            }

            float3 color_sample_direction = sample_normal;
            if (info.probe_relative_sample_dir)
            {
                color_sample_direction = -shading_to_probe_direction;
            }

            const float3 color_sample = pgi_sample_probe_color(
                globals,
                reg_settings,
                color_sample_direction,
                probes,
                color_sample_stable_index,
                info.color_filter_nearest
            ).rgb;

            accum += probe_weight * sqrt(color_sample.rgb);
            weight_accum += probe_weight;
        }
    }

    if (weight_accum == 0)
    {
        return float4(0,0,0,0);
    }
    else
    {
        const float integration_weight = info.sample_mode == PGI_SAMPLE_MODE_IRRADIANCE ? (2.0f * 3.141f) : 1.0f;
        weight_accum = max(min_weight_bias, weight_accum);
        return float4(clamp(square(accum * rcp(weight_accum)), float3(0,0,0), float3(1,1,1) * 100000.0f) * integration_weight, weight_accum);
    }
}

func pgi_select_cascade_smooth_spherical(PGISettings settings, /*viewrel = worldpos - campos*/ float3 viewrel_position) -> float
{
    if (settings.debug_force_cascade != -1)
    {
        return float(settings.debug_force_cascade);
    }
    const float3 cascade0_safe_range = settings.cascades[0].probe_spacing * float3(settings.probe_count/2 - 2);
    const float3 normalized_manhattan_cascade0_center_dist = viewrel_position * rcp(cascade0_safe_range);
    const float normalized_cascade0_center_dist = length(normalized_manhattan_cascade0_center_dist);

    const float blend_area_percentage = 1.0f - settings.cascade_blend;
    const float cascade = max(0.0f, log2(normalized_cascade0_center_dist) + 1.0f);
    const float cascade_blend = smoothstep(floor(cascade) + blend_area_percentage, ceil(cascade), cascade);
    const float selected_cascade = lerp(floor(cascade), ceil(cascade), cascade_blend);
    return selected_cascade;
}

func pgi_is_pos_in_cascade(PGISettings settings, PGICascade cascade, /*viewrel = worldpos - campos*/ float3 world_position, int cascade_index) -> bool
{
    if (settings.debug_force_cascade != -1)
    {
        return cascade_index == settings.debug_force_cascade;
    }
    const float3 grid_coord = pgi_grid_coord_of_position(settings, cascade, world_position);
    const int3 base_probe = int3(floor(grid_coord));
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