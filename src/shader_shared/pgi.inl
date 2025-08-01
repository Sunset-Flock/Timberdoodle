#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

#define PGI_DEBUG_PROBE_DRAW_MODE_OFF 0
#define PGI_DEBUG_PROBE_DRAW_MODE_IRRADIANCE 1
#define PGI_DEBUG_PROBE_DRAW_MODE_DISTANCE 2
#define PGI_DEBUG_PROBE_DRAW_MODE_UNCERTAINTY 3
#define PGI_DEBUG_PROBE_DRAW_MODE_TEXEL 4
#define PGI_DEBUG_PROBE_DRAW_MODE_UV 5
#define PGI_DEBUG_PROBE_DRAW_MODE_NORMAL 6
#define PGI_DEBUG_PROBE_DRAW_MODE_HYSTERESIS 7

#define PGI_UPDATE_RATE_FULL 0
#define PGI_UPDATE_RATE_1_OF_2 1
#define PGI_UPDATE_RATE_1_OF_8 2
#define PGI_UPDATE_RATE_1_OF_16 3
#define PGI_UPDATE_RATE_1_OF_32 4
#define PGI_UPDATE_RATE_1_OF_64 5

#define PGI_DEBUG_PROBE_MESH_INDICES 960
#define PGI_MAX_UPDATES_PER_FRAME (1u << 14u)
#define PGI_TRACE_TEX_PROBES_X (1u << 8u)
#define PGI_MAX_REQUESTED_PROBES (1u << 18u)
#define PGI_MAX_CASCADES (16)

struct PGICascade
{
    daxa_i32vec3 window_to_stable_index_offset;
    daxa_f32vec3 window_base_position;
    daxa_i32vec3 window_movement_frame_to_frame;
    daxa_f32vec3 probe_spacing;             // probe_range / probe_count
    daxa_f32vec3 probe_spacing_rcp;         // 1.0f / probe_spacing
    daxa_f32 max_visibility_distance;       // length(probe_spacing) * 1.5f
};

struct PGISettings
{
    daxa_b32 enabled TIDO_DEFAULT_VALUE(true);
    daxa_i32 update_rate TIDO_DEFAULT_VALUE(PGI_UPDATE_RATE_1_OF_8);
    daxa_i32 debug_force_cascade TIDO_DEFAULT_VALUE(-1);
    daxa_i32 debug_probe_draw_mode TIDO_DEFAULT_VALUE(PGI_DEBUG_PROBE_DRAW_MODE_OFF);
    daxa_b32 debug_probe_influence TIDO_DEFAULT_VALUE(false);
    daxa_b32 debug_draw_repositioning TIDO_DEFAULT_VALUE(false);
    daxa_b32 debug_draw_grid TIDO_DEFAULT_VALUE(false);
    daxa_b32 debug_draw_repositioning_forces TIDO_DEFAULT_VALUE(false);
    daxa_i32 probe_irradiance_resolution TIDO_DEFAULT_VALUE(6);
    daxa_i32 probe_trace_resolution TIDO_DEFAULT_VALUE(16);
    daxa_i32 probe_visibility_resolution TIDO_DEFAULT_VALUE(16);
    daxa_b32 probe_repositioning TIDO_DEFAULT_VALUE(true);
    daxa_b32 probe_repositioning_spring_force TIDO_DEFAULT_VALUE(true);
    daxa_i32 cascade_count TIDO_DEFAULT_VALUE(8);
    daxa_f32 cascade_blend TIDO_DEFAULT_VALUE(0.3f);
    // Non photorealistic factor.
    // Allows lights past the cosine cutoff to still contribute to a probes lighting.
    // Helps a lot with edge lighting where the probe resolution is not good enough to calculate bounce light.
    daxa_f32 cos_wrap_around TIDO_DEFAULT_VALUE(0.0f);
    daxa_f32vec3 probe_range TIDO_DEFAULT_VALUE(64 TIDO_COMMA 64 TIDO_COMMA 64);
    daxa_i32vec3 probe_count TIDO_DEFAULT_VALUE(32 TIDO_COMMA 32 TIDO_COMMA 32);
    daxa_i32vec3 debug_probe_index TIDO_DEFAULT_VALUE(0 TIDO_COMMA 0 TIDO_COMMA 0);
    // Calculated by Renderer
    daxa_f32vec3 probe_count_rcp;
    daxa_u32vec3 probe_count_log2; 
    daxa_f32 irradiance_resolution_w_border;
    daxa_f32 irradiance_resolution_w_border_rcp;
    daxa_f32 visibility_resolution_w_border;
    daxa_f32 visibility_resolution_w_border_rcp;
    PGICascade cascades[PGI_MAX_CASCADES];
};