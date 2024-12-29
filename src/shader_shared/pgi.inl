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

struct PGISettings
{
    daxa_b32 enabled TIDO_DEFAULT_VALUE(false);
    daxa_b32 fixed_center TIDO_DEFAULT_VALUE(true);
    daxa_f32vec3 fixed_center_position TIDO_DEFAULT_VALUE(0 TIDO_COMMA 0 TIDO_COMMA 8);
    daxa_i32 debug_probe_draw_mode TIDO_DEFAULT_VALUE(PGI_DEBUG_PROBE_DRAW_MODE_OFF);
    daxa_b32 debug_probe_influence TIDO_DEFAULT_VALUE(false);
    daxa_b32 debug_draw_repositioning TIDO_DEFAULT_VALUE(false);
    daxa_i32 probe_radiance_resolution TIDO_DEFAULT_VALUE(6);
    daxa_i32 probe_trace_resolution TIDO_DEFAULT_VALUE(6);
    daxa_i32 probe_visibility_resolution TIDO_DEFAULT_VALUE(12);
    daxa_b32 probe_repositioning TIDO_DEFAULT_VALUE(true);
    daxa_b32 probe_repositioning_spring_force TIDO_DEFAULT_VALUE(true);
    // Non photorealistic factor.
    // Allows lights past the cosine cutoff to still contribute to a probes lighting.
    // Helps a lot with edge lighting where the probe resolution is not good enough to calculate bounce light.
    daxa_f32 cos_wrap_around TIDO_DEFAULT_VALUE(0.3f);
    daxa_f32vec3 probe_range TIDO_DEFAULT_VALUE(32 TIDO_COMMA 32 TIDO_COMMA 32);
    daxa_i32vec3 probe_count TIDO_DEFAULT_VALUE(32 TIDO_COMMA 32 TIDO_COMMA 32);
    daxa_i32vec3 debug_probe_index TIDO_DEFAULT_VALUE(0 TIDO_COMMA 0 TIDO_COMMA 0);
    // Calculated by Renderer
    daxa_f32vec3 probe_spacing;             // probe_range / probe_count
    daxa_f32vec3 probe_spacing_rcp;         // 1.0f / probe_spacing
    daxa_f32 max_visibility_distance;       // length(probe_spacing) * 1.5f
};