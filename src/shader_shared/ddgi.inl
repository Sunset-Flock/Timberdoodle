#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

struct DDGISettings
{
    daxa_f32vec3 probe_range TIDO_DEFAULT_VALUE(32 TIDO_COMMA 32 TIDO_COMMA 16);
    daxa_i32vec3 probe_count TIDO_DEFAULT_VALUE(48 TIDO_COMMA 48 TIDO_COMMA 24);
    daxa_f32vec3 fixed_center_position TIDO_DEFAULT_VALUE(0 TIDO_COMMA 0 TIDO_COMMA 0);
    daxa_b32 fixed_center TIDO_DEFAULT_VALUE(false);
    daxa_b32 draw_debug_probes TIDO_DEFAULT_VALUE(false);
    daxa_i32 probe_surface_resolution TIDO_DEFAULT_VALUE(6);
    daxa_i32 probe_rays_per_frame TIDO_DEFAULT_VALUE(6);
};