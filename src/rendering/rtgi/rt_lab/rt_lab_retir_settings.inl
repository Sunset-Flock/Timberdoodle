#pragma once

#include "daxa/daxa.inl"

// ReTIR settings (rt_lab_retir.hpp): temporal only reservoir reuse whose per ray intermediate estimates all go to the denoiser. Own settings, the shared
// ReSTIR passes (trace, shade, filter) read the RtLab3Settings the CPU builds from these (rt_lab_retir_to_lab3_settings).
struct RtLabRetirSettings
{
    daxa_i32 temporal_enabled;
    daxa_f32 temporal_max_m;            // cap of the previous reservoir confidence M
    daxa_f32 temporal_plane_tolerance;  // plane distance in lab pixel widths
    daxa_f32 temporal_normal_threshold; // min dot of the face normals
    daxa_i32 temporal_visibility;       // shadow ray from the receiver to the reused sample
    daxa_i32 temporal_max_age;          // sample age cap in frames (0 = off)
    daxa_f32 temporal_max_age_jitter;   // 0..1: the age limit of a reuse is cap x (1 - jitter x random)
    daxa_f32 temporal_partial_scale;    // M multiplier of a candidate that only partly matches
    daxa_f32 jacobian_max;
    daxa_f32 min_reconnection_distance;
};
