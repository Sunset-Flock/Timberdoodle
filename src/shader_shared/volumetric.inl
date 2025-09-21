#pragma once

#include "daxa/daxa.inl"

#include "shared.inl"

#define HENYEY_GREENSTEIN 0
#define HENYEY_GREENSTEIN_OCTAVES 1
#define DRAINE 2

struct VolumetricSettings
{
    daxa_i32 enable             TIDO_DEFAULT_VALUE(1);
    daxa_u32vec3 size           TIDO_DEFAULT_VALUE(512 TIDO_COMMA 512 TIDO_COMMA 64);
    daxa_f32vec3 position       TIDO_DEFAULT_VALUE(1000.0f TIDO_COMMA -1000.0f TIDO_COMMA 000.0f);
    daxa_f32 scale              TIDO_DEFAULT_VALUE(20.0f);
    daxa_i32vec2 debug_pixel    TIDO_DEFAULT_VALUE(-1 TIDO_COMMA -1);       

    daxa_i32 secondary_steps    TIDO_DEFAULT_VALUE(2);

    // Clouds material
    daxa_f32 clouds_albedo              TIDO_DEFAULT_VALUE(1.0f);
    daxa_f32 clouds_density_scale       TIDO_DEFAULT_VALUE(0.1f);

    // Phase function params
    daxa_u32 use_density_modulated_g    TIDO_DEFAULT_VALUE(0u);
    daxa_i32 phase_function_model       TIDO_DEFAULT_VALUE(HENYEY_GREENSTEIN_OCTAVES);
    daxa_f32 g                          TIDO_DEFAULT_VALUE(0.660f);
    // Only Draine Phase
    daxa_f32 diameter                   TIDO_DEFAULT_VALUE(20.0f);
    // Multi octave Hg
    daxa_f32 w_0                        TIDO_DEFAULT_VALUE(1.32f);
    daxa_f32 w_1                        TIDO_DEFAULT_VALUE(2.7f);
    daxa_i32 octaves                    TIDO_DEFAULT_VALUE(2);
};