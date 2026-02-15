#pragma once

#include "daxa/daxa.inl"

#include "shared.inl"

#define HENYEY_GREENSTEIN 0
#define HENYEY_GREENSTEIN_OCTAVES 1
#define DRAINE 2

struct VolumetricSettings
{
    daxa_i32 enable             TIDO_DEFAULT_VALUE(1);
    daxa_i32vec2 debug_pixel    TIDO_DEFAULT_VALUE(-1 TIDO_COMMA -1);       

    daxa_i32 secondary_steps    TIDO_DEFAULT_VALUE(2);

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

struct CloudVolumeInstance
{
    daxa_ImageViewId cloud_data_texture;
    daxa_ImageViewId cloud_sdf_texture;
    daxa_ImageViewId detail_noise_texture;

    daxa_u32vec3 texture_size;

    daxa_f32mat4x3 transform;
    daxa_f32 albedo;
    daxa_f32 density_scale;
};
DAXA_DECL_BUFFER_PTR_ALIGN(CloudVolumeInstance, 8)

struct CloudVolumeInstancesBufferHead
{   
    daxa_u32 count;
    daxa_BufferPtr(CloudVolumeInstance) instances;
};
DAXA_DECL_BUFFER_PTR_ALIGN(CloudVolumeInstancesBufferHead, 8)