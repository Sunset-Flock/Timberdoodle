#pragma once

#include "daxa/daxa.inl"

#define MAX_LIGHT_INSTANCES_PER_FRAME 128
#define LIGHT_MASK_VOLUME_CHUNK_SIZE 8

struct LightSettings
{
    daxa_i32vec3 mask_volume_cell_count;
    daxa_f32vec3 mask_volume_size;
    daxa_f32vec3 mask_volume_min_pos;
    // Calculated in resolve:
    daxa_f32vec3 mask_volume_cell_size;
    daxa_u32 point_light_count;
    daxa_u32 spot_light_count;
    daxa_u32 first_spot_light_instance;
    daxa_u32 light_count;
    daxa_u32vec4 point_light_mask;
    daxa_u32vec4 spot_light_mask;
    #if defined(__cplusplus)
        LightSettings()
            : mask_volume_cell_count{ 128, 128, 64 },
            mask_volume_size{ 1024, 1024, 512 }
        {
        }
        auto operator==(LightSettings const & other) const -> bool
        {
            return std::memcmp(this, &other, sizeof(LightSettings)) == 0;
        }
    #endif
};