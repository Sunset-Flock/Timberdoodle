#pragma once

#include "daxa/daxa.inl"

#define AMBIENT_OCCLUSION_MODE_NONE 0
#define AMBIENT_OCCLUSION_MODE_RTAO 1

struct AoSettings
{
    daxa_i32 mode;
    daxa_i32 sample_count;
    daxa_f32 ao_range;
    daxa_f32 denoiser_accumulation_max_epsi;
    #if defined(__cplusplus)
        auto operator==(AoSettings const & other) const -> bool
        {
            return std::memcmp(this, &other, sizeof(AoSettings)) == 0;
        }
        auto operator!=(AoSettings const & other) const -> bool
        {
            return std::memcmp(this, &other, sizeof(AoSettings)) != 0;
        }
        AoSettings()
            : mode{ AMBIENT_OCCLUSION_MODE_NONE },
            sample_count{ 1 },
            ao_range{ 1.5f },
            denoiser_accumulation_max_epsi{ 0.95f }
        {}
    #endif
};