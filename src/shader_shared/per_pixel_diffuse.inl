#pragma once

#include "daxa/daxa.inl"

#define PER_PIXEL_DIFFUSE_MODE_NONE 0
#define PER_PIXEL_DIFFUSE_MODE_RTAO 1
#define PER_PIXEL_DIFFUSE_MODE_SHORT_RANGE_RTGI 2
#define PER_PIXEL_DIFFUSE_MODE_FULL_RTGI 3

struct PerPixelDiffuseSettings
{
    daxa_i32 mode;
    daxa_i32 sample_count;
    daxa_i32 debug_primary_trace;
    daxa_f32 ao_range;
    daxa_f32 short_range_rtgi_range;
    daxa_f32 denoiser_accumulation_max_epsi;
    #if defined(__cplusplus)
        auto operator==(PerPixelDiffuseSettings const & other) const -> bool
        {
            return std::memcmp(this, &other, sizeof(PerPixelDiffuseSettings)) == 0;
        }
        auto operator!=(PerPixelDiffuseSettings const & other) const -> bool
        {
            return std::memcmp(this, &other, sizeof(PerPixelDiffuseSettings)) != 0;
        }
        PerPixelDiffuseSettings()
            : mode{ PER_PIXEL_DIFFUSE_MODE_RTAO },
            sample_count{ 1 },
            debug_primary_trace{ 0 },
            ao_range{ 0.75f },
            short_range_rtgi_range{ 1.5f },
            denoiser_accumulation_max_epsi{ 0.95f }
        {}
    #endif
};