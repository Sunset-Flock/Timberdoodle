#pragma once

#include "daxa/daxa.inl"

#define PER_PIXEL_DIFFUSE_MODE_NONE 0
#define PER_PIXEL_DIFFUSE_MODE_RTAO 1
#define PER_PIXEL_DIFFUSE_MODE_RTGI 2
#define PER_PIXEL_DIFFUSE_MODE_RTGI_HYBRID 3

struct PerPixelDiffuseSettings
{
    daxa_i32 mode;
    daxa_i32 sample_count;
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
            sample_count{ 1 }
        {}
    #endif
};