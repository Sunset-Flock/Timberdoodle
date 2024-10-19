#include "daxa/daxa.inl"

#include "shared.inl"

struct VolumetricSettings
{
    daxa_i32 enabled;
    daxa_f32 global_fog_density;
#if defined(__cplusplus)
    VolumetricSettings()
        : enabled{1},
          global_fog_density{0.0f}
    {
    }
#endif
};
