
#pragma once

#define DAXA_RAY_TRACING 1

#include "rtgi_trace_diffuse.inl"

#define RTGI_USE_PGI_RADIANCE_ON_MISS 0
#define RTGI_USE_PGI_RADIANCE_ON_HIT 1

[[vk::push_constant]] RtgiTraceDiffusePush rtgi_trace_diffuse_push;

#define RTGI_PGI_RADIANCE_CACHE_TMIN 0.0f

#if RTGI_USE_PGI_RADIANCE_ON_MISS
static const float TMAX = 4.0f;
#else
static const float TMAX = 100000000000.0f; // Arbitrary large value
#endif

struct RayPayload
{
    uint2 dtid;
    float3 color;    
    float t;
    bool skip_sky_shader;
};