
#pragma once

#define DAXA_RAY_TRACING 1

#include "rtgi_trace_diffuse.inl"

// I currently do not like the quality tradeoff of using probes for miss shading.
// The performance is also questionable, as the miss shading requests far more probes, causing pgi performance loss.
#define RTGI_USE_PGI_RADIANCE_ON_MISS 0
#define RTGI_USE_PGI_RADIANCE_ON_HIT 1

#define RTGI_USE_PGI_RADIANCE_ON_MISS_TMAX_SCALE 2

[[vk::push_constant]] RtgiTraceDiffusePush rtgi_trace_diffuse_push;

#define RTGI_PGI_RADIANCE_CACHE_TMIN 0.0f

struct RayPayload
{
    uint2 dtid;
    float3 color;    
    float t;
    bool skip_sky_shader;
};