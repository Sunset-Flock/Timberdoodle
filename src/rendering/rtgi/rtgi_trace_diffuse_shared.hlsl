
#pragma once

#define DAXA_RAY_TRACING 1

#include "rtgi_trace_diffuse.inl"

[[vk::push_constant]] RtgiTraceDiffusePush rtgi_trace_diffuse_push;

static const float3 sky_color = float3(0.5f, 0.7f, 1.0f);
static const float TMAX = 100000000000.0f; // Arbitrary large value

struct RayPayload
{
    float3 color;    
    float t;
    bool skip_sky_shader;
};