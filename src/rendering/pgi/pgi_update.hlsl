#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "pgi_update.inl"
#include "../../shader_lib/pgi.hlsl"
#include "../../shader_lib/misc.hlsl"
#include "../../shader_lib/debug.glsl"
#include "../../shader_lib/shading.hlsl"
#include "../../shader_lib/raytracing.hlsl"
#include "../../shader_lib/SH.hlsl"

[[vk::push_constant]] PGIUpdateProbesPush update_probes_push;

[shader("compute")]
[numthreads(PGI_UPDATE_WG_XY,PGI_UPDATE_WG_XY,PGI_UPDATE_WG_Z)]
func entry_update_probes(
    uint3 dtid : SV_DispatchThreadID,
)
{
    let push = update_probes_push;
    PGISettings settings = push.attach.globals.pgi_settings;

    let probe_texel = (dtid.xy % settings.probe_surface_resolution);
    let probe_index = uint3(dtid.xy / settings.probe_surface_resolution, dtid.z);
    uint frame_index = push.attach.globals.frame_index;
    const uint thread_seed = (dtid.x * 1023 + dtid.y * 31 + dtid.z + frame_index * 17);
    rand_seed(thread_seed);

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }
    
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : push.attach.globals.camera.position;
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_anchor, probe_index);
    float2 probe_texel_uv = (float2(probe_texel) + 0.5f) * rcp(settings.probe_surface_resolution);
    float3 probe_texel_normal = pgi_probe_uv_to_probe_normal(probe_texel_uv);

    uint3 probe_texture_base_index = pgi_probe_texture_base_offset(settings, probe_index);
    float3 probe_range = float3(settings.probe_range) * rcp(settings.probe_count) * 2.0f;
    float max_probe_range = max(probe_range.x, max(probe_range.y, probe_range.z));
    
    uint3 probe_texture_index = probe_texture_base_index + uint3(probe_texel, 0);


    float3 prev_frame_radiance = push.attach.probe_radiance.get()[probe_texture_index].rgb;

    int s = settings.probe_surface_resolution;

    float3 cosine_convoluted_trace_result = float3(0.0f,0.0f,0.0f);
    float acc_weight = 0.0f;
    for (int y = 0; y < s; ++y)
    {
        for (int x = 0; x < s; ++x)
        {
            float2 sample_uv = (float2(x,y) + 0.5f) * rcp(s);
            float3 sample_normal = pgi_probe_uv_to_probe_normal(sample_uv);
            float cos_weight = max(0.0f, dot(sample_normal, probe_texel_normal));
            int3 sample_texture_index = probe_texture_base_index + uint3(x,y,0);
            if (cos_weight > 0.0f)
            {
                float3 sample = push.attach.trace_result.get()[sample_texture_index].rgb;
                cosine_convoluted_trace_result += sample * cos_weight;
                acc_weight += cos_weight;
            }
        }
    }
    cosine_convoluted_trace_result *= rcp(acc_weight);


    push.attach.probe_radiance.get()[probe_texture_index].rgb = lerp(prev_frame_radiance, cosine_convoluted_trace_result, 0.005f);
}