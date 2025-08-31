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
#include "../../shader_lib/raytracing.hlsl"
#include "../../shader_lib/SH.hlsl"

struct DrawDebugProbesVertexToPixel
{
    float4 position : SV_Position;
    float3 probe_position;
    float3 normal;
    nointerpolation int4 probe_index;
};

[[vk::push_constant]] PGIDrawDebugProbesPush draw_debug_probe_p;

[shader("vertex")]
func entry_vertex_draw_debug_probes(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> DrawDebugProbesVertexToPixel
{
    let push = draw_debug_probe_p;
    PGISettings* settings = &push.attach.globals.pgi_settings;

    int4 probe_index = {};
    {
        uint indirect_index = instance_index;
        // We want to draw all active probes, not only the updated probes:
        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index + PGI_MAX_UPDATES_PER_FRAME];
        probe_index = pgi_unpack_indirect_probe(indirect_package);
    }

    PGISettings reg_settings = *settings;
    PGICascade reg_cascade = settings->cascades[probe_index.w];
    
    var position = push.probe_mesh_positions[vertex_index];
    var normal = position;
    position *= 0.03f * reg_cascade.max_visibility_distance;

    PGIProbeInfo probe_info = PGIProbeInfo::load(reg_settings, reg_cascade, push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, probe_info, probe_index);
    position += probe_position;

    float4x4 viewproj = push.attach.globals.view_camera.view_proj;

    if (probe_info.validity < 0.1)
    {
        DrawDebugProbesVertexToPixel ret = {};
        return ret;
    }

    DrawDebugProbesVertexToPixel ret = {};
    ret.position = mul(viewproj, float4(position, 1));
    ret.normal = normal;
    ret.probe_index = probe_index;
    ret.probe_position = probe_position;
    return ret;
}

struct DrawDebugProbesFragmentOut
{
    float4 color : SV_Target;
};

[shader("fragment")]
func entry_fragment_draw_debug_probes(DrawDebugProbesVertexToPixel vertToPix) -> DrawDebugProbesFragmentOut
{
    let push = draw_debug_probe_p;
    PGISettings* settings = &push.attach.globals.pgi_settings;
    PGISettings reg_settings = *settings;
    PGICascade reg_cascade = settings->cascades[vertToPix.probe_index.w];
    int3 stable_index = pgi_probe_to_stable_index(reg_settings, reg_cascade, vertToPix.probe_index);
    

    float3 view_ray = -vertToPix.normal;
    float4 color_hysteresis = pgi_sample_probe_color(push.attach.globals, reg_settings, vertToPix.normal, push.attach.probe_color.get(), stable_index);
    float3 color = color_hysteresis.rgb;
    float hysteresis = color_hysteresis.a;
    float2 visibility = 0.01f * pgi_sample_probe_visibility(push.attach.globals, reg_settings, vertToPix.normal, push.attach.probe_visibility.get(), stable_index);
    float mean = abs(visibility.x);
    float mean2 = visibility.y;

    float2 uv = pgi_probe_normal_to_probe_uv(vertToPix.normal);
    float2 texel = floor(uv * reg_settings.probe_visibility_resolution) * rcp(reg_settings.probe_visibility_resolution);

    float exposure = deref(push.attach.exposure);
    color *= exposure;
    visibility *= exposure;

    float3 draw_color = (float3)0;
    switch(reg_settings.debug_probe_draw_mode)
    {
        case PGI_DEBUG_PROBE_DRAW_MODE_OFF: break;
        case PGI_DEBUG_PROBE_DRAW_MODE_IRRADIANCE: draw_color = color; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_DISTANCE: draw_color = visibility.xxx; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_UNCERTAINTY: draw_color = visibility.yyy; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_TEXEL: draw_color = float3(texel,0); break;
        case PGI_DEBUG_PROBE_DRAW_MODE_UV: draw_color = float3(uv,1); break;
        case PGI_DEBUG_PROBE_DRAW_MODE_NORMAL: draw_color = vertToPix.normal * 0.5f + 0.5f; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_HYSTERESIS: draw_color = TurboColormap(hysteresis); break;
    }

    #if defined(DEBUG_PROBE_TEXEL_UPDATE)
    bool debug_mode = any(reg_settings.debug_probe_index != 0);
    if (debug_mode)
    {
        draw_color = float3(visibility,0) * 100;
    }
    #endif

    return DrawDebugProbesFragmentOut(float4(draw_color,1));
}