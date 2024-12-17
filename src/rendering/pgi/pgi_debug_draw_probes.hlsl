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

struct DrawDebugProbesVertexToPixel
{
    float4 position : SV_Position;
    float3 probe_position;
    float3 normal;
    nointerpolation uint3 probe_index;
};

[[vk::push_constant]] PGIDrawDebugProbesPush draw_debug_probe_p;

[shader("vertex")]
func entry_vertex_draw_debug_probes(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> DrawDebugProbesVertexToPixel
{
    let push = draw_debug_probe_p;
    PGISettings settings = push.attach.globals.pgi_settings;
    var position = push.probe_mesh_positions[vertex_index];
    var normal = position;
    position *= 0.125f;

    uint probes_per_z_slice = (push.attach.globals.pgi_settings.probe_count.x * push.attach.globals.pgi_settings.probe_count.y);
    uint probe_z = instance_index / probes_per_z_slice;
    uint probes_per_y_row = push.attach.globals.pgi_settings.probe_count.x;
    uint probe_y = (instance_index - probe_z * probes_per_z_slice) / probes_per_y_row;
    uint probe_x = (instance_index - probe_z * probes_per_z_slice - probe_y * probes_per_y_row);

    float3 probe_anchor = push.attach.globals.pgi_settings.fixed_center ? push.attach.globals.pgi_settings.fixed_center_position : push.attach.globals.camera.position;

    uint3 probe_index = uint3(probe_x, probe_y, probe_z);
    position += pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_anchor, probe_index);

    float4x4* viewproj = {};
    if (push.attach.globals.settings.draw_from_observer != 0)
    {
        viewproj = &push.attach.globals.observer_camera.view_proj;
    }
    else
    {
        viewproj = &push.attach.globals.camera.view_proj;
    }

    DrawDebugProbesVertexToPixel ret = {};
    ret.position = mul(*viewproj, float4(position, 1));
    ret.normal = normal;
    ret.probe_index = probe_index;
    ret.probe_position = pgi_probe_index_to_worldspace(settings, probe_anchor, probe_index);
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
    PGISettings settings = push.attach.globals.pgi_settings;
    

    float3 view_ray = -vertToPix.normal;
    float3 radiance = pgi_sample_probe_irradiance(push.attach.globals, settings, vertToPix.normal, push.attach.probe_radiance.get(), vertToPix.probe_index);
    float2 visibility = 0.01f * pgi_sample_probe_visibility(push.attach.globals, settings, vertToPix.normal, push.attach.probe_visibility.get(), vertToPix.probe_index);
    float mean = abs(visibility.x);
    float mean2 = visibility.y;

    float2 uv = pgi_probe_normal_to_probe_uv(vertToPix.normal);
    float2 texel = floor(uv * settings.probe_visibility_resolution) * rcp(settings.probe_visibility_resolution);

    float3 draw_color = (float3)0;
    switch(settings.debug_probe_draw_mode)
    {
        case PGI_DEBUG_PROBE_DRAW_MODE_OFF: break;
        case PGI_DEBUG_PROBE_DRAW_MODE_IRRADIANCE: draw_color = radiance; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_DISTANCE: draw_color = visibility.xxx; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_UNCERTAINTY: draw_color = visibility.yyy; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_TEXEL: draw_color = float3(texel,1); break;
        case PGI_DEBUG_PROBE_DRAW_MODE_UV: draw_color = float3(uv,1); break;
        case PGI_DEBUG_PROBE_DRAW_MODE_NORMAL: draw_color = vertToPix.normal * 0.5f + 0.5f; break;
    }
    return DrawDebugProbesFragmentOut(float4(draw_color,1));
}