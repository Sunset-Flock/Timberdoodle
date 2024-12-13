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
    float3 radiance = pgi_sample_irradiance_probe(push.attach.globals, settings, vertToPix.normal, push.attach.probe_radiance.get(), vertToPix.probe_index);

    return DrawDebugProbesFragmentOut(float4(radiance,1));
}