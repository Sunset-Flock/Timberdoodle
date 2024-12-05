#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "ddgi_update.inl"
#include "../../shader_lib/ddgi.hlsl"

struct DrawDebugProbesVertexToPixel
{
    float4 position : SV_Position;
    float3 normal;
};

[[vk::push_constant]] DDGIDrawDebugProbesPush draw_debug_probe_p;

[shader("vertex")]
func entry_vertex_draw_debug_probes(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID) -> DrawDebugProbesVertexToPixel
{
    let push = draw_debug_probe_p;
    var position = push.probe_mesh_positions[vertex_index];
    var normal = position;
    position *= 0.125f;

    uint probes_per_z_slice = (push.attach.globals.ddgi_settings.probe_count.x * push.attach.globals.ddgi_settings.probe_count.y);
    uint probe_z = instance_index / probes_per_z_slice;
    uint probes_per_y_row = push.attach.globals.ddgi_settings.probe_count.x;
    uint probe_y = (instance_index - probe_z * probes_per_z_slice) / probes_per_y_row;
    uint probe_x = (instance_index - probe_z * probes_per_z_slice - probe_y * probes_per_y_row);

    float3 probe_anchor = push.attach.globals.ddgi_settings.fixed_center ? push.attach.globals.ddgi_settings.fixed_center_position : push.attach.globals.camera.position;

    uint3 probe_index = uint3(probe_x, probe_y, probe_z);
    position +=ddgi_probe_index_to_worldspace(push.attach.globals.ddgi_settings, probe_anchor, probe_index);

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
    const float depth_bufer_depth = push.attach.scene_depth_image.get()[int2(vertToPix.position.xy)];
    if ( depth_bufer_depth > vertToPix.position.z ) { discard; }
    return DrawDebugProbesFragmentOut(float4(vertToPix.normal * 0.5f + 0.5f,1));
}