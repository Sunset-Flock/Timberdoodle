#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "ddgi_update.inl"

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
    position *= 0.25f;

    let grid_index = float3(instance_index & 0x7, (instance_index >> 3) & 0x7, (instance_index >> 6) & 0x7);
    position += (grid_index - float3(2,2,2)) * 2;

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