#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "ddgi_update.inl"
#include "../../shader_lib/ddgi.hlsl"
#include "../../shader_lib/misc.hlsl"
#include "../../shader_lib/debug.glsl"

struct DrawDebugProbesVertexToPixel
{
    float4 position : SV_Position;
    float3 normal;
    nointerpolation uint3 probe_index;
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
    position += ddgi_probe_index_to_worldspace(push.attach.globals.ddgi_settings, probe_anchor, probe_index);

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
    DDGISettings settings = push.attach.globals.ddgi_settings;
    //return DrawDebugProbesFragmentOut(float4(vertToPix.normal * 0.5f + 0.5f,1));
    float2 octa_index = floor(float(settings.probe_surface_resolution) * map_octahedral(vertToPix.normal));
    uint3 probe_texture_base_index = ddgi_probe_base_texture_index(push.attach.globals.ddgi_settings, vertToPix.probe_index, push.attach.globals.frame_index);
    uint3 probe_texture_index = probe_texture_base_index + uint3(octa_index.x, octa_index.y, 0);
    float4 radiance = push.attach.probe_radiance.get()[probe_texture_index];
    return DrawDebugProbesFragmentOut(float4(radiance.rgb,1));
}

func trace_ray(RaytracingAccelerationStructure tlas, float3 pos, float3 light_dir) -> float
{
    RayQuery<RAY_FLAG_CULL_NON_OPAQUE |
        RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;

    const float t_min = 0.01f;
    const float t_max = 100.0f;

    RayDesc my_ray = {
        pos,
        t_min,
        light_dir,
        t_max,
    };

    // Set up a trace.  No work is done yet.
    q.TraceRayInline(
        tlas,
        0, // OR'd with flags above
        0xFFFF,
        my_ray);

    // Proceed() below is where behind-the-scenes traversal happens,
    // including the heaviest of any driver inlined code.
    // In this simplest of scenarios, Proceed() only needs
    // to be called once rather than a loop:
    // Based on the template specialization above,
    // traversal completion is guaranteed.
    q.Proceed();


    bool hit = false;
    // Examine and act on the result of the traversal.
    // Was a hit committed?
    if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit = true;
    }

    if (!hit)
    {
        return 100.0f;
    }
    else
    {
        return q.CandidateTriangleRayT();
    }
}

[[vk::push_constant]] DDGIUpdateProbesPush update_probes_push;

[shader("compute")]
[numthreads(DDGI_UPDATE_WG_XYZ,DDGI_UPDATE_WG_XYZ,DDGI_UPDATE_WG_XYZ)]
func entry_update_probes(
    uint3 dtid : SV_DispatchThreadID,
)
{
    let push = update_probes_push;
    DDGISettings settings = push.attach.globals.ddgi_settings;

    let probe_ray_index = dtid.x % settings.probe_surface_resolution;
    let probe_index = uint3(dtid.x / settings.probe_surface_resolution, dtid.y, dtid.z);

    if (any(greaterThanEqual(probe_index, settings.probe_count)))
    {
        return;
    }
    
    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : push.attach.globals.camera.position;
    float3 probe_position = ddgi_probe_index_to_worldspace(push.attach.globals.ddgi_settings, probe_anchor, probe_index);

    uint3 probe_texture_base_index = ddgi_probe_base_texture_index(settings, probe_index, push.attach.globals.frame_index);
    uint3 probe_texture_base_index_prev = ddgi_probe_base_texture_index_prev_frame(settings, probe_index, push.attach.globals.frame_index);
    float3 probe_range = float3(settings.probe_range) * rcp(settings.probe_count) * 2.0f;
    float max_probe_range = max(probe_range.x, max(probe_range.y, probe_range.z));
    
    const uint thread_seed = (dtid.x * 1023 + dtid.y * 31 + dtid.z + push.attach.globals.frame_index * 17);
    rand_seed(thread_seed);
    float noise = rand();
    uint2 probe_octa_index = uint2(probe_ray_index, uint(noise * settings.probe_surface_resolution));
    uint3 probe_texture_index = probe_texture_base_index + uint3(probe_octa_index, 0);
    uint3 probe_texture_index_prev_frame = probe_texture_base_index + uint3(probe_octa_index, 0);

    float2 octa_texel_size = rcp(float(settings.probe_surface_resolution));
    float texel_noise_x = 0;//rand();
    float texel_noise_y = 0;//rand();
    float2 octa_position = (probe_octa_index + float2(texel_noise_x, texel_noise_y)) * rcp(settings.probe_surface_resolution);

    float3 probe_texel_dir = unmap_octahedral(octa_position);

    float dist = min(max_probe_range, trace_ray(daxa::RayTracingAccelerationStructureTable[push.attach.tlas.index()], probe_position, normalize(probe_texel_dir)));

    debug_draw_line(push.attach.globals.debug, ShaderDebugLineDraw(probe_position, probe_position + probe_texel_dir * dist, float3(0.4,0.4,0.8), 0));
    debug_draw_circle(push.attach.globals.debug, ShaderDebugCircleDraw(probe_position + probe_texel_dir * dist, float3(1,1,0), 0.01, 0));

    float4 prev_frame_radiance = push.attach.probe_radiance.get()[probe_texture_index_prev_frame];
    push.attach.probe_radiance.get()[probe_texture_index] = lerp(prev_frame_radiance, float4(dist.xxx,1), 1.0f);
}