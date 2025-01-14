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
    position *= 0.03f * settings.max_visibility_distance;

    int3 probe_index = {};
    if (settings.enable_indirect_sparse)
    {
        uint indirect_index = instance_index;
        // We want to draw all active probes, not only the updated probes
        uint active_probes_offset = settings.probe_count.x * settings.probe_count.y * settings.probe_count.z;
        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index + active_probes_offset];
        probe_index = int3(
            (indirect_package >> 0) & ((1u << 10u) - 1),
            (indirect_package >> 10) & ((1u << 10u) - 1),
            (indirect_package >> 20) & ((1u << 10u) - 1),
        );
    }
    else
    {
        uint probes_per_z_slice = (push.attach.globals.pgi_settings.probe_count.x * push.attach.globals.pgi_settings.probe_count.y);
        uint probe_z = instance_index / probes_per_z_slice;
        uint probes_per_y_row = push.attach.globals.pgi_settings.probe_count.x;
        uint probe_y = (instance_index - probe_z * probes_per_z_slice) / probes_per_y_row;
        uint probe_x = (instance_index - probe_z * probes_per_z_slice - probe_y * probes_per_y_row);

        probe_index = uint3(probe_x, probe_y, probe_z);
    }


    PGIProbeInfo probe_info = PGIProbeInfo::load(settings, push.attach.probe_info.get(), probe_index);
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_info, probe_index);
    position += probe_position;

    float4x4* viewproj = {};
    if (push.attach.globals.settings.draw_from_observer != 0)
    {
        viewproj = &push.attach.globals.observer_camera.view_proj;
    }
    else
    {
        viewproj = &push.attach.globals.camera.view_proj;
    }

    if (probe_info.validity < 0.1)
    {
        DrawDebugProbesVertexToPixel ret = {};
        return ret;
    }

    DrawDebugProbesVertexToPixel ret = {};
    ret.position = mul(*viewproj, float4(position, 1));
    ret.normal = normal;
    ret.probe_index = probe_index;
    ret.probe_position = probe_position;
    return ret;
}

struct DrawDebugProbesFragmentOut
{
    float4 color : SV_Target;
};

// TODO: Precalculate and move to RenderGlobalData 
float compute_exposure(PostprocessSettings post_settings, float average_luminance) 
{
    const float exposure_bias = post_settings.exposure_bias;
    const float calibration = post_settings.calibration;
    const float sensor_sensitivity = post_settings.sensor_sensitivity;
    const float ev100 = log2(average_luminance * sensor_sensitivity * exposure_bias / calibration);
	const float exposure = 1.0 / (1.2 * exp2(ev100));
	return exposure;
}

[shader("fragment")]
func entry_fragment_draw_debug_probes(DrawDebugProbesVertexToPixel vertToPix) -> DrawDebugProbesFragmentOut
{
    let push = draw_debug_probe_p;
    PGISettings settings = push.attach.globals.pgi_settings;
    int3 stable_index = pgi_probe_to_stable_index(settings, vertToPix.probe_index);
    

    float3 view_ray = -vertToPix.normal;
    float4 irradiance_hysteresis = pgi_sample_probe_irradiance(push.attach.globals, settings, vertToPix.normal, push.attach.probe_radiance.get(), stable_index);
    float3 irradiance = irradiance_hysteresis.rgb;
    float hysteresis = irradiance_hysteresis.a;
    float2 visibility = 0.01f * pgi_sample_probe_visibility(push.attach.globals, settings, vertToPix.normal, push.attach.probe_visibility.get(), stable_index);
    float mean = abs(visibility.x);
    float mean2 = visibility.y;

    float2 uv = pgi_probe_normal_to_probe_uv(vertToPix.normal);
    float2 texel = floor(uv * settings.probe_visibility_resolution) * rcp(settings.probe_visibility_resolution);

    float exposure = compute_exposure(push.attach.globals.postprocess_settings, deref(push.attach.luminance_average));
    irradiance *= exposure;
    visibility *= exposure;

    float3 draw_color = (float3)0;
    switch(settings.debug_probe_draw_mode)
    {
        case PGI_DEBUG_PROBE_DRAW_MODE_OFF: break;
        case PGI_DEBUG_PROBE_DRAW_MODE_IRRADIANCE: draw_color = irradiance; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_DISTANCE: draw_color = visibility.xxx; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_UNCERTAINTY: draw_color = visibility.yyy; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_TEXEL: draw_color = float3(texel,1); break;
        case PGI_DEBUG_PROBE_DRAW_MODE_UV: draw_color = float3(uv,1); break;
        case PGI_DEBUG_PROBE_DRAW_MODE_NORMAL: draw_color = vertToPix.normal * 0.5f + 0.5f; break;
        case PGI_DEBUG_PROBE_DRAW_MODE_HYSTERESIS: draw_color = square((hysteresis - 0.7) * (1.0f / (0.7))) * float3(0,1,0); break;
    }
    return DrawDebugProbesFragmentOut(float4(draw_color,1));
}