#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "pgi_update.inl"
#include "../../shader_lib/pgi.hlsl"
#include "../../shader_lib/misc.hlsl"
#include "../../shader_lib/transform.hlsl"

[[vk::push_constant]] PGIEvalScreenIrradiancePush enty_eval_screen_irradiance_push;

[shader("compute")]
[numthreads(PGI_EVAL_SCREEN_IRRADIANCE_XY, PGI_EVAL_SCREEN_IRRADIANCE_XY, 1)]
func enty_eval_screen_irradiance(uint2 dtid : SV_DispatchThreadID)
{
    let push = enty_eval_screen_irradiance_push;
    let globals = push.attach.globals;

    if (any(dtid >= push.irradiance_image_size))
    {
        return;
    }

    float2 downscaled_pixel_index = {};
    float downscaled_depth = {};
    float3 downscaled_normal = {};
    {
        downscaled_pixel_index = float2(dtid) * 2.0f + 0.5f;

        float4 depths = float4(
            push.attach.view_cam_depth.get()[dtid * 2 + uint2(0,0)],
            push.attach.view_cam_depth.get()[dtid * 2 + uint2(1,0)],
            push.attach.view_cam_depth.get()[dtid * 2 + uint2(0,1)],
            push.attach.view_cam_depth.get()[dtid * 2 + uint2(1,1)]
        );
        downscaled_depth = (depths[0] + depths[1] + depths[2] + depths[3]) * 0.25f;

        uint4 compressed_mapped_normals = uint4(
            push.attach.view_cam_mapped_normals.get()[dtid * 2 + uint2(0,0)],
            push.attach.view_cam_mapped_normals.get()[dtid * 2 + uint2(1,0)],
            push.attach.view_cam_mapped_normals.get()[dtid * 2 + uint2(0,1)],
            push.attach.view_cam_mapped_normals.get()[dtid * 2 + uint2(1,1)]
        );
        half3 mapped_normals[4] = {
            half3(uncompress_normal_octahedral_32(compressed_mapped_normals[0])),
            half3(uncompress_normal_octahedral_32(compressed_mapped_normals[1])),
            half3(uncompress_normal_octahedral_32(compressed_mapped_normals[2])),
            half3(uncompress_normal_octahedral_32(compressed_mapped_normals[3]))
        };
        downscaled_normal = normalize(float3(mapped_normals[0] + mapped_normals[1] + mapped_normals[2] + mapped_normals[3]));
    }

    let camera = push.attach.globals.camera;
    float3 ws_position = pixel_index_to_world_space(camera, downscaled_pixel_index, downscaled_depth);
    float3 primary_ray = normalize(ws_position - globals.camera.position);

    float3 pgi_irradiance = pgi_sample_irradiance(
        globals,
        globals.pgi_settings,
        ws_position,
        downscaled_normal,
        downscaled_normal,
        primary_ray,
        push.attach.probe_radiance.get(),
        push.attach.probe_visibility.get(),
        push.attach.probe_info.get(),
        push.attach.probe_requests.get(),
        /*request probes*/true
    );

    push.attach.pgi_irradiance.get()[dtid].rgb = pgi_irradiance;
}