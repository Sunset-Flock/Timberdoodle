#pragma once

#include "rt_lab_telemetry.inl"

#include "rt_lab_shared.hlsl"

[[vk::push_constant]] RtLabTelemetryPush rt_lab_telemetry_push;

func rt_lab_telemetry_store(RtLabTelemetryPush push, uint index, float value)
{
    push.attach.telemetry[push.slot * RT_LAB_TELEMETRY_SLOT_WORDS + index] = asuint(value);
}

func rt_lab_telemetry_store(RtLabTelemetryPush push, uint index, uint value)
{
    push.attach.telemetry[push.slot * RT_LAB_TELEMETRY_SLOT_WORDS + index] = value;
}

[shader("compute")]
[numthreads(RT_LAB_TELEMETRY_THREADS, 1, 1)]
func entry_rt_lab_telemetry(uint thread_index : SV_GroupIndex)
{
    let push = rt_lab_telemetry_push;
    const CameraInfo camera = push.attach.globals.view_camera;
    const uint2 lab_pixel = push.lab_pixel;
    const uint2 screen_pixel = push.screen_pixel;

    const float depth = push.attach.lab_depth.get()[lab_pixel];
    const bool geometry = depth > 0.0f;
    const uint2 alloc = push.attach.ray_alloc.get()[lab_pixel];
    const uint ray_list_count = push.attach.ray_counters->ray_count;
    uint ray_count = 0u;
    if (geometry && alloc.x < ray_list_count)
    {
        ray_count = min(alloc.y, ray_list_count - alloc.x);
    }
    const uint stored = min(ray_count, uint(RT_LAB_TELEMETRY_MAX_RAYS));

    const float3 face_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[lab_pixel]);
    const float3 detail_normal = uncompress_normal_octahedral_32(push.attach.detail_normal.get()[screen_pixel]);
    const float4 albedo_metalness = push.attach.half_res_albedo.get()[screen_pixel >> 1u];
    const float3 diffuse_albedo = albedo_metalness.rgb * (1.0f - albedo_metalness.a);

    if (thread_index == 0u)
    {
        const uint2 size = uint2(push.attach.globals.settings.render_target_size);
        const float2 ndc = rt_lab_ndc(screen_pixel, rcp(float2(size)));
        const float3 world_position = geometry ? linear_depth_to_world_space(camera, ndc, depth) : float3(0.0f, 0.0f, 0.0f);
        const float3 view_direction = geometry ? normalize(world_position - camera.position) : float3(0.0f, 0.0f, 0.0f);
        rt_lab_telemetry_store(push, 0u, push.stamp);
        rt_lab_telemetry_store(push, 1u, ray_count);
        rt_lab_telemetry_store(push, 2u, stored);
        rt_lab_telemetry_store(push, 3u, geometry ? 1u : 0u);
        rt_lab_telemetry_store(push, 4u, screen_pixel.x);
        rt_lab_telemetry_store(push, 5u, screen_pixel.y);
        rt_lab_telemetry_store(push, 6u, lab_pixel.x);
        rt_lab_telemetry_store(push, 7u, lab_pixel.y);
        rt_lab_telemetry_store(push, 8u, world_position.x);
        rt_lab_telemetry_store(push, 9u, world_position.y);
        rt_lab_telemetry_store(push, 10u, world_position.z);
        rt_lab_telemetry_store(push, 11u, depth);
        rt_lab_telemetry_store(push, 12u, face_normal.x);
        rt_lab_telemetry_store(push, 13u, face_normal.y);
        rt_lab_telemetry_store(push, 14u, face_normal.z);
        rt_lab_telemetry_store(push, 15u, albedo_metalness.a);
        rt_lab_telemetry_store(push, 16u, detail_normal.x);
        rt_lab_telemetry_store(push, 17u, detail_normal.y);
        rt_lab_telemetry_store(push, 18u, detail_normal.z);
        rt_lab_telemetry_store(push, 19u, geometry ? rt_lab_pixel_width_ws(camera, float(push.attach.globals.settings.render_target_size.y), depth) : 0.0f);
        rt_lab_telemetry_store(push, 20u, albedo_metalness.r);
        rt_lab_telemetry_store(push, 21u, albedo_metalness.g);
        rt_lab_telemetry_store(push, 22u, albedo_metalness.b);
        rt_lab_telemetry_store(push, 23u, push.attach.globals.exposure);
        rt_lab_telemetry_store(push, 24u, view_direction.x);
        rt_lab_telemetry_store(push, 25u, view_direction.y);
        rt_lab_telemetry_store(push, 26u, view_direction.z);
    }

    for (uint k = thread_index; k < stored; k += RT_LAB_TELEMETRY_THREADS)
    {
        const RtLabRay ray = push.attach.rays[alloc.x + k];
        const float3 radiance = ray.radiance / RTGI_RADIANCE_SCALE;
        const float3 direction = uncompress_normal_octahedral_32(ray.direction);
        const float hit_t = push.attach.ray_lengths[alloc.x + k];
        const float cos_face = max(dot(face_normal, direction), 1e-3f);
        const float cos_detail = max(dot(detail_normal, direction), 0.0f);
        const float3 modulated = diffuse_albedo * radiance * (cos_detail / cos_face);
        const uint base = RT_LAB_TELEMETRY_HEADER_WORDS + k * RT_LAB_TELEMETRY_RAY_WORDS;
        rt_lab_telemetry_store(push, base + 0u, radiance.x);
        rt_lab_telemetry_store(push, base + 1u, radiance.y);
        rt_lab_telemetry_store(push, base + 2u, radiance.z);
        rt_lab_telemetry_store(push, base + 3u, hit_t);
        rt_lab_telemetry_store(push, base + 4u, modulated.x);
        rt_lab_telemetry_store(push, base + 5u, modulated.y);
        rt_lab_telemetry_store(push, base + 6u, modulated.z);
        rt_lab_telemetry_store(push, base + 7u, cos_detail);
        rt_lab_telemetry_store(push, base + 8u, direction.x);
        rt_lab_telemetry_store(push, base + 9u, direction.y);
        rt_lab_telemetry_store(push, base + 10u, direction.z);
        rt_lab_telemetry_store(push, base + 11u, cos_face);
    }
}
