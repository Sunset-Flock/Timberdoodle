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

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/misc.hlsl"

[[vk::push_constant]] PGITraceProbeLightingPush pgi_trace_probe_lighting_push;

#define PI 3.1415926535897932384626433832795

float rand_normal_dist() {
    float theta = 2.0 * PI * rand();
    float rho = sqrt(-2.0 * log(rand()));
    return rho * cos(theta);
}

float3 rand_dir() {
    return normalize(float3(
        rand_normal_dist(),
        rand_normal_dist(),
        rand_normal_dist()));
}

float3 rand_hemi_dir(float3 nrm) {
    float3 result = rand_dir();
    return result * sign(dot(nrm, result));
}

struct RayPayload
{
    bool hit;
    float4 color_depth;
    int3 probe_index;
}

struct PGILightVisibilityTester : LightVisibilityTesterI
{
    RaytracingAccelerationStructure tlas;
    RenderGlobalData* globals;
    float sun_light(MaterialPointData material_point, float3 incoming_ray)
    {
        let sky = globals->sky_settings;

        float t_max = 10000.0f;
        float3 start = rt_calc_ray_start(material_point.position, material_point.geometry_normal, incoming_ray);
        float3 dir = sky.sun_direction;
        
        
        RayDesc ray = {};
        ray.Direction = dir;
        ray.Origin = start;
        ray.TMax = t_max;
        ray.TMin = 0.0f;

        RayPayload payload;
        payload.hit = true; // Only runs miss shader. Miss shader writes false.
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 0, 0, ray, payload);

        bool path_occluded = payload.hit;

        return path_occluded ? 0.0f : 1.0f;
    }
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        return 0.0f;
    }
}

#define RAYGEN_SHADING 0

[shader("raygeneration")]
void entry_ray_gen()
{
    let push = pgi_trace_probe_lighting_push;
    PGISettings settings = push.attach.globals.pgi_settings;
    int3 dtid = DispatchRaysIndex().xyz;

    uint indirect_index = {};
    int3 probe_index = {};
    int2 probe_texel = {};
    {
        indirect_index = dtid.x / (settings.probe_trace_resolution * settings.probe_trace_resolution);
        uint local_index = (dtid.x - indirect_index * (settings.probe_trace_resolution * settings.probe_trace_resolution));
        probe_texel.y = local_index / settings.probe_trace_resolution;
        probe_texel.x = local_index - settings.probe_trace_resolution * probe_texel.y;

        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = int3(
            (indirect_package >> 0) & ((1u << 10u) - 1),
            (indirect_package >> 10) & ((1u << 10u) - 1),
            (indirect_package >> 20) & ((1u << 10u) - 1),
        );
    }

    uint frame_index = push.attach.globals.frame_index;

    // Seed is the same for all threads processing a probe.
    // This is important to be able to efficiently reconstruct data between tracing and probe texel updates.
    float2 in_texel_offset = pgi_probe_trace_noise(probe_index, frame_index);

    PGIProbeInfo probe_info = PGIProbeInfo::load(settings, push.attach.probe_info.get(), probe_index);

    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_info, probe_index);

    uint3 probe_texture_base_index = uint3(pgi_indirect_index_to_trace_tex_offset(settings, indirect_index), 0);
    uint3 probe_texture_index = probe_texture_base_index + uint3(probe_texel, 0);
    uint3 trace_result_texture_index = probe_texture_index;

    float2 probe_uv = float2(float2(probe_texel) + in_texel_offset) * rcp(settings.probe_trace_resolution);

    float3 probe_normal = pgi_probe_uv_to_probe_normal(probe_uv);

    // Randomly Offset Origin tangentially to the trace direction
    // Reduces artifacts caused by specific probe positioning.
    float3 sample_origin = probe_position;
    float3 sample_direction = probe_normal;
    if (false && probe_info.validity > 1.0f)
    {
        rand_seed((probe_index.x * 231) ^ (probe_index.y * 3) ^ (probe_index.z * 523) ^ (probe_texel.x * 132) ^ (probe_texel.y * 311) ^ (push.attach.globals.frame_index * 213));
        float3 rand3 = float3(rand(), rand(), rand()) * 2.0f - 1.0f;
        float3 tangent = normalize(cross(sample_direction, rand3));

        int3 stable_index = pgi_probe_to_stable_index(settings, probe_index);
        float2 vis = pgi_sample_probe_visibility(push.attach.globals,settings, tangent, push.attach.probe_visibility.get(), stable_index);

        float offset_len = clamp(settings.probe_spacing.x * 0.5f * 0.33f, 0.01f, vis.x * 0.8f) * rand();
        float3 offset = tangent * offset_len;
        sample_origin += offset;

        ShaderDebugLineDraw line = {};
        line.start = probe_position;
        line.end = sample_origin;
        line.color = float3(0,0,1);
        //debug_draw_line(push.attach.globals.debug, line);
        ShaderDebugCircleDraw circle = {};
        circle.position = sample_origin;
        circle.radius = 0.01;
        circle.color = float3(1,1,1);
        //debug_draw_circle(push.attach.globals.debug, circle);
    }

    RayDesc ray = {};
    ray.Direction = sample_direction;
    ray.Origin = sample_origin;
    ray.TMax = 100000.0f;
    ray.TMin = 0.0f;

    RayPayload payload;
    payload.probe_index = probe_index;

    TraceRay(push.attach.tlas.get(), RAY_FLAG_FORCE_OPAQUE, ~0, 0, 0, 0, ray, payload);

    RWTexture2DArray<float4> trace_result_tex = push.attach.trace_result.get();

    trace_result_tex[trace_result_texture_index] = payload.color_depth;
}

[shader("anyhit")]
void entry_any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // let push = pgi_trace_probe_lighting_push;
    // if (!rt_is_alpha_hit(
    //     push.attach.globals,
    //     push.attach.mesh_instances,
    //     push.scene.meshes,
    //     push.scene.materials,
    //     attr.barycentrics))
    // {
    //     IgnoreHit();
    // }
}

[shader("closesthit")]
void entry_closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = pgi_trace_probe_lighting_push;

    payload.hit = true;

    PGISettings pgi_settings = push.attach.globals.pgi_settings;
    float3 hit_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    uint request_mode = pgi_get_probe_request_mode(
        push.attach.globals,
        push.attach.globals.pgi_settings,
        push.attach.probe_requests.get(),
        payload.probe_index);
    request_mode += 1; // direct(0) becomes indirect(1), indirect(1) becomes none(2) 
    // let in_pgi_volume = all(hit_position > pgi_settings.window_base_position) && all(hit_position < (pgi_settings.window_base_position + pgi_settings.probe_spacing * pgi_settings.probe_count));
    // let pgi_require_hq_eval = RayTCurrent() < push.attach.globals.pgi_settings.probe_spacing.x * 30;
    // if (false && in_pgi_volume && !pgi_require_hq_eval)
    // {
    //     payload.color_depth.rgb = pgi_sample_irradiance_nearest(
    //         push.attach.globals, 
    //         push.attach.globals.pgi_settings, 
    //         WorldRayOrigin() + WorldRayDirection() * RayTCurrent(), 
    //         -WorldRayDirection(), 
    //         -WorldRayDirection(), 
    //         WorldRayDirection(), 
    //         push.attach.probe_radiance.get(), 
    //         push.attach.probe_visibility.get(), 
    //         push.attach.probe_info.get(), 
    //         push.attach.probe_requests.get(), 
    //         request_mode,
    //         true);
    //     payload.color_depth.a = RayTCurrent();
    // }
    // else
    {
        MeshInstance* mi = push.attach.mesh_instances.instances;
        TriangleGeometry tri_geo = rt_get_triangle_geo(
            attr.barycentrics,
            InstanceID(),
            GeometryIndex(),
            PrimitiveIndex(),
            push.scene.meshes,
            push.scene.entity_to_meshgroup,
            push.scene.mesh_groups,
            mi
        );
        TriangleGeometryPoint tri_point = rt_get_triangle_geo_point(
            tri_geo,
            push.scene.meshes,
            push.scene.entity_to_meshgroup,
            push.scene.mesh_groups,
            push.scene.entity_combined_transforms
        );
        MaterialPointData material_point = evaluate_material<SHADING_QUALITY_LOW>(
            push.attach.globals,
            tri_geo,
            tri_point
        );
        bool double_sided_or_blend = ((material_point.material_flags & MATERIAL_FLAG_DOUBLE_SIDED) != MATERIAL_FLAG_NONE);
        bool backface = dot(WorldRayDirection(), tri_point.face_normal) > 0.01f && !double_sided_or_blend;
        payload.color_depth.rgb = float3(0,0,0);
        if (!backface)
        {
            PGILightVisibilityTester light_vis_tester = PGILightVisibilityTester( push.attach.tlas.get(), push.attach.globals);
            payload.color_depth.rgb = shade_material<SHADING_QUALITY_LOW>(
                push.attach.globals, 
                push.attach.sky_transmittance,
                push.attach.sky,
                material_point, 
                WorldRayOrigin(),
                WorldRayDirection(), 
                light_vis_tester, 
                push.attach.probe_radiance.get(), 
                push.attach.probe_visibility.get(), 
                push.attach.probe_info.get(),
                push.attach.probe_requests.get(),
                request_mode
            ).rgb;

            payload.color_depth.a = RayTCurrent();
        }

        if (backface)
        {
            payload.color_depth.a = -RayTCurrent();
        }
    }

}

[shader("miss")]
void entry_miss(inout RayPayload payload)
{
    let push = pgi_trace_probe_lighting_push;

    payload.hit = false;

    #if RAYGEN_SHADING == 0
    payload.color_depth.rgb = shade_sky(push.attach.globals, push.attach.sky_transmittance, push.attach.sky, WorldRayDirection());
    payload.color_depth.a = 1000.0f;
    #endif
}