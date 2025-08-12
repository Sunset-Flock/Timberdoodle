
#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtgi_trace_diffuse.inl"
#include "rtgi_trace_diffuse_shared.hlsl"

#include "shader_lib/misc.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/shading.hlsl"
#include "shader_lib/vsm_sampling.hlsl"

func trace_shadow_ray(RaytracingAccelerationStructure tlas, float3 position, float3 light_position, float3 flat_normal, float3 incoming_ray) -> bool
{
    float3 start = rt_calc_ray_start(position, flat_normal, incoming_ray);

    RayDesc ray = {};
    ray.Direction = normalize(light_position - position);
    ray.Origin = start;
    ray.TMax = length(light_position - position) * 1.01f;
    ray.TMin = 0.0f;

    RayPayload payload = {};
    payload.skip_sky_shader = true;
    TraceRay(tlas, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, ~0, 0, 0, 0, ray, payload);

    return payload.t == TMAX;
}


struct RtgiLightVisibilityTester : LightVisibilityTesterI
{
    RaytracingAccelerationStructure tlas;
    RenderGlobalData* globals;
    float sun_light(MaterialPointData material_point, float3 incoming_ray)
    {
        let sky = globals->sky_settings;
        let light_visible = trace_shadow_ray(tlas, material_point.position, material_point.position + sky.sun_direction * 1000000, material_point.face_normal, incoming_ray);
        return light_visible ? 1.0f : 0.0f;
    }
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        let push = rtgi_trace_diffuse_push;

        GPUPointLight point_light = globals.scene.point_lights[light_index];
        float3 to_light = point_light.position - material_point.position;
        float3 to_light_dir = normalize(to_light);

        //return 1.0f;
        let RAYTRACED_POINT_SHADOWS = false;
        if (RAYTRACED_POINT_SHADOWS)
        {        
            let light_visible = trace_shadow_ray(tlas, material_point.position, point_light.position, material_point.face_normal, incoming_ray);
            return light_visible ? 1.0f : 0.0f;
        }
        else
        {
            const float point_norm_dot = dot(material_point.position, to_light_dir);
            return get_vsm_point_shadow_coarse(
                push.attach.globals,
                push.attach.vsm_globals,
                push.attach.vsm_memory_block.get(),
                &(push.attach.vsm_point_spot_page_table[0]),
                push.attach.vsm_point_lights,
                material_point.normal, 
                material_point.position,
                light_index,
                point_norm_dot);
        }
    }
    float spot_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        let push = rtgi_trace_diffuse_push;

        GPUSpotLight spot_light = globals.scene.spot_lights[light_index];

        //return 1.0f;
        let RAYTRACED_POINT_SHADOWS = false;
        if (RAYTRACED_POINT_SHADOWS)
        {        
            let light_visible = trace_shadow_ray(tlas, material_point.position, spot_light.position, material_point.face_normal, incoming_ray);
            return light_visible ? 1.0f : 0.0f;
        }
        else
        {
            return get_vsm_spot_shadow_coarse(
                push.attach.globals,
                push.attach.vsm_globals,
                push.attach.vsm_memory_block.get(),
                &(push.attach.vsm_point_spot_page_table[0]),
                push.attach.vsm_spot_lights,
                material_point.normal, 
                material_point.position,
                light_index);
        }
    }
}

[shader("closesthit")]
void closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = rtgi_trace_diffuse_push;

    MeshInstance* mi = push.attach.mesh_instances.instances;
    TriangleGeometry tri_geo = rt_get_triangle_geo(
        attr.barycentrics,
        InstanceID(),
        GeometryIndex(),
        PrimitiveIndex(),
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.entity_to_meshgroup,
        push.attach.globals.scene.mesh_groups,
        mi
    );
    TriangleGeometryPoint tri_point = rt_get_triangle_geo_point(
        tri_geo,
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.entity_to_meshgroup,
        push.attach.globals.scene.mesh_groups,
        push.attach.globals.scene.entity_combined_transforms
    );
    MaterialPointData material_point = evaluate_material<SHADING_QUALITY_LOW>(
        push.attach.globals,
        tri_geo,
        tri_point
    );
    bool double_sided_or_blend = ((material_point.material_flags & MATERIAL_FLAG_DOUBLE_SIDED) != MATERIAL_FLAG_NONE);
    RtgiLightVisibilityTester light_vis_tester = RtgiLightVisibilityTester(push.attach.tlas.get(), push.attach.globals);

    const float indirect_ao_range = 0.5f;
    const float pgi_enabled = push.attach.globals.pgi_settings.enabled ? 1.0f : 0.0f;
    const float ambient_occlusion = (1.0f - max(0.0f,(indirect_ao_range - RayTCurrent()))/indirect_ao_range) * pgi_enabled;

    payload.color.rgb = shade_material<SHADING_QUALITY_HIGH>(
        push.attach.globals, 
        push.attach.sky_transmittance,
        push.attach.sky,
        material_point, 
        push.attach.globals.view_camera.position,
        WorldRayDirection(), 
        light_vis_tester, 
        push.attach.light_mask_volume.get(),
        push.attach.pgi_irradiance.get(), 
        push.attach.pgi_visibility.get(), 
        push.attach.pgi_info.get(),
        push.attach.pgi_requests.get_formatted(),
        PGI_PROBE_REQUEST_MODE_INDIRECT,
        ambient_occlusion
    ).rgb;

    payload.t = RayTCurrent();
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    let push = rtgi_trace_diffuse_push;

    if (!payload.skip_sky_shader)
    {
        #if RTGI_SHORT_MODE
        const bool stochastic_conservative_sampling = true;
        payload.color = pgi_sample_irradiance(
                push.attach.globals, 
                &push.attach.globals.pgi_settings, 
                WorldRayOrigin(), 
                WorldRayDirection(), 
                WorldRayDirection(), 
                push.attach.globals.view_camera.position,
                push.attach.pgi_irradiance.get(), 
                push.attach.pgi_visibility.get(), 
                push.attach.pgi_info.get(), 
                push.attach.pgi_requests.get_formatted(), 
                PGI_PROBE_REQUEST_MODE_DIRECT,
                false,
                stochastic_conservative_sampling);
        #else
        payload.color = shade_sky(push.attach.globals, push.attach.sky_transmittance, push.attach.sky, WorldRayDirection());
        #endif
    }
    payload.t = TMAX;
}
