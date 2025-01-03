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
    float t;
    float2 barycentrics;
    uint primitive_index;
    uint geometry_index;
    uint instance_id;
}

[shader("raygeneration")]
void entry_ray_gen()
{
    let push = pgi_trace_probe_lighting_push;
    PGISettings settings = push.attach.globals.pgi_settings;
    const int3 dtid = DispatchRaysIndex().xyz;
    uint frame_index = push.attach.globals.frame_index;

    let probe_texel = (dtid.xy % settings.probe_trace_resolution);
    let probe_index = uint3(dtid.xy / settings.probe_trace_resolution, dtid.z);

    // Seed is the same for all threads processing a probe.
    // This is important to be able to efficiently reconstruct data between tracing and probe texel updates.
    float2 in_texel_offset = pgi_probe_trace_noise(probe_index, frame_index);

    PGIProbeInfo probe_info = PGIProbeInfo::load(push.attach.probe_info.get(), probe_index);

    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : push.attach.globals.camera.position;
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_info, probe_anchor, probe_index);

    uint3 probe_texture_base_index = pgi_probe_texture_base_offset<NO_BORDER>(settings, settings.probe_trace_resolution, probe_index);
    uint3 probe_texture_index = probe_texture_base_index + uint3(probe_texel, 0);
    uint3 trace_result_texture_index = probe_texture_index;

    float2 probe_uv = float2(float2(probe_texel) + in_texel_offset) * rcp(settings.probe_trace_resolution);

    float3 probe_normal = pgi_probe_uv_to_probe_normal(probe_uv);

    RayDesc ray = {};
    ray.Direction = probe_normal;
    ray.Origin = probe_position;
    ray.TMax = 1000.0f;
    ray.TMin = 0.0f;

    RayPayload payload;

    TraceRay(push.attach.tlas.get(), {}, ~0, 0, 0, 0, ray, payload);

    float4 color_depth = {};
    if (payload.hit)
    {
        float3 hit_point = probe_position + probe_normal * payload.t;
        MeshInstance* mi = push.attach.mesh_instances.instances;
        TriangleGeometry tri_geo = rt_get_triangle_geo(
            payload.barycentrics,
            payload.instance_id,
            payload.geometry_index,
            payload.primitive_index,
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
        MaterialPointData material_point = evaluate_material(
            push.attach.globals,
            tri_geo,
            tri_point
        );
        RTLightVisibilityTester light_vis_tester = RTLightVisibilityTester( push.attach.tlas.get(), push.attach.globals );
        light_vis_tester.origin = probe_position;
        color_depth.rgb = shade_material(
            push.attach.globals, 
            push.attach.sky_transmittance,
            push.attach.sky,
            material_point, 
            probe_normal, 
            light_vis_tester, 
            push.attach.probe_radiance.get(), 
            push.attach.probe_visibility.get(), 
            push.attach.probe_info.get(),
            push.attach.tlas.get()
        ).rgb;

        float distance = payload.t;
        bool backface = dot(probe_normal, tri_point.face_normal) > 0.01f;
        bool double_sided_or_blend = ((material_point.material_flags & MATERIAL_FLAG_DOUBLE_SIDED) != MATERIAL_FLAG_NONE);
        if (backface && !double_sided_or_blend)
        {
            distance *= -1.0f;
        }

        color_depth.a = distance;
    }
    else
    {
        color_depth.rgb = shade_sky(push.attach.globals, push.attach.sky_transmittance, push.attach.sky, probe_normal);
        color_depth.a = 1000.0f;
    }

    push.attach.trace_result.get()[trace_result_texture_index] = color_depth;//payload.color_depth;
}

[shader("anyhit")]
void entry_any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = pgi_trace_probe_lighting_push;
    const float3 hit_location = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const uint primitive_index = PrimitiveIndex() * 3;
    GPUMesh *mesh = {};
    const uint mesh_instance_index = InstanceID();
    MeshInstance* mesh_instance = push.attach.mesh_instances.instances + mesh_instance_index;
    mesh = push.attach.globals.scene.meshes + mesh_instance->mesh_index;

    const int primitive_indices[3] = {
        mesh.primitive_indices[primitive_index],
        mesh.primitive_indices[primitive_index + 1],
        mesh.primitive_indices[primitive_index + 2],
    };

    const float2 uvs[3] = {
        mesh.vertex_uvs[primitive_indices[0]],
        mesh.vertex_uvs[primitive_indices[1]],
        mesh.vertex_uvs[primitive_indices[2]],
    };
    const float2 interp_uv = uvs[0] + attr.barycentrics.x * (uvs[1] - uvs[0]) + attr.barycentrics.y * (uvs[2] - uvs[0]);
    const GPUMaterial *material = &push.attach.globals.scene.materials[mesh.material_index];
    
    if (mesh.material_index == INVALID_MANIFEST_INDEX)
    {
        return;
    }

    const bool has_opacity_texture = !material.opacity_texture_id.is_empty();
    const bool alpha_discard_enabled = material.alpha_discard_enabled;
    if(has_opacity_texture && alpha_discard_enabled)
    {
        let oppacity_tex = Texture2D<float>::get(material.opacity_texture_id);
        let oppacity = oppacity_tex.SampleLevel(SamplerState::get(push.attach.globals->samplers.linear_repeat), interp_uv, 2).r;
        if(oppacity < 0.5) {
            IgnoreHit();
        }
    }
}

[shader("closesthit")]
void entry_closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.hit = true;
    payload.t = RayTCurrent();
    payload.barycentrics = attr.barycentrics;
    payload.primitive_index = PrimitiveIndex();
    payload.instance_id = InstanceID();
    payload.geometry_index = GeometryIndex();
}

[shader("miss")]
void entry_miss(inout RayPayload payload)
{
    payload.hit = false;
}