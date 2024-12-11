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
}

[shader("raygeneration")]
void entry_ray_gen()
{
    let push = pgi_trace_probe_lighting_push;
    PGISettings settings = push.attach.globals.pgi_settings;
    const int3 dtid = DispatchRaysIndex().xyz;
    uint frame_index = push.attach.globals.frame_index;
    const uint thread_seed = (dtid.x * 1023 + dtid.y * 31 + dtid.z + frame_index * 17);
    rand_seed(thread_seed);

    let probe_texel = (dtid.xy % settings.probe_surface_resolution);
    let probe_index = uint3(dtid.xy / settings.probe_surface_resolution, dtid.z);

    float3 probe_anchor = settings.fixed_center ? settings.fixed_center_position : push.attach.globals.camera.position;
    float3 probe_position = pgi_probe_index_to_worldspace(push.attach.globals.pgi_settings, probe_anchor, probe_index);

    uint3 probe_texture_base_index = pgi_probe_texture_base_offset(settings, probe_index);
    uint3 probe_texture_index = probe_texture_base_index + uint3(probe_texel, 0);
    uint3 trace_result_texture_index = probe_texture_index;

    float2 in_texel_offset = { rand(), rand() };
    float2 probe_uv = float2(float2(probe_texel) + in_texel_offset) * rcp(settings.probe_surface_resolution);

    float3 probe_normal = pgi_probe_uv_to_probe_normal(probe_uv);

    RayDesc ray = {};
    ray.Direction = probe_normal;
    ray.Origin = probe_position;
    ray.TMax = 1000.0f;
    ray.TMin = 0.01f;

    RayPayload payload;

    TraceRay(push.attach.tlas.get(), {}, ~0, 0, 0, 0, ray, payload);

    push.attach.trace_result.get()[trace_result_texture_index] = payload.color_depth;
}

[shader("anyhit")]
void entry_any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = pgi_trace_probe_lighting_push;
    const float3 hit_location = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const uint geometry_index = GeometryIndex();
    const uint primitive_index = PrimitiveIndex() * 3;
    GPUMesh *mesh = {};

    const uint entity_index = InstanceID();
    const uint mesh_group_index = push.attach.globals.scene.entity_to_meshgroup[entity_index];
    const GPUMeshGroup * mesh_group = &push.attach.globals.scene.mesh_groups[mesh_group_index];
    // TODO: We always select the most detailed lod here.
    const uint lod = 0;
    const uint mesh_index = mesh_group->mesh_lod_group_indices[geometry_index] * MAX_MESHES_PER_LOD_GROUP + lod;
    mesh = push.attach.globals.scene.meshes + mesh_index;

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
    let push = pgi_trace_probe_lighting_push;
    payload.hit = true;

    float3 hit_point = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    TriangleGeometry tri_geo = rt_get_triangle_geo(
        attr.barycentrics,
        InstanceID(),
        GeometryIndex(),
        PrimitiveIndex(),
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.entity_to_meshgroup,
        push.attach.globals.scene.mesh_groups
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
    light_vis_tester.origin = WorldRayOrigin();
    payload.color_depth.rgb = shade_material(push.attach.globals, material_point, WorldRayDirection(), light_vis_tester, push.attach.probe_radiance.get(), push.attach.tlas.get()).rgb;

    payload.color_depth.a = RayTCurrent();
}

[shader("miss")]
void entry_miss(inout RayPayload payload)
{
    let push = pgi_trace_probe_lighting_push;
    payload.hit = false;
    payload.color_depth.rgb = shade_sky(push.attach.globals, push.attach.sky_transmittance, push.attach.sky, WorldRayDirection());
    payload.color_depth.a = RayTCurrent();
}