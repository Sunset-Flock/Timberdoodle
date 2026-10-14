#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtgi_trace.inl"
#include "rtgi_shade.inl"

#include "shader_lib/misc.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/shading.hlsl"
#include "shader_lib/vsm_sampling.hlsl"

#include "rtgi_shared.hlsl"

// RTGI ray shading. Two users, ONE shading core:
//   - the ray tracing pipelines that shade in their hit shaders, ray generation in rtgi_trace.hlsl:
//     "Trace RTGI" (classic per-pixel trace): closest_hit (sharp or coarse per ray) / miss.
//     "Trace RTGI Pioneer" (pioneer trace, diffuse rays only): closest_hit_coarse / miss. Own pipeline so its register
//     count is not set by the full material + light shading of closest_hit (NVIDIA allocates the pipeline maximum).
//   - the split trace / shade (repacked ray-list path): entry_shade_sharp / entry_shade_coarse / entry_shade_miss,
//     compute passes over the shade lists the split trace (ray_gen_split, rtgi_trace.hlsl) recorded.
// Sharp hits (specular rays) get full material + light shading, coarse hits (diffuse rays) only the PGI radiance
// cache with short range AO, misses the sky.

[[vk::push_constant]] RtgiShadePush rtgi_shade_push;

// == Shading core ===============================================================================================

// Everything the shading core reads. The trace and shade task heads name these attachments the same, so both fill
// it with RTGI_SHADE_CONTEXT(push.attach).
struct RtgiShadeContext
{
    RenderGlobalData* globals;
    MeshInstancesBufferHead* mesh_instances;
    daxa_u64 tlas;
    daxa_ImageViewId sky;
    daxa_ImageViewId sky_transmittance;
    daxa::Texture2DArrayIndex<daxa_u32vec4> light_mask_volume;
    daxa::Texture2DArrayIndex<daxa_f32vec4> pgi_color;
    daxa::Texture2DArrayIndex<daxa_f32vec2> pgi_visibility;
    daxa::Texture2DArrayIndex<daxa_f32vec4> pgi_info;
    daxa::RWTexture2DArrayIndex<daxa_u32> pgi_requests;
    VSMGlobals* vsm_globals;
    VSMPointLight* vsm_point_lights;
    VSMSpotLight* vsm_spot_lights;
    daxa::Texture2DId<daxa_f32> vsm_memory_block;
    daxa::RWTexture2DArrayId<daxa_u32>* vsm_point_spot_page_table;
};
#define RTGI_SHADE_CONTEXT(ATTACH) RtgiShadeContext( \
    ATTACH.globals, ATTACH.mesh_instances, ATTACH.tlas, ATTACH.sky, ATTACH.sky_transmittance, ATTACH.light_mask_volume, \
    ATTACH.pgi_color, ATTACH.pgi_visibility, ATTACH.pgi_info, ATTACH.pgi_requests, \
    ATTACH.vsm_globals, ATTACH.vsm_point_lights, ATTACH.vsm_spot_lights, ATTACH.vsm_memory_block, &(ATTACH.vsm_point_spot_page_table[0]))

// Shadow ray as an inline ray query with the alpha test done in the traversal loop (works in the closest hit and in
// compute alike).
func rtgi_shade_shadow_ray(RtgiShadeContext ctx, float3 position, float3 light_position, float3 flat_normal, float3 incoming_ray) -> bool
{
    RayDesc ray = {};
    ray.Direction = normalize(light_position - position);
    ray.Origin = rt_calc_ray_start(position, flat_normal, incoming_ray);
    ray.TMax = length(light_position - position) * 1.01f;
    ray.TMin = 0.0f;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(RaytracingAccelerationStructure::get(ctx.tlas), 0, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            rt_is_alpha_hit(
                ctx.globals,
                ctx.mesh_instances,
                ctx.globals.scene.meshes,
                ctx.globals.scene.materials,
                q.CandidateTriangleBarycentrics(),
                q.CandidatePrimitiveIndex(), q.CandidateInstanceID(), ray.Origin, ray.Direction, q.CandidateTriangleRayT()))
        {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

// Sun: ray traced shadow. Point / spot lights: coarse VSM lookup.
struct RtgiLightVisibilityTester : LightVisibilityTesterI
{
    RtgiShadeContext ctx;
    float sun_light(MaterialPointData material_point, float3 incoming_ray)
    {
        let sky = ctx.globals->sky_settings;
        let light_visible = rtgi_shade_shadow_ray(ctx, material_point.position, material_point.position + sky.sun_direction * 1000000, material_point.face_normal, incoming_ray);
        return light_visible ? 1.0f : 0.0f;
    }
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        GPUPointLight point_light = ctx.globals.scene.point_lights[light_index];
        const float3 to_light_dir = normalize(point_light.position - material_point.position);
        const float point_norm_dot = dot(material_point.position, to_light_dir);
        return get_vsm_point_shadow_coarse(
            ctx.globals,
            ctx.vsm_globals,
            ctx.vsm_memory_block.get(),
            ctx.vsm_point_spot_page_table,
            ctx.vsm_point_lights,
            material_point.normal,
            material_point.position,
            light_index,
            point_norm_dot);
    }
    float spot_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        return get_vsm_spot_shadow_coarse(
            ctx.globals,
            ctx.vsm_globals,
            ctx.vsm_memory_block.get(),
            ctx.vsm_point_spot_page_table,
            ctx.vsm_spot_lights,
            material_point.normal,
            material_point.position,
            light_index);
    }
}

func rtgi_shade_hit_triangle(RtgiShadeContext ctx, uint instance_id, uint primitive_index, float2 barycentrics) -> TriangleGeometry
{
    return rt_get_triangle_geo(
        barycentrics,
        instance_id,
        0u, // geometry index, unused (one geometry per BLAS)
        primitive_index,
        ctx.globals.scene.meshes,
        ctx.globals.scene.entity_to_meshgroup,
        ctx.globals.scene.mesh_groups,
        ctx.mesh_instances.instances
    );
}

// Full hit geometry (sharp shading: material evaluation needs normals, uvs, tangents).
func rtgi_shade_hit_geometry(RtgiShadeContext ctx, uint instance_id, uint primitive_index, float2 barycentrics, out TriangleGeometry tri_geo, out TriangleGeometryPoint tri_point)
{
    tri_geo = rtgi_shade_hit_triangle(ctx, instance_id, primitive_index, barycentrics);
    tri_point = rt_get_triangle_geo_point(
        tri_geo,
        ctx.globals.scene.meshes,
        ctx.globals.scene.entity_to_meshgroup,
        ctx.globals.scene.mesh_groups,
        ctx.globals.scene.entity_combined_transforms
    );
}

// Coarse hit geometry: position + face normal only.
func rtgi_shade_hit_position_face_normal(RtgiShadeContext ctx, uint instance_id, uint primitive_index, float2 barycentrics, out float3 world_position, out float3 face_normal)
{
    rt_get_triangle_position_face_normal(
        rtgi_shade_hit_triangle(ctx, instance_id, primitive_index, barycentrics),
        ctx.globals.scene.meshes,
        ctx.globals.scene.entity_combined_transforms,
        world_position,
        face_normal);
}

// Sharp hit: full material + light shading.
func rtgi_shade_sharp_hit(RtgiShadeContext ctx, TriangleGeometry tri_geo, TriangleGeometryPoint tri_point, float3 ray_direction, float t) -> float3
{
    MaterialPointData material_point = evaluate_material<SHADING_QUALITY_LOW>(
        ctx.globals,
        tri_geo,
        tri_point
    );

    const float indirect_ao_range = ctx.globals.rtgi_settings.shading_ao_range;
    const float pgi_enabled = ctx.globals.pgi_settings.enabled ? 1.0f : 0.0f;
    const float ambient_occlusion = (1.0f - max(0.0f, (indirect_ao_range - t)) / indirect_ao_range) * pgi_enabled;

    return shade_material<SHADING_QUALITY_LOW>(
        ctx.globals,
        ctx.sky_transmittance,
        ctx.sky,
        material_point,
        ctx.globals.view_camera.position,
        ray_direction,
        RtgiLightVisibilityTester(ctx),
        ctx.light_mask_volume.get(),
        ctx.pgi_color.get(),
        ctx.pgi_visibility.get(),
        ctx.pgi_info.get(),
        ctx.pgi_requests.get_formatted(),
        PGI_REQUEST_MODE_INDIRECT,
        ambient_occlusion
    ).rgb;
}

// Coarse hit: PGI radiance with short range AO.
func rtgi_shade_coarse_hit(RtgiShadeContext ctx, float3 position, float3 face_normal, float t) -> float3
{
    const float pgi_enabled = ctx.globals.pgi_settings.enabled ? 1.0f : 0.0f;
    // PGI can not be trusted at short hit ranges as its very leaky for fine details.
    // Apply strong range based AO to the PGI radiance.
    const float indirect_ao_range = ctx.globals.rtgi_settings.shading_ao_range;
    const float relative_ao_t = min(indirect_ao_range, t) / indirect_ao_range;
    const float ambient_occlusion = square(relative_ao_t) * pgi_enabled;

    PGISampleInfo info = PGISampleInfoNearestSurfaceRadiance();
    return pgi_sample_probe_volume(
        ctx.globals, &ctx.globals.pgi_settings, info,
        position, ctx.globals.view_camera.position, face_normal, face_normal,
        ctx.pgi_color.get(),
        ctx.pgi_visibility.get(),
        ctx.pgi_info.get(),
        ctx.pgi_requests.get()
    ) * ambient_occlusion;
}

func rtgi_shade_miss(RtgiShadeContext ctx, float3 ray_direction) -> float3
{
    return shade_sky(ctx.globals, ctx.sky_transmittance, ctx.sky, ray_direction);
}

// == Hit shaders of the shading ray tracing pipelines ("Trace RTGI", "Trace RTGI Pioneer") ======================

// Coarse hit of a ray tracing pipeline: the hit position comes from the ray, only the face normal is fetched.
func rtgi_closest_hit_coarse(RtgiShadeContext ctx, float2 barycentrics) -> float3
{
    const float t = RayTCurrent();
    float3 unused_position;
    float3 face_normal;
    rtgi_shade_hit_position_face_normal(ctx, InstanceID(), PrimitiveIndex(), barycentrics, unused_position, face_normal);
    return rtgi_shade_coarse_hit(ctx, WorldRayOrigin() + WorldRayDirection() * t, face_normal, t);
}

[shader("closesthit")]
void closest_hit(inout RtgiRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let ctx = RTGI_SHADE_CONTEXT(rtgi_trace_push.attach);
    if (payload.specular)
    {
        TriangleGeometry tri_geo;
        TriangleGeometryPoint tri_point;
        rtgi_shade_hit_geometry(ctx, InstanceID(), PrimitiveIndex(), attr.barycentrics, tri_geo, tri_point);
        payload.color = rtgi_shade_sharp_hit(ctx, tri_geo, tri_point, WorldRayDirection(), RayTCurrent());
    }
    else
    {
        payload.color = rtgi_closest_hit_coarse(ctx, attr.barycentrics);
    }
    payload.t = RayTCurrent();
}

[shader("closesthit")]
void closest_hit_coarse(inout RtgiRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.color = rtgi_closest_hit_coarse(RTGI_SHADE_CONTEXT(rtgi_trace_push.attach), attr.barycentrics);
    payload.t = RayTCurrent();
}

[shader("miss")]
void miss(inout RtgiRayPayload payload)
{
    payload.color = rtgi_shade_miss(RTGI_SHADE_CONTEXT(rtgi_trace_push.attach), WorldRayDirection());
    payload.t = RTGI_RAY_MISS_T;
}

// == Split trace / shade compute passes (RtgiShadeH) ============================================================

// Ray-list index of entry (group_id, group_index) of shade list shade_class. false = past the end of the list.
// Thread 0 also writes the list's ray count to the readback (statistics; a pass with an empty list dispatches nothing,
// the readback is cleared every frame -> 0).
func rtgi_shade_list_ray_index(uint3 group_id, uint group_index, uint shade_class, out uint ray_index) -> bool
{
    let push = rtgi_shade_push;
    ray_index = 0u;
    RtgiShadeCounters* counters = push.attach.shade_counters;
    const uint list_index = (group_id.y * RTGI_SHADE_DISPATCH_X_MAX + group_id.x) * RTGI_SHADE_HITS_X + group_index;
    uint count;
    switch (shade_class)
    {
        case RTGI_SHADE_SHARP: count = counters->sharp_count; break;
        case RTGI_SHADE_MISS:  count = counters->miss_count; break;
        default:               count = counters->coarse_count; break;
    }
    if (list_index == 0u)
    {
        switch (shade_class)
        {
            case RTGI_SHADE_SHARP: push.attach.globals.readback.rtgi_shade_sharp_rays = count; break;
            case RTGI_SHADE_MISS:  push.attach.globals.readback.rtgi_shade_miss_rays = count; break;
            default:               push.attach.globals.readback.rtgi_shade_coarse_rays = count; break;
        }
    }
    if (list_index >= count) { return false; }
    const uint list_capacity = rtgi_ray_list_capacity(push.attach.globals.settings.render_target_size >> 1u);
    ray_index = push.attach.shade_list[rtgi_shade_list_slot(shade_class, list_index, list_capacity)];
    return true;
}

// The trace stored the ray's sampling weight in .radiance.x.
func rtgi_shade_write_result(uint ray_index, float weight, float3 color)
{
    rtgi_shade_push.attach.ray_result[ray_index].radiance = color * weight * RTGI_RADIANCE_SCALE;
}

[shader("compute")]
[numthreads(RTGI_SHADE_HITS_X, 1, 1)]
func entry_shade_sharp(uint3 group_id : SV_GroupID, uint group_index : SV_GroupIndex)
{
    let push = rtgi_shade_push;
    uint ray_index;
    if (!rtgi_shade_list_ray_index(group_id, group_index, RTGI_SHADE_SHARP, ray_index)) { return; }

    let ctx = RTGI_SHADE_CONTEXT(push.attach);
    const RtgiRayResult result = push.attach.ray_result[ray_index];
    const RtgiRayHit hit = push.attach.ray_hits[ray_index];

    TriangleGeometry tri_geo;
    TriangleGeometryPoint tri_point;
    rtgi_shade_hit_geometry(ctx, hit.instance_id, hit.primitive_index, hit.barycentrics, tri_geo, tri_point);

    const float3 color = rtgi_shade_sharp_hit(ctx, tri_geo, tri_point, uncompress_normal_octahedral_32(result.packed_dir), result.t);
    rtgi_shade_write_result(ray_index, result.radiance.x, color);
}

[shader("compute")]
[numthreads(RTGI_SHADE_HITS_X, 1, 1)]
func entry_shade_coarse(uint3 group_id : SV_GroupID, uint group_index : SV_GroupIndex)
{
    let push = rtgi_shade_push;
    uint ray_index;
    if (!rtgi_shade_list_ray_index(group_id, group_index, RTGI_SHADE_COARSE, ray_index)) { return; }

    let ctx = RTGI_SHADE_CONTEXT(push.attach);
    const RtgiRayResult result = push.attach.ray_result[ray_index];
    const RtgiRayHit hit = push.attach.ray_hits[ray_index];

    float3 world_position;
    float3 face_normal;
    rtgi_shade_hit_position_face_normal(ctx, hit.instance_id, hit.primitive_index, hit.barycentrics, world_position, face_normal);

    const float3 color = rtgi_shade_coarse_hit(ctx, world_position, face_normal, result.t);
    rtgi_shade_write_result(ray_index, result.radiance.x, color);
}

[shader("compute")]
[numthreads(RTGI_SHADE_HITS_X, 1, 1)]
func entry_shade_miss(uint3 group_id : SV_GroupID, uint group_index : SV_GroupIndex)
{
    let push = rtgi_shade_push;
    uint ray_index;
    if (!rtgi_shade_list_ray_index(group_id, group_index, RTGI_SHADE_MISS, ray_index)) { return; }

    const RtgiRayResult result = push.attach.ray_result[ray_index];
    const float3 color = rtgi_shade_miss(RTGI_SHADE_CONTEXT(push.attach), uncompress_normal_octahedral_32(result.packed_dir));
    rtgi_shade_write_result(ray_index, result.radiance.x, color);
}
