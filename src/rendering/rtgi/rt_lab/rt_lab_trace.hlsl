#pragma once

// MUST come before the first daxa include (daxa.inl is pragma once): enables RaytracingAccelerationStructure::get.
#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rt_lab_trace.inl"

#include "shader_lib/raytracing.hlsl"
#include "../rtgi_shade.hlsl" // RTGI shading core: RTGI_SHADE_CONTEXT, rtgi_shade_coarse_hit_pgi, rtgi_coarse_hit_ao, rtgi_shade_miss
#include "rt_lab_shared.hlsl"

[[vk::push_constant]] RtLabTracePush rt_lab_trace_push;


// Hit shading of the lab rays. 1 = fine: full material + lights at the hit like the RTGI sharp path (evaluate_material +
// shade_material, sun / light shadow rays), PGI only as the hit's indirect term, scaled by the lab's short range AO on hit.
// 0 = coarse: PGI radiance at the hit with the lab's short range AO (the RTGI coarse path).
#define RT_LAB_FINE_HIT_SHADING 1

// Fine hit shading (RT_LAB_FINE_HIT_SHADING): rtgi_shade_sharp_hit with the lab's hit AO (hit_ao_*) on the PGI indirect term.
func rt_lab_shade_fine_hit(RtgiShadeContext ctx, RtLabSettings settings, uint instance_id, uint primitive_index, float2 barycentrics,
    float3 ray_direction, float t) -> float3
{
    TriangleGeometry tri_geo;
    TriangleGeometryPoint tri_point;
    rtgi_shade_hit_geometry(ctx, instance_id, primitive_index, barycentrics, tri_geo, tri_point);
    MaterialPointData material_point = evaluate_material<SHADING_QUALITY_LOW>(ctx.globals, tri_geo, tri_point);
    const float pgi_enabled = ctx.globals.pgi_settings.enabled ? 1.0f : 0.0f;
    const float ambient_occlusion = (settings.hit_ao_enabled ? rtgi_coarse_hit_ao(t, max(settings.hit_ao_range, 1e-4f)) : 1.0f) * pgi_enabled;
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

// Traces one lab ray (closest hit, alpha test in the traversal loop: masked geometry is non-opaque) and shades the hit
// (RT_LAB_FINE_HIT_SHADING), sky on miss. Shared by the trace and the temporal gradient re-trace, which must shade
// identically.
// hit_seed (rt_lab_hit_seed): rand() is seeded with it right before the hit is shaded, so the trace and the re-trace
// draw the same random numbers in the shading.
func rt_lab_trace_radiance(RtgiShadeContext ctx, RtLabSettings settings, RayDesc ray, uint hit_seed) -> float3
{
    RayQuery<RAY_FLAG_NONE> q;
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
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        rand_seed(hit_seed);
        const float t = q.CommittedRayT();
#if RT_LAB_FINE_HIT_SHADING
        return rt_lab_shade_fine_hit(ctx, settings, q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), ray.Direction, t);
#else
        float3 hit_position;
        float3 hit_face_normal;
        rtgi_shade_hit_position_face_normal(ctx, q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), hit_position, hit_face_normal);
        float3 color = rtgi_shade_coarse_hit_pgi(ctx, ray.Origin + ray.Direction * t, hit_face_normal);
        if (settings.hit_ao_enabled)
        {
            color *= rtgi_coarse_hit_ao(t, max(settings.hit_ao_range, 1e-4f));
        }
        return color;
#endif
    }
    return rtgi_shade_miss(ctx, ray.Direction);
}

[shader("compute")]
[numthreads(RT_LAB_TRACE_THREADS, 1, 1)]
func entry_rt_lab_trace(uint ray_index : SV_DispatchThreadID)
{
    let push = rt_lab_trace_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (ray_index >= min(push.attach.ray_counters->ray_count, rt_lab_ray_budget(push.size, settings, push.ray_capacity)))
    {
        return;
    }
    const uint packed_pixel = push.attach.ray_list[ray_index];
    const uint2 dtid = uint2(packed_pixel & 0xFFFFu, packed_pixel >> 16u);
    // Index of this ray among its pixel's rays: seeds its direction (ray 0 = the old 1 ray per pixel seed).
    const uint pixel_ray = ray_index - push.attach.ray_alloc.get()[dtid].x;
    const float depth = push.attach.lab_depth.get()[dtid];
    const float3 face_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);

    // Cosine distributed direction around the face normal.
    const uint ray_seed = rt_lab_ray_seed(settings, push.attach.globals.trunk_flt_frame_index, dtid, pixel_ray);
    rand_seed(ray_seed);
    const float3 tangent = normalize(cross(face_normal, float3(0, 0, 1) + 0.0001f));
    const float3x3 tbn = transpose(float3x3(tangent, cross(tangent, face_normal), face_normal));
    // Quantized like the stored direction (RtLabRay, gradient record): the temporal gradient re-trace then fires the
    // bit-identical ray. With the exact direction the re-traced ray differs slightly and its luminance change at edges /
    // shadow boundaries showed up as a gradient in static scenes.
    const float3 direction = uncompress_normal_octahedral_32(compress_normal_octahedral_32(mul(tbn, rand_cosine_sample_hemi())));

    // RTGI diffuse ray start (rt_lab_ray_launch).
    float3 world_pos;
    float t_min;
    RayDesc ray = {};
    ray.Origin    = rt_lab_ray_launch(push.attach.globals.view_camera, dtid, push.size, depth, face_normal, world_pos, t_min);
    ray.Direction = direction;
    ray.TMin      = t_min;
    ray.TMax      = RT_LAB_RAY_T_MAX;

    let ctx = RTGI_SHADE_CONTEXT(push.attach);
    const float3 color = rt_lab_trace_radiance(ctx, settings, ray, rt_lab_hit_seed(ray_seed));

    push.attach.rays[ray_index] = RtLabRay(color * RTGI_RADIANCE_SCALE, compress_normal_octahedral_32(direction));
}

[[vk::push_constant]] RtLabGradientTracePush rt_lab_gradient_trace_push;

// Temporal gradient re-trace (rt_lab_trace.inl), one thread per last frame gradient record.
[shader("compute")]
[numthreads(RT_LAB_RAYS_X, RT_LAB_RAYS_Y, 1)]
func entry_rt_lab_gradient_trace(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_gradient_trace_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const uint4 data = push.attach.gradient_record_data.get()[dtid];
    if ((data.w & RT_LAB_GRADIENT_RECORD_MAGIC_MASK) != RT_LAB_GRADIENT_RECORD_MAGIC)
    {
        return; // no ray in this cell last frame (or an uninitialized record)
    }
    const float4 origin = push.attach.gradient_record_origin.get()[dtid];
    const float4 surface = push.attach.gradient_record_surface.get()[dtid];
    if (any(!isfinite(origin)) || any(!isfinite(surface)))
    {
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;

    // Nearest current lab pixel of the recorded surface point.
    const float4 clip = mul(camera.view_proj, float4(surface.xyz, 1.0f));
    if (clip.w <= 0.0f)
    {
        return;
    }
    const float2 uv = (clip.xy / clip.w) * 0.5f + 0.5f;
    const int2 pixel = int2(floor(uv * float2(push.lab_size)));
    if (any(pixel < 0) || any(pixel >= int2(push.lab_size)))
    {
        return;
    }
    // Same surface test as the temporal reprojection (face normal + plane distance).
    const bool half_res = push.lab_size.x != push.attach.globals.settings.render_target_size.x;
    uint packed_normal;
    const float depth = rt_lab_load_geometry(push.attach.view_cam_depth.get(), push.attach.view_cam_face_normals.get(),
        push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get(), camera, half_res, uint2(pixel), packed_normal);
    if (depth == 0.0f)
    {
        return;
    }
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(pixel), rcp(float2(push.lab_size))), depth);
    const float3 normal = uncompress_normal_octahedral_32(packed_normal);
    const float3 recorded_normal = uncompress_normal_octahedral_32(data.y);
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.lab_size.y), depth));
    if (dot(normal, recorded_normal) < settings.disocclusion_normal_threshold ||
        rt_lab_plane_weight(surface.xyz, recorded_normal, position, normal, rcp_pixel_width_ws, settings.disocclusion_plane_tolerance) <= 0.0f)
    {
        return;
    }

    // The identical ray, shaded the same way as the trace.
    RayDesc ray = {};
    ray.Origin    = origin.xyz;
    ray.Direction = uncompress_normal_octahedral_32(data.x);
    ray.TMin      = origin.w;
    ray.TMax      = RT_LAB_RAY_T_MAX;
    let ctx = RTGI_SHADE_CONTEXT(push.attach);
    // The recorded hit seed: the shading draws the same random numbers as last frame's trace.
    const float new_luminance = rt_lab_luminance(rt_lab_trace_radiance(ctx, settings, ray, data.w & RT_LAB_HIT_SEED_MASK) * RTGI_RADIANCE_SCALE);
    const float old_luminance = surface.w;
    const float exposure = push.attach.globals.exposure;
    // RIS weight of the selection (rt_lab_gradient_select, 1 = uniform): the cell's mean change, not the picked ray's.
    const float ris_weight = asfloat(data.z);
    const float delta = rt_lab_exposed(new_luminance - old_luminance, exposure) * ris_weight;
    const float maximum = rt_lab_exposed(max(new_luminance, old_luminance), exposure) * ris_weight;
    if (!isfinite(delta) || !(maximum > 0.0f))
    {
        return;
    }
    // f16 packed: one 32 bit store, so colliding cells keep one whole sample.
    const uint packed = (f32tof16(clamp(delta, -65000.0f, 65000.0f)) & 0xFFFFu) | (f32tof16(min(maximum, 65000.0f)) << 16u);
    push.attach.gradient_samples.get()[uint2(pixel) / RT_LAB_GRADIENT_CELL] = packed;
}
