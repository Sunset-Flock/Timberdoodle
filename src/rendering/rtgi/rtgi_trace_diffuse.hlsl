#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtgi_trace_diffuse.inl"
#include "rtgi_trace_diffuse_shared.hlsl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/pgi.hlsl"

#include "rtgi_shared.hlsl"
#include "shader_lib/debug.glsl"

#define GOLDEN_RATIO 1.6181
#define PI 3.1415926535897932384626433832795

#define STBN_ENABLED 0
#define STBN_WDITH 128
#define STBN_SIZE uint3(STBN_WDITH,STBN_WDITH,32)
#define STBN_GRID_SIZE uint3(STBN_WDITH,STBN_WDITH,32)

float2 rand_stbn2d(Texture2DArray<float4> stbn2d_image, uint2 pixel, int frame)
{
    const uint z_wrap = 0;//frame / STBN_SIZE.z;
    const uint2 xy_wrap = pixel / STBN_SIZE.xy;
    const uint z = frame % STBN_SIZE.z;// + xy_wrap.x + xy_wrap.y * 17;
    pixel = (pixel + uint2(GOLDEN_RATIO * float2(STBN_SIZE.xy * z_wrap))) % STBN_SIZE.xy;
    return stbn2d_image[uint3(pixel,z)].xy;
}

float3 rand_stbnCosDir(Texture2DArray<float4> stbn2d_image, uint2 pixel, int frame)
{
    pixel = pixel % STBN_GRID_SIZE.xy;
    const uint z_wrap = frame / STBN_SIZE.z;
    const uint z = frame % STBN_SIZE.z;
    pixel = (pixel + uint2(GOLDEN_RATIO * float2(STBN_SIZE.xy * z_wrap))) % STBN_SIZE.xy;
    return stbn2d_image[uint3(pixel,z)].xyz * 2.0f - 1.0f;
}

float2 rand_concentric_sample_disc_stbn(uint2 pixel_frame)
{
    let push = rtgi_trace_diffuse_push;
    float2 rr = rand_stbn2d(Texture2DArray<float4>::get(push.attach.globals.stbn2d), pixel_frame.xy, push.attach.globals.frame_index);
    rr = abs(rr);
    float r = rr.x;
    float theta = rr.y * 2 * PI;
    return float2(cos(theta), sin(theta)) * r;
}

float3 rand_cosine_sample_hemi_stbn(uint2 pixel_frame)
{
    float2 d = rand_concentric_sample_disc_stbn(pixel_frame);
    float z = sqrt(max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return float3(d.x, d.y, z);
}

interface TraceRayInterface
{
    static void trace_and_shade(RayDesc ray, uint flags, inout RayPayload payload);
};

struct RTPipelineTraceRay : TraceRayInterface
{
    static void trace_and_shade(RayDesc ray, uint flags, inout RayPayload payload)
    {
        TraceRay(RaytracingAccelerationStructure::get(rtgi_trace_diffuse_push.attach.tlas), flags, ~0, 0, 0, 0, ray, payload);
    }
}

__generic<TRACE_FUNCTOR : TraceRayInterface>
void shade_ray_gen(uint2 dtid)
{
    let clk_start = clockARB();
    let push = rtgi_trace_diffuse_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    const float depth = push.attach.view_cam_half_res_depth.get()[dtid];
    const float2 pixel_index = float2(dtid.xy * 2u) + 0.5f;
    const CameraInfo camera = push.attach.globals.view_camera;
    const float3 world_position = pixel_index_to_world_space(camera, pixel_index, depth);
    const float3 face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[dtid].r);
    const float3 primary_ray = normalize(world_position - push.attach.globals.view_camera.position);
            
    if (push.debug_primary_trace)
    {
        RayPayload payload = {};
        
        RayDesc ray = {};
        ray.Origin = camera.position;
        ray.TMax = 1000000000.0f;
        ray.TMin = 0.0f;
        ray.Direction = primary_ray;

        payload.color = float3(0,0,0);
        TRACE_FUNCTOR::trace_and_shade(ray, 0, payload);

        float4 value = float4(payload.color, 1.0f);
        push.attach.diffuse_raw.get()[dtid.xy] = float4(payload.color,payload.t);
        return;
    }

    float acc_ray_shortness = 0.0f;        // mean ray shortness [0,1] over the rays; stored in .a
    float3 mean_perceptual_rgb = float3(0, 0, 0); // geometric mean (mean log rgb) over the rays; stored in .rgb

    const uint prime_shift0 = 257;   // just over typical period of frame time roughly (32 - 255 accum frames)
    const uint prime_shift1 = 9629;  // just over typical period of frame width x (480 - 8192)
    const uint prime_shift2 = 10069; // just over typical period of frame height y (480 - 8192)
    const uint frame_seed = rtgi_settings.animate_noise ? push.attach.globals.frame_index * prime_shift0 : 0u;
    const uint thread_seed =
        frame_seed +
        dtid.x * prime_shift1 +
        dtid.y * prime_shift2;

    rand_seed(thread_seed);
    float2 rr_stbn = rand_stbn2d(Texture2DArray<float4>::get(push.attach.globals.stbn2d), dtid.xy, push.attach.globals.frame_index);
    float2 rr = float2(rand(), rand());

    const float2 half_res_inv_render_target_size = push.attach.globals.settings.render_target_size_inv * 2.0f;
    const float  ws_px_size = depth > 0.0f ? calc_pixel_width_ws(half_res_inv_render_target_size, camera.near_plane, depth) : 0.0f;

    // --- Determine this pixel's ray count (0 for sky) ---
    uint samples = 0u;
    if (depth > 0.0f)
    {
        const uint total_extra_ray_demands = push.attach.ray_demand->total_extra_rays;
        const uint max_extra_rays = (push.attach.globals.settings.render_target_size.x * push.attach.globals.settings.render_target_size.y) / 4;
        const float relative_allowed_rays = min(1.0f, float(max_extra_rays) / (float(total_extra_ray_demands) + 0.0001f));
        const float reproj_sample_count = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[dtid.xy]);
        const uint desired_extra_samples = calc_desired_extra_rays(reproj_sample_count, rtgi_settings.fast_convergence_samples);
        const uint allowed_extra_samples = uint(float(desired_extra_samples) * relative_allowed_rays);
        // Redistribution off: trace a fixed max(floor(ray_budget), 1) rays per pixel. On: adaptive base + extra.
        samples = rtgi_settings.use_ray_redistribution
            ? (1u + allowed_extra_samples)
            : calc_fixed_rays_per_pixel(rtgi_settings.ray_percentage);
    }

    // --- Wave-coalesced ray-list allocation: exclusive prefix sum of the ray counts within the wave, then
    // ONE atomic per wave to reserve that wave's contiguous slice of the global ray list. Same ray_list /
    // ray_result / pixel_ray_alloc structures the repacked (distribute-rays) path uses. ---
    const uint lane_prefix = WavePrefixSum(samples);
    const uint wave_total  = WaveActiveSum(samples);
    uint wave_base = 0u;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(push.attach.ray_demand->ray_list_count, wave_total, wave_base);
    }
    wave_base = WaveReadLaneFirst(wave_base);
    const uint my_offset = wave_base + lane_prefix;

    // Hard capacity clamp so a rounding/overflow edge can never write past the ray_result buffer.
    const uint2 half_res = push.attach.globals.settings.render_target_size >> 1u;
    const uint  ray_list_capacity = half_res.x * half_res.y * RTGI_RAY_LIST_CAPACITY_MUL;
    uint write_count = samples;
    if (my_offset >= ray_list_capacity)               write_count = 0u;
    else if (my_offset + samples > ray_list_capacity) write_count = ray_list_capacity - my_offset;

    if (write_count > 0u)
    {
        const float  inv_samples   = rcp(float(write_count));
        const float3 world_tangent = normalize(cross(face_normal, float3(0, 0, 1) + 0.0001f));
        const float3x3 tbn         = transpose(float3x3(world_tangent, cross(world_tangent, face_normal), face_normal));
        const float3 sample_pos    = rt_calc_ray_start(world_position, face_normal, primary_ray);

        for (uint i = 0u; i < write_count; ++i)
        {
            float3 importance_rand_hemi_sample;
            if (STBN_ENABLED)
            {
                const uint stbn_frame_seed = rtgi_settings.animate_noise ? push.attach.globals.frame_index : 0u;
                rand_seed(stbn_frame_seed + i * prime_shift1);
                importance_rand_hemi_sample = rand_stbnCosDir(Texture2DArray<float4>::get(push.attach.globals.stbnCosDir), pixel_index, (rtgi_settings.animate_noise ? push.attach.globals.frame_index : 0) + rand());
            }
            else
            {
                importance_rand_hemi_sample = rand_cosine_sample_hemi();
            }

            RayPayload payload = {};
            payload.dtid = dtid;

            #if RTGI_USE_PGI_RADIANCE_ON_MISS
            float pgi_cascade = pgi_select_cascade_smooth_spherical(push.attach.globals.pgi_settings, sample_pos - push.attach.globals.view_camera.position);
            float t_max = float(1u << uint(ceil(pgi_cascade))) * push.attach.globals.pgi_settings.cascades[0].probe_spacing.x * RTGI_USE_PGI_RADIANCE_ON_MISS_TMAX_SCALE;
            #else
            float t_max = 100000000000.0f;
            #endif

            RayDesc ray = {};
            ray.Origin    = sample_pos - primary_ray * ws_px_size;
            ray.TMax      = t_max;
            ray.TMin      = ws_px_size * 0.5f;
            const float3 sample_dir = mul(tbn, importance_rand_hemi_sample);
            ray.Direction = sample_dir;
            const uint flags = {};
            TRACE_FUNCTOR::trace_and_shade(ray, flags, payload);

            // Write this ray's result into the shared ray list (same layout the blend pass produced), so
            // the pre-filter re-blends (and per-ray firefly-clamps) it identically to the repacked path.
            const float3 ray_rgb = payload.color * VALUE_MULTIPLIER;
            push.attach.ray_result[my_offset + i] = RtgiRayResult(ray_rgb, payload.t, compress_normal_octahedral_32(sample_dir));

            mean_perceptual_rgb += linear_to_perceptual(ray_rgb, push.attach.globals.inv_exposure) * inv_samples;
            acc_ray_shortness   += calc_ray_shortness(payload.t, ws_px_size, rtgi_settings.max_visibility_pixel_range) * inv_samples;
        }
    }

    // Same outputs the distribute pass produces: per-pixel ray-list offset, ray count, and the log-rgb /
    // shortness the pre-filter reads. diffuse / diffuse2 are NOT produced (the pre-filter re-blends rays).
    push.attach.pixel_ray_alloc.get()[dtid.xy] = my_offset;
    push.attach.ray_count_image.get()[dtid.xy] = write_count;
    push.attach.perceptual_rgb_shortness.get()[dtid.xy] = float4(mean_perceptual_rgb, acc_ray_shortness);

    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_TRACE_CLOCKS)
    {
        let clk_end = clockARB();
        const uint clocks = uint(clk_end - clk_start);
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(clocks * 0.0001f * push.attach.globals.settings.debug_visualization_scale), 1.0f + push.attach.globals.settings.debug_visualization_blend), 2);
    }
}

// Ray-list body: traces one ray from the flat ray list built by the allocate pass. Dispatched as
// (128, 1, ceil(max_rays/128)); the flat ray index is z*128 + x.
void ray_gen_from_list_body()
{
    let push = rtgi_trace_diffuse_push;
    let clk_start = clockARB();
    const uint ray_index = DispatchRaysIndex().z * 128u + DispatchRaysIndex().x;

    if (ray_index >= push.attach.ray_demand->ray_list_count)
        return;

    const RtgiRayEntry entry   = push.attach.ray_list[ray_index];
    const uint2 pixel_xy       = uint2(entry.packed_xy & 0xFFFFu, entry.packed_xy >> 16u);
    // Cap with an UNSIGNED literal: `min(32, ...)` promotes the compare to signed, so a garbage-large
    // sample_index (e.g. from a ray-list slot reserved but never written under the capacity clamp) turns
    // negative, survives the min, and wraps back to a massive uint — driving the skip loop below for
    // billions of iterations and hanging the GPU. 32u keeps the compare unsigned so the cap always holds.
    const uint  sample_index   = min(32u, entry.sample_index);

    const float depth = push.attach.view_cam_half_res_depth.get()[pixel_xy];
    if (depth == 0.0f)
    {
        push.attach.ray_result[ray_index] = RtgiRayResult(float3(0.0f, 0.0f, 0.0f), 0.0f, 0u);
        return;
    }

    let rtgi_settings = push.attach.globals.rtgi_settings;
    const CameraInfo camera   = push.attach.globals.view_camera;
    const float2 pixel_index  = float2(pixel_xy * 2u) + 0.5f;
    const float3 world_pos    = pixel_index_to_world_space(camera, pixel_index, depth);
    const float3 face_normal  = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[pixel_xy].r);
    const float3 primary_ray  = normalize(world_pos - camera.position);
    const float2 half_res_inv_render_target_size = push.attach.globals.settings.render_target_size_inv * 2.0f;
    const float  ws_px_size   = calc_pixel_width_ws(half_res_inv_render_target_size, camera.near_plane, depth);

    const uint prime_shift0 = 257u;
    const uint prime_shift1 = 9629u;
    const uint prime_shift2 = 10069u;
    const uint prime_shift3 = 6151u;
    const uint frame_seed   = rtgi_settings.animate_noise ? push.attach.globals.frame_index * prime_shift0 : 0u;
    // Fold the reprojected history length into the seed. frame_index alone can alias across frames, so a
    // pixel that shot N rays one frame could re-draw near-identical directions the next. The history count
    // advances by the pixel's rays-shot each frame (fastest exactly when many rays/frame make repeats most
    // likely), giving an extra decorrelating dimension. Gated by animate_noise so frozen-noise stays frozen.
    const float history_count = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[pixel_xy]);
    const uint history_seed   = rtgi_settings.animate_noise ? uint(max(history_count, 0.0f)) * prime_shift3 : 0u;
    // Seed ONCE per pixel+frame (NOT per ray). Each ray is a separate shader invocation, so we advance
    // the per-pixel RNG sequence to this ray's slot instead of folding sample_index into the seed:
    // re-seeding per ray with base + sample_index*prime does NOT decorrelate (a single PCG step barely
    // mixes nearby seeds), which made a pixel's N rays near-duplicates. rand_cosine_sample_hemi draws 2
    // values, so skip sample_index*2 to land on a fresh, decorrelated pair — mirroring how the classic
    // per-pixel trace draws sequentially across its sample loop.
    rand_seed(frame_seed + history_seed + pixel_xy.x * prime_shift1 + pixel_xy.y * prime_shift2);
    [loop] for (uint skip = 0u; skip < sample_index * 2u; ++skip) { rand(); }

    const float3 world_tangent = normalize(cross(face_normal, float3(0, 0, 1) + 0.0001f));
    const float3x3 tbn         = transpose(float3x3(world_tangent, cross(world_tangent, face_normal), face_normal));
    const float3 sample_dir    = mul(tbn, rand_cosine_sample_hemi());

    RayPayload payload = {};
    payload.dtid = pixel_xy;

    // Match the classic per-pixel trace's ray setup exactly (see shade_ray_gen): back-offset the
    // origin by one pixel width and use an effectively-unbounded TMax. (shading_ao_range is NOT a ray length
    // clamp here — the classic path ignores it for TMax, so clamping to it made every ray miss.)
    const float3 sample_pos = rt_calc_ray_start(world_pos, face_normal, primary_ray);
    RayDesc ray = {};
    ray.Origin    = sample_pos - primary_ray * ws_px_size;
    ray.Direction = sample_dir;
    ray.TMin      = ws_px_size * 0.5f;
    ray.TMax      = 100000000000.0f;

    const uint flags = {};
    RTPipelineTraceRay::trace_and_shade(ray, flags, payload);

    // Store the raw hit distance; the blend pass converts it to bounded shortness [0,1] per ray and
    // averages over the pixel's rays into the ray-length texture for a stable denoiser guide.
    push.attach.ray_result[ray_index] = RtgiRayResult(payload.color * VALUE_MULTIPLIER, payload.t, compress_normal_octahedral_32(sample_dir));

    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_TRACE_CLOCKS)
    {
        let clk_end = clockARB();
        const uint clocks = uint(clk_end - clk_start);
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, pixel_xy, float4(Heatmap(clocks * 0.0001f * push.attach.globals.settings.debug_visualization_scale), 1.0f + push.attach.globals.settings.debug_visualization_blend), 2);
    }
}

// Single raygen entry point. Switches on the setting so both trace paths can share one pipeline
// (avoids daxa's single-handle raygen SBT limitation). The task graph only ever dispatches one of
// the two paths per frame — with the matching dispatch shape — based on the same setting.
[shader("raygeneration")]
void ray_gen()
{
    if (rtgi_trace_diffuse_push.attach.globals.rtgi_settings.use_repacked_ray_dispatch)
    {
        ray_gen_from_list_body();
    }
    else
    {
        shade_ray_gen<RTPipelineTraceRay>(DispatchRaysIndex().xy);
    }
}

[shader("anyhit")]
void any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = rtgi_trace_diffuse_push;

    if (!rt_is_alpha_hit(
        push.attach.globals,
        push.attach.mesh_instances,
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.materials,
        attr.barycentrics,
        PrimitiveIndex(), InstanceID(), WorldRayOrigin(), WorldRayDirection(), RayTCurrent()))
    {
        IgnoreHit();
    }
}