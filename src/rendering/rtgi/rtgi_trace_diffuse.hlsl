#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtgi_trace_diffuse.inl"
#include "rtgi_trace_diffuse_shared.hlsl"
#include "rtgi_guide_resample.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/pgi.hlsl"

#include "rtgi_shared.hlsl"
#include "rtgi_guided_sampling.hlsl"
#include "rtgi_specular_sampling.hlsl"
#include "shader_lib/debug.glsl"

#define GOLDEN_RATIO 1.6181
#define PI 3.1415926535897932384626433832795

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
    float2 rr = rand_stbn2d(Texture2DArray<float4>::get(push.attach.globals.stbn2d), pixel_frame.xy, push.attach.globals.trunk_flt_frame_index);
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

void rtgi_trace_and_shade(RayDesc ray, uint flags, inout RayPayload payload)
{
    TraceRay(RaytracingAccelerationStructure::get(rtgi_trace_diffuse_push.attach.tlas), flags, ~0, 0, 0, 0, ray, payload);
}

// Traces ONE specular ray for a half-res pixel. Uses the caller's rand() state (seed it before calling).
// Returns the ray-list result (radiance already weighted + RTGI_RADIANCE_SCALE scaled).
func rtgi_trace_specular_ray(uint2 pixel_xy, float3 world_pos, float3 face_normal, float3 primary_ray, float ws_px_size, float4 guide_sh_y) -> RtgiRayResult
{
    let push = rtgi_trace_diffuse_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    const float3 view_dir = -primary_ray;
    const uint normal_roughness = push.attach.view_cam_half_res_normal_roughness.get()[pixel_xy];
    const float3 detail_normal = unpack_normal_roughness_normal(normal_roughness);
    const float3 shading_normal = rtgi_specular_shading_normal(detail_normal, view_dir);
    const float roughness = unpack_normal_roughness_roughness(normal_roughness);

    float weight = 1.0f;
    const bool guide_valid = rtgi_settings.pioneer_guiding_enabled != 0 && dot(guide_sh_y.xyz, guide_sh_y.xyz) > 1e-12f;
    const float3 dir = rtgi_sample_specular_dir(
        shading_normal, face_normal, view_dir, roughness,
        guide_sh_y, guide_valid, rtgi_settings.specular_guide_concentration,
        rtgi_settings.specular_guide_mix, weight);

    if (weight <= 0.0f)
    {
        return RtgiRayResult(float3(0, 0, 0), 0.0f, compress_normal_octahedral_32(dir));
    }

    RayPayload payload = {};
    payload.dtid = pixel_xy;
    payload.specular = true;

    const float3 sample_pos = rt_calc_ray_start(world_pos, face_normal, primary_ray);
    RayDesc ray = {};
    ray.Origin    = sample_pos - primary_ray * ws_px_size;
    ray.Direction = dir;
    ray.TMin      = ws_px_size * 0.5f;
    ray.TMax      = 100000000000.0f;
    rtgi_trace_and_shade(ray, 0, payload);

    return RtgiRayResult(payload.color * weight * RTGI_RADIANCE_SCALE, min(payload.t, RTGI_SPECULAR_MAX_HIT_DISTANCE), compress_normal_octahedral_32(dir));
}

// Specular rays use their own RNG stream so they never correlate with the pixel's diffuse rays.
func rtgi_specular_seed(uint pixel_seed, uint sample_index)
{
    rand_seed(pixel_seed ^ 0x68E31DA4u);
    [loop] for (uint skip = 0u; skip < sample_index * 4u; ++skip) { rand(); }
}

void shade_ray_gen(uint2 dtid)
{
    let clk_start = clockARB();
    let push = rtgi_trace_diffuse_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    const float depth = push.attach.view_cam_half_res_depth.get()[dtid];
    const float2 pixel_index = float2(dtid.xy * 2u) + 0.5f;
    const CameraInfo camera = push.attach.globals.view_camera;
    const float3 world_position = rtgi_half_res_depth_to_world_space(camera, (pixel_index + 0.5f) * camera.inv_screen_size * 2.0f - 1.0f, depth);
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
        rtgi_trace_and_shade(ray, 0, payload);

        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid.xy, float4(payload.color, 2.0f), 2);
        return;
    }

    float acc_ray_shortness = 0.0f;        // mean ray shortness [0,1] over the rays; stored in .a
    float3 mean_perceptual_rgb = float3(0, 0, 0); // geometric mean (mean log rgb) over the rays; stored in .rgb
    float3 mean_specular_perceptual_rgb = float3(0, 0, 0);
    float  mean_specular_hit = 0.0f;

    const uint prime_shift0 = 257;   // just over typical period of frame time roughly (32 - 255 accum frames)
    const uint prime_shift1 = 9629;  // just over typical period of frame width x (480 - 8192)
    const uint prime_shift2 = 10069; // just over typical period of frame height y (480 - 8192)
    const uint frame_seed = rtgi_settings.animate_noise ? push.attach.globals.trunk_flt_frame_index * prime_shift0 : 0u;
    const uint thread_seed =
        frame_seed +
        dtid.x * prime_shift1 +
        dtid.y * prime_shift2;

    rand_seed(thread_seed);
    float2 rr_stbn = rand_stbn2d(Texture2DArray<float4>::get(push.attach.globals.stbn2d), dtid.xy, push.attach.globals.trunk_flt_frame_index);
    float2 rr = float2(rand(), rand());

    const float2 half_res_inv_render_target_size = push.attach.globals.settings.render_target_size_inv * 2.0f;
    const float  ws_px_size = depth > 0.0f ? rtgi_half_res_pixel_width_ws(half_res_inv_render_target_size, camera.near_plane, depth) : 0.0f;

    if (all(dtid.xy == uint2(0, 0)))
    {
        // Classic path: base rays are always traced, extras are capped at a quarter-res worth of rays.
        const uint total_geo = push.attach.ray_counters->total_geo_rays;
        push.attach.globals.readback.rtgi_requested_base_rays = total_geo;
        push.attach.globals.readback.rtgi_requested_extra_rays = push.attach.ray_counters->total_extra_rays;
        push.attach.globals.readback.rtgi_ray_budget = total_geo + (push.attach.globals.settings.render_target_size.x * push.attach.globals.settings.render_target_size.y) / 4;
    }

    // --- Determine this pixel's ray counts (0 for sky): diffuse + specular from one budget ---
    uint samples = 0u;
    uint specular_samples = 0u;
    if (depth > 0.0f)
    {
        const uint total_extra_ray_demands = push.attach.ray_counters->total_extra_rays;
        const uint max_extra_rays = (push.attach.globals.settings.render_target_size.x * push.attach.globals.settings.render_target_size.y) / 4;
        const float relative_allowed_rays = min(1.0f, float(max_extra_rays) / (float(total_extra_ray_demands) + 0.0001f));
        const float reproj_sample_count = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[dtid.xy]);
        const float2 impact_factors = push.attach.ray_impact.get()[dtid.xy];
        const RtgiRayImpact impact = { impact_factors.x, impact_factors.y };
        const RtgiRayDemand demand = rtgi_calc_ray_demand(reproj_sample_count, push.attach.specular_sample_count.get()[dtid.xy], impact, rtgi_settings);
        const uint base = rtgi_ray_demand_base(demand);
        const uint allowed_extra_samples = uint(float(demand.diffuse + demand.specular - base) * relative_allowed_rays);
        if (rtgi_settings.use_ray_redistribution)
        {
            uint extra_diffuse, extra_specular;
            rtgi_split_extra_rays(allowed_extra_samples, demand, extra_diffuse, extra_specular);
            specular_samples = (demand.specular > 0u ? 1u : 0u) + extra_specular;
            samples = 1u + extra_diffuse + specular_samples;
        }
        else
        {
            // Redistribution off: fixed rays per pixel, at least one per active signal (calc_fixed_ray_split).
            calc_fixed_ray_split(rtgi_settings.ray_percentage, demand.specular > 0u, samples, specular_samples);
        }
    }

    // --- Wave-coalesced ray-list allocation: exclusive prefix sum of the ray counts within the wave, then
    // ONE atomic per wave to reserve that wave's contiguous slice of the global ray list. Same ray_list /
    // ray_result / pixel_ray_alloc structures the repacked (distribute-rays) path uses. ---
    const uint lane_prefix = WavePrefixSum(samples);
    const uint wave_total  = WaveActiveSum(samples);
    uint wave_base = 0u;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(push.attach.ray_counters->ray_list_count, wave_total, wave_base);
    }
    wave_base = WaveReadLaneFirst(wave_base);
    const uint my_offset = wave_base + lane_prefix;

    // Hard capacity clamp so a rounding/overflow edge can never write past the ray_result buffer.
    const uint2 half_res = push.attach.globals.settings.render_target_size >> 1u;
    const uint  ray_list_capacity = half_res.x * half_res.y * RTGI_RAY_LIST_CAPACITY_MUL;
    uint write_count = samples;
    if (my_offset >= ray_list_capacity)               write_count = 0u;
    else if (my_offset + samples > ray_list_capacity) write_count = ray_list_capacity - my_offset;

    // Entries [0, diffuse_write_count) are diffuse, the rest specular.
    const uint diffuse_write_count  = min(samples - specular_samples, write_count);
    const uint specular_write_count = write_count - diffuse_write_count;
    {
        // Statistics: rays actually traced, per signal.
        const uint wave_diffuse = WaveActiveSum(diffuse_write_count);
        const uint wave_specular = WaveActiveSum(specular_write_count);
        if (WaveIsFirstLane())
        {
            InterlockedAdd(push.attach.ray_counters->shot_diffuse_rays, wave_diffuse);
            InterlockedAdd(push.attach.ray_counters->shot_specular_rays, wave_specular);
        }
    }
    var clk_after_diffuse = clk_start; // splits the pixel's clocks into its diffuse and specular rays (debug)
    if (write_count > 0u)
    {
        const float  inv_samples   = rcp(float(max(diffuse_write_count, 1u)));
        const float3 world_tangent = normalize(cross(face_normal, float3(0, 0, 1) + 0.0001f));
        const float3x3 tbn         = transpose(float3x3(world_tangent, cross(world_tangent, face_normal), face_normal));
        const float3 sample_pos    = rt_calc_ray_start(world_position, face_normal, primary_ray);

        // Pioneer-guided direction support -- see the matching block (and rtgi_guided_sampling.hlsl) in
        // ray_gen_from_list_body for the full explanation. Read once per pixel, reused by every ray below.
        // Already half-res-pixel-aligned, no addressing needed.
        float4 pixel_guide_sh_y = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (rtgi_settings.pioneer_guiding_enabled)
        {
            pixel_guide_sh_y = push.attach.guide_sh_y.get()[dtid];
        }

        for (uint i = 0u; i < diffuse_write_count; ++i)
        {
            float3 sample_dir;
            // Correction weight for pioneer-guided sampling; 1.0 (no-op) unless that branch fires below --
            // see the matching comment in ray_gen_from_list_body.
            float guided_weight = 1.0f;
            if (rtgi_settings.trace_use_stbn != 0)
            {
                const uint stbn_frame_seed = rtgi_settings.animate_noise ? push.attach.globals.trunk_flt_frame_index : 0u;
                rand_seed(stbn_frame_seed + i * prime_shift1);
                const float3 importance_rand_hemi_sample = rand_stbnCosDir(Texture2DArray<float4>::get(push.attach.globals.stbnCosDir), pixel_index, (rtgi_settings.animate_noise ? push.attach.globals.trunk_flt_frame_index : 0) + rand());
                sample_dir = mul(tbn, importance_rand_hemi_sample);
            }
            else if (rtgi_settings.pioneer_guiding_enabled)
            {
                // Returns a WORLD-space direction directly (builds its own basis internally) -- does NOT
                // go through `tbn`, unlike the other two branches.
                sample_dir = rtgi_sample_guided_diffuse_dir(
                    face_normal, pixel_guide_sh_y, rtgi_settings.guide_concentration, 1.0f, guided_weight);
            }
            else
            {
                sample_dir = mul(tbn, rand_cosine_sample_hemi());
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
            ray.Direction = sample_dir;
            const uint flags = {};
            rtgi_trace_and_shade(ray, flags, payload);

            // Write this ray's result into the shared ray list (same layout the blend pass produced), so
            // the pre-filter re-blends (and per-ray firefly-clamps) it identically to the repacked path.
            // guided_weight folds in the pioneer-guided-sampling correction (1.0 = no-op unless it fired).
            const float3 ray_rgb = payload.color * guided_weight * RTGI_RADIANCE_SCALE;
            push.attach.ray_result[my_offset + i] = RtgiRayResult(ray_rgb, payload.t, compress_normal_octahedral_32(sample_dir));

            mean_perceptual_rgb += linear_to_perceptual_rgb(ray_rgb, push.attach.globals.inv_exposure) * inv_samples;
            acc_ray_shortness   += calc_ray_shortness(payload.t, ws_px_size, rtgi_settings.max_visibility_pixel_range) * inv_samples;
        }

        clk_after_diffuse = clockARB();

        // Specular rays (own RNG stream) in the slots after the diffuse ones.
        const float inv_specular_samples = rcp(float(max(specular_write_count, 1u)));
        for (uint i = 0u; i < specular_write_count; ++i)
        {
            rtgi_specular_seed(thread_seed, i);
            const RtgiRayResult spec_result = rtgi_trace_specular_ray(dtid, world_position, face_normal, primary_ray, ws_px_size, pixel_guide_sh_y);
            push.attach.ray_result[my_offset + diffuse_write_count + i] = spec_result;
            mean_specular_perceptual_rgb += linear_to_perceptual_rgb(spec_result.radiance, push.attach.globals.inv_exposure) * inv_specular_samples;
            mean_specular_hit += spec_result.t * inv_specular_samples;
        }
    }

    // Same outputs the distribute pass produces: per-pixel ray-list offset, ray count, and the log-rgb /
    // shortness the pre-filter reads. diffuse / diffuse2 are NOT produced (the pre-filter re-blends rays).
    push.attach.pixel_ray_alloc.get()[dtid.xy] = my_offset;
    push.attach.ray_count_image.get()[dtid.xy] = diffuse_write_count;
    push.attach.specular_ray_count_image.get()[dtid.xy] = specular_write_count;
    push.attach.perceptual_rgb_shortness.get()[dtid.xy] = float4(mean_perceptual_rgb, acc_ray_shortness);
    push.attach.specular_perceptual_rgb_hit.get()[dtid.xy] = float4(mean_specular_perceptual_rgb, mean_specular_hit);

    let trace_debug_mode = push.attach.globals.settings.debug_draw_mode;
    if (trace_debug_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_TRACE_CLOCKS || trace_debug_mode == DEBUG_DRAW_MODE_RTGI_SPECULAR_TRACE_CLOCKS)
    {
        let clk_end = clockARB();
        const uint clocks = trace_debug_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_TRACE_CLOCKS ? uint(clk_after_diffuse - clk_start) : uint(clk_end - clk_after_diffuse);
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

    if (ray_index >= min(push.attach.ray_counters->ray_list_count, rtgi_ray_list_limit(push.attach.globals.settings.render_target_size >> 1u, push.attach.globals.rtgi_settings)))
        return;

    const RtgiRayEntry entry   = push.attach.ray_list[ray_index];
    const uint2 pixel_xy       = uint2(entry.packed_xy & 0xFFFFu, entry.packed_xy >> 16u);
    const uint  sample_index   = entry.sample_index;

    const float depth = push.attach.view_cam_half_res_depth.get()[pixel_xy];
    if (depth == 0.0f)
    {
        push.attach.ray_result[ray_index] = RtgiRayResult(float3(0.0f, 0.0f, 0.0f), 0.0f, 0u);
        return;
    }

    let rtgi_settings = push.attach.globals.rtgi_settings;
    const CameraInfo camera   = push.attach.globals.view_camera;
    const float2 pixel_index  = float2(pixel_xy * 2u) + 0.5f;
    const float3 world_pos    = rtgi_half_res_depth_to_world_space(camera, (pixel_index + 0.5f) * camera.inv_screen_size * 2.0f - 1.0f, depth);
    const float3 face_normal  = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[pixel_xy].r);
    const float3 primary_ray  = normalize(world_pos - camera.position);
    const float2 half_res_inv_render_target_size = push.attach.globals.settings.render_target_size_inv * 2.0f;
    const float  ws_px_size   = rtgi_half_res_pixel_width_ws(half_res_inv_render_target_size, camera.near_plane, depth);

    const uint prime_shift0 = 257u;
    const uint prime_shift1 = 9629u;
    const uint prime_shift2 = 10069u;
    const uint prime_shift3 = 6151u;
    const uint frame_seed   = rtgi_settings.animate_noise ? push.attach.globals.trunk_flt_frame_index * prime_shift0 : 0u;
    // Fold the reprojected history length into the seed. frame_index alone can alias across frames, so a
    // pixel that shot N rays one frame could re-draw near-identical directions the next. The history count
    // advances by the pixel's rays-shot each frame (fastest exactly when many rays/frame make repeats most
    // likely), giving an extra decorrelating dimension. Gated by animate_noise so frozen-noise stays frozen.
    const float history_count = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[pixel_xy]);
    const uint history_seed   = rtgi_settings.animate_noise ? uint(max(history_count, 0.0f)) * prime_shift3 : 0u;
    const float3 world_tangent = normalize(cross(face_normal, float3(0, 0, 1) + 0.0001f));
    const float3x3 tbn         = transpose(float3x3(world_tangent, cross(world_tangent, face_normal), face_normal));
    // A pixel's entries are [diffuse 0..n_d) then [specular 0..n_s); the diffuse count says which this one is.
    const uint diffuse_ray_count = push.attach.ray_count_image.get()[pixel_xy];
    const bool is_specular_ray = sample_index >= diffuse_ray_count;

    // Pioneer-guided direction support: read the same-frame, lag-free guide built by the pioneer trace +
    // spatial RIS resample (rtgi_guide_resample.hlsl) -- already half-res-pixel-aligned, no addressing
    // needed. Feeds rtgi_sample_guided_diffuse_dir below. See rtgi_guided_sampling.hlsl.
    float4 pixel_guide_sh_y = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (rtgi_settings.pioneer_guiding_enabled)
    {
        pixel_guide_sh_y = push.attach.guide_sh_y.get()[pixel_xy];
    }

    // Seed ONCE per pixel+frame (NOT per ray). Each ray is a separate shader invocation, so we advance
    // the per-pixel RNG sequence to this ray's slot instead of folding sample_index into the seed:
    // re-seeding per ray with base + sample_index*prime does NOT decorrelate (a single PCG step barely
    // mixes nearby seeds), which made a pixel's N rays near-duplicates. Downstream draws (2 for plain
    // cosine, a variable count for guided sampling) start from this ray's own decorrelated slot regardless
    // of how many rand() calls they end up consuming — mirrors how the classic per-pixel trace draws
    // sequentially across its sample loop. Harmless no-op for the STBN branch below, which never calls rand().
    if (is_specular_ray)
    {
        rtgi_specular_seed(frame_seed + history_seed + pixel_xy.x * prime_shift1 + pixel_xy.y * prime_shift2, sample_index - diffuse_ray_count);
        push.attach.ray_result[ray_index] = rtgi_trace_specular_ray(pixel_xy, world_pos, face_normal, primary_ray, ws_px_size, pixel_guide_sh_y);
        if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_SPECULAR_TRACE_CLOCKS)
        {
            const uint clocks = uint(clockARB() - clk_start);
            write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, pixel_xy, float4(Heatmap(clocks * 0.0001f * push.attach.globals.settings.debug_visualization_scale), 1.0f + push.attach.globals.settings.debug_visualization_blend), 2);
        }
        return;
    }

    rand_seed(frame_seed + history_seed + pixel_xy.x * prime_shift1 + pixel_xy.y * prime_shift2);
    [loop] for (uint skip = 0u; skip < sample_index * 2u; ++skip) { rand(); }

    // Ray direction. The repacked/redistribution path previously ALWAYS drew rand_cosine_sample_hemi, so
    // trace_use_stbn did nothing here — STBN only worked on the classic per-pixel path (shade_ray_gen).
    // Mirror that path: when trace_use_stbn is set, draw the blue-noise cosine direction indexed by the
    // pixel's screen coordinate (pixel_index, the same full-res coord shade_ray_gen uses).
    float3 sample_dir;
    // Correction weight for pioneer-guided sampling (importance-sampling ratio); multiplied into the
    // traced radiance below. Stays 1.0 (no-op) for STBN/plain-cosine sampling, which need no such weight
    // because their pdf's cos(theta)/PI cancels exactly against the Lambertian estimator's own numerator.
    float guided_weight = 1.0f;
    if (rtgi_settings.trace_use_stbn != 0)
    {
        // STBN z-slice = frame + sample_index. The per-ray offset MUST be an integer slice step: a pixel's
        // rays are decorrelated by reading different (blue-noise-decorrelated) temporal slices at the same
        // xy texel. rand() CANNOT do this -- it returns [0,1) and the frame arg is an int, so `+ rand()`
        // truncates to zero and every ray of the pixel reads the same slice -> IDENTICAL direction.
        // animate_noise off -> slices 0..N-1 per pixel, frozen across frames but still distinct per ray.
        const int stbn_frame = (rtgi_settings.animate_noise ? int(push.attach.globals.trunk_flt_frame_index) : 0) + int(sample_index);
        const float3 importance_rand_hemi_sample = rand_stbnCosDir(Texture2DArray<float4>::get(push.attach.globals.stbnCosDir), pixel_index, stbn_frame);
        sample_dir = mul(tbn, importance_rand_hemi_sample);
    }
    else if (rtgi_settings.pioneer_guiding_enabled)
    {
        // rtgi_sample_guided_diffuse_dir builds its own basis internally (around either face_normal or the
        // SH dominant direction) and returns a WORLD-space direction directly -- unlike the other two
        // branches, it does NOT go through the pre-built `tbn` here.
        sample_dir = rtgi_sample_guided_diffuse_dir(
            face_normal, pixel_guide_sh_y, rtgi_settings.guide_concentration, 1.0f, guided_weight);
    }
    else
    {
        sample_dir = mul(tbn, rand_cosine_sample_hemi());
    }

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
    rtgi_trace_and_shade(ray, flags, payload);

    // Store the raw hit distance; the blend pass converts it to bounded shortness [0,1] per ray and
    // averages over the pixel's rays into the ray-length texture for a stable denoiser guide.
    // guided_weight folds in the pioneer-guided-sampling correction (1.0 = no-op unless that path fired).
    push.attach.ray_result[ray_index] = RtgiRayResult(payload.color * guided_weight * RTGI_RADIANCE_SCALE, payload.t, compress_normal_octahedral_32(sample_dir));

    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_TRACE_CLOCKS)
    {
        let clk_end = clockARB();
        const uint clocks = uint(clk_end - clk_start);
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, pixel_xy, float4(Heatmap(clocks * 0.0001f * push.attach.globals.settings.debug_visualization_scale), 1.0f + push.attach.globals.settings.debug_visualization_blend), 2);
    }
}

// Pioneer ray direction distribution: 1 = uniform hemisphere, 0 = cosine-weighted (see pioneer_ray_gen).
#define RTGI_GUIDE_PIONEER_UNIFORM_HEMISPHERE 1

// Pioneer trace, gated by rtgi_settings.pioneer_guiding_enabled. Dispatched at a SPARSE grid --
// 1/RTGI_GUIDE_PIONEER_GRID_DIV resolution of the half-res trace grid in each axis -- one ray per cell.
// Which half-res pixel each cell samples ROTATES every frame (a simple mod-DIV cycle through the
// DIVxDIV sub-positions), so the sparse pattern isn't fixed; entry_guide_resample_horizontal/vertical
// (rtgi_guide_resample.hlsl) MUST use the exact same rotation formula to find each cell's sampled pixel
// back. This pass writes raw per-cell data only -- no averaging, no history -- rtgi_guide_resample.hlsl's
// separable resample is what actually produces the per-pixel guide the main trace reads.
void pioneer_ray_gen(uint2 pioneer_dtid)
{
    let push = rtgi_trace_diffuse_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    // Same rotation formula entry_guide_resample uses to map a candidate cell back to a half-res pixel.
    // Gated by animate_noise so a frozen frame keeps a fixed pioneer pattern instead of still cycling.
    // trunk_flt_frame_index, not raw frame_index -- RTGI convention (see globals.inl); mod-DIV is
    // unaffected since DIV (4) divides the 4096 truncation period evenly.
    const uint rotation_frame = rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
    const uint2 rotation = uint2(
        rotation_frame % RTGI_GUIDE_PIONEER_GRID_DIV,
        (rotation_frame / RTGI_GUIDE_PIONEER_GRID_DIV) % RTGI_GUIDE_PIONEER_GRID_DIV);
    const uint2 pixel_xy = pioneer_dtid * RTGI_GUIDE_PIONEER_GRID_DIV + rotation;

    const uint2 half_res_size = push.attach.globals.settings.render_target_size >> 1u;
    // .w < 0 is the invalid/no-data sentinel entry_guide_resample skips as a RIS candidate -- brightness
    // Y is otherwise always >= 0, so it's unambiguous. Covers both the out-of-bounds edge cells (grid is
    // ceil-divided, so the rotated pixel can fall outside on the last row/column) and disocclusion.
    if (any(pixel_xy >= half_res_size))
    {
        push.attach.pioneer_hit_y.get()[pioneer_dtid] = float4(0.0f, 0.0f, 0.0f, -1.0f);
        return;
    }

    const float depth = push.attach.view_cam_half_res_depth.get()[pixel_xy];
    if (depth == 0.0f)
    {
        push.attach.pioneer_hit_y.get()[pioneer_dtid] = float4(0.0f, 0.0f, 0.0f, -1.0f);
        return;
    }

    const CameraInfo camera = push.attach.globals.view_camera;
    const float2 pixel_index = float2(pixel_xy * 2u) + 0.5f;
    const float3 world_position = rtgi_half_res_depth_to_world_space(camera, (pixel_index + 0.5f) * camera.inv_screen_size * 2.0f - 1.0f, depth);
    const float3 face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[pixel_xy].r);
    const float3 primary_ray = normalize(world_position - camera.position);
    const float2 half_res_inv_render_target_size = push.attach.globals.settings.render_target_size_inv * 2.0f;
    const float ws_px_size = rtgi_half_res_pixel_width_ws(half_res_inv_render_target_size, camera.near_plane, depth);

    // Unguided sampling -- pioneer rays ARE the raw signal being gathered, so they must not guide off anything
    // themselves. RTGI_GUIDE_PIONEER_UNIFORM_HEMISPHERE (default 1): uniform over the hemisphere, so the search for
    // bright spots covers grazing directions as densely as the pole (they also feed the rough specular guide mix);
    // 0 = cosine-weighted (density matches diffuse importance). The guide resample weights candidates by brightness
    // only (no sampling pdf), so either distribution plugs in unchanged. Seeded per pixel+frame like the other paths.
    const uint prime_shift1 = 9629u;
    const uint prime_shift2 = 10069u;
    const uint frame_seed = rtgi_settings.animate_noise ? push.attach.globals.trunk_flt_frame_index * 257u : 0u;
    rand_seed(frame_seed + pixel_xy.x * prime_shift1 + pixel_xy.y * prime_shift2);

    const float3 world_tangent = normalize(cross(face_normal, float3(0, 0, 1) + 0.0001f));
    const float3x3 tbn = transpose(float3x3(world_tangent, cross(world_tangent, face_normal), face_normal));
#if RTGI_GUIDE_PIONEER_UNIFORM_HEMISPHERE
    // Uniform hemisphere: cos(theta) uniform in [0, 1).
    const float cos_theta = rand();
    const float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    const float phi = rand() * 2.0f * PI;
    const float3 local_dir = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
#else
    const float3 local_dir = rand_cosine_sample_hemi();
#endif
    const float3 sample_dir = mul(tbn, local_dir);

    RayPayload payload = {};
    payload.dtid = pixel_xy;

    const float3 sample_pos = rt_calc_ray_start(world_position, face_normal, primary_ray);
    RayDesc ray = {};
    ray.Origin    = sample_pos - primary_ray * ws_px_size;
    ray.Direction = sample_dir;
    ray.TMin      = ws_px_size * 0.5f;
    // Bounded (unlike the main trace's effectively-unbounded rays) -- the pioneer trace only needs nearby
    // bounce lighting to build a useful direction guide; see the field comment in rtgi.inl.
    ray.TMax      = rtgi_settings.guide_pioneer_trace_max_distance;
    const uint flags = {};
    rtgi_trace_and_shade(ray, flags, payload);

    // Store a RECONNECTION payload, not a baked SH-Y direction: the ray's actual hit position X (world
    // space) plus the direction-independent brightness Y measured there. The traced surface is Lambertian
    // (this whole pipeline only does diffuse GI), so its exitant radiance is the SAME in every exit
    // direction -- Y needs no per-receiver re-derivation, ever. What changes per receiver is only the
    // DIRECTION from the receiver to X, and that's deliberately not computed here: baking `sample_dir` in
    // now would make it only valid for THIS pixel, defeating the point of storing X at all. Every reuse
    // downstream (H/V passes) just carries (X, Y) forward unchanged; entry_guide_resolve is the one place
    // that turns this into a real direction, computed fresh for whichever half-res pixel actually consumes
    // it as a guide. On a sky miss payload.t is a huge sentinel (see shade_miss) that's WAY past this
    // ray's own TMax -- clamp to ray.TMax (the pioneer trace's bounded, finite max distance) before
    // building the position. Two reasons: (1) pioneer_hit_y is stored as R16G16B16A16_SFLOAT (half
    // float, max finite ~65504) -- origin + direction*1e12 overflows that to +-Inf on write, which then
    // turns into a NaN reconnection direction downstream (entry_guide_resolve) that silently slips past
    // the `dot(normal, recon_dir) <= 0` guard (NaN comparisons are always false) and NaNs the guide's
    // SH-Y, which rtgi_sh_dominant_direction then falls back to the surface normal for -- i.e. exactly
    // the sky-miss-guided rays (bright, likely to win the RIS pick) silently losing their real direction.
    // (2) ray.TMax is still "far enough" for the same reason 1e12 was meant to be: any nearby receiver's
    // parallax against a point that far along sample_dir is negligible, so reconnection still reproduces
    // ~sample_dir for every receiver, same as the original intent, just representable.
    const float3 hit_position = ray.Origin + ray.Direction * min(payload.t, ray.TMax);
    const float Y = linear_to_y_co_cg(payload.color).x;
    // max(...,0) keeps Y unambiguous against the negative invalid sentinel above.
    push.attach.pioneer_hit_y.get()[pioneer_dtid] = float4(hit_position, max(Y, 0.0f));
}

// Single raygen entry point. Switches on the setting so all trace paths can share one pipeline
// (avoids daxa's single-handle raygen SBT limitation). The task graph only ever dispatches one of
// the paths per task — with the matching dispatch shape — based on push/settings values.
[shader("raygeneration")]
void ray_gen()
{
    if (rtgi_trace_diffuse_push.is_pioneer_pass)
    {
        pioneer_ray_gen(DispatchRaysIndex().xy);
        return;
    }
    if (rtgi_trace_diffuse_push.attach.globals.rtgi_settings.use_repacked_ray_dispatch)
    {
        ray_gen_from_list_body();
    }
    else
    {
        shade_ray_gen(DispatchRaysIndex().xy);
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