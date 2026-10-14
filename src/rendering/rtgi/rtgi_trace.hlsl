#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtgi_trace.inl"
#include "rtgi_guide_resample.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/pgi.hlsl"
#include "shader_lib/debug.glsl"

#include "rtgi_shared.hlsl"
#include "rtgi_guided_sampling.hlsl"
#include "rtgi_specular_sampling.hlsl"

// RTGI ray generation + tracing, diffuse and specular rays. Three ray tracing pipelines (see rtgi.hpp):
//   "Trace RTGI" (rtgi_trace_compile_info): ray_gen (classic per-pixel trace) + any_hit here, the closest hit / miss
//     shaders that shade the rays live in rtgi_shade.hlsl.
//   "Trace RTGI Pioneer" (rtgi_trace_pioneer_compile_info): ray_gen_pioneer + any_hit here, coarse-only closest hit
//     + miss in rtgi_shade.hlsl.
//   "Trace RTGI Split" (rtgi_trace_split_compile_info): repacked ray-list trace, ray_gen_split + hit shaders that only
//     record the hit and bin the ray into a shade list; the compute passes in rtgi_shade.hlsl shade them afterwards.

// == Shared ray setup ===========================================================================================

// RNG seed primes.
#define RTGI_SEED_FRAME 257u    // just over typical period of frame time roughly (32 - 255 accum frames)
#define RTGI_SEED_X 9629u       // just over typical period of frame width x (480 - 8192)
#define RTGI_SEED_Y 10069u      // just over typical period of frame height y (480 - 8192)
#define RTGI_SEED_HISTORY 6151u

// Per pixel + frame RNG seed. extra: additional decorrelation term (0 = none).
func rtgi_pixel_seed(uint2 pixel_xy, uint extra) -> uint
{
    let push = rtgi_trace_push;
    const uint frame_seed = push.attach.globals.rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) * RTGI_SEED_FRAME : 0u;
    return frame_seed + extra + pixel_xy.x * RTGI_SEED_X + pixel_xy.y * RTGI_SEED_Y;
}

// Specular rays use their own RNG stream so they never correlate with the pixel's diffuse rays.
func rtgi_specular_seed(uint pixel_seed, uint sample_index)
{
    rand_seed(pixel_seed ^ 0x68E31DA4u);
    [loop] for (uint skip = 0u; skip < sample_index * 4u; ++skip) { rand(); }
}

#define GOLDEN_RATIO 1.6181
#define STBN_WDITH 128
#define STBN_SIZE uint3(STBN_WDITH,STBN_WDITH,32)
#define STBN_GRID_SIZE uint3(STBN_WDITH,STBN_WDITH,32)

float3 rand_stbnCosDir(Texture2DArray<float4> stbn2d_image, uint2 pixel, int frame)
{
    pixel = pixel % STBN_GRID_SIZE.xy;
    const uint z_wrap = frame / STBN_SIZE.z;
    const uint z = frame % STBN_SIZE.z;
    pixel = (pixel + uint2(GOLDEN_RATIO * float2(STBN_SIZE.xy * z_wrap))) % STBN_SIZE.xy;
    return stbn2d_image[uint3(pixel,z)].xyz * 2.0f - 1.0f;
}

// The half-res pixel a ray starts from.
struct RtgiTracePixel
{
    float depth;        // linear depth, 0 = sky
    float3 world_pos;
    float3 face_normal;
    float ws_px_size;   // world space half-res pixel width at the pixel (0 for sky)
};

func rtgi_load_trace_pixel(uint2 pixel_xy) -> RtgiTracePixel
{
    let push = rtgi_trace_push;
    const CameraInfo camera = push.attach.globals.view_camera;
    RtgiTracePixel ret;
    ret.depth = push.attach.view_cam_half_res_depth.get()[pixel_xy];
    const float2 ndc = (float2(pixel_xy * 2u) + 1.0f) * camera.inv_screen_size * 2.0f - 1.0f;
    ret.world_pos = rtgi_half_res_depth_to_world_space(camera, ndc, ret.depth);
    ret.face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[pixel_xy].r);
    const float2 half_res_inv_render_target_size = push.attach.globals.settings.render_target_size_inv * 2.0f;
    ret.ws_px_size = ret.depth > 0.0f ? rtgi_half_res_pixel_width_ws(half_res_inv_render_target_size, camera.near_plane, ret.depth) : 0.0f;
    return ret;
}

// Diffuse (and pioneer) ray from a half-res pixel: back-offset the origin by one pixel width along the view ray.
func rtgi_diffuse_ray_desc(RtgiTracePixel pixel, float3 primary_ray, float3 direction, float t_max) -> RayDesc
{
    RayDesc ray = {};
    ray.Origin    = rt_calc_ray_start(pixel.world_pos, pixel.face_normal, primary_ray) - primary_ray * pixel.ws_px_size;
    ray.Direction = direction;
    ray.TMin      = pixel.ws_px_size * 0.5f;
    ray.TMax      = t_max;
    return ray;
}

// Diffuse ray direction: STBN blue noise cosine (trace_use_stbn), pioneer guided (valid guide) or plain cosine
// (rand(), seed before calling). guided_weight: importance sampling correction of the guided direction, 1 otherwise
// (the cosine pdf cancels against the Lambertian estimator). stbn_frame: STBN z slice; MUST step by an integer per ray
// of a pixel, the rays are decorrelated by reading different temporal slices at the same texel.
func rtgi_sample_diffuse_dir(float3 face_normal, uint2 full_res_pixel, int stbn_frame, float4 guide_sh_y, bool use_guide, out float guided_weight) -> float3
{
    let push = rtgi_trace_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    guided_weight = 1.0f;
    if (rtgi_settings.trace_use_stbn == 0 && rtgi_settings.pioneer_guiding_enabled && use_guide && rtgi_guide_is_valid(guide_sh_y))
    {
        // Builds its own basis (around the face normal or the guide direction), returns a world space direction.
        return rtgi_sample_guided_diffuse_dir(face_normal, guide_sh_y, rtgi_settings.guide_concentration, 1.0f, guided_weight);
    }
    const float3 world_tangent = normalize(cross(face_normal, float3(0, 0, 1) + 0.0001f));
    const float3x3 tbn = transpose(float3x3(world_tangent, cross(world_tangent, face_normal), face_normal));
    if (rtgi_settings.trace_use_stbn != 0)
    {
        return mul(tbn, rand_stbnCosDir(Texture2DArray<float4>::get(push.attach.globals.stbnCosDir), full_res_pixel, stbn_frame));
    }
    return mul(tbn, rand_cosine_sample_hemi());
}

// STBN z slice of ray `ray_index_in_signal` of a pixel this frame.
func rtgi_stbn_frame(uint ray_index_in_signal) -> int
{
    let push = rtgi_trace_push;
    return (push.attach.globals.rtgi_settings.animate_noise ? int(push.attach.globals.trunk_flt_frame_index) : 0) + int(ray_index_in_signal);
}

// Builds ONE specular ray for a half-res pixel. Uses the caller's rand() state (seed it before calling).
// weight = sampling weight of the ray (0 = sample rejected, do not trace).
func rtgi_specular_ray_desc(uint2 pixel_xy, RtgiTracePixel pixel, float3 primary_ray, float4 guide_sh_y, out float weight) -> RayDesc
{
    let push = rtgi_trace_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    const float3 view_dir = -primary_ray;
    const uint normal_roughness = push.attach.view_cam_half_res_normal_roughness.get()[pixel_xy];
    const float3 detail_normal = unpack_normal_roughness_normal(normal_roughness);
    const float3 shading_normal = rtgi_specular_shading_normal(detail_normal, view_dir);
    const float roughness = unpack_normal_roughness_roughness(normal_roughness);

    weight = 1.0f;
    const bool guide_valid = rtgi_settings.pioneer_guiding_enabled != 0 && rtgi_guide_is_valid(guide_sh_y);
    const float3 dir = rtgi_sample_specular_dir(
        shading_normal, pixel.face_normal, view_dir, roughness,
        guide_sh_y, guide_valid, rtgi_settings.specular_guide_concentration,
        rtgi_settings.specular_guide_mix, weight);

    RayDesc ray = {};
    // The half-res position is reconstructed from ONE representative sub-pixel's depth at the half-res pixel
    // center, so it can sit off the true surface. Pull back along the view ray (covers the error at steep views)
    // AND lift along the face normal by one pixel width: at grazing views the view-ray pullback lifts the origin
    // by only ws_px_size * NoV, so near-tangent reflection rays would start on / below the surface and hit it
    // (black grazing reflections).
    ray.Origin    = rt_calc_ray_start(pixel.world_pos, pixel.face_normal, primary_ray) - primary_ray * pixel.ws_px_size + pixel.face_normal * pixel.ws_px_size;
    ray.Direction = dir;
    ray.TMin      = pixel.ws_px_size * 0.5f;
    ray.TMax      = RTGI_RAY_T_MAX;
    return ray;
}

// Alpha test of the any hit shaders of both pipelines.
func rtgi_any_hit_is_alpha_hit(float2 barycentrics) -> bool
{
    let push = rtgi_trace_push;
    return rt_is_alpha_hit(
        push.attach.globals,
        push.attach.mesh_instances,
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.materials,
        barycentrics,
        PrimitiveIndex(), InstanceID(), WorldRayOrigin(), WorldRayDirection(), RayTCurrent());
}

// == Per-ray guide fetch =========================================================================
// Why per ray: if all rays of a pixel (or all pixels of a neighborhood) shared one guide per frame, a super-sampled
// pixel would bake that frame's guide into many samples at once (stripes / splotches where the allocator put
// bursts). Each ray instead picks its own guide, so burst rays and neighbouring pixels follow different guides.
// Each ray reads a pioneer-RESOLUTION guide cell
// (pioneer_guide_hit_y: the vertical resample's RIS pick per pioneer cell, a hit position + brightness), taken
// from the pre-blur's Poisson disc sequence (g_Poisson16, rtgi_shared.hlsl) scaled to
// RTGI_GUIDE_STOCHASTIC_FETCH_CELL_RADIUS pioneer cells around the pixel's own cell. ONE sequence per pixel,
// walked across the pixel's rays: ray i, attempt a uses tap (i * ATTEMPTS + a) mod 16, rotated by a random
// per-pixel angle and offset from the pixel's continuous position inside its cell.
// The cell's hit is RECONNECTED to THIS ray's own position (exact direction, unlike reading a neighbor's
// already-resolved per-pixel guide, which was reconnected from the neighbor). The cell must be geometry, have a
// pick, pass the same distance gate the resolve uses (RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD), and the reconnected
// direction must lie in this pixel's hemisphere; after RTGI_GUIDE_STOCHASTIC_FETCH_ATTEMPTS failed taps the ray
// gets no guide (zero) and samples unguided. Any guide works for correctness (the pdf weight is computed for the guide
// actually used). Uses no RNG, so the ray direction sequence (rand()) is untouched.
#define RTGI_GUIDE_STOCHASTIC_FETCH_CELL_RADIUS 4.0f
#define RTGI_GUIDE_STOCHASTIC_FETCH_ATTEMPTS 3u

// 1 = only the FIRST ray of each signal per pixel is guided, further burst rays sample plain cosine / GGX VNDF
// (the older fix for burst correlation, before the per-ray fetch decorrelated guides). 0 = every ray is guided.
#define RTGI_GUIDE_FIRST_RAY_ONLY 0

// A fetched guide is usable when it is non-zero (rtgi_fetch_ray_guide returns zero when no cell qualified).
func rtgi_guide_is_valid(float4 guide_sh_y) -> bool
{
    return dot(guide_sh_y.xyz, guide_sh_y.xyz) > 1e-12f;
}

// Whether ray `ray_index_in_signal` (0-based within its signal) of a pixel uses the guide.
func rtgi_ray_uses_guide(uint ray_index_in_signal) -> bool
{
    return RTGI_GUIDE_FIRST_RAY_ONLY == 0 || ray_index_in_signal == 0u;
}

// Debug view: the guide direction a pixel's FIRST ray (diffuse ray 0) used, as color. Black = no guide (unguided).
void rtgi_guide_direction_debug_draw(uint2 pixel_xy, float4 guide_sh_y)
{
    let push = rtgi_trace_push;
    if (push.attach.globals.settings.debug_draw_mode != DEBUG_DRAW_MODE_RTGI_GUIDE_DIRECTION) { return; }
    if (push.attach.globals.rtgi_settings.trace_use_stbn != 0) { guide_sh_y = float4(0.0f, 0.0f, 0.0f, 0.0f); } // STBN ignores the guide
    const float moment_len = length(guide_sh_y.xyz);
    const float3 color = moment_len > 1e-8f ? (guide_sh_y.xyz / moment_len) * 0.5f + 0.5f : float3(0.0f, 0.0f, 0.0f);
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, pixel_xy, float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend), 2);
}

// Per-pixel hash for the guide fetch kernel rotation (PCG). Static per pixel: the pioneer cells themselves rotate
// and get new picks every frame.
func rtgi_guide_fetch_pixel_hash(uint2 pixel_xy) -> uint
{
    const uint state = (pixel_xy.x * 9629u ^ pixel_xy.y * 10069u ^ 0x2545F491u) * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// ray_key: the ray's index within the pixel; diffuse and specular rays each walk the sequence from tap 0.
// half_res_depth: the pixel's linear depth (world_pos was reconstructed from it).
func rtgi_fetch_ray_guide(uint2 pixel_xy, float3 world_pos, float half_res_depth, float3 face_normal, float ws_px_size, uint ray_key) -> float4
{
    let push = rtgi_trace_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    if (rtgi_settings.pioneer_guiding_enabled == 0) { return float4(0.0f, 0.0f, 0.0f, 0.0f); }

    const CameraInfo camera = push.attach.globals.view_camera;
    const uint2 half_res_size = push.attach.globals.settings.render_target_size >> 1u;
    // Same pioneer cell rotation + grid as ray_gen_pioneer and the resample / resolve passes -- MUST stay in sync.
    const uint rotation_frame = rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
    const int2 rotation = int2(
        int(rotation_frame % RTGI_GUIDE_PIONEER_GRID_DIV),
        int((rotation_frame / RTGI_GUIDE_PIONEER_GRID_DIV) % RTGI_GUIDE_PIONEER_GRID_DIV));
    const int2 pioneer_grid_size = int2((half_res_size + RTGI_GUIDE_PIONEER_GRID_DIV - 1u) / RTGI_GUIDE_PIONEER_GRID_DIV);
    // The pixel's continuous position in pioneer-cell space (so pixels at different spots inside one cell reach
    // different neighbor cells), and a per-pixel random rotation of the Poisson kernel: without it every pixel
    // walks the same tap sequence, so whole 4x4 blocks pick the same cells and the guides stay spatially uniform.
    const float2 own_cell_pos = (float2(int2(pixel_xy) - rotation) + 0.5f) / float(RTGI_GUIDE_PIONEER_GRID_DIV);
    const uint pixel_hash = rtgi_guide_fetch_pixel_hash(pixel_xy);
    const float kernel_angle = float(pixel_hash >> 8u) * (6.28318530718f / 16777216.0f);
    float kernel_sin, kernel_cos;
    sincos(kernel_angle, kernel_sin, kernel_cos);
    const float2x2 kernel_rotation = float2x2(kernel_cos, -kernel_sin, kernel_sin, kernel_cos);
    // Distance gate in VIEW space: same distance as in world space (rigid transform), but far fewer live registers
    // than unprojecting every reference pixel to world space (3 inv_view_proj columns + camera position across the
    // loop). The view ray is affine in NDC and the perspective inverse projection maps an NDC offset through its
    // diagonal only, so ref - own = own_ray * (z_ref - z_own) + (d_ndc * diag, 0) * z_ref.
    const float2 own_ndc = (float2(pixel_xy * 2u) + 1.0f) * camera.inv_screen_size * 2.0f - 1.0f;
    const float3 own_ray_vs = camera_view_ray_vs(camera, own_ndc);
    const float2 ndc_per_pixel_vs = float2(camera.inv_proj[0][0], camera.inv_proj[1][1]) * camera.inv_screen_size * 4.0f;
    for (uint attempt = 0u; attempt < RTGI_GUIDE_STOCHASTIC_FETCH_ATTEMPTS; ++attempt)
    {
        const float2 tap = mul(kernel_rotation, g_Poisson16[(ray_key * RTGI_GUIDE_STOCHASTIC_FETCH_ATTEMPTS + attempt) & 15u].xy);
        const int2 cell = clamp(int2(floor(own_cell_pos + tap * RTGI_GUIDE_STOCHASTIC_FETCH_CELL_RADIUS)), int2(0, 0), pioneer_grid_size - 1);
        const int2 ref_pixel = cell * int(RTGI_GUIDE_PIONEER_GRID_DIV) + rotation;
        if (any(ref_pixel < 0) || any(ref_pixel >= int2(half_res_size))) { continue; }
        const float ref_depth = push.attach.view_cam_half_res_depth.get()[uint2(ref_pixel)];
        if (ref_depth == 0.0f) { continue; }
        const float4 hit_y = push.attach.pioneer_guide_hit_y.get()[uint2(cell)]; // .xyz hit position, .w brightness
        if (hit_y.w <= 0.0f) { continue; }
        // Same distance-only gate as the resolve (the cell's reference pixel vs this pixel).
        const float3 ref_offset_vs = own_ray_vs * (ref_depth - half_res_depth) + float3(float2(ref_pixel - int2(pixel_xy)) * ndc_per_pixel_vs, 0.0f) * ref_depth;
        const float ref_px_size = rtgi_half_res_pixel_width_ws(rcp(float2(half_res_size)), camera.near_plane, ref_depth);
        if (length(ref_offset_vs) > RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD * min(ws_px_size, ref_px_size)) { continue; }
        // Reconnect the cell's picked hit to THIS pixel.
        const float3 to_hit = hit_y.xyz - world_pos;
        const float dist = length(to_hit);
        if (dist < 1e-4f) { continue; }
        const float3 dir = to_hit / dist;
        if (dot(face_normal, dir) <= 0.0f) { continue; }
        return y_to_sh(hit_y.w, dir);
    }
    return float4(0.0f, 0.0f, 0.0f, 0.0f); // no valid cell: unguided
}

// This ray's guide, zero = unguided (plain cosine / GGX VNDF).
func rtgi_fetch_ray_guide_if_used(uint2 pixel_xy, RtgiTracePixel pixel, uint ray_index_in_signal, uint ray_key) -> float4
{
    return rtgi_ray_uses_guide(ray_index_in_signal)
        ? rtgi_fetch_ray_guide(pixel_xy, pixel.world_pos, pixel.depth, pixel.face_normal, pixel.ws_px_size, ray_key)
        : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// == Shading pipelines ("Trace RTGI": classic per-pixel trace, "Trace RTGI Pioneer": pioneer trace) ==========
// Rays are shaded in the closest hit / miss shaders (rtgi_shade.hlsl).

void rtgi_trace_and_shade(RayDesc ray, inout RtgiRayPayload payload)
{
    TraceRay(RaytracingAccelerationStructure::get(rtgi_trace_push.attach.tlas), 0, ~0, 0, 0, 0, ray, payload);
}

// Classic per-pixel trace ("Trace RTGI", use_repacked_ray_dispatch == 0): each pixel picks its own ray counts, allocates its slice
// of the ray list and traces + shades its rays in a loop. Produces the same outputs as the distribute pass + split
// trace + blend pass of the repacked path.
[shader("raygeneration")]
void ray_gen()
{
    const uint2 dtid = DispatchRaysIndex().xy;
    let clk_start = clockARB();
    let push = rtgi_trace_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    const CameraInfo camera = push.attach.globals.view_camera;

    const RtgiTracePixel pixel = rtgi_load_trace_pixel(dtid);
    const float3 primary_ray = normalize(pixel.world_pos - camera.position);

    if (push.debug_primary_trace)
    {
        RayDesc ray = {};
        ray.Origin = camera.position;
        ray.TMax = RTGI_RAY_T_MAX;
        ray.TMin = 0.0f;
        ray.Direction = primary_ray;

        RtgiRayPayload payload = {};
        rtgi_trace_and_shade(ray, payload);

        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid.xy, float4(payload.color, 2.0f), 2);
        return;
    }

    if (all(dtid.xy == uint2(0, 0)))
    {
        // Classic path: base rays are always traced, extras are capped at a quarter-res worth of rays.
        const uint total_geo = push.attach.ray_counters->total_geo_rays;
        push.attach.globals.readback.rtgi_requested_base_rays = total_geo;
        push.attach.globals.readback.rtgi_requested_extra_rays = push.attach.ray_counters->total_extra_rays;
        push.attach.globals.readback.rtgi_ray_budget = total_geo + (push.attach.globals.settings.render_target_size.x * push.attach.globals.settings.render_target_size.y) / 4;
    }

    // --- This pixel's ray counts (0 for sky): diffuse + specular from one budget ---
    uint samples = 0u;
    uint specular_samples = 0u;
    if (pixel.depth > 0.0f)
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

    // --- Wave-coalesced ray-list allocation: exclusive prefix sum of the ray counts within the wave, then ONE atomic
    // per wave to reserve that wave's contiguous slice of the global ray list (same ray_list / ray_result /
    // pixel_ray_alloc structures the repacked path uses). ---
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
    const uint ray_list_capacity = rtgi_ray_list_capacity(push.attach.globals.settings.render_target_size >> 1u);
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

    float  mean_ray_shortness = 0.0f;                     // mean ray shortness [0,1] over the rays; stored in .a
    float3 mean_perceptual_rgb = float3(0, 0, 0);          // geometric mean (mean log rgb) over the rays; stored in .rgb
    float3 mean_specular_perceptual_rgb = float3(0, 0, 0);
    float  mean_specular_hit = 0.0f;
    var clk_after_diffuse = clk_start; // splits the pixel's clocks into its diffuse and specular rays (debug)
    if (write_count > 0u)
    {
        const uint pixel_seed = rtgi_pixel_seed(dtid, 0u);
        rand_seed(pixel_seed);

        const float inv_samples = rcp(float(max(diffuse_write_count, 1u)));
        for (uint i = 0u; i < diffuse_write_count; ++i)
        {
            const float4 guide_sh_y = rtgi_fetch_ray_guide_if_used(dtid, pixel, i, i);
            if (i == 0u) { rtgi_guide_direction_debug_draw(dtid, guide_sh_y); }
            float guided_weight;
            const float3 sample_dir = rtgi_sample_diffuse_dir(pixel.face_normal, dtid * 2u, rtgi_stbn_frame(i), guide_sh_y, rtgi_ray_uses_guide(i), guided_weight);

            RtgiRayPayload payload = {};
            rtgi_trace_and_shade(rtgi_diffuse_ray_desc(pixel, primary_ray, sample_dir, RTGI_RAY_T_MAX), payload);

            // Write this ray's result into the shared ray list (same layout the split trace + shade produce), so the
            // pre-filter re-blends (and per-ray firefly-clamps) it identically to the repacked path.
            const float3 ray_rgb = payload.color * guided_weight * RTGI_RADIANCE_SCALE;
            push.attach.ray_result[my_offset + i] = RtgiRayResult(ray_rgb, payload.t, compress_normal_octahedral_32(sample_dir));

            mean_perceptual_rgb += linear_to_perceptual_rgb(ray_rgb, push.attach.globals.inv_exposure) * inv_samples;
            mean_ray_shortness  += calc_ray_shortness(payload.t, pixel.ws_px_size, rtgi_settings.max_visibility_pixel_range) * inv_samples;
        }

        clk_after_diffuse = clockARB();

        // Specular rays (own RNG stream) in the slots after the diffuse ones.
        const float inv_specular_samples = rcp(float(max(specular_write_count, 1u)));
        for (uint i = 0u; i < specular_write_count; ++i)
        {
            rtgi_specular_seed(pixel_seed, i);
            const float4 guide_sh_y = rtgi_fetch_ray_guide_if_used(dtid, pixel, i, i);
            float weight;
            const RayDesc ray = rtgi_specular_ray_desc(dtid, pixel, primary_ray, guide_sh_y, weight);
            RtgiRayResult result = RtgiRayResult(float3(0, 0, 0), 0.0f, compress_normal_octahedral_32(ray.Direction));
            if (weight > 0.0f)
            {
                RtgiRayPayload payload = {};
                payload.specular = true;
                rtgi_trace_and_shade(ray, payload);
                result.radiance = payload.color * weight * RTGI_RADIANCE_SCALE;
                result.t = min(payload.t, RTGI_SPECULAR_MAX_HIT_DISTANCE);
            }
            push.attach.ray_result[my_offset + diffuse_write_count + i] = result;
            mean_specular_perceptual_rgb += linear_to_perceptual_rgb(result.radiance, push.attach.globals.inv_exposure) * inv_specular_samples;
            mean_specular_hit += result.t * inv_specular_samples;
        }
    }

    // Same outputs the distribute + blend passes produce: per-pixel ray-list offset, ray counts, and the log-rgb /
    // shortness the pre-filter reads.
    push.attach.pixel_ray_alloc.get()[dtid.xy] = my_offset;
    push.attach.ray_count_image.get()[dtid.xy] = diffuse_write_count;
    push.attach.specular_ray_count_image.get()[dtid.xy] = specular_write_count;
    push.attach.perceptual_rgb_shortness.get()[dtid.xy] = float4(mean_perceptual_rgb, mean_ray_shortness);
    push.attach.specular_perceptual_rgb_hit.get()[dtid.xy] = float4(mean_specular_perceptual_rgb, mean_specular_hit);

    let trace_debug_mode = push.attach.globals.settings.debug_draw_mode;
    if (trace_debug_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_TRACE_CLOCKS || trace_debug_mode == DEBUG_DRAW_MODE_RTGI_SPECULAR_TRACE_CLOCKS)
    {
        let clk_end = clockARB();
        const uint clocks = trace_debug_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_TRACE_CLOCKS ? uint(clk_after_diffuse - clk_start) : uint(clk_end - clk_after_diffuse);
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(clocks * 0.0001f * push.attach.globals.settings.debug_visualization_scale), 1.0f + push.attach.globals.settings.debug_visualization_blend), 2);
    }
}

// Pioneer ray direction distribution: 1 = uniform hemisphere, 0 = cosine-weighted (see ray_gen_pioneer).
#define RTGI_GUIDE_PIONEER_UNIFORM_HEMISPHERE 1

// Pioneer trace ("Trace RTGI Pioneer"), gated by rtgi_settings.pioneer_guiding_enabled. Dispatched at a SPARSE grid --
// 1/RTGI_GUIDE_PIONEER_GRID_DIV resolution of the half-res trace grid in each axis -- one ray per cell.
// Which half-res pixel each cell samples ROTATES every frame (a simple mod-DIV cycle through the
// DIVxDIV sub-positions), so the sparse pattern isn't fixed; entry_guide_resample_horizontal/vertical
// (rtgi_guide_resample.hlsl) MUST use the exact same rotation formula to find each cell's sampled pixel
// back. This pass writes raw per-cell data only -- no averaging, no history -- rtgi_guide_resample.hlsl's
// separable resample is what actually produces the guide picks the main trace reads.
[shader("raygeneration")]
void ray_gen_pioneer()
{
    const uint2 pioneer_dtid = DispatchRaysIndex().xy;
    let push = rtgi_trace_push;
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

    // .w < 0 is the invalid/no-data sentinel entry_guide_resample skips as a RIS candidate -- brightness
    // Y is otherwise always >= 0, so it's unambiguous. Covers both the out-of-bounds edge cells (grid is
    // ceil-divided, so the rotated pixel can fall outside on the last row/column) and sky.
    const uint2 half_res_size = push.attach.globals.settings.render_target_size >> 1u;
    if (any(pixel_xy >= half_res_size))
    {
        push.attach.pioneer_hit_y.get()[pioneer_dtid] = float4(0.0f, 0.0f, 0.0f, -1.0f);
        return;
    }
    const RtgiTracePixel pixel = rtgi_load_trace_pixel(pixel_xy);
    if (pixel.depth == 0.0f)
    {
        push.attach.pioneer_hit_y.get()[pioneer_dtid] = float4(0.0f, 0.0f, 0.0f, -1.0f);
        return;
    }
    const float3 primary_ray = normalize(pixel.world_pos - push.attach.globals.view_camera.position);

    // Unguided sampling -- pioneer rays ARE the raw signal being gathered, so they must not guide off anything
    // themselves. RTGI_GUIDE_PIONEER_UNIFORM_HEMISPHERE (default 1): uniform over the hemisphere, so the search for
    // bright spots covers grazing directions as densely as the pole (they also feed the rough specular guide mix);
    // 0 = cosine-weighted (density matches diffuse importance). The guide resample weights candidates by brightness
    // only (no sampling pdf), so either distribution plugs in unchanged.
    rand_seed(rtgi_pixel_seed(pixel_xy, 0u));
    const float3 world_tangent = normalize(cross(pixel.face_normal, float3(0, 0, 1) + 0.0001f));
    const float3x3 tbn = transpose(float3x3(world_tangent, cross(world_tangent, pixel.face_normal), pixel.face_normal));
#if RTGI_GUIDE_PIONEER_UNIFORM_HEMISPHERE
    // Uniform hemisphere: cos(theta) uniform in [0, 1).
    const float cos_theta = rand();
    const float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    const float phi = rand() * 2.0f * PI;
    const float3 local_dir = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
#else
    const float3 local_dir = rand_cosine_sample_hemi();
#endif

    // Bounded (unlike the main trace's effectively-unbounded rays) -- the pioneer trace only needs nearby
    // bounce lighting to build a useful direction guide; see the field comment in rtgi.inl.
    const RayDesc ray = rtgi_diffuse_ray_desc(pixel, primary_ray, mul(tbn, local_dir), rtgi_settings.guide_pioneer_trace_max_distance);
    RtgiRayPayload payload = {};
    rtgi_trace_and_shade(ray, payload);

    // Store a RECONNECTION payload, not a baked SH-Y direction: the ray's actual hit position X (world space) plus the
    // direction-independent brightness Y measured there. The hit is shaded diffuse (Lambertian), so its exitant
    // radiance is the SAME in every exit direction -- Y needs no per-receiver re-derivation. What changes per receiver
    // is only the DIRECTION from the receiver to X; rtgi_fetch_ray_guide computes it fresh for whichever pixel consumes
    // the guide, the resample passes carry (X, Y) forward unchanged.
    // A sky miss reports t = RTGI_RAY_MISS_T, WAY past this ray's TMax: clamp to TMax before building the position.
    // (1) pioneer_hit_y is R16G16B16A16_SFLOAT (max ~65504): origin + direction * 1e12 overflows to +-Inf, which turns
    // into a NaN reconnection direction downstream that slips past the hemisphere test (NaN compares false) -- exactly
    // the bright sky-miss guides would silently lose their direction. (2) TMax is still "far enough": the parallax of
    // nearby receivers against a point that far along the ray is negligible, reconnection still reproduces ~the ray.
    const float3 hit_position = ray.Origin + ray.Direction * min(payload.t, ray.TMax);
    const float Y = linear_to_y_co_cg(payload.color).x;
    // max(...,0) keeps Y unambiguous against the negative invalid sentinel above.
    push.attach.pioneer_hit_y.get()[pioneer_dtid] = float4(hit_position, max(Y, 0.0f));
}

[shader("anyhit")]
void any_hit(inout RtgiRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    if (!rtgi_any_hit_is_alpha_hit(attr.barycentrics)) { IgnoreHit(); }
}

// == Split trace ("Trace RTGI Split", repacked ray-list path) ===================================================
// Own ray tracing pipeline (rtgi_trace_split_compile_info) whose hit shaders only record the hit and classify it, so
// the pipeline's register footprint is only the ray setup + traversal (+ alpha test any hit). Shading runs afterwards
// in rtgi_shade.hlsl, one compute pass per shade list (sharp / coarse / miss). Everything the raygen still needs after
// TraceRay rides in the payload (only ray_index stays live across the trace): the ray result's weight + direction are
// written before it.

// One ray of the flat ray list, ready to trace.
struct RtgiListRay
{
    RayDesc ray;
    float weight;       // sampling correction (guided diffuse / specular MIS); 1 = none
    uint packed_dir;    // octahedral ray direction (ray-list result)
    bool specular;
    bool sky;           // the pixel has no geometry: nothing to trace, result is zero
    float roughness;    // specular rays: GGX roughness of the sampled lobe
};

// Builds ray-list entry ray_index: pixel, RNG slot, guide fetch, direction and origin.
func rtgi_setup_list_ray(uint ray_index) -> RtgiListRay
{
    let push = rtgi_trace_push;
    RtgiListRay ret = {};
    ret.weight = 1.0f;

    const RtgiRayEntry entry = push.attach.ray_list[ray_index];
    const uint2 pixel_xy     = uint2(entry.packed_xy & 0xFFFFu, entry.packed_xy >> 16u);
    const uint  sample_index = entry.sample_index;

    const RtgiTracePixel pixel = rtgi_load_trace_pixel(pixel_xy);
    if (pixel.depth == 0.0f)
    {
        ret.sky = true;
        return ret;
    }

    // ORDER MATTERS for register pressure (the split trace's footprint): the guide fetch loop runs FIRST, everything
    // only needed after it (primary ray, seeds, tbn) is computed after it instead of being kept live through it.
    // A pixel's entries are [diffuse 0..n_d) then [specular 0..n_s); the diffuse count says which this one is.
    const uint diffuse_ray_count = push.attach.ray_count_image.get()[pixel_xy];
    ret.specular = sample_index >= diffuse_ray_count;
    const uint ray_index_in_signal = ret.specular ? sample_index - diffuse_ray_count : sample_index;
    const float4 guide_sh_y = rtgi_fetch_ray_guide_if_used(pixel_xy, pixel, ray_index_in_signal, sample_index);

    const float3 primary_ray = normalize(pixel.world_pos - push.attach.globals.view_camera.position);
    // Fold the reprojected history length into the seed. frame_index alone can alias across frames, so a pixel that
    // shot N rays one frame could re-draw near-identical directions the next. The history count advances by the
    // pixel's rays-shot each frame (fastest exactly when many rays/frame make repeats most likely), giving an extra
    // decorrelating dimension. Gated by animate_noise so frozen-noise stays frozen.
    const float history_count = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[pixel_xy]);
    const uint history_seed = push.attach.globals.rtgi_settings.animate_noise ? uint(max(history_count, 0.0f)) * RTGI_SEED_HISTORY : 0u;
    const uint pixel_seed = rtgi_pixel_seed(pixel_xy, history_seed);

    // Seed ONCE per pixel+frame (NOT per ray) and advance the sequence to this ray's slot: re-seeding per ray with
    // base + sample_index * prime does NOT decorrelate (a single PCG step barely mixes nearby seeds), which made a
    // pixel's N rays near-duplicates. Mirrors how the classic per-pixel trace draws sequentially across its rays.
    if (ret.specular)
    {
        rtgi_specular_seed(pixel_seed, ray_index_in_signal);
        ret.ray = rtgi_specular_ray_desc(pixel_xy, pixel, primary_ray, guide_sh_y, ret.weight);
        ret.roughness = unpack_normal_roughness_roughness(push.attach.view_cam_half_res_normal_roughness.get()[pixel_xy]);
        ret.packed_dir = compress_normal_octahedral_32(ret.ray.Direction);
        return ret;
    }

    rand_seed(pixel_seed);
    [loop] for (uint skip = 0u; skip < sample_index * 2u; ++skip) { rand(); }
    if (sample_index == 0u) { rtgi_guide_direction_debug_draw(pixel_xy, guide_sh_y); }
    const float3 sample_dir = rtgi_sample_diffuse_dir(pixel.face_normal, pixel_xy * 2u, rtgi_stbn_frame(sample_index), guide_sh_y, rtgi_ray_uses_guide(ray_index_in_signal), ret.weight);
    ret.ray = rtgi_diffuse_ray_desc(pixel, primary_ray, sample_dir, RTGI_RAY_T_MAX);
    ret.packed_dir = compress_normal_octahedral_32(sample_dir);
    return ret;
}

struct RtgiSplitPayload
{
    float t;              // in: miss t (stored as-is on a miss), out: hit t clamped to the in value
    uint instance_id;     // RTGI_RAY_HIT_MISS = miss (pre-set by the raygen, overwritten on hit)
    uint primitive_index;
    float2 barycentrics;
    // in: RTGI_SPLIT_DIFFUSE_RAY for diffuse rays, else the specular cone width per hit distance (2 * lobe tan,
    // 0 = always sharp). out (hits only): 1 = sharp, 0 = coarse.
    float sharp;
};
#define RTGI_SPLIT_DIFFUSE_RAY -1.0f

// Coarse specular: the GGX cone footprint at the hit must cover at least this many PGI probe spacings.
#define RTGI_SPLIT_COARSE_SPECULAR_FOOTPRINT_PROBES 1.0f

// Sharp hits get full material + light shading, coarse hits only the PGI radiance. Matches the full shading
// pipeline's closest hit: specular rays need sharp hits (mirror reflections), diffuse rays take the PGI radiance cache
// shortcut. With split_coarse_wide_specular, a specular hit whose GGX cone is already as wide at the hit as the PGI
// probe grid there (the spatial resolution PGI radiance is projected at) is shaded coarse too: full shading would
// resolve detail the lobe blurs away anyway. Called by the closest hit (ray origin / direction / t are free there
// instead of being kept live across TraceRay in the raygen).
func rtgi_split_hit_is_sharp(float cone_per_t, float3 hit_position, float t) -> bool
{
    if (cone_per_t == RTGI_SPLIT_DIFFUSE_RAY) { return false; }
    if (cone_per_t <= 0.0f) { return true; }
    let push = rtgi_trace_push;
    // By POINTER: the cascade lookup below indexes dynamically. On a local copy of PGISettings that pulls all
    // PGI_MAX_CASCADES cascades into registers (measured 255 + spills); through the pointer it is one load.
    PGISettings* pgi = &push.attach.globals.pgi_settings;
    const float cascade = ceil(pgi_select_cascade_smooth_spherical(*pgi, hit_position - push.attach.globals.view_camera.position));
    if (cascade >= float(pgi->cascade_count)) { return true; } // outside PGI: coarse shading would be black
    const float3 spacing = pgi->cascades[uint(cascade)].probe_spacing;
    const float probe_spacing = max(spacing.x, max(spacing.y, spacing.z));
    return cone_per_t * t < probe_spacing * RTGI_SPLIT_COARSE_SPECULAR_FOOTPRINT_PROBES;
}

// Appends this ray to its shade list (RTGI_SHADE_*, see RtgiShadeCounters). Per-thread atomics on purpose: a
// wave-coalesced append (first lane atomic + WaveReadLaneFirst broadcast) is NOT safe after TraceRay, ray tracing
// stages do not guarantee the lanes reconverge after the first-lane branch, so lanes broadcast a wrong base, write
// wrong slots and leave other slots holding ray indices of earlier frames (stale hit records -> page faults).
void rtgi_split_append_shade(uint ray_index, uint shade_class)
{
    let push = rtgi_trace_push;
    RtgiShadeCounters* counters = push.attach.shade_counters;
    uint k = 0u;
    DispatchIndirectStruct* dispatch;
    switch (shade_class)
    {
        case RTGI_SHADE_SHARP: InterlockedAdd(counters->sharp_count, 1u, k);  dispatch = &counters->sharp_dispatch;  break;
        case RTGI_SHADE_MISS:  InterlockedAdd(counters->miss_count, 1u, k);   dispatch = &counters->miss_dispatch;   break;
        default:               InterlockedAdd(counters->coarse_count, 1u, k); dispatch = &counters->coarse_dispatch; break;
    }
    const uint list_capacity = rtgi_ray_list_capacity(push.attach.globals.settings.render_target_size >> 1u);
    push.attach.shade_list[rtgi_shade_list_slot(shade_class, k, list_capacity)] = ray_index;
    // The entry that opens a new shade group grows the indirect dispatch.
    if ((k % RTGI_SHADE_HITS_X) == 0u)
    {
        const uint groups = k / RTGI_SHADE_HITS_X + 1u;
        InterlockedMax(dispatch->x, min(groups, RTGI_SHADE_DISPATCH_X_MAX));
        InterlockedMax(dispatch->y, (groups + RTGI_SHADE_DISPATCH_X_MAX - 1u) / RTGI_SHADE_DISPATCH_X_MAX);
    }
}

[shader("raygeneration")]
void ray_gen_split()
{
    let push = rtgi_trace_push;
    const uint ray_index = DispatchRaysIndex().z * 128u + DispatchRaysIndex().x;

    if (ray_index == 0u)
    {
        // The counters are cleared to 0 every frame; x / y grow by atomic max in rtgi_split_append_shade.
        push.attach.shade_counters->sharp_dispatch.z = 1u;
        push.attach.shade_counters->coarse_dispatch.z = 1u;
        push.attach.shade_counters->miss_dispatch.z = 1u;
    }

    if (ray_index >= min(push.attach.ray_counters->ray_list_count, rtgi_ray_list_limit(push.attach.globals.settings.render_target_size >> 1u, push.attach.globals.rtgi_settings)))
        return;

    const RtgiListRay list_ray = rtgi_setup_list_ray(ray_index);
    if (list_ray.sky)
    {
        push.attach.ray_result[ray_index] = RtgiRayResult(float3(0.0f, 0.0f, 0.0f), 0.0f, 0u);
        return;
    }
    if (list_ray.specular && list_ray.weight <= 0.0f)
    {
        push.attach.ray_result[ray_index] = RtgiRayResult(float3(0, 0, 0), 0.0f, list_ray.packed_dir);
        return;
    }

    // Written now so weight / direction are not live across the trace; .t is filled in after it. The shade pass
    // replaces .radiance with the shaded value times this weight.
    push.attach.ray_result[ray_index] = RtgiRayResult(float3(list_ray.weight, 0.0f, 0.0f), 0.0f, list_ray.packed_dir);

    RtgiSplitPayload payload = {};
    // A miss leaves the payload untouched: same t as the shading pipelines' miss shader (specular clamped like there).
    payload.t = list_ray.specular ? RTGI_SPECULAR_MAX_HIT_DISTANCE : RTGI_RAY_MISS_T;
    payload.instance_id = RTGI_RAY_HIT_MISS;
    const bool coarse_wide_specular = push.attach.globals.rtgi_settings.split_coarse_wide_specular != 0 && push.attach.globals.pgi_settings.enabled;
    payload.sharp = !list_ray.specular ? RTGI_SPLIT_DIFFUSE_RAY : (coarse_wide_specular ? 2.0f * rtgi_specular_lobe_tan(list_ray.roughness) : 0.0f);
    TraceRay(RaytracingAccelerationStructure::get(push.attach.tlas), 0, ~0, 0, 0, 0, list_ray.ray, payload);

    push.attach.ray_result[ray_index].t = payload.t;
    const bool miss = payload.instance_id == RTGI_RAY_HIT_MISS;
    if (!miss)
    {
        push.attach.ray_hits[ray_index] = RtgiRayHit(payload.instance_id, payload.primitive_index, payload.barycentrics);
    }
    rtgi_split_append_shade(ray_index, miss ? RTGI_SHADE_MISS : (payload.sharp > 0.5f ? RTGI_SHADE_SHARP : RTGI_SHADE_COARSE));
}

[shader("closesthit")]
void closest_hit_split(inout RtgiSplitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    const float t = RayTCurrent();
    payload.sharp = rtgi_split_hit_is_sharp(payload.sharp, WorldRayOrigin() + WorldRayDirection() * t, t) ? 1.0f : 0.0f;
    payload.t = min(t, payload.t);
    payload.instance_id = InstanceID();
    payload.primitive_index = PrimitiveIndex();
    payload.barycentrics = attr.barycentrics;
}

[shader("miss")]
void miss_split(inout RtgiSplitPayload payload)
{
    // The payload is pre-initialized to a miss by ray_gen_split.
}

[shader("anyhit")]
void any_hit_split(inout RtgiSplitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    if (!rtgi_any_hit_is_alpha_hit(attr.barycentrics)) { IgnoreHit(); }
}
