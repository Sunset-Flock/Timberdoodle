#pragma once

#include "rt_lab_rays.inl"

#include "rt_lab_shared.hlsl"
#include "rt_lab_reproject.hlsl"
#include "shader_lib/debug.glsl"

// == Demand ===========================================================================================================

[[vk::push_constant]] RtLabDemandPush rt_lab_demand_push;

// Variance laziness (RtLabSettings::variance_laziness_*): factor on the reprojected history length = convergence target /
// samples needed for the perceptual target error, >= 1. Per sample perceptual variance = luma variance / (mean + floor)^2
// (log space derivative), from last frame's variance guide estimate and history mean at the nearest previous pixel (the
// history length already carries the surface test).
func rt_lab_variance_laziness(Texture2D<float4> history, Texture2D<float4> variance, CameraInfo* prev_camera, RtLabSettings settings,
    uint2 size, float3 position, float exposure) -> float
{
    const float perceptual_variance = rt_lab_previous_perceptual_variance(history, variance, prev_camera, settings, size, position, exposure);
    if (perceptual_variance < 0.0f)
    {
        return 1.0f;
    }
    const float target = max(settings.fast_convergence_samples, 1.0f);
    const float needed = clamp(perceptual_variance / square(max(settings.variance_laziness_target_error, 1e-3f)),
        min(max(settings.variance_laziness_min_samples, 1.0f), target), target);
    return target / needed;
}

// Writes the lab geometry, then the pixel's desired ray count (RTGI curve: calc_desired_extra_rays over the reprojected
// history length, amplitude = the extras up to max_rays_per_pixel) and totals the demand.
[shader("compute")]
[numthreads(RT_LAB_RAYS_X, RT_LAB_RAYS_Y, 1)]
func entry_rt_lab_demand(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_demand_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;

    const bool half_res = push.size.x != push.attach.globals.settings.render_target_size.x;
    uint packed_normal;
    const float depth = rt_lab_load_geometry(push.attach.view_cam_depth.get(), push.attach.view_cam_face_normals.get(),
        push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get(), camera, half_res, dtid, packed_normal);
    // Half res: tag which full res sub-pixel the representative depth came from (rt_lab_ray_launch starts rays there).
    uint subpixel = 0u;
    if (half_res && depth > 0.0f)
    {
        float best_difference = 1e30f;
        const uint2 full_size = push.attach.globals.settings.render_target_size;
        for (uint i = 0u; i < 4u; ++i)
        {
            const uint2 full_res_pixel = min(dtid * 2u + uint2(i & 1u, i >> 1u), full_size - 1u);
            const float ndc_depth = push.attach.view_cam_depth.get()[full_res_pixel];
            if (ndc_depth > 0.0f)
            {
                const float difference = abs(camera.near_plane / ndc_depth - depth);
                if (difference < best_difference)
                {
                    best_difference = difference;
                    subpixel = i;
                }
            }
        }
    }
    push.attach.lab_depth.get()[dtid] = rt_lab_tag_subpixel(depth, subpixel);
    const float2 size = float2(push.size);
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(size)), depth);
    const float3 normal = uncompress_normal_octahedral_32(packed_normal);
    const float rcp_pixel_width_ws = depth > 0.0f ? rcp(rt_lab_pixel_width_ws(camera, size.y, depth)) : 0.0f;
    // Temporal gradient: how much of the reprojected history survives (1 - lambda); cap mode: unscaled lambda.
    const bool gradient_cap = push.gradient_bound != 0u && settings.temporal_gradient_cap_mode != 0;
    // temporal_gradient_pixel_relative: made relative to this pixel's own previous history luminance.
    float gradient_lambda = 0.0f;
    if (push.gradient_bound != 0u && depth > 0.0f)
    {
        const bool packed = rt_lab_gradient_packed(settings);
        const float4 cells = rt_lab_temporal_gradient_full(push.attach.temporal_gradient.get(), camera, push.size, dtid, position, normal, rcp_pixel_width_ws, packed);
        float2 gradient = cells.xz;
        if (packed)
        {
            gradient = push.history_bound != 0u
                ? rt_lab_gradient_pixel_relative(settings, cells, push.attach.history.get(), &push.attach.globals.view_camera_prev_frame,
                      push.size, position, push.attach.globals.exposure)
                : float2(0.0f, 0.0f);
        }
        const bool kalman = (settings.temporal_gradient_cap_mode & RT_LAB_GRADIENT_CAP_KALMAN) != 0;
        // Kalman: the information loss L, unscaled, no deadzone (rt_lab_gradient_history_cap).
        gradient_lambda = kalman ? max(gradient.x, 0.0f) : saturate(rt_lab_gradient_deadzone(settings, gradient.x) * (gradient_cap ? 1.0f : settings.temporal_gradient_scale));
    }
    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RT_LAB_TEMPORAL_GRADIENT)
    {
        // lambda: 0 = history kept (cold), 1 = history dropped (hot); magenta = gradient off.
        const uint resolution_scale = max(push.attach.globals.settings.render_target_size.x / push.size.x, 1u);
        const float3 color = depth == 0.0f ? float3(0.0f, 0.0f, 0.0f) : (push.gradient_bound != 0u ? Heatmap(gradient_lambda) : float3(1.0f, 0.0f, 1.0f) * 0.25f);
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
            float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend), resolution_scale);
    }
    push.attach.lab_face_normal.get()[dtid] = packed_normal;

    uint desired = 0u;
    if (depth != 0.0f)
    {
        desired = 1u;
        if (settings.redistribution_enabled != 0)
        {
            // Exactly the history length the temporal pass reprojects (occluder aware soft normalize + temporal gradient).
            float history_length = push.history_bound != 0u
                ? rt_lab_reproject_history_length(push.attach.history.get(), push.attach.lab_depth_history.get(), push.attach.lab_face_normal_history.get(),
                      &push.attach.globals.view_camera_prev_frame, &push.attach.globals.rt_lab_settings, int2(push.size), position, normal, rcp_pixel_width_ws)
                : 0.0f;
            // Variance laziness BEFORE the gradient cut: a cut history still asks for the full extras.
            if (settings.variance_laziness_enabled != 0 && push.history_bound != 0u && history_length > 0.0f)
            {
                history_length *= rt_lab_variance_laziness(push.attach.history.get(), push.attach.variance_history.get(),
                    &push.attach.globals.view_camera_prev_frame, settings, push.size, position, push.attach.globals.exposure);
            }
            if (gradient_cap)
            {
                // Same cap as the temporal pass (same inputs), after the laziness: a capped history still asks for extras.
                if (push.history_bound != 0u && history_length > 0.0f)
                {
                    const float perceptual_variance = rt_lab_previous_perceptual_variance(push.attach.history.get(), push.attach.variance_history.get(),
                        &push.attach.globals.view_camera_prev_frame, settings, push.size, position, push.attach.globals.exposure);
                    history_length = min(history_length, rt_lab_gradient_history_cap(settings, gradient_lambda, perceptual_variance, history_length));
                }
            }
            else
            {
                history_length *= 1.0f - gradient_lambda;
            }
            const uint max_rays = uint(max(settings.max_rays_per_pixel, 1));
            const float extra = calc_desired_extra_rays(history_length, settings.fast_convergence_samples, float(max_rays - 1u));
            desired = min(1u + uint(extra + 0.5f), max_rays);
        }
    }
    push.attach.ray_demand.get()[dtid] = desired;

    const uint wave_base = WaveActiveSum(desired > 0u ? 1u : 0u);
    const uint wave_extra = WaveActiveSum(desired > 0u ? desired - 1u : 0u);
    if (WaveIsFirstLane())
    {
        InterlockedAdd(push.attach.ray_counters->base_demand, wave_base);
        InterlockedAdd(push.attach.ray_counters->extra_demand, wave_extra);
    }
}

// == Distribute =======================================================================================================

[[vk::push_constant]] RtLabDistributePush rt_lab_distribute_push;

// 8x8 ordered dither (Bayer) rank in [0, 63], finest level in the high bits: taking the lowest ranks picks a spatially
// even subset (checkerboard at 32). Same as the RTGI distribute pass.
func rt_lab_bayer_m2(uint a, uint b) -> uint
{
    return 2u * ((a ^ b) & 1u) + (a & 1u);
}
// Base ray gradient records per 8x8 tile (temporal_gradient_all_base_rays): min_ray_budget of the tile's pixels. Same
// formula in C++ (rt_lab.cpp, record image size).
func rt_lab_gradient_base_slots(RtLabSettings settings) -> uint
{
    return clamp(uint(round(saturate(settings.min_ray_budget) * 64.0f)), 1u, 64u);
}

func rt_lab_bayer_8x8(uint2 p) -> uint
{
    return 16u * rt_lab_bayer_m2(p.x, p.y) + 4u * rt_lab_bayer_m2(p.x >> 1u, p.y >> 1u) + rt_lab_bayer_m2(p.x >> 2u, p.y >> 2u);
}

#define RT_LAB_TILE_PIXELS (RT_LAB_RAYS_X * RT_LAB_RAYS_Y)

// Per frame rotation of the base ray Bayer rank order (distribute; the gradient's base ray records use the same).
func rt_lab_base_rank_rotation(RtLabSettings settings, float trunk_flt_frame_index) -> uint
{
    const uint frame = settings.animate_noise ? uint(trunk_flt_frame_index) : 0u;
    return uint(frac(float(frame) * 0.61803398875f) * float(RT_LAB_TILE_PIXELS));
}
groupshared uint gs_desired[RT_LAB_TILE_PIXELS];   // per tile pixel (row major), 0 = sky / out of bounds
groupshared uint gs_rays[RT_LAB_TILE_PIXELS];      // allocated rays
groupshared uint gs_offset[RT_LAB_TILE_PIXELS];    // ray list offset
groupshared uint gs_pixel_by_rank[RT_LAB_TILE_PIXELS]; // inverse Bayer: tile pixel of each rank

// Tile based budget drain (RTGI rule, one workgroup = one 8x8 tile). Enough budget: every pixel gets its demand. Else
// global fractions: the base request is min_ray_budget of the geometry pixels, paid first; the extras get the rest at one
// ratio of their demand; what the extras leave raises the base coverage (up to every pixel). Each tile then gets
// floor(fraction x its own demand) of both, so the frame never exceeds the budget and no ray is ever dropped.
// Within the tile (thread 0, serial over the 64 pixels):
//   base:  the tile's base rays go to its geometry pixels in Bayer rank order, the rank rotated per frame -> spatially
//          even, exact count.
//   extra: cumulative scan over the extra demand (linear priority, freshest pixels get the most), phase rotated per
//          frame and tile, each pixel capped at its own extra demand.
// The tile's rays are reserved in the ray list with one atomic.
[shader("compute")]
[numthreads(RT_LAB_RAYS_X, RT_LAB_RAYS_Y, 1)]
func entry_rt_lab_distribute(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rt_lab_distribute_push;
    let settings = push.attach.globals.rt_lab_settings;
    const uint slot = gtid.y * RT_LAB_RAYS_X + gtid.x;
    const bool in_bounds = all(dtid < push.size);
    gs_desired[slot] = in_bounds ? push.attach.ray_demand.get()[dtid] : 0u;
    gs_pixel_by_rank[rt_lab_bayer_8x8(gtid)] = slot;
    GroupMemoryBarrierWithGroupSync();

    if (slot == 0u)
    {
        const uint budget = rt_lab_ray_budget(push.size, settings, push.ray_capacity);
        const uint base_demand = push.attach.ray_counters->base_demand;
        const uint extra_demand = push.attach.ray_counters->extra_demand;
        uint tile_total = 0u;
        if (base_demand + extra_demand <= budget)
        {
            for (uint i = 0u; i < RT_LAB_TILE_PIXELS; ++i)
            {
                gs_rays[i] = gs_desired[i];
            }
        }
        else
        {
            // Global fractions (identical in every tile).
            const float min_budget = saturate(settings.min_ray_budget);
            const float min_base_spend = min(float(base_demand) * min_budget, float(budget));
            const float extra_ratio = extra_demand > 0u ? saturate((float(budget) - min_base_spend) / float(extra_demand)) : 0.0f;
            const float base_fraction = base_demand > 0u
                ? min(min(max((float(budget) - float(extra_demand) * extra_ratio) / float(base_demand), min_budget), 1.0f), float(budget) / float(base_demand))
                : 0.0f;

            uint tile_base_demand = 0u;
            uint tile_extra_demand = 0u;
            for (uint i = 0u; i < RT_LAB_TILE_PIXELS; ++i)
            {
                tile_base_demand += gs_desired[i] > 0u ? 1u : 0u;
                tile_extra_demand += gs_desired[i] > 0u ? gs_desired[i] - 1u : 0u;
                gs_rays[i] = 0u;
            }
            const uint tile_base_budget = min(uint(float(tile_base_demand) * base_fraction), tile_base_demand);
            const uint tile_extra_budget = min(uint(float(tile_extra_demand) * extra_ratio), tile_extra_demand);

            // Base: geometry pixels in rotated Bayer rank order until the tile's base budget is used up.
            const uint frame = settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
            const uint rank_rotation = rt_lab_base_rank_rotation(settings, push.attach.globals.trunk_flt_frame_index);
            uint base_given = 0u;
            for (uint r = 0u; r < RT_LAB_TILE_PIXELS && base_given < tile_base_budget; ++r)
            {
                const uint i = gs_pixel_by_rank[(r + rank_rotation) % RT_LAB_TILE_PIXELS];
                if (gs_desired[i] > 0u)
                {
                    gs_rays[i] = 1u;
                    ++base_given;
                }
            }
            // Extras: cumulative scan, target(prefix) = floor(prefix / total x budget + phase), differenced.
            const uint tile_id = gid.y * ((push.size.x + RT_LAB_RAYS_X - 1u) / RT_LAB_RAYS_X) + gid.x;
            const float phase = frac(float(frame) * 0.61803398875f + float(tile_id) * 0.7548776662f + 0.5f);
            const float inv_extra_demand = tile_extra_demand > 0u ? rcp(float(tile_extra_demand)) : 0.0f;
            uint accumulated_demand = 0u;
            uint extra_given = 0u;
            for (uint i = 0u; i < RT_LAB_TILE_PIXELS; ++i)
            {
                const uint extra_i = gs_desired[i] > 0u ? gs_desired[i] - 1u : 0u;
                accumulated_demand += extra_i;
                const uint target = tile_extra_demand > 0u
                    ? min(uint(saturate(float(accumulated_demand) * inv_extra_demand) * float(tile_extra_budget) + phase), tile_extra_budget)
                    : 0u;
                const uint extra = min(target - min(extra_given, target), extra_i);
                extra_given = max(extra_given, target);
                gs_rays[i] += extra;
            }
        }
        for (uint i = 0u; i < RT_LAB_TILE_PIXELS; ++i)
        {
            gs_offset[i] = tile_total;
            tile_total += gs_rays[i];
        }
        uint tile_offset = 0u;
        InterlockedAdd(push.attach.ray_counters->ray_count, tile_total, tile_offset);
        for (uint i = 0u; i < RT_LAB_TILE_PIXELS; ++i)
        {
            gs_offset[i] += tile_offset;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // The budget math never exceeds the budget; the clamp only guards the buffer.
    const uint budget = rt_lab_ray_budget(push.size, settings, push.ray_capacity);

    if (!in_bounds)
    {
        return;
    }
    const uint offset = gs_offset[slot];
    const uint count = offset < budget ? min(gs_rays[slot], budget - offset) : 0u;
    push.attach.ray_alloc.get()[dtid] = uint2(offset, count);
    const uint packed_pixel = dtid.x | (dtid.y << 16u);
    for (uint k = 0u; k < count; ++k)
    {
        push.attach.ray_list[offset + k] = packed_pixel;
    }

    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    if (debug_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_RAYS_SHOT || debug_mode == DEBUG_DRAW_MODE_RTGI_TOTAL_RAYS_SHOT)
    {
        // Rays per pixel, full heat = max_rays_per_pixel.
        const uint resolution_scale = max(push.attach.globals.settings.render_target_size.x / push.size.x, 1u);
        const float3 color = gs_desired[slot] == 0u ? float3(0.0f, 0.0f, 0.0f) : Heatmap(float(count) / float(max(settings.max_rays_per_pixel, 1)));
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
            float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend), resolution_scale);
    }
}

// == Temporal gradient blur ==========================================================================================

[[vk::push_constant]] RtLabGradientBlurPush rt_lab_gradient_blur_push;

// Unpacks one gradient sample (rt_lab_trace.hlsl): .x signed luminance change, .y max(old, new), both exposed. (0, 0) = none.
func rt_lab_unpack_gradient_sample(uint packed) -> float2
{
    return float2(f16tof32(packed & 0xFFFFu), f16tof32(packed >> 16u));
}

// Representative geometry of a gradient cell (rt_lab_gradient_cell_pixel) from the g-buffer. Returns the linear depth
// (0 = sky).
func rt_lab_gradient_cell_geometry(RtLabGradientBlurPush push, CameraInfo camera, uint2 cell, out float3 position, out uint packed_normal) -> float
{
    const bool half_res = push.lab_size.x != push.attach.globals.settings.render_target_size.x;
    return rt_lab_cell_geometry(push.attach.view_cam_depth.get(), push.attach.view_cam_face_normals.get(),
        push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get(), camera, push.lab_size, half_res, cell, position, packed_normal);
}

// Last frame's gradient history (gradient_history, vertical pass) of the cell whose representative point is `position`:
// nearest previous cell, same surface by a plane test (RT_LAB_GRADIENT_HISTORY_PLANE_TOLERANCE cell widths).
func rt_lab_gradient_previous_cell(RtLabGradientBlurPush push, float3 position, float3 normal, float rcp_pixel_width_ws, out float4 previous) -> bool
{
    previous = float4(0.0f, 0.0f, 0.0f, 0.0f);
    CameraInfo* prev_camera = &push.attach.globals.view_camera_prev_frame;
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    if (!(prev_clip.w > 0.0f) || any(prev_uv <= 0.0f) || any(prev_uv >= 1.0f))
    {
        return false;
    }
    const uint2 prev_cell = min(uint2(prev_uv * float2(push.lab_size)) / RT_LAB_GRADIENT_CELL, push.size - 1u);
    previous = push.attach.gradient_history.get()[prev_cell];
    if (!(previous.w > 0.0f) || !all(isfinite(previous)))
    {
        return false;
    }
    const float3 prev_position = linear_depth_to_world_space(*prev_camera,
        rt_lab_ndc(rt_lab_gradient_cell_pixel(prev_cell, push.lab_size), rcp(float2(push.lab_size))), previous.w);
    const float plane_distance = abs(dot(prev_position - position, normal)) * rcp_pixel_width_ws;
    return plane_distance <= RT_LAB_GRADIENT_HISTORY_PLANE_TOLERANCE * float(RT_LAB_GRADIENT_CELL);
}

[shader("compute")]
[numthreads(RT_LAB_RAYS_X, RT_LAB_RAYS_Y, 1)]
func entry_rt_lab_gradient_blur(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_gradient_blur_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const bool vertical = push.vertical != 0u;

    // Center cell geometry. Sky: no lambda, and the upscale skips the cell.
    float3 center_position;
    uint center_packed_normal;
    const float center_depth = rt_lab_gradient_cell_geometry(push, camera, dtid, center_position, center_packed_normal);
    if (center_depth == 0.0f)
    {
        if (vertical)
        {
            push.attach.temporal_gradient.get()[dtid] = uint4(asuint(0.0f), asuint(0.0f), 0u, 0u);
            push.attach.gradient_history_dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else
        {
            push.attach.blur_sums_dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            push.attach.blur_moments_dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }
    const float3 center_normal = uncompress_normal_octahedral_32(center_packed_normal);
    // Distance weight in lab pixel widths at the center's depth (rt_lab_gradient_distance_weight).
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.lab_size.y), center_depth));

    const int radius = clamp(settings.temporal_gradient_blur_radius, 0, RT_LAB_GRADIENT_MAX_BLUR_RADIUS);
    const float sigma = max(float(radius) * 0.5f, 0.5f);
    const float rcp_two_sigma_sq = rcp(2.0f * sigma * sigma);
    const int2 direction = vertical ? int2(0, 1) : int2(1, 0);
    float4 sums = float4(0.0f, 0.0f, 0.0f, 0.0f); // .x sum w delta, .y sum w max, .z sum w samples, .w sum w old; cells without a sample add 0
    float3 moments = float3(0.0f, 0.0f, 0.0f);   // sum w^2 delta^2, sum w^2 delta old, sum w^2 old^2 (Kalman)
    for (int d = -radius; d <= radius; ++d)
    {
        const int2 cell = int2(dtid) + direction * d;
        if (any(cell < 0) || any(cell >= int2(push.size)))
        {
            continue;
        }
        float weight = exp(-float(d * d) * rcp_two_sigma_sq);
        if (d != 0)
        {
            float3 tap_position;
            uint tap_packed_normal;
            if (rt_lab_gradient_cell_geometry(push, camera, uint2(cell), tap_position, tap_packed_normal) == 0.0f)
            {
                continue;
            }
            weight *= rt_lab_gradient_distance_weight(camera.position, center_position, center_normal, tap_position,
                uncompress_normal_octahedral_32(tap_packed_normal), rcp_pixel_width_ws, float(abs(d) * RT_LAB_GRADIENT_CELL));
            if (weight <= 0.0f)
            {
                continue;
            }
        }
        float4 tap_sample;
        float3 tap_moments;
        if (vertical)
        {
            tap_sample = push.attach.blur_sums.get()[cell];
            tap_moments = push.attach.blur_moments.get()[cell].xyz;
        }
        else
        {
            if (push.per_pixel != 0u)
            {
                // Cell sums as one sample (cluster): the moments of the cell sums.
                tap_sample = float4(float2(float(asint(push.attach.gradient_delta_sum.get()[cell])), float(push.attach.gradient_max_sum.get()[cell])) / RT_LAB_GRADIENT_SUM_SCALE,
                    float(push.attach.gradient_count_sum.get()[cell]), float(push.attach.gradient_old_sum.get()[cell]) / RT_LAB_GRADIENT_SUM_SCALE);
            }
            else
            {
                const float2 packed_sample = rt_lab_unpack_gradient_sample(push.attach.gradient_samples.get()[cell]);
                // A valid sample has max > 0; old = max - delta when brightening, max when darkening.
                tap_sample = float4(packed_sample, packed_sample.y > 0.0f ? 1.0f : 0.0f, packed_sample.x >= 0.0f ? packed_sample.y - packed_sample.x : packed_sample.y);
            }
            tap_moments = float3(square(tap_sample.x), tap_sample.x * tap_sample.w, square(tap_sample.w));
        }
        sums += tap_sample * weight;
        moments += tap_moments * square(weight);
    }
    if (!vertical)
    {
        push.attach.blur_sums_dst.get()[dtid] = sums;
        push.attach.blur_moments_dst.get()[dtid] = float4(moments, 0.0f);
        return;
    }
    if ((settings.temporal_gradient_cap_mode & RT_LAB_GRADIENT_CAP_KALMAN) != 0)
    {
        // == Kalman temporal gradient (rt_lab_rays.inl) ==
        // Measurement: log ratio of the re-traced sums and its variance (ratio estimator, delta method).
        float z = 0.0f;
        float measurement_variance = -1.0f; // < 0: no samples in reach
        if (sums.z > 1e-3f)
        {
            const float old_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * sums.z;
            const float sum_old = max(sums.w, old_floor);
            const float sum_new = max(sums.w + sums.x, old_floor);
            const float ratio = sum_new / sum_old;
            const float ratio_minus_one = ratio - 1.0f;
            const float residual = max(moments.x - 2.0f * ratio_minus_one * moments.y + square(ratio_minus_one) * moments.z, 0.0f);
            z = log(ratio);
            measurement_variance = residual / square(sum_new) + RT_LAB_GRADIENT_KALMAN_MIN_VARIANCE;
        }
        float x = z;
        float p = max(measurement_variance, 0.0f);
        float4 previous;
        if (settings.reset_history == 0 && rt_lab_gradient_previous_cell(push, center_position, center_normal, rcp_pixel_width_ws, previous))
        {
            const float p_predicted = previous.y + max(settings.temporal_gradient_kalman_q, 0.0f);
            if (measurement_variance < 0.0f)
            {
                x = previous.x * RT_LAB_GRADIENT_KALMAN_NO_SAMPLE_DECAY;
                p = p_predicted;
            }
            else
            {
                const float innovation = z - previous.x;
                const float innovation_variance = p_predicted + measurement_variance;
                // Outside the gate: a shock, reset to this frame's measurement (x = z, p = R).
                if (square(innovation) <= square(max(settings.temporal_gradient_kalman_gate, 0.0f)) * innovation_variance)
                {
                    const float gain = p_predicted / innovation_variance;
                    x = previous.x + gain * innovation;
                    p = (1.0f - gain) * p_predicted;
                }
            }
        }
        push.attach.gradient_history_dst.get()[dtid] = float4(x, p, 0.0f, center_depth);
        // Wiener shrunk correction, information loss = uncertainty + the change the clamp could not correct.
        const float shrink = square(x) / max(square(x) + p, 1e-12f);
        const float correction = clamp(shrink * x, -RT_LAB_GRADIENT_KALMAN_MAX_LOG, RT_LAB_GRADIENT_KALMAN_MAX_LOG);
        const float loss = p + square(x - correction);
        push.attach.temporal_gradient.get()[dtid] = uint4(asuint(loss), asuint(center_depth), center_packed_normal, asuint(correction));
        return;
    }
    // This frame's sums: the signed values (history rescale) never use the smoothed ones, the rescale is applied once
    // per frame (smoothed signed changes would rescale the history again every frame they linger).
    const float4 current_sums = sums;
    // Temporal smoothing of the sums (temporal_gradient_history_alpha): nearest previous cell of this cell's
    // representative point, same surface by a plane test (RT_LAB_GRADIENT_HISTORY_PLANE_TOLERANCE cell widths).
    {
        // Shock bypass: a big change of THIS frame resets the smoothing (instant reaction to single frame flashes).
        const float current_lambda = sums.y > 0.0f ? saturate(abs(sums.x) / max(sums.y, RT_LAB_RELATIVE_DIFFERENCE_FLOOR)) : 0.0f;
        const float shock = max(settings.temporal_gradient_history_shock, 1e-3f);
        const float alpha = lerp(saturate(settings.temporal_gradient_history_alpha), 1.0f, smoothstep(0.5f * shock, shock, current_lambda));
        float4 previous;
        if (alpha < 1.0f && settings.reset_history == 0 && rt_lab_gradient_previous_cell(push, center_position, center_normal, rcp_pixel_width_ws, previous))
        {
            sums.xyz = lerp(previous.xyz, sums.xyz, alpha);
        }
        push.attach.gradient_history_dst.get()[dtid] = float4(sums.xyz, center_depth);
    }
    // Denominator floor: RT_LAB_RELATIVE_DIFFERENCE_FLOOR exposed, so changes in the dark do not reset the history.
    const float lambda = sums.y > 0.0f ? saturate(abs(sums.x) / max(sums.y, RT_LAB_RELATIVE_DIFFERENCE_FLOOR)) : 0.0f;
    // .w = signed lambda of THIS frame (sign of the summed change), kept undilated for the history rescale.
    const float current_lambda_abs = current_sums.y > 0.0f ? saturate(abs(current_sums.x) / max(current_sums.y, RT_LAB_RELATIVE_DIFFERENCE_FLOOR)) : 0.0f;
    const float signed_lambda = current_sums.x < 0.0f ? -current_lambda_abs : current_lambda_abs;
    if (rt_lab_gradient_packed(settings))
    {
        // + the absolute weighted mean exposed delta of the samples in reach, made relative per pixel (rt_lab_gradient_pixel_relative).
        const float mean_delta = sums.z > 1e-6f ? sums.x / sums.z : 0.0f;
        const float current_mean_delta = current_sums.z > 1e-6f ? current_sums.x / current_sums.z : 0.0f;
        push.attach.temporal_gradient.get()[dtid] = uint4(rt_lab_gradient_pack_pair(lambda, abs(mean_delta)), asuint(center_depth), center_packed_normal,
            rt_lab_gradient_pack_pair(signed_lambda, current_mean_delta));
        return;
    }
    push.attach.temporal_gradient.get()[dtid] = uint4(asuint(lambda), asuint(center_depth), center_packed_normal, asuint(signed_lambda));
}

// == Temporal gradient dilation (rt_lab_rays.inl) ======================================================================

[[vk::push_constant]] RtLabGradientDilatePush rt_lab_gradient_dilate_push;

[shader("compute")]
[numthreads(RT_LAB_RAYS_X, RT_LAB_RAYS_Y, 1)]
func entry_rt_lab_gradient_dilate(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_gradient_dilate_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const uint4 center = push.attach.src.get()[dtid];
    const float center_depth = asfloat(center.y);
    if (!(center_depth > 0.0f))
    {
        push.attach.dst.get()[dtid] = center; // sky: nothing to dilate into
        return;
    }
    const float2 inv_lab_size = rcp(float2(push.lab_size));
    const float3 center_position = linear_depth_to_world_space(camera, rt_lab_ndc(rt_lab_gradient_cell_pixel(dtid, push.lab_size), inv_lab_size), center_depth);
    const float3 center_normal = uncompress_normal_octahedral_32(center.z);
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.lab_size.y), center_depth));

    const int radius = clamp(settings.temporal_gradient_dilation_radius, 0, RT_LAB_GRADIENT_MAX_BLUR_RADIUS);
    const float sigma = max(float(radius) * 0.5f, 0.5f);
    const float rcp_two_sigma_sq = rcp(2.0f * sigma * sigma);
    const int2 direction = push.vertical != 0u ? int2(0, 1) : int2(1, 0);
    // temporal_gradient_pixel_relative: lambda and |mean delta| (f16 pair) dilated separately.
    const bool packed = rt_lab_gradient_packed(settings);
    float2 lambda = packed ? rt_lab_gradient_unpack_pair(center.x) : float2(asfloat(center.x), 0.0f);
    for (int d = -radius; d <= radius; ++d)
    {
        if (d == 0)
        {
            continue;
        }
        const int2 cell = int2(dtid) + direction * d;
        if (any(cell < 0) || any(cell >= int2(push.size)))
        {
            continue;
        }
        const uint4 tap = push.attach.src.get()[cell];
        const float2 tap_lambda = packed ? rt_lab_gradient_unpack_pair(tap.x) : float2(asfloat(tap.x), 0.0f);
        const float tap_depth = asfloat(tap.y);
        if (!any(tap_lambda > lambda) || !(tap_depth > 0.0f))
        {
            continue; // cannot raise the max
        }
        const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(rt_lab_gradient_cell_pixel(uint2(cell), push.lab_size), inv_lab_size), tap_depth);
        const float weight = exp(-float(d * d) * rcp_two_sigma_sq) *
            rt_lab_gradient_distance_weight(camera.position, center_position, center_normal, tap_position, uncompress_normal_octahedral_32(tap.z),
                rcp_pixel_width_ws, float(abs(d) * RT_LAB_GRADIENT_CELL));
        lambda = max(lambda, tap_lambda * weight);
    }
    // .w: undilated signed lambda (+ signed mean delta when packed).
    push.attach.dst.get()[dtid] = uint4(packed ? rt_lab_gradient_pack_pair(lambda.x, lambda.y) : asuint(lambda.x), center.y, center.z, center.w);
}

// == Gather ===========================================================================================================

[[vk::push_constant]] RtLabGatherPush rt_lab_gather_push;

// Per gather thread (= lab pixel of the 8x8 tile, row major): ray count of its pixel, for the temporal gradient sample
// selection.
groupshared uint gs_gather_rays[RT_LAB_TILE_PIXELS];

// Temporal gradient sample selection of the cell whose corner pixel is `corner` (all its rays are traced by now): the
// first ray of the first pixel with rays (pixel order rotated per frame and cell).
// Writes the cell's record (rt_lab_trace.inl: origin + t min, surface + luminance, direction + face normal + magic) or
// invalidates it when the cell has no ray. Out of bounds pixels have 0 rays in the groupshared array.
func rt_lab_gradient_select(RtLabGatherPush push, RtLabSettings settings, uint2 corner, uint2 corner_gtid)
{
    const uint2 cell = corner / RT_LAB_GRADIENT_CELL;
    const uint cell_pixels = RT_LAB_GRADIENT_CELL * RT_LAB_GRADIENT_CELL;
    const uint frame = settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
    const uint cell_hash = cell.x * 73856093u ^ cell.y * 19349663u ^ frame * 83492791u;
    uint2 selected_pixel = corner;
    uint selected_ray = RT_LAB_GRADIENT_NO_RAY;
    const uint start = cell_hash % cell_pixels;
    for (uint j = 0u; j < cell_pixels; ++j)
    {
        const uint local = (start + j) % cell_pixels;
        const uint2 tile_pixel = corner_gtid + uint2(local % RT_LAB_GRADIENT_CELL, local / RT_LAB_GRADIENT_CELL);
        if (gs_gather_rays[tile_pixel.y * RT_LAB_RAYS_X + tile_pixel.x] > 0u)
        {
            selected_pixel = corner + (tile_pixel - corner_gtid);
            selected_ray = push.attach.ray_alloc.get()[selected_pixel].x;
            break;
        }
    }

    if (selected_ray == RT_LAB_GRADIENT_NO_RAY)
    {
        push.attach.gradient_record_data.get()[cell] = uint4(0u, 0u, 0u, 0u);
        return;
    }
    rt_lab_gradient_write_record(push, settings, cell, selected_pixel, selected_ray);
}

// Writes a gradient record at record (rt_lab_trace.inl: origin + t min, surface + luminance, direction + face normal +
// magic) for ray `ray_index` of `pixel`.
func rt_lab_gradient_write_record(RtLabGatherPush push, RtLabSettings settings, uint2 record, uint2 pixel, uint ray_index)
{
    // RAW ray (never a pre filter output like filtered_rays): the re-trace shades raw radiance, both sides must match.
    const RtLabRay ray = push.attach.rays[ray_index];
    const float depth = push.attach.lab_depth.get()[pixel];
    const uint packed_normal = push.attach.lab_face_normal.get()[pixel];
    float3 world_pos;
    float t_min;
    const float3 origin = rt_lab_ray_launch(push.attach.globals.view_camera, pixel, push.size, depth, uncompress_normal_octahedral_32(packed_normal), world_pos, t_min);
    push.attach.gradient_record_origin.get()[record] = float4(origin, t_min);
    push.attach.gradient_record_surface.get()[record] = float4(world_pos, rt_lab_luminance(ray.radiance));
    // The ray's hit shading seed (same derivation as the trace), so the re-trace reproduces its shading.
    const uint pixel_ray = ray_index - push.attach.ray_alloc.get()[pixel].x;
    const uint hit_seed = rt_lab_hit_seed(rt_lab_ray_seed(settings, push.attach.globals.trunk_flt_frame_index, pixel, pixel_ray));
    push.attach.gradient_record_data.get()[record] = uint4(ray.direction, packed_normal, 0u, RT_LAB_GRADIENT_RECORD_MAGIC | hit_seed);
}

[shader("compute")]
[numthreads(RT_LAB_RAYS_X, RT_LAB_RAYS_Y, 1)]
func entry_rt_lab_gather(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rt_lab_gather_push;
    const bool in_bounds = all(dtid < push.size);
    const uint2 alloc = in_bounds ? push.attach.ray_alloc.get()[dtid] : uint2(0u, 0u);
    float3 sum = float3(0.0f, 0.0f, 0.0f);        // raw rays (the gradient selection works on them)
    float3 output_sum = float3(0.0f, 0.0f, 0.0f); // the rays of the pixel mean (ray filter output when bound)
    for (uint k = 0u; k < alloc.y; ++k)
    {
        sum += push.attach.rays[alloc.x + k].radiance;
        output_sum += push.filtered_bound != 0u ? push.attach.filtered_rays[alloc.x + k].radiance : push.attach.rays[alloc.x + k].radiance;
    }
    if (in_bounds)
    {
        const float count = float(alloc.y);
        push.attach.lab_raw_diffuse.get()[dtid] = alloc.y > 0u ? float4(output_sum / count, count) : float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (push.raw_luminance_bound != 0u)
        {
            push.attach.lab_raw_luminance.get()[dtid] = alloc.y > 0u ? rt_lab_luminance(sum / count) : 0.0f;
        }
    }

    // Base rays (temporal_gradient_all_base_rays): the pixels of the guaranteed base share (min_ray_budget of the tile:
    // the first ranks of the distribute's rotated Bayer order, same 8x8 tiles) record their ray 0. Rate = min_ray_budget
    // (0.125: 8 per tile, ~2 per 4x4 cell). With sky in the tile approximate (the distribute ranks geometry pixels only).
    // DENSE records: slot (tile.x x slots + rank, tile.y), every slot rewritten each frame (invalid without a ray), so the
    // re-trace dispatches only over real records (a sparse lab res layout left 7 of 8 lanes idle in every wave).
    if (push.gradient_enabled != 0u && push.gradient_per_pixel != 0u)
    {
        let settings = push.attach.globals.rt_lab_settings;
        const uint slots = rt_lab_gradient_base_slots(settings);
        const uint rank = (rt_lab_bayer_8x8(gtid) + RT_LAB_TILE_PIXELS - rt_lab_base_rank_rotation(settings, push.attach.globals.trunk_flt_frame_index)) % RT_LAB_TILE_PIXELS;
        if (rank < slots)
        {
            const uint2 record = uint2(gid.x * slots + rank, gid.y);
            if (in_bounds && alloc.y > 0u)
            {
                rt_lab_gradient_write_record(push, settings, record, dtid, alloc.x);
            }
            else
            {
                push.attach.gradient_record_data.get()[record] = uint4(0u, 0u, 0u, 0u);
            }
        }
        return;
    }
    // Uniform branch (push constant): every thread reaches the barrier.
    if (push.gradient_enabled != 0u)
    {
        const uint slot = gtid.y * RT_LAB_RAYS_X + gtid.x;
        gs_gather_rays[slot] = alloc.y;
        GroupMemoryBarrierWithGroupSync();
        // The corner pixel of a cell is in bounds whenever the cell exists.
        if (in_bounds && all((dtid % RT_LAB_GRADIENT_CELL) == 0u))
        {
            rt_lab_gradient_select(push, push.attach.globals.rt_lab_settings, dtid, gtid);
        }
    }
}

// == Ray filter (RtLabSettings::ray_filter_*) ========================================================================

[[vk::push_constant]] RtLabRayFilterPush rt_lab_ray_filter_push;

// One thread per ray list entry: statistics of all rays of the (2r + 1)^2 neighbor pixels (pixel distance is the only
// acceptance test, no plane / normal test; the pixel's own OTHER rays included, the ray itself never), then the ray is
// clamped against them, hue preserving (all channels scaled by one factor):
//   RT_LAB_RAY_FILTER_VARIANCE:   luminance mean / std -> ray luminance clamped to mean + sigmas x std (and, with
//                                 ray_filter_clamp_low, raised to mean - sigmas x std)
//   RT_LAB_RAY_FILTER_PERCEPTUAL: like the RTGI firefly clamp (rgb mode): per channel ceiling = exp(mean of log rgb +
//                                 ray_filter_tolerance), log with the RTGI perceptual floor; the channel most over its
//                                 ceiling decides the factor
// No neighbor ray: unchanged.
[shader("compute")]
[numthreads(RT_LAB_TRACE_THREADS, 1, 1)]
func entry_rt_lab_ray_filter(uint ray_index : SV_DispatchThreadID)
{
    let push = rt_lab_ray_filter_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (ray_index >= min(push.attach.ray_counters->ray_count, rt_lab_ray_budget(push.size, settings, push.ray_capacity)))
    {
        return;
    }
    const RtLabRay ray = push.attach.rays[ray_index];
    const uint packed_pixel = push.attach.ray_list[ray_index];
    const uint2 dtid = uint2(packed_pixel & 0xFFFFu, packed_pixel >> 16u);
    const bool perceptual = settings.ray_filter_mode == RT_LAB_RAY_FILTER_PERCEPTUAL;
    // RTGI perceptual floor (calc_perceptual_radiance_floor): keeps log() of dark values bounded.
    const float log_floor = RTGI_RADIANCE_SCALE * 1e-3f / max(push.attach.globals.exposure, 1e-12f);
    const int radius = clamp(settings.ray_filter_radius, 0, RT_LAB_RAY_FILTER_MAX_RADIUS);

    float3 sum = float3(0.0f, 0.0f, 0.0f); // variance: (lum, lum^2, -); perceptual: sum of log rgb
    float sum_squared = 0.0f;
    float count = 0.0f;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            const int2 neighbor = int2(dtid) + int2(x, y);
            if (any(neighbor < 0) || any(neighbor >= int2(push.size)))
            {
                continue;
            }
            if (push.attach.lab_depth.get()[neighbor] == 0.0f)
            {
                continue; // sky: no rays
            }
            const uint2 alloc = push.attach.ray_alloc.get()[neighbor];
            for (uint k = 0u; k < alloc.y; ++k)
            {
                const uint index = alloc.x + k;
                if (index == ray_index)
                {
                    continue; // the ray itself is never part of its own distribution
                }
                const float3 radiance = push.attach.rays[index].radiance;
                if (perceptual)
                {
                    sum += log(max(radiance, log_floor));
                }
                else
                {
                    const float luminance = rt_lab_luminance(radiance);
                    sum.x += luminance;
                    sum_squared += luminance * luminance;
                }
                count += 1.0f;
            }
        }
    }

    float factor = 1.0f;
    if (count > 0.0f)
    {
        if (perceptual)
        {
            const float3 ceiling = exp(sum / count + max(settings.ray_filter_tolerance, 0.0f));
            const float3 over = ray.radiance / max(ceiling, 1e-12f);
            factor = 1.0f / max(max(over.x, max(over.y, over.z)), 1.0f);
        }
        else
        {
            const float mean = sum.x / count;
            const float std = sqrt(max(sum_squared / count - mean * mean, 0.0f));
            const float luminance = rt_lab_luminance(ray.radiance);
            const float ceiling = mean + settings.ray_filter_sigmas * std;
            const float floor_value = mean - settings.ray_filter_sigmas * std;
            if (luminance > ceiling)
            {
                factor = ceiling / luminance;
            }
            else if (settings.ray_filter_clamp_low != 0 && luminance < floor_value && luminance > 0.0f)
            {
                factor = floor_value / luminance;
            }
        }
    }
    RtLabRay filtered = ray;
    filtered.radiance = ray.radiance * (isfinite(factor) ? max(factor, 0.0f) : 1.0f);
    push.attach.filtered_rays[ray_index] = filtered;
}
