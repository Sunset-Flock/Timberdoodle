#pragma once

#include "rt_lab_rays.inl"

#include "rt_lab_shared.hlsl"
#include "rt_lab_reproject.hlsl"
#include "shader_lib/debug.glsl"

// == Demand ===========================================================================================================

[[vk::push_constant]] RtLabDemandPush rt_lab_demand_push;

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
    push.attach.lab_face_normal.get()[dtid] = packed_normal;

    uint desired = 0u;
    if (depth != 0.0f)
    {
        desired = 1u;
        if (settings.redistribution_enabled != 0)
        {
            // Exactly the history length the temporal pass reprojects (occluder aware soft normalize + temporal gradient 2 cut).
            float history_length = push.history_bound != 0u
                ? rt_lab_reproject_history_length(push.attach.history.get(), push.attach.lab_depth_history.get(), push.attach.lab_face_normal_history.get(),
                      &push.attach.globals.view_camera_prev_frame, &push.attach.globals.rt_lab_settings, int2(push.size), position, normal, rcp_pixel_width_ws)
                : 0.0f;
            // Temporal gradient 2: the history length the temporal pass keeps (same call, same inputs). A cut history asks for rays.
            if (push.gradient2_bound != 0u && push.history_bound != 0u && settings.temporal_gradient2_scale_history != 0 && history_length > 0.0f)
            {
                const float perceptual_variance = rt_lab_previous_perceptual_variance(push.attach.history.get(), push.attach.variance_history.get(),
                    &push.attach.globals.view_camera_prev_frame, settings, push.size, position, push.attach.globals.exposure);
                history_length *= rt_lab_gradient2_history_keep(push.attach.gradient2.get()[dtid], history_length, perceptual_variance);
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
        const float max_base = saturate(settings.max_base_coverage);
        const bool base_capped = max_base > 0.0f && max_base < 1.0f;
        // SURPLUS: the demand (shaped by max_rays_per_pixel and the history) is below the budget. The budget is a hard AND exact total, so the
        // demand only decides WHERE the rays go: every pixel's share is scaled up, rays = ratio x demand (rounded by the cumulative scan below).
        // With a base coverage limit the base rays keep their schedule, the scaled share is the extras: ratio = (budget - base) / total demand.
        bool surplus = false;
        float surplus_ratio = 0.0f;
        if (base_demand > 0u)
        {
            const float weight_total = float(base_demand + extra_demand);
            if (base_capped)
            {
                const float base_total = min(max_base * float(base_demand), float(budget));
                const float extra_budget = float(budget) - base_total;
                surplus = float(extra_demand) < extra_budget;
                surplus_ratio = extra_budget / weight_total;
            }
            else
            {
                surplus = base_demand + extra_demand < budget;
                surplus_ratio = float(budget) / weight_total;
            }
        }
        if (!base_capped && !surplus && base_demand + extra_demand <= budget)
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
            float base_fraction = base_demand > 0u
                ? min(min(max((float(budget) - float(extra_demand) * extra_ratio) / float(base_demand), min_budget), 1.0f), float(budget) / float(base_demand))
                : 0.0f;
            if (base_capped)
            {
                // The leftover budget never raises the base coverage above the limit.
                base_fraction = min(base_fraction, max_base);
            }
            if (surplus)
            {
                // Everything above the base schedule is the scaled share.
                base_fraction = base_capped ? max_base : 0.0f;
            }

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
            if (base_capped)
            {
                // Fixed schedule over ALL 64 tile positions (sky included): ranks [rotation, rotation + slots), the rotation advances by `slots`
                // per frame, so every pixel gets exactly one base ray every 64 / slots frames. The counted version below rounds the tile's
                // budget down: a tile with fewer than 1 / coverage geometry pixels (silhouettes, foliage) got NO base ray, ever.
                const uint slots = uint(base_fraction * float(RT_LAB_TILE_PIXELS) + 1e-3f);
                const uint scheduled_rotation = (frame * slots) % RT_LAB_TILE_PIXELS;
                for (uint r = 0u; r < slots; ++r)
                {
                    const uint i = gs_pixel_by_rank[(r + scheduled_rotation) % RT_LAB_TILE_PIXELS];
                    if (gs_desired[i] > 0u)
                    {
                        gs_rays[i] = 1u;
                        ++base_given;
                    }
                }
            }
            for (uint r = 0u; !base_capped && r < RT_LAB_TILE_PIXELS && base_given < tile_base_budget; ++r)
            {
                const uint i = gs_pixel_by_rank[(r + rank_rotation) % RT_LAB_TILE_PIXELS];
                if (gs_desired[i] > 0u)
                {
                    gs_rays[i] = 1u;
                    ++base_given;
                }
            }
            // Extras: cumulative scan, target(prefix) = floor(prefix / total x budget + phase), differenced.
            // Surplus: the scan covers the whole demand (base ray included in the weight) and has no per pixel cap.
            const uint tile_id = gid.y * ((push.size.x + RT_LAB_RAYS_X - 1u) / RT_LAB_RAYS_X) + gid.x;
            const float phase = frac(float(frame) * 0.61803398875f + float(tile_id) * 0.7548776662f + 0.5f);
            const uint scan_total = surplus ? tile_base_demand + tile_extra_demand : tile_extra_demand;
            const uint scan_budget = surplus ? uint(surplus_ratio * float(scan_total) + phase) : tile_extra_budget;
            const float inv_extra_demand = scan_total > 0u ? rcp(float(scan_total)) : 0.0f;
            uint accumulated_demand = 0u;
            uint extra_given = 0u;
            for (uint i = 0u; i < RT_LAB_TILE_PIXELS; ++i)
            {
                const uint extra_i = surplus ? gs_desired[i] : (gs_desired[i] > 0u ? gs_desired[i] - 1u : 0u);
                accumulated_demand += extra_i;
                const uint target = scan_total > 0u
                    ? min(uint(saturate(float(accumulated_demand) * inv_extra_demand) * float(scan_budget) + phase), scan_budget)
                    : 0u;
                const uint extra = min(target - min(extra_given, target), surplus ? scan_budget : extra_i);
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
    float3 sum = float3(0.0f, 0.0f, 0.0f); // the rays of the pixel mean (ray filter output when bound)
    for (uint k = 0u; k < alloc.y; ++k)
    {
        sum += push.filtered_bound != 0u ? push.attach.filtered_rays[alloc.x + k].radiance : push.attach.rays[alloc.x + k].radiance;
    }
    if (in_bounds)
    {
        const float count = float(alloc.y);
        push.attach.lab_raw_diffuse.get()[dtid] = alloc.y > 0u ? float4(sum / count, count) : float4(0.0f, 0.0f, 0.0f, 0.0f);
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
//   RT_LAB_RAY_FILTER_ABSOLUTE:   per channel ceiling = linear mean rgb x ray_filter_factor (floored at the RTGI perceptual
//                                 floor); the channel most over its ceiling decides the factor
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

    float3 sum = float3(0.0f, 0.0f, 0.0f); // absolute: sum of rgb; perceptual: sum of log rgb
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
                sum += perceptual ? log(max(radiance, log_floor)) : radiance;
                count += 1.0f;
            }
        }
    }

    float factor = 1.0f;
    if (count > 0.0f)
    {
        const float3 ceiling = perceptual
            ? exp(sum / count + max(settings.ray_filter_tolerance, 0.0f))
            : max(sum / count, log_floor) * max(settings.ray_filter_factor, 1.0f);
        const float3 over = ray.radiance / max(ceiling, 1e-12f);
        factor = 1.0f / max(max(over.x, max(over.y, over.z)), 1.0f);
    }
    RtLabRay filtered = ray;
    filtered.radiance = ray.radiance * (isfinite(factor) ? max(factor, 0.0f) : 1.0f);
    push.attach.filtered_rays[ray_index] = filtered;
}
