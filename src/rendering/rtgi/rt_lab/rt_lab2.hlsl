#pragma once

#include "rt_lab2.inl"
#include "rt_lab_integrator.hlsl"

#include "rt_lab_shared.hlsl"
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtLab2QueryPush rt_lab2_query_push;
[[vk::push_constant]] RtLab2CellPassPush rt_lab2_reset_push;
[[vk::push_constant]] RtLab2CellPassPush rt_lab2_prepare_push;
[[vk::push_constant]] RtLab2CellPassPush rt_lab2_cell_integrate_push;
[[vk::push_constant]] RtLab2AccumulatePush rt_lab2_accumulate_push;
[[vk::push_constant]] RtLab2LookupPush rt_lab2_lookup_push;
[[vk::push_constant]] RtLab2IntegratorPush rt_lab2_integrator_push;
[[vk::push_constant]] RtLab2ProbePush rt_lab2_probe_push;
[[vk::push_constant]] RtLab2DemandPush rt_lab2_demand_push;
[[vk::push_constant]] RtLab2RaysDebugPush rt_lab2_rays_debug_push;
[[vk::push_constant]] RtLab2StatsPush rt_lab2_stats_push;
[[vk::push_constant]] RtLab2SelectPush rt_lab2_select_push;
[[vk::push_constant]] RtLab2GradientRecordPush rt_lab2_gradient_record_push;
[[vk::push_constant]] RtLab2GradientSpatialPush rt_lab2_gradient_spatial_push;
[[vk::push_constant]] RtLab2DistributePush rt_lab2_distribute_push;

// == Cell keys (a copy of rtgi_hash_grid_shared.hlsl, own settings) =====================================================
// A cell is (cascade level, integer cell coordinate at that level's cell size, face normal bucket). The level follows the
// distance to the camera, so a cell covers roughly cell_size_px pixels.

struct RtLab2Key
{
    int3 coord;
    uint level;
    uint normal_bucket;
};

func rt_lab2_level(float distance, float pixel_width_per_distance, RtLab2Settings s) -> uint
{
    const float target_cell_size = distance * pixel_width_per_distance * s.cell_size_px;
    const float levels = ceil(log2(max(target_cell_size / max(s.min_cell_size, 1e-4f), 1.0f)));
    return uint(min(levels, float(RT_LAB2_MAX_LEVEL)));
}

func rt_lab2_cell_size(uint level, RtLab2Settings s) -> float
{
    return max(s.min_cell_size, 1e-4f) * float(1u << level);
}

func rt_lab2_normal_bucket(float3 normal) -> uint
{
    const float3 a = abs(normal);
    const uint axis = (a.x >= a.y && a.x >= a.z) ? 0u : (a.y >= a.z ? 1u : 2u);
    return axis * 2u + (normal[axis] < 0.0f ? 1u : 0u);
}

func rt_lab2_make_key(float3 position, float3 normal, float3 camera_position, float pixel_width_per_distance, RtLab2Settings s) -> RtLab2Key
{
    RtLab2Key key;
    key.level = rt_lab2_level(length(position - camera_position), pixel_width_per_distance, s);
    key.coord = int3(floor(position / rt_lab2_cell_size(key.level, s)));
    key.normal_bucket = rt_lab2_normal_bucket(normal);
    return key;
}

// Stochastic cascade selection (RtLab2Settings::level_blend_width). The continuous level l = log2(target cell size / min cell size) picks the
// primary cascade ceil(l) (cell sizes between 1x and 2x the target). u in (0, 1]: 1 = the cell is about as big as the target (the level is
// about to change to the next coarser one), 0 = it just changed. Near both transitions a tap uses the neighbor cascade with probability
// 0.5 (1 - smoothstep(0, width, distance to the transition)): 0.5 exactly at the transition (the mixture is continuous across it), 0 beyond the region.
struct RtLab2LevelBlend
{
    uint primary;
    uint other;
    float p_other; // 0 = no blend (outside the region, or width 0)
};

func rt_lab2_level_blend(float distance, float pixel_width_per_distance, RtLab2Settings s) -> RtLab2LevelBlend
{
    RtLab2LevelBlend blend;
    const float target_ratio = distance * pixel_width_per_distance * s.cell_size_px / max(s.min_cell_size, 1e-4f);
    blend.primary = rt_lab2_level(distance, pixel_width_per_distance, s);
    blend.other = blend.primary;
    blend.p_other = 0.0f;
    const float width = saturate(s.level_blend_width);
    if (width <= 0.0f || target_ratio <= 1.0f)
    {
        return blend;
    }
    const float l = log2(target_ratio);
    const float u = l - (ceil(l) - 1.0f);
    const float p_up = blend.primary < RT_LAB2_MAX_LEVEL ? 0.5f * (1.0f - smoothstep(0.0f, width, 1.0f - u)) : 0.0f;
    const float p_down = blend.primary > 0u ? 0.5f * (1.0f - smoothstep(0.0f, width, u)) : 0.0f;
    if (p_up >= p_down)
    {
        blend.other = blend.primary + 1u;
        blend.p_other = p_up;
    }
    else
    {
        blend.other = blend.primary - 1u;
        blend.p_other = p_down;
    }
    return blend;
}

func rt_lab2_make_key_at_level(float3 position, float3 normal, uint level, RtLab2Settings s) -> RtLab2Key
{
    RtLab2Key key;
    key.level = level;
    key.coord = int3(floor(position / rt_lab2_cell_size(level, s)));
    key.normal_bucket = rt_lab2_normal_bucket(normal);
    return key;
}

func rt_lab2_hash_u32(uint x) -> uint
{
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

func rt_lab2_hash_key(RtLab2Key key, uint seed) -> uint
{
    uint h = rt_lab2_hash_u32(seed ^ uint(key.coord.x));
    h = rt_lab2_hash_u32(h ^ uint(key.coord.y));
    h = rt_lab2_hash_u32(h ^ uint(key.coord.z));
    h = rt_lab2_hash_u32(h ^ (key.level | (key.normal_bucket << 8u)));
    return h;
}

func rt_lab2_slot_hash(RtLab2Key key) -> uint
{
    return rt_lab2_hash_key(key, 0x9E3779B9u);
}

func rt_lab2_checksum(RtLab2Key key) -> uint
{
    return max(rt_lab2_hash_key(key, 0x85EBCA6Bu), 1u); // 0 == empty slot
}

// Number of cell slots (a power of two): 2^cell_capacity_log2, 0 = the default.
func rt_lab2_capacity(RtLab2Settings s) -> uint
{
    const float log2_capacity = s.cell_capacity_log2 > 0.0f ? s.cell_capacity_log2 : float(RT_LAB2_CELL_CAPACITY_LOG2_DEFAULT);
    return 1u << uint(clamp(log2_capacity, 12.0f, 24.0f));
}

func rt_lab2_probe_slot(uint slot_hash, uint probe, uint mask) -> uint
{
    return (slot_hash + probe) & mask;
}

// Lookup without insertion (the whole probe window is scanned: freed cells leave holes).
func rt_lab2_find(RtLab2Cell* cells, RtLab2Key key, uint mask) -> uint
{
    const uint slot_hash = rt_lab2_slot_hash(key);
    const uint checksum = rt_lab2_checksum(key);
    for (uint probe = 0u; probe < RT_LAB2_MAX_PROBES; ++probe)
    {
        const uint slot = rt_lab2_probe_slot(slot_hash, probe, mask);
        if (cells[slot].key == checksum)
        {
            return slot;
        }
    }
    return RT_LAB2_INVALID_CELL;
}

// Lookup, inserting the key into the first free slot of its probe window when it is not present (see
// rtgi_hash_grid_find_or_insert for the race notes).
func rt_lab2_find_or_insert(RtLab2Cell* cells, RtLab2Key key, uint mask) -> uint
{
    const uint slot_hash = rt_lab2_slot_hash(key);
    const uint checksum = rt_lab2_checksum(key);
    uint first_free_probe = RT_LAB2_MAX_PROBES;
    for (uint probe = 0u; probe < RT_LAB2_MAX_PROBES; ++probe)
    {
        const uint slot = rt_lab2_probe_slot(slot_hash, probe, mask);
        const uint slot_key = cells[slot].key;
        if (slot_key == checksum)
        {
            return slot;
        }
        if (slot_key == 0u && first_free_probe == RT_LAB2_MAX_PROBES)
        {
            first_free_probe = probe;
        }
    }
    for (uint probe = first_free_probe; probe < RT_LAB2_MAX_PROBES; ++probe)
    {
        const uint slot = rt_lab2_probe_slot(slot_hash, probe, mask);
        uint prev_key = 0u;
        InterlockedCompareExchange(cells[slot].key, 0u, checksum, prev_key);
        if (prev_key == 0u || prev_key == checksum)
        {
            if (prev_key == 0u)
            {
                cells[slot].level = key.level; // only the inserting pixel writes it (statistics)
            }
            return slot;
        }
    }
    return RT_LAB2_INVALID_CELL;
}

func rt_lab2_pixel_width_per_distance(CameraInfo camera, float size_y) -> float
{
    return 2.0f / (abs(camera.proj[1][1]) * size_y);
}

func rt_lab2_cell_radiance(RtLab2Cell* cell) -> float3
{
    return float3(cell.radiance_r, cell.radiance_g, cell.radiance_b);
}

// == Reset ============================================================================================================
// Empties every cell (history reset of a test phase). Dispatched before the query, only on reset frames.
[shader("compute")]
[numthreads(RT_LAB2_CELL_PASS_X, 1, 1)]
func entry_rt_lab2_reset(uint dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_reset_push;
    const RtLab2Settings s = *push.settings;
    if (dtid >= rt_lab2_capacity(s))
    {
        return;
    }
    RtLab2Cell* cell = &push.attach.cells[dtid];
    cell.key = 0u;
    cell.frames_untouched = 0u;
    cell.pixel_count = 0u;
    cell.ray_count = 0u;
    cell.sum_r = 0;
    cell.sum_g = 0;
    cell.sum_b = 0;
    cell.radiance_r = 0.0f;
    cell.radiance_g = 0.0f;
    cell.radiance_b = 0.0f;
    cell.sample_count = 0.0f;
    cell.exposure = 0.0f;
    cell.last_pixel_count = 0u;
    cell.last_ray_count = 0u;
    cell.last_mean_r = 0.0f;
    cell.last_mean_g = 0.0f;
    cell.last_mean_b = 0.0f;
    cell.rec_next = 0u;
    cell.rec_valid = 0u;
    cell.meas_z = 0.0f;
    cell.meas_r = 0.0f;
    cell.gk_x = 0.0f;
    cell.gk_p = 0.0f;
    cell.gk_pending = 0.0f;
    cell.gk_applied = 0.0f;
    cell.gk_loss = 0.0f;
    cell.grad_count = 0u;
    cell.grad_sum_prec = 0;
    cell.grad_sum_zprec = 0;
}

// Pixels of one wave mostly share a cell: one atomic per distinct cell instead of one per pixel. Every lane must call it (wave ops).
// Returns true on exactly ONE lane per cell and frame: the lane that raised the cell's pixel count from 0 (the debug box draw uses it).
func rt_lab2_count_pixel(RtLab2Cell* cells, uint cell_index) -> bool
{
    bool first_in_cell = false;
    bool pending = cell_index != RT_LAB2_INVALID_CELL;
    while (WaveActiveAnyTrue(pending))
    {
        const uint current = WaveActiveMin(pending ? cell_index : 0xFFFFFFFFu);
        const bool mine = pending && cell_index == current;
        const uint count = WaveActiveCountBits(mine);
        if (mine && WavePrefixCountBits(mine) == 0u)
        {
            uint previous_count = 0u;
            InterlockedAdd(cells[current].pixel_count, count, previous_count);
            first_in_cell = previous_count == 0u;
        }
        pending = pending && !mine;
    }
    return first_in_cell;
}

// Debug: the world space box of a queried cell (same toggle as the RTGI hash grid: rtgi_settings.hash_grid_debug_draw_cells).
func rt_lab2_debug_cell_box(RtLab2Key key, uint cell_index, RtLab2Settings s) -> ShaderDebugAABBDraw
{
    const float cell_size = rt_lab2_cell_size(key.level, s);
    ShaderDebugAABBDraw aabb;
    aabb.position = (float3(key.coord) + 0.5f) * cell_size;
    aabb.size = cell_size.xxx;
    aabb.color = hsv2rgb(float3(IdFloatScramble(cell_index & 0xFFFFu), 0.8f, 1.0f));
    aabb.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
    return aabb;
}

// == Query ============================================================================================================
// Every geometry pixel finds (or inserts) its cell and counts itself into the cell's pixel_count (the cell's exposure
// of this frame). Runs where a reprojection would, before the rays are distributed.
[shader("compute")]
[numthreads(RT_LAB2_PASS_X, RT_LAB2_PASS_Y, 1)]
func entry_rt_lab2_query(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_query_push;
    const bool in_bounds = all(dtid < push.size);
    uint cell_index = RT_LAB2_INVALID_CELL;
    uint cell_index2 = RT_LAB2_INVALID_CELL;
    RtLab2Key key;
    RtLab2Key key2;
    key.level = 0u;
    key.coord = int3(0, 0, 0);
    key.normal_bucket = 0u;
    key2 = key;
    if (in_bounds)
    {
        const RtLab2Settings s = *push.settings;
        const CameraInfo camera = push.attach.globals.view_camera;
        const float depth = push.attach.lab_depth.get()[dtid];
        if (depth > 0.0f)
        {
            const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(float2(push.size))), depth);
            const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
            const RtLab2LevelBlend blend = rt_lab2_level_blend(length(position - camera.position), rt_lab2_pixel_width_per_distance(camera, float(push.size.y)), s);
            key = rt_lab2_make_key_at_level(position, normal, blend.primary, s);
            cell_index = rt_lab2_find_or_insert(push.attach.cells, key, rt_lab2_capacity(s) - 1u);
            if (blend.p_other > 0.0f)
            {
                // Inside the cascade blend region the neighbor cascade's cell is fed by this pixel too.
                key2 = rt_lab2_make_key_at_level(position, normal, blend.other, s);
                cell_index2 = rt_lab2_find_or_insert(push.attach.cells, key2, rt_lab2_capacity(s) - 1u);
            }
        }
        push.attach.pixel_cell.get()[dtid] = cell_index;
        push.attach.pixel_cell2.get()[dtid] = cell_index2;
    }

    const bool first = rt_lab2_count_pixel(push.attach.cells, cell_index);
    const bool first2 = rt_lab2_count_pixel(push.attach.cells, cell_index2);
    if (push.attach.globals.rtgi_settings.hash_grid_debug_draw_cells != 0 && (first || first2))
    {
        const RtLab2Settings s = *push.settings;
        if (first)
        {
            debug_draw_aabb(push.attach.globals.debug, rt_lab2_debug_cell_box(key, cell_index, s));
        }
        if (first2)
        {
            debug_draw_aabb(push.attach.globals.debug, rt_lab2_debug_cell_box(key2, cell_index2, s));
        }
    }
}

// == Cell prepare =====================================================================================================
// One thread per cell slot. Exposure = the number of pixels that see the cell, smoothed over frames. When it GROWS, part
// of the cell that the history never saw appears (the camera comes closer, an occluder moves away): the history is
// scaled down by (old / new)^strength. A shrinking exposure keeps the history (what stays visible was in it). Cells no
// pixel queries age and are freed.
[shader("compute")]
[numthreads(RT_LAB2_CELL_PASS_X, 1, 1)]
func entry_rt_lab2_prepare(uint dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_prepare_push;
    const RtLab2Settings s = *push.settings;
    if (dtid >= rt_lab2_capacity(s))
    {
        return;
    }
    RtLab2Cell* cell = &push.attach.cells[dtid];
    if (cell.key == 0u)
    {
        return;
    }
    const uint pixel_count = cell.pixel_count;
    if (pixel_count > 0u)
    {
        cell.frames_untouched = 0u;
        const float now = float(pixel_count);
        // The coverage of the history: last frame's pixel count, the smoothed one when the cell was not queried last frame.
        const float previous = cell.last_pixel_count > 0u ? float(cell.last_pixel_count) : cell.exposure;
        if (previous > 0.0f && s.exposure_strength > 0.0f && cell.sample_count > 0.0f)
        {
            // The new frame counts proportionally to its pixel coverage relative to the history's: coverage 4 -> 16 pixels = the new frame
            // counts 4x more against the history (history samples / 4), 16 -> 4 = 4x less (history samples x 4, the window cap still limits them).
            const float limit = 1.0f / max(s.exposure_min_scale, 1e-3f);
            cell.sample_count *= clamp(pow(previous / now, s.exposure_strength), s.exposure_min_scale, limit);
        }
        cell.exposure = cell.exposure <= 0.0f ? now : lerp(cell.exposure, now, saturate(s.exposure_alpha));
    }
    else
    {
        cell.frames_untouched = cell.frames_untouched + 1u;
        if (cell.frames_untouched > uint(max(s.max_untouched_frames, 0)))
        {
            cell.key = 0u;
            cell.frames_untouched = 0u;
            cell.pixel_count = 0u;
            cell.ray_count = 0u;
            cell.sum_r = 0;
            cell.sum_g = 0;
            cell.sum_b = 0;
            cell.radiance_r = 0.0f;
            cell.radiance_g = 0.0f;
            cell.radiance_b = 0.0f;
            cell.sample_count = 0.0f;
            cell.exposure = 0.0f;
            cell.last_pixel_count = 0u;
            cell.last_ray_count = 0u;
            cell.last_mean_r = 0.0f;
            cell.last_mean_g = 0.0f;
            cell.last_mean_b = 0.0f;
            cell.rec_next = 0u;
            cell.rec_valid = 0u;
            cell.meas_z = 0.0f;
            cell.meas_r = 0.0f;
            cell.gk_x = 0.0f;
            cell.gk_p = 0.0f;
            cell.gk_pending = 0.0f;
            cell.gk_applied = 0.0f;
            cell.gk_loss = 0.0f;
            cell.grad_count = 0u;
            cell.grad_sum_prec = 0;
            cell.grad_sum_zprec = 0;
        }
    }
}

// The history window of a cell in samples (RtLab2Settings::history_samples_per_pixel): proportional to the pixels that see the cell.
func rt_lab2_history_cap(RtLab2Cell* cell, RtLab2Settings s) -> float
{
    const float upper = max(s.max_history_samples, 1.0f);
    if (s.history_samples_per_pixel <= 0.0f)
    {
        return upper;
    }
    return clamp(s.history_samples_per_pixel * max(cell.exposure, 1.0f), min(max(s.min_history_samples, 1.0f), upper), upper);
}

// == Demand ===========================================================================================================
// The pixel's cell decides how many rays the pixel asks for. x = cell samples / (target x the cell's history window): a cell with no data
// asks for the maximum (RTGI curve: amplitude exp(-4 x), 0 at / after the target), a converged cell for the base ray only. The old demand pass
// wrote 1 ray for every geometry pixel (and the base total), the extras are added here.
[shader("compute")]
[numthreads(RT_LAB2_PASS_X, RT_LAB2_PASS_Y, 1)]
func entry_rt_lab2_demand(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_demand_push;
    uint extra_rays = 0u;
    if (all(dtid < push.size))
    {
        const RtLab2Settings s = *push.settings;
        const uint desired = push.attach.ray_demand.get()[dtid];
        const uint cell_index = push.attach.pixel_cell.get()[dtid];
        const uint max_rays = uint(max(s.max_rays_per_pixel, 1));
        if (desired > 0u && cell_index != RT_LAB2_INVALID_CELL && max_rays > 1u)
        {
            RtLab2Cell* cell = &push.attach.cells[cell_index];
            const float target = max(s.redistribution_target, 1e-3f) * rt_lab2_history_cap(cell, s);
            const float extra = calc_desired_extra_rays(cell.sample_count, target, float(max_rays - 1u));
            extra_rays = min(uint(extra + 0.5f), max_rays - 1u);
            push.attach.ray_demand.get()[dtid] = desired + extra_rays;
        }
    }
    const uint wave_extra = WaveActiveSum(extra_rays);
    if (WaveIsFirstLane() && wave_extra > 0u)
    {
        InterlockedAdd(push.attach.ray_counters->extra_demand, wave_extra);
    }
}

#define RT_LAB2_TILE_PIXELS 64

// Per frame rotation of the Bayer rank order (the old distribute pass's rule).
func rt_lab_base_rank_rotation_lab2(RtLabSettings settings, float trunk_flt_frame_index) -> uint
{
    const uint frame = settings.animate_noise != 0 ? uint(trunk_flt_frame_index) : 0u;
    return uint(frac(float(frame) * 0.61803398875f) * float(RT_LAB2_TILE_PIXELS));
}

// == Ray redistribution on quad candidates ============================================================================
func rt_lab2_bayer_m2(uint a, uint b) -> uint
{
    return 2u * ((a ^ b) & 1u) + (a & 1u);
}

// 8x8 ordered dither rank in [0, 63] (same as the old distribute pass).
func rt_lab2_bayer_8x8(uint2 p) -> uint
{
    return 16u * rt_lab2_bayer_m2(p.x, p.y) + 4u * rt_lab2_bayer_m2(p.x >> 1u, p.y >> 1u) + rt_lab2_bayer_m2(p.x >> 2u, p.y >> 2u);
}

// The candidate of every quad of lab pixels and its ray demand: the sub pixel rotates every frame (+ a per quad offset so neighbor quads pick different
// ones), the next sub pixel replaces a sky pixel. Demand = 1 base ray + the extras the pixel's cell asks for (calc_desired_extra_rays over its
// history fill, max_rays_per_pixel). Totals into counters.
[shader("compute")]
[numthreads(RT_LAB2_PASS_X, RT_LAB2_PASS_Y, 1)]
func entry_rt_lab2_select(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_select_push;
    uint packed = 0u;
    uint base_count = 0u;
    uint extra_count = 0u;
    if (all(dtid < push.size))
    {
        const RtLab2Settings s = *push.settings;
        const uint frame = push.attach.globals.rt_lab_settings.animate_noise != 0 ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
        const uint first_sub = (frame + (dtid.x & 1u) + ((dtid.y & 1u) << 1u)) & 3u;
        for (uint i = 0u; i < 4u; ++i)
        {
            const uint sub = (first_sub + i) & 3u;
            const uint2 pixel = dtid * 2u + uint2(sub & 1u, sub >> 1u);
            if (packed == 0u && all(pixel < push.lab_size) && push.attach.lab_depth.get()[pixel] > 0.0f)
            {
                uint demand = 1u;
                const uint max_rays = uint(clamp(s.max_rays_per_pixel, 1, 32));
                const uint cell_index = push.attach.pixel_cell.get()[pixel];
                if (s.redistribution_enabled != 0 && max_rays > 1u && cell_index != RT_LAB2_INVALID_CELL)
                {
                    RtLab2Cell* cell = &push.attach.cells[cell_index];
                    const float target = max(s.redistribution_target, 1e-3f) * rt_lab2_history_cap(cell, s);
                    const float extra = calc_desired_extra_rays(cell.sample_count, target, float(max_rays - 1u));
                    demand = min(1u + uint(extra + 0.5f), max_rays);
                }
                packed = demand | (sub << 8u);
                base_count = 1u;
                extra_count = demand - 1u;
            }
        }
        push.attach.candidate.get()[dtid] = packed;
    }
    const uint wave_base = WaveActiveSum(base_count);
    const uint wave_extra = WaveActiveSum(extra_count);
    if (WaveIsFirstLane())
    {
        InterlockedAdd(push.attach.counters[0], wave_base);
        InterlockedAdd(push.attach.counters[1], wave_extra);
    }
}

groupshared uint gs_quad_scan[RT_LAB2_TILE_PIXELS];
groupshared uint gs_quad_order[RT_LAB2_TILE_PIXELS];
groupshared uint gs_quad_tile_base;
groupshared uint gs_quad_tile_extra;
groupshared uint gs_quad_tile_offset;

// Inclusive prefix sum over the 64 threads of the group (Hillis Steele). Every thread of the group must call it.
func rt_lab2_scan64(uint value, uint slot) -> uint
{
    gs_quad_scan[slot] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < RT_LAB2_TILE_PIXELS; stride <<= 1u)
    {
        const uint other = slot >= stride ? gs_quad_scan[slot - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gs_quad_scan[slot] += other;
        GroupMemoryBarrierWithGroupSync();
    }
    const uint result = gs_quad_scan[slot];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

// The budget drain of the old distribute pass (RTGI rule), one workgroup per 8x8 tile of quads, parallel: the global fractions (base request min_ray_budget of the
// candidates paid first, extras get the rest at one ratio, what they leave raises the base coverage) give every tile floor(fraction x its demand) of base and extra
// rays. Base rays go to the geometry candidates in rotated Bayer order, extras by a cumulative demand scan with a per frame / tile phase. Then ray list offsets
// by a scan (one atomic per tile), the ray list entries and ray_alloc (written for all 4 pixels of the quad: only the candidate can have rays).
[shader("compute")]
[numthreads(RT_LAB2_PASS_X, RT_LAB2_PASS_Y, 1)]
func entry_rt_lab2_distribute(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rt_lab2_distribute_push;
    const RtLabSettings lab_settings = push.attach.globals.rt_lab_settings;
    const uint slot = gtid.y * RT_LAB2_PASS_X + gtid.x;
    const bool in_bounds = all(dtid < push.size);
    const uint packed = in_bounds ? push.attach.candidate.get()[dtid] : 0u;
    const uint demand = packed & 0xFFu;
    const uint sub = (packed >> 8u) & 3u;
    const uint base_present = demand > 0u ? 1u : 0u;
    const uint extra_demand_i = demand > 0u ? demand - 1u : 0u;

    if (slot == 0u)
    {
        gs_quad_tile_base = 0u;
        gs_quad_tile_extra = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    InterlockedAdd(gs_quad_tile_base, base_present);
    InterlockedAdd(gs_quad_tile_extra, extra_demand_i);
    GroupMemoryBarrierWithGroupSync();
    const uint tile_base_demand = gs_quad_tile_base;
    const uint tile_extra_demand = gs_quad_tile_extra;

    // Global fractions (identical in every tile).
    const uint pixel_count = push.lab_size.x * push.lab_size.y;
    const uint budget = min(uint(float(pixel_count) * max(lab_settings.ray_budget, 0.0f)), push.ray_capacity);
    const uint global_base = push.attach.counters[0];
    const uint global_extra = push.attach.counters[1];
    uint base_given = base_present;
    uint extra_given = extra_demand_i;
    if (global_base + global_extra > budget)
    {
        const float min_budget = saturate(lab_settings.min_ray_budget);
        const float min_base_spend = min(float(global_base) * min_budget, float(budget));
        const float extra_ratio = global_extra > 0u ? saturate((float(budget) - min_base_spend) / float(global_extra)) : 0.0f;
        const float base_fraction = global_base > 0u
            ? min(min(max((float(budget) - float(global_extra) * extra_ratio) / float(global_base), min_budget), 1.0f), float(budget) / float(global_base))
            : 0.0f;
        const uint tile_base_budget = min(uint(float(tile_base_demand) * base_fraction), tile_base_demand);
        const uint tile_extra_budget = min(uint(float(tile_extra_demand) * extra_ratio), tile_extra_demand);

        // Base: the first tile_base_budget geometry candidates in the rotated Bayer order.
        const uint rotation = rt_lab_base_rank_rotation_lab2(lab_settings, push.attach.globals.trunk_flt_frame_index);
        const uint order = (rt_lab2_bayer_8x8(gtid) + RT_LAB2_TILE_PIXELS - rotation) % RT_LAB2_TILE_PIXELS;
        gs_quad_order[order] = base_present;
        GroupMemoryBarrierWithGroupSync();
        const uint ordered_present = gs_quad_order[slot];
        GroupMemoryBarrierWithGroupSync();
        const uint ordered_scan = rt_lab2_scan64(ordered_present, slot);
        gs_quad_order[slot] = ordered_scan; // by order
        GroupMemoryBarrierWithGroupSync();
        const uint k = gs_quad_order[order] - base_present; // exclusive geometry count in order
        base_given = (base_present != 0u && k < tile_base_budget) ? 1u : 0u;

        // Extras: cumulative scan in slot order, target(prefix) = floor(prefix / total x budget + phase), differenced.
        const uint frame = lab_settings.animate_noise != 0 ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
        const uint tile_id = gid.y * ((push.size.x + RT_LAB2_PASS_X - 1u) / RT_LAB2_PASS_X) + gid.x;
        const float phase = frac(float(frame) * 0.61803398875f + float(tile_id) * 0.7548776662f + 0.5f);
        const float inverse_extra = tile_extra_demand > 0u ? rcp(float(tile_extra_demand)) : 0.0f;
        const uint inclusive = rt_lab2_scan64(extra_demand_i, slot);
        const uint exclusive = inclusive - extra_demand_i;
        const uint target_now = tile_extra_demand > 0u ? min(uint(saturate(float(inclusive) * inverse_extra) * float(tile_extra_budget) + phase), tile_extra_budget) : 0u;
        const uint target_before = tile_extra_demand > 0u ? min(uint(saturate(float(exclusive) * inverse_extra) * float(tile_extra_budget) + phase), tile_extra_budget) : 0u;
        extra_given = min(target_now - min(target_before, target_now), extra_demand_i);
    }

    // Ray list offsets.
    const uint rays = base_given + extra_given;
    const uint ray_inclusive = rt_lab2_scan64(rays, slot);
    const uint ray_exclusive = ray_inclusive - rays;
    if (slot == RT_LAB2_TILE_PIXELS - 1u)
    {
        uint tile_offset = 0u;
        InterlockedAdd(push.attach.ray_counters->ray_count, ray_inclusive, tile_offset);
        gs_quad_tile_offset = tile_offset;
    }
    GroupMemoryBarrierWithGroupSync();
    if (!in_bounds)
    {
        return;
    }
    const uint offset = gs_quad_tile_offset + ray_exclusive;
    const uint count = offset < budget ? min(rays, budget - offset) : 0u;
    const uint2 chosen = dtid * 2u + uint2(sub & 1u, sub >> 1u);
    for (uint i = 0u; i < 4u; ++i)
    {
        const uint2 pixel = dtid * 2u + uint2(i & 1u, i >> 1u);
        if (all(pixel < push.lab_size))
        {
            const bool is_chosen = demand > 0u && all(pixel == chosen);
            push.attach.ray_alloc.get()[pixel] = is_chosen ? uint2(offset, count) : uint2(0u, 0u);
        }
    }
    const uint packed_pixel = chosen.x | (chosen.y << 16u);
    for (uint k = 0u; k < count; ++k)
    {
        push.attach.ray_list[offset + k] = packed_pixel;
    }
}

// == Rays shot debug view ==============================================================================================
[shader("compute")]
[numthreads(RT_LAB2_PASS_X, RT_LAB2_PASS_Y, 1)]
func entry_rt_lab2_rays_debug(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_rays_debug_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    if (debug_mode != DEBUG_DRAW_MODE_RTGI_DIFFUSE_RAYS_SHOT && debug_mode != DEBUG_DRAW_MODE_RTGI_TOTAL_RAYS_SHOT)
    {
        return;
    }
    const RtLab2Settings s = *push.settings;
    const float count = float(push.attach.ray_alloc.get()[dtid].y);
    const float3 color = push.attach.lab_depth.get()[dtid] <= 0.0f ? float3(0.0f, 0.0f, 0.0f) : Heatmap(count / float(max(s.max_rays_per_pixel, 1)));
    // write_debug_image keeps the first writer unless the new alpha is larger (and the alpha ranges select the compositing, so it must not change): the old distribute
    // pass draws this view first, correctly normalized since the overrides give it the real max rays (rt_lab2_frame_update). Equal alpha = this never overwrites it.
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
}

// == Accumulate =======================================================================================================
// The pixel's ray mean (the old lab's gather: lab_raw_diffuse, .a = ray count) goes into the pixel's cell (and the neighbor cascade's cell
// inside the level blend region). Pixels of one wave mostly share a cell: summed per cell in the wave first, then one 64 bit fixed point
// atomic per channel. Every lane must call it (wave ops).
func rt_lab2_accumulate_cell(RtLab2Cell* cells, uint cell_index, float3 ray_sum, float ray_count)
{
    bool pending = cell_index != RT_LAB2_INVALID_CELL;
    while (WaveActiveAnyTrue(pending))
    {
        const uint current = WaveActiveMin(pending ? cell_index : 0xFFFFFFFFu);
        const bool mine = pending && cell_index == current;
        const float3 sum = WaveActiveSum(mine ? ray_sum : float3(0.0f, 0.0f, 0.0f));
        const float count = WaveActiveSum(mine ? ray_count : 0.0f);
        if (mine && WavePrefixCountBits(mine) == 0u)
        {
            RtLab2Cell* cell = &cells[current];
            AtomicAddU64(cell.sum_r, uint64_t(min(sum.r * RT_LAB2_RADIANCE_FIXED_SCALE + 0.5f, 1.0e15f)));
            AtomicAddU64(cell.sum_g, uint64_t(min(sum.g * RT_LAB2_RADIANCE_FIXED_SCALE + 0.5f, 1.0e15f)));
            AtomicAddU64(cell.sum_b, uint64_t(min(sum.b * RT_LAB2_RADIANCE_FIXED_SCALE + 0.5f, 1.0e15f)));
            InterlockedAdd(cell.ray_count, uint(count + 0.5f));
        }
        pending = pending && !mine;
    }
}

[shader("compute")]
[numthreads(RT_LAB2_PASS_X, RT_LAB2_PASS_Y, 1)]
func entry_rt_lab2_accumulate(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_accumulate_push;
    const bool in_bounds = all(dtid < push.size);
    uint cell_index = RT_LAB2_INVALID_CELL;
    uint cell_index2 = RT_LAB2_INVALID_CELL;
    float3 ray_sum = float3(0.0f, 0.0f, 0.0f);
    float ray_count = 0.0f;
    if (in_bounds)
    {
        const RtLab2Settings s = *push.settings;
        const uint candidate = push.attach.pixel_cell.get()[dtid];
        const float4 raw = push.attach.lab_raw_diffuse.get()[dtid];
        if (candidate != RT_LAB2_INVALID_CELL && raw.a > 0.0f && all(isfinite(raw.rgb)))
        {
            const float3 mean = max(raw.rgb, 0.0f);
            cell_index = candidate;
            cell_index2 = push.attach.pixel_cell2.get()[dtid];
            ray_count = raw.a;
            ray_sum = mean * raw.a;
        }
    }
    rt_lab2_accumulate_cell(push.attach.cells, cell_index, ray_sum, ray_count);
    rt_lab2_accumulate_cell(push.attach.cells, cell_index2, ray_sum, ray_count);
}

// == Cell integrate ===================================================================================================
// One thread per cell slot: the frame's ray sums become the cell mean, merged into the history with the sample count
// blend n_new / (n_old + n_new) (n_old capped at max_history_samples = the window). Clears the per frame counters.
[shader("compute")]
[numthreads(RT_LAB2_CELL_PASS_X, 1, 1)]
func entry_rt_lab2_cell_integrate(uint dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_cell_integrate_push;
    const RtLab2Settings s = *push.settings;
    if (dtid >= rt_lab2_capacity(s))
    {
        return;
    }
    RtLab2Cell* cell = &push.attach.cells[dtid];
    if (cell.key == 0u)
    {
        return;
    }
    const float history_before = cell.radiance_r + cell.radiance_g + cell.radiance_b;
    const float samples_before = cell.sample_count;
    // Temporal gradient: the cell's gradient (the integrated spatial blur of all its half res pixels, else its own re-trace) is accumulated as EVIDENCE of a change of the lighting
    // that the history does not know yet: S = decay S + z, V = decay^2 V + variance (a leaky sum of the per frame changes and its variance). A consistent small drift (1 % per
    // frame: far below the noise of ONE frame's measurement) builds up significance |S| / sqrt(V) over frames; once it passes the threshold a smooth fraction of S is applied to
    // the history (x exp) and taken out of S. A single measurement far outside the noise (shock gate) is applied at once. The sample count is cut by the unapplied
    // evidence S^2 (a real change the history lags behind), not by the filter's variance (that would also cut static cells).
    // (Fields reused: gradient_kalman_q = decay, gradient_kalman_gate = significance threshold in sigmas, gradient_glide = fraction of S consumed per frame once significant.)
    if (s.gradient_enabled != 0 && cell.sample_count > 0.0f)
    {
        float z = 0.0f;
        float variance = 0.0f;
        bool measured = false;
        if (cell.grad_count > 0u && cell.grad_sum_prec > 0)
        {
            const float prec_sum = float(cell.grad_sum_prec) / RT_LAB2_GRADIENT_PREC_SCALE;
            const float zprec_sum = float(cell.grad_sum_zprec) / RT_LAB2_GRADIENT_PREC_SCALE;
            z = zprec_sum / max(prec_sum, 1e-6f) - RT_LAB2_GRADIENT_Z_BIAS;
            variance = float(cell.grad_count) / max(prec_sum, 1e-6f); // 1 / the mean precision of the pixels (they share taps: not a sum)
            measured = true;
        }
        else if (cell.meas_r > 0.0f)
        {
            z = cell.meas_z;
            variance = cell.meas_r;
            measured = true;
        }
        const float decay = clamp(s.gradient_kalman_q, 0.0f, 0.999f);
        const float threshold = max(s.gradient_kalman_gate, 0.1f);
        float evidence = cell.gk_x;
        float evidence_variance = cell.gk_p;
        float shock = 0.0f;
        if (measured)
        {
            variance = max(variance, 1e-6f);
            const float shock_gate = threshold + 3.0f;
            if (z * z > shock_gate * shock_gate * variance)
            {
                shock = z; // far outside the noise of one measurement: applied at once, the evidence restarts
                evidence = 0.0f;
                evidence_variance = 0.0f;
            }
            else
            {
                evidence = decay * evidence + z;
                evidence_variance = decay * decay * evidence_variance + variance;
            }
        }
        else
        {
            evidence *= decay;
            evidence_variance *= decay * decay;
        }
        const float significance = abs(evidence) / sqrt(max(evidence_variance, 1e-8f));
        const float consumed = smoothstep(threshold, threshold + 1.0f, significance) * saturate(s.gradient_glide);
        const float applied_evidence = consumed * evidence;
        evidence -= applied_evidence;
        evidence_variance *= (1.0f - consumed) * (1.0f - consumed);
        // One frame never moves the history by more than a factor e^1.4 = 4: repeated bogus measurements cannot run away (the rays pull back, ~x20 per frame did not).
        const float applied = clamp(shock + applied_evidence, -1.4f, 1.4f);
        const float loss = evidence * evidence;
        cell.gk_x = evidence;
        cell.gk_p = evidence_variance;
        cell.gk_pending = 0.0f;
        cell.gk_applied = applied;
        cell.gk_loss = loss;
        const float scale = exp(applied);
        cell.radiance_r *= scale;
        cell.radiance_g *= scale;
        cell.radiance_b *= scale;
        const float samples = cell.sample_count;
        // Cut = k S^2 + gain S^4 (gain = gradient_sample_variance): small drifts (sun sweep, S ~ 0.05) stay lightly cut, large changes (night, S ~ 0.3) cut hard.
        cell.sample_count = max(min(samples, 2.0f), samples / (1.0f + samples * (max(s.gradient_cap_k, 0.0f) * loss + max(s.gradient_sample_variance, 0.0f) * loss * loss)));
    }
    else
    {
        cell.gk_applied = 0.0f;
        cell.gk_loss = 0.0f;
    }
    const uint rays = cell.ray_count;
    if (rays > 0u)
    {
        const float inverse = 1.0f / (float(rays) * RT_LAB2_RADIANCE_FIXED_SCALE);
        const float3 mean = float3(float(cell.sum_r), float(cell.sum_g), float(cell.sum_b)) * inverse;
        const float n_new = s.max_frame_samples > 0.0f ? min(float(rays), s.max_frame_samples) : float(rays); // the mean stays, only the weight of the frame is capped
        const float cap = rt_lab2_history_cap(cell, s);
        float3 radiance = mean;
        float sample_count = min(n_new, cap);
        if (cell.sample_count > 0.0f)
        {
            const float n_old = min(cell.sample_count, cap);
            radiance = lerp(rt_lab2_cell_radiance(cell), mean, n_new / (n_old + n_new));
            sample_count = min(n_old + n_new, cap);
        }
        cell.radiance_r = radiance.r;
        cell.radiance_g = radiance.g;
        cell.radiance_b = radiance.b;
        cell.sample_count = sample_count;
        cell.last_mean_r = mean.r;
        cell.last_mean_g = mean.g;
        cell.last_mean_b = mean.b;
    }
    // A non finite or negative cell must not live on (it would poison the lookup and the integrator history for good): drop its state.
    const float radiance_sum = cell.radiance_r + cell.radiance_g + cell.radiance_b;
    if (!isfinite(radiance_sum) || !isfinite(cell.sample_count) || !isfinite(cell.gk_x) || !isfinite(cell.gk_p) || cell.radiance_r < 0.0f || cell.radiance_g < 0.0f ||
        cell.radiance_b < 0.0f || radiance_sum > 1e15f)
    {
        cell.radiance_r = 0.0f;
        cell.radiance_g = 0.0f;
        cell.radiance_b = 0.0f;
        cell.sample_count = 0.0f;
        cell.gk_x = 0.0f;
        cell.gk_p = 0.0f;
        cell.gk_pending = 0.0f;
        cell.gk_applied = 0.0f;
        cell.gk_loss = 0.0f;
    }
    // The cell's total log change of this frame (gradient correction AND merged rays): the pixels' integrator history follows it (it blends in at 3 - 13 % per frame, the
    // pixels lagged the cells by 2 - 5x after a fast light change). A deadzone of one standard error of the cell's mean keeps the merge noise out.
    cell.gk_pending = 0.0f;
    if (samples_before > 0.0f && cell.sample_count > 0.0f)
    {
        const float change = log((cell.radiance_r + cell.radiance_g + cell.radiance_b + 1e-2f) / (history_before + 1e-2f));
        const float standard_error = rsqrt(max(samples_before, 1.0f));
        cell.gk_pending = clamp(sign(change) * max(abs(change) - standard_error, 0.0f), -1.4f, 1.4f);
    }
    cell.last_pixel_count = cell.pixel_count;
    cell.last_ray_count = rays;
    cell.sum_r = 0;
    cell.sum_g = 0;
    cell.sum_b = 0;
    cell.ray_count = 0u;
    cell.pixel_count = 0u;
    cell.grad_count = 0u;
    cell.grad_sum_prec = 0;
    cell.grad_sum_zprec = 0;
    cell.meas_r = 0.0f;
    cell.rec_valid = min(cell.rec_next, uint(RT_LAB2_GRADIENT_RECORDS)); // the re-trace of the next frame
    cell.rec_next = 0u;
}

// == Temporal gradient ================================================================================================
// Record: one thread per ray list entry. A ray wins a record slot of its pixel's cell with the probability that spreads the slots over the cell's rays (the cell's
// last ray count), at most the coverage limited number of slots. The record = everything the re-trace needs to launch the SAME ray (origin, quantized direction,
// the hit shading seed) + the luminance now. The re-trace of the next frame runs before this pass overwrites the slots.
[shader("compute")]
[numthreads(RT_LAB_TRACE_THREADS, 1, 1)]
func entry_rt_lab2_gradient_record(uint ray_index : SV_DispatchThreadID)
{
    let push = rt_lab2_gradient_record_push;
    const RtLabSettings lab_settings = push.attach.globals.rt_lab_settings;
    if (ray_index >= min(push.attach.ray_counters->ray_count, rt_lab_ray_budget(push.size, lab_settings, push.ray_capacity)))
    {
        return;
    }
    const RtLab2Settings s = *push.settings;
    const uint packed_pixel = push.attach.ray_list[ray_index];
    const uint2 pixel = uint2(packed_pixel & 0xFFFFu, packed_pixel >> 16u);
    const uint cell_index = push.attach.pixel_cell.get()[pixel];
    if (cell_index == RT_LAB2_INVALID_CELL)
    {
        return;
    }
    RtLab2Cell* cell = &push.attach.cells[cell_index];
    const float coverage = max(cell.exposure, float(cell.last_pixel_count));
    const uint allowed = clamp(uint(ceil(float(RT_LAB2_GRADIENT_RECORDS) * coverage / max(s.gradient_full_coverage, 1.0f))), 1u, uint(RT_LAB2_GRADIENT_RECORDS));
    const float probability = min(1.0f, 1.5f * float(allowed) / max(float(cell.last_ray_count), 1.0f));
    const uint frame = lab_settings.animate_noise != 0 ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
    const float random = float(rt_lab2_hash_u32(ray_index * 0x9E3779B1u ^ rt_lab2_hash_u32(frame + 0x7F4A7C15u))) * (1.0f / 4294967296.0f);
    if (random >= probability)
    {
        return;
    }
    uint slot = 0u;
    InterlockedAdd(cell.rec_next, 1u, slot);
    if (slot >= allowed)
    {
        return;
    }
    const float depth = push.attach.lab_depth.get()[pixel];
    const float3 face_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[pixel]);
    float3 world_pos;
    float t_min;
    const float3 origin = rt_lab_ray_launch(push.attach.globals.view_camera, pixel, push.size, depth, face_normal, world_pos, t_min);
    const RtLabRay ray = push.attach.rays[ray_index];
    const uint pixel_ray = ray_index - push.attach.ray_alloc.get()[pixel].x;
    const uint hit_seed = rt_lab_hit_seed(rt_lab_ray_seed(lab_settings, push.attach.globals.trunk_flt_frame_index, pixel, pixel_ray));
    RtLab2GradientRecord record;
    record.origin = origin;
    record.t_min = t_min;
    record.direction = ray.direction;
    record.data = RT_LAB_GRADIENT_RECORD_MAGIC | hit_seed;
    record.old_luminance = rt_lab_luminance(ray.radiance);
    record.pad = 0u;
    push.attach.records[cell_index * uint(RT_LAB2_GRADIENT_RECORDS) + slot] = record;
}

// Sums of precision weighted gradient values into the cells, grouped per cell inside the wave. Every lane must call it (wave ops).
// Sub pixels of every quad evaluated per frame (1 = one rotating candidate per quad, 4 = every full res pixel: small cells get a pooled gradient every frame).
#ifndef RT_LAB2_GRADIENT_SUBPIXELS
#define RT_LAB2_GRADIENT_SUBPIXELS 4u
#endif
func rt_lab2_gradient_add(RtLab2Cell* cells, uint cell_index, float precision, float z)
{
    bool pending = cell_index != RT_LAB2_INVALID_CELL;
    const float biased = clamp(z, -RT_LAB2_GRADIENT_MAX_LOG, RT_LAB2_GRADIENT_MAX_LOG) + RT_LAB2_GRADIENT_Z_BIAS;
    while (WaveActiveAnyTrue(pending))
    {
        const uint current = WaveActiveMin(pending ? cell_index : 0xFFFFFFFFu);
        const bool mine = pending && cell_index == current;
        const float precision_sum = WaveActiveSum(mine ? precision : 0.0f);
        const float weighted_sum = WaveActiveSum(mine ? precision * biased : 0.0f);
        const uint count = WaveActiveCountBits(mine);
        if (mine && WavePrefixCountBits(mine) == 0u)
        {
            RtLab2Cell* cell = &cells[current];
            AtomicAddU64(cell.grad_sum_prec, uint64_t(precision_sum * RT_LAB2_GRADIENT_PREC_SCALE + 0.5f));
            AtomicAddU64(cell.grad_sum_zprec, uint64_t(weighted_sum * RT_LAB2_GRADIENT_PREC_SCALE + 0.5f));
            InterlockedAdd(cell.grad_count, count);
        }
        pending = pending && !mine;
    }
}

// Spatial: one thread per half res pixel. Candidate = the first geometry pixel of its quad (rotating start, like the ray selection). Taps: the pixel itself + gradient_blur_taps
// (up to 32) on a Vogel spiral (golden angle, per pixel rotation, jittered stratified radius) whose radius follows a gaussian truncated at the blur radius (sigma = radius / 2):
// the gaussian is in the SAMPLING, the tap weight is the old lab's coarse relative world distance + normal edge test (rt_lab_gradient_distance_weight: tolerant on slopes, stops
// depth jumps). A cell counts ONCE (its measurement is one measurement, however many taps land in it). Weights = edge weight x precision (1 / variance of the cell's
// measurement). The blurred gradient is that weighted mean; its variance is the larger of what the measurements' own variances give and the spread of the cells' values
// (dispersion / effective count): a region whose cells disagree is not trusted. Integrated into the pixel's cell.
[shader("compute")]
[numthreads(RT_LAB2_PASS_X, RT_LAB2_PASS_Y, 1)]
func entry_rt_lab2_gradient_spatial(uint2 dtid : SV_DispatchThreadID)
{
    for (uint sub_pass = 0u; sub_pass < RT_LAB2_GRADIENT_SUBPIXELS; ++sub_pass)
    {
        uint cell_out = RT_LAB2_INVALID_CELL;
        uint center_cell2 = RT_LAB2_INVALID_CELL;
        float precision_out = 0.0f;
        float z_out = 0.0f;
        let push = rt_lab2_gradient_spatial_push;
        if (all(dtid < push.size))
        {
            const RtLab2Settings s = *push.settings;
            const CameraInfo camera = push.attach.globals.view_camera;
            const uint frame = push.attach.globals.rt_lab_settings.animate_noise != 0 ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
            const uint first_sub = (RT_LAB2_GRADIENT_SUBPIXELS > 1u) ? sub_pass : ((frame + (dtid.x & 1u) + ((dtid.y & 1u) << 1u)) & 3u);
            bool found = false;
            uint2 center_pixel = uint2(0u, 0u);
            for (uint i = 0u; i < ((RT_LAB2_GRADIENT_SUBPIXELS > 1u) ? 1u : 4u); ++i)
            {
                const uint sub = (first_sub + i) & 3u;
                const uint2 pixel = dtid * 2u + uint2(sub & 1u, sub >> 1u);
                if (!found && all(pixel < push.lab_size) && push.attach.lab_depth.get()[pixel] > 0.0f)
                {
                    found = true;
                    center_pixel = pixel;
                }
            }
            const uint center_cell = found ? push.attach.pixel_cell.get()[center_pixel] : RT_LAB2_INVALID_CELL;
            center_cell2 = found ? push.attach.pixel_cell2.get()[center_pixel] : RT_LAB2_INVALID_CELL;
            if (center_cell != RT_LAB2_INVALID_CELL)
            {
                const float2 inv_size = rcp(float2(push.lab_size));
                const float center_depth = push.attach.lab_depth.get()[center_pixel];
                const float3 center_position = linear_depth_to_world_space(camera, rt_lab_ndc(center_pixel, inv_size), center_depth);
                const float3 center_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[center_pixel]);
                const float rcp_center_pixel_width = rcp(rt_lab_pixel_width_ws(camera, float(push.lab_size.y), center_depth));
                rand_seed(rt_lab2_hash_u32(dtid.x | (dtid.y << 16u)) ^ rt_lab2_hash_u32(frame * 0x2545F491u + 0x1B873593u + sub_pass * 0x9E3779B9u));
                const uint taps = max(uint(clamp(s.gradient_blur_taps, 0, 32)) / 2, 1u);
                const float sigma = max(s.gradient_blur_radius, 1.0f) * 0.5f;
                const float theta = 6.2831853f * rand();
                uint seen[66];
                float tap_weight[66];
                float tap_edge[66];
                float tap_z[66];
                float tap_variance[66];
                uint count = 0u;
                for (uint t = 0u; t <= taps; ++t)
                {
                    int2 tap = int2(center_pixel);
                    float offset_px = 0.0f;
                    if (t > 0u)
                    {
                        const float u = saturate((float(t) - 0.5f + 0.8f * (rand() - 0.5f)) / float(taps));
                        offset_px = sigma * sqrt(-2.0f * log(max(1.0f - u * 0.8646647f, 1e-4f))); // gaussian truncated at 2 sigma = the blur radius
                        const float angle = theta + float(t) * 2.3999632f;
                        tap = clamp(int2(center_pixel) + int2(round(float2(cos(angle), sin(angle)) * offset_px)), int2(0, 0), int2(push.lab_size) - 1);
                    }
                    // A pixel inside the cascade blend region feeds the cells of BOTH cascades (pixel_cell, pixel_cell2): both are taps and both receive the pooled gradient.
                    for (uint c = 0u; c < 2u; ++c)
                    {
                        const uint cell_index = c == 0u ? push.attach.pixel_cell.get()[uint2(tap)] : push.attach.pixel_cell2.get()[uint2(tap)];
                        if (cell_index == RT_LAB2_INVALID_CELL)
                        {
                            continue;
                        }
                        bool duplicate = false;
                        for (uint k = 0u; k < count; ++k)
                        {
                            duplicate = duplicate || seen[k] == cell_index;
                        }
                        const float measurement_variance = push.attach.cells[cell_index].meas_r;
                        if (duplicate || !(measurement_variance > 0.0f))
                        {
                            continue;
                        }
                        float edge_weight = 1.0f;
                        if (t > 0u)
                        {
                            const float tap_depth = push.attach.lab_depth.get()[uint2(tap)];
                            if (tap_depth <= 0.0f)
                            {
                                continue;
                            }
                            const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[uint2(tap)]);
                            const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
                            edge_weight = rt_lab_gradient_distance_weight(camera.position, center_position, center_normal, tap_position, tap_normal, rcp_center_pixel_width,
                                rcp(rt_lab_pixel_width_ws(camera, float(push.lab_size.y), tap_depth)), offset_px);
                            if (edge_weight < 0.02f)
                            {
                                continue;
                            }
                        }
                        seen[count] = cell_index;
                        tap_edge[count] = edge_weight;
                        tap_variance[count] = measurement_variance;
                        tap_z[count] = push.attach.cells[cell_index].meas_z;
                        tap_weight[count] = edge_weight * min(1.0f / measurement_variance, RT_LAB2_GRADIENT_PREC_MAX);
                        count += 1u;
                    }
                }
                if (count > 0u)
                {
                    float weight_sum = 0.0f;
                    float weight_square_sum = 0.0f;
                    float z_sum = 0.0f;
                    float own_variance_numerator = 0.0f; // sum w^2 variance
                    for (uint k = 0u; k < count; ++k)
                    {
                        weight_sum += tap_weight[k];
                        weight_square_sum += tap_weight[k] * tap_weight[k];
                        z_sum += tap_weight[k] * tap_z[k];
                        own_variance_numerator += tap_weight[k] * tap_weight[k] * tap_variance[k];
                    }
                    const float z_mean = z_sum / weight_sum;
                    float variance = own_variance_numerator / (weight_sum * weight_sum);
                    if (count >= 3u)
                    {
                        float dispersion = 0.0f;
                        for (uint k = 0u; k < count; ++k)
                        {
                            dispersion += tap_weight[k] * (tap_z[k] - z_mean) * (tap_z[k] - z_mean);
                        }
                        dispersion /= weight_sum;
                        const float effective_count = weight_sum * weight_sum / weight_square_sum;
                        variance = max(variance, dispersion / max(effective_count, 1.0f));
                    }
                    cell_out = center_cell;
                    z_out = z_mean;
                    precision_out = min(1.0f / max(variance, 1e-6f), RT_LAB2_GRADIENT_PREC_MAX);
                }
            }
        }
        rt_lab2_gradient_add(push.attach.cells, cell_out, precision_out, z_out);
        rt_lab2_gradient_add(push.attach.cells, cell_out == RT_LAB2_INVALID_CELL ? RT_LAB2_INVALID_CELL : center_cell2, precision_out, z_out);
    }
}

// == Lookup ===========================================================================================================
// Fully stochastic filter. Every tap draws a random offset from the kernel distribution (per axis, in cells of the tap's cascade, scaled by
// lookup_distance) and reads the cell that contains the point; the cascade of the tap is picked stochastically around the level transitions
// (rt_lab2_level_blend). There are no weights: the mean of the taps is the filtered value (the kernel convolved with the piecewise constant
// cells, smoother than an interpolation of the cell centers), the noise is averaged by the integrator.
func rt_lab2_sample_kernel(int kernel, float scale) -> float3
{
    float3 d;
    if (kernel == RT_LAB2_KERNEL_TRILINEAR)
    {
        // Triangular: the difference of two uniforms.
        const float3 a = float3(rand(), rand(), rand());
        const float3 b = float3(rand(), rand(), rand());
        d = a - b;
    }
    else if (kernel == RT_LAB2_KERNEL_CUBIC)
    {
        // Cubic B-spline = the sum of 4 uniforms over [-0.5, 0.5].
        d = float3(-2.0f, -2.0f, -2.0f);
        for (uint i = 0u; i < 4u; ++i)
        {
            d += float3(rand(), rand(), rand());
        }
    }
    else
    {
        // Gaussian (Box-Muller), sigma 0.75 cells.
        const float3 u1 = max(float3(rand(), rand(), rand()), float3(1e-6f, 1e-6f, 1e-6f));
        const float3 u2 = float3(rand(), rand(), rand());
        d = sqrt(-2.0f * log(u1)) * cos(2.0f * PI * u2) * 0.75f;
    }
    return d * scale;
}

// The cell around the tap at cell offset k (0..3 per axis = -1..+2 from i0). Offsets scale with lookup_distance and can be
// projected onto the surface plane. false = no cell, or a cell without enough samples.
func rt_lab2_tap_samples(RtLab2Cell* cells, RtLab2Key base_key, float3 position, float3 normal, float cell_size, int3 i0, int3 k, RtLab2Settings s, out float3 radiance, out float samples) -> bool
{
    radiance = float3(0.0f, 0.0f, 0.0f);
    samples = 0.0f;
    const float3 center = (float3(i0 + k - 1) + 0.5f) * cell_size;
    float3 offset = (center - position) * s.lookup_distance;
    if (s.lookup_project != 0)
    {
        offset -= normal * dot(normal, offset);
    }
    RtLab2Key key = base_key;
    key.coord = int3(floor((position + offset) / cell_size));
    const uint cell = rt_lab2_find(cells, key, rt_lab2_capacity(s) - 1u);
    if (cell == RT_LAB2_INVALID_CELL)
    {
        return false;
    }
    if (cells[cell].sample_count < max(float(s.lookup_min_samples), 1e-6f))
    {
        return false;
    }
    radiance = rt_lab2_cell_radiance(&cells[cell]);
    samples = cells[cell].sample_count;
    return true;
}

func rt_lab2_tap(RtLab2Cell* cells, RtLab2Key base_key, float3 position, float3 normal, float cell_size, int3 i0, int3 k, RtLab2Settings s, out float3 radiance) -> bool
{
    float samples;
    return rt_lab2_tap_samples(cells, base_key, position, normal, cell_size, i0, k, s, radiance, samples);
}

[shader("compute")]
[numthreads(RT_LAB2_PASS_X, RT_LAB2_PASS_Y, 1)]
func entry_rt_lab2_lookup(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_lookup_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const RtLab2Settings s = *push.settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth <= 0.0f)
    {
        push.attach.lookup.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(float2(push.size))), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    const RtLab2LevelBlend blend = rt_lab2_level_blend(length(position - camera.position), rt_lab2_pixel_width_per_distance(camera, float(push.size.y)), s);
    const uint normal_bucket = rt_lab2_normal_bucket(normal);
    const uint taps = clamp(uint(max(s.lookup_taps, 1)), 1u, uint(RT_LAB2_MAX_LOOKUP_TAPS));
    const uint frame = s.animate_noise != 0 ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
    rand_seed(rt_lab2_hash_u32(dtid.x | (dtid.y << 16u)) ^ rt_lab2_hash_u32(frame + 0x9E3779B9u));

    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float found = 0.0f;
    for (uint i = 0u; i < taps; ++i)
    {
        // The cascade of this tap.
        const bool use_other = blend.p_other > 0.0f && rand() < blend.p_other;
        const uint level = use_other ? blend.other : blend.primary;
        const float cell_size = rt_lab2_cell_size(level, s);
        float3 offset = rt_lab2_sample_kernel(s.lookup_kernel, s.lookup_distance) * cell_size;
        if (s.lookup_project != 0)
        {
            offset -= normal * dot(normal, offset);
        }
        RtLab2Key key;
        key.level = level;
        key.normal_bucket = normal_bucket;
        key.coord = int3(floor((position + offset) / cell_size));
        const uint cell = rt_lab2_find(push.attach.cells, key, rt_lab2_capacity(s) - 1u);
        if (cell != RT_LAB2_INVALID_CELL && push.attach.cells[cell].sample_count >= max(float(s.lookup_min_samples), 1e-6f))
        {
            sum += rt_lab2_cell_radiance(&push.attach.cells[cell]);
            found += 1.0f;
        }
    }

    // No tap found a cell: the pixel's own cell.
    const RtLab2Key own_key = rt_lab2_make_key_at_level(position, normal, blend.primary, s);
    const uint own_cell = rt_lab2_find(push.attach.cells, own_key, rt_lab2_capacity(s) - 1u);
    float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (found > 0.0f)
    {
        result = float4(sum / found, 1.0f);
    }
    else if (own_cell != RT_LAB2_INVALID_CELL && push.attach.cells[own_cell].sample_count > 0.0f)
    {
        result = float4(rt_lab2_cell_radiance(&push.attach.cells[own_cell]), 1.0f);
    }
    push.attach.lookup.get()[dtid] = result;

    if (debug_mode == DEBUG_DRAW_MODE_RT_LAB2_CELLS || debug_mode == DEBUG_DRAW_MODE_RT_LAB2_CELL_SAMPLES ||
        debug_mode == DEBUG_DRAW_MODE_RT_LAB2_CELL_EXPOSURE || debug_mode == DEBUG_DRAW_MODE_RT_LAB2_LOOKUP ||
        debug_mode == DEBUG_DRAW_MODE_RT_LAB2_TEMPORAL_GRADIENT || debug_mode == DEBUG_DRAW_MODE_RT_LAB2_TEMPORAL_GRADIENT_LOSS)
    {
        float3 color = float3(1.0f, 0.0f, 1.0f) * 0.25f; // no cell
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB2_LOOKUP)
        {
            color = result.rgb / RTGI_RADIANCE_SCALE * PI * push.attach.globals.exposure;
        }
        else if (own_cell != RT_LAB2_INVALID_CELL)
        {
            RtLab2Cell* cell = &push.attach.cells[own_cell];
            if (debug_mode == DEBUG_DRAW_MODE_RT_LAB2_CELLS)
            {
                color = hsv2rgb(float3(IdFloatScramble(own_cell & 0xFFFFu), 0.8f, 1.0f));
            }
            else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB2_TEMPORAL_GRADIENT)
            {
                // The temporal gradient of the cell as a log change: what was applied to the history this frame + the evidence collected but not applied yet.
                // Red = the light got brighter, blue = darker, full color at a change of e^0.06 (6 %); dark gray = the cell is quiet.
                const float change = cell.gk_applied + cell.gk_x;
                const float magnitude = saturate(abs(change) * 16.0f);
                color = float3(0.04f, 0.04f, 0.04f) + (change > 0.0f ? float3(magnitude, 0.0f, 0.0f) : float3(0.0f, 0.0f, magnitude));
            }
            else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB2_TEMPORAL_GRADIENT_LOSS)
            {
                color = Heatmap(saturate(cell.gk_loss * 20.0f));
            }
            else if (debug_mode == DEBUG_DRAW_MODE_RT_LAB2_CELL_SAMPLES)
            {
                color = Heatmap(saturate(cell.sample_count / rt_lab2_history_cap(cell, s)));
            }
            else
            {
                color = Heatmap(saturate(log2(1.0f + cell.exposure) / 12.0f));
            }
        }
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
            float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
    }
}

// == Integrator =======================================================================================================
// TAA / temporal analyzer hybrid on the lookup. The window variance of THIS frame's lookup decides how much history is
// kept: a quiet window means the lookup is already accurate (trust it, small history), a noisy window keeps more history
// and moves the new value toward the window mean. The history is clamped into mean +- gamma sigma of the window.

[shader("compute")]
[numthreads(RT_LAB2_PASS_X, RT_LAB2_PASS_Y, 1)]
func entry_rt_lab2_integrator(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_integrator_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const RtLab2Settings s = *push.settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    CameraInfo* prev_camera = &push.attach.globals.view_camera_prev_frame;
    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth <= 0.0f)
    {
        push.attach.integrated.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    const float4 center = push.attach.lookup.get()[dtid];
    if (s.integrator_enabled == 0)
    {
        push.attach.integrated.get()[dtid] = float4(center.rgb, center.a > 0.5f ? 1.0f : 0.0f);
        return;
    }
    // The shared temporal resolve core (rt_lab_integrator.hlsl) with this lab's settings; the temporal gradient's applied correction scales the history.
    RtLabIntegratorSettings p;
    p.window_radius = s.window_radius;
    p.variance_gamma = s.variance_gamma;
    p.alpha_min = s.alpha_min;
    p.alpha_max = s.alpha_max;
    p.variance_reference = s.variance_reference;
    p.spatial_strength = s.spatial_strength;
    p.max_frames = s.max_frames;
    p.plane_tolerance = s.plane_tolerance;
    p.normal_threshold = s.normal_threshold;
    p.reset = s.reset;
    float history_scale = 1.0f;
    if (s.gradient_enabled != 0 && s.gradient_correct_integrator != 0)
    {
        const uint history_cell = push.attach.pixel_cell.get()[dtid];
        if (history_cell != RT_LAB2_INVALID_CELL)
        {
            history_scale = exp(push.attach.cells[history_cell].gk_pending);
        }
    }
    float noise;
    const float4 integrated = rt_lab_integrate(push.attach.lookup.get(), push.attach.history.get(), push.attach.lab_depth.get(), push.attach.lab_face_normal.get(),
        push.attach.lab_depth_history.get(), push.attach.lab_face_normal_history.get(), push.attach.globals, dtid, push.size, p, 0.5f, history_scale, noise);
    const float frames = integrated.a;
    push.attach.integrated.get()[dtid] = integrated;

    if (debug_mode == DEBUG_DRAW_MODE_RT_LAB2_VARIANCE || debug_mode == DEBUG_DRAW_MODE_RT_LAB2_HISTORY)
    {
        const float3 color = debug_mode == DEBUG_DRAW_MODE_RT_LAB2_VARIANCE ? Heatmap(noise) : Heatmap(saturate(frames / float(max(s.max_frames, 1))));
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
            float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
    }
}

// == Statistics =======================================================================================================
// One thread per cell slot (after the cell integrate pass: the last_* fields hold the frame that just ended). Plain atomics: only live cells count.
func rt_lab2_stats_bucket(float value) -> uint
{
    return min(uint(max(log2(1.0f + max(value, 0.0f)), 0.0f)), RT_LAB2_STATS_BUCKETS - 1u);
}

[shader("compute")]
[numthreads(RT_LAB2_CELL_PASS_X, 1, 1)]
func entry_rt_lab2_stats(uint dtid : SV_DispatchThreadID)
{
    let push = rt_lab2_stats_push;
    const RtLab2Settings s = *push.settings;
    if (dtid >= rt_lab2_capacity(s))
    {
        return;
    }
    RtLab2Cell* cell = &push.attach.cells[dtid];
    if (cell.key == 0u)
    {
        return;
    }
    const uint base = push.slot * RT_LAB2_STATS_SLOT_WORDS;
    const uint level = min(cell.level, RT_LAB2_STATS_BUCKETS - 1u);
    const bool touched = cell.last_pixel_count > 0u;
    InterlockedAdd(push.attach.stats[base + 0u], 1u);
    InterlockedAdd(push.attach.stats[base + RT_LAB2_STATS_LEVEL_LIVE + level], 1u);
    InterlockedAdd(push.attach.stats[base + RT_LAB2_STATS_UNTOUCHED + rt_lab2_stats_bucket(float(cell.frames_untouched))], 1u);
    if (cell.sample_count > 0.0f)
    {
        InterlockedAdd(push.attach.stats[base + 2u], 1u);
    }
    if (cell.last_ray_count > 0u)
    {
        InterlockedAdd(push.attach.stats[base + 6u], 1u);
        InterlockedAdd(push.attach.stats[base + 5u], cell.last_ray_count);
    }
    if (!touched)
    {
        InterlockedAdd(push.attach.stats[base + 3u], 1u);
        return;
    }
    InterlockedAdd(push.attach.stats[base + 1u], 1u);
    InterlockedAdd(push.attach.stats[base + 4u], cell.last_pixel_count);
    InterlockedAdd(push.attach.stats[base + RT_LAB2_STATS_LEVEL_TOUCHED + level], 1u);
    InterlockedAdd(push.attach.stats[base + RT_LAB2_STATS_COVERAGE + rt_lab2_stats_bucket(float(cell.last_pixel_count))], 1u);
    InterlockedAdd(push.attach.stats[base + RT_LAB2_STATS_RAYS + rt_lab2_stats_bucket(float(cell.last_ray_count))], 1u);
    if (cell.sample_count > 0.0f)
    {
        const float fill = saturate(cell.sample_count / rt_lab2_history_cap(cell, s));
        InterlockedAdd(push.attach.stats[base + RT_LAB2_STATS_FILL + min(uint(fill * float(RT_LAB2_STATS_BUCKETS)), RT_LAB2_STATS_BUCKETS - 1u)], 1u);
    }
}

// == Live cell probes =================================================================================================
// One thread per probe: copies the cell under the probe's pixel (and its trilinear corner taps) into the host visible ring.
[shader("compute")]
[numthreads(RT_LAB2_PROBE_COUNT, 1, 1)]
func entry_rt_lab2_probe(uint index : SV_DispatchThreadID)
{
    let push = rt_lab2_probe_push;
    if (index >= RT_LAB2_PROBE_COUNT)
    {
        return;
    }
    const uint slot_base = push.slot * RT_LAB2_PROBE_SLOT_WORDS;
    if (index == 0u)
    {
        push.attach.probe[slot_base + 1u] = push.active_mask;
    }
    if (((push.active_mask >> index) & 1u) == 0u)
    {
        return;
    }
    const uint base = slot_base + RT_LAB2_PROBE_HEADER_WORDS + index * RT_LAB2_PROBE_WORDS;
    for (uint i = 0u; i < RT_LAB2_PROBE_WORDS; ++i)
    {
        push.attach.probe[base + i] = 0u;
    }
    const RtLab2Settings s = *push.settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const uint2 pixel = min(push.screen_pixel[index], push.size - 1u);
    push.attach.probe[base + 1u] = pixel.x;
    push.attach.probe[base + 2u] = pixel.y;
    const float depth = push.attach.lab_depth.get()[pixel];
    uint flags = 0u;
    if (depth > 0.0f)
    {
        flags |= RT_LAB2_PROBE_FLAG_GEOMETRY;
        const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(pixel, rcp(float2(push.size))), depth);
        const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[pixel]);
        const RtLab2Key key = rt_lab2_make_key(position, normal, camera.position, rt_lab2_pixel_width_per_distance(camera, float(push.size.y)), s);
        const float cell_size = rt_lab2_cell_size(key.level, s);
        const uint cell_index = rt_lab2_find(push.attach.cells, key, rt_lab2_capacity(s) - 1u);
        push.attach.probe[base + 4u] = key.level;
        push.attach.probe[base + 5u] = key.normal_bucket;
        push.attach.probe[base + 8u] = asuint(key.coord.x);
        push.attach.probe[base + 9u] = asuint(key.coord.y);
        push.attach.probe[base + 10u] = asuint(key.coord.z);
        push.attach.probe[base + 11u] = asuint(cell_size);
        push.attach.probe[base + 12u] = asuint(position.x);
        push.attach.probe[base + 13u] = asuint(position.y);
        push.attach.probe[base + 14u] = asuint(position.z);
        push.attach.probe[base + 15u] = asuint(depth);
        push.attach.probe[base + 16u] = asuint(normal.x);
        push.attach.probe[base + 17u] = asuint(normal.y);
        push.attach.probe[base + 18u] = asuint(normal.z);
        push.attach.probe[base + 23u] = asuint(push.attach.globals.exposure);
        push.attach.probe[base + 3u] = cell_index;
        if (cell_index != RT_LAB2_INVALID_CELL)
        {
            flags |= RT_LAB2_PROBE_FLAG_CELL;
            RtLab2Cell* cell = &push.attach.cells[cell_index];
            const float3 history = rt_lab2_cell_radiance(cell) / RTGI_RADIANCE_SCALE;
            const float3 mean = float3(cell.last_mean_r, cell.last_mean_g, cell.last_mean_b) / RTGI_RADIANCE_SCALE;
            push.attach.probe[base + 6u] = cell.frames_untouched;
            push.attach.probe[base + 7u] = cell.key;
            push.attach.probe[base + 19u] = asuint(cell.sample_count);
            push.attach.probe[base + 20u] = asuint(history.x);
            push.attach.probe[base + 21u] = asuint(history.y);
            push.attach.probe[base + 22u] = asuint(history.z);
            push.attach.probe[base + 24u] = asuint(cell.exposure);
            push.attach.probe[base + 25u] = cell.last_pixel_count;
            push.attach.probe[base + 26u] = cell.last_ray_count;
            push.attach.probe[base + 72u] = asuint(cell.gk_x);
            push.attach.probe[base + 73u] = asuint(cell.gk_p);
            push.attach.probe[base + 74u] = asuint(cell.gk_applied);
            push.attach.probe[base + 75u] = asuint(cell.gk_loss);
            push.attach.probe[base + 76u] = asuint(cell.meas_z);
            push.attach.probe[base + 77u] = asuint(cell.meas_r);
            push.attach.probe[base + 78u] = cell.rec_valid;
            push.attach.probe[base + 79u] = asuint(cell.gk_pending);
            push.attach.probe[base + 28u] = asuint(mean.x);
            push.attach.probe[base + 29u] = asuint(mean.y);
            push.attach.probe[base + 30u] = asuint(mean.z);
        }
        // The trilinear corner taps (offsets 0 / +1 per axis), same projection / distance as the lookup.
        const float3 u = position / cell_size - 0.5f;
        const int3 i0 = int3(floor(u));
        const float3 t = u - float3(i0);
        for (uint tap = 0u; tap < 8u; ++tap)
        {
            const int3 k = int3(1 + int(tap & 1u), 1 + int((tap >> 1u) & 1u), 1 + int((tap >> 2u) & 1u));
            const float weight = ((tap & 1u) != 0u ? t.x : 1.0f - t.x) * (((tap >> 1u) & 1u) != 0u ? t.y : 1.0f - t.y) * (((tap >> 2u) & 1u) != 0u ? t.z : 1.0f - t.z);
            float3 radiance;
            float samples;
            const bool found = rt_lab2_tap_samples(push.attach.cells, key, position, normal, cell_size, i0, k, s, radiance, samples);
            const uint tap_base = base + 40u + tap * 4u;
            push.attach.probe[tap_base + 0u] = found ? 1u : 0u;
            push.attach.probe[tap_base + 1u] = asuint(weight);
            push.attach.probe[tap_base + 2u] = asuint(samples);
            push.attach.probe[tap_base + 3u] = asuint(rt_lab_luminance(radiance) / RTGI_RADIANCE_SCALE);
        }
        const float4 lookup = push.attach.lookup.get()[pixel];
        if (lookup.a > 0.5f)
        {
            flags |= RT_LAB2_PROBE_FLAG_LOOKUP;
        }
        push.attach.probe[base + 31u] = lookup.a > 0.5f ? 1u : 0u;
        push.attach.probe[base + 32u] = asuint(lookup.x / RTGI_RADIANCE_SCALE);
        push.attach.probe[base + 33u] = asuint(lookup.y / RTGI_RADIANCE_SCALE);
        push.attach.probe[base + 34u] = asuint(lookup.z / RTGI_RADIANCE_SCALE);
        const float4 integrated = push.attach.integrated.get()[pixel];
        push.attach.probe[base + 36u] = asuint(integrated.x / RTGI_RADIANCE_SCALE);
        push.attach.probe[base + 37u] = asuint(integrated.y / RTGI_RADIANCE_SCALE);
        push.attach.probe[base + 38u] = asuint(integrated.z / RTGI_RADIANCE_SCALE);
        push.attach.probe[base + 39u] = asuint(integrated.w);
    }
    push.attach.probe[base + 0u] = flags;
    // The stamp last (the CPU also checks it against the stamp it issued).
    if (index == 0u)
    {
        push.attach.probe[slot_base + 0u] = push.stamp;
    }
}
