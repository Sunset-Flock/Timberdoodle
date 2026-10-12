#pragma once

#include "rtgi_hash_grid.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"
#include "rtgi_hash_grid_shared.hlsl"
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtgiHashGridQueryPush rtgi_hash_grid_query_push;
[[vk::push_constant]] RtgiHashGridAllocatePush rtgi_hash_grid_allocate_push;
[[vk::push_constant]] RtgiHashGridScatterPush rtgi_hash_grid_scatter_push;
[[vk::push_constant]] RtgiHashGridBlendPush rtgi_hash_grid_blend_push;
[[vk::push_constant]] RtgiHashGridDebugPush rtgi_hash_grid_debug_push;

// A newly created cell initializes itself from a matching cell (same normal bucket) one cascade up (the parent cell)
// or down (the 8 child cells): the candidate with the most samples donates its full color and a fraction of its sample
// count, so the new cell starts converged-ish instead of black and the first real data still blends in quickly.
// A smaller cell seeded from its larger parent keeps less (RTGI_HASH_GRID_INIT_FROM_LARGER_SAMPLE_FRACTION): the
// parent's samples were spread over ~4x the surface, and its averaged color is only a coarse guess for the child.
#define RTGI_HASH_GRID_INIT_FROM_NEIGHBOR_LEVELS 1
#define RTGI_HASH_GRID_INIT_SAMPLE_FRACTION 0.5f
#define RTGI_HASH_GRID_INIT_FROM_LARGER_SAMPLE_FRACTION 0.0f
// The blend runs one thread per cell. Cells close to the camera (cascade 0 clamp) can cover thousands of pixels, and
// one thread walking all of them stalls the whole pass. Sum at most this many pixels per cell: an even stride through
// the cell's pixel list with a per-frame rotating start, and scale the ray total back up to all pixels. Those pixels
// are strongly correlated (same small surface patch, pre-filtered input), so the subset mean is as good as the full one.
#define RTGI_HASH_GRID_BLEND_MAX_PIXELS 64u
// Floor of the stored average sample count, so a written cell always reads as holding data (sample_count > 0).
#define RTGI_HASH_GRID_MIN_STORED_SAMPLE_COUNT 0.25f

func rtgi_hash_grid_init_consider(RtgiHashGridCell* cells, RtgiHashGridKey key, inout uint best_cell, inout float best_samples)
{
    const uint cell = rtgi_hash_grid_find(cells, key);
    if (cell != RTGI_HASH_GRID_INVALID_CELL)
    {
        const float samples = cells[cell].sample_count;
        if (samples > best_samples)
        {
            best_samples = samples;
            best_cell = cell;
        }
    }
}

func rtgi_hash_grid_init_from_neighbor_levels(RtgiHashGridCell* cells, uint new_cell, RtgiHashGridKey key)
{
    uint best_cell = RTGI_HASH_GRID_INVALID_CELL;
    float best_samples = 0.0f;
    if (key.level < RTGI_HASH_GRID_MAX_LEVEL)
    {
        RtgiHashGridKey parent = key;
        parent.level = key.level + 1u;
        parent.coord = key.coord >> 1; // arithmetic shift = floor division for negative coords too
        rtgi_hash_grid_init_consider(cells, parent, best_cell, best_samples);
    }
    const uint parent_cell = best_cell; // the parent is considered first; a child only wins with more samples
    if (key.level > 0u)
    {
        RtgiHashGridKey child = key;
        child.level = key.level - 1u;
        for (uint i = 0u; i < 8u; ++i)
        {
            child.coord = key.coord * 2 + int3(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
            rtgi_hash_grid_init_consider(cells, child, best_cell, best_samples);
        }
    }
    if (best_cell != RTGI_HASH_GRID_INVALID_CELL)
    {
        cells[new_cell].sh_y = cells[best_cell].sh_y;
        cells[new_cell].cocg = cells[best_cell].cocg;
        const float fraction = (best_cell == parent_cell) ? RTGI_HASH_GRID_INIT_FROM_LARGER_SAMPLE_FRACTION : RTGI_HASH_GRID_INIT_SAMPLE_FRACTION;
        cells[new_cell].sample_count = best_samples * fraction;
    }
}

// === Query ===================================================================================================
// Every geometry pixel finds (or inserts) its cell and counts itself into the cell's pixel_count. The returned old
// count is the pixel's index within the cell's pixel list, so the scatter pass needs no further atomics.
[shader("compute")]
[numthreads(RTGI_HASH_GRID_QUERY_X, RTGI_HASH_GRID_QUERY_Y, 1)]
func entry_hash_grid_query(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_hash_grid_query_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    const bool in_bounds = all(dtid < push.size);

    bool failed = false;
    if (in_bounds)
    {
        uint cell_index = RTGI_HASH_GRID_INVALID_CELL;
        uint index_in_cell = 0u;
        const float depth = push.attach.view_cam_half_res_depth.get()[dtid];
        if (depth != 0.0f)
        {
            CameraInfo* camera = &push.attach.globals->view_camera;
            const float2 inv_half_res_size = rcp(float2(push.size));
            const float2 ndc = (float2(dtid) + 0.5f) * inv_half_res_size * 2.0f - 1.0f;
            const float3 position_ws = rtgi_half_res_depth_to_world_space(*camera, ndc, depth);
            const float3 normal_ws = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[dtid]);
            const RtgiHashGridKey key = rtgi_hash_grid_make_key(position_ws, normal_ws, camera.position,
                rtgi_hash_grid_pixel_width_per_distance(inv_half_res_size.y), rtgi_settings);
            bool inserted = false;
            cell_index = rtgi_hash_grid_find_or_insert(push.attach.hash_grid, key, inserted);
            failed = cell_index == RTGI_HASH_GRID_INVALID_CELL;
#if RTGI_HASH_GRID_INIT_FROM_NEIGHBOR_LEVELS
            if (inserted)
            {
                rtgi_hash_grid_init_from_neighbor_levels(push.attach.hash_grid, cell_index, key);
            }
#endif
            if (!failed)
            {
                InterlockedAdd(push.attach.hash_grid[cell_index].pixel_count, 1u, index_in_cell);
                // The first pixel of a cell this frame draws its box: exactly one draw per queried cell.
                if (rtgi_settings.hash_grid_debug_draw_cells != 0 && index_in_cell == 0u)
                {
                    const float cell_size = rtgi_hash_grid_cell_size(key.level, rtgi_settings);
                    ShaderDebugAABBDraw aabb;
                    aabb.position = (float3(key.coord) + 0.5f) * cell_size;
                    aabb.size = cell_size.xxx;
                    aabb.color = hsv2rgb(float3(IdFloatScramble(cell_index & 0xFFFFu), 0.8f, 1.0f));
                    aabb.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
                    debug_draw_aabb(push.attach.globals.debug, aabb);
                }
            }
        }
        push.attach.pixel_cell.get()[dtid] = uint2(cell_index, index_in_cell);
    }

    const uint failed_count = WaveActiveCountBits(failed);
    if (WaveIsFirstLane() && failed_count > 0u)
    {
        InterlockedAdd(push.attach.globals.readback.rtgi_hash_grid_failed_queries, failed_count);
    }
}

// === Allocate ================================================================================================
// One thread per cell slot. Untouched cells age and get freed. Touched cells reserve their pixel list segment: a single
// 64 bit atomic adds 1 to the segment count (high 32 bits) and pixel_count to the list offset (low 32 bits). Every
// queried pixel is counted exactly once, so the list never needs more than one entry per half-res pixel.
[shader("compute")]
[numthreads(RTGI_HASH_GRID_CELL_PASS_X, 1, 1)]
func entry_hash_grid_allocate(uint dtid : SV_DispatchThreadID)
{
    let push = rtgi_hash_grid_allocate_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    RtgiHashGridCell* cell = &push.attach.hash_grid[min(dtid, RTGI_HASH_GRID_CAPACITY - 1u)];
    const bool occupied = dtid < RTGI_HASH_GRID_CAPACITY && cell.key != 0u;

    bool touched = false;
    bool alive = occupied;
    if (occupied)
    {
        const uint pixel_count = cell.pixel_count;
        touched = pixel_count > 0u;
        if (touched)
        {
            cell.frames_untouched = 0u;
            const uint64_t prev = AtomicAddU64(push.attach.list_header->merged_segment_count_pixel_offset, (uint64_t(1) << 32) | uint64_t(pixel_count));
            const uint segment = uint(prev >> 32);
            cell.list_offset = uint(prev);
            push.attach.segment_cells[segment] = dtid;
        }
        else
        {
            cell.frames_untouched = cell.frames_untouched + 1u;
            if (cell.frames_untouched > uint(max(rtgi_settings.hash_grid_max_untouched_frames, 0)))
            {
                // Free the slot. Data is zeroed so the next key inserted here starts empty.
                cell.sh_y = float4(0.0f, 0.0f, 0.0f, 0.0f);
                cell.cocg = float2(0.0f, 0.0f);
                cell.sample_count = 0.0f;
                cell.frames_untouched = 0u;
                cell.key = 0u;
                alive = false;
            }
        }
    }

    const uint live_count = WaveActiveCountBits(alive);
    const uint touched_count = WaveActiveCountBits(touched);
    if (WaveIsFirstLane())
    {
        if (live_count > 0u) { InterlockedAdd(push.attach.globals.readback.rtgi_hash_grid_live_cells, live_count); }
        if (touched_count > 0u) { InterlockedAdd(push.attach.globals.readback.rtgi_hash_grid_touched_cells, touched_count); }
    }
}

// === Scatter =================================================================================================
[shader("compute")]
[numthreads(RTGI_HASH_GRID_QUERY_X, RTGI_HASH_GRID_QUERY_Y, 1)]
func entry_hash_grid_scatter(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_hash_grid_scatter_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const uint2 pixel_cell = push.attach.pixel_cell.get()[dtid];
    if (pixel_cell.x == RTGI_HASH_GRID_INVALID_CELL)
    {
        return;
    }
    push.attach.pixel_list[push.attach.hash_grid[pixel_cell.x].list_offset + pixel_cell.y] = rtgi_hash_grid_pack_pixel(dtid);
}

// === Blend ===================================================================================================
// One thread per touched cell (segment), dispatched over the whole capacity; segments past the count exit. Touched
// cells are packed at the front, so the active threads stay dense.
[shader("compute")]
[numthreads(RTGI_HASH_GRID_CELL_PASS_X, 1, 1)]
func entry_hash_grid_blend(uint dtid : SV_DispatchThreadID)
{
    let push = rtgi_hash_grid_blend_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    const uint segment_count = uint(push.attach.list_header->merged_segment_count_pixel_offset >> 32);
    if (dtid >= segment_count)
    {
        return;
    }
    RtgiHashGridCell* cell = &push.attach.hash_grid[push.attach.segment_cells[dtid]];
    const uint pixel_count = cell.pixel_count;
    const uint list_offset = cell.list_offset;
    cell.pixel_count = 0u;

    // Plain mean of the cell's pixels' temporally accumulated diffuse (post temporal: the input already holds the full
    // temporal result, so pixels are not weighted by rays). Large cells visit a strided subset (see
    // RTGI_HASH_GRID_BLEND_MAX_PIXELS).
    float4 sh_y_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float2 cocg_sum = float2(0.0f, 0.0f);
    float sample_count_sum = 0.0f;
    float valid_pixels = 0.0f;
    const uint visit_count = min(pixel_count, RTGI_HASH_GRID_BLEND_MAX_PIXELS);
    const float visit_stride = float(pixel_count) / float(max(visit_count, 1u));
    const uint visit_start = rtgi_hash_grid_hash_u32(push.attach.globals.frame_index ^ dtid);
    for (uint v = 0u; v < visit_count; ++v)
    {
        const uint i = (uint(float(v) * visit_stride) + visit_start) % pixel_count;
        const uint2 pixel = rtgi_hash_grid_unpack_pixel(push.attach.pixel_list[list_offset + i]);
        const float4 sh_y = push.attach.diffuse.get()[pixel];
        const float2 cocg = push.attach.diffuse2.get()[pixel];
        if (all(isfinite(sh_y)) && all(isfinite(cocg)))
        {
            sh_y_sum += sh_y;
            cocg_sum += cocg;
            sample_count_sum += max(rtgi_unpack_normal_count(push.attach.sample_count.get()[pixel]), 0.0f);
            valid_pixels += 1.0f;
        }
    }
    if (valid_pixels > 0.0f)
    {
        // No accumulation in the grid: the cell IS this frame's mean of the temporal result. Untouched cells keep their
        // last value until freed (hash_grid_max_untouched_frames). sample_count = the pixels' average temporal history
        // length, used to re-initialize revived pixels (temporal accumulate). Floored so a written cell always reads as
        // holding data (> 0).
        cell.sh_y = sh_y_sum / valid_pixels;
        cell.cocg = cocg_sum / valid_pixels;
        cell.sample_count = max(sample_count_sum / valid_pixels, RTGI_HASH_GRID_MIN_STORED_SAMPLE_COUNT);
    }
}

// === Debug visualization =====================================================================================
[shader("compute")]
[numthreads(RTGI_HASH_GRID_DEBUG_X, RTGI_HASH_GRID_DEBUG_Y, 1)]
func entry_hash_grid_debug(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_hash_grid_debug_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    if (any(dtid >= push.size))
    {
        return;
    }
    const float ndc_depth = push.attach.view_cam_depth.get()[dtid];
    if (ndc_depth == 0.0f)
    {
        return; // sky
    }
    CameraInfo* camera = &push.attach.globals->view_camera;
    const float2 inv_size = rcp(float2(push.size));
    const float2 ndc = (float2(dtid) + 0.5f) * inv_size * 2.0f - 1.0f;
    const float linear_depth = camera.near_plane / ndc_depth; // infinite reversed-Z
    const float3 position_ws = linear_depth_to_world_space(*camera, ndc, linear_depth);
    const float3 face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_face_normals.get()[dtid]);
    // Same level math as the half-res query (half-res pixel width = 2 full-res pixels).
    const float pixel_width_per_distance = rtgi_hash_grid_pixel_width_per_distance(inv_size.y * 2.0f);
    RtgiHashGridKey key = rtgi_hash_grid_make_key(position_ws, face_normal, camera.position, pixel_width_per_distance, rtgi_settings);
    uint cell_index = rtgi_hash_grid_find(push.attach.hash_grid, key);
    if (debug_mode == DEBUG_DRAW_MODE_RTGI_HASH_GRID_RADIANCE_JITTERED)
    {
        // Stochastic smoothing test: random in-plane lookup offset, new every pixel and frame, no temporal filter.
        const uint seed = rtgi_hash_grid_hash_u32(rtgi_hash_grid_pack_pixel(dtid) ^ rtgi_hash_grid_hash_u32(push.attach.globals.frame_index));
        const uint jittered_cell_index = rtgi_hash_grid_find_jittered(push.attach.hash_grid, position_ws, face_normal, camera.position,
            pixel_width_per_distance, rtgi_settings, seed, key);
        if (jittered_cell_index != RTGI_HASH_GRID_INVALID_CELL)
        {
            cell_index = jittered_cell_index;
        }
    }
    const float debug_alpha = 2.0f + push.attach.globals.settings.debug_visualization_blend;

    float3 color = float3(0.0f, 0.0f, 0.0f);
    if (cell_index == RTGI_HASH_GRID_INVALID_CELL)
    {
        // No cell (not inserted yet / probe window full): dark red stripes.
        color = float3(((dtid.x + dtid.y) / 4u) % 2u == 0u ? 0.35f : 0.05f, 0.0f, 0.0f);
    }
    else
    {
        RtgiHashGridCell cell = push.attach.hash_grid[cell_index];
        if (debug_mode == DEBUG_DRAW_MODE_RTGI_HASH_GRID_CELLS)
        {
            // Random hue per cell, brightness steps per cascade level so level borders stay visible.
            const float level_shade = 1.0f - 0.5f * float(key.level & 1u);
            color = hsv2rgb(float3(float(cell.key & 0xFFFFu) * (1.0f / 65536.0f), 0.75f, level_shade));
            // Darken cell borders (within ~1 full-res pixel of a cell face).
            const float cell_size = rtgi_hash_grid_cell_size(key.level, rtgi_settings);
            const float pixel_width_ws = length(position_ws - camera.position) * inv_size.y * 2.0f;
            const float3 cell_frac = frac(position_ws / cell_size);
            const float3 edge_dist = min(cell_frac, 1.0f - cell_frac) * cell_size;
            // Ignore the axis along the normal: faces parallel to the surface are not visible as lines.
            const uint normal_axis = key.normal_bucket / 2u;
            float min_edge = 1e30f;
            [unroll] for (uint axis = 0u; axis < 3u; ++axis)
            {
                if (axis != normal_axis) { min_edge = min(min_edge, edge_dist[axis]); }
            }
            if (min_edge < pixel_width_ws)
            {
                color *= 0.25f;
            }
        }
        else if (debug_mode == DEBUG_DRAW_MODE_RTGI_HASH_GRID_RADIANCE || debug_mode == DEBUG_DRAW_MODE_RTGI_HASH_GRID_RADIANCE_JITTERED)
        {
            // Same units as DEBUG_DRAW_MODE_INDIRECT_DIFFUSE (irradiance = resolved radiance * pi), exposed.
            const float3 detail_normal = uncompress_normal_octahedral_32(push.attach.view_camera_detail_normal_image.get()[dtid]);
            const float3 radiance = sh_resolve_diffuse(cell.sh_y, cell.cocg, detail_normal) / RTGI_RADIANCE_SCALE;
            color = radiance * PI * push.attach.globals.exposure;
        }
        else // DEBUG_DRAW_MODE_RTGI_HASH_GRID_SAMPLE_COUNT
        {
            // Age: bright = updated this frame, dark = about to be freed.
            color = Heatmap(1.0f - float(cell.frames_untouched) / float(max(rtgi_settings.hash_grid_max_untouched_frames, 1)));
        }
    }
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(color, debug_alpha), 1);
}
